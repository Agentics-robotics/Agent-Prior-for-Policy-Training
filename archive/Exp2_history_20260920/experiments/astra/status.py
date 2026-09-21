"""Read-only progress snapshot for the additional APPL study."""
from collections import Counter
import argparse
import json
import os
import time

from appl.io import read
from .runner import BASE, NAMES


def recovery_snapshot():
    """Prefer the latest recorded recovery, while retaining original attempt counts."""
    rounds = ('evaluation_recovery', 'evaluation_recovery_20260918', 'evaluation_followup_20260918')
    root = next((BASE / name for name in reversed(rounds)
                 if (BASE / name / 'status.json').exists()), None)
    if root is None:
        return None
    status = read(root / 'status.json')
    original = read(BASE / 'results.json')
    rows = {(r['task'], r['condition'], r['seed']): r for r in original['episodes']
            if r['method'] == 'APPL_6_xhigh'}
    attempts = Counter()
    for round_name in rounds:
        folder = BASE / round_name
        for path in (folder / 'jobs').glob('*/*/*/process_result.json'):
            name, condition, seed = path.parent.relative_to(folder / 'jobs').parts
            if read(path)['returncode']:
                if folder == root:
                    attempts['interrupted'] += 1
            else:
                result = read(folder / name / 'evaluation/APPL' / condition / seed / 'result.json')
                rows[name, condition, int(seed)] = dict(result, completed=True)
                if folder == root:
                    attempts['success' if result['success'] else 'task_failure'] += 1
    complete = [r for r in rows.values() if r['completed']]
    outcomes = Counter('success' if r['success'] else 'task_failure' for r in complete)
    tasks = {name: {condition: dict(
        completed=sum(r['completed'] for k, r in rows.items() if k[:2] == (name, condition)),
        success=sum(r['success'] is True for k, r in rows.items() if k[:2] == (name, condition)),
        planned=30) for condition in ('ID', 'OOD')} for name in NAMES}
    return dict(stage='evaluation_recovery', round=root.name, finished=sum(attempts.values()),
        pending=len(status['pending']), active=len(status['active']), process_failures=status['failed'],
        outcomes=dict(attempts), overall_outcomes=dict(outcomes), overall_completed=len(complete),
        overall_unknown=len(rows)-len(complete), planned_outcomes=len(rows), tasks=tasks,
        original_interrupted_attempts_retained=original['summary']['new_unknown'],
        execution_ended=(root / 'execution.json').exists(),
        completion=(root / 'completion.json').exists(),
        status_age_seconds=round(time.time()-status['time']))


def snapshot():
    resumed = recovery_snapshot()
    if resumed is not None:
        return resumed
    status = read(BASE / 'status.json')
    owner = read(BASE / 'coordinator.json')
    try:
        os.kill(owner['pid'], 0)
        visible = True
    except ProcessLookupError:
        visible = False
    age = round(time.time() - status['time'])
    tasks = {}
    running = []
    failures = []
    design_interruptions = {}
    for path in (BASE / 'incidents').glob('*/incident.json'):
        incident = read(path)
        if incident.get('status') == 'design_budget_exhausted':
            design_interruptions[incident['task'], incident['policy']] = path
    for name in NAMES:
        root = BASE / name
        counts = Counter()
        for assignment in sorted(root.glob('policies/*/*/assignment.json')):
            folder = assignment.parent
            policy_id = read(assignment)['policy_id']
            counts['assigned'] += 1
            counts['submitted'] += (folder / 'submission.json').exists()
            counts['trained'] += (folder / 'training/result.json').exists()
            checked = root / 'inference_checks' / policy_id / 'result.json'
            counts['checked'] += checked.exists()
            if not checked.exists():
                progress = folder / 'training/progress.json'
                row = dict(task=name, policy=policy_id)
                if (folder / 'training/result.json').exists():
                    row['stage'] = 'trained_waiting_check'
                elif progress.exists():
                    p = read(progress)
                    row.update(stage='training', step=p['step'], updates=p['updates'],
                               elapsed_seconds=round(p['elapsed_seconds']))
                elif (folder / 'submission.json').exists():
                    row['stage'] = 'submitted'
                else:
                    row['stage'] = 'design'
                    incident = design_interruptions.get((name, policy_id))
                    if incident is not None:
                        row['stage'] = 'design_budget_exhausted'
                        row['incident'] = str(incident.relative_to(BASE))
                    responses = sorted((folder / 'design/api').glob('*.response.json'))
                    if responses:
                        response = read(responses[-1])
                        if response['http_status'] != 200:
                            row['stage'] = 'interrupted_design'
                            row['http_status'] = response['http_status']
                            failures.append(dict(path=str(responses[-1].relative_to(BASE)),
                                                 http_status=response['http_status']))
                running.append(row)
        for condition in ('ID', 'OOD'):
            results = [read(p) for p in root.glob(f'evaluation/APPL/{condition}/*/result.json')]
            counts[condition + '_completed'] = len(results)
            counts[condition + '_success'] = sum(r['success'] for r in results)
        tasks[name] = dict(counts)
        for path in sorted(root.rglob('failure.json')):
            # Bounded pre-submission interface failures remain in check receipts;
            # they are not terminal failures if the API subsequently submits.
            if 'checks' not in path.relative_to(root).parts:
                failures.append(dict(path=str(path.relative_to(BASE)), **read(path)))
    # A sandboxed reader may have a private PID namespace; absence there does
    # not establish that the host coordinator has exited.
    auxiliary = []
    for name in ('preparation_overlap', 'training_overlap', 'allocation_5gpu',
                 'preparation_continuation', 'preparation_continuation_after_interface_failure'):
        for receipt in sorted((BASE / name).glob('*/process_result.json')):
            result = read(receipt)
            auxiliary.append(dict(worker=str(receipt.parent.relative_to(BASE)), **result))
    return dict(stage=status['stage'], coordinator_pid_visible=visible,
                status_age_seconds=age, heartbeat_recent=age < 30,
                active_jobs=len(status['active']), finished_jobs=status['finished'],
                pending_jobs=status['pending'], failed_jobs=status['failed'],
                tasks=tasks, running_policies=running, failures=failures,
                auxiliary_processes=auxiliary,
                completion=(BASE / 'completion.json').exists())


def compact_snapshot():
    """Count only exited episode processes; do not infer completion from files in flight."""
    resumed = recovery_snapshot()
    if resumed is not None:
        return {k:v for k,v in resumed.items() if k != 'tasks'}
    status=read(BASE / 'status.json');outcomes=Counter()
    for receipt in (BASE / 'jobs/evaluation').glob('*/*/*/process_result.json'):
        name,condition,seed=receipt.parent.relative_to(BASE / 'jobs/evaluation').parts
        if read(receipt)['returncode']:
            outcome='interrupted'
        else:
            result=read(BASE / name / 'evaluation/APPL' / condition / seed / 'result.json')
            outcome='success' if result['success'] else 'task_failure'
        outcomes[outcome]+=1
    return dict(stage=status['stage'], finished=status['finished'], pending=status['pending'],
        active=len(status['active']), process_failures=status['failed'], outcomes=dict(outcomes),
        completion=(BASE / 'completion.json').exists())


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compact',action='store_true')
    args=parser.parse_args()
    print(json.dumps(compact_snapshot() if args.compact else snapshot(), ensure_ascii=False))
