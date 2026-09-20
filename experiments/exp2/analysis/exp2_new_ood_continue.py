"""Authorized OOD-only continuation; reuse the frozen Exp2_new evaluator verbatim."""
import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import fcntl
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from appl.io import ROOT, PIXI, MANIFEST, atomic, digest, read
from appl.journal import lock
from experiments.exp2.exp2_new import common, budget, run, report

BASE = common.BASE
OUTPUT = BASE / 'ood_continuation_20260920'
MODULE = 'experiments.exp2.analysis.exp2_new_ood_continue'
GPUS = [0, 1, 2, 3, 7]


def select_pending(plan, completed):
    return [c for c in plan['cells'] if c['condition'] == 'OOD' and c['index'] not in completed]


def episode_root(cell):
    if cell['condition'] != 'OOD':
        raise ValueError('This authorization schedules OOD only')
    return OUTPUT / 'episodes' / cell['task'] / 'OOD' / str(cell['seed'])


class ContinuationLedger(budget.Ledger):
    def __init__(self):
        super().__init__(OUTPUT / 'budget')

    @contextmanager
    def transaction(self):
        # The original serial runner deliberately used a nonblocking file lock.
        # Queue only short local ledger transactions across the five new workers.
        # Network calls never hold this lock and are never retried.
        with (self.root / 'parallel.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def reserve(self, key, n, maximum):
        with self.transaction():
            return super().reserve(key, n, maximum)

    def settle(self, key, body, http_status):
        with self.transaction():
            return super().settle(key, body, http_status)

    def stop(self, reason):
        with self.transaction():
            return super().stop(reason)


class ContinuationClient(budget.BudgetClient):
    def __init__(self, journal):
        super().__init__(journal)
        self.ledger = ContinuationLedger()


def configure_storage():
    # Only output destinations and the newly authorized monetary ledger change.
    # The original evaluator still reads its original frozen plan/config/checks.
    run.episode_root = episode_root
    report.episode_root = episode_root
    run.BudgetClient = ContinuationClient
    run.Ledger = ContinuationLedger


def prepare():
    plan = common.verify()
    for task in common.TASKS:
        common.verify(task)
    prior = read(BASE / 'completion.json')
    assert prior['passed'] and prior['remaining_own_GPU_processes'] == []
    completed = {e['index'] for e in prior['episodes'] if e['completed']}
    pending = select_pending(plan, completed)
    assert len(pending) == 68 and len(completed) == 17
    assert all(c['condition'] == 'OOD' for c in pending)
    credential = Path('/tmp') / f'exp2_new_credentials_{os.getuid()}' / 'credential.json'
    assert credential.is_file() and not credential.stat().st_mode & 0o077
    if OUTPUT.exists():
        raise ValueError('Continuation already prepared; preserve its records')
    originals = {str(p.relative_to(BASE)): digest(p) for p in BASE.rglob('*') if p.is_file() and not p.is_symlink()}
    OUTPUT.mkdir()
    atomic(OUTPUT / 'preservation.json', originals)
    authorization = dict(date='2026-09-20', additional_budget_SGD=500, new_cap_USD=390,
        planning_USD_per_SGD=0.78, planning_basis='Reuse the previous conservative budgeting ratio; not a live FX quote or invoice conversion.',
        previous_authorization=str(BASE / 'budget/authorization.json'), old_remaining_USD_not_reallocated=0.4322685,
        instruction='Continue after adding SGD 500; finish all 75 OOD layouts, then stop and report.',
        scope='68 outstanding OOD cells; restart the one budget-interrupted cell from the same initial state; reuse all seven completed OOD cells.',
        ID_admission=False, automatic_transport_retry=False, new_training=False)
    atomic(OUTPUT / 'authorization.json', authorization)
    frozen = dict(created=time.time(), original_plan_sha256=digest(BASE / 'plan.json'),
        old_completion_sha256=digest(BASE / 'completion.json'), cells=pending,
        completed_original_indices=sorted(completed), physical_devices=GPUS,
        maximum_active_devices=5, maximum_jobs_per_device=1,
        script_sha256=digest(__file__), preservation_sha256=digest(OUTPUT / 'preservation.json'),
        authorization_sha256=digest(OUTPUT / 'authorization.json'),
        reset_retest_index=17, reset_retest_original=str(common.episode_root(plan['cells'][17])),
        stop_after='75 validated OOD outcomes; no further ID work',
        scientific_changes=[], storage_change='Independent continuation output and budget; original evaluator/config/API prompts reused.')
    atomic(OUTPUT / 'plan.json', frozen)
    atomic(OUTPUT / 'plan_sha256.json', dict(sha256=digest(OUTPUT / 'plan.json')))
    (OUTPUT / 'source.py').write_text(Path(__file__).read_text())
    ContinuationLedger().initialize('390', str(OUTPUT / 'authorization.json'))
    summarize()
    print(dict(prepared=True, new_OOD=len(pending), reused_OOD=7, new_USD_cap=390, devices=GPUS), flush=True)


def verify():
    plan = read(OUTPUT / 'plan.json')
    assert digest(OUTPUT / 'plan.json') == read(OUTPUT / 'plan_sha256.json')['sha256']
    assert digest(__file__) == plan['script_sha256']
    assert digest(BASE / 'plan.json') == plan['original_plan_sha256']
    assert digest(BASE / 'completion.json') == plan['old_completion_sha256']
    assert digest(OUTPUT / 'authorization.json') == plan['authorization_sha256']
    common.verify()
    return plan


def command(*args):
    return [PIXI, 'run', '--manifest-path', str(MANIFEST), '--locked', '--no-install',
            'python', '-m', MODULE, *map(str, args)]


def supervise():
    from appl.gpu import identity
    plan = verify()
    jobs = OUTPUT / 'supervisor'
    jobs.mkdir(exist_ok=False)
    active = {}; next_cell = 0; finished = []; stop = None
    with lock(OUTPUT / 'supervisor.lock'):
        while active or (next_cell < len(plan['cells']) and not stop):
            ledger = read(OUTPUT / 'budget/ledger.json')
            stop = stop or ledger['stopped']
            for gpu, item in list(active.items()):
                process, cell, stdout, stderr = item
                code = process.poll()
                if code is None:
                    continue
                stdout.close(); stderr.close()
                atomic(jobs / str(cell['index']) / 'process_result.json', dict(returncode=code, finished=time.time()))
                finished.append(dict(index=cell['index'], returncode=code, gpu=gpu))
                del active[gpu]
                if code:
                    stop = stop or 'An episode stopped; no further admission or automatic retry'
                    if not read(OUTPUT / 'budget/ledger.json')['stopped']:
                        ContinuationLedger().stop(stop)
                summary = summarize()
                print(dict(finished=cell['index'], returncode=code, OOD_completed=summary['OOD_completed'],
                           OOD_successes=summary['OOD_successes'], new_cost_USD=summary['costs']['new_reported_USD']), flush=True)
            for gpu in GPUS:
                if stop or next_cell == len(plan['cells']):
                    break
                if gpu in active:
                    continue
                ident = identity(gpu)
                occupancy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                    '--format=csv,noheader,nounits'], text=True)
                if ident['uuid'] in occupancy:
                    stop = 'Selected GPU became occupied; preserve unrelated process and stop admission'
                    ContinuationLedger().stop(stop)
                    break
                cell = plan['cells'][next_cell]
                job = jobs / str(cell['index']); job.mkdir(exist_ok=False)
                args = command('evaluate', '--cell', cell['index'], '--gpu', gpu)
                atomic(job / 'process.json', dict(command=args, device=ident, occupancy=occupancy, started=time.time()))
                stdout = (job / 'stdout.log').open('w'); stderr = (job / 'stderr.log').open('w')
                process = subprocess.Popen(args, cwd=ROOT, stdout=stdout, stderr=stderr)
                atomic(job / 'pid.json', dict(pid=process.pid))
                active[gpu] = (process, cell, stdout, stderr); next_cell += 1
                print(dict(started=cell['index'], task=cell['task'], seed=cell['seed'], gpu=gpu), flush=True)
            if active:
                time.sleep(1)
        atomic(jobs / 'completion.json', dict(attempted=next_cell, planned=len(plan['cells']), processes=finished,
            stopped=stop, all_scheduled_parents_exited=not active, maximum_active_devices=5,
            ID_episodes_scheduled=0))
        summarize()


def dump_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def summarize():
    plan = read(OUTPUT / 'plan.json'); original = read(BASE / 'plan.json')
    continued = {c['index'] for c in plan['cells']}; rows = []; cards = []
    for cell in original['cells']:
        if cell['condition'] != 'OOD':
            continue
        new = cell['index'] in continued
        root = episode_root(cell) if new else common.episode_root(cell)
        process = (OUTPUT / 'supervisor' if new else BASE / 'supervisor/evaluate') / str(cell['index']) / 'process_result.json'
        audit = read(root / 'audit.json') if (root / 'audit.json').exists() else {}
        complete = bool(audit.get('passed') and audit.get('completed') and process.exists()
            and read(process)['returncode'] == 0 and (root / 'replay_provenance.json').exists())
        result = read(root / 'result.json') if complete else None
        if result:
            assert digest(root / 'result.json') == audit['result_sha256']
        status = 'completed' if complete else 'interrupted' if (root / 'interruption.json').exists() else 'running' if root.exists() else 'not_started'
        row = dict(index=cell['index'], task=cell['task'], seed=cell['seed'], condition='OOD', status=status,
            success=result['success'] if result else None, steps=result['steps'] if result else None,
            source='continuation' if new else 'original', root=str(root))
        for method, reference in cell['baseline_roots'].items():
            row[method] = read(Path(reference) / 'result.json')['success']
        rows.append(row)
        if (root / 'replay.mp4').exists():
            title = f"{cell['task']} / OOD / {cell['seed']}: {status}"
            if result:
                title += f" — {'success' if result['success'] else 'failure'}, {result['steps']} steps"
            url = os.path.relpath(root / 'replay.mp4', OUTPUT)
            cards.append(f'<article><h3>{html.escape(title)}</h3><video controls preload="none" src="{url}"></video></article>')
    groups = []
    for task in common.TASKS + ['all']:
        selected = [r for r in rows if task == 'all' or r['task'] == task]
        done = [r for r in selected if r['status'] == 'completed']
        groups.append(dict(task=task, planned=len(selected), completed=len(done), APPL_successes=sum(r['success'] for r in done),
            naive_DP_matched=sum(r['naive_DP'] for r in done), SinglePrior_matched=sum(r['SinglePrior_6_xhigh'] for r in done),
            naive_DP_full=sum(r['naive_DP'] for r in selected), SinglePrior_full=sum(r['SinglePrior_6_xhigh'] for r in selected)))
    ledger = read(OUTPUT / 'budget/ledger.json'); records = ledger['records']
    known = sum(r.get('settled_nano_usd', 0) for r in records) / budget.NANO
    held = sum(r['reserved_nano_usd'] for r in records if 'settled_nano_usd' not in r) / budget.NANO
    costs = dict(new_reported_USD=known, unknown_reserved_USD=held, new_cap_USD=390,
        old_reported_USD=read(BASE / 'completion.json')['costs']['reported_usd'],
        generation_attempts=len(records), stopped=ledger['stopped'],
        input_tokens=sum((r.get('usage') or {}).get('input_tokens', 0) for r in records),
        output_tokens=sum((r.get('usage') or {}).get('output_tokens', 0) for r in records))
    counts = Counter(r['status'] for r in rows)
    value = dict(OOD_planned=75, OOD_completed=counts['completed'], OOD_successes=groups[-1]['APPL_successes'],
        OOD_interrupted=counts['interrupted'], OOD_running=counts['running'], OOD_not_started=counts['not_started'],
        unchanged_ID_completed=10, unchanged_ID_successes=8, groups=groups, costs=costs)
    atomic(OUTPUT / 'results.json', value); dump_csv(OUTPUT / 'episodes.csv', rows); dump_csv(OUTPUT / 'summary.csv', groups)
    lines = ['# Exp2_new: OOD-only continuation', '',
        f"OOD: **{value['OOD_successes']}/{value['OOD_completed']} completed**; interrupted {counts['interrupted']}, running {counts['running']}, unstarted {counts['not_started']}; planned 75.", '',
        'The user added SGD 500 and requested completing all 75 OOD layouts, then stopping. Seven completed original OOD outcomes are reused. The one previous budget interruption is preserved and explicitly retested from the same reset; no completed failure is retested. No ID trial is admitted. Existing ID results remain 8/10.', '',
        'The unchanged original Exp2_new evaluator uses GPT-6 Astra/xhigh, all 36 frozen policies, original prompts/handoff documents, corrected binary gripper execution, DDPM100 and the hidden 5,000-step cap. No training, segmentation, design or API-output editing occurs. Only output directories, monetary ledger and scheduling change; at most five GPUs run concurrently.', '',
        '## Identical-layout comparison', '',
        '| Task | Completed / planned | APPL successes | Naive DP on matched layouts | SinglePrior on matched layouts |',
        '| --- | ---: | ---: | ---: | ---: |']
    for g in groups:
        n = g['completed']
        lines.append(f"| {g['task']} | {n}/{g['planned']} | {g['APPL_successes']}/{n} | {g['naive_DP_matched']}/{n} | {g['SinglePrior_matched']}/{n} |")
    lines += ['', '## Accounting', '', '```json', json.dumps(costs, indent=2), '```', '',
        'The new USD 390 cap uses the previous conservative planning ratio of USD 0.78 per SGD; it is not a live currency quote. Old spending and unused old allowance are not reassigned. Usage-based estimates are not invoices. Unknown reservations remain separate. No automatic transport retries or fallback credentials/models are allowed.', '',
        '## Interpretation and evidence', '',
        'These are previously examined layouts with one frozen training seed. Budget or provider interruptions are unknown outcomes, not task failures. Use only matched completed layouts until all 75 outcomes pass audit. Parallel scheduling may affect service-side randomness or cache costs; model sampling is not seeded. The seven reused outcomes retain their original execution provenance.', '',
        '- [Frozen continuation plan](plan.json)', '- [Authorization and budget](authorization.json)',
        '- [Machine-readable results](results.json)', '- [Every layout](episodes.csv)', '- [Videos](replays.html)',
        '- [New request ledger](budget/ledger.json)', '- [Original report, retained unchanged](../REPORT.md)',
        '- [Original interrupted prefix](../APPL_GPT6_astra_xhigh/buffer_swap/OOD/30101/interruption.json)']
    (OUTPUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    (OUTPUT / 'replays.html').write_text('<!doctype html><meta charset="utf-8"><title>Exp2_new OOD</title><style>body{font:16px sans-serif;max-width:1100px;margin:auto}article{display:inline-block;vertical-align:top;width:48%;margin:1%}video{width:100%}</style><h1>Exp2_new: OOD</h1><p>Original frames, approximately 6x simulation speed. Interrupted prefixes are not outcomes.</p>' + ''.join(cards))
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'supervise', 'evaluate', 'report'])
    parser.add_argument('--cell', type=int); parser.add_argument('--gpu', type=int)
    parser.add_argument('--device-isolated', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'supervise':
        supervise()
    elif args.action == 'report':
        print(summarize(), flush=True)
    else:
        frozen = verify()
        cell = next(c for c in frozen['cells'] if c['index'] == args.cell)
        assert args.gpu in GPUS
        if not args.device_isolated:
            from appl.gpu import launch
            launch(args.gpu, sys.argv[1:], module=MODULE)
        configure_storage()
        raise SystemExit(run.evaluate(cell))


if __name__ == '__main__':
    main()
