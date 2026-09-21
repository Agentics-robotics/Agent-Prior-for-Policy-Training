"""Resource-only continuation: wait for a busy GPU without stopping other jobs."""
import argparse
from collections import Counter
import csv
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from appl.io import ROOT, PIXI, MANIFEST, atomic, digest, read
from appl.journal import lock
from experiments.exp2.exp2_new import common, budget, run
from experiments.exp2.analysis import exp2_new_ood_continue as previous

BASE = common.BASE
PRIOR = BASE / 'ood_continuation_20260920'
OUTPUT = BASE / 'ood_resource_resume_20260920'
GPUS = [1, 2, 3, 4, 7]
MODULE = 'experiments.exp2.analysis.exp2_new_ood_resume'


def configure_storage():
    previous.OUTPUT = OUTPUT
    previous.configure_storage()


def episode_root(cell):
    if cell['condition'] != 'OOD':
        raise ValueError('OOD only')
    return OUTPUT / 'episodes' / cell['task'] / 'OOD' / str(cell['seed'])


def available_devices(active, occupancy, identities):
    return [gpu for gpu in GPUS if gpu not in active and identities[gpu]['uuid'] not in occupancy]


def prepare():
    plan = common.verify()
    proof = read(PRIOR / 'completion.json')
    assert proof['passed'] and proof['remaining_own_GPU_processes'] == []
    assert proof['new_reported_USD'] == 195.2687485 and proof['unknown_reserved_USD'] == 0
    selected = {}
    with (PRIOR / 'episodes.csv').open() as stream:
        for row in csv.DictReader(stream):
            if row['status'] == 'completed':
                selected[row['index']] = row['root']
    assert len(selected) == 45
    cells = [c for c in plan['cells'] if c['condition'] == 'OOD' and str(c['index']) not in selected]
    assert len(cells) == 30
    interrupted = []
    for cell in cells:
        root = PRIOR / 'episodes' / cell['task'] / 'OOD' / str(cell['seed'])
        if (root / 'interruption.json').exists():
            assert read(root / 'interruption.json')['reason'] == 'Selected GPU became occupied; preserve unrelated process and stop admission'
            interrupted.append(dict(index=cell['index'], root=str(root), interruption=read(root / 'interruption.json')))
    assert len(interrupted) == 4
    assert not OUTPUT.exists()
    preserved = {str(p.relative_to(BASE)): digest(p) for p in BASE.rglob('*') if p.is_file() and not p.is_symlink()}
    OUTPUT.mkdir()
    atomic(OUTPUT / 'preservation.json', preserved)
    authorization = dict(date='2026-09-20', governing_authorization=str(PRIOR / 'authorization.json'),
        additional_funding_requested=False, originally_added_SGD=500, original_added_cap_USD=390,
        prior_continuation_cost_USD=195.2687485, remaining_cap_USD=194.7312515,
        reason='A transient GPU 0 process triggered an overly strict local admission stop. No API error occurred. Resume the already-authorized OOD objective and retain all interrupted attempts.',
        remedy='Only skip currently occupied GPU slots; let other active episodes continue. Preserve unrelated processes. No transport retry.',
        scope='30 unfinished OOD layouts including four reset retests; no completed failure retests; no ID work.')
    atomic(OUTPUT / 'authorization.json', authorization)
    frozen = dict(created=time.time(), cells=cells, selected_completed_roots=selected, interrupted_retests=interrupted,
        original_plan_sha256=digest(BASE / 'plan.json'), prior_completion_sha256=digest(PRIOR / 'completion.json'),
        script_sha256=digest(__file__), inherited_storage_wrapper_sha256=digest(previous.__file__),
        preservation_sha256=digest(OUTPUT / 'preservation.json'), authorization_sha256=digest(OUTPUT / 'authorization.json'),
        physical_devices=GPUS, maximum_active_devices=5, scientific_changes=[], ID_admission=False)
    atomic(OUTPUT / 'plan.json', frozen)
    atomic(OUTPUT / 'plan_sha256.json', dict(sha256=digest(OUTPUT / 'plan.json')))
    (OUTPUT / 'source.py').write_text(Path(__file__).read_text())
    previous.ContinuationLedger().initialize('194.7312515', str(OUTPUT / 'authorization.json'))
    summarize()
    print(dict(prepared=True, reused_OOD=45, remaining_OOD=30, preserved_retests=4, remaining_USD=194.7312515), flush=True)


def verify():
    plan = read(OUTPUT / 'plan.json')
    assert digest(OUTPUT / 'plan.json') == read(OUTPUT / 'plan_sha256.json')['sha256']
    assert digest(__file__) == plan['script_sha256']
    assert digest(previous.__file__) == plan['inherited_storage_wrapper_sha256']
    assert digest(BASE / 'plan.json') == plan['original_plan_sha256']
    assert digest(PRIOR / 'completion.json') == plan['prior_completion_sha256']
    assert digest(OUTPUT / 'authorization.json') == plan['authorization_sha256']
    common.verify()
    return plan


def command(*args):
    return [PIXI, 'run', '--manifest-path', str(MANIFEST), '--locked', '--no-install',
            'python', '-m', MODULE, *map(str, args)]


def supervise():
    from appl.gpu import identity
    plan = verify(); jobs = OUTPUT / 'supervisor'; jobs.mkdir(exist_ok=False)
    active = {}; next_cell = 0; finished = []; stop = None
    with lock(OUTPUT / 'supervisor.lock'):
        while active or (next_cell < len(plan['cells']) and not stop):
            stop = stop or read(OUTPUT / 'budget/ledger.json')['stopped']
            for gpu, item in list(active.items()):
                process, cell, stdout, stderr = item
                code = process.poll()
                if code is None:
                    continue
                stdout.close(); stderr.close()
                atomic(jobs / str(cell['index']) / 'process_result.json', dict(returncode=code, finished=time.time()))
                finished.append(dict(index=cell['index'], returncode=code, gpu=gpu)); del active[gpu]
                if code:
                    stop = stop or 'Episode stopped; no further admission or automatic retry'
                    if not read(OUTPUT / 'budget/ledger.json')['stopped']:
                        previous.ContinuationLedger().stop(stop)
                summary = summarize()
                print(dict(finished=cell['index'], returncode=code, OOD_completed=summary['OOD_completed'],
                    OOD_successes=summary['OOD_successes'], remaining_phase_cost_USD=summary['costs']['phase_reported_USD']), flush=True)
            if not stop and next_cell < len(plan['cells']):
                occupancy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                    '--format=csv,noheader,nounits'], text=True)
                identities = {gpu: identity(gpu) for gpu in GPUS}
                free = available_devices(active, occupancy, identities)
                for gpu in free:
                    if next_cell == len(plan['cells']):
                        break
                    if read(OUTPUT / 'budget/ledger.json')['stopped']:
                        break
                    cell = plan['cells'][next_cell]; job = jobs / str(cell['index']); job.mkdir(exist_ok=False)
                    args = command('evaluate', '--cell', cell['index'], '--gpu', gpu)
                    atomic(job / 'process.json', dict(command=args, device=identities[gpu], occupancy=occupancy, started=time.time()))
                    stdout = (job / 'stdout.log').open('w'); stderr = (job / 'stderr.log').open('w')
                    process = subprocess.Popen(args, cwd=ROOT, stdout=stdout, stderr=stderr)
                    atomic(job / 'pid.json', dict(pid=process.pid))
                    active[gpu] = (process, cell, stdout, stderr); next_cell += 1
                    print(dict(started=cell['index'], task=cell['task'], seed=cell['seed'], gpu=gpu), flush=True)
                if not free and not active:
                    atomic(jobs / 'waiting_for_GPU.json', dict(time=time.time(), occupancy=occupancy, preserved_unrelated_processes=True))
            if active or (next_cell < len(plan['cells']) and not stop):
                time.sleep(1)
        atomic(jobs / 'completion.json', dict(attempted=next_cell, planned=len(plan['cells']), processes=finished,
            stopped=stop, all_scheduled_parents_exited=True, maximum_active_devices=5, ID_episodes_scheduled=0))
        summarize()


def summarize():
    plan = read(OUTPUT / 'plan.json'); original = read(BASE / 'plan.json'); rows = []; cards = []
    for cell in original['cells']:
        if cell['condition'] != 'OOD':
            continue
        reused = plan['selected_completed_roots'].get(str(cell['index']))
        root = Path(reused) if reused else episode_root(cell)
        audit = read(root / 'audit.json') if (root / 'audit.json').exists() else {}
        proc = OUTPUT / 'supervisor' / str(cell['index']) / 'process_result.json'
        done = bool(audit.get('passed') and audit.get('completed') and (reused or (proc.exists() and read(proc)['returncode'] == 0)))
        result = read(root / 'result.json') if done else None
        if result:
            assert digest(root / 'result.json') == audit['result_sha256']
        status = 'completed' if done else 'interrupted' if (root / 'interruption.json').exists() else 'running' if root.exists() else 'not_started'
        row = dict(index=cell['index'], task=cell['task'], seed=cell['seed'], condition='OOD', status=status,
            success=result['success'] if result else None, steps=result['steps'] if result else None,
            source='previous_validated' if reused else 'resource_resume', root=str(root))
        for method, reference in cell['baseline_roots'].items():
            row[method] = read(Path(reference) / 'result.json')['success']
        rows.append(row)
        if (root / 'replay.mp4').exists():
            title = f"{cell['task']} / {cell['seed']}: {status}"
            if result:
                title += f" — {'success' if result['success'] else 'failure'}, {result['steps']} steps"
            cards.append(f'<article><h3>{html.escape(title)}</h3><video controls preload="none" src="{os.path.relpath(root / "replay.mp4", OUTPUT)}"></video></article>')
    groups = []
    for task in common.TASKS + ['all']:
        selected = [r for r in rows if task == 'all' or r['task'] == task]
        done = [r for r in selected if r['status'] == 'completed']
        groups.append(dict(task=task, planned=len(selected), completed=len(done), APPL_successes=sum(r['success'] for r in done),
            naive_DP_matched=sum(r['naive_DP'] for r in done), SinglePrior_matched=sum(r['SinglePrior_6_xhigh'] for r in done)))
    ledger = read(OUTPUT / 'budget/ledger.json'); records = ledger['records']
    known = sum(r.get('settled_nano_usd', 0) for r in records) / budget.NANO
    held = sum(r['reserved_nano_usd'] for r in records if 'settled_nano_usd' not in r) / budget.NANO
    costs = dict(phase_reported_USD=known, phase_unknown_reserved_USD=held, phase_cap_USD=194.7312515,
        prior_continuation_USD=195.2687485, additional_500_SGD_total_reported_USD=195.2687485 + known,
        additional_500_SGD_cap_USD=390, original_100_SGD_round_USD=77.5677315,
        generation_attempts=len(records), stopped=ledger['stopped'])
    counts = Counter(r['status'] for r in rows)
    value = dict(OOD_planned=75, OOD_completed=counts['completed'], OOD_successes=groups[-1]['APPL_successes'],
        OOD_interrupted=counts['interrupted'], OOD_running=counts['running'], OOD_not_started=counts['not_started'],
        unchanged_ID_completed=10, unchanged_ID_successes=8, groups=groups, costs=costs)
    atomic(OUTPUT / 'results.json', value); previous.dump_csv(OUTPUT / 'episodes.csv', rows); previous.dump_csv(OUTPUT / 'summary.csv', groups)
    lines = ['# Exp2_new: complete OOD comparison', '',
        f"Completed OOD: **{value['OOD_completed']}/75**; successes **{value['OOD_successes']}**; interrupted {counts['interrupted']}, running {counts['running']}, unstarted {counts['not_started']}.", '',
        'All methods use the same 15 original position-OOD layouts per task. APPL reuses 36 frozen GPT-6 Astra/xhigh policies; the two corrected-executor baselines reuse their existing results. Original prompts, tools, DDPM100, 8-action chunks, binary gripper, hidden 5,000-step cap and geometric goals remain unchanged. No retraining, segmentation, policy design or API-output editing occurs. ID remains 8/10 and receives no new trials.', '',
        '| Task | Completed / planned | APPL | Naive DP, matched | SinglePrior, matched |',
        '| --- | ---: | ---: | ---: | ---: |']
    for g in groups:
        n = g['completed']
        lines.append(f"| {g['task']} | {n}/{g['planned']} | {g['APPL_successes']}/{n} | {g['naive_DP_matched']}/{n} | {g['SinglePrior_matched']}/{n} |")
    lines += ['', '## Execution history and costs', '',
        'Seven original OOD results and 38 completed continuation results are reused. A transient GPU 0 process triggered an overly strict resource guard in the first continuation, interrupting four other episodes. Their prefixes and all charges are retained. This resource-only revision waits for busy slots without stopping unrelated active episodes, uses GPUs 1/2/3/4/7, and explicitly retests only those four unknown outcomes from the same reset. No completed failure is retested, and no transport request is automatically retried.', '',
        'The user authorized SGD 500 additional funds. The USD 390 conservative allowance is shared across both continuation phases; the second phase receives only the unspent USD 194.7312515. Old spending and interrupted costs are included, not erased. Values are usage estimates, not invoices.', '',
        '```json', json.dumps(costs, indent=2), '```', '',
        'These are previously examined layouts and one frozen training seed. Use matched completed subsets until all 75 outcomes are validated. Provider/API randomness and the executor correction prevent treating differences from the original gateway study as a single-factor model ablation.', '',
        '- [Selected outcomes](episodes.csv)', '- [Machine-readable results](results.json)', '- [All selected videos](replays.html)',
        '- [This phase plan](plan.json)', '- [Resource recovery and remaining budget](authorization.json)',
        '- [This phase budget ledger](budget/ledger.json)', '- [Previous phase audit](../ood_continuation_20260920/completion.json)',
        '- [Previous phase report and interruption evidence](../ood_continuation_20260920/ANALYSIS.md)',
        '- [Original budget-limited run](../ANALYSIS.md)']
    (OUTPUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    (OUTPUT / 'replays.html').write_text('<!doctype html><meta charset="utf-8"><title>Exp2_new OOD</title><style>body{font:16px sans-serif;max-width:1100px;margin:auto}article{display:inline-block;vertical-align:top;width:48%;margin:1%}video{width:100%}</style><h1>Exp2_new OOD</h1><p>Selected original-frame videos; approximately 6x simulation speed.</p>' + ''.join(cards))
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'supervise', 'evaluate', 'report'])
    p.add_argument('--cell', type=int); p.add_argument('--gpu', type=int)
    p.add_argument('--device-isolated', action='store_true'); args = p.parse_args()
    configure_storage()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'supervise':
        supervise()
    elif args.action == 'report':
        print(summarize(), flush=True)
    else:
        plan = verify(); cell = next(c for c in plan['cells'] if c['index'] == args.cell)
        assert args.gpu in GPUS
        if not args.device_isolated:
            from appl.gpu import launch
            launch(args.gpu, sys.argv[1:], module=MODULE)
        raise SystemExit(run.evaluate(cell))


if __name__ == '__main__':
    main()
