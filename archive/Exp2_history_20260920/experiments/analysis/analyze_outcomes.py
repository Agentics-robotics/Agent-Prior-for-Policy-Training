"""Exploratory measurements of completed frozen trials, without rerunning them."""
from collections import Counter
from datetime import datetime, timezone
import json
import numpy as np

from appl.io import ROOT, read, atomic, digest
from appl.prior_policies.data import vector, SLICES
from appl.scaleup.protocol import BASE, NAMES, config

OUTPUT = ROOT / 'experiments/exp2/analysis/outcome_diagnostics.json'


def field_name(index):
    for name, (start, stop) in SLICES.items():
        if start <= index < stop:
            return f'{name}[{index - start}]'
    raise ValueError(index)


def analyze(root, norm):
    result = read(root / 'result.json')
    initial = read(root / 'initial_state.json')
    states = [initial]
    first_goals = {}
    actions = []
    if (root / 'trace.jsonl').exists():
        with (root / 'trace.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                assert row['step'] == len(states)
                states.append(row['state'])
                actions.append(row['action'])
                for name, met in row['metrics'].items():
                    if met and name not in first_goals:
                        first_goals[name] = row['step']
    assert len(states) == result['steps'] + 1
    observations = np.stack([vector(s) for s in states])
    # State t is the input of action t+1. The final state has no subsequent action.
    inputs = observations[:-1]
    maximum = None
    if len(inputs):
        encoded = (inputs - norm['mean']) / norm['std']
        step, index = np.unravel_index(np.abs(encoded).argmax(), encoded.shape)
        maximum = dict(abs_value=float(abs(encoded[step, index])),
            signed_value=float(encoded[step, index]), field=field_name(index),
            state_step=int(step), next_action_step=int(step + 1),
            raw_value=float(inputs[step, index]), mean=norm['mean'][index], scale=norm['std'][index])
    width = observations[:, 7:9].sum(1)
    lift = {}
    for color in ('red', 'blue'):
        start, _ = SLICES[color + '_pose']
        z = observations[:, start + 2]
        indices = np.flatnonzero(z > z[0] + .03)
        lift[color] = dict(initial_z_m=float(z[0]), peak_z_m=float(z.max()),
            first_3cm_rise_step=int(indices[0]) if len(indices) else None)
    longest = current = 0
    for value in width:
        current = current + 1 if value <= .005 else 0
        longest = max(longest, current)
    return dict(task=result['task_id'], method=result['method'], condition=result['condition'], seed=result['seed'],
        result_sha256=digest(root / 'result.json'), trace_sha256=digest(root / 'trace.jsonl') if actions else None,
        success=result['success'], status=result['status'], steps=result['steps'],
        first_goals=first_goals, final_goals=result['final'], maximum_normalized_action_input=maximum,
        object_heights=lift, finger_width_min_m=float(width.min()), finger_width_max_m=float(width.max()),
        longest_5mm_or_less_width_run_states=longest,
        gripper_action_range=[float(np.min(np.asarray(actions)[:, 7])), float(np.max(np.asarray(actions)[:, 7]))] if actions else None)


def main():
    previous = read(OUTPUT)['episodes'] if OUTPUT.exists() else []
    cache = {(r['task'], r['method'], r['condition'], r['seed']): r for r in previous}
    rows = []
    for name in NAMES:
        cfg = config(name)
        shared = read(cfg['output'] / 'normalization.json')['normalizer']
        original = read(ROOT / 'runs/exp2/m0/training/result.json')['spec']['normalizer'] if name == 'drawer_exchange' else shared
        for result_path in sorted((BASE / name / 'evaluation').glob('*/*/*/result.json')):
            root = result_path.parent
            method, condition, seed = root.parts[-3:]
            exited = BASE / 'batch/jobs' / name / method / condition / seed / 'process_result.json'
            if not exited.exists():
                continue
            assert read(exited)['returncode'] == 0
            key = (name, method, condition, int(seed))
            if key in cache:
                row = cache[key]
                assert row['result_sha256'] == digest(result_path)
                if row['trace_sha256']:
                    assert row['trace_sha256'] == digest(root / 'trace.jsonl')
            else:
                row = analyze(root, original if method == 'naive_DP' else shared)
            rows.append(row)
    groups = []
    for name in NAMES:
        for condition in ('ID', 'OOD'):
            for method in ('naive_DP', 'APPL'):
                group = [r for r in rows if (r['task'], r['condition'], r['method']) == (name, condition, method)]
                maxima = [r['maximum_normalized_action_input']['abs_value'] for r in group if r['maximum_normalized_action_input']]
                groups.append(dict(task=name, condition=condition, method=method, completed=len(group),
                    success=sum(r['success'] for r in group), stop_status_counts=dict(Counter(r['status'] for r in group)),
                    ever_goal_counts=dict(Counter(k for r in group for k in r['first_goals'] if k != 'success')),
                    final_goal_counts=dict(Counter(k for r in group for k, v in r['final_goals'].items() if v and k != 'success')),
                    ever_3cm_rise_counts={c: sum(r['object_heights'][c]['first_3cm_rise_step'] is not None for r in group) for c in ('red', 'blue')},
                    normalized_maximum_range=[min(maxima), max(maxima)] if maxima else None))
    atomic(OUTPUT, dict(reviewed_utc=datetime.now(timezone.utc).isoformat(), completed=len(rows), planned=600,
        all_completed=len(rows) == 600, episodes=rows, groups=groups,
        scope='Exploratory description of recorded frozen trials; zero new API requests, optimizer updates or physical steps.',
        interpretation='A 3 cm rise and <=5 mm finger width are descriptive analysis conventions, not changes to success or proof of grasp. Normalized magnitudes above training support do not alone establish a normalization bug. Action inputs exclude the terminal state.'))
    print(dict(completed=len(rows), output=str(OUTPUT)), flush=True)


if __name__ == '__main__':
    main()
