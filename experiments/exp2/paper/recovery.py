"""Apply an audited later recovery to reporting without editing frozen results."""
from collections import Counter
import copy
from pathlib import Path
import shutil

from appl.io import ROOT, atomic, digest, read
from appl.scaleup.report import wilson

RECOVERY = ROOT / 'runs/exp2/M1_scaleup/evaluation_recovery_20260919'


def apply(original, destination):
    receipt = read(RECOVERY / 'completion.json')
    assert receipt['passed'] and receipt['status'] == 'complete'
    assert read(RECOVERY / 'process_result.json')['returncode'] == 0
    for path, sha in read(RECOVERY / 'original_artifacts.json').items():
        assert digest(path) == sha, path
    replacement = receipt['replacement']
    episode = Path(replacement['episode_root'])
    for filename, key in [('result.json', 'result_sha256'), ('trace.jsonl', 'trace_sha256'),
                          ('replay.mp4', 'video_sha256')]:
        assert digest(episode / filename) == receipt[key]
    result = copy.deepcopy(original)
    key = lambda r: (r['task'], r['condition'], r['seed'], r['method'])
    index, = [i for i, r in enumerate(result['episodes']) if key(r) == key(replacement)]
    previous = result['episodes'][index]
    assert not previous['completed'] and previous['status'] == 'interrupted'
    result['episodes'][index] = replacement
    assert sum(a != b for a, b in zip(original['episodes'], result['episodes'])) == 1
    for group in result['groups']:
        rows = [r for r in result['episodes'] if (r['task'], r['condition'], r['method']) ==
                (group['task'], group['condition'], group['method'])]
        n = sum(r['completed'] for r in rows)
        s = sum(r['success'] is True for r in rows)
        group.update(completed=n, success=s, unknown=30-n, rate=s/30 if n == 30 else None,
                     wilson95=wilson(s, 30) if n == 30 else None,
                     success_by={str(cap): sum(r['success_by_' + str(cap)] for r in rows)
                                 for cap in (1500, 3000, 5000)})
    for pair in result['paired']:
        select = lambda method: {r['seed']: r for r in result['episodes']
            if (r['task'], r['condition'], r['method']) == (pair['task'], pair['condition'], method)}
        fresh, reference = select('SinglePrior_6_xhigh'), select(pair['reference'])
        counts = Counter((reference[s]['success'], fresh[s]['success']) for s in fresh if reference[s]['completed'])
        pair.update(complete_pairs=sum(counts.values()), single_only=counts[(False, True)],
                    reference_only=counts[(True, False)], both_success=counts[(True, True)],
                    both_failure=counts[(False, False)])
    result['reporting_recovery'] = dict(completion=str(RECOVERY / 'completion.json'),
        completion_sha256=digest(RECOVERY / 'completion.json'), original_unknown=previous,
        replaced_cells=1, frozen_canonical_results_unchanged=True)
    dest = destination / 'recovery_20260919'
    dest.mkdir(exist_ok=True)
    copies = []
    for source, name in [(RECOVERY / n, n) for n in
        ('completion.json', 'authorization.json', 'preflight.json', 'allocation.json',
         'process.json', 'process_result.json', 'study_freeze.json', 'frozen_recovery_executor.py')]:
        shutil.copyfile(source, dest / name)
        copies.append(dict(source=str(source), bundle_file=name, sha256=digest(source)))
    for name in ('replay.mp4', 'result.json', 'initial_state.json', 'initial.png', 'final.png', 'invocations.json'):
        shutil.copyfile(episode / name, dest / name)
        copies.append(dict(source=str(episode / name), bundle_file=name, sha256=digest(episode / name)))
    atomic(dest / 'index.json', dict(files=copies))
    (dest / 'REPORT.md').write_text(
        '# Authorized recovery of the last unknown outcome\n\n'
        'On 2026-09-19 the user authorized one reset of buffer_swap / ID / 20116. '
        'The original GPT-5.5/high episode stopped after 650 physical steps because '
        'the provider returned an upstream service error. It was neither a task failure nor a success.\n\n'
        f"The replacement {'succeeded' if replacement['success'] else 'completed unsuccessfully'} "
        f"after {replacement['steps']} steps. The main table now contains 1,200 complete cells and zero unknowns. "
        'The original interruption, all earlier frozen reports and the preceding ZIP remain retained. '
        'No completed task failure was retested.\n\n'
        'The actual model/effort remains GPT-5.5/high to reproduce the historical comparison. '
        'New experiment defaults remain xhigh. Initial state, initial prompt and first request '
        'match the original exactly; all actions, API-selected policies and literal stop rules '
        'passed replay audit. No training or policy/prompt changes occurred. The derived source '
        'manifest records the previously audited gradient-disable correction in the zero-update '
        'reload check; the evaluation path is unchanged.\n\n'
        '[Completion and API usage](completion.json) · [Authorization](authorization.json) · '
        '[Recorded video](replay.mp4) · [API invocations](invocations.json) · [Copy provenance](index.json)\n')
    return result


def add_accounting(original):
    value = copy.deepcopy(original)
    added = read(RECOVERY / 'completion.json')['API_accounting']
    for key in ('states', 'usage'):
        counts = Counter(value[key])
        counts.update(added[key])
        value[key] = dict(counts)
    return value
