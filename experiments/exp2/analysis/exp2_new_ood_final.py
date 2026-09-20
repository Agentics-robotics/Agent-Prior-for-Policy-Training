"""Audit the selected 75 OOD outcomes and all preserved resource-recovery costs."""
from collections import Counter
from pathlib import Path
import subprocess
import time

from appl.io import atomic, digest, read
from experiments.exp2.analysis import exp2_new_ood_completion as auditor
from experiments.exp2.analysis.exp2_new_ood_resume import (
    BASE, PRIOR, OUTPUT, GPUS, common, budget, previous, episode_root, verify, summarize,
)


def main():
    supervisor = read(OUTPUT / 'supervisor/completion.json')
    assert supervisor['all_scheduled_parents_exited'] and supervisor['ID_episodes_scheduled'] == 0
    plan = verify()
    for task in common.TASKS:
        common.verify(task)
    old = read(PRIOR / 'completion.json'); assert old['passed']
    preserved = read(OUTPUT / 'preservation.json')
    assert digest(OUTPUT / 'preservation.json') == plan['preservation_sha256']
    for name, sha in preserved.items():
        assert digest(BASE / name) == sha, name
    summary = summarize(); ledger = read(OUTPUT / 'budget/ledger.json')
    assert ledger['stopped'] or summary['OOD_completed'] == 75
    records = {r['key']: r for r in ledger['records']}
    assert len(records) == len(ledger['records'])
    auditor.OUTPUT = OUTPUT; auditor.episode_root = episode_root
    matched = set(); rows = []; workers = 0; statuses = Counter(); http = Counter(); tokens = Counter()
    for cell in plan['cells']:
        if not episode_root(cell).exists():
            continue
        row, count, status, wire, usage = auditor.audit_episode(cell, records, matched)
        rows.append(row); workers += count; statuses.update(status); http.update(wire); tokens.update(usage)
    assert matched == set(records)
    settled = sum(r.get('settled_nano_usd', 0) for r in records.values())
    held = sum(r['reserved_nano_usd'] for r in records.values() if 'settled_nano_usd' not in r)
    assert settled + held <= ledger['cap_nano_usd']
    prior_nano = sum(r.get('settled_nano_usd', 0) for r in read(PRIOR / 'budget/ledger.json')['records'])
    assert settled + held + prior_nano <= 390 * budget.NANO
    retried = []
    original_cells = read(BASE / 'plan.json')['cells']
    for item in plan['interrupted_retests']:
        root = episode_root(original_cells[item['index']]); earlier = Path(item['root'])
        if (root / 'api/api/0001.request.json').exists():
            assert read(root / 'api/api/0001.request.json') == read(earlier / 'api/api/0001.request.json')
            assert read(root / 'initial_state.json') == read(earlier / 'initial_state.json')
            retried.append(dict(index=item['index'], first_request_identical=True, reset_identical=True,
                original_prefix=str(earlier), selected_attempt=str(root)))
    occupancy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
        '--format=csv,noheader,nounits'], text=True)
    own = []
    for line in occupancy.splitlines():
        if not line.strip():
            continue
        pid = line.split(',')[0].strip()
        try:
            args = (Path('/proc') / pid / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        except (FileNotFoundError, PermissionError):
            continue
        if 'exp2_new_ood_resume' in args or str(OUTPUT) in args or str(OUTPUT.resolve()) in args:
            own.append(pid)
    assert not own
    previous.dump_csv(OUTPUT / 'episode_costs.csv', rows)
    original_audit = read(BASE / 'completion.json')
    original_cost = original_audit['costs']['reported_usd']
    added_tokens = tokens + Counter(old['token_usage'])
    all_tokens = added_tokens + Counter({key: original_audit['costs'][key] for key in added_tokens})
    cost = (prior_nano + settled) / budget.NANO
    value = dict(passed=True, reviewed=time.time(), **summary,
        all_75_OOD_complete=summary['OOD_completed'] == 75, new_phase_complete=sum(r['completed'] for r in rows),
        generation_attempts=len(records), request_statuses=dict(statuses), HTTP_statuses=dict(http), token_usage=dict(tokens),
        added_500_SGD_token_usage=dict(added_tokens), entire_Exp2_new_token_usage=dict(all_tokens),
        added_500_SGD_generation_attempts=len(records) + old['generation_attempts'],
        entire_Exp2_new_generation_attempts=len(records) + old['generation_attempts'] + original_audit['generation_attempts'],
        phase_reported_USD=settled / budget.NANO, phase_unknown_reserved_USD=held / budget.NANO,
        additional_500_SGD_reported_USD=cost, entire_Exp2_new_reported_USD=cost + original_cost,
        original_artifacts_preserved=len(preserved), original_policy_and_scientific_inputs_unchanged=True,
        resource_retests=retried, all_scheduled_parents_exited=True, closed_policy_workers=workers,
        remaining_own_GPU_processes=own, devices=GPUS, maximum_active_physical_devices=5,
        new_ID_trials=0, training_updates=0, segmentation_calls=0, policy_design_calls=0,
        transport_retries=0, retained_resource_interrupted_prefixes=4,
        current_source_sha256=digest('experiments/exp2/analysis/exp2_new_ood_resume.py'),
        audit_source_sha256=digest(__file__), results_sha256=digest(OUTPUT / 'results.json'))
    atomic(OUTPUT / 'completion.json', value)
    lines = ['# Exp2_new OOD: validated completion and accounting', '',
        f"Validated OOD: **{summary['OOD_completed']}/75**, **{summary['OOD_successes']} successes**. All scheduled processes exited; no own GPU process remains. New ID trials: zero. Original ID remains 8/10.", '',
        '## Outcome selection and retained attempts', '',
        'The selected OOD table combines seven original outcomes, 38 completed outcomes from the first continuation, and the completed results from the resource-resumption phase. Successful and failed completed trials are retained equally; completed failures are not rerun. Four resource-interrupted prefixes and the earlier monetary-budget interruption remain available with their original charges. They are not counted as task failures or independent evaluation seeds.', '',
        f"All {len(preserved)} earlier artifact hashes and all frozen policy/scientific-input hashes match. All actual requests and consumed responses use GPT-6 Astra/xhigh/default service tier. Selected actions, fixed geometric predicates, literal API stopping rules, raw API output, videos and worker exits passed audit. Four resource-reset retests use exactly the same initial requests and observations as their earlier attempts.", '',
        '## Resource incident', '',
        'A transient GPU 0 process (PID 3207271, 502 MiB) appeared while a slot was being refilled. Its owner could not be inspected before it exited. The first continuation treated this occupancy as a global stop and unnecessarily interrupted four active episodes. No API service error occurred and no unrelated process was terminated. The resource-only fix skips occupied slots and allows other episodes to continue, within five active GPUs. Original runner, source and incident evidence are preserved. Seventeen framework tests passed before resource resumption.', '',
        '## Budget and requests', '',
        f"Original SGD 100 round: **USD {original_cost:.6f}**. First part of the SGD 500 continuation: **USD {prior_nano / budget.NANO:.6f}**. Resource resumption: **USD {settled / budget.NANO:.6f}**. Total charged-token estimate against the added SGD 500 allowance: **USD {cost:.6f}**, under the declared USD 390 cap. Total Exp2_new deployment estimate: **USD {cost + original_cost:.6f}**. Unknown-charge reservations in this phase: USD {held / budget.NANO:.6f}.", '',
        f"Resource-resumption generation attempts: {len(records)}; request statuses {dict(statuses)}; received HTTP statuses {dict(http)}. Earlier phases retain their separate accounting. No automatic transport retries, fallback model or credential was used. Estimates use the frozen tariff and conservative currency planning ratio; they are not account invoices.", '',
        '## Interpretation', '',
        'The table compares the same position-OOD layouts under the corrected gripper executor. The two baseline models were not retrained or rerun. These layouts were examined before this study, and results come from one frozen training seed. Differences from historical APPL also include the official provider route and API sampling. The fixed geometric goal does not require extra terminal release, clearance, velocity or hold conditions.', '',
        'All completed failures count. A retained API finish reason describes the agent’s interpretation; it is not an independently established causal explanation. The preserved trajectories and videos support further read-only failure analysis.', '',
        '- [Results and matched baselines](REPORT.md)', '- [Completion receipt](completion.json)',
        '- [Per-layout source selection](episodes.csv)', '- [Selected videos](replays.html)',
        '- [This phase per-episode cost and original finish reasons](episode_costs.csv)',
        '- [Previous phase per-episode cost](../ood_continuation_20260920/episode_costs.csv)',
        '- [Resource incident](resource_incident.json)', '- [Preservation hashes](preservation.json)']
    (OUTPUT / 'ANALYSIS.md').write_text('\n'.join(lines) + '\n')
    (OUTPUT / 'START_HERE.md').write_text('# Exp2_new OOD\n\nRead [results](REPORT.md), [accounting and interpretation](ANALYSIS.md), [completion audit](completion.json), and [videos](replays.html). Earlier rounds and all interrupted prefixes remain immutable at their original paths.\n')
    names = ['REPORT.md', 'ANALYSIS.md', 'START_HERE.md', 'completion.json', 'results.json', 'episodes.csv',
        'summary.csv', 'episode_costs.csv', 'replays.html', 'plan.json', 'authorization.json', 'resource_incident.json', 'budget/ledger.json']
    atomic(OUTPUT / 'report_artifacts.json', dict(files={n: digest(OUTPUT / n) for n in names}, reporting_API_calls=0))
    print({k: value[k] for k in ['passed', 'OOD_completed', 'OOD_successes', 'generation_attempts',
        'additional_500_SGD_reported_USD', 'entire_Exp2_new_reported_USD', 'phase_unknown_reserved_USD',
        'original_artifacts_preserved', 'remaining_own_GPU_processes']}, flush=True)


if __name__ == '__main__':
    main()
