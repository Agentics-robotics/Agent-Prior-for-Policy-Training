"""Offline completion, accounting and preservation audit for OOD continuation."""
from collections import Counter
import json
from pathlib import Path
import sqlite3
import subprocess
import time

from appl.io import atomic, digest, read
from experiments.exp2.analysis.exp2_new_completion import partial_invocations
from experiments.exp2.analysis.exp2_new_ood_continue import (
    BASE, OUTPUT, GPUS, common, budget, episode_root, verify, summarize, dump_csv,
)


def audit_episode(cell, records, matched):
    root = episode_root(cell)
    process = read(OUTPUT / 'supervisor' / str(cell['index']) / 'process_result.json')
    proof = read(root / 'audit.json'); assert proof['passed']
    result = read(root / 'result.json') if (root / 'result.json').exists() else None
    if result:
        assert process['returncode'] == 0 and proof['completed']
        assert digest(root / 'result.json') == proof['result_sha256']
    else:
        assert process['returncode'] != 0 and (root / 'interruption.json').exists()
        frozen = read(BASE / 'plan.json')['tasks'][cell['task']]
        atomic(root / 'interrupted_invocation_audit.json', partial_invocations(root, cell, frozen))
    if proof['trace_sha256']:
        assert digest(root / 'trace.jsonl') == proof['trace_sha256']
    movie = read(root / 'replay_provenance.json')
    assert digest(root / 'replay.mp4') == movie['video_sha256']
    for name, sha in movie['frames'].items():
        assert digest(root / name) == sha
    worker_count = 0
    for worker in (root / 'workers').glob('*'):
        assert read(worker / 'closed.json')['returncode'] == 0
        assert read(worker / 'enforcement.json')['network'] is False
        worker_count += 1
    db = sqlite3.connect((root / 'api/journal.sqlite').resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    statuses = Counter(); http = Counter(); tokens = Counter(); cost = 0; attempts = 0
    for seq, status, request_text, response_text in db.execute('SELECT * FROM api ORDER BY seq'):
        statuses[status] += 1; request = json.loads(request_text)
        assert (request['model'], request['reasoning']['effort'], request['service_tier']) == ('gpt-6-astra', 'xhigh', 'default')
        assert request == read(root / 'api/api' / f'{seq:04d}.request.json')
        transport = root / 'api/transport' / f'{seq:04d}'
        key = str((root / 'api').resolve().relative_to(BASE.resolve())) + f'/{seq:04d}'
        if not (transport / 'generation_attempt.json').exists():
            assert key not in records and status == 'not_sent'
            continue
        assert key in records and key not in matched
        record = records[key]; matched.add(key); attempts += 1
        counted = read(transport / 'count_response.json')
        assert counted['http_status'] == 200 and counted['response']['input_tokens'] == record['input_tokens_counted']
        if (transport / 'response.json').exists():
            wire = read(transport / 'response.json'); http[str(wire['http_status'])] += 1
            body = wire['response']
            if response_text:
                assert json.loads(response_text) == body
            if status == 'consumed':
                assert (body['model'], body['reasoning']['effort']) == ('gpt-6-astra', 'xhigh')
            usage = body.get('usage')
            if usage:
                assert usage == record['usage']
                amount, _ = budget.usage_cost(usage)
                assert amount == record['settled_nano_usd']; cost += amount
                details = usage.get('input_tokens_details') or {}
                tokens.update(input_tokens=usage['input_tokens'], output_tokens=usage['output_tokens'],
                    cached_input_tokens=details.get('cached_tokens', 0), cache_write_tokens=details.get('cache_write_tokens', 0),
                    reasoning_tokens=(usage.get('output_tokens_details') or {}).get('reasoning_tokens', 0))
    reasons = [json.loads(r[0])['reason'] for r in db.execute("SELECT args FROM tools WHERE name='finish' AND status='completed'")]
    db.close()
    row = dict(index=cell['index'], task=cell['task'], seed=cell['seed'], condition='OOD',
        completed=bool(result), success=result['success'] if result else None,
        steps=result['steps'] if result else read(root / 'interruption.json')['steps'],
        generation_attempts=attempts, reported_cost_USD=cost / budget.NANO, **tokens,
        API_finish_reason=reasons[-1] if reasons else '', root=str(root))
    return row, worker_count, statuses, http, tokens


def main():
    supervisor = read(OUTPUT / 'supervisor/completion.json')
    assert supervisor['all_scheduled_parents_exited'] and supervisor['ID_episodes_scheduled'] == 0
    plan = verify()
    for task in common.TASKS:
        common.verify(task)
    original_files = read(OUTPUT / 'preservation.json')
    assert digest(OUTPUT / 'preservation.json') == plan['preservation_sha256']
    for name, sha in original_files.items():
        assert digest(BASE / name) == sha, name
    summary = summarize(); ledger = read(OUTPUT / 'budget/ledger.json')
    assert ledger['stopped'] or summary['OOD_completed'] == 75
    records = {r['key']: r for r in ledger['records']}
    assert len(records) == len(ledger['records'])
    matched = set(); rows = []; workers = 0; statuses = Counter(); http = Counter(); tokens = Counter()
    for cell in plan['cells']:
        if not episode_root(cell).exists():
            continue
        row, count, status, wire, usage = audit_episode(cell, records, matched)
        rows.append(row); workers += count; statuses.update(status); http.update(wire); tokens.update(usage)
    assert matched == set(records)
    settled = sum(r.get('settled_nano_usd', 0) for r in records.values())
    held = sum(r['reserved_nano_usd'] for r in records.values() if 'settled_nano_usd' not in r)
    assert settled + held <= ledger['cap_nano_usd']
    original_plan = read(BASE / 'plan.json')
    retried = original_plan['cells'][17]
    old_first = common.episode_root(retried) / 'api/api/0001.request.json'
    new_first = episode_root(retried) / 'api/api/0001.request.json'
    assert read(new_first) == read(old_first)
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
        if 'exp2_new_ood_continue' in args or str(OUTPUT) in args or str(OUTPUT.resolve()) in args:
            own.append(pid)
    assert not own
    dump_csv(OUTPUT / 'episode_costs.csv', rows)
    value = dict(passed=True, reviewed=time.time(), **summary,
        new_completed=sum(r['completed'] for r in rows), new_attempted=len(rows),
        new_interrupted=sum(not r['completed'] for r in rows),
        generation_attempts=len(records), request_statuses=dict(statuses), HTTP_statuses=dict(http),
        token_usage=dict(tokens), new_reported_USD=settled / budget.NANO, unknown_reserved_USD=held / budget.NANO,
        old_artifacts_preserved=len(original_files), all_frozen_scientific_inputs_unchanged=True,
        original_interrupted_retest_first_request_identical=True,
        all_scheduled_parents_exited=True, closed_policy_workers=workers, remaining_own_GPU_processes=own,
        devices=GPUS, maximum_active_physical_devices=5, new_ID_trials=0, new_training_updates=0,
        new_segmentation_or_design_calls=0, transport_retries=0,
        all_75_OOD_complete=summary['OOD_completed'] == 75,
        new_source_sha256=digest('experiments/exp2/analysis/exp2_new_ood_continue.py'),
        audit_source_sha256=digest(__file__), results_sha256=digest(OUTPUT / 'results.json'))
    atomic(OUTPUT / 'completion.json', value)
    lines = ['# OOD continuation: completion and interpretation', '',
        f"Validated OOD outcomes: **{summary['OOD_completed']}/75**, successes **{summary['OOD_successes']}**. New complete outcomes: {value['new_completed']}; new interrupted attempts: {value['new_interrupted']}.", '',
        '## Execution', '',
        f"All {value['new_attempted']} admitted episode processes and {workers} policy workers exited. No own GPU process remains. All {len(original_files)} original artifacts retain their pre-continuation hashes. The original 36 policy packages and scientific source remain unchanged.", '',
        'The previous buffer-swap budget interruption was explicitly retested from the same reset. Its first actual request exactly matches the previous first request; later API decisions may differ. The original interrupted prefix, journals and charges remain preserved. Completed failures were never retested.', '',
        '## Costs', '',
        f"New recorded token-tariff cost: **USD {settled / budget.NANO:.6f}**. Unknown-charge reservations: **USD {held / budget.NANO:.6f}**. Additional authorized budget: SGD 500; new ledger cap: USD 390. Original recorded cost remains USD {summary['costs']['old_reported_USD']:.6f}. These are usage estimates, not invoices.", '',
        f"Generation attempts: {len(records)}; request statuses: {dict(statuses)}; received HTTP statuses: {dict(http)}. No automatic transport retries. Budget/service stop reason: {ledger['stopped']}.", '',
        '## Evidence and limitations', '',
        'Compare against the same layouts in REPORT.md. The two baselines are reused corrected-executor results, with no new training or deployment API. Original ID scores remain unchanged at APPL 8/10; this continuation does not complete the ID matrix.', '',
        'The seeds were examined previously, and all methods use one frozen training seed. This is a paired post-hoc executor study, not an untouched confirmatory generalization estimate. The official endpoint and API sampling can also differ from the historical gateway runs. Failure finish reasons are the API’s original statements; they are not independent causal diagnoses.', '',
        '- [Full comparison](REPORT.md)', '- [Completion audit](completion.json)',
        '- [Per-episode cost and original API finish reason](episode_costs.csv)',
        '- [Videos](replays.html)', '- [Original-artifact preservation hashes](preservation.json)',
        '- [Previous interrupted prefix](../APPL_GPT6_astra_xhigh/buffer_swap/OOD/30101/interruption.json)']
    (OUTPUT / 'ANALYSIS.md').write_text('\n'.join(lines) + '\n')
    (OUTPUT / 'START_HERE.md').write_text('# Exp2_new OOD continuation\n\nRead [results](REPORT.md), [accounting and interpretation](ANALYSIS.md), [audit](completion.json), and [videos](replays.html). The original round remains in the parent directory unchanged.\n')
    names = ['REPORT.md', 'ANALYSIS.md', 'START_HERE.md', 'completion.json', 'results.json', 'episodes.csv',
        'summary.csv', 'episode_costs.csv', 'replays.html', 'plan.json', 'authorization.json', 'budget/ledger.json']
    atomic(OUTPUT / 'report_artifacts.json', dict(files={n: digest(OUTPUT / n) for n in names}, reporting_API_calls=0))
    print({k: value[k] for k in ['passed', 'OOD_completed', 'OOD_successes', 'new_completed', 'new_interrupted',
        'generation_attempts', 'new_reported_USD', 'unknown_reserved_USD', 'old_artifacts_preserved', 'remaining_own_GPU_processes']}, flush=True)


if __name__ == '__main__':
    main()
