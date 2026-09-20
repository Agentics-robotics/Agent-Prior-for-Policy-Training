"""Audit retained evaluation interruptions and describe a bounded retest proposal.

This is an offline analysis. It performs no API requests or simulator steps and
does not authorize or execute the proposed retests.
"""
from collections import Counter
import json
import time

from appl.io import atomic, digest, object_hash, read
from appl.prior_policies.report import api_accounting
from appl.scaleup.report import audit_episode
from experiments.exp2.analysis.audit_scaleup import audit_api
from .runner import BASE, OLD, api_identity, config


def main():
    target = BASE / 'incidents/evaluation_outage'
    freeze = read(BASE / 'study_freeze.json')
    rows = []
    for path in sorted(BASE.glob('*/evaluation/APPL/*/*/failure.json')):
        root = path.parent
        name, condition, seed = path.parts[-6], path.parts[-3], int(path.parts[-2])
        failure = read(path)
        job = BASE / 'jobs/evaluation' / name / condition / str(seed)
        assert read(job / 'process_result.json')['returncode'] != 0
        assert not (root / 'result.json').exists()
        assert failure['error'] == 'Provider HTTP failure retained; no automatic retry'
        initial = read(root / 'initial_state.json')
        assert initial == read(OLD / name / 'evaluation/naive_DP' / condition / str(seed) / 'initial_state.json')
        contract = read(config(name)['completion_contract'])
        measured = audit_episode(root, contract, None)
        assert measured['steps'] == failure['physical_steps']
        assert measured['first_success'] is None
        trace_path = root / 'trace.jsonl'
        trace = [json.loads(line) for line in trace_path.read_text().splitlines()] if trace_path.exists() else []
        identity = api_identity(root / 'api/journal.sqlite', require_consumed=False)
        api = audit_api(root, contract, trace, initial, None, freeze['study']['tasks'][name], True)
        response_path = sorted((root / 'api/api').glob('*.response.json'))[-1]
        response = read(response_path)
        assert response['http_status'] in (429, 502, 503)
        rows.append(dict(task=name, condition=condition, seed=seed,
            root=str(root), process_root=str(job), physical_steps=measured['steps'],
            failed_API_call=int(response_path.name.split('.')[0]),
            http_status=response['http_status'], provider_error=response['response']['error'],
            final_outcome_known=False, paired_reset_verified=True,
            prefix_audit=measured, API_audit=api, model_identity=identity,
            API_accounting=api_accounting(root / 'api/journal.sqlite'),
            failure_sha256=digest(path), response_sha256=digest(response_path),
            trace_sha256=digest(trace_path) if trace_path.exists() else None))
    results = read(BASE / 'results.json')
    assert len(rows) == results['summary']['new_unknown'] == 89
    record = dict(reviewed=time.time(), passed=True, interrupted=len(rows),
        HTTP_status_counts=dict(Counter(row['http_status'] for row in rows)),
        zero_step_attempts=sum(row['physical_steps'] == 0 for row in rows),
        partial_attempts=sum(row['physical_steps'] > 0 for row in rows),
        completed_outcomes=results['summary']['new_completed'],
        confirmed_successes=results['summary']['new_successes'],
        completed_API_audit_sha256=digest(BASE / 'evaluation_audit.json'),
        original_completion_sha256=digest(BASE / 'completion.json'),
        study_sha256=freeze['study_sha256'], audit_source_sha256=digest(__file__),
        API_requests_added=0, simulator_steps_added=0, automatic_retries=0, records=rows)
    atomic(target / 'audit.json', record)
    plan = dict(status='proposal_only_not_authorized_or_executed',
        scope='One fresh reset retest for each of the 89 HTTP-interrupted new-study cells only.',
        cases=[dict(task=r['task'], condition=r['condition'], seed=r['seed'],
            original_root=r['root'], original_failure_sha256=r['failure_sha256']) for r in rows],
        completed_211_cells='Retain unchanged; no retest, including all 70 task failures.',
        outputs=str(BASE / 'evaluation_recovery'),
        original_attempts='Retain in place, including original unknown outcomes and all API costs.',
        protocol='Same frozen models, inputs, paired reset seeds, English prompts, gpt-6-astra/xhigh, and executor-only 5000-step limit.',
        execution='First run one zero-step interrupted cell as the actual retest. Admit remaining cells only if it exits without transport error. At any new transport error, stop admitting cells and let active cells exit; no second retry.',
        concurrency=dict(physical_GPUs=[1,2,3,5], episode_slots=4, API_requests=4),
        maximum_new_episodes=89, maximum_steps_per_episode=5000,
        maximum_API_requests_per_episode=128, additional_training_updates=0,
        exact_physics_resume=False,
        comparability='Reset retests, not continuations; API outputs are not seeded, so partial trajectories can differ.',
        outcome_rule='Use the one declared retest per interrupted cell; do not select among attempts by success. Keep original and retest results separately.',
        reason_confirmation_needed='The recorded four-design recovery authorization does not cover these later evaluation interruptions; scoped AGENTS.md forbids automatic provider retries.',
        audit_sha256=digest(target / 'audit.json'), study_sha256=freeze['study_sha256'])
    plan['proposal_sha256'] = object_hash(plan)
    atomic(target / 'retest_proposal.json', plan)
    print({k: record[k] for k in ('passed','interrupted','HTTP_status_counts','zero_step_attempts','partial_attempts','completed_outcomes','confirmed_successes')}, flush=True)


if __name__ == '__main__':
    main()
