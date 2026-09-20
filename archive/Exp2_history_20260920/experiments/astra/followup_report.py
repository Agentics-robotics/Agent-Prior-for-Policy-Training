"""Audit the 21-case follow-up and combine declared attempts without overwriting history."""
from collections import Counter
import csv
import json
import math
from pathlib import Path
import shutil

from appl.io import atomic, digest, read
from appl.prior_policies.report import api_accounting
from appl.scaleup.report import audit_episode, video, wilson
from experiments.exp2.analysis.audit_scaleup import audit_api
from . import followup_evaluation as followup
from . import recovery_report as rendering
from .report import verify_video
from .runner import BASE, OLD, NAMES, api_identity, config


def generate():
    followup.bind()
    out, prior = followup.OUT, followup.PRIOR
    planned = followup.authorized_cases()
    execution = read(out / 'execution.json')
    if len(execution['finished']) + len(execution['unstarted']) != len(planned):
        raise ValueError('Follow-up schedule does not match the authorized cases')
    before = read(out / 'original_artifacts.json')
    if before != followup.inventory():
        raise ValueError('Earlier artifacts changed')
    earlier = read(prior / 'results.json')
    rows = [dict(r) for r in earlier['episodes']]
    frozen = read(BASE / 'study_freeze.json')['study']
    audits, accounts, clips = [], [], []
    for case in planned:
        name, condition, seed = followup.key(case)
        root = out / name / 'evaluation/APPL' / condition / str(seed)
        job = out / 'jobs' / name / condition / str(seed)
        index, = [i for i, r in enumerate(rows) if
                  (r['task'], r['condition'], r['seed'], r['method']) ==
                  (name, condition, seed, 'APPL_6_xhigh')]
        if rows[index]['completed']:
            raise ValueError('A completed outcome cannot receive another attempt')
        if not (job / 'process_result.json').exists():
            if root.exists():
                raise ValueError('Follow-up episode has no final process receipt')
            continue
        process = read(job / 'process_result.json')
        result = read(root / 'result.json') if (root / 'result.json').exists() else None
        if (process['returncode'] == 0) != (result is not None):
            raise ValueError('Process exit and result disagree')
        contract = read(config(name)['completion_contract'])
        measured = audit_episode(root, contract, result)
        original_root = Path(case['original_root'])
        initial = read(root / 'initial_state.json')
        if (initial != read(original_root / 'initial_state.json') or initial != read(
                OLD / name / 'evaluation/naive_DP' / condition / str(seed) / 'initial_state.json')):
            raise ValueError('Paired reset changed')
        if read(root / 'prompt.json') != read(original_root / 'prompt.json'):
            raise ValueError('Initial prompt changed')
        if read(root / 'api/api/0001.request.json') != read(original_root / 'api/api/0001.request.json'):
            raise ValueError('First request changed')
        trace_path = root / 'trace.jsonl'
        trace = [json.loads(line) for line in trace_path.read_text().splitlines()] if trace_path.exists() else []
        if result:
            if result['steps'] > 5000 or (result['success'] and measured['first_success'] != result['steps']):
                raise ValueError('Invalid success or physical stopping boundary')
        else:
            failure = read(root / 'failure.json')
            if failure['error'] != 'Provider HTTP failure retained; no automatic retry':
                raise ValueError('Non-HTTP interruption needs separate diagnosis')
            if failure['physical_steps'] != measured['steps'] or measured['first_success'] is not None:
                raise ValueError('Interrupted prefix mismatch')
        api = audit_api(root, contract, trace, initial, result, frozen['tasks'][name], result is None)
        identity = api_identity(root / 'api/journal.sqlite', require_consumed=result is not None)
        accounts.append(dict(task=name, condition=condition, seed=seed,
                             **api_accounting(root / 'api/journal.sqlite')))
        audits.append(dict(task=name, condition=condition, seed=seed, passed=True,
            completed=result is not None, prefix=measured, API_audit=api, model_identity=identity,
            initial_state_identical=True, initial_prompt_identical=True, first_request_identical=True,
            result_sha256=digest(root / 'result.json') if result else None,
            failure_sha256=digest(root / 'failure.json') if not result else None,
            trace_sha256=digest(trace_path) if trace_path.exists() else None))
        row = dict(task=name, condition=condition, seed=seed, method='APPL_6_xhigh',
            completed=result is not None, success=result['success'] if result else None,
            status=result['status'] if result else 'interrupted', **measured,
            reused_original_result=False, selected_attempt='authorized_followup_20260918',
            original_episode_root=str(original_root), episode_root=str(root))
        for cap in (1500, 3000, 5000):
            row['success_by_' + str(cap)] = measured['first_success'] is not None and measured['first_success'] <= cap
        rows[index] = row
        path = video(root)
        if path:
            clips.append(dict(task=name, condition=condition, seed=seed, path=str(path),
                              sha256=digest(path), validation=verify_video(root)))
    groups, paired = [], []
    for name in NAMES:
        for condition in ('ID', 'OOD'):
            for method in rendering.METHODS:
                es = [r for r in rows if (r['task'], r['condition'], r['method']) == (name, condition, method)]
                if len(es) != 30:
                    raise ValueError('Incomplete comparison matrix')
                successes, complete = sum(r['success'] is True for r in es), sum(r['completed'] for r in es)
                groups.append(dict(task=name, condition=condition, method=method, success=successes,
                    completed=complete, planned=30, unknown=30-complete,
                    rate=successes/30 if complete == 30 else None,
                    wilson95=wilson(successes, 30) if complete == 30 else None))
            for reference in rendering.METHODS[:2]:
                new = {r['seed']: r for r in rows if (r['task'], r['condition'], r['method']) == (name, condition, 'APPL_6_xhigh')}
                old = {r['seed']: r for r in rows if (r['task'], r['condition'], r['method']) == (name, condition, reference)}
                cells = Counter((old[s]['success'], new[s]['success']) for s in new if old[s]['completed'] and new[s]['completed'])
                win, loss = cells[(False, True)], cells[(True, False)]
                n = win + loss
                p = min(1., 2*sum(math.comb(n, i) for i in range(min(win, loss)+1))/2**n) if n else 1.
                paired.append(dict(task=name, condition=condition, reference=reference,
                    complete_pairs=sum(cells.values()), new_only=win, reference_only=loss,
                    both_success=cells[(True, True)], both_failure=cells[(False, False)],
                    exact_mcnemar_p=p, interpretation='Descriptive complete pairs; unadjusted, one training seed, reused layouts.'))
    usage, states = Counter(earlier['summary']['reported_API_usage']), Counter(earlier['summary']['API_states'])
    for account in accounts:
        usage.update(account['usage'])
        states.update(account['states'])
    fresh = [r for r in rows if r['method'] == 'APPL_6_xhigh']
    failures = [r for r in fresh if r['completed'] and not r['success']]
    summary = dict(earlier['summary'], complete=sum(r['completed'] for r in fresh),
        successes=sum(r['success'] is True for r in fresh), task_failures=len(failures),
        unknown=sum(not r['completed'] for r in fresh),
        resumed_attempts=earlier['summary']['resumed_attempts']+len(audits),
        resumed_complete_outcomes=earlier['summary']['resumed_complete_outcomes']+sum(a['completed'] for a in audits),
        followup_attempts=len(audits), followup_complete_outcomes=sum(a['completed'] for a in audits),
        API_states=dict(states), reported_API_usage=dict(usage),
        failure_statuses=dict(Counter(r['status'] for r in failures)),
        failure_step_range=[min(r['steps'] for r in failures), max(r['steps'] for r in failures)] if failures else None,
        success_by={str(cap):sum(r['success_by_'+str(cap)] for r in fresh) for cap in (1500,3000,5000)},
        original_files_verified_unchanged=len(before), all_300_outcomes_complete=all(r['completed'] for r in fresh))
    shutil.copyfile(__file__, out / 'report_source.py')
    shutil.copyfile(rendering.__file__, out / 'report_rendering_source.py')
    atomic(out / 'results.json', dict(summary=summary, groups=groups, episodes=rows, paired=paired,
        followup_API_accounting=accounts, previous_results_sha256=digest(prior / 'results.json'),
        original_results_sha256=digest(BASE / 'results.json'), execution_sha256=digest(out / 'execution.json'),
        study_sha256=read(BASE / 'study_freeze.json')['study_sha256'],
        report_source_sha256=digest(__file__), rendering_source_sha256=digest(rendering.__file__)))
    atomic(out / 'evaluation_audit.json', dict(passed=True, followup_records=audits,
        retained_previous_audit_sha256=digest(prior / 'evaluation_audit.json'),
        retained_original_audit_sha256=digest(BASE / 'evaluation_audit.json'),
        retained_original_interruption_audit_sha256=digest(BASE / 'incidents/evaluation_outage/audit.json'),
        earlier_files_unchanged=len(before)))
    atomic(out / 'videos.json', dict(followup_videos=clips,
        retained_previous_videos_sha256=digest(prior / 'videos.json'),
        retained_original_videos_sha256=digest(BASE / 'videos.json')))
    with (out / 'episodes.csv').open('w') as handle:
        fields = ['task','condition','method','seed','status','completed','success','steps',
                  'first_success','selected_attempt','episode_root']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in rows)
    rendering.render(out, rows, groups, summary)
    rendering.analyze(out, rows, groups, summary, paired)
    with (out / 'ANALYSIS.md').open('a') as handle:
        handle.write('\n## Second explicitly authorized continuation\n\n'
            'The service-restored round stopped after 69 attempts: 68 complete outcomes and one HTTP 502 '
            '(two_block_sort / OOD / 30027, step 679), leaving 20 cases unstarted. '
            'The user explicitly authorized this exact 21-case follow-up. All 69 preceding attempts, '
            'including the interrupted trace and its costs, remain in '
            '[the preceding report](../evaluation_recovery_20260918/REPORT.md). '
            'The cumulative resumed-attempt count includes both rounds; no completed task failure was retested.\n')
    evidence = read(out / 'analysis_evidence.json')
    evidence['analysis_sha256'] = digest(out / 'ANALYSIS.md')
    atomic(out / 'analysis_evidence.json', evidence)
    receipt = dict(all_300_new_outcomes_complete=summary['all_300_outcomes_complete'],
        original_complete_outcomes_retained=211, prior_recovery_complete_outcomes_retained=68,
        followup_outcomes_audited=len(audits), followup_complete_outcomes=summary['followup_complete_outcomes'],
        complete_outcomes=summary['complete'], successes=summary['successes'], task_failures=summary['task_failures'],
        unknown=summary['unknown'], original_files_unchanged=len(before), selected_new_study_videos=300,
        new_video_validations=len(clips), original_attempts_retained=True, previous_attempts_retained=True,
        automatic_retries=0, added_training_updates=0,
        artifacts={name:digest(out/name) for name in ('results.json','episodes.csv','REPORT.md','ANALYSIS.md',
            'analysis_evidence.json','evaluation_audit.json','videos.json','comparison.png',
            'replays.html','paired_examples.html','report_source.py','report_rendering_source.py')})
    atomic(out / 'report_receipt.json', receipt)
    if summary['all_300_outcomes_complete']:
        if not execution['all_21_completed'] or len(audits) != 21 or len(clips) != 21:
            raise ValueError('Completion requires all authorized outcomes and videos')
        atomic(out / 'completion.json', dict(status='complete', **receipt))
    return summary


if __name__ == '__main__':
    print(generate(), flush=True)
