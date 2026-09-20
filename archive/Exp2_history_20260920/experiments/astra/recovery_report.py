"""Audit the authorized resumed evaluations and assemble an explicit outcome view.

Original results stay in place. Each originally unknown cell is represented by
its one declared resumed attempt, never by a success-selected attempt.
"""
from collections import Counter
import csv
import html
import json
import math
import os
from pathlib import Path
import shutil

from appl.io import ROOT, atomic, digest, read
from appl.prior_policies.report import api_accounting
from appl.scaleup.report import audit_episode, video, wilson
from experiments.exp2.analysis.audit_scaleup import audit_api
from . import recover_evaluation as recovery
from .report import verify_video
from .runner import BASE, OLD, NAMES, api_identity, config, original_config

METHODS = ['naive_DP', 'APPL_5_5_high', 'APPL_6_xhigh']
LABELS = ['DP (reused)', 'APPL 5.5/high (reused)', 'APPL 6/xhigh']


def generate():
    recovery.select_round('20260918')
    out = recovery.OUT
    execution = read(out / 'execution.json')
    shutil.copyfile(__file__, out / 'report_source.py')
    planned = recovery.authorized_cases()
    if len(execution['finished']) + len(execution['unstarted']) != len(planned):
        raise ValueError('Resumed schedule does not match the authorized matrix')
    before = read(out / 'original_artifacts.json')
    if before != recovery.inventory():
        raise ValueError('Original artifacts changed')
    frozen = read(BASE / 'study_freeze.json')['study']
    original = read(BASE / 'results.json')
    rows = [dict(r, selected_attempt='original') for r in original['episodes']]
    audits, accounts, clips = [], [], []
    for case in planned:
        name, condition, seed = recovery.key(case)
        old_root = Path(case['original_root'])
        root = out / name / 'evaluation/APPL' / condition / str(seed)
        matches = [i for i, row in enumerate(rows) if
                   (row['task'], row['condition'], row['seed'], row['method']) ==
                   (name, condition, seed, 'APPL_6_xhigh')]
        index, = matches
        if rows[index]['completed']:
            raise ValueError('Cannot replace an originally complete outcome')
        job = out / 'jobs' / name / condition / str(seed)
        if not (job / 'process_result.json').exists():
            if root.exists():
                raise ValueError('Resumed attempt has no final process receipt')
            rows[index]['selected_attempt'] = 'original_unknown_retest_not_started'
            continue
        process = read(job / 'process_result.json')
        result = read(root / 'result.json') if (root / 'result.json').exists() else None
        if (process['returncode'] == 0) != (result is not None):
            raise ValueError('Result and process exit disagree')
        contract = read(config(name)['completion_contract'])
        measured = audit_episode(root, contract, result)
        initial = read(root / 'initial_state.json')
        if initial != read(old_root / 'initial_state.json') or initial != read(
                OLD / name / 'evaluation/naive_DP' / condition / str(seed) / 'initial_state.json'):
            raise ValueError('Resumed reset differs from the original paired layout')
        if read(root / 'prompt.json') != read(old_root / 'prompt.json'):
            raise ValueError('Initial API prompt/input changed')
        if read(root / 'api/api/0001.request.json') != read(old_root / 'api/api/0001.request.json'):
            raise ValueError('First API request changed')
        trace_path = root / 'trace.jsonl'
        trace = [json.loads(line) for line in trace_path.read_text().splitlines()] if trace_path.exists() else []
        if result:
            if result['steps'] > 5000 or (result['success'] and measured['first_success'] != result['steps']):
                raise ValueError('Incorrect physical or success stopping boundary')
        else:
            failure = read(root / 'failure.json')
            if failure['error'] != 'Provider HTTP failure retained; no automatic retry':
                raise ValueError('Non-HTTP interruption requires separate diagnosis')
            if failure['physical_steps'] != measured['steps'] or measured['first_success'] is not None:
                raise ValueError('Interrupted prefix mismatch')
        api = audit_api(root, contract, trace, initial, result, frozen['tasks'][name], result is None)
        identity = api_identity(root / 'api/journal.sqlite', require_consumed=result is not None)
        account = api_accounting(root / 'api/journal.sqlite')
        accounts.append(dict(task=name, condition=condition, seed=seed, **account))
        audits.append(dict(task=name, condition=condition, seed=seed, passed=True,
            completed=result is not None, prefix=measured, API_audit=api, model_identity=identity,
            initial_state_identical=True, initial_prompt_identical=True, first_request_identical=True,
            result_sha256=digest(root / 'result.json') if result else None,
            failure_sha256=digest(root / 'failure.json') if not result else None,
            trace_sha256=digest(trace_path) if trace_path.exists() else None))
        row = dict(task=name, condition=condition, seed=seed, method='APPL_6_xhigh',
            completed=result is not None, success=result['success'] if result else None,
            status=result['status'] if result else 'interrupted', **measured,
            reused_original_result=False, selected_attempt='authorized_reset_retest_20260918',
            original_episode_root=str(old_root), episode_root=str(root))
        for cap in (1500, 3000, 5000):
            row['success_by_' + str(cap)] = measured['first_success'] is not None and measured['first_success'] <= cap
        rows[index] = row
        path = video(root)
        if path:
            clips.append(dict(task=name, condition=condition, seed=seed, path=str(path),
                              sha256=digest(path), validation=verify_video(root)))
    groups = []
    for name in NAMES:
        for condition in ('ID', 'OOD'):
            for method in METHODS:
                es = [r for r in rows if (r['task'], r['condition'], r['method']) == (name, condition, method)]
                if len(es) != 30:
                    raise ValueError('Incomplete task/split/method matrix')
                success = sum(e['success'] is True for e in es)
                completed = sum(e['completed'] for e in es)
                groups.append(dict(task=name, condition=condition, method=method,
                    success=success, completed=completed, planned=30, unknown=30-completed,
                    rate=success/30 if completed == 30 else None,
                    wilson95=wilson(success, 30) if completed == 30 else None))
    paired = []
    for name in NAMES:
        for condition in ('ID', 'OOD'):
            for reference in METHODS[:2]:
                new = {r['seed']: r for r in rows if (r['task'], r['condition'], r['method']) == (name, condition, METHODS[2])}
                old = {r['seed']: r for r in rows if (r['task'], r['condition'], r['method']) == (name, condition, reference)}
                cells = Counter((old[s]['success'], new[s]['success']) for s in new if old[s]['completed'] and new[s]['completed'])
                win, loss = cells[(False, True)], cells[(True, False)]
                n = win + loss
                p = min(1., 2*sum(math.comb(n, i) for i in range(min(win, loss)+1))/2**n) if n else 1.
                paired.append(dict(task=name, condition=condition, reference=reference,
                    complete_pairs=sum(cells.values()), new_only=win, reference_only=loss,
                    both_success=cells[(True, True)], both_failure=cells[(False, False)],
                    exact_mcnemar_p=p, interpretation='Descriptive complete pairs; unadjusted, one training seed, reused layouts.'))
    usage = Counter(original['summary']['reported_API_usage'])
    states = Counter(original['summary']['API_states'])
    previous = read(BASE / 'evaluation_recovery/validation.json')['API_accounting']
    for account in [previous, *accounts]:
        usage.update(account['usage'])
        states.update(account['states'])
    fresh = [r for r in rows if r['method'] == METHODS[2]]
    failures = [r for r in fresh if r['completed'] and not r['success']]
    summary = dict(model='gpt-6-astra', effort='xhigh', planned=300,
        complete=sum(r['completed'] for r in fresh), successes=sum(r['success'] is True for r in fresh),
        task_failures=len(failures), unknown=sum(not r['completed'] for r in fresh),
        original_complete_outcomes=211, resumed_attempts=len(audits),
        resumed_complete_outcomes=sum(a['completed'] for a in audits),
        previous_zero_step_interrupted_retests=1, API_states=dict(states), reported_API_usage=dict(usage),
        failed_request_cost='Unreported usage remains unknown; no currency bill inferred.',
        total_formal_training_updates=720000, added_training_updates=0,
        failure_statuses=dict(Counter(r['status'] for r in failures)),
        failure_step_range=[min(r['steps'] for r in failures), max(r['steps'] for r in failures)] if failures else None,
        success_by={str(cap): sum(r['success_by_'+str(cap)] for r in fresh) for cap in (1500, 3000, 5000)},
        original_files_verified_unchanged=len(before), all_300_outcomes_complete=all(r['completed'] for r in fresh))
    atomic(out / 'results.json', dict(summary=summary, groups=groups, episodes=rows,
        paired=paired, resumed_API_accounting=accounts, previous_interrupted_retest_accounting=previous,
        original_results_sha256=digest(BASE / 'results.json'), execution_sha256=digest(out / 'execution.json'),
        study_sha256=read(BASE / 'study_freeze.json')['study_sha256'], report_source_sha256=digest(__file__)))
    atomic(out / 'evaluation_audit.json', dict(passed=True, resumed_records=audits,
        retained_original_complete_audit_sha256=digest(BASE / 'evaluation_audit.json'),
        retained_original_interruption_audit_sha256=digest(BASE / 'incidents/evaluation_outage/audit.json'),
        original_files_unchanged=len(before)))
    atomic(out / 'videos.json', dict(resumed_videos=clips,
        retained_original_videos_sha256=digest(BASE / 'videos.json')))
    with (out / 'episodes.csv').open('w') as handle:
        fields = ['task','condition','method','seed','status','completed','success','steps',
                  'first_success','selected_attempt','episode_root']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in rows)
    render(out, rows, groups, summary)
    analyze(out, rows, groups, summary, paired)
    receipt = dict(all_300_new_outcomes_complete=summary['all_300_outcomes_complete'],
        original_complete_outcomes_retained=211, resumed_outcomes_audited=len(audits),
        complete_outcomes=summary['complete'], successes=summary['successes'],
        task_failures=summary['task_failures'], unknown=summary['unknown'],
        original_files_unchanged=len(before), selected_new_study_videos=300,
        new_video_validations=len(clips), original_video_validation_sha256=digest(BASE/'videos.json'),
        original_attempts_retained=True, previous_zero_step_retest_retained=True,
        automatic_retries=0, added_training_updates=0,
        artifacts={name:digest(out/name) for name in ('results.json','episodes.csv','REPORT.md','ANALYSIS.md',
            'analysis_evidence.json','evaluation_audit.json','videos.json','comparison.png',
            'replays.html','paired_examples.html','report_source.py')})
    atomic(out/'report_receipt.json',receipt)
    if summary['all_300_outcomes_complete']:
        if not execution['all_89_completed'] or len(audits)!=89 or len(clips)!=89:
            raise ValueError('Completion requires the entire authorized matrix and its videos')
        atomic(out/'completion.json',dict(status='complete',**receipt))
    return summary


def analyze(out, rows, groups, summary, paired):
    fresh = [r for r in rows if r['method'] == METHODS[2]]
    segmentation = []
    for name in NAMES:
        record = dict(task=name)
        for label, cfg in [('old', original_config(name)), ('new', config(name))]:
            path = ROOT / cfg['dataset'] / 'manifest.json'
            manifest = read(path)
            record[label] = dict(manifest=str(path),sha256=digest(path),checks=manifest['checks'],
                                 skills=[d['skill_id'] for d in manifest['datasets']])
        segmentation.append(record)
    lines = ['# Measured analysis after the authorized resumption', '',
        f"The new GPT-6 Astra/xhigh study has {summary['complete']} complete outcomes of 300: {summary['successes']} successes, {summary['task_failures']} task failures and {summary['unknown']} unknown outcomes.",
        '', '## Five-task comparison', '',
        '| Task | DP ID | APPL 5.5 ID | APPL 6 ID | DP OOD | APPL 5.5 OOD | APPL 6 OOD |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for name in NAMES:
        cells = []
        for condition in ('ID', 'OOD'):
            for method in METHODS:
                g, = [g for g in groups if (g['task'],g['condition'],g['method']) == (name,condition,method)]
                cells.append(f"{g['success']}/30"+(f"; {g['unknown']} unknown" if g['unknown'] else ''))
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines += ['', 'Each denominator is the declared 30 cells. An unknown outcome is not a task failure. The old 5.5/high buffer ID interruption remains unretouched and separate from the new-study recoveries.',
        '', '## Paired evidence relative to the old APPL library', '',
        '| Task | Split | Complete pairs | Both succeed | New only | Old only | Both fail |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for row in paired:
        if row['reference'] == METHODS[1]:
            lines.append(f"| {row['task']} | {row['condition']} | {row['complete_pairs']} | {row['both_success']} | {row['new_only']} | {row['reference_only']} | {row['both_failure']} |")
    lines += ['', 'These counts use identical initial-layout seeds. Exact paired tests are recorded in results.json as descriptive, unadjusted statistics; there is one training seed and the layouts were analyzed previously. They do not isolate the influence of model family or reasoning effort.',
        '', '## Stopping and incomplete task goals', '',
        f"Failure termination statuses: `{summary['failure_statuses']}`. Failure step range: `{summary['failure_step_range']}`. Confirmed successes by step 1500/3000/5000: `{summary['success_by']}`.",
        'An API finish before the physical cap is an observed stopping decision. It does not establish that running longer would succeed, or that the global step cap caused the failure.', '']
    final_patterns = {}
    for name in NAMES:
        failures = [r for r in fresh if r['task']==name and r['completed'] and not r['success']]
        counts = Counter()
        for row in failures:
            final = read(Path(row['episode_root'])/'result.json')['final']
            counts[tuple(k for k, value in sorted(final.items()) if k!='success' and value)] += 1
        final_patterns[name] = [dict(achieved_goals=list(k),count=v) for k,v in counts.items()]
        lines.append(f"- {name}: {len(failures)} task failures; achieved-goal patterns: "+'; '.join(f"{list(k) or ['none']}: {v}" for k,v in counts.items())+'.')
    lines += ['', 'Original failed trajectories include missed grasps, empty closed grippers and displaced blocks, with the API explicitly ending after unsuccessful recovery attempts. Those original observations remain in the [pre-recovery analysis](../ANALYSIS.md). Final predicates and public API finish explanations support trajectory inspection; they are not controlled causal tests of architecture, policy switching or recovery coverage.',
        '', '## What changed and what is controlled', '',
        '- Training demonstrations, task geometry, target predicates, paired seeds, complete-demonstration normalization recipe, DDPM100 and execution chunk of 8 are retained. The API does not receive the private 5000-step cap.',
        '- Model and reasoning effort changed together from GPT-5.5/high to GPT-6 Astra/xhigh. API regenerated segmentation, priors, policy architecture/loss and handoff documents, then selected policies at inference. The comparison concerns this full pipeline.',
        '- New versus old policy counts by task: drawer 9/18, sort 6/6, buffer 9/9, unstack 6/12, tray 6/6. The aggregate prior-training budgets are 720,000 versus 1,020,000 updates. DP training compute is also not matched to the policy portfolios.',
        '- The retained one-line EMA reload verification correction matches requires_grad flags. It changes the verification gate only, not sampling, saved weights or deployment behavior. Its original evidence remains under the parent incidents directory.',
        '- Environmental and diffusion seeds are fixed; API generation is not seeded. Interrupted cells were reset, not resumed from a saved simulator state. No completed task failure was retested or used to tune a prior or prompt.',
        '', '## Outage recovery and cost', '',
        'The original 300 attempts left 89 unknowns (82 HTTP 503, two HTTP 429 and five HTTP 502). A separately authorized first recovery failed on HTTP 503 before its first action. After the user reported service restoration, the present round was explicitly authorized for those same 89 cells. Original attempts, failed requests and their possible unreported costs remain in place.',
        f"Current resumption: {summary['resumed_attempts']} attempts, {summary['resumed_complete_outcomes']} complete outcomes. Added training updates: zero. Original files verified unchanged: {summary['original_files_verified_unchanged']}.",
        f"Whole-study API states including retained attempts: `{summary['API_states']}`. Reported usage: `{summary['reported_API_usage']}`. Reported zero usage on an HTTP error is not evidence of zero charge.",
        '', '## Viewing and provenance', '',
        '[Comparison and uncertainty plot](REPORT.md) · [All 300 selected new-study videos](replays.html) · [Fixed paired examples](paired_examples.html) · [Exact API execution audit](evaluation_audit.json) · [Raw outcomes](episodes.csv) · [36 API policy pipelines](../POLICIES.md)', '']
    section = ['', '## Segmentation differences in the retained API outputs', '',
        '| Task | Old skills | New skills | Old overlapping actions | New overlapping actions |',
        '| --- | --- | --- | --- | --- |']
    for record in segmentation:
        old, new = record['old']['checks'], record['new']['checks']
        section.append(f"| {record['task']} | {old['skill_datasets']} | {new['skill_datasets']} | {old['overlapping_actions']} | {new['overlapping_actions']} |")
    section += ['', 'The drawer API changed from six skills (opening, red regrasp, red placement, blue acquisition, blue transport, blue release/retreat) to three broader skills (open_access, evacuate_red, insert_blue). Both manifests cover all 12 demonstrations and 12,809 unique actions with zero exclusions; reported overlap increased from 5,220 to 6,626 actions. Thus its regression cannot simply be described as missing demonstration coverage or fewer overlap actions. Coarser skill decomposition, learned phase distinctions, handoff behavior and the smaller total training budget are possible contributors, not isolated causal findings. No changes are made to the tested library.', '']
    position = lines.index('## What changed and what is controlled')
    lines[position:position] = section
    (out/'ANALYSIS.md').write_text('\n'.join(lines))
    atomic(out/'analysis_evidence.json',dict(failure_goal_patterns=final_patterns,segmentation_comparison=segmentation,
        results_sha256=digest(out/'results.json'), analysis_sha256=digest(out/'ANALYSIS.md'),
        new_API_requests=0,new_simulator_steps=0))


def render(out, rows, groups, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    colors = ['#8894a5', '#d99d38', '#167d9a']
    figure, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=True)
    for ri, condition in enumerate(('ID','OOD')):
        for ci, name in enumerate(NAMES):
            ax = axes[ri, ci]
            for x, method in enumerate(METHODS):
                g, = [g for g in groups if (g['task'],g['condition'],g['method']) == (name,condition,method)]
                rate = g['success']/30
                ax.bar(x, rate, color=colors[x])
                if g['unknown']:
                    ax.bar(x,g['unknown']/30,bottom=rate,color='none',edgecolor=colors[x],hatch='///')
                else:
                    lo,hi = g['wilson95']
                    ax.errorbar(x,rate,yerr=[[max(0,rate-lo)],[max(0,hi-rate)]],fmt='none',color='black',capsize=3)
                ax.text(x,1.03,f"{g['success']}/30"+(f" + {g['unknown']}?" if g['unknown'] else ''),ha='center',fontsize=9)
            ax.set_xticks([0,1,2],['DP','5.5/high','6/xhigh'])
            ax.set_ylim(0,1.13)
            ax.set_title(name.replace('_',' ')+' / '+condition,fontsize=10)
            ax.grid(axis='y',alpha=.2)
    handles = [Patch(facecolor=c,label=l) for c,l in zip(colors,LABELS)]
    handles.append(Patch(facecolor='none',edgecolor='gray',hatch='///',label='Unknown outcomes'))
    figure.legend(handles=handles,loc='lower center',ncol=4)
    figure.tight_layout(rect=(0,.06,1,1))
    figure.savefig(out/'comparison.png',dpi=180)
    plt.close(figure)
    lines = ['# GPT-6 Astra / xhigh: results after authorized evaluation recovery','',
        f"Complete new-study outcomes: **{summary['complete']}/300**; successes: **{summary['successes']}**; task failures: **{summary['task_failures']}**; unknown: **{summary['unknown']}**.",
        'Original 211 complete outcomes are retained; only originally HTTP-interrupted cells receive the declared reset retest. All original attempts and the earlier zero-step interrupted retest remain separate.','',
        '| Task | Split | DP | APPL 5.5/high | APPL 6/xhigh |','| --- | --- | --- | --- | --- |']
    for name in NAMES:
        for condition in ('ID','OOD'):
            cells = []
            for method in METHODS:
                g, = [g for g in groups if (g['task'],g['condition'],g['method']) == (name,condition,method)]
                cells.append(f"{g['success']}/30"+(f"; {g['unknown']} unknown" if g['unknown'] else ''))
            lines.append('| '+name+' | '+condition+' | '+' | '.join(cells)+' |')
    lines += ['', '![Three-method comparison](comparison.png)','',
        'Error bars are Wilson 95% intervals for complete groups. Hatching marks unknown outcomes; it is not a confidence interval. The old APPL comparison retains its separate historical missing outcome.',
        '', '## Protocol and interpretation', '',
        'The five tasks use the same 12 demonstrations per task, paired initial layouts, complete-demonstration normalization recipe and geometric goals. All new policies use 20,000 updates, seed 0, final EMA, DDPM100, history 2, horizon 16 and execution 8. The 5000-step cap is hidden from the inference API. API alone owns segmentation, priors, policy source and deployment decisions.',
        'The additional library contains 36 policies and 720,000 formal updates; the retained 5.5/high library contains 51 and 1,020,000 historical updates. Both API model and reasoning effort changed, along with the API-generated library. This is not a single-factor model or effort ablation. These are follow-up paired layouts already analyzed previously, not untouched hidden tests. API generation is unseeded.',
        'Recovery changes output paths and scheduling only. Every resumed initial state, prompt and first request is compared exactly with the original. Every recorded action, literal API stop condition, model identity and geometric result is audited. Task failures are retained without performance-driven retesting.',
        '', '## Accounting', '',
        f"Additional training updates: 0. Resumed attempts: {summary['resumed_attempts']}; completed resumed outcomes: {summary['resumed_complete_outcomes']}. One earlier zero-step recovery interruption is also retained and charged separately.",
        f"Full new-study API states including all interrupted attempts: `{summary['API_states']}`. Reported token totals: `{summary['reported_API_usage']}`. Failed requests may have unreported usage; no currency bill is inferred.",
        '', '[Measured analysis](ANALYSIS.md) · [All selected new-study replays](replays.html) · [Fixed paired examples](paired_examples.html) · [API policies](../POLICIES.md) · [Raw results](results.json) · [Resumed execution audit](evaluation_audit.json) · [Original attempt report](../REPORT.md)', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    style = '<style>body{font:16px system-ui;max-width:1200px;margin:24px auto}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}article{padding:12px;background:#f3f5f7}video{width:100%}</style>'
    def card(row, label):
        path = Path(row['episode_root'])/'replay.mp4'
        if not path.exists():
            raise ValueError('Missing selected replay: '+str(path))
        outcome = 'unknown' if not row['completed'] else 'success' if row['success'] else 'failure'
        return f'<article data-task="{row["task"]}"><h3>{html.escape(label)}</h3><p>{outcome}; {row["steps"]} steps; {row["selected_attempt"]}</p><video controls preload="none" src="{html.escape(os.path.relpath(path,out))}"></video></article>'
    page = ['<!doctype html><meta charset="utf-8"><title>APPL 6/xhigh replays</title>',style,
            '<h1>APPL 6/xhigh: all 300 selected attempts</h1><p>Original recorded frames, approximately 6x physical speed. Unknown clips show only the retained prefix.</p>',
            '<select onchange="document.querySelectorAll(\'article\').forEach(x=>x.hidden=this.value!==\'all\'&&x.dataset.task!==this.value)"><option value="all">All tasks</option>'+''.join(f'<option>{n}</option>' for n in NAMES)+'</select><div class="grid">']
    page += [card(r,f"{r['task']} / {r['condition']} / {r['seed']}") for r in rows if r['method']==METHODS[2]]
    page.append('</div>')
    (out/'replays.html').write_text('\n'.join(page))
    page = ['<!doctype html><meta charset="utf-8"><title>Paired comparison</title>',style,
            '<h1>Fixed paired examples</h1><p>First declared ID and OOD seed per task, irrespective of outcome; original frames only.</p>']
    for name in NAMES:
        for condition in ('ID','OOD'):
            seed = config(name)['evaluation']['seeds' if condition=='ID' else 'ood_seeds'][0]
            page.append(f'<h2>{name} / {condition} / {seed}</h2><div class="grid">')
            for method,label in zip(METHODS,LABELS):
                row, = [r for r in rows if (r['task'],r['condition'],r['seed'],r['method'])==(name,condition,seed,method)]
                page.append(card(row,label))
            page.append('</div>')
    (out/'paired_examples.html').write_text('\n'.join(page))


if __name__ == '__main__':
    print(generate(), flush=True)
