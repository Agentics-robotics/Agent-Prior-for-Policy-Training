"""Summarize the finished original matrix without changing scientific artifacts."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import sqlite3

import numpy as np

from appl.io import ROOT, atomic, digest, read
from appl.scaleup.protocol import BASE, NAMES


def plot_comparison(groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    lookup = {(g['task'], g['condition'], g['method']): g for g in groups}
    fig, axes = plt.subplots(2, 5, figsize=(16, 7), sharey=True)
    for row, condition in enumerate(('ID', 'OOD')):
        for col, name in enumerate(NAMES):
            ax = axes[row, col]
            for x, method, color in ((0, 'naive_DP', '#8d99ae'), (1, 'APPL', '#247ba0')):
                g = lookup[name, condition, method]
                confirmed, upper = np.asarray(g['success_count_bounds']) / 30
                ax.bar(x, confirmed, color=color, width=.65)
                if g['unknown']:
                    ax.bar(x, upper-confirmed, bottom=confirmed, color='white',
                           edgecolor='black', hatch='///', width=.65)
                    label = f"{g['success']} confirmed\n{g['unknown']} unknown"
                    top = upper
                else:
                    lo, hi = g['wilson95']
                    ax.errorbar(x, confirmed, yerr=[[max(0, confirmed-lo)], [max(0, hi-confirmed)]],
                                fmt='none', color='black', capsize=3)
                    label = f"{g['success']}/30"
                    top = hi
                ax.text(x, top+.035, label, ha='center', fontsize=9)
            ax.set_xticks([0, 1], ['DP', 'APPL'])
            ax.set_ylim(0, 1.18)
            ax.set_yticks([0, .25, .5, .75, 1], ['0%', '25%', '50%', '75%', '100%'])
            ax.set_title(name.replace('_', ' ').title()+' / '+condition, fontsize=10)
            ax.grid(axis='y', alpha=.2)
            if col == 0:
                ax.set_ylabel('Successes among 30 planned layouts')
    fig.suptitle('Five tasks · 12 demonstrations each · DDPM100 · 5000-step cap', fontsize=15)
    fig.legend(handles=[Patch(facecolor='white', edgecolor='black', hatch='///', label='Unknown outcome range')],
               loc='lower center', frameon=False)
    fig.text(.5, .048, 'Error bars: 95% Wilson intervals for complete groups only. APPL and DP are not compute-matched.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .94))
    path = BASE / 'comparison_with_unknowns.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    completion = read(BASE / 'completion.json')
    assert completion['evidence_checks_passed'] and completion['attempted_episodes'] == 600
    result = read(BASE / 'results.json')
    diagnostics_path = ROOT / 'experiments/exp2/analysis/outcome_diagnostics.json'
    diagnostics = read(diagnostics_path)
    assert diagnostics['completed'] == result['completed'] == completion['completed_episodes']
    for path, expected in completion['artifacts'].items():
        assert digest(path) == expected, path
    rows = diagnostics['episodes']
    groups = []
    for original in result['groups']:
        key = tuple(original[k] for k in ('task', 'condition', 'method'))
        selected = [r for r in rows if tuple(r[k] for k in ('task', 'condition', 'method')) == key]
        assert len(selected) == original['completed']
        failures = [r for r in selected if not r['success']]
        maxima = [r['maximum_normalized_action_input']['abs_value'] for r in selected
                  if r['maximum_normalized_action_input']]
        groups.append(dict(**original,
            success_count_bounds=[original['success'], original['success'] + original['unknown']],
            successful_step_median=float(np.median([r['steps'] for r in selected if r['success']]))
                if any(r['success'] for r in selected) else None,
            termination_counts=dict(Counter(r['status'] for r in selected)),
            failures=len(failures),
            failed_without_3cm_rise={c: sum(r['object_heights'][c]['first_3cm_rise_step'] is None
                                          for r in failures) for c in ('red', 'blue')},
            failed_with_100_consecutive_narrow_width_states=sum(
                r['longest_5mm_or_less_width_run_states'] >= 100 for r in failures),
            normalized_input_episode_max_quantiles=dict(zip(('median', 'p90', 'maximum'),
                [float(v) for v in np.quantile(maxima, [.5, .9, 1])])) if maxima else None))
    states = result['api']['states']
    assert set(states) <= {'consumed', 'http_failed'}, states
    formal = Counter()
    for row in result['training']:
        formal[row['owner']] += row['optimizer_steps']
    aggregate = {}
    for method in ('naive_DP', 'APPL'):
        selected = [g for g in groups if g['method'] == method]
        stops = Counter()
        for g in selected:
            stops.update(g['termination_counts'])
        aggregate[method] = dict(failures=sum(g['failures'] for g in selected),
            failed_with_100_consecutive_narrow_width_states=sum(
                g['failed_with_100_consecutive_narrow_width_states'] for g in selected),
            termination_counts=dict(stops),
            new_task_ID_successes=sum(g['success'] for g in selected
                if g['task'] != 'drawer_exchange' and g['condition'] == 'ID'))
    gpu_counts = Counter()
    finish_calls = []
    for episode in result['episodes']:
        relative = (Path(episode['task']) / 'evaluation' / episode['method'] /
                    episode['condition'] / str(episode['seed']))
        plan = read(BASE / relative / 'plan.json')
        job = read(BASE / 'batch/jobs' / episode['task'] / episode['method'] /
                   episode['condition'] / str(episode['seed']) / 'process_result.json')
        assert plan['device']['physical_gpu'] == job['gpu']
        gpu_counts[job['gpu']] += 1
        if episode['method'] == 'APPL' and episode['completed']:
            database = (BASE / relative / 'api/journal.sqlite').resolve()
            wal = database.with_name(database.name+'-wal')
            assert not wal.exists() or wal.stat().st_size == 0
            db = sqlite3.connect('file:'+str(database)+'?mode=ro&immutable=1', uri=True)
            try:
                reasons = [json.loads(args)['reason'] for args, in db.execute(
                    "SELECT args FROM tools WHERE name='finish' AND status='completed' ORDER BY rowid")]
            finally:
                db.close()
            if episode['status'] == 'agent_finished':
                assert reasons, str(relative)
            finish_calls.append(dict(task=episode['task'], condition=episode['condition'],
                seed=episode['seed'], status=episode['status'], steps=episode['steps'],
                verbatim_API_finish_reasons=reasons))
    assert sum(gpu_counts.values()) == 600 and set(gpu_counts) <= {1, 2, 3, 7}
    summary = dict(reviewed_utc=datetime.now(timezone.utc).isoformat(),
        scope='Post-hoc analysis of the original 600 planned attempts, not supplemental retests.',
        completed=result['completed'], unknown=600-result['completed'], groups=groups,
        new_formal_optimizer_updates=dict(formal),
        confirmed_interface_updates=sum(r['confirmed_updates'] for r in result['interface_checks']),
        failed_interface_attempts=[r['path'] for r in result['interface_checks'] if not r['passed']],
        unconfirmed_interface_update_upper_bound=sum(r['unconfirmed_update_upper_bound']
                                                     for r in result['interface_checks']),
        api_states=states, reported_api_usage=result['api']['usage'],
        aggregate_failure_diagnostics=aggregate,
        original_evaluation_attempts_by_physical_gpu=dict(gpu_counts),
        API_finish_tool_audit=finish_calls,
        source_hashes={str(p): digest(p) for p in (BASE/'completion.json', BASE/'results.json', diagnostics_path)},
        script_sha256=digest(__file__), additional_training=0, additional_physical_steps=0,
        additional_API_calls=0)
    plot = plot_comparison(groups)
    summary['comparison_plot_sha256'] = digest(plot)
    atomic(BASE / 'analysis_summary.json', summary)
    lookup = {(g['task'], g['condition'], g['method']): g for g in groups}
    lines = ['# Five-task analysis', '',
        f"The original matrix has {result['completed']} complete outcomes and {600-result['completed']} unknown outcomes across 600 planned attempts.",
        'Unknown outcomes are retained separately; they are neither policy failures nor silently replaced trials.', '',
        '## Final comparison', '',
        '| Task | DP ID | APPL ID | DP position OOD | APPL position OOD |',
        '| --- | ---: | ---: | ---: | ---: |']
    def display(g):
        return (f"{g['success']}/30" if not g['unknown'] else
                f"{g['success']} confirmed; {g['unknown']} unknown / 30")
    for name in NAMES:
        lines.append('| '+name+' | '+' | '.join(display(lookup[name,c,m])
            for c in ('ID','OOD') for m in ('naive_DP','APPL'))+' |')
    lines += ['', '![Five-task comparison including unknown outcomes](comparison_with_unknowns.png)', '',
        'Complete groups have 95% Wilson intervals in [the report](REPORT.md). '
        'One training seed and one API decision trajectory per layout do not measure training or API sampling variability.', '',
        '### Main interpretation', '',
        f"The four new tasks yield only {aggregate['naive_DP']['new_task_ID_successes']}/120 naive-DP ID successes. "
        'The verified implementation-equivalence results below exclude specific migration mistakes; '
        'they do not isolate why this monolithic learned baseline performs poorly.', '',
        f"Among {aggregate['naive_DP']['failures']} completed DP failures, "
        f"{aggregate['naive_DP']['failed_with_100_consecutive_narrow_width_states']} contain at least 100 consecutive "
        'state snapshots with finger width <=5 mm. The corresponding APPL count is '
        f"{aggregate['APPL']['failed_with_100_consecutive_narrow_width_states']}/{aggregate['APPL']['failures']}. "
        'No original demonstration contains such a narrow-width state. This supports missing recovery-state '
        'coverage as a concrete hypothesis; finger width alone is not a grasp/contact classifier or causal proof.', '',
        f"APPL has {aggregate['APPL']['termination_counts'].get('agent_finished',0)} unsuccessful voluntary finishes "
        f"and {aggregate['APPL']['termination_counts'].get('physical_budget_exhausted',0)} failures at the physical cap. "
        'Most remaining failures therefore did not consume the whole 5000-step allowance. '
        'Increasing a hidden cap alone does not extend a recorded trajectory that the API already chose to end.', '',
        'The study holds demonstration count, DDPM100 sampling and the declared training recipe fixed. '
        'It does not test whether additional demonstrations, larger networks, more updates or more denoising '
        'steps would solve these failures. Recovery coverage, phase/handoff representation and API stopping '
        'decisions remain candidates for a separately declared follow-up study.', '',
        '## What additional physical steps changed', '',
        'The following are success prefixes of the same frozen 5000-step trajectories. '
        'They are not separate budget experiments and do not change API decisions. '
        'The executor hides the total and remaining physical budget from the API.', '',
        '| Task | Split | Method | By 1500 | By 3000 | By 5000 | Unknown final outcomes |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for g in groups:
        lines.append(f"| {g['task']} | {g['condition']} | {g['method']} | "
                     +' | '.join(str(g['success_by'][str(c)]) for c in (1500,3000,5000))
                     +f" | {g['unknown']} |")
    for method in ('naive_DP', 'APPL'):
        selected = [g for g in groups if g['method'] == method]
        totals = {cap: sum(g['success_by'][str(cap)] for g in selected) for cap in (1500,3000,5000)}
        lines += ['', f"Across {sum(g['completed'] for g in selected)} complete {method} outcomes: "
            f"{totals[1500]} succeeded by 1500, {totals[3000]} by 3000 and {totals[5000]} by 5000 steps. "
            f"Thus {totals[5000]-totals[1500]} observed completions occur after 1500, "
            f"including {totals[5000]-totals[3000]} after 3000. "
            f"Unknown final outcomes: {sum(g['unknown'] for g in selected)}."]
    lines += ['', 'Counts are confirmed successes among the 30 planned layouts per row. '
        'An interruption before a given cap leaves that trajectory unknown at the cap.', '',
        '## Failure evidence and its limits', '',
        'Initial-state ID is a property of the reset distribution. After a missed grasp or poor placement, '
        'the policy can reach states absent from all twelve demonstrations. More diffusion sampling steps '
        'or a larger episode budget do not by themselves add examples of how to recover from those states.', '',
        'Every original dataset has zero observations with summed finger-joint width at or below 5 mm. '
        'The four new datasets have a minimum width near 36.5 mm. '
        '[Training support](../../../experiments/exp2/analysis/training_gripper_support.json) '
        'therefore does not contain the fully closed states observed after several empty grasps. '
        'Width is not a contact sensor; this is evidence of missing state support, not proof of a single failure cause.', '',
        '| Task | Split | Method | Completed failures | No red 3 cm rise | No blue 3 cm rise | >=100 consecutive narrow-width states |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for g in groups:
        lines.append(f"| {g['task']} | {g['condition']} | {g['method']} | {g['failures']} | "
            f"{g['failed_without_3cm_rise']['red']} | {g['failed_without_3cm_rise']['blue']} | "
            f"{g['failed_with_100_consecutive_narrow_width_states']} |")
    lines += ['', 'These descriptive counts can overlap. A 3 cm rise is not proof of a successful grasp; '
        'narrow width means <=5 mm and includes state snapshots. These analysis conventions do not alter task success.', '',
        '### Normalization and baseline implementation', '',
        'The read-only [baseline equivalence audit](../../../experiments/exp2/analysis/naive_equivalence_audit.json) '
        'verified all four 60000-update models, complete-demo training windows, action transforms and normalization '
        'against the original M0 implementation. With the same final EMA and RNG, DDPM100 outputs match exactly '
        'on the checked original training histories. This excludes those particular migration errors; '
        'it is not a proof that every component or modeling choice is optimal.', '',
        '| Task | Split | Method | Median episode max normalized input | 90th percentile | Largest |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for g in groups:
        q=g['normalized_input_episode_max_quantiles']
        if q:
            lines.append(f"| {g['task']} | {g['condition']} | {g['method']} | {q['median']:.2f} | {q['p90']:.2f} | {q['maximum']:.2f} |")
    lines += ['', 'These are normalized observations actually preceding actions, excluding the terminal state. '
        'A large value alone does not establish a normalization bug. In particular, a velocity far outside '
        'training support can produce a large normalized input even with the verified full-demo scale.', '',
        '### Illustrative recorded trajectories', '',
        '- Sorting ID seed 20000: DP failed at 5000 steps; APPL succeeded at 680. '
        'The DP input magnitude peaked at 2.68; the blue block never rose 3 cm. '
        'This case is not explained by the old roughly 700-fold quaternion amplification.',
        '- Sorting OOD seed 30000: APPL voluntarily stopped after 882 steps. '
        'Its gripper stayed fully closed after step 81. Recorded raw and executed commands match and '
        'the native controller maps them to closure; the framework did not rewrite an opening command.',
        '- Drawer OOD seed 6406: APPL succeeded at step 3492 after 42 policy calls. '
        'The red goal was first met at 2445 and the blue goal at 3492. This post-hoc example '
        'shows a successful late recovery; the full prefix table establishes how frequent late successes are.', '',
        'The first two examples are the first predeclared ID/OOD sorting seeds. '
        'The late drawer example was selected after observing its late success.', '',
        '## Termination behavior', '',
        '| Task | Split | Method | Recorded termination counts |',
        '| --- | --- | --- | --- |']
    for g in groups:
        counts=', '.join(f'{k}: {v}' for k,v in sorted(g['termination_counts'].items()))
        lines.append(f"| {g['task']} | {g['condition']} | {g['method']} | {counts} |")
    lines += ['', 'An API `agent_finished` failure is its own decision to stop; an unused physical budget '
        'does not imply that the executor forced termination. `physical_budget_exhausted` is the private cap. '
        'Every `agent_finished` outcome was checked against a completed original API `finish` tool call; '
        'verbatim reasons are retained in the machine-readable analysis, as API explanations rather than '
        'independently established causes. API-request exhaustion raises a separate error. '
        'Transport-interrupted attempts are excluded from the completed-outcome counts above.', '',
        '## Costs and interpretation', '',
        f"New formal optimizer updates: naive DP {formal['developer_naive']:,}; API prior policies {formal['Runtime_API_prior']:,}. "
        f"Interface checks add {summary['confirmed_interface_updates']} confirmed updates, with "
        f"{summary['unconfirmed_interface_update_upper_bound']} additional updates as the retained uncertainty upper bound.", '',
        'Two unsubmitted API drafts failed tensor-shape interface checks: buffer-swap blue-transfer h02 '
        'and unstack blue-placement h03. Runtime API corrected its unsubmitted source and both passed '
        'subsequent checks before formal training. The failed source/check receipts remain available '
        'in the raw result table; they are not additional formal models or concealed retries.', '',
        'The study adds 4 naive models and 33 API prior models; the original drawer reuses 1 naive model '
        'and 18 API priors. Each naive model uses 60000 updates; each prior uses 20000. '
        'Consequently the full evaluated portfolio represents 300000 naive and 1020000 APPL formal updates. '
        'Update counts are not FLOP-normalized and model architectures can differ.', '',
        f"Verified physical GPU records for all 600 original evaluation attempts: {dict(gpu_counts)}. "
        'Only physical GPUs 1, 2, 3 and 7 occur; multiple processes share each card.', '',
        f"New-study API request states: {states}. Reported usage: {result['api']['usage']}. "
        'These include the four new segmentation/design stages and original-matrix inference, '
        'excluding historical drawer design. Failed-request usage can be unknown. Currency costs are not '
        'asserted because provider billing/rates are unavailable; raw journals and reported token counts are retained.', '',
        'This compares the complete APPL system with naive DP. APPL also gains skill decomposition, '
        'multiple separately trained models and observation-dependent API selection. The result cannot '
        'isolate the causal contribution of a prior, overlap or model count. A future matched ablation '
        'would be needed; none was run using these test results.', '',
        'The four new tasks use developer-designed scripted demonstrations with twelve fixed training seeds. '
        'Collection success is not learned-policy success. OOD changes initial positions only. '
        'The physical simulator pauses while awaiting the API, so these results do not demonstrate '
        'real-time control. The agreed geometric goal may be met before release or stable rest.', '',
        '## Evidence and viewing', '',
        '- [Final matrix, uncertainty and paired comparisons](results.json)',
        '- [Five-task paired examples](paired_examples.html): first declared ID/OOD seed per task, without success-based selection.',
        '- [All original replay clips](replays.html)',
        '- [Completion and artifact audit](completion.json)',
        '- [API-authored policy and handoff documents](POLICIES.md)',
        '- [Machine-readable analysis](analysis_summary.json)', '',
        'This analysis creates no new optimizer updates, API calls, physical steps or edited API outputs.', '']
    document = '\n'.join(lines).replace('](../../../experiments/', f']({ROOT}/experiments/')
    for target in re.findall(r'\]\(([^)]+)\)', document):
        path = Path(target) if target.startswith('/') else BASE / target
        assert path.exists(), target
    (BASE / 'ANALYSIS.md').write_text(document)
    summary['analysis_markdown_sha256'] = digest(BASE / 'ANALYSIS.md')
    atomic(BASE / 'analysis_summary.json', summary)
    print(dict(complete_outcomes=result['completed'], unknown=600-result['completed'],
               analysis=str(BASE/'ANALYSIS.md')), flush=True)


if __name__ == '__main__':
    main()
