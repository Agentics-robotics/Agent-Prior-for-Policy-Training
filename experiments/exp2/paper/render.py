"""Narrative, source catalog and completion receipt for the Exp2 paper bundle."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

from appl.io import ROOT, atomic, digest, read
from .build import BASE, OUT, METHOD, METHODS, LABELS, NAMES, TASK_LABELS, link, repo_path, table


def render(result,tasks,models,segments,aggregates,prefixes,compute):
    reviewed=datetime.now(timezone.utc).isoformat()
    groups=result['groups'];episodes=result['episodes'];summary=result['summary']
    get=lambda task,condition,method:next(g for g in groups if (g['task'],g['condition'],g['method'])==(task,condition,method))
    def cell(g):
        return f"{g['success']}/30"+(f"; {g['unknown']} unknown" if g['unknown'] else f" ({g['success']/30:.1%})")
    from .recovery import add_accounting
    old_api=add_accounting(read(ROOT/'runs/exp2/M1_scaleup/results.json')['api'])
    costs=[dict(method='APPL_5_5_high',scope='Scale-up incremental API usage: four new tasks design/segmentation plus five-task evaluation and the authorized 2026-09-19 reset; excludes reused historical drawer design.',
                states=old_api['states'],usage=old_api['usage']),
           dict(method='APPL_6_xhigh',scope='Entire additional five-task APPL study, including retained interruptions and authorized continuations.',
                states=read(ROOT/'experiments/exp2/analysis/astra_token_cost_20260918.json')['states'],
                usage=read(ROOT/'experiments/exp2/analysis/astra_token_cost_20260918.json')['usage']),
           dict(method=METHOD,scope='All five offline designs, including one retained tray HTTP failure; no segmentation and zero deployment API requests.',
                states=summary['design_API_states'],usage=summary['reported_design_API_usage'])]
    atomic(OUT/'API_COSTS.json',dict(accounting=costs,
        caution='Scopes differ. Reported usage is not an invoice; unreported failed-request usage remains unknown. Cached input is a subset of input. Excludes Codex conversation and GPU billing.'))
    table(OUT/'tables/API_costs.csv',costs)
    claims=[]
    for a in aggregates:
        claims.append(dict(claim_id=f"result_{a['method']}_{a['condition']}",kind='Measured outcome',
            statement=f"{a['success']} confirmed successes, {a['completed']} complete outcomes and {a['unknown']} unknown among 150 planned {a['condition']} layouts for {a['method']}.",
            evidence='tables/aggregate.csv; tables/episodes.csv; results.json',
            limitation='One training replicate; later studies use previously examined paired layouts.'))
    claims += [
        dict(claim_id='single_no_runtime_API',kind='Execution contract and audit',
            statement='SinglePrior has one learned full-task policy per task and zero deployment API calls.',
            evidence=repo_path(BASE/'completion.json')+'; METHODS.md',limitation='Offline API design still incurs cost.'),
        dict(claim_id='drawer_regression',kind='Measured task-specific result',
            statement='Drawer ID success decreased from 29/30 in APPL 5.5 to 8/30 in APPL 6.',
            evidence='tables/results.csv; supporting_reports/DRAWER_ASTRA_REGRESSION.md',
            limitation='Model, effort, segmentation, policies and total training changed together.'),
        dict(claim_id='coverage_hypothesis',kind='Interpretation, not causal identification',
            statement='Observed empty-gripper and failed-lift states support investigating missing recovery-state coverage.',
            evidence='supporting_reports/SCALEUP_ANALYSIS.md; supporting_reports/training_gripper_support.json',
            limitation='Finger width and height change do not establish contact or a unique failure cause.'),
        dict(claim_id='agent_only_ablation_absent',kind='Design limitation',
            statement='SinglePrior versus APPL is not an isolated runtime-agent ablation.',
            evidence='METHODS.md; tables/compute.csv; tables/segmentation.csv',
            limitation='Decomposition, model count, training windows and aggregate compute also differ.'),
        dict(claim_id='terminal_semantics',kind='Executed success contract',
            statement='Main-matrix success requires simultaneous geometric predicates at one observation; release or stable settling is not required.',
            evidence='METHODS.md; tables/tasks.csv',limitation='Historical strict M0 results must remain separate.'),
    ]
    table(OUT/'CLAIMS.csv',claims)
    lines=['# Exp2: API-designed priors for long-horizon manipulation','',
        f'Final evidence review: {reviewed}.','',
        '## 1. Executive summary','',
        'The main comparison contains five tasks, twelve original demonstrations per task, four methods, thirty paired ID and thirty paired position-OOD layouts per task. There are **1,200 intended method-layout cells, 1,200 complete outcomes and zero unresolved unknowns**. The last APPL 5.5 interruption was resolved by one user-authorized reset on 2026-09-19; the interrupted attempt remains retained. See the [recovery audit and video](recovery_20260919/REPORT.md). Every single-policy trial is complete. Historical diagnostic and preliminary studies are reported separately.',
        '', '| Method | ID successes / 150 planned | Position-OOD successes / 150 planned |',
        '| --- | ---: | ---: |']
    for method in METHODS:
        cells=[]
        for c in ('ID','OOD'):
            a=next(a for a in aggregates if a['method']==method and a['condition']==c)
            cells.append(f"{a['success']}/150"+(f"; {a['unknown']} unknown" if a['unknown'] else f" ({a['success']/150:.1%})"))
        lines.append('| '+LABELS[method]+' | '+' | '.join(cells)+' |')
    single=sum(r['success'] is True for r in episodes if r['method']==METHOD)
    lines+=['',f'The offline single-prior baseline achieves **{single}/300** successes with five API-authored full-task models and **zero deployment API calls**. Naive DP achieves 17/300. APPL 5.5 records 180 successes among 299 complete outcomes; APPL 6 records 197/300. These totals describe the tested systems, not isolated causal effects of priors, model family, reasoning effort or runtime agents.',
        '', '[Executed methods](METHODS.md) · [Development history](HISTORY.md) · [Writing instructions](WRITING_GUIDE.md) · [Machine-readable outcomes](results.json) · [Source catalog](ARTIFACTS.json)',
        '', '## 2. Tasks and demonstrations','',
        '![Original task cameras](figures/tasks.png)','',
        'Figure 1. Original reset camera frames from the first predeclared ID layout of each task. Source images are 128x128; the display does not add visual detail.',
        '', '| Task | Demonstrations | Original actions | Mean control steps per demonstration | Mean simulated duration |',
        '| --- | ---: | ---: | ---: | ---: |']
    for t in tasks:
        lines.append(f"| {t['label']} | 12 | {t['total_actions']:,} | {t['mean_actions']:.1f} | {t['mean_duration_sim_seconds']:.1f} s |")
    lines+=['','The tasks use the same Panda joint-position interface and 20 Hz control. The new four tasks were designed and their demonstrations collected as developer-owned preparation. OOD changes only initial XY position; it does not change goals, objects or dynamics. Exact IDs, file hashes, layouts, geometry and goal-contract paths are in [task metadata](tasks.json) and [tables](tables/).',
        '', '## 3. Methods and controls','',
        '![Executed protocol](figures/protocol.png)','',
        'Figure 2. Executed data, API, training and deployment roles. The common evaluation interface does not imply equal model count or compute. [PDF](figures/protocol.pdf) · [Mathematical interfaces](FORMALISM.md).',
        '',
        'Naive DP uses one standard full-task diffusion U-Net. SinglePrior uses one full-task DP whose representation, architecture and/or auxiliary loss are chosen and implemented by GPT-6 Astra/xhigh. APPL additionally uses API-selected trajectory segmentation, independent prior policies and runtime selection with observation-based invocation stops. The framework supplies bounded interfaces and independent geometric evaluation; it does not rewrite API policy source, heuristics, handoffs or runtime decisions.',
        '', 'All methods use the same twelve demonstrations, full-demo normalization, history 2, horizon 16, execution 8, DDPM100 and training seed 0. Full-task models receive 60,000 updates; each APPL prior receives 20,000. Final EMA checkpoints are used. All current success predicates are identical across methods and require one simultaneous geometric conjunction, without release, clearance, velocity or sustained-hold requirements. See [METHODS.md](METHODS.md) for exact thresholds and action alignment.',
        '', '| Method | Evaluated models | Total formal updates represented | Parameters per model, min–max |',
        '| --- | ---: | ---: | ---: |']
    for c in compute:
        lines.append(f"| {LABELS[c['method']]} | {c['models']} | {c['formal_updates']:,} | {c['minimum_model_parameters']:,}–{c['maximum_model_parameters']:,} |")
    lines+=['','Counts include reused drawer models. Original scale-up newly trained four naive models and 33 APPL models; the additional APPL 6 round trained all 36 models anew; SinglePrior trained five new models. Interface-check updates are separate from formal training; [check receipts](tables/API_interface_checks.csv) give confirmed updates and retained uncertainty bounds for evaluated API packages. Parameter totals, source versions, training receipts and checkpoint hashes are in [POLICY_INDEX.csv](POLICY_INDEX.csv). All 92 API-authored source packages and prior documents are copied byte-for-byte into [policy_cards](policy_cards/).',
        '', '## 4. Main results','',
        '| Task | Split | Naive DP | Single prior 6/xhigh | APPL 5.5/high | APPL 6/xhigh |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for name in NAMES:
        for condition in ('ID','OOD'):
            lines.append('| '+TASK_LABELS[name]+' | '+condition+' | '+' | '.join(cell(get(name,condition,m)) for m in METHODS)+' |')
    lines+=['','![Main comparison](figures/main_results.png)','',
        'Figure 3. Successes among thirty planned layouts per task and condition. Error bars are descriptive 95% Wilson intervals for the thirty complete outcomes in each group. There is one training replicate; these intervals do not measure training-seed or repeated-agent variance. Vector exports: [PDF](figures/main_results.pdf), [SVG](figures/main_results.svg).',
        '', '### What the additional baseline shows','',
        f'SinglePrior changes the number of observed successes relative to naive DP by {single-17:+d} across the 300 paired layouts. This comparison holds data and update counts fixed, while allowing API-designed architectures/objectives and their compute to differ. The table is the evidence for task-specific gains or regressions; a low training loss is not counted as manipulation success.',
        '', f"Its ID results are strongly task-dependent: buffer exchange {get('buffer_swap','ID',METHOD)['success']}/30, unstacking {get('unstack_sort','ID',METHOD)['success']}/30 and tray packing {get('tray_pack','ID',METHOD)['success']}/30, but two-block sorting {get('two_block_sort','ID',METHOD)['success']}/30. The tested full-task policy approach can complete several of these long-horizon tasks without runtime API decisions; this does not make every task or prior design reliable.",
        '', f"The drawer comparison also changes with this baseline: SinglePrior records {get('drawer_exchange','ID',METHOD)['success']}/30 ID successes with the same offline API model/effort as APPL 6, versus {get('drawer_exchange','ID','APPL_6_xhigh')['success']}/30 for that APPL system. Conversely, APPL 6 records {get('two_block_sort','ID','APPL_6_xhigh')['success']}/30 on sorting. These observations identify task-specific differences between complete pipelines; they do not isolate a benefit or harm from runtime API decisions alone.",
        '', f"SinglePrior's aggregate drops from {summary['success_by_condition']['ID']}/150 ID successes to {summary['success_by_condition']['OOD']}/150 under the larger initial-position offsets. APPL 6 records 73/150 OOD successes. Thus offline prior design improves the tested full-task baseline while leaving a substantial position-generalization gap. The present comparison does not identify which combination of skill decomposition, additional training, alternative policies and runtime selection accounts for that gap.",
        '', 'The single-policy result tests whether offline API prior design is useful without deployment-time API decisions. Comparing it with APPL also changes segmentation, policy count, aggregate training and handoff structure. An agent-only ablation would require a separate matched library and execution comparison. No such isolated experiment was run.',
        '', '### Paired outcomes','',
        'The new baseline uses the same initial states as the three earlier methods. '+link(BASE/'paired_examples.html','four-method videos of first predeclared seeds')+' avoid selecting clips based on outcomes. [Paired win/loss counts](tables/paired_single_prior.csv) compare SinglePrior with each reference using only complete pairs. '+link(ROOT/'runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/results.json','APPL 6 versus 5.5 paired counts')+' remain available as the pre-recovery historical comparison. The current selected outcomes include the documented 2026-09-19 APPL 5.5 reset. Later rounds use previously examined layouts, not a fresh untouched test set.',
        '', '## 5. Step budgets and termination','',
        '| Method | Success by 1500 | By 3000 | By 5000 | Unknown final outcomes |',
        '| --- | ---: | ---: | ---: | ---: |']
    for row in prefixes:
        lines.append(f"| {LABELS[row['method']]} | {row['1500']} | {row['3000']} | {row['5000']} | {row['unknown']} |")
    old_prefix=next(r for r in prefixes if r['method']=='APPL_5_5_high')
    lines+=['','These are prefixes of the same frozen 5,000-step trajectories, not separate budget interventions. The API never receives the total/remaining physical cap. It may choose to finish early; the executor independently evaluates success.',
        '', f"Naive DP recorded no additional success after step 1500. APPL 5.5 recorded {old_prefix['5000']-old_prefix['1500']} later successes, including {old_prefix['5000']-old_prefix['3000']} after 3000; most of its complete failures were voluntary API finishes. All 103 APPL 6 task failures were API finishes at steps 437–2408; none exhausted the 5000-step cap. Therefore increasing that cap alone would not extend those already terminated decision trajectories.",
        '', f"SinglePrior recorded {summary['success_by']['5000']-summary['success_by']['1500']} successes after 1500 and {summary['success_by']['5000']-summary['success_by']['3000']} after 3000. Its unsuccessful complete trials run to the physical cap. Exact statuses for all methods are in [termination.csv](tables/termination.csv).",
        '', '## 6. Failure evidence and interpretation','',
        '### Closed-loop state coverage','',
        'Initial-state ID does not guarantee that a learned rollout stays within demonstration support. Retained analyses find missed grasps, extended empty closed-gripper states and failure to lift a required block. The original successful demonstrations contain no summed finger width at or below 5 mm; the four new task datasets have minimum width near 36.5 mm. Finger width is not a contact classifier, and a 3 cm rise is not proof of a secure grasp. These measurements support a state-coverage hypothesis without establishing a single cause.',
        '', 'The old four new-task naive implementations were checked against M0 for training windows, normalization, action transforms and seeded DDPM100 output on tested histories. The measured equality rules out those specific migration errors. The repaired quaternion scaling does not imply that all later failures disappear, nor that every modeling choice is optimal.',
        '', 'A further untested hypothesis is ambiguity in the expert action targets around nearly stationary grasp/release holds. The expert includes fixed dwell periods, while policies receive only two causal observations. Choosing when to leave a hold may therefore be difficult to infer from very similar inputs. This concerns imitation of expert timing, not proof that the physical task requires a clock. No longer-history or action-commitment ablation was run, so this hypothesis remains separate from the measured failure counts.',
        '', link(ROOT/'runs/exp2/M1_scaleup/ANALYSIS.md','Original five-task failure analysis')+' · '+link(ROOT/'experiments/exp2/analysis/naive_equivalence_audit.json','baseline equivalence audit')+' · '+link(ROOT/'experiments/exp2/analysis/training_gripper_support.json','demonstration gripper support'),
        '', '### Why newer APPL regressed on drawer exchange','',
        'APPL 6 achieved 8/30 drawer ID successes versus 29/30 for APPL 5.5. Its API changed six narrower skills into three broader skills and reduced the drawer portfolio from eighteen to nine models. Both segmentations retained all twelve demonstrations and all 12,809 actions; measured overlapping actions increased from 5,220 to 6,626. Missing overall coverage or a smaller overlap count therefore does not explain the regression.',
        '', 'A read-only review found that all thirty new ID episodes opened the drawer, but only eight lifted blue, and those eight completed the task. Of twenty-six episodes initially selecting red-transfer h03, sixteen returned when the drawer retracted below the API-selected 0.26 m condition; only one of those later succeeded. The evidence points to learned execution, handoff and recovery interactions. It does not isolate which representation, policy boundary, training-budget or agent decision caused the regression.',
        '', link(ROOT/'experiments/exp2/analysis/drawer_astra_regression/REPORT.md','Drawer regression report')+' · '+link(ROOT/'experiments/exp2/analysis/drawer_astra_regression/evidence.json','episode evidence'),
        '', '### Single-policy failures','',
        'The following counts describe final task failures and whether each geometric goal was ever observed during that recorded trajectory. Achieving goals at different times does not satisfy the required simultaneous conjunction. These counts are descriptive and do not, alone, diagnose a grasp or causal mechanism.',
        '', '| Task | Split | Failed episodes | Ever-achieved goal counts among failures |',
        '| --- | --- | ---: | --- |']
    failure_rows=[]
    for name in NAMES:
        for condition in ('ID','OOD'):
            failed=[r for r in episodes if (r['task'],r['condition'],r['method'])==(name,condition,METHOD) and r['success'] is False]
            goals=Counter(k for r in failed for k in r['first_goals'] if k!='success')
            failure_rows.append(dict(task=name,condition=condition,failed=len(failed),ever_goal_counts=dict(goals)))
            lines.append(f"| {TASK_LABELS[name]} | {condition} | {len(failed)} | `{dict(goals)}` |")
    table(OUT/'tables/single_policy_failure_goals.csv',failure_rows)
    diagnosis=read(OUT/'outcome_diagnostics.json')
    new_failures=[r for r in diagnosis['episodes'] if r['method']==METHOD and not r['success']]
    no_red=sum(r['object_heights']['red']['first_3cm_rise_step'] is None for r in new_failures)
    no_blue=sum(r['object_heights']['blue']['first_3cm_rise_step'] is None for r in new_failures)
    narrow=sum(r['longest_5mm_or_less_width_run_states']>=100 for r in new_failures)
    sorting=next(r for r in diagnosis['groups'] if (r['task'],r['condition'],r['method'])==('two_block_sort','ID',METHOD))
    maximum=max(r['maximum_normalized_action_input']['abs_value'] for r in diagnosis['episodes'] if r['method']==METHOD)
    lines+=['',f'Among the {len(new_failures)} unsuccessful SinglePrior trajectories, {no_red} never raised red more than 3 cm above its reset height, {no_blue} never raised blue by that amount, and {narrow} contained at least 100 consecutive state snapshots with summed finger width at or below 5 mm. These categories can overlap. They are descriptive state-support measurements, not contact ground truth or modified success criteria.',
        '', 'The same measurements were extracted for all 1,200 complete trajectories, with retained diagnostic results reused only after result/trace hash verification. [Failure and normalization table](tables/failure_diagnostics.csv) · [Per-episode measurements](outcome_diagnostics.json). Normalization statistics concern the common 47-dimensional representation of states preceding actions, excluding the terminal state; they do not capture every custom API feature or prove a scaling bug.']
    lines += ['', f"Sorting illustrates why failure categories must be separated: {sorting['failed_without_red_3cm_rise']}/30 ID episodes never raised red by 3 cm, and {sorting['failed_without_blue_3cm_rise']}/30 never raised blue by that amount, while only {sorting['failed_with_100_narrow_width_states']}/30 showed the sustained narrow-finger-width pattern. This locates missing manipulation progress, without classifying every failure as an empty grasp.",
        '', f"Across all 300 SinglePrior trajectories, the largest absolute coordinate under the shared 47-dimensional observation normalizer was {maximum:.2f}; sorting ID reached {sorting['normalized_state_max_largest']:.2f}. The old amplification into the hundreds was not observed in these normalized inputs. This does not measure every custom API feature or establish that normalization choices are optimal.",
        '', 'Trajectory durations differ, particularly because APPL may finish early. These failure categories are descriptive and can overlap; their counts are not duration-matched estimates of recovery probability.']
    lines+=['','## 7. API and compute accounting','',
        '| Method/accounting scope | Consumed requests | Retained HTTP failures | Reported total tokens | Cached input tokens |',
        '| --- | ---: | ---: | ---: | ---: |']
    for cost in costs:
        states=cost['states'];usage=cost['usage']
        lines.append(f"| {LABELS[cost['method']]} | {states.get('consumed',0):,} | {states.get('http_failed',0):,} | {usage.get('total_tokens',0):,} | {usage.get('cached_input_tokens',0):,} |")
    lines+=['','The APPL 5.5 row is the incremental scale-up ledger, includes its one authorized 2026-09-19 reset, and excludes reused historical drawer design. APPL 6 covers its whole additional five-task study including all authorized recovery attempts. SinglePrior covers all five offline designs and its one retained tray failure; deployment requests are zero. These scopes must not be treated as identical end-to-end accounting boundaries. Exact input/output splits and scope notes are in [API_COSTS.json](API_COSTS.json).',
        '', 'Cached tokens are a subset of input tokens. HTTP failures can have unreported usage and unknown charges. Counts exclude this Codex conversation, GPU billing and unexecuted work. The retained Astra cost audit estimated $1,859.05 for identical reported usage at the audit’s specified direct-API rates; this is a historical counterfactual estimate, not a provider invoice or a newly checked price. '+link(ROOT/'experiments/exp2/analysis/astra_token_cost_20260918.json','Exact rates, scope and uncertainty')+'.',
        '', 'Training receipts record parameters, optimizer updates and elapsed time. Shared GPUs and different architectures make elapsed time or update counts unsuitable substitutes for matched FLOPs. Physical simulation pauses for API calls, so reported success does not establish online real-time feasibility.',
        '', '[Recorded deployment timings](tables/deployment_timing.csv) and [per-episode records](tables/deployment_episode_timing.csv) cover the 1,200 selected complete outcomes. These elapsed times include the actual execution conditions and differ with resource sharing, early termination and trajectory length. They exclude interrupted attempts and preparation, and must not be interpreted as exclusive GPU-hours or a matched latency benchmark.',
        '', '## 8. Development studies and reproducibility','',
        'The older M0 2/5 development and 4/15 confirmation results used different seeds and stricter terminal semantics. Preliminary APPL 1500/3000/5000 results reused the same five ID resets; the 5000 revision also changed feedback and context handling. They explain development but are not pooled into the main comparison. See [HISTORY.md](HISTORY.md).',
        '', 'The main studies retain original inputs, source hashes, API requests/responses, submission versions, checkpoints, paired reset states, per-action traces and original-frame videos. Interrupted attempts are retained; explicitly authorized evaluation recoveries reset the affected initial states. No completed task failure was rerun to replace its score. The old 5.5 interruption is preserved, and its authorized same-setting reset supplies the selected outcome; [recovery provenance](recovery_20260919/REPORT.md).',
        '', 'The 1,200 planned method-layout cells correspond to **1,292 recorded reset attempts**, including attempts interrupted before any action. APPL 6 used 391 attempts: 300 completed outcomes and 91 retained interruptions. APPL 5.5 used 301 attempts: 300 complete outcomes and one retained interruption. DP and SinglePrior each used 300 attempts. Thus there are 92 interrupted attempts in the main-study history and zero unresolved selected outcomes. These resets are not independent new layouts. The [attempt ledger](evaluation_attempts.json) preserves every directory and outcome hash; [summary](tables/evaluation_attempt_summary.csv).',
        '', 'SinglePrior evaluation used the authorized maximum of five physical GPUs (0/1/4/6/7). A separately recorded resource wrapper added devices 4/6 for 150 unstarted predeclared cells after ten zero-update, zero-physics action-equivalence checks. Resource configuration copies changed only device allocation. Original admission was guarded before the reserved range; completed outcomes were reused without repeating physical trials. Actual supplemental process exits, results and video hashes are audited in '+link(BASE/'evaluation_supplement/completed.json','the resource-completion receipt')+'. This changes resource scheduling, not model/source/data/seeds/sampling or goals.',
        '', 'Thirty predeclared cells from the original half were also scheduled early on the same original GPUs 0/1/7, using the unchanged original evaluation command and configurations. An admission guard prevents their original coordinator from reaching them before clean completion; its later completed-result path reuses them. These are part of the same 300 physical trials. Their actual process exits, outcomes and video hashes are separately audited in '+link(BASE/'evaluation_tail/completed.json','the thirty-cell scheduling receipt')+'.',
        '', 'All three supervised coordinators exited successfully and all 300 actual policy workers have clean closure receipts. The final [resource snapshot](resource_release.json) found no identified GPU process belonging to this study; unrelated GPU processes were preserved.',
        '', 'All project commands use the locked Exp2 Pixi environment. For read-only report regeneration after experiment completion:',
        '', '```bash', '/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked --no-install python -m experiments.exp2.paper.build build', '```',
        '', '## 9. Limits and claims supported','',
        'The report supports comparisons of these executed pipelines on these paired layouts, plus measured descriptions of their failures. It does not establish an isolated effect of a runtime agent, overlap, auxiliary loss, model family or reasoning effort. Each model has one training seed; later tests reuse layouts already examined. Goals are geometric and can be satisfied while an object remains grasped. OOD is position-only, demonstrations are scripted successful trajectories, and no real robot is tested. Recovery-support, matched-compute, matched-library agent ablations and repeated training/API seeds remain unexecuted follow-ups.',
        '', '## 10. Artifact access','',
        '- [Machine-readable complete comparison](results.json), [all tables](tables/), [policy/source index](POLICY_INDEX.csv).',
        '- [Actual physical-execution audit](physical_execution_audit.json) identifies the 300 new simulations and their real process records separately from later result-reuse jobs.',
        '- [Original-frame task and result figures](figures/); PDF/SVG exports support manuscript layout.',
        '- '+link(BASE/'paired_examples.html','four-method paired video examples')+'; '+link(BASE/'replays.html','all 300 new single-policy videos')+'.',
        '- '+link(ROOT/'runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/replays.html','all final APPL 6 videos')+'; '+link(ROOT/'runs/exp2/M1_scaleup/replays.html','original DP and APPL 5.5 videos')+'.',
        '- [Portable paired examples](execution_examples/index.html) include forty original videos, reset/results and twenty APPL invocation/notebook records, selected by the first predeclared seed. Larger replay collections remain in the repository.',
        '- [API-authored prior catalog](PRIOR_CATALOG.md) and [source packages](policy_cards/) retain exact API content; large checkpoints and raw trajectories remain at their indexed repository paths.',
        '- [Claim/evidence ledger](CLAIMS.csv), [provenance catalog](ARTIFACTS.json), [bundle completion/hash receipt](completion.json), [paper-writing handoff](WRITING_GUIDE.md).']
    lines += ['','[Supporting historical and diagnostic reports](supporting_reports/README.md) are also included as byte-identical copies. Their original relative links retain the repository context listed in the supporting-report index.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    atomic(OUT/'results.json',result)
    evidence={}
    def add(path,role,expected=None):
        p=ROOT/repo_path(path);assert p.exists(),p
        evidence[repo_path(p)]=dict(role=role,sha256=expected or digest(p),
            hash_source='Frozen training receipt' if expected else 'Recomputed during paper build')
    for path,role in [(BASE/'completion.json','New baseline completion'),(BASE/'results.json','Four-method outcomes'),
        (BASE/'study_freeze.json','Five frozen single-policy models and layouts'),
        (BASE/'design_completion.json','API identity, authorship and deduplicated usage'),
        (BASE/'incidents/tray_design_http502/continuation.json','Authorized exact-request design recovery'),
        (BASE/'evaluation_supplement/allocation.json','Resource-only evaluation allocation'),
        (BASE/'evaluation_supplement/checks_completed.json','Ten zero-update action-equivalence checks'),
        (BASE/'evaluation_supplement/completed.json','150 supplemental original physical trials'),
        (ROOT/'experiments/exp2/analysis/single_policy_evaluation_supplement.py','Frozen resource scheduling wrapper'),
        (BASE/'evaluation_tail/allocation.json','Resource-only scheduling of thirty original cells'),
        (BASE/'evaluation_tail/completed.json','Thirty original physical trials scheduled early'),
        (ROOT/'experiments/exp2/analysis/single_policy_evaluation_tail.py','Frozen thirty-cell resource wrapper'),
        (ROOT/'runs/exp2/M1_scaleup/completion.json','Original scale-up completion'),
        (ROOT/'runs/exp2/M1_scaleup/evaluation_recovery_20260919/completion.json','Authorized completion of the original unknown cell'),
        (ROOT/'runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/completion.json','Completed APPL 6 matrix'),
        (ROOT/'experiments/exp2/M0_REPORT.md','Frozen historical M0 report'),
        (ROOT/'experiments/exp2/SCALEUP.md','Original scale-up protocol'),
        (ROOT/'experiments/exp2/ASTRA_XHIGH.md','Additional APPL 6 protocol'),
        (ROOT/'experiments/exp2/SINGLE_POLICY.md','Single-policy protocol'),
        (ROOT/'environments/exp2/pixi.lock','Environment lock')]:add(path,role)
    for row in models:
        add(row['training_receipt'],'Formal training receipt')
        add(row['checkpoint'],'Model checkpoint',row['checkpoint_sha256'])
        for key in ('prior_document','handoff_document','authorship_audit'):
            if key in row:add(row[key],key)
    for row in episodes:
        root=Path(row['episode_root'])
        if row['completed']:add(root/'result.json','Selected complete physical outcome')
        add(root/'initial_state.json','Paired reset state')
    atomic(OUT/'ARTIFACTS.json',dict(reviewed_utc=reviewed,repository_root=str(ROOT),
        paths_relative_to_repository=True,artifacts=evidence,
        checkpoints_note='Checkpoint hashes copied from frozen training receipts; documentary source hashes recomputed. New-model checkpoint validation is in the completed experiment receipt.'))
    (OUT/'START_HERE.md').write_text('''# Exp2 论文材料入口

本包已在全部四组方法的结果核验后生成。先读 [英文完整报告](REPORT.md)，再读
[方法](METHODS.md)、[历史与协议区别](HISTORY.md) 和 [论文写作 AI 指引](WRITING_GUIDE.md)。

- `tables/`：结果、配对比较、任务、训练数据、种子/初态、计算量和 API 用量。
- `policy_cards/`：92 个 API 策略的原文档和源码，逐文件保持原样。
- `PRIOR_CATALOG.md`：按方法和任务排列的 API prior 原文摘要及源码入口。
- `execution_examples/index.html`：40 段可离线播放的原始视频，以及 APPL 调用与 notebook。
- `POLICY_INDEX.csv`：全部 97 个模型的来源、参数量、更新数、checkpoint 路径及哈希。
- `figures/`：可直接排版的 PNG/PDF/SVG 图。
- `FORMALISM.md`：数据窗口、扩散损失及 API 调用接口的数学定义。
- `framework/`、`protocols/`：冻结框架、环境锁、规范和配置的原样副本，供离线查阅。
- `ARTIFACTS.json`：原始证据路径；大模型权重、原始轨迹和视频保留在仓库运行目录。
- `completion.json`：完成状态与本包文件哈希。

可将本目录的 ZIP 交给写作 AI；它能直接读取主要报告、表格和 API 策略原文。
需要检查原始视频或 checkpoint 时，再按索引访问同一仓库。旧 APPL 5.5 中断已按授权补测，当前 1,200 个结果完整；补测记录与视频见
`recovery_20260919/`。原中断保留，不计为任务失败。不要将系统比较写成已经完成的单因素消融。
''')
    files={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*')
        if p.is_file() and p!=OUT/'completion.json' and '__pycache__' not in p.parts}
    atomic(OUT/'completion.json',dict(status='complete',reviewed_utc=reviewed,planned_cells=1200,
        complete_outcomes=1200,unknown_outcomes=0,models=97,API_authored_packages=92,
        work_scope='Read-only paper rebuild; the separately executed authorized recovery is recorded below.',
        recovery_receipt='recovery_20260919/completion.json',
        additional_API_calls=0,additional_training_updates=0,additional_simulator_steps=0,files=files))
    archive=OUT.parent/'Exp2_paper_bundle.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:z.write(path,Path('Exp2_paper')/path.relative_to(OUT))
    print(dict(report=str(OUT/'REPORT.md'),archive=str(archive),bundle_files=len(files)+1,
        complete_outcomes=1200,unknown=0),flush=True)
