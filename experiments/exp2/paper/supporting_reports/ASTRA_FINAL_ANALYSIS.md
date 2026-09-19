# Measured analysis after the authorized resumption

The new GPT-6 Astra/xhigh study has 300 complete outcomes of 300: 197 successes, 103 task failures and 0 unknown outcomes.

## Five-task comparison

| Task | DP ID | APPL 5.5 ID | APPL 6 ID | DP OOD | APPL 5.5 OOD | APPL 6 OOD |
| --- | --- | --- | --- | --- | --- | --- |
| drawer_exchange | 15/30 | 29/30 | 8/30 | 0/30 | 14/30 | 7/30 |
| two_block_sort | 1/30 | 23/30 | 28/30 | 0/30 | 4/30 | 18/30 |
| buffer_swap | 0/30 | 28/30; 1 unknown | 30/30 | 0/30 | 13/30 | 8/30 |
| unstack_sort | 1/30 | 29/30 | 30/30 | 0/30 | 15/30 | 17/30 |
| tray_pack | 0/30 | 21/30 | 28/30 | 0/30 | 4/30 | 23/30 |

Each denominator is the declared 30 cells. An unknown outcome is not a task failure. The old 5.5/high buffer ID interruption remains unretouched and separate from the new-study recoveries.

## Paired evidence relative to the old APPL library

| Task | Split | Complete pairs | Both succeed | New only | Old only | Both fail |
| --- | --- | --- | --- | --- | --- | --- |
| drawer_exchange | ID | 30 | 7 | 1 | 22 | 0 |
| drawer_exchange | OOD | 30 | 5 | 2 | 9 | 14 |
| two_block_sort | ID | 30 | 21 | 7 | 2 | 0 |
| two_block_sort | OOD | 30 | 3 | 15 | 1 | 11 |
| buffer_swap | ID | 29 | 28 | 1 | 0 | 0 |
| buffer_swap | OOD | 30 | 4 | 4 | 9 | 13 |
| unstack_sort | ID | 30 | 29 | 1 | 0 | 0 |
| unstack_sort | OOD | 30 | 12 | 5 | 3 | 10 |
| tray_pack | ID | 30 | 19 | 9 | 2 | 0 |
| tray_pack | OOD | 30 | 4 | 19 | 0 | 7 |

These counts use identical initial-layout seeds. Exact paired tests are recorded in results.json as descriptive, unadjusted statistics; there is one training seed and the layouts were analyzed previously. They do not isolate the influence of model family or reasoning effort.

## Stopping and incomplete task goals

Failure termination statuses: `{'agent_finished': 103}`. Failure step range: `[437, 2408]`. Confirmed successes by step 1500/3000/5000: `{'1500': 188, '3000': 197, '5000': 197}`.
An API finish before the physical cap is an observed stopping decision. It does not establish that running longer would succeed, or that the global step cap caused the failure.

- drawer_exchange: 45 task failures; achieved-goal patterns: ['drawer_open']: 13; ['red_on_pad']: 19; ['none']: 11; ['blue_inside', 'red_on_pad']: 1; ['drawer_open', 'red_on_pad']: 1.
- two_block_sort: 14 task failures; achieved-goal patterns: ['red_at_goal']: 13; ['none']: 1.
- buffer_swap: 22 task failures; achieved-goal patterns: ['blue_at_goal']: 4; ['none']: 18.
- unstack_sort: 13 task failures; achieved-goal patterns: ['red_at_goal']: 4; ['none']: 9.
- tray_pack: 9 task failures; achieved-goal patterns: ['none']: 3; ['red_at_goal']: 6.

Original failed trajectories include missed grasps, empty closed grippers and displaced blocks, with the API explicitly ending after unsuccessful recovery attempts. Those original observations remain in the [pre-recovery analysis](../ANALYSIS.md). Final predicates and public API finish explanations support trajectory inspection; they are not controlled causal tests of architecture, policy switching or recovery coverage.


## Segmentation differences in the retained API outputs

| Task | Old skills | New skills | Old overlapping actions | New overlapping actions |
| --- | --- | --- | --- | --- |
| drawer_exchange | 6 | 3 | 5220 | 6626 |
| two_block_sort | 2 | 2 | 3600 | 4538 |
| buffer_swap | 3 | 3 | 3600 | 5832 |
| unstack_sort | 4 | 2 | 3840 | 4638 |
| tray_pack | 2 | 2 | 2400 | 4521 |

The drawer API changed from six skills (opening, red regrasp, red placement, blue acquisition, blue transport, blue release/retreat) to three broader skills (open_access, evacuate_red, insert_blue). Both manifests cover all 12 demonstrations and 12,809 unique actions with zero exclusions; reported overlap increased from 5,220 to 6,626 actions. Thus its regression cannot simply be described as missing demonstration coverage or fewer overlap actions. Coarser skill decomposition, learned phase distinctions, handoff behavior and the smaller total training budget are possible contributors, not isolated causal findings. No changes are made to the tested library.

## What changed and what is controlled

- Training demonstrations, task geometry, target predicates, paired seeds, complete-demonstration normalization recipe, DDPM100 and execution chunk of 8 are retained. The API does not receive the private 5000-step cap.
- Model and reasoning effort changed together from GPT-5.5/high to GPT-6 Astra/xhigh. API regenerated segmentation, priors, policy architecture/loss and handoff documents, then selected policies at inference. The comparison concerns this full pipeline.
- New versus old policy counts by task: drawer 9/18, sort 6/6, buffer 9/9, unstack 6/12, tray 6/6. The aggregate prior-training budgets are 720,000 versus 1,020,000 updates. DP training compute is also not matched to the policy portfolios.
- The retained one-line EMA reload verification correction matches requires_grad flags. It changes the verification gate only, not sampling, saved weights or deployment behavior. Its original evidence remains under the parent incidents directory.
- Environmental and diffusion seeds are fixed; API generation is not seeded. Interrupted cells were reset, not resumed from a saved simulator state. No completed task failure was retested or used to tune a prior or prompt.

## Outage recovery and cost

The original 300 attempts left 89 unknowns (82 HTTP 503, two HTTP 429 and five HTTP 502). A separately authorized first recovery failed on HTTP 503 before its first action. After the user reported service restoration, the present round was explicitly authorized for those same 89 cells. Original attempts, failed requests and their possible unreported costs remain in place.
Current resumption: 90 attempts, 89 complete outcomes. Added training updates: zero. Original files verified unchanged: 32209.
Whole-study API states including retained attempts: `{'consumed': 4742, 'http_failed': 94}`. Reported usage: `{'cached_input_tokens': 11169664, 'input_tokens': 185052860, 'output_tokens': 2180887, 'total_tokens': 187233747}`. Reported zero usage on an HTTP error is not evidence of zero charge.

## Viewing and provenance

[Comparison and uncertainty plot](REPORT.md) · [All 300 selected new-study videos](replays.html) · [Fixed paired examples](paired_examples.html) · [Exact API execution audit](evaluation_audit.json) · [Raw outcomes](episodes.csv) · [36 API policy pipelines](../POLICIES.md)

## Second explicitly authorized continuation

The service-restored round stopped after 69 attempts: 68 complete outcomes and one HTTP 502 (two_block_sort / OOD / 30027, step 679), leaving 20 cases unstarted. The user explicitly authorized this exact 21-case follow-up. All 69 preceding attempts, including the interrupted trace and its costs, remain in [the preceding report](../evaluation_recovery_20260918/REPORT.md). The cumulative resumed-attempt count includes both rounds; no completed task failure was retested.
