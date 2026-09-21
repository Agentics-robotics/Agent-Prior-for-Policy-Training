# Five-task analysis

The original matrix has 599 complete outcomes and 1 unknown outcomes across 600 planned attempts.
Unknown outcomes are retained separately; they are neither policy failures nor silently replaced trials.

## Final comparison

| Task | DP ID | APPL ID | DP position OOD | APPL position OOD |
| --- | ---: | ---: | ---: | ---: |
| drawer_exchange | 15/30 | 29/30 | 0/30 | 14/30 |
| two_block_sort | 1/30 | 23/30 | 0/30 | 4/30 |
| buffer_swap | 0/30 | 28 confirmed; 1 unknown / 30 | 0/30 | 13/30 |
| unstack_sort | 1/30 | 29/30 | 0/30 | 15/30 |
| tray_pack | 0/30 | 21/30 | 0/30 | 4/30 |

![Five-task comparison including unknown outcomes](comparison_with_unknowns.png)

Complete groups have 95% Wilson intervals in [the report](REPORT.md). One training seed and one API decision trajectory per layout do not measure training or API sampling variability.

### Main interpretation

The four new tasks yield only 2/120 naive-DP ID successes. The verified implementation-equivalence results below exclude specific migration mistakes; they do not isolate why this monolithic learned baseline performs poorly.

Among 283 completed DP failures, 270 contain at least 100 consecutive state snapshots with finger width <=5 mm. The corresponding APPL count is 105/119. No original demonstration contains such a narrow-width state. This supports missing recovery-state coverage as a concrete hypothesis; finger width alone is not a grasp/contact classifier or causal proof.

APPL has 115 unsuccessful voluntary finishes and 4 failures at the physical cap. Most remaining failures therefore did not consume the whole 5000-step allowance. Increasing a hidden cap alone does not extend a recorded trajectory that the API already chose to end.

The study holds demonstration count, DDPM100 sampling and the declared training recipe fixed. It does not test whether additional demonstrations, larger networks, more updates or more denoising steps would solve these failures. Recovery coverage, phase/handoff representation and API stopping decisions remain candidates for a separately declared follow-up study.

## What additional physical steps changed

The following are success prefixes of the same frozen 5000-step trajectories. They are not separate budget experiments and do not change API decisions. The executor hides the total and remaining physical budget from the API.

| Task | Split | Method | By 1500 | By 3000 | By 5000 | Unknown final outcomes |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| drawer_exchange | ID | naive_DP | 15 | 15 | 15 | 0 |
| drawer_exchange | ID | APPL | 27 | 28 | 29 | 0 |
| drawer_exchange | OOD | naive_DP | 0 | 0 | 0 | 0 |
| drawer_exchange | OOD | APPL | 6 | 12 | 14 | 0 |
| two_block_sort | ID | naive_DP | 1 | 1 | 1 | 0 |
| two_block_sort | ID | APPL | 22 | 23 | 23 | 0 |
| two_block_sort | OOD | naive_DP | 0 | 0 | 0 | 0 |
| two_block_sort | OOD | APPL | 2 | 4 | 4 | 0 |
| buffer_swap | ID | naive_DP | 0 | 0 | 0 | 0 |
| buffer_swap | ID | APPL | 28 | 28 | 28 | 1 |
| buffer_swap | OOD | naive_DP | 0 | 0 | 0 | 0 |
| buffer_swap | OOD | APPL | 7 | 13 | 13 | 0 |
| unstack_sort | ID | naive_DP | 1 | 1 | 1 | 0 |
| unstack_sort | ID | APPL | 29 | 29 | 29 | 0 |
| unstack_sort | OOD | naive_DP | 0 | 0 | 0 | 0 |
| unstack_sort | OOD | APPL | 15 | 15 | 15 | 0 |
| tray_pack | ID | naive_DP | 0 | 0 | 0 | 0 |
| tray_pack | ID | APPL | 20 | 21 | 21 | 0 |
| tray_pack | OOD | naive_DP | 0 | 0 | 0 | 0 |
| tray_pack | OOD | APPL | 2 | 4 | 4 | 0 |

Across 300 complete naive_DP outcomes: 17 succeeded by 1500, 17 by 3000 and 17 by 5000 steps. Thus 0 observed completions occur after 1500, including 0 after 3000. Unknown final outcomes: 0.

Across 299 complete APPL outcomes: 158 succeeded by 1500, 177 by 3000 and 180 by 5000 steps. Thus 22 observed completions occur after 1500, including 3 after 3000. Unknown final outcomes: 1.

Counts are confirmed successes among the 30 planned layouts per row. An interruption before a given cap leaves that trajectory unknown at the cap.

## Failure evidence and its limits

Initial-state ID is a property of the reset distribution. After a missed grasp or poor placement, the policy can reach states absent from all twelve demonstrations. More diffusion sampling steps or a larger episode budget do not by themselves add examples of how to recover from those states.

Every original dataset has zero observations with summed finger-joint width at or below 5 mm. The four new datasets have a minimum width near 36.5 mm. [Training support](/home/users/oscar/agent_learning/Agent-Prior-for-Policy-Training/experiments/exp2/analysis/training_gripper_support.json) therefore does not contain the fully closed states observed after several empty grasps. Width is not a contact sensor; this is evidence of missing state support, not proof of a single failure cause.

| Task | Split | Method | Completed failures | No red 3 cm rise | No blue 3 cm rise | >=100 consecutive narrow-width states |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| drawer_exchange | ID | naive_DP | 15 | 10 | 15 | 15 |
| drawer_exchange | ID | APPL | 1 | 1 | 1 | 0 |
| drawer_exchange | OOD | naive_DP | 30 | 25 | 29 | 28 |
| drawer_exchange | OOD | APPL | 16 | 14 | 16 | 7 |
| two_block_sort | ID | naive_DP | 29 | 19 | 29 | 26 |
| two_block_sort | ID | APPL | 7 | 1 | 3 | 6 |
| two_block_sort | OOD | naive_DP | 30 | 27 | 30 | 26 |
| two_block_sort | OOD | APPL | 26 | 13 | 22 | 26 |
| buffer_swap | ID | naive_DP | 30 | 17 | 24 | 30 |
| buffer_swap | ID | APPL | 1 | 0 | 0 | 1 |
| buffer_swap | OOD | naive_DP | 30 | 28 | 30 | 29 |
| buffer_swap | OOD | APPL | 17 | 6 | 16 | 14 |
| unstack_sort | ID | naive_DP | 29 | 6 | 28 | 27 |
| unstack_sort | ID | APPL | 1 | 0 | 0 | 1 |
| unstack_sort | OOD | naive_DP | 30 | 24 | 29 | 30 |
| unstack_sort | OOD | APPL | 15 | 14 | 14 | 15 |
| tray_pack | ID | naive_DP | 30 | 28 | 30 | 29 |
| tray_pack | ID | APPL | 9 | 0 | 6 | 9 |
| tray_pack | OOD | naive_DP | 30 | 30 | 30 | 30 |
| tray_pack | OOD | APPL | 26 | 18 | 25 | 26 |

These descriptive counts can overlap. A 3 cm rise is not proof of a successful grasp; narrow width means <=5 mm and includes state snapshots. These analysis conventions do not alter task success.

### Normalization and baseline implementation

The read-only [baseline equivalence audit](/home/users/oscar/agent_learning/Agent-Prior-for-Policy-Training/experiments/exp2/analysis/naive_equivalence_audit.json) verified all four 60000-update models, complete-demo training windows, action transforms and normalization against the original M0 implementation. With the same final EMA and RNG, DDPM100 outputs match exactly on the checked original training histories. This excludes those particular migration errors; it is not a proof that every component or modeling choice is optimal.

| Task | Split | Method | Median episode max normalized input | 90th percentile | Largest |
| --- | --- | --- | ---: | ---: | ---: |
| drawer_exchange | ID | naive_DP | 1.43 | 1.66 | 6.18 |
| drawer_exchange | ID | APPL | 1.58 | 4.83 | 26.98 |
| drawer_exchange | OOD | naive_DP | 1.43 | 2.57 | 14.51 |
| drawer_exchange | OOD | APPL | 15.44 | 21.59 | 22.61 |
| two_block_sort | ID | naive_DP | 2.68 | 2.80 | 6.34 |
| two_block_sort | ID | APPL | 2.22 | 16.83 | 22.69 |
| two_block_sort | OOD | naive_DP | 2.68 | 2.89 | 3.77 |
| two_block_sort | OOD | APPL | 6.83 | 18.75 | 22.04 |
| buffer_swap | ID | naive_DP | 2.68 | 2.68 | 2.77 |
| buffer_swap | ID | APPL | 1.05 | 1.39 | 14.76 |
| buffer_swap | OOD | naive_DP | 2.68 | 4.67 | 6.91 |
| buffer_swap | OOD | APPL | 3.12 | 13.20 | 14.41 |
| unstack_sort | ID | naive_DP | 3.91 | 4.47 | 7.76 |
| unstack_sort | ID | APPL | 1.15 | 1.43 | 8.93 |
| unstack_sort | OOD | naive_DP | 2.68 | 3.70 | 5.65 |
| unstack_sort | OOD | APPL | 2.80 | 7.37 | 8.52 |
| tray_pack | ID | naive_DP | 2.68 | 2.68 | 21.21 |
| tray_pack | ID | APPL | 3.27 | 11.64 | 22.26 |
| tray_pack | OOD | naive_DP | 2.68 | 2.68 | 2.68 |
| tray_pack | OOD | APPL | 7.37 | 16.12 | 42.69 |

These are normalized observations actually preceding actions, excluding the terminal state. A large value alone does not establish a normalization bug. In particular, a velocity far outside training support can produce a large normalized input even with the verified full-demo scale.

### Illustrative recorded trajectories

- Sorting ID seed 20000: DP failed at 5000 steps; APPL succeeded at 680. The DP input magnitude peaked at 2.68; the blue block never rose 3 cm. This case is not explained by the old roughly 700-fold quaternion amplification.
- Sorting OOD seed 30000: APPL voluntarily stopped after 882 steps. Its gripper stayed fully closed after step 81. Recorded raw and executed commands match and the native controller maps them to closure; the framework did not rewrite an opening command.
- Drawer OOD seed 6406: APPL succeeded at step 3492 after 42 policy calls. The red goal was first met at 2445 and the blue goal at 3492. This post-hoc example shows a successful late recovery; the full prefix table establishes how frequent late successes are.

The first two examples are the first predeclared ID/OOD sorting seeds. The late drawer example was selected after observing its late success.

## Termination behavior

| Task | Split | Method | Recorded termination counts |
| --- | --- | --- | --- |
| drawer_exchange | ID | naive_DP | physical_budget_exhausted: 15, succeeded: 15 |
| drawer_exchange | ID | APPL | agent_finished: 1, succeeded: 29 |
| drawer_exchange | OOD | naive_DP | physical_budget_exhausted: 30 |
| drawer_exchange | OOD | APPL | agent_finished: 16, succeeded: 14 |
| two_block_sort | ID | naive_DP | physical_budget_exhausted: 29, succeeded: 1 |
| two_block_sort | ID | APPL | agent_finished: 7, succeeded: 23 |
| two_block_sort | OOD | naive_DP | physical_budget_exhausted: 30 |
| two_block_sort | OOD | APPL | agent_finished: 23, physical_budget_exhausted: 3, succeeded: 4 |
| buffer_swap | ID | naive_DP | physical_budget_exhausted: 30 |
| buffer_swap | ID | APPL | physical_budget_exhausted: 1, succeeded: 28 |
| buffer_swap | OOD | naive_DP | physical_budget_exhausted: 30 |
| buffer_swap | OOD | APPL | agent_finished: 17, succeeded: 13 |
| unstack_sort | ID | naive_DP | physical_budget_exhausted: 29, succeeded: 1 |
| unstack_sort | ID | APPL | agent_finished: 1, succeeded: 29 |
| unstack_sort | OOD | naive_DP | physical_budget_exhausted: 30 |
| unstack_sort | OOD | APPL | agent_finished: 15, succeeded: 15 |
| tray_pack | ID | naive_DP | physical_budget_exhausted: 30 |
| tray_pack | ID | APPL | agent_finished: 9, succeeded: 21 |
| tray_pack | OOD | naive_DP | physical_budget_exhausted: 30 |
| tray_pack | OOD | APPL | agent_finished: 26, succeeded: 4 |

An API `agent_finished` failure is its own decision to stop; an unused physical budget does not imply that the executor forced termination. `physical_budget_exhausted` is the private cap. Every `agent_finished` outcome was checked against a completed original API `finish` tool call; verbatim reasons are retained in the machine-readable analysis, as API explanations rather than independently established causes. API-request exhaustion raises a separate error. Transport-interrupted attempts are excluded from the completed-outcome counts above.

## Costs and interpretation

New formal optimizer updates: naive DP 240,000; API prior policies 660,000. Interface checks add 66 confirmed updates, with 4 additional updates as the retained uncertainty upper bound.

Two unsubmitted API drafts failed tensor-shape interface checks: buffer-swap blue-transfer h02 and unstack blue-placement h03. Runtime API corrected its unsubmitted source and both passed subsequent checks before formal training. The failed source/check receipts remain available in the raw result table; they are not additional formal models or concealed retries.

The study adds 4 naive models and 33 API prior models; the original drawer reuses 1 naive model and 18 API priors. Each naive model uses 60000 updates; each prior uses 20000. Consequently the full evaluated portfolio represents 300000 naive and 1020000 APPL formal updates. Update counts are not FLOP-normalized and model architectures can differ.

Verified physical GPU records for all 600 original evaluation attempts: {1: 145, 7: 154, 3: 149, 2: 152}. Only physical GPUs 1, 2, 3 and 7 occur; multiple processes share each card.

New-study API request states: {'consumed': 5619, 'http_failed': 1}. Reported usage: {'cached_input_tokens': 125503488, 'input_tokens': 195335211, 'output_tokens': 3414583, 'total_tokens': 198749794}. These include the four new segmentation/design stages and original-matrix inference, excluding historical drawer design. Failed-request usage can be unknown. Currency costs are not asserted because provider billing/rates are unavailable; raw journals and reported token counts are retained.

This compares the complete APPL system with naive DP. APPL also gains skill decomposition, multiple separately trained models and observation-dependent API selection. The result cannot isolate the causal contribution of a prior, overlap or model count. A future matched ablation would be needed; none was run using these test results.

The four new tasks use developer-designed scripted demonstrations with twelve fixed training seeds. Collection success is not learned-policy success. OOD changes initial positions only. The physical simulator pauses while awaiting the API, so these results do not demonstrate real-time control. The agreed geometric goal may be met before release or stable rest.

## Evidence and viewing

- [Final matrix, uncertainty and paired comparisons](results.json)
- [Five-task paired examples](paired_examples.html): first declared ID/OOD seed per task, without success-based selection.
- [All original replay clips](replays.html)
- [Completion and artifact audit](completion.json)
- [API-authored policy and handoff documents](POLICIES.md)
- [Machine-readable analysis](analysis_summary.json)

This analysis creates no new optimizer updates, API calls, physical steps or edited API outputs.
