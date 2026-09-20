# GPT-6 Astra / xhigh: results after authorized evaluation recovery

Complete new-study outcomes: **300/300**; successes: **197**; task failures: **103**; unknown: **0**.
Original 211 complete outcomes are retained; only originally HTTP-interrupted cells receive the declared reset retest. All original attempts and the earlier zero-step interrupted retest remain separate.

| Task | Split | DP | APPL 5.5/high | APPL 6/xhigh |
| --- | --- | --- | --- | --- |
| drawer_exchange | ID | 15/30 | 29/30 | 8/30 |
| drawer_exchange | OOD | 0/30 | 14/30 | 7/30 |
| two_block_sort | ID | 1/30 | 23/30 | 28/30 |
| two_block_sort | OOD | 0/30 | 4/30 | 18/30 |
| buffer_swap | ID | 0/30 | 28/30; 1 unknown | 30/30 |
| buffer_swap | OOD | 0/30 | 13/30 | 8/30 |
| unstack_sort | ID | 1/30 | 29/30 | 30/30 |
| unstack_sort | OOD | 0/30 | 15/30 | 17/30 |
| tray_pack | ID | 0/30 | 21/30 | 28/30 |
| tray_pack | OOD | 0/30 | 4/30 | 23/30 |

![Three-method comparison](comparison.png)

Error bars are Wilson 95% intervals for complete groups. Hatching marks unknown outcomes; it is not a confidence interval. The old APPL comparison retains its separate historical missing outcome.

## Protocol and interpretation

The five tasks use the same 12 demonstrations per task, paired initial layouts, complete-demonstration normalization recipe and geometric goals. All new policies use 20,000 updates, seed 0, final EMA, DDPM100, history 2, horizon 16 and execution 8. The 5000-step cap is hidden from the inference API. API alone owns segmentation, priors, policy source and deployment decisions.
The additional library contains 36 policies and 720,000 formal updates; the retained 5.5/high library contains 51 and 1,020,000 historical updates. Both API model and reasoning effort changed, along with the API-generated library. This is not a single-factor model or effort ablation. These are follow-up paired layouts already analyzed previously, not untouched hidden tests. API generation is unseeded.
Recovery changes output paths and scheduling only. Every resumed initial state, prompt and first request is compared exactly with the original. Every recorded action, literal API stop condition, model identity and geometric result is audited. Task failures are retained without performance-driven retesting.

## Accounting

Additional training updates: 0. Resumed attempts: 90; completed resumed outcomes: 89. One earlier zero-step recovery interruption is also retained and charged separately.
Full new-study API states including all interrupted attempts: `{'consumed': 4742, 'http_failed': 94}`. Reported token totals: `{'cached_input_tokens': 11169664, 'input_tokens': 185052860, 'output_tokens': 2180887, 'total_tokens': 187233747}`. Failed requests may have unreported usage; no currency bill is inferred.

[Measured analysis](ANALYSIS.md) · [All selected new-study replays](replays.html) · [Fixed paired examples](paired_examples.html) · [API policies](../POLICIES.md) · [Raw results](results.json) · [Resumed execution audit](evaluation_audit.json) · [Original attempt report](../REPORT.md)
