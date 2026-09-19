# Five-task DP versus APPL

Completed outcomes: 599/600. Unknown/interrupted outcomes remain separate.

Both methods use paired initial observations, DDPM100 / execution 8, and the same geometric goals with a 5000-step cap.
API sees no total/remaining episode budget. Each trained library was frozen before test. Training is not compute-matched: APPL trains multiple policies.

| Task | Condition | DP successes / 30 | APPL successes / 30 | Unknown DP / APPL |
| --- | --- | ---: | ---: | ---: |
| drawer_exchange | ID | 15 | 29 | 0 / 0 |
| drawer_exchange | OOD | 0 | 14 | 0 / 0 |
| two_block_sort | ID | 1 | 23 | 0 / 0 |
| two_block_sort | OOD | 0 | 4 | 0 / 0 |
| buffer_swap | ID | 0 | 28 | 0 / 1 |
| buffer_swap | OOD | 0 | 13 | 0 / 0 |
| unstack_sort | ID | 1 | 29 | 0 / 0 |
| unstack_sort | OOD | 0 | 15 | 0 / 0 |
| tray_pack | ID | 0 | 21 | 0 / 0 |
| tray_pack | OOD | 0 | 4 | 0 / 0 |

Success counts with unknown outcomes are confirmed successes, not complete success-rate estimates.
Each task has one training seed and twelve demonstrations; these intervals concern test layouts, not training-seed variability.
Environment and diffusion seeds are fixed; API generation is not seeded. One API decision trajectory is tested per layout.
Simulation pauses during API calls. API latency and elapsed time are recorded separately; physical-step success does not establish real-time deployment.
Geometric success can precede release or stable rest. Position OOD keeps the task and dynamics fixed; it does not test new objects or robots.

![Comparison](comparison.png)

[All outcomes](episodes.csv) · [Raw summary and costs](results.json) · [Replay browser](replays.html) · [Data validation](data_validation.json)
