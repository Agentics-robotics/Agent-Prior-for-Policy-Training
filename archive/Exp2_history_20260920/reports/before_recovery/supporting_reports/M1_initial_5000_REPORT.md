# M1_v2: API stopping conditions with private 5000-step limit

The same 18 frozen policies and initial seeds are reused. Additional training updates: 0.
These are new episodes from reset. The total and remaining physical budget are hidden from API input. This is a system revision with a new prompt, API-authored stop conditions and bounded context; it is not a single-factor budget comparison.

ID successes: 4/5 completed (5 planned). Of those, 3 succeeded within the first 1500 steps, and 1 required later steps.

| Seed | Scope | Original success | New success | New steps | Within original budget | Status |
| --- | --- | --- | --- | ---: | --- | --- |
| [1000](evaluation/1000/REPORT.md) | training reset | True | True | 987 | True | succeeded |
| [6200](evaluation/6200/REPORT.md) | same-seed ID retest | False | True | 1529 | False | succeeded |
| [6201](evaluation/6201/REPORT.md) | same-seed ID retest | False | False | 2070 | False | agent_finished |
| [6202](evaluation/6202/REPORT.md) | same-seed ID retest | False | True | 1453 | True | succeeded |
| [6203](evaluation/6203/REPORT.md) | same-seed ID retest | True | True | 1197 | True | succeeded |
| [6204](evaluation/6204/REPORT.md) | same-seed ID retest | True | True | 1058 | True | succeeded |

## Interpretation

Success remains the simultaneous drawer-open, red-on-pad, blue-inside conjunction. No terminal release, speed or hold condition is added. A new success beyond the old budget shows that this new trajectory needed additional execution time; it does not prove that the old episode would have followed the same continuation.

## Incremental accounting

Inference API requests: 94; input tokens: 3,236,552 (cached 2,092,288); output tokens: 40,379; cumulative request seconds: 948.2. Original design/training costs remain in the parent report and are not charged again. Dollar invoice unavailable.
Execution failures: 0. No automatic transport retries. Current library hashes verified: True.

[Original 1500-step report](../REPORT.md) · [Machine-readable comparison](summary.json) · [New freeze](library_freeze.json) · [Budget amendment](setup/amendment.json)

## Feedback protocol

The API supplies each invocation duration, numeric stopping conditions and notebook. The executor checks those exact conditions after every physical action and returns control at their first match; it never selects a successor or invents a skill threshold. All state/action trajectories and original API outputs remain saved.
API requests retain the original initial message, latest complete read of every policy, latest invocation/notebook and recent complete exchanges. Older invocations are available through read_invocation. No API text is rewritten. The private API request ceiling is recorded in the configuration and costs are incremental.

[Detailed analysis and remaining failure](ANALYSIS.md) · [Six recorded replays](visualizations/README.md) · [Completion and validation](completion.json)
