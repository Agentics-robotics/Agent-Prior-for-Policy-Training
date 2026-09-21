# M1_v2 inference revision: results and failure analysis

Reviewed 2026-09-16. This analysis is framework-authored; policy designs, invocation choices, stopping conditions and notebooks remain the original Runtime API outputs.

## Result

The new round completed all six attempts: **4/5 same-seed ID retests succeeded**, and the training-reset diagnostic succeeded (1/1). No episode reached the private 5000-step cap. There was no infrastructure interruption. All 18 frozen policies were reused; additional training updates: **0**.

| Seed | Scope | Original 1500-step round | Previous 3000-step round | New result | New physical steps | API requests |
| --- | --- | --- | --- | --- | ---: | ---: |
| 1000 | Training reset | Success | Failure | Success | 987 | 14 |
| 6200 | Same-seed ID retest | Failure | Failure | Success | 1529 | 18 |
| 6201 | Same-seed ID retest | Failure | Success | Failure | 2070 | 16 |
| 6202 | Same-seed ID retest | Failure | Interrupted | Success | 1453 | 18 |
| 6203 | Same-seed ID retest | Success | Failure | Success | 1197 | 15 |
| 6204 | Same-seed ID retest | Success | Success | Success | 1058 | 13 |

Within the new trajectories, three ID seeds succeeded by 1500 steps (6202, 6203, 6204), and a fourth at step 1529 (6200). There were **no additional successes after step 3000**. The failure, 6201, voluntarily finished at 2070 with 2930 physical steps unused. The larger ceiling was available, but its full size was not needed by any success.

These seeds were previously inspected. The revision changed feedback, prompt, context selection, budget visibility, total cap and API request allowance together. The observed 4/5 is a small development retest, not a new independent 80% generalization estimate or a causal ablation of any one change. The previous 3000-step run had two successes, two task failures and one interrupted/unknown outcome; it must not be treated as five fully evaluated failures/successes.

## What changed, and who decides when to stop

1. **One invocation:** API chooses a frozen policy, a maximum duration of 1–300 steps and numeric stopping groups. The framework executes actions and checks the literal conditions after every step. It returns at the first match or chosen duration; API then decides whether to continue, switch or finish. The API is not queried on every physical step.
2. **The whole episode:** API can explicitly call `finish`. Independently, the executor ends when all three measured task goals hold or when its private 5000-step cap is reached. The API is not told the total/remaining budget; it can see elapsed steps as a timestamp and the 300-step per-invocation limit.
3. **Feedback and memory:** current state, metric ranges and the latest invocation replace repeated full invocation-history payloads. Each API-authored notebook is preserved verbatim. Original task, policy documents, latest invocation and six recent complete exchanges remain in request context; older invocations are readable by tool. The complete raw history stays on disk.

Across 54 invocations, **30 returned on an API-authored condition before the requested duration**, five ended on measured task success, and 19 ran their requested duration. No framework-authored semantic threshold or successor policy was inserted. API conditions and notebook strings were checked against the actual raw function-call arguments.

### Concrete successful handoff

In seed 6203, API ran blue-grasp h02 from step 793 with a maximum duration of 260. It selected the conjunction `blue_z > 0.20`, `finger_width < 0.045`, `tcp_blue_distance < 0.07`. At step **1004**, after **211** steps, these conditions first held and control returned 49 steps before the requested endpoint. API then selected blue-transport h03; the three task predicates became true at **1197**. This demonstrates that the new interface exposes a useful transfer state immediately. It does not prove what the unexecuted remaining 49 grasp steps would have done.

The original failure review found blue blocks that rose and fell inside a single invocation, while the API saw only its endpoint. The new interface directly addresses that information/timing problem. Metric extrema additionally preserve evidence of intermediate progress even when a condition does not fire.

## Remaining failure: seed 6201

The drawer needed to exceed 0.26 m. It ended at 0.2403867 m; red remained at drawer-front height (~0.063 m), and blue remained on the table (~0.020 m). All three final goals were false. No stopping condition fired during this episode.

| Steps | API choice | Observed result |
| --- | --- | --- |
| 0–300 | Drawer h03 | Drawer ended at 0.2227 m; within-call peak 0.2460 m. |
| 300–400 | Continue h03 | End 0.2405 m; peak 0.2548 m, still short of the goal. |
| 400–550 | Switch to h02 | Fingers reopened and TCP retreated; drawer stayed near 0.2404 m. |
| 550–1030 | Drawer h01, then h03 | Neither recovered the missing opening progress. |
| 1030–1810 | Red grasp h01 twice, then h03 | Red never lifted. These attempts did not establish a grasp: fingers stayed largely open, and the closest recorded TCP–red distance during these attempts was about 9.7 cm. |
| 1810–2070 | Drawer h01 again | No further opening; API then called `finish`. |

### A directly observed alternative from the previous round

The first **400** physical steps are exactly equal between this run and the successful previous 3000-step run: policy IDs, raw actions, clipped actions and full states match. At step 401 the branches separate. The previous run continued **h03** for another 200 steps and reached drawer position **0.2963881 m** at step 600; it eventually succeeded at 1782. The current run switched to **h02**, leaving the drawer at **0.2403945 m** at step 600.

Thus the existing h03 had a recorded successful continuation from the same prefix. The current API interpreted slow/non-monotonic progress as stalling and switched before that continuation occurred. Once h02 released and retreated, calling h03 later started from a different physical state and policy context; it was not equivalent to uninterrupted h03 continuation. This is evidence of a continuation/selection error plus unsuccessful recovery, not evidence that 5000 steps is too short or that the learned library lacks any solution for this initial state.

The API later judged the remaining learned policies unlikely to recover. Its complete unmodified final rationale is saved in [analysis_evidence.json](analysis_evidence.json) and the [raw journal](evaluation/6201/api/journal.sqlite). The decision to quit at 2070 was its own; the executor did not tell it that it was running out of time. We cannot conclude that more attempts from its final state would have succeeded.

## Other seeds and what the improvement supports

- **6200:** several initial red-grasp/recovery attempts failed, but API restored the drawer opening before completing red placement and blue transfer. It returned from blue-grasp h02 at 1409 on its own lift conditions, then succeeded at 1529. This particular new trajectory needed 29 steps beyond 1500; it did not need 3000–5000.
- **6202:** it recovered from failed red-grasp variants, returned from blue-grasp at 1331 and succeeded at 1453. The previous run was interrupted by excessive API context at step 2740; this time all 18 requests completed.
- **6203:** timely blue handoff at 1004 led to success at 1197. The drawer was nearly stationary at success, unlike the original 1500-step run’s high-speed terminal drawer event.
- **6204:** it selected red-grasp h02 and blue-grasp h03, succeeded at 1058, and remained successful across all three rounds.
- **Diagnostic 1000:** seven API condition returns led to success at 987; this reset comes from training and is reported separately from ID.

Together these observations support prioritizing handoff timing and policy continuation/recovery decisions. They do not isolate an effect of hiding the budget, nor justify retraining, more diffusion steps or a looser task threshold on their own. No goal threshold was relaxed in this revision.

## Context, reproducibility and limitations

Maximum request input was **56,969 tokens**, versus 261,961 in the last successful response preceding the previous context failure. All **94** requests completed and were consumed. Projection was reconstructed request-by-request from the full raw journal; retained API output items and tool exchanges matched exactly. This demonstrates that the tested episodes avoided the old growth problem, not a universal token bound for arbitrary future tool use.

Initial states matched the original six resets exactly; policy weights, normalization, DDPM settings and seeded workers were preserved. Provider requests did not supply a generation seed or temperature. This revision also deliberately changes API inputs/tools, so trajectory differences cannot be attributed only to API sampling randomness.

The success definition is still the simultaneous geometric conjunction. In successful seeds 1000, 6200 and 6202, terminal finger width was about 0.0365 m with TCP–blue separation about 0.009 m, consistent with the blue block still being held. Release/settling was not required and no continuation was run to establish it. All five successful terminal drawer speeds had absolute value below 0.001 m/s; this does not establish a sustained stable whole-task terminal state.

One retained API condition in seed 6204’s final invocation included both `blue_x < -0.25` and `blue_x > -0.09` in the same AND group. That optional group was unsatisfiable and never triggered; it was not edited. Another group and the independent whole-task predicate covered success. Generic contradiction feedback could be investigated in a future revision, but was not silently added after this result.

## Validation and incremental accounting

- **39** framework tests passed, including literal stopping, hidden cap enforcement, request context provenance, unchanged legacy protocol and actual Gym wrapper limits.
- **8,294** recorded physical steps and all **94** outgoing requests audited; all 30 condition stops occurred on the first satisfying step. All workers closed successfully.
- All **3,747** original M1_v2 and budget_3000 files retained their hashes; the original 18-policy library identity, configuration, normalizer and native environment were preserved. Exp1 and M0 were not modified.
- Additional training: **0**; automatic transport retries: **0**; API outputs rewritten: **0**; detected budget disclosures: **0**.
- Incremental API usage: **3,236,552 input tokens**, including **2,092,288 cached input tokens**, and **40,379 output tokens**; **948.2 cumulative API seconds**. Cached input is a subset of input, not an additional charge. Dollar invoice is unavailable. Original design/training costs are not counted again.

[Machine-readable completion](completion.json) · [Step/context audit](evaluation_audit.json) · [Detailed comparison](analysis_evidence.json) · [Preservation receipt](setup/parent_preservation_after.json) · [Camera replays](visualizations/README.md) · [Original failure review](../budget_3000/FAILURE_REVIEW.md)
