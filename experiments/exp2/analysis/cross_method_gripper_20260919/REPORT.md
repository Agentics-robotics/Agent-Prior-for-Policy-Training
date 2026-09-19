# Does the gripper problem also affect the prior-based methods?

Reviewed 2026-09-19. Read-only analysis of all **1,200 selected formal evaluation outcomes**. No new simulation, training, GPU experiments, or Runtime API requests. All 92 API policy source files match their exported paper copies. Original results and API output are unchanged. [Completion and source audit](completion.json).

## Conclusion

**All three prior-based methods use the same continuous gripper execution interface, and all three have recorded trajectories with progressive finger closure followed by closure in free space.** The frequency and circumstances differ substantially. The single-policy prior is not immune; the new APPL has fewer examples under the conservative screen used here, particularly in ID. Some old-APPL episodes recover and succeed after an event.

The preceding [naive-DP study](../dp_diagnosis_20260919/REPORT.md) demonstrated a finger-state feedback dependency through controlled input intervention and a live binary-gripper comparison. **This new cross-method review is observational.** It identifies similar trajectories and a shared interface, not proof that the same feedback mechanism causes every event or failure. Binary-gripper interventions have not yet been run on these three methods.

## 1. Shared interface, different learned functions

The three compared methods are:

- `SinglePrior_6_xhigh`: one API-designed full-task policy; no agent at deployment.
- `APPL_5_5_high`: the original API-designed skill library and runtime selector.
- `APPL_6_xhigh`: the later Astra/xhigh skill library and runtime selector.

For all three, `appl.prior_policies.engine.sample` produces a continuous eight-dimensional action through DDPM and `denormalize_action`. The two deployment paths then call `np.clip(raw, space.low, space.high)` and submit the result to the same environment controller. Neither deployment path applies a sign function or requires the gripper to remain fully open. Across all 1,200 selected formal traces, **the executor changes the raw gripper value on zero steps**; recorded intermediate values are passed through.

Evidence in the live, frozen-equivalent framework:

- [APPL deployment](../../../../src/appl/prior_policies/deploy.py): `DeploymentTools.invoke`, lines 194–201.
- [Single-policy deployment](../../single_policy/evaluate.py): evaluation loop, lines 83–87.
- [Shared sampling](../../../../src/appl/prior_policies/engine.py): `sample` and `LoadedPolicy.action`.
- [Action decoding](../../../../src/appl/public.py): `denormalize_action`.

An API prior changes the learned representation, model structure or loss, so it can change sensitivity to finger errors. It does not automatically change the native controller's interpretation of an intermediate command. For example, the single sorting prior includes learned finger forecasts and relative object features; these are learned conditions, not an executable guarantee against premature closing. Some Astra policies have explicit learned gripper-persistence heads and losses, but their final action still passes through continuous DDPM decoding.

## 2. Conservative trajectory screen

The screen operates on the four added block tasks. Drawer is excluded from this count because a handle is an additional legitimate grasp target. Each method has **120 ID and 120 position-OOD episodes** in the screened subset.

A flagged event must satisfy all of the following:

1. A positive-to-negative gripper-command transition, with the same policy throughout 24 preceding and 20 subsequent actions.
2. All preceding 24 commands nonnegative, at least eight strictly between 0 and 0.95.
3. Total finger opening decreases at least 12 mm, from at least 70 mm to at most 65 mm before the negative action.
4. At that time the TCP is more than 80 mm above both cube centers; both cubes remain within 10 mm of their event positions during the preceding window.
5. Within the following 20 actions the opening reaches at most 5 mm and neither cube moves more than 10 mm.

This excludes visibly carried cubes and policy-switch boundaries, and focuses on progressive closure rather than every low gripper command. It is still a screening heuristic, not contact ground truth: intentional empty-hand closing can be flagged, and other grasp/timing failures can be missed. The 1 cm movement bound uses full XYZ displacement. Counts below are **episodes containing a signature**, not failures causally attributed to it.

| Method | Flagged ID episodes / 120 | Flagged OOD episodes / 120 | Flagged episodes that eventually succeed, ID / OOD |
| --- | ---: | ---: | ---: |
| Naive DP | 68 | 91 | 0 / 0 |
| Single-policy prior, Astra/xhigh | **24** | **52** | 0 / 0 |
| APPL GPT-5.5/high | **15** | **32** | **13 / 4** |
| APPL Astra/xhigh | **0** | **10** | 0 / 0 |

All fifteen flagged old-APPL ID episodes are tray trials. Single-prior ID flags are 23 sorting trials and one tray trial. Astra APPL's ten flagged OOD episodes are one sorting, three buffer-swap and six unstack trials. Zero flagged Astra ID episodes is not proof of immunity or absence of other empty-grasp mechanisms.

Raw episode-level windows and hashes: [episodes.json](episodes.json). Full method/task/condition counts: [summary.csv](summary.csv). In the machine summary, `task=all` includes all 150 episodes per condition, including the thirty unscreened drawer trials; the flagged counts remain the same. The table above explicitly uses only the 120 screened block episodes.

## 3. Concrete examples

### Single-policy prior: sorting ID seed 20000

The gripper command at recorded steps 32/40/48/56/64 is approximately **0.986 / 0.967 / 0.913 / 0.772 / 0.393**. Actual opening falls from 79.3 mm to 49.8 mm. Action 65 commands **−0.415** while the input TCP is **186 mm above the blocks**; the fingers subsequently close to zero and neither object moves. The episode fails after 5,000 steps.

This is strong observational evidence that the single prior did not prevent the progressive closure pattern seen in naive DP. The sorting prior's explicit finger forecasting is not sufficient to guarantee stable execution.

[Original trace](../../../../runs/exp2/single_policy_astra_xhigh/two_block_sort/evaluation/SinglePrior_6_xhigh/ID/20000/trace.jsonl).

### Old APPL: tray ID seed 20301, followed by recovery

During `acquire_block__h03`, opening drops from approximately 74 mm to 55 mm. Action 419 becomes negative while the TCP is **162 mm above the higher cube**. The gripper closes empty.

The retained API invocation journal then shows:

1. The failed `acquire_block__h03` call ends at step 586.
2. At steps **587–592**, the API selects `place_block__h02`, explicitly identifying the empty closed gripper and using its release behavior to reopen it.
3. At step **593**, it selects **`acquire_block__h02`**, a different contact-phase prior, explicitly citing the failed h03 grasp timing.
4. Blue is grasped and transported; `place_block__h03` completes the task at **step 910**.

This is direct journal evidence of an agent-mediated recovery in one episode. The 13/15 flagged ID episodes that eventually succeed do not by themselves prove that the agent was necessary for all thirteen recoveries; that would require a matched intervention.

[Original invocation journal](../../../../runs/exp2/M1_scaleup/tray_pack/evaluation/APPL/ID/20301/invocations.json), [trace](../../../../runs/exp2/M1_scaleup/tray_pack/evaluation/APPL/ID/20301/trace.jsonl), and [result](../../../../runs/exp2/M1_scaleup/tray_pack/evaluation/APPL/ID/20301/result.json).

### New APPL: unstack OOD seed 30204

Within `acquire_lift__h03`, commands at actions 97/105/113/121 are approximately **0.916 / 0.814 / 0.521 / −0.120**. Input opening falls from 77.0 mm to 55.4 mm; by action 121 the TCP is **91 mm above the upper cube**. The fingers subsequently close empty and both cubes remain essentially stationary. The episode fails.

Here the arm is already moving upward when the command becomes negative. This could involve an incorrect grasp/lift timing prediction as well as finger-state feedback. The observed signature alone cannot determine which deviation initiated the failure.

[Original trace](../../../../runs/exp2/M1_astra_xhigh/unstack_sort/evaluation/APPL/OOD/30204/trace.jsonl).

## 4. The intermediate-state coverage issue is also shared

All four methods ultimately train from the same twelve successful demonstrations per task. API segmentation and larger overlap add temporal context and reuse samples; they do not create new examples of offset placement, empty grasps or recovery states.

Relative features and geometry/phase losses may reduce a model's sensitivity to irrelevant position differences. APPL can also select the next policy after a geometric subgoal succeeds, or choose another recovery policy after a failure. These are plausible ways to tolerate deviations, not guarantees. The original drawer regression and recorded recovery failures already show that skill handoffs and retries can still fail.

The current read-only screen does not measure the causal contribution of intermediate-state coverage separately for each prior. It would be incorrect to attribute every APPL failure to the naive-DP gripper finding, or to conclude that high aggregate success means this interface is harmless.

## 5. Implication for the comparison

The previous naive-DP diagnosis is not evidence that only naive DP is exposed to the interface weakness. The common interface is shared, but learned policies have different sensitivity and APPL has additional recovery opportunities. The newer APPL's lower flagged rate also does not isolate the effect of the language model: priors, segmentation, networks, training amounts and policy selection differ.

A fair next experiment would compare original versus binary-gripper execution **within each of the four frozen methods**, keeping policy source/checkpoints and initial seeds fixed and recording it as a new inference configuration. Start with a declared small diagnostic set, then freeze the rule before a broader comparison. Do not combine a repaired method's scores with other methods' historical scores and call that a controlled comparison. Binarization can also worsen individual episodes, as already observed for drawer DP.

This report adds no evidence that more DDPM steps or more training alone would repair these other methods. Their individual causal effects remain to be tested.
