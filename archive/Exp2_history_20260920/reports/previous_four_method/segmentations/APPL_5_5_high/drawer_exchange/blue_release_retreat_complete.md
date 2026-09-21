# blue_release_retreat_complete

Locally center/release the blue block inside the open drawer and retreat while preserving the completed task state.

## Segmentation

This final dataset begins at index 930, before insertion is finished, when blue is still carried high near the drawer mouth, continues through the open/release state at index 1030, and includes all remaining actions to the final observation so settling/retreat context is retained. Under the task contract, completion is achieved when drawer_open, red_on_pad and blue_inside hold at one observation; release and retreat are not required but retained as useful interface context.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo1000`: [930, 1065)
- `demo1001`: [930, 1077)
- `demo1002`: [930, 1063)
- `demo1003`: [930, 1066)
- `demo1004`: [930, 1065)
- `demo1005`: [930, 1070)
- `demo1006`: [930, 1064)
- `demo1007`: [930, 1069)
- `demo1008`: [930, 1071)
- `demo1009`: [930, 1065)
- `demo1010`: [930, 1069)
- `demo1011`: [930, 1065)

## Heuristic 1

Completion-predicate local insertion prior: guide final blue release using drawer/red/blue success predicates and drawer-frame error.

**Evidence inspected by the API**

- `demo1000` observations: 930, 1030, 1065
- `demo1001` observations: 930, 1030, 1077
- `demo1011` observations: 930, 1030, 1065

**Interpretation**

Late actions are local corrections and retreat around the task completion boundary. Providing explicit predicate features and future completion supervision should focus the learner on the evaluator-relevant geometry rather than arbitrary final joint pose.

**Applicability**

Transfers when the drawer is open, red is on the pad, blue is already near/inside the drawer mouth, and task completion is blue_inside plus drawer_open plus red_on_pad. Requires state observations for all three predicates.

**Implications for a future training/inference pipeline**

Inputs causal full state. Compute current predicates red_on_pad, drawer_open, blue_inside from task contract and drawer center. Candidate DP conditions on drawer-frame blue error and predicate vector; train auxiliary classifier for future completion C_{t+H}. At inference sample actions that maximize DP likelihood plus predicted completion score, without using future labels.

**Assumptions and limitations**

Completion rule does not require gripper release or object velocity, though retreat is retained for interface safety. The dataset cannot show recovery if blue is dropped outside the drawer. Compare predicate-conditioned local policy to pure end-of-demo cloning; expected benefit is fewer near-threshold cavity misses.

### Handoff interface

**Entry conditions**

Supported entry at index 930 has blue carried high near x -0.17 to -0.18,y 0.065-0.080,z ~0.32, fingers closed, drawer_position ~0.297, red already on pad. It may also enter later when blue is lower/near release.

**Exit conditions**

Task-complete useful end state: drawer_position >0.26, red_pose inside pad bounds at z ~0.02, blue_pose inside drawer cavity with z about 0.063, and gripper open/retreated upward by final observation (tcp z ~0.323).

**Failure signatures**

blue final xy not inside drawer bounds, z outside [0.053,0.074], drawer closes below threshold, or tcp remains pressing on blue causing displacement.

**Overlap role**

[930,1030) is shared with transport so both learn local centering/descent before and during release; this final policy also includes post-release retreat/settle through the last action.

**Successor readiness**

No successor for task, but a monitor can declare completion when current predicates satisfy drawer_open AND red_on_pad AND blue_inside at one observation.


## Heuristic 2

Support-before-retreat prior: open and withdraw only after blue is likely supported inside the drawer.

**Evidence inspected by the API**

- `demo1002` observations: 930, 1030, 1063
- `demo1003` observations: 930, 1030, 1066
- `demo1008` observations: 930, 1030, 1071

**Interpretation**

At index 1030 the gripper is open while tcp is already lifting, and final observations keep blue fixed in the drawer. A support latent separates object placement from arm retreat, reducing disturbances during withdrawal.

**Applicability**

Useful when the gripper can open after blue reaches a stable support surface inside the drawer and then retreat vertically without disturbing it. Requires qpos finger state and blue_pose.

**Implications for a future training/inference pipeline**

Train a release-support latent r_t. Labels from future: blue_pose remains within cavity after gripper qpos opens and tcp z increases. Inputs are causal blue_pose,tcp_pose,drawer_position,qpos. Decoder conditions on r_t: if not supported, continue small descent/centering; if supported, open and retreat. Objective adds future blue displacement prediction after release.

**Assumptions and limitations**

No force or contact label confirms support, and no failed early releases are present. Falsify by ablation without support/stability auxiliary on perturbations in release height; prior should reduce blue-out-of-drawer after retreat.

### Handoff interface

**Entry conditions**

Enter with blue held over/in the open drawer, fingers closed or just opening, and blue z descending toward the 0.053-0.074 band. Red must already be on pad and drawer open.

**Exit conditions**

Fingers open near 0.04 m, blue remains in drawer at z ~0.063, tcp retreats upward to z about 0.30-0.323; final qvels nearly zero in many demos.

**Failure signatures**

blue follows tcp upward after opening, object is pushed out during retreat, or gripper opens before blue is supported in the drawer.

**Overlap role**

The overlap with transport includes release initiation; both policies learn the open/retreat transition, reducing reliance on an exact release timestep.

**Successor readiness**

Completion monitor needs stable object predicates; if blue moves with tcp after opening, retreat should be slowed or aborted by a trained candidate, though no recovery examples exist.


## Heuristic 3

Near-goal residual stabilizer: specialize final actions to small drawer-frame corrections, release and retreat from near-completion states.

**Evidence inspected by the API**

- `demo1006` observations: 930, 1030, 1064
- `demo1007` observations: 930, 1030, 1069
- `demo1010` observations: 930, 1030, 1069

**Interpretation**

The final segment is much narrower in state distribution and task objective than long-range transport. A residual stabilizer should learn smaller corrections and be testably better at avoiding threshold misses while explicitly acknowledging it is not a general transporter.

**Applicability**

Applies when final motion can be seen as a short-horizon stabilizing controller around the drawer cavity, not a long-range planner. Requires near-goal entry; not meant for earlier blue transport.

**Implications for a future training/inference pipeline**

Implement as residual local policy around a nominal drawer-frame controller: u_nom drives blue_error to zero and opens at small error; DP predicts residual joint action Δa and gripper residual. Inputs causal full state; training targets are residuals obtained by subtracting nominal IK/action approximation from demonstration actions. At inference only activate when ||blue_xy-drawer_xy|| is within a demonstrated near-goal envelope (about states at 930 onward).

**Assumptions and limitations**

This prior assumes near-goal initialization and cannot transport blue from the table. Compare against the full transport policy on early entries; this local policy should fail there but be more precise near the drawer. Failure to improve near-goal precision would falsify it.

### Handoff interface

**Entry conditions**

Blue should already be above/near drawer xy, drawer open, red on pad, gripper closed/closing around blue. States far from the drawer at table blue start are outside support.

**Exit conditions**

Final observation has blue stationary at z about 0.063, drawer_position ~0.297, tcp high/clear, gripper open; action sequence ends with low qvel.

**Failure signatures**

Large blue-drawer error at entry, oscillatory corrections around cavity, or retreat before convergence into the z/xy acceptance box.

**Overlap role**

[930,1030) supplies near-goal entries with the arm still descending, so the local controller can learn to take over before exact predicate satisfaction.

**Successor readiness**

No manipulation successor; downstream evaluator/monitor should check predicates rather than require a particular tcp pose.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
