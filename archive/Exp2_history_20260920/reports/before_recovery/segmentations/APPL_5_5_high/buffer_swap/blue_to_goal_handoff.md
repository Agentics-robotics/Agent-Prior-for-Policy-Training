# blue_to_goal_with_red_handoff

Pick the blue block from the upper/red-goal region, place it at blue_goal, and continue the demonstrated handoff toward reacquiring the buffered red block.

## Segmentation

Each segment starts inside the expanded predecessor overlap after red has been buffered, before or during blue acquisition, and continues through the complete blue transfer to blue_goal plus an expanded successor overlap into red reacquisition and lift. Start index 320 is high/open with red buffered; stop index 860 is after blue placement and during/after red pickup from the buffer. The grouped demonstrations share the same middle learning problem: move blue from the initially upper/red-goal region to the lower blue_goal while preserving the buffered red for the final transfer.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo10100`: [320, 860)
- `demo10101`: [320, 860)
- `demo10102`: [320, 860)
- `demo10103`: [320, 860)
- `demo10104`: [320, 860)
- `demo10105`: [320, 860)
- `demo10106`: [320, 860)
- `demo10107`: [320, 860)
- `demo10108`: [320, 860)
- `demo10109`: [320, 860)
- `demo10110`: [320, 860)
- `demo10111`: [320, 860)

## Heuristic 1

Active-object reflection prior: learn the blue transfer in a coordinate system tied to the active object and its goal, with approximate y-reflection symmetry to exploit the repeated pick-carry-place structure across the swap.

**Evidence inspected by the API**

- `demo10100` observations: 320, 460, 550, 600, 650, 700, 860
- `demo10103` observations: 320, 460, 700, 860
- `demo10111` observations: 320, 460, 700, 860

**Interpretation**

At index 320 in demo10100 red is buffered and blue remains at the upper region; by 460 the gripper closes over blue; by 600 blue is carried near the lower goal and by 650 it is placed near [-0.352,-0.201,0.02]. This resembles the first transfer but with active object blue and target blue_goal. Demo10103 and demo10111 show the same sequence with different starting blue y/pose. The learning problem is sample-efficient generalization over spatial direction and color role. This prior assumes a reusable reflected manipulation schema; it differs from the placement prior by emphasizing symmetry in representation rather than terminal accuracy.

**Applicability**

Transfers to the middle phase of the same buffer-swap task when red is already buffered and the active object is the blue block initially near the red goal. It assumes the world goals are fixed at red_goal=[-0.35,0.2,0.02] and blue_goal=[-0.35,-0.2,0.02] or represented in the observation, with similar top-down grasps and table geometry.

**Implications for a future training/inference pipeline**

Build a policy with a y-reflection/object-swap equivariant encoder. Causal inputs are qpos/qvel, tcp_pose, red_pose, blue_pose, red_goal, blue_goal; choose active object o=blue. Encode relative features in an active-object/goal frame: phi=[tcp-o, blue_goal-o, red_pose-o, red_goal-blue_goal, qpos fingers]. Add an augmentation during training reflecting y -> -y and swapping roles where physically valid within the task schema; actions remain absolute joint targets but the encoder shares weights for reflected spatial channels and lets the decoder map to joint space. Objective is diffusion action loss plus consistency loss f(T obs)=T_a f(obs) in TCP/task-space latent before joint decoding. Future labels are only used to mark active phase for training; inference uses current observations.

**Assumptions and limitations**

The symmetry is approximate: initial red-to-buffer and blue-to-goal have different source/target locations and successor states, so a strict reflection could overconstrain actions. A falsifying ablation is a non-equivariant goal-conditioned DP with equal capacity; if it handles y-direction swaps equally well under object pose perturbations, the equivariance is not needed. The data cannot establish general color invariance beyond this two-block swap.

### Handoff interface

**Entry conditions**

Measured entry spans the overlap from skill 1: at index 320 red_pose is buffered near [-0.181,0,0.02], blue_pose remains on the upper region, qpos[7:9] are open near 0.04 m, and tcp_pose is high near z 0.28-0.30 m; by index 460 TCP is low over blue and qpos[7:9] about 0.018 m as blue grasp closes. The prior assumes red has been released and should not be carried.

**Exit conditions**

Useful exit spans the overlap with skill 3: after blue is placed at blue_goal near [-0.352,-0.200,0.02], TCP retreats high/open near index 700 and then proceeds to red-buffer acquisition, with red lifted by index 860. The policy's own blue subgoal is satisfied once blue_at_goal and no blue carry are observed; it may continue into red pickup for handoff.

**Failure signatures**

Blue_pose does not follow TCP after closure, blue z drops during carry before reaching blue_goal, red_pose leaves the buffer during blue transport, or blue is released outside the blue_goal tolerance. For mirror prior specifically, failures may appear as y-sign confusion causing motion toward the wrong marked region.

**Overlap role**

This heuristic uses both expanded overlaps: [320,460) learns entry from red-handoff into blue contact, and [700,860) learns exit from blue placement into red reacquisition. It treats the middle blue transfer as a reflected analogue of the first transfer plus a goal-directed place.

**Successor readiness**

Skill 3 requires blue stable at blue_goal, red still at buffer, gripper open and TCP in the demonstrated corridor from high above blue to low/closing/lifting red. Measured support includes high/open at index 700 and red-lift states at index 860; not supported are cases where blue is still in hand or mis-placed.


## Heuristic 2

Goal-anchored blue placement prior: represent and predict blue error to blue_goal and train an auxiliary success predicate, so the policy prioritizes accurate containment over merely replaying the middle trajectory.

**Evidence inspected by the API**

- `demo10100` observations: 550, 600, 650, 700
- `demo10101` observations: 650, 700, 860
- `demo10107` observations: 700

**Interpretation**

In demo10100 the carried blue moves from upper region to z 0.293 at index 550, to near the goal but elevated at index 600, and to [-0.352,-0.201,0.02] with open fingers by index 650-700. Across demo10101 and demo10107 the inspected index 700 states have blue at the lower goal and open gripper. The hard part is final spatial accuracy and release timing; encoding goal error and a containment predicate should focus learning on the success condition. This differs from the symmetry prior because it does not assume mirrored dynamics; it imposes an explicit terminal objective.

**Applicability**

Best when success is sensitive to final blue placement within the goal square and z tolerance. It requires reliable blue_pose and blue_goal observations in the world frame, and assumes the robot can release by opening fingers after lowering to table height over the goal.

**Implications for a future training/inference pipeline**

Use a goal-anchored DP: causal inputs include e_blue=blue_pose.xyz-blue_goal and e_tcp_goal=tcp_pose.xyz-blue_goal, plus full qpos/qvel and red_pose. Add auxiliary heads predicting future min ||blue_pose-blue_goal|| over horizon and a binary blue_at_goal predicate computed from future/current observations using task contract for training labels. Objective: diffusion action loss + lambda1 L2(predicted final blue pose, blue_goal) + lambda2 BCE(blue_at_goal). During deployment the policy conditions on current goal error and may terminate/hand off when the learned predicate is high and gripper state is open. Actions remain pd_joint_pos absolute joint targets and gripper_command.

**Assumptions and limitations**

The demonstrations do not include failed or marginal placements, so the terminal loss may not teach correction after overshoot. If a baseline trained without explicit goal-error features achieves equal containment under perturbed blue starts/goals, this prior is falsified. It also cannot prove that gripper release is necessary for task success, since the contract does not require release, although demos release.

### Handoff interface

**Entry conditions**

Can enter from the predecessor when red is buffered and blue is either untouched at the upper region or already in the early grasp part of the overlap. It uses blue_pose-blue_goal as a primary input, so it expects blue_pose to be observable and not occluded by invalid contact.

**Exit conditions**

Its central exit condition is blue_at_goal: inspected index 650-700 states show blue_pose around x -0.352, y -0.200, z 0.02 and qpos fingers open. For handoff, the policy may also continue to index 860, where red is being lifted; however the goal-anchored mechanism should mark blue placement complete before successor transfer.

**Failure signatures**

Nonzero blue goal error after release, blue z above table when the policy believes placement complete, or gripper opening while tcp z is still high. A late failure is disturbing the placed blue during red handoff.

**Overlap role**

The entry overlap supplies early blue approach and grasp states; the exit overlap supplies states after the placement objective is achieved but before and during red pickup, letting a successor take over without a single fixed release timestep.

**Successor readiness**

The next policy requires blue_at_goal to be true and stable enough that red motion will not collide with it, red_pose at the central buffer, and gripper not carrying blue. The measured state at index 700 satisfies this with TCP high and open; index 860 additionally shows red in hand. Tolerance to a blue block slightly outside the goal is hypothesized only within the task contract half-width, not demonstrated.


## Heuristic 3

Contact-mode temporal prior: decompose the middle skill into inferred approach, grasp, carry, place, retreat, and red-handoff modes, enabling a long-horizon policy to condition actions on manipulation state rather than absolute time.

**Evidence inspected by the API**

- `demo10100` observations: 320, 460, 500, 550, 600, 650, 700, 780, 860
- `demo10102` observations: 320, 460, 700, 860
- `demo10107` observations: 320, 460, 700, 860

**Interpretation**

The middle skill is the longest and includes multiple contact changes. Demo10100 shows entry at 320, blue contact at 460, blue carried high at 500-550, lowered near goal at 600, released by 650-700, then red approach/pick by 780-860. Similar endpoints appear in demo10102 and demo10107, though demo10107 red is slightly lower at 860. A flat policy must infer mode from time and state; a latent temporal organization should improve long-horizon credit assignment. This differs from the goal prior by modeling temporal structure and handoff modes rather than only terminal blue accuracy.

**Applicability**

Transfers when the policy must cover a long temporal interval from late predecessor handoff through blue manipulation and into early successor handoff, including contact/non-contact mode switches. It requires stable observations of object z, TCP z, and finger aperture to infer modes.

**Implications for a future training/inference pipeline**

Train a switching/recurrent DP. Causal encoder h_t=GRU([qpos,qvel,tcp_pose,red_pose,blue_pose,goals]) outputs mode probabilities m_t over {entry_approach_blue, blue_grasp_lift, blue_carry, blue_place_release, retreat, red_handoff}. Mode labels are weakly computed from demonstration events/futures for training: gripper_command sign, blue z>0.05, distance to blue_goal, red z>0.05. The diffusion denoiser is mixture-of-experts sum_m p(m|h) eps_theta_m. Objective combines action denoising, mode cross-entropy, and transition smoothness. Deployment uses only causal h_t and mode probabilities to decode actions and handoff confidence.

**Assumptions and limitations**

Latent modes are inferred from smooth successful sequences and may not align with true contact if a grasp misses. A falsification is to remove the mode/recurrent structure and use a flat chunking DP; if long-horizon handoff and release timing are unchanged, the temporal prior is unnecessary. The data cannot teach branching recovery between modes.

### Handoff interface

**Entry conditions**

Acceptable entries include high/open above the buffer after red release (index 320) through low/closing contact on blue (index 460). The latent-mode policy should infer whether it is still in approach, grasp, carry, place, retreat, or red-handoff from causal qpos, qvel, tcp_pose, and object poses.

**Exit conditions**

The useful exit spans blue placed/open/high through red reacquired/lifted. In measured data index 700 has blue placed and TCP high/open; index 860 has red z about 0.238-0.279 m and fingers near 0.018 m. Either can be a valid exit depending on successor confidence.

**Failure signatures**

Mode posterior inconsistent with observations, e.g. carry mode when qpos fingers are open and blue z is table height, or handoff mode while blue not at goal. Other failures are skipped lift phases, premature lowering, or lingering closed gripper after release.

**Overlap role**

Both overlaps are treated as shared temporal modes, not discarded boundary tails. The entry overlap is approach/grasp mode for blue; the exit overlap is retreat/approach/grasp mode for red. Training both in one policy lets the selector switch under timing uncertainty.

**Successor readiness**

Skill 3 needs the exit mode posterior to be red-handoff or later, blue_at_goal true, and either gripper open/high near red approach or already closing/lifting red as seen at index 860. It remains unsupported to switch while the mode is blue-carry or blue-place.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
