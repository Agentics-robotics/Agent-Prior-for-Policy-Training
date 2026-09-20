# red_transport_place_pad

Carry the red block to the marked outside pad, release it there, and retreat toward the blue block.

## Segmentation

Segments start before red is fully lifted (index 455) to share the grasp-to-carry transition, continue through high transport (530), pad placement and opening (around 600-650), and include retreat/travel toward blue until index 760. This groups actions whose subgoal is to move red from carried state to a released, task-satisfying pad state and hand off during the post-release transit.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo1000`: [455, 760)
- `demo1001`: [455, 760)
- `demo1002`: [455, 760)
- `demo1003`: [455, 760)
- `demo1004`: [455, 760)
- `demo1005`: [455, 760)
- `demo1006`: [455, 760)
- `demo1007`: [455, 760)
- `demo1008`: [455, 760)
- `demo1009`: [455, 760)
- `demo1010`: [455, 760)
- `demo1011`: [455, 760)

## Heuristic 1

Pad-frame red placement: control the carried red block relative to the outside pad goal, improving precision of place and release.

**Evidence inspected by the API**

- `demo1000` observations: 455, 530, 650, 760
- `demo1002` observations: 455, 530, 650, 760
- `demo1008` observations: 455, 530, 650, 760

**Interpretation**

Across demos red is carried from variable drawer y to a very consistent pad location. Expressing the motion in red-goal error can focus learning on reducing the object placement error rather than memorizing the exact joint path.

**Applicability**

Transfers when the target outside pad is fixed at world [-0.18,-0.30,0.02], red is top-grasped and carried, and a red center in the pad xy region with z 0.014-0.031 completes the red part of the task. Requires world-frame red_pose and tcp_pose.

**Implications for a future training/inference pipeline**

Use causal inputs qpos,qvel,tcp_pose,red_pose,drawer_position and constant goal g_red=[-0.18,-0.30,0.02]. Encode red-to-goal error and tcp-red offset. Diffusion decoder predicts joint/gripper actions; add auxiliary future-label losses for red_on_pad predicate computed from future red_pose and task contract. Inference computes predicate from current red_pose only.

**Assumptions and limitations**

The pad is fixed in all demos; this prior may overfit to a single world goal and would not transfer to moved pads without explicit goal input. Compare pad-frame representation to absolute replay under artificial pad perturbations; success on current fixed task alone cannot prove generality.

### Handoff interface

**Entry conditions**

Supported entries include index 455/530 while fingers are closed around red (finger qpos about 0.018 m), drawer open, red z rising to ~0.31 m, and tcp co-located with red. The policy can take over before full cruising speed due to overlap.

**Exit conditions**

Red is placed on the outside pad: by index 650 red_pose is near [-0.176,-0.298 to -0.304,0.020] with gripper open near 0.04 m and tcp retreating upward; by index 760 the arm is traveling toward blue.

**Failure signatures**

red final xy outside pad half-size, red z remains high while fingers open, red follows tcp after release, or tcp retreats before red has reached pad height.

**Overlap role**

[455,530) shares grasp-to-carry with predecessor; [650,760) shares release/retreat/global transit with the blue approach skill, supporting successor starts while arm is still leaving the pad.

**Successor readiness**

blue_approach_grasp needs red settled on pad, gripper open, no red attachment, drawer still open, and the arm clear enough to move toward blue.


## Heuristic 2

High-carry then soft-place temporal prior: separate transport clearance from final descent/release to reduce collisions and drops.

**Evidence inspected by the API**

- `demo1001` observations: 530, 650, 760
- `demo1005` observations: 530, 650, 760
- `demo1011` observations: 530, 650, 760

**Interpretation**

The trajectory has a high transport plateau from index 530 then a controlled descent/release near index 650. A height-stage bottleneck addresses the difficulty of not opening early while still making a precise low-z placement.

**Applicability**

Useful when red must be transported over obstacles and then placed softly. Requires sufficient vertical workspace and world z in tcp/red poses.

**Implications for a future training/inference pipeline**

Candidate is a staged latent policy with z_t in {carry_high, descend, open, retreat}; labels are derived from red_pose z, gripper qpos and future red velocity. Inputs are causal state. Decoder conditions on stage and goal error; train with diffusion loss + stage CE + hinge red_z>0.25 during carry_high until near pad_xy. At inference a recurrent stage filter updates from current red_pose/qpos.

**Assumptions and limitations**

No failed drops are shown; soft-placement stability is inferred from success only. Ablate a height-stage bottleneck versus unconstrained DP; falsify if no reduction in low-clearance path errors or premature opening.

### Handoff interface

**Entry conditions**

Enter with red carried high (red z about 0.307 at index 530 or already approaching pad), fingers closed near 0.018 m, and drawer open. It may start in overlap from the initial lift.

**Exit conditions**

Exit once red has descended to z about 0.020 on the pad, fingers have opened to 0.04 m, and tcp is retreating upward and laterally toward blue by index 760.

**Failure signatures**

red travels laterally at low clearance, collides with drawer/table, or release occurs at high z causing drop/roll.

**Overlap role**

The successor blue skill shares the retreat after red release so it learns the arm leaves the placement zone before blue approach rather than assuming a static post-place pose.

**Successor readiness**

Next skill needs open gripper and tcp no longer constrained by red; red pose should be pad-stable with z in [0.014,0.031].


## Heuristic 3

Carry-release object dynamics prior: model red as rigidly attached until gripper opens, then stationary on the pad.

**Evidence inspected by the API**

- `demo1003` observations: 455, 530, 650
- `demo1004` observations: 455, 530, 650
- `demo1009` observations: 455, 530, 650

**Interpretation**

During transport the red block's motion is mostly governed by tcp motion; at release it becomes stationary while tcp retreats. Encoding this switch can improve generalization of carrying across start variations and reduce object slippage.

**Applicability**

Applies when the red block remains rigidly attached during carry and release is detected by opening fingers. Requires reliable red_pose and gripper qpos observations.

**Implications for a future training/inference pipeline**

Train an object-centric dynamics auxiliary: while fingers closed, predict red_pose_{t+1} from tcp motion using a learned rigid offset r=T_tcp^{-1}red; after open, predict red stationary. Inputs causal state, labels future red_pose from demonstrations. Diffusion decoder is conditioned on predicted attachment and r. Deployment rolls one-step object prediction to score/action-sample consistency.

**Assumptions and limitations**

Rigid attachment may not hold for loose grasps or rotations; demos do not contain red slipping. Compare to policy without attachment consistency auxiliary; if object pose prediction is not improved, the prior is unsupported.

### Handoff interface

**Entry conditions**

Supported starts include attached red in the shared lift interval; qpos finger gap about 0.018 m and tcp-red relative pose stable. Drawer_position remains >0.26 m.

**Exit conditions**

Fingers open near 0.04 m and red no longer co-moves with tcp; red_pose is in pad region and tcp has lifted away to z ~0.20-0.26 by index 650 or is transiting by 760.

**Failure signatures**

tcp-red offset changes before intended release, red slips during transport, or red continues to move with tcp after gripper opens.

**Overlap role**

Both predecessor and successor share object-attachment transitions: first from grasp to carry, then from release to free-retreat.

**Successor readiness**

Successor can start when attachment latent for red is false, gripper is open, and tcp path toward blue is unobstructed by red.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
