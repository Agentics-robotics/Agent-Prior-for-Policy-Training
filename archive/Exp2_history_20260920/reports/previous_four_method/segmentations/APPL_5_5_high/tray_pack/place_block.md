# Place selected block in tray and release/retreat

With the active block already grasped or entering the goal funnel, align it to its tray target, lower/support it, open the gripper, and retreat/transition while preserving the other block.

## Segmentation

This dataset groups red and blue placements because the shared problem is carrying a held active block into its goal region, lowering to tray height, opening the gripper, and retreating without disturbing the other block. Red placement starts at action 185 while the red is still high and moving, not at a fixed pre-place pose, and continues to 420 so it includes release, upward retreat, and the early transition toward the blue object. Blue placement starts at 580 while blue is still high/in transit and continues to each trajectory's final action, retaining release and final settling/retreat context needed by the task completion contract. The red tail overlaps with blue acquisition so the successor sees broad open-gripper retreat states.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo10300`: [185, 420)
- `demo10300`: [580, 742)
- `demo10301`: [185, 420)
- `demo10301`: [580, 742)
- `demo10302`: [185, 420)
- `demo10302`: [580, 752)
- `demo10303`: [185, 420)
- `demo10303`: [580, 743)
- `demo10304`: [185, 420)
- `demo10304`: [580, 737)
- `demo10305`: [185, 420)
- `demo10305`: [580, 752)
- `demo10306`: [185, 420)
- `demo10306`: [580, 749)
- `demo10307`: [185, 420)
- `demo10307`: [580, 730)
- `demo10308`: [185, 420)
- `demo10308`: [580, 737)
- `demo10309`: [185, 420)
- `demo10309`: [580, 741)
- `demo10310`: [185, 420)
- `demo10310`: [580, 730)
- `demo10311`: [185, 420)
- `demo10311`: [580, 742)

## Heuristic 1

Active-goal placement funnel: express placement as reducing the active block's world-frame error to its goal, sharing one precision controller across red and blue. This should improve final XY/Z accuracy and transfer to modest target variations because the policy sees residual goal error directly.

**Evidence inspected by the API**

- `demo10300` observations: 185, 250, 300, 330, 420, 580, 625, 650, 700, 742
- `demo10301` observations: 185, 250, 300, 330, 420, 580, 625, 650, 700, 742
- `demo10302` observations: 185, 250, 330, 420, 580, 625, 752
- `demo10310` observations: 185, 250, 330, 420, 580, 625, 730

**Interpretation**

Both objects are placed by aligning the grasped block to its color goal and descending. In demo10300, the red is high near the goal at 250, settled at z≈0.036 by 300, and the tcp has retreated by 330/420. The blue is high near its goal at 625, low/settling at 650, released/open by 700 and final. The same pattern appears in demo10301 and across final endpoints. The hard learning problem is precise placement after a long carry; goal-frame features make the final action depend on residual error rather than absolute color/location. This differs from the release-phase prior below, which focuses on support/open timing, and from the noninterference prior, which focuses on the inactive object/tray context.

**Applicability**

Transfers to top-down placement of the same cuboid blocks into fixed or observed goal regions inside the same tray geometry, with active block pose and goal available in world coordinates. It assumes the object is already grasped or very near the goal funnel at entry and that the controller can command absolute joint targets with similar compliance. It should cover both red and blue placement if a role selector provides the currently held active object.

**Implications for a future training/inference pipeline**

Use active role r and build features e_t=[p_obj^r-p_goal^r, p_tcp-p_obj^r, object orientation yaw/quat, finger_width, p_inactive-p_goal_inactive, qpos,qvel]. A goal-funnel diffusion policy predicts joint targets and gripper command with an auxiliary head for next active object pose relative to goal. Training labels include active role from segment identity and release timing from gripper/finger observations; goal-relative errors are computed from current observations, not futures. Objective L=L_action+lambda ||(p_obj_{t+1}-p_goal)-hat_e_{t+1}||^2+lambda_goal terminal-funnel weighting for samples near release. At deployment, all features are causal; the policy samples actions conditioned on current relative error and opens only when the learned decoder probability of support/release is high. The specific prior is translation into the active goal frame and heavier loss near goal-funnel convergence.

**Assumptions and limitations**

Goal-relative placement assumes a good grasp and does not learn recovery from dropping a block outside the tray. If the role selector misidentifies the active block, the same relative controller will place the wrong object. Falsify by comparing to absolute-state and color-specific baselines under shifted goals/starts; if relative errors do not improve or if role-conditioned sharing hurts final precision, this prior is unsupported. Successful data cannot establish tolerance to clutter or tray pose changes beyond the fixed fixture.

### Handoff interface

**Entry conditions**

Measured entries: red place begins at observation 185 with the red block already grasped/elevated (z≈0.28 m in most demos, z≈0.286 for demo10304, z≈0.281 for demo10311) and fingers closed ≈0.018 m. Blue place begins at 580 with the blue block grasped and elevated (z≈0.30 m in many demos) while moving toward the blue goal. Hypothesized tolerance includes entries as low as observed at the acquire/place overlap endpoint (blue z≈0.13 m in demos 10307/10310 at 625).

**Exit conditions**

Exit is a block resting within its goal region in the tray with pose z≈0.036 m and the gripper open/retreating. Red exits are visible by index 330 while the tcp retreats above the red goal; blue exits by the final observation with both red and blue at goal and tcp z≈0.316 m, fingers open ≈0.04 m. A full release is not required by the task contract, but the demonstrations release and retreat.

**Failure signatures**

Active object remains outside the goal XY tolerance after opening, object z stays high after intended placement, gripper opens while the object is not supported, tcp pushes the object out of the tray, or the inactive object moves noticeably. Premature acquisition-to-place handoff appears as open fingers or a non-held object at entry.

**Overlap role**

The goal-relative policy is trained on the early carry-in overlap [185,250) and [580,625), so it can take over while the predecessor is still transporting the block. Its red tail [330,420) overlaps acquire_blue, teaching it to finish release/retreat while the next policy begins approach.

**Successor readiness**

For red placement, acquire_block(blue) is ready when red_pose is near red_goal with z≈0.036 m, fingers are open, and tcp has retreated enough to avoid dragging the red block (examples index 330-420). For final blue placement there is no successor other than task completion; readiness is both goal predicates true.


## Heuristic 2

Support-gated release and retreat: treat placement as carry -> descend -> supported-settle -> open -> retreat, with gripper opening only after the object is supported near goal height. This should improve release timing and reduce dragging/lifting the placed block.

**Evidence inspected by the API**

- `demo10300` observations: 250, 300, 330, 650, 700, 742
- `demo10301` observations: 250, 300, 330, 650, 700, 742
- `demo10307` observations: 580, 625, 730
- `demo10310` observations: 580, 625, 730

**Interpretation**

The demonstrations show a consistent hold-lower-open-retreat sequence. In demo10300, red is still held high at 250; by 300 red is at z≈0.036 with qpos fingers opening to ≈0.04, and by 330 the tcp has retreated. Blue remains held at 625, is low at 650 with fingers still closed, then is open/retreating by 700 and final. Demo10307/10310 show faster descent by 625 but still follow the same release logic. This addresses a diffusion policy's tendency to average across closed descent and open retreat; it differs from the goal-funnel prior by explicitly gating release on support state.

**Applicability**

Applies when the block can be quasi-statically lowered into the tray and released by opening the same mimic gripper after support contact. It assumes the tray floor height and object height are similar to the demonstrations (goal z≈0.036 m) and that support can be inferred from object z/velocity and finger width; no force sensing is available.

**Implications for a future training/inference pipeline**

Train a phase-conditioned placement model with latent z_t in {carry_to_funnel, descend, supported_settle, open_release, retreat}. Infer labels from current/future demo observations: supported when active_pose.z is within goal z tolerance (~0.036±0.011) and gripper/tcp vertical velocity is small; release when gripper_command switches to +1 and finger qpos increases; retreat when tcp.z rises while object remains fixed. Use DP_theta(a|o,z) with a gripper decoder gated by z: gripper_cmd=-1 except in open_release/retreat. Add auxiliary support probability and next object height predictions. Inference uses only past/current object pose, tcp_pose and finger state to update z; future labels are training-only. This prior changes the temporal computation and gripper gating, not just input coordinates.

**Assumptions and limitations**

Support/contact is inferred from kinematics; no force/torque labels confirm when the tray supports the block. Expected failure is gripper opening at the wrong height or vertical retreat dragging the block. Falsify by ablation removing the latent release/support classifier while keeping goal-relative features; if release timing is unchanged, this prior is unnecessary. Dataset cannot establish behavior with tilted blocks, soft contacts, or disturbances during release.

### Handoff interface

**Entry conditions**

Requires a held active block in or near the goal funnel, fingers closed around ≈0.018 m, and a descent path into the tray. Red entries at 185/250 have the object high and moving; blue entries at 580/625 range from high carry to early descent. The prior assumes enough grasp stability for a controlled descent, which is observed but not stress-tested.

**Exit conditions**

Exit after the phase posterior reaches released/retreat: active object z≈0.036 m, finger qpos≈0.04 m or gripper command +1, tcp has risen to a safe clearance (e.g. final tcp z≈0.316 m; red retreat at 330 often z≈0.21-0.30 m). Object velocity thresholds are not part of the task contract, but observed final qvel is near zero.

**Failure signatures**

Opening before object z approaches tray height, closing during retreat, object following the gripper upward after open, or repeated vertical poking. Premature successor handoff is detectable if acquire_blue starts before gripper is open and red object stable.

**Overlap role**

The latent phases deliberately span adjacent skills: acquisition and placement both learn the high-carry phase before descent, while red place and blue acquisition both learn open-gripper retreat. This avoids a brittle cut where release and retreat belong to only one policy.

**Successor readiness**

For blue acquisition, successor readiness is released/open plus tcp moving upward/away from red: examples include red at z≈0.036 by 300-330 and open fingers by 330; acquire_blue can take over anywhere through 330-420. For task end, readiness is simply both objects at their goals with no need for further gripper action.


## Heuristic 3

Tray packing noninterference: condition placement on both objects and tray geometry, optimizing active-goal accuracy while preserving inactive object pose and containment. This should improve robustness when placing the second block and during retreat/transition because task success is conjunctive.

**Evidence inspected by the API**

- `demo10300` observations: 185, 250, 300, 330, 420
- `demo10300` observations: 580, 625, 650, 700, 742
- `demo10305` observations: 185, 250, 330, 420, 580, 625, 752
- `demo10306` observations: 330, 420, 580, 625, 749

**Interpretation**

Throughout red placement the blue block stays at its table pose, and after red is placed it remains fixed while the arm retreats and handles blue. Demo10300 shows red at goal by 330 and unchanged through 742 while blue is acquired and placed; demo10305/10306 show the same invariance at indices 330, 420, 580, 625 and final. The final task success requires both goals simultaneously, so a policy that only optimizes the active block may inadvertently disturb the other block when generalized. This prior adds explicit noninterference and tray-containment structure, unlike the previous two which focus on the active object alone.

**Applicability**

Applies to packing multiple objects into the same tray where the inactive object should remain undisturbed and tray walls define containment. It depends on reliable poses for both objects and fixed tray geometry from the task contract. It is most relevant to the second placement and the red-to-blue transition tail; it may be less critical for single-object placement but should regularize both.

**Implications for a future training/inference pipeline**

Augment place policy inputs with inactive object pose, both goal boxes, and analytic tray-wall signed distances from the task contract. Add auxiliary losses: inactive-pose constancy prediction ||hat_p_inactive(t+1)-p_inactive(t)||, active containment margin, and optional differentiable keep-out cost for tcp/object paths near inactive block and tray walls. A diffusion policy can be guided at sampling by score = log pi_theta - alpha collision_cost(path, tray, inactive) - beta goal_error. All quantities are causal except future path/object deltas used as training targets. The action output remains the action contract. This candidate differs by encoding a multi-object/tray scene prior, not only role-relative placement or release timing.

**Assumptions and limitations**

No demonstrations include near-collisions or displaced inactive blocks, so obstacle costs are extrapolations from invariance rather than corrective behavior. Expected failure is overconservative motion or ignoring a displaced inactive block because the training set never reacts. Falsify with an ablation that masks inactive object/tray features; if masking does not increase disturbance/collision under goal shifts, the noninterference prior is not useful. Dataset cannot establish recovery if a placed block is bumped.

### Handoff interface

**Entry conditions**

Can take over when one block is held and the other block has either not yet been placed (red placement: blue still on table at z≈0.020 m) or is already placed (blue placement: red at red_goal z≈0.036 m from index 330 onward). It assumes the inactive object's pose is observable and should be treated as a keep-out/non-disturb region.

**Exit conditions**

Exit when the active block is in its target region and the inactive block has not moved outside its current/goal tolerance. For red placement, blue should remain on the table until acquire_blue begins; for blue placement, red should remain at red_goal while blue is placed and the tcp retreats.

**Failure signatures**

Inactive block pose changes during active placement, tcp path crosses too close to the placed block/tray wall, active object ends outside full rotated XY containment, or the policy pushes a block while retreating. A premature handoff is indicated if acquire_blue starts after red placement with red displaced or gripper not open.

**Overlap role**

The large [330,420) overlap is particularly important for this prior: it teaches that red must remain undisturbed while the arm retreats and heads to blue. The [580,625) overlap teaches entry to blue placement with red already occupying the tray, so both acquisition and placement learn the two-object context.

**Successor readiness**

acquire_block(blue) needs red stable in its goal, gripper open, and a collision-free path from the retreat pose to the blue block. The final completion consumer needs both predicates red_at_goal and blue_at_goal true for at least the required single observation; no sustained hold is required.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
