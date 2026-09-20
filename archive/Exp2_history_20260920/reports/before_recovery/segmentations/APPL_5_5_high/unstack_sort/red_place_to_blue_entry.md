# Red place, release, and transition to blue lift entry

Carry the grasped red block to the red pad, release and retreat from it, then switch to the blue block and reach a blue-lift entry state.

## Segmentation

Segments [130,520) begin while red is already grasped and being lifted, continue through transport to the red pad, descent, release, retreat, approach to blue, blue grasp closure, and the early blue lift. The start overlaps Skill 1 before red transport has settled; the stop at 520 is after inspected demonstrations show blue lifted high from the source, giving the following blue policies a supported takeover state. This grouping is intentionally broader than pure red placement to learn the red-to-blue handoff.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo10200`: [130, 520)
- `demo10201`: [130, 520)
- `demo10202`: [130, 520)
- `demo10203`: [130, 520)
- `demo10204`: [130, 520)
- `demo10205`: [130, 520)
- `demo10206`: [130, 520)
- `demo10207`: [130, 520)
- `demo10208`: [130, 520)
- `demo10209`: [130, 520)
- `demo10210`: [130, 520)
- `demo10211`: [130, 520)

## Heuristic 1

Red goal-relative carry/place prior: while red is grasped, actions should depend primarily on red_pose-to-red_goal error and the maintained tcp-red grasp offset, improving spatial generalization of placement.

**Evidence inspected by the API**

- `demo10200` observations: 130, 190, 260, 280, 330
- `demo10207` observations: 130, 190, 260, 280, 330
- `demo10204` observations: 130, 190, 260, 330

**Interpretation**

The data show a smooth transition from red high in hand to red at the marked red pad: demo10200 red moves from index 190 high and off-goal to index 260/280 at table height near red_goal, then stays fixed by 330. Encoding the red_goal error directly addresses the learning difficulty of mapping varied initial stack poses to a single precise goal. This prior is about goal-directed object servoing, not release timing or object switching.

**Applicability**

Transfers to the red-object phase of the same task when red is securely grasped and its goal is fixed at red_goal in world coordinates. It relies on red_pose being observable while grasped and on a top-down grasp that keeps red near a stable tcp offset. It should not be assumed for non-grasped pushing or when red_goal changes outside the reachable tabletop region.

**Implications for a future training/inference pipeline**

Use causal inputs qpos,qvel,tcp_pose,red_pose,blue_pose,red_goal,blue_goal. Represent red placement by e_goal = red_goal - red_pose.xyz and grasp offset e_tcp = tcp_pose.xyz - red_pose.xyz. Candidate DP denoiser conditions on [e_goal,e_tcp,blue_pose,phase scalar] and predicts absolute joint targets and gripper. Auxiliary training labels from future demonstrations: time-to-red-at-goal and future red_goal error. Objective: action denoising + beta||red_pose_{t+H}.xyz-red_goal||_Huber for predicted latent rollout + binary red_at_goal classifier. Deployment samples actions using current e_goal only, without future labels.

**Assumptions and limitations**

The red-goal servo is learned only for one red_goal and one approach direction; it may overfit to absolute goal position. It cannot prove that release is unnecessary because demonstrations do release even though the task contract does not require release. Falsify by comparing against a non-goal-conditioned raw policy on perturbed initial stack positions; no improvement in red placement accuracy would refute the benefit.

### Handoff interface

**Entry conditions**

Can enter during the Skill 1/2 overlap with red already in hand: qpos fingers around 0.018 m, gripper command closed, red_pose z above the stack (demo10200 index 130 z 0.155 m) or high carry (index 190 z 0.302 m), blue_pose still at table z about 0.020 m.

**Exit conditions**

For this prior the useful end state is red placed within the red_goal basin and the robot on its way to blue acquisition. At demo10200 index 280 red_pose is at about [-0.241,-0.248,0.020] with fingers open; by index 330 tcp_pose is high above the red goal and ready to retreat/switch.

**Failure signatures**

Red_pose does not converge to red_goal before opening, red tilts or follows the gripper after release, blue is displaced during red carry, or red remains high when the policy switches attention to blue.

**Overlap role**

The [130,190) overlap with the predecessor lets this policy learn from early red lift/carry states; the [330,520) overlap with the successor lets it continue through retreat and blue approach rather than ending at red release.

**Successor readiness**

The blue acquisition successor needs red_at_goal supported by red_pose within the pad tolerance, gripper open, tcp_pose clear enough to move to blue, and blue_pose still at source. Demonstrated support includes red settled by 280 and tcp high by 330, with further approach to blue by 460-520.


## Heuristic 2

Release-reset prior: after red reaches the pad, the policy should explicitly switch from closed carry to open retreat, using stable red placement and open fingers as the state that enables the next object pickup.

**Evidence inspected by the API**

- `demo10200` observations: 260, 280, 300, 330, 420, 520
- `demo10207` observations: 260, 280, 330, 460, 520
- `demo10201` observations: 260, 300, 330, 420, 520

**Interpretation**

The inspected trajectory does not simply stop after placing red; it opens at the table, then retreats upward and across before blue pickup. Without an explicit reset prior, a policy may average closed-carry and open-retreat commands around the release boundary. Learning release as a manipulation-state transition should improve handoff reliability to the next object. This differs from the goal prior by focusing on gripper/object decoupling and availability for the next grasp.

**Applicability**

Applies when the robot must release red and leave it stable before approaching blue. It depends on gripper command/finger position observations and on tabletop support capturing the red cube after descent. It transfers to tasks where a released object should remain at a goal while the arm retreats, but not to tasks requiring continuous in-hand manipulation after placement.

**Implications for a future training/inference pipeline**

Train a policy with a release-reset latent variable r_t in {carry, descend_contact, open_release, retreat_switch}. Causal inputs: qpos fingers, tcp_pose, red_pose, blue_pose, goals. Training labels: r_t computed from gripper_command sign, finger aperture, and red_pose near red_goal; future labels mark red stationary after opening. Action decoder: diffusion predicts joint/gripper horizons; a gripper head is regularized with BCE to command open only in open_release/retreat. Additional loss predicts red_pose_{t+k} after release to encourage no-drag. Deployment infers r_t from current/history and gates gripper open/arm retreat behavior.

**Assumptions and limitations**

The reset/release event is inferred from gripper and object motion; contacts or object settling forces are not measured. If the object is slightly wedged or the table friction differs, release may not leave red stable. An ablation should remove explicit release/reset labels and compare frequency of dragging red toward blue. The successful dataset cannot establish recovery if red is dropped outside the pad.

### Handoff interface

**Entry conditions**

Enter after or during red descent with red still grasped, such as demo10200 index 260 red at z 0.020 near red_goal with qpos fingers still near 0.018 m, or earlier high-carry states in the overlap. The policy assumes blue remains ungrasped at source.

**Exit conditions**

Exit when red is released and stationary at goal, gripper is open (finger qpos about 0.04 m by demo10200 index 280/300), and tcp_pose is either high above red or moving toward blue. It may continue until blue is grasped/lifted by index 520 in the expanded overlap.

**Failure signatures**

Opening before red z is at table, red moves with the gripper after release, tcp retreats horizontally while still closed on red, or fingers fail to reopen before blue approach. Any of these gives the successor a bad object-switch state.

**Overlap role**

The [330,520) overlap is deliberately large: both this skill and blue acquisition see the post-release retreat, open-finger approach to blue, blue closing, and blue lift. This prior treats that overlap as a learned reset process rather than an instantaneous cut.

**Successor readiness**

Successor requires red_pose at red_goal and gripper available for blue. Demonstrated states include red fixed at z 0.020 from 330 onward while tcp travels to blue and fingers remain open until the close near blue; hypothesized tolerance to off-goal red is not demonstrated.


## Heuristic 3

Sequential object-attention prior: encode a learned red-to-blue target switch triggered by red_at_goal and gripper release, reducing wrong-object ambiguity during the long transition to blue pickup.

**Evidence inspected by the API**

- `demo10200` observations: 330, 420, 460, 520
- `demo10207` observations: 330, 420, 460, 500, 520
- `demo10201` observations: 330, 420, 460, 520

**Interpretation**

At index 330 red is already fixed at the red pad while the robot has not yet completed blue acquisition; by 420-520 tcp and blue_pose dominate the motion. A policy that must infer which object matters from raw concatenated poses risks confusion because both goals remain present. An explicit attention switch uses the observed red_at_goal predicate to decide when blue becomes the control target. This differs from release-reset by changing the policy representation of object relevance.

**Applicability**

Applies to ordered two-object sorting where red must be finished before blue can be manipulated. It assumes object identities are known from red_pose and blue_pose fields and that the task order red-then-blue is valid. It is not a general unordered planner unless trained with additional permutations.

**Implications for a future training/inference pipeline**

Build a target-conditioned or mixture-of-experts diffusion policy with attention logits alpha_red, alpha_blue. Inputs at inference: object poses/goals and qpos/qvel; no future labels. Training labels for alpha come from demonstration futures: target=red until red_at_goal/open release, then transition target=blue by approach/blue lift. Compute object-centric embeddings phi_i=[tcp-object_i, object_i-goal_i, finger, is_at_goal_i]. Denoiser condition = sum_i alpha_i phi_i plus global qpos. Loss = action denoising + CE(alpha,label) + consistency encouraging alpha_blue after red_at_goal. Deployment infers alpha online and samples actions from the selected/weighted object context.

**Assumptions and limitations**

The demonstrations always use the same task order, so the attention switch might encode chronology rather than a learned predicate. It may fail if blue should be done first or if red is already at goal initially. Falsify with an ablation that removes explicit object-attention/target token; benefit should show as fewer regrasp-red or wrong-object actions. The dataset cannot prove autonomous task planning.

### Handoff interface

**Entry conditions**

The policy can enter with red in hand during early overlap or after red placement. For the attention switch itself, measured support is red_pose fixed at red_goal and qpos fingers open around 0.04 m near index 330; blue remains at source and becomes the new attended object during indices 420-520.

**Exit conditions**

Exit when attention has switched to blue and a secure blue grasp/lift entry exists: blue_pose z has risen substantially, e.g. demo10200 index 520 blue z 0.280 m or demo10207 index 520 blue z 0.278 m, with fingers closed around 0.018 m.

**Failure signatures**

Attention switches before red_at_goal, robot returns toward red after release instead of blue, blue approach occurs with fingers still closed on red, or red_pose changes while blue is being lifted.

**Overlap role**

The [330,520) overlap is the full object-switch corridor. Both policies learn states with red released, robot retreating, then blue closing/lifting, so successor takeover can occur before or after actual blue contact.

**Successor readiness**

The next policy needs a target-object flag or attention state selecting blue, red fixed at goal, tcp near or above blue, and gripper either open for approach or closed/lifting blue. Demonstrated range spans high retreat at 330, low blue contact at 460, and lift at 520.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
