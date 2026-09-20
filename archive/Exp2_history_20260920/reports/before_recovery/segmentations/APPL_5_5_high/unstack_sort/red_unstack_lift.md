# Red unstack grasp and lift

Grasp the top red block from the initial stack, lift it clear of the blue block, and begin carrying it toward the red goal while keeping the blue block stationary.

## Segmentation

Segments [0,190) start from the initial stacked red-on-blue scene with the gripper open and include approach, closure on the red cube, vertical lifting, and the first part of lateral travel. The stop at 190 was chosen because inspected observations show red high above the table and moving toward the red goal while blue remains at the source (e.g. demo10200 and demo10207 index 190). It intentionally overlaps the next skill from 130 onward, where red is already grasped and lifted but placement has not finished.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo10200`: [0, 190)
- `demo10201`: [0, 190)
- `demo10202`: [0, 190)
- `demo10203`: [0, 190)
- `demo10204`: [0, 190)
- `demo10205`: [0, 190)
- `demo10206`: [0, 190)
- `demo10207`: [0, 190)
- `demo10208`: [0, 190)
- `demo10209`: [0, 190)
- `demo10210`: [0, 190)
- `demo10211`: [0, 190)

## Heuristic 1

Red-relative top-grasp prior: learn unstacking in a coordinate frame centered on red_pose, with blue_pose as the support/obstacle, so the policy generalizes over stack xy shifts rather than memorizing absolute joint paths.

**Evidence inspected by the API**

- `demo10200` observations: 0, 80, 100, 130, 190
- `demo10207` observations: 0, 90, 100, 130, 190
- `demo10204` observations: 0, 130, 190

**Interpretation**

Across demonstrations the TCP descends toward the red top with the gripper open, closes near the stacked block, and then red_pose moves with tcp_pose while blue_pose remains on the table. The absolute joint trajectories differ with initial stack xy, but the local relationship between tcp_pose, red_pose, and blue_pose is nearly invariant. Encoding that relation should reduce sample complexity and generalize to small stack translations. This differs from the phase prior below: the representation bias says what coordinates to use, not how to segment time or contacts.

**Applicability**

Transfers when a red cube is initially stacked directly on the blue cube, tcp_pose is above the workspace, top-down parallel-jaw grasping is feasible, and red_pose/blue_pose are available in world coordinates. It depends on the same Panda arm, gripper width range, block size, and gravity/contact behavior because the learned joint targets and gripper closure timing are contact dependent.

**Implications for a future training/inference pipeline**

Candidate policy encodes observations in red-relative coordinates: r_tcp = tcp_pose.xyz - red_pose.xyz, r_blue = blue_pose.xyz - red_pose.xyz, and target grasp offset g=(0,0,approximately 0.00 to 0.02 m above red top/contact). Causal inputs at inference: qpos,qvel,tcp_pose,red_pose,blue_pose,goals. Training labels from demonstrations: joint actions and optionally Cartesian finite-difference delta of tcp_pose. Train a diffusion policy over short horizons of absolute arm_joint_positions plus gripper_command, but condition its denoising network on r_tcp, red/blue relative pose, and finger aperture; optionally decode a Cartesian delta through a learned joint-target head. Loss = action denoising + auxiliary L2 on next r_tcp and red z. At deployment compute red-relative features online and sample joint/gripper commands; no future information is used.

**Assumptions and limitations**

The demonstrations only show successful centered top grasps. The prior may fail for yawed or offset grasps, partial side contacts, or if the top red cube is not directly supported by the blue cube. A falsifying ablation would compare object-relative features against raw joint/state diffusion on held-out initial stack xy; failure would appear as more blue disturbance or missed grasps. The dataset cannot prove recovery after bumping the stack.

### Handoff interface

**Entry conditions**

Measured support: observations like demo10200 index 0 and demo10207 index 0, with qpos finger channels about 0.04 m each, gripper_command open, tcp_pose z about 0.442 m, red_pose z about 0.060 m directly above blue_pose z about 0.020 m. Hypothesized tolerance: initial stack xy within the demonstrated range roughly x -0.339 to -0.361 m and y -0.011 to 0.012 m.

**Exit conditions**

Useful end state for the next policy: red is pinched, qpos finger channels near 0.018 m, red_pose has risen well above the blue cube (e.g. z 0.30 m at demo10200 index 190 and demo10207 index 190), blue_pose remains near table z 0.020 m, and tcp_pose is moving laterally toward red_goal.

**Failure signatures**

No red lift after gripper closure, finger channels remain near 0.04 m despite close command, red_pose z stays near 0.06 m or blue_pose shifts/tilts while red is pulled, or tcp_pose-red_pose offset grows unexpectedly. These indicate transfer to red placement is premature.

**Overlap role**

The [130,190) overlap teaches both this skill and the red placement skill to handle a grasped red cube being raised and starting lateral travel. This prior assumes the successor may enter while vertical lift is still in progress, not only at a fixed high pose.

**Successor readiness**

Successor needs red_pose, red_goal, blue_pose, tcp_pose and qpos fingers showing a secure pinch; it can accept red z above roughly 0.15 m as seen at index 130 or about 0.30 m at index 190, with gripper command still closing/closed.


## Heuristic 2

Event-gated grasp-lift prior: represent unstacking as a small ordered latent state machine whose modes gate arm motion and gripper command, improving contact timing and reducing averaging across approach, close, and lift behaviors.

**Evidence inspected by the API**

- `demo10200` observations: 80, 100, 130, 190
- `demo10207` observations: 90, 100, 130, 190
- `demo10201` observations: 80, 100, 140, 180

**Interpretation**

At inspected states the command and object motion form distinct events: open approach at indices 80/90, close command and finger aperture near 0.018 m at 100, red lifted at 130, and lateral carry by 190. A single stationary policy must learn sharp gripper timing and continuous arm motion simultaneously; a phase latent variable can separate these distributions and make the close/lift transition learnable from few demonstrations.

**Applicability**

Applies when the grasp/lift sequence has the same qualitative event order: open approach, gripper close/contact, red begins rising, then early transport. It requires observable finger aperture in qpos and object pose changes; it is less appropriate for suction, side grasps, or tasks where contact is not visible in the object motion.

**Implications for a future training/inference pipeline**

Train a latent-mode diffusion policy with modes m in {approach_open, close_contact, lift_carry}. Causal inputs: qpos/qvel including finger aperture, tcp_pose, red_pose, blue_pose. Training labels: modes computed from demonstration futures/thresholds, e.g. close when gripper_command<0 and finger qpos decreases, lift when future red_pose.z increases above initial. Objective: denoising loss for action horizon plus cross-entropy for m_t and transition consistency p(m_{t+1}|m_t,o_t). Deployment pseudocode: infer p(m_t|history) with a GRU/transformer; condition diffusion denoiser eps_theta(a_h,o_t,m_t); sample action horizon. The specific prior is temporal event structure rather than coordinate choice.

**Assumptions and limitations**

Event labels are inferred from successful demonstrations, not directly measured contacts. If the gripper closes around empty space, the latent model may still advance unless trained with object-motion checks. Falsify by comparing against a non-phase diffusion policy and by withholding closure timing perturbations; a true benefit should reduce early/late gripper close errors. Dataset cannot establish robustness to failed contact detection.

### Handoff interface

**Entry conditions**

Policy can take over from initial or pre-grasp states with gripper open and red still stacked. It models internal phases and therefore can also start within the demonstrated overlap at a post-close lift state such as demo10200 index 130, where qpos fingers are about 0.018 m and red_pose z is 0.155 m.

**Exit conditions**

Exit when the latent phase is lift/early-transport and red_pose z is high enough that lateral motion has begun (e.g. demo10200 index 190 red x,y shifted toward red_goal, z 0.302 m).

**Failure signatures**

Mode posterior oscillates between close and lift, gripper opens during lift, red_pose does not rise after the close phase, or qvel/finger velocities show continuing closure with no object motion. Such signatures mean the successor may receive an ungrasped object.

**Overlap role**

In [130,190), both policies learn the transition from lift to carry. This phase prior explicitly exposes the successor to varying latent modes so it need not assume a single boundary timestep.

**Successor readiness**

The red placing successor needs the mode estimate or observable equivalent: closed fingers, red_pose moving with tcp_pose, and blue stable. Demonstrated tolerance includes red z from about 0.13-0.16 m at index 130 to about 0.30 m at index 190.


## Heuristic 3

Support-preservation prior: treat blue_pose as a passive support/obstacle and bias the policy to increase red-blue clearance before lateral motion, minimizing unintended blue movement during red unstacking.

**Evidence inspected by the API**

- `demo10202` observations: 0, 130, 190
- `demo10208` observations: 0, 130, 190
- `demo10211` observations: 0, 130, 190

**Interpretation**

In all inspected endpoints, red moves from z 0.06 to high z while blue remains at table height and nearly fixed xy. The most important generalization issue for unstacking is avoiding coupled motion of the bottom cube; explicitly representing blue as an object to preserve should bias the policy toward vertical lift before lateral travel. This differs from the event prior by adding a physical constraint on non-target object motion.

**Applicability**

Applies when the lower blue cube should remain at the source while the red top cube is removed. It relies on accurate blue_pose and the physical need for vertical clearance before lateral transport. It transfers to similar stacked-block unstacking but not to tasks where the support object should move together with the top object.

**Implications for a future training/inference pipeline**

Augment the diffusion objective with blue-preservation and clearance auxiliaries. Inference inputs are causal states including red_pose and blue_pose. Training labels: future blue displacement and red-blue clearance from demonstrations. Mechanism: denoiser conditioned on c = [tcp-red, red-blue, finger_aperture]; loss = L_DDPM(action) + lambda1||blue_pose_{t+k}.xy-blue_pose_t.xy||^2 + lambda2 max(0, h_min-(red_z-blue_z)) during lateral velocity labels. Optionally reject sampled action horizons whose predicted next-state model violates blue stability. Deployment samples actions and selects the lowest predicted blue-disturbance trajectory.

**Assumptions and limitations**

The blue-stability objective is observational, not a measured contact force constraint. It may over-penalize necessary incidental motion in other environments, and cannot recover if the blue cube is already displaced. An ablation should remove blue-motion auxiliary loss/cost; falsification would be no reduction in blue perturbation or grasp failure rates. Successful demos cannot prove safety under aggressive lateral disturbances.

### Handoff interface

**Entry conditions**

Enter with red_pose and blue_pose vertically aligned, blue_pose z near 0.020 m, and no prior blue displacement. The policy assumes the blue cube is a passive support to preserve, as in all index-0 observations.

**Exit conditions**

Exit when red has been lifted clear while blue_pose remains approximately unchanged at the source. Examples: demo10202 index 190 red z 0.302 m while blue z 0.020 m; demo10208 index 190 red z 0.303 m while blue remains table-level.

**Failure signatures**

Blue_pose xy changes by more than a few millimetres or blue orientation tilts during red lift, red/blue z separation stays close to the initial 0.04 m, or lateral tcp motion occurs before red has cleared the blue. These signs predict poor handoff to red placement and later blue pickup.

**Overlap role**

The overlap [130,190) includes the safety-critical moment where vertical clearance becomes sufficient and lateral red transport begins. Both adjacent policies learn to preserve blue while carrying red.

**Successor readiness**

The successor can assume blue remains at the original source and red is clear. Demonstrated readiness includes red-blue z separation larger than 0.14 m by index 130 and about 0.28 m by index 190; hypothesized tolerance is not validated outside this range.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
