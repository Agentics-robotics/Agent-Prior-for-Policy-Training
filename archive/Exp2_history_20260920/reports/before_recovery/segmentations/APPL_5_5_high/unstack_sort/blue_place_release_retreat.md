# Blue final place, release, and retreat

Place the blue block on its goal pad, release it, and retreat while keeping both red and blue at their goals.

## Segmentation

Segments [520,end) begin while blue is already lifted high, before exact final placement, and continue through transport/descent, table contact, gripper opening, and terminal retreat/settling. Starting at 520 creates a large overlap with blue acquisition/transport so this policy can take over from high source-carry or goal-near carry states rather than a single final descent pose. The final observations are retained because the task completion contract needs both objects at goals and the demonstrations include useful release/retreat context.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo10200`: [520, 732)
- `demo10201`: [520, 717)
- `demo10202`: [520, 729)
- `demo10203`: [520, 728)
- `demo10204`: [520, 721)
- `demo10205`: [520, 723)
- `demo10206`: [520, 729)
- `demo10207`: [520, 739)
- `demo10208`: [520, 737)
- `demo10209`: [520, 723)
- `demo10210`: [520, 730)
- `demo10211`: [520, 721)

## Heuristic 1

Blue goal-basin servo prior: near final placement, represent actions around blue_pose-to-blue_goal error and train an auxiliary final-success critic to improve precise pad containment.

**Evidence inspected by the API**

- `demo10200` observations: 520, 590, 620, 640, 660, 700, 732
- `demo10207` observations: 520, 590, 640, 660, 700, 739
- `demo10210` observations: 520, 590, 730

**Interpretation**

Inspected final phases show blue moving from high carry near the goal through descent to table-level containment. The main learning challenge is precision: the pad tolerance is small relative to long arm joint motions. A goal-basin representation and success critic directly bias final corrections toward blue_at_goal rather than replaying time indices. This differs from release timing and retreat priors by focusing on placement accuracy before/while contact is made.

**Applicability**

Applies when blue is securely grasped or recently lifted and must be placed on the fixed blue_goal pad. It relies on blue_pose/blue_goal observability and similar table/contact geometry. It transfers to small variations in approach height but not to cases where the grasp is slipping or blue is not in hand at entry.

**Implications for a future training/inference pipeline**

Construct a local goal-basin DP with features e_xy=blue_goal.xy-blue_pose.xy, e_z=blue_goal.z-blue_pose.z, tcp-blue, finger aperture, red_at_goal flag. Causal inputs: current fields only. Training labels: action horizons and final blue_at_goal predicate computed from future observations. Mechanism: denoiser predicts joint/gripper actions; an auxiliary critic V(o)=probability blue_at_goal within H is trained from demonstration futures and used to rank sampled action horizons. Objective = action denoising + gamma BCE(V, future_success) + Huber on predicted terminal blue-goal error. Deployment samples several horizons and executes the first action from the highest V/lowest denoising-cost sample.

**Assumptions and limitations**

This local servo is demonstrated only for one blue_goal and several similar approach paths. It may fail if entering with large xy error not seen at 520 or if the grasp is loose. Falsify by comparing against raw action diffusion and measuring blue_at_goal accuracy and final nudging. The dataset cannot prove general pad placement under unseen goal locations.

### Handoff interface

**Entry conditions**

Can enter during overlap from blue high carry: demo10200 index 520 blue z 0.280 over source, index 590 near goal high at z 0.295; demo10207 index 520 source-high and 590 goal-high. Gripper is closed with qpos fingers near 0.018 m and red is already at goal.

**Exit conditions**

Exit/final state is blue within the blue_goal pad at table height, red still at red_goal, and robot retreated or at least not disturbing objects. Examples: demo10200 final index 732 blue at [-0.241,0.248,0.020], red at [-0.241,-0.248,0.020], fingers open.

**Failure signatures**

Blue xy error grows during descent, blue contacts table outside the pad, gripper opens while blue z is still high, or tcp pushes blue after release. These signal local goal-servo failure.

**Overlap role**

The [520,590) overlap with the predecessor lets this final policy start before blue has reached the exact goal or descent stage. It learns to continue from high source carry, high goal carry, or early descent states.

**Successor readiness**

No successor manipulation is required; for a hypothetical monitor, readiness is red_at_goal AND blue_at_goal in world frame with object z within tolerance. The task contract does not require gripper release, but demonstrations support release and retreat.


## Heuristic 2

Supported-release prior: gate gripper opening on blue being both aligned with blue_goal and supported by the table, reducing premature drops or post-contact dragging.

**Evidence inspected by the API**

- `demo10200` observations: 590, 620, 640, 660, 680
- `demo10207` observations: 590, 640, 660, 700
- `demo10201` observations: 590, 640, 660

**Interpretation**

The demonstrations show blue kept closed during high transport/descent and gripper opening only when blue is near table height at the goal, e.g. transition around 640-660. Without a release prior, diffusion may average open/closed actions near the boundary or drop early from high states included in the overlap. This prior is a manipulation-contact timing bias distinct from goal-servo accuracy.

**Applicability**

Applies to placing a grasped block on a table where opening should occur only after the block is supported and aligned. It depends on gripper command/finger observations and table contact dynamics. It does not transfer to tasks where release in midair or dropping is acceptable.

**Implications for a future training/inference pipeline**

Train a release-gated DP with event variable u_t in {carry_closed, descend_closed, release_open, retreat_open}. Causal inputs: blue_pose, blue_goal, tcp_pose, qpos fingers, red_pose. Training labels from demonstrations/futures: u_t based on gripper_command, finger aperture, and blue_pose.z/xy relative to goal; these labels are not available at inference but inferred by a small recurrent classifier. Denoiser conditions on u_t and predicts absolute joint targets and gripper_command. Add loss penalizing sampled open command when inferred P(supported/aligned) is low: support predictor S(o_t) trained from future stable-at-goal labels. Deployment keeps gripper closed until S is high.

**Assumptions and limitations**

Release timing is not supervised by contact force, only by pose/finger trajectories; the object might be supported slightly earlier or later under different friction. A falsifying comparison would remove the release-event classifier; if premature drops do not increase, the prior may be unnecessary. Successful data cannot establish how to recover from a bounce or a stuck object.

### Handoff interface

**Entry conditions**

Enter with blue in gripper, usually high or descending: at index 590 blue is near goal but above table in many trajectories; by index 640 demo10200 blue is near table z 0.023 and still closed; at index 660 gripper opens as blue is at z 0.020.

**Exit conditions**

Exit when fingers have opened and blue remains at blue_goal table height. Demonstrated examples include demo10200 index 660 qpos fingers approximately 0.040 m with blue z 0.020, followed by retreat at 680.

**Failure signatures**

Open command before blue z approaches 0.02 m and xy is within the pad, continued closed grasp after object is table-supported causing dragging, or asymmetric finger motion indicating object wedged in the gripper.

**Overlap role**

The prior uses the C/D overlap to learn that high-carry states should keep gripper closed, while later table contact states should open. This prevents the predecessor's transport state from being confused with release.

**Successor readiness**

Since this is final, successor readiness means the monitor can evaluate success after release or while stable. Demonstrations support open-finger final states; tolerance for no-release success is allowed by task contract but not strongly represented in the demonstrations.


## Heuristic 3

Post-release verification and retreat prior: after blue placement, continue learning open-finger retreat and object-stability prediction so the final policy avoids disturbing already successful object poses.

**Evidence inspected by the API**

- `demo10200` observations: 660, 680, 700, 720, 732
- `demo10207` observations: 660, 700, 739
- `demo10201` observations: 660, 700, 717
- `demo10210` observations: 730

**Interpretation**

Final observations show the expert keeps the gripper open and retreats upward after blue is placed, while both objects stay at their goals. Including this behavior in the skill prevents the learned policy from ending with the gripper still near the block, which could bump the object on controller settling. This is different from release timing: it optimizes post-release non-disturbance and terminal confidence.

**Applicability**

Applies to final task completion when both blocks are at their goals and the robot should stop influencing them. It depends on the demonstration convention of retreating upward with open fingers after placement. It transfers to tasks where final object stability matters, even though this task contract does not require tcp clearance or sustained hold.

**Implications for a future training/inference pipeline**

Add terminal-state and no-disturbance auxiliary heads to a final DP. Inputs: all object poses/goals, tcp_pose, qpos/qvel. Training labels: future task_success, future object displacement after release, and terminal/retreat phase from demonstration futures. Objective: action denoising + BCE(success_{t+H}) + lambda||red/blue_pose_{t+H}-red/blue_pose_t|| for post-release states + optional imitation of open gripper. Deployment computes success probability; if high and object-disturbance risk low, the policy can output retreat/open or a hold action according to sampled horizon. This prior specifically changes terminal objective and inference selection.

**Assumptions and limitations**

The task contract has no velocity threshold, sustained hold, or tcp clearance requirement, so retreat/stability is a demonstration convention rather than necessary for success. A policy that stops immediately after blue_at_goal may still satisfy the contract. Falsify by comparing with a no-retreat terminal policy; if object disturbance is not reduced, the prior adds unnecessary horizon. The data cannot prove robustness to delayed settling or collisions after final pose.

### Handoff interface

**Entry conditions**

Enter after release or near-release with blue at table height and fingers opening/open, as at demo10200 index 660. It can also cover retreat states through final observations where both objects are already at goals.

**Exit conditions**

Exit is the materialized task-complete state: red_pose at red_goal, blue_pose at blue_goal, both z about 0.020 m, gripper open qpos about 0.04 m, and tcp_pose retreated high around z 0.30 m in final frames.

**Failure signatures**

Object pose changes during retreat, tcp remains low enough to bump the blue cube, fingers close again after release, or final blue/red velocities/poses indicate sliding. These signs mean stopping/monitoring should not declare stable completion.

**Overlap role**

Although this is the terminal skill, retaining actions through final settling teaches post-release retreat instead of truncating at first contact. It shares early release/descent states with Skill 3 via [520,590) and then continues final verification context.

**Successor readiness**

No successor policy is expected. A completion monitor can use the task contract predicates red_at_goal and blue_at_goal for at least one observation; demonstrations additionally show open gripper and tcp clearance, which are useful but not contract-required.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
