# Acquire selected block and carry toward its tray goal

Move from an open-gripper start or transition state to a stable grasp of the active table block, lift it, and carry it into the handoff funnel toward that block's tray goal.

## Segmentation

This dataset groups both red-first and blue-second acquisitions because the core learning problem is top-down approach, gripper closure, lift, and carry of the active block. The red occurrences start at action 0 from the common home/open configuration; by observation 250 the red block is grasped and elevated/near the red goal across demonstrations. The blue occurrences start at action 330, intentionally before the red-place retreat has fully transitioned to a blue pregrasp pose, so the skill learns to take over from a broad handoff state with the gripper open and red already settled. They stop at observation 625, when the blue is grasped and elevated or beginning descent toward its goal. These stops are not exact subgoal predicates; they are expanded transition windows chosen from observed state changes.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo10300`: [0, 250)
- `demo10300`: [330, 625)
- `demo10301`: [0, 250)
- `demo10301`: [330, 625)
- `demo10302`: [0, 250)
- `demo10302`: [330, 625)
- `demo10303`: [0, 250)
- `demo10303`: [330, 625)
- `demo10304`: [0, 250)
- `demo10304`: [330, 625)
- `demo10305`: [0, 250)
- `demo10305`: [330, 625)
- `demo10306`: [0, 250)
- `demo10306`: [330, 625)
- `demo10307`: [0, 250)
- `demo10307`: [330, 625)
- `demo10308`: [0, 250)
- `demo10308`: [330, 625)
- `demo10309`: [0, 250)
- `demo10309`: [330, 625)
- `demo10310`: [0, 250)
- `demo10310`: [330, 625)
- `demo10311`: [0, 250)
- `demo10311`: [330, 625)

## Heuristic 1

Active-object relative acquisition: represent pickup in the selected block's coordinate/goal frame, sharing one policy across red-first and blue-second occurrences. This should reduce sample complexity and improve transfer to modest object-position changes because the policy learns relative approach, closure and lift rather than memorizing two absolute joint-space routes.

**Evidence inspected by the API**

- `demo10300` observations: 0, 100, 150, 185, 250
- `demo10302` observations: 0, 185, 250, 330, 500, 580, 625
- `demo10301` observations: 330, 420, 500, 580, 625
- `demo10307` observations: 185, 250, 330, 580, 625

**Interpretation**

Across trajectories the same top-down pattern occurs for red and blue despite different absolute Y positions and different entry poses. At demo10300 index 0 the red is on the table while the arm is high/open; by 185-250 the red is grasped and elevated/near its goal. Later, after red placement, demo10301 index 330 shows the gripper open above the tray and the blue still on the table; by 580-625 the blue is grasped and carried high. The learning difficulty is that a joint-space policy must represent two spatially separated pickups and two very different entry poses. The prior assumes the manipulation law is mostly a function of active object-relative geometry and task stage. This differs from the phase and waypoint priors below: it changes the coordinate/role representation rather than imposing a temporal state machine or geometric via-points.

**Applicability**

Transfers when a single active cuboid block sits on the table/tray approach surface and the gripper can reach it top-down with the same Panda, mimic gripper and world-frame observation schema. It assumes the held/active object can be inferred from causal observations plus task stage: before red_at_goal the red block is active; after red is at its goal and the gripper is open, the blue block is active. It should tolerate modest initial XY variation like the demonstrations (red table positions roughly x -0.39 to -0.41, y -0.25 to -0.27; blue roughly x -0.39 to -0.41, y 0.25 to 0.27), but not unseen block shapes or side grasps.

**Implications for a future training/inference pipeline**

Candidate policy input at inference: qpos, qvel, tcp_pose, red_pose, blue_pose, red_goal, blue_goal in world/native frames. Add a causal role module g(o_t)->r_t in {red,blue}, initialized from segment labels and optionally constrained by predicates red_at_goal/blue_at_goal computed from current poses. Form role-normalized features x_t=[qpos,qvel, finger_width, p_tcp-p_obj^r, quat_rel(tcp,obj), p_goal^r-p_obj^r, p_inactive-p_obj^r]. A diffusion/sequence policy predicts either normalized joint targets directly or a Cartesian delta plus learned IK head: a_t=[q_arm_target, gripper_cmd]. Training labels include role r from segment membership/future-free stage labels; future-computed labels may be used only to supervise g during training. Deployment computes r_t causally, samples action sequence pi_theta(a_{t:t+H}|x_{t-K:t},r_t), and decodes to the action contract. The concrete prior is removal of color/absolute-position dependence by sharing weights across red and blue in the active-object frame.

**Assumptions and limitations**

The demonstrations never show ambiguity in task order, failed grasps, or both blocks already partially moved, so active-role inference may be brittle. Expected failure is confident pursuit of the inactive block or oscillation between role choices. A falsifying ablation is a plain absolute-state diffusion policy or a policy given the correct role one-hot: if learned role-relative features do not improve transfer to shifted block starts or if oracle role greatly outperforms causal role selection, this prior is not the main benefit. Successful demonstrations cannot establish recovery from missed grasp or from off-table block positions.

### Handoff interface

**Entry conditions**

Measured support: red acquisition starts at observation 0 with qpos fingers ≈0.04 m, tcp_pose about [-0.384,0,0.442] m, red_pose on table z≈0.020 m. Blue acquisition starts at observation 330 with red already at its goal z≈0.036 m, fingers open ≈0.04 m, and tcp_pose still retreating above the red goal/tray (e.g. demo10300 z≈0.279 m; demo10301 z≈0.213 m; demo10304 z≈0.305 m). The role prior assumes a causal stage selector can choose red first, blue second from object poses and gripper state.

**Exit conditions**

Useful end state is a stable grasp of the active block, fingers near 0.018 m, active block lifted and carried toward/over its goal at z commonly 0.23-0.31 m by index 250 for red and 0.13-0.31 m by index 625 for blue; gripper_command remains -1.0. The handoff does not require zero velocity; several endpoint qvel entries show ongoing transport/descent.

**Failure signatures**

Wrong active-object selection (tcp moves to red after red_at_goal or to blue before red is placed), gripper_command closing while tcp is not above the active block, finger qpos staying ≈0.04 after closure command, active object z remaining ≈0.02 m after lift phase, active object slipping away from tcp, or collision-induced jumps in inactive block pose.

**Overlap role**

The prior explicitly learns the overlapped carry-in/carry-out states: red [185,250) is both late acquisition and early placement; [330,420) is both red-place retreat and blue-acquire approach; blue [580,625) is both late acquisition and early placement. The role representation should remain valid while the predecessor's manipulation state is still changing, not only at a discrete cut.

**Successor readiness**

place_block is ready when the active block is grasped, tcp and active block are above or approaching the correct goal in world frame, and the role selector exposes active_pose/active_goal. The successor needs relative active-object-to-goal features and assurance that the gripper is closed around the block; tolerances beyond the demonstrated height/XY ranges are hypotheses.


## Heuristic 2

Contact-phase bottleneck for pickup: model acquisition as approach -> align -> close -> lift -> carry, with phase-conditioned gripper and arm actions. This should improve grasp timing and avoid averaging across open-loop approach and closed-gripper lift.

**Evidence inspected by the API**

- `demo10300` observations: 100, 150, 185, 250, 450, 500, 550, 625
- `demo10301` observations: 100, 150, 185, 250, 450, 500, 550, 625
- `demo10305` observations: 185, 250, 330, 500, 580, 625

**Interpretation**

At both pickups, gripper opening and object motion create a discrete bottleneck. Demo10300 shows red approach open at index 100, closed/lifted by 150, and high carry by 185-250. Blue similarly is open approaching around 450, closed with the blue just leaving the surface by 500, lifted by 550 and carried by 580-625. A monolithic diffusion model can blur this switch and produce averaged gripper commands; the phase prior makes the contact event an explicit latent variable. Unlike the active-object prior, this is about temporal/contact organization rather than spatial role sharing.

**Applicability**

Applies when grasp acquisition has a clear top-down contact event observable through gripper width, gripper command and object height: open fingers approach, close near block, object z rises with tcp after contact. It relies on the same gripper mechanics and roughly the same friction/contact behavior as the demonstrations. It is less tied to exact initial XY positions than a monolithic sequence, but it depends on the order and quality of the contact transition.

**Implications for a future training/inference pipeline**

Train a latent-state sequence model z_t in {approach, align, close, lift, carry}. Inputs are causal observations plus relative active-object features. Training labels are computed from demonstration observations: close onset when gripper_command becomes -1 and finger qpos decreases; lift when active_pose.z exceeds table z by >~0.05 m; carry when z>~0.12 m and object-tcp offset is stable. Loss: diffusion action loss plus cross-entropy/ELBO for z_t and auxiliary predictions of next active z and finger width. Pseudocode: z_t ~ HMM/Transformer(o_{t-K:t}); a_{t:t+H} ~ DP_theta(o,z); L=L_action+lambda1 CE(z,z*)+lambda2 ||hat_z_obj(t+1)-z_obj(t+1)||^2. At inference, phase posterior is computed only from current/past qpos, tcp_pose and object poses; no future labels are used. The gripper decoder is phase-conditioned so closure commands are concentrated in the close/lift transition.

**Assumptions and limitations**

Phase labels are inferred from successful demonstrations, not force sensors, so true contact timing is uncertain. A failure signature is a correct geometric approach with wrong gripper timing, especially closing too early above the block or too late after contact. Falsify by comparing to a representation-identical policy without latent phase/contact auxiliary losses; if timing and grasp success do not improve under initial-pose perturbations, the bottleneck prior is not useful. Dataset cannot prove behavior under collision, partial grasps, or variable object heights.

### Handoff interface

**Entry conditions**

This phase prior can take over when the gripper is open (finger qpos around 0.04 m) and the active block is either on the table z≈0.020 m or the arm is in the demonstrated red-to-blue transition with no block held and red already placed. Initial tcp may be high/home (red) or retreating above the tray at z≈0.21-0.31 m (blue).

**Exit conditions**

Exit is after the latent phase reaches carry/lift: gripper command and finger qpos indicate closure (command -1; observed finger qpos ≈0.018 m), active block z has risen well above table, and tcp-object relative pose remains small enough to infer a grasp. The successor place phase can start before a precise hover target, as in the [580,625) overlap.

**Failure signatures**

The latent phase advances to lift without object z increasing, fingers do not close to ≈0.018 m, object pose lags behind tcp, or the model emits repeated open/close toggles near contact. Premature successor transfer is visible if place_block receives an ungrasped object or open fingers during its descend phase.

**Overlap role**

The overlapped intervals intentionally label ambiguous phase boundaries for both policies: late acquisition phases include the beginning of place descent/carry, and acquire_blue includes the open-gripper retreat from red placement before the approach phase has fully stabilized. The phase model should learn that adjacent policies share these contact/carry states.

**Successor readiness**

place_block needs the phase posterior p(carry/lift)>threshold or equivalent measured evidence: closed fingers and active object lifted. The demonstrated handoff supports heights from about 0.13 m to 0.31 m; robustness to lower or slipping carries is untested.


## Heuristic 3

Clearance via-point acquisition: constrain the learned pickup to pass through reusable hover, grasp, lift-clearance and goal-approach waypoints in world/active-object coordinates. This should improve long-range transport generalization and reduce unsafe low paths.

**Evidence inspected by the API**

- `demo10300` observations: 0, 50, 100, 150, 185, 250
- `demo10300` observations: 330, 420, 450, 500, 550, 580, 625
- `demo10304` observations: 185, 250, 580, 625
- `demo10310` observations: 185, 250, 580, 625

**Interpretation**

The inspected motions are not arbitrary curves: the arm descends near the table block, closes, lifts to a clearance height, then translates with the object high before placing. Demo10300 red moves from high home to near red at 100, red z≈0.284 at 185 and near the goal at 250; blue is approached from the red retreat, closed near 500, lifted around 550-580 and high near the blue goal at 625. These repeated geometric milestones address the learning difficulty of long-horizon continuous motion from sparse demos. This differs from the phase prior by adding explicit spatial waypoints and clearance constraints.

**Applicability**

Applies when top-down pickup and transport must avoid the table, tray rim and previously placed block using similar object/tray geometry. It assumes a reliable world-frame tcp_pose and object poses and that the robot can follow Cartesian via-points with the same kinematic reach. It is most useful for position generalization within the tray/table workspace; it is not intended for radically different obstacles or side grasps.

**Implications for a future training/inference pipeline**

Augment the policy with a learned or analytic waypoint head. From demonstration futures compute labels w1=pregrasp hover, w2=grasp, w3=lift-clearance, w4=handoff-above-goal in world frame for the active role. At inference compute current progress by nearest waypoint segment using only current tcp/active poses. The diffusion policy conditions on [relative state, w_next-p_tcp, progress] and either outputs joint targets directly or desired Cartesian delta clipped toward w_next, then a learned IK/action head maps to arm_joint_positions; gripper command is tied to waypoint phase. Add auxiliary MSE on predicted w_next and barrier losses for z above table/tray rim during lift/carry. This concrete mechanism encodes the prior as explicit subgoal geometry rather than only sequence history.

**Assumptions and limitations**

The via-points are inferred, not annotated, and some demonstrations (e.g. 10304/10307/10310) have red/blue endpoints at lower heights, so a rigid waypoint schedule could overconstrain. Falsification: compare to an unconstrained object-relative policy and to a phase-only policy under shifted starts; if waypoint prediction does not reduce collisions or improve target approach, the assumed geometric skeleton is unnecessary. Dataset cannot prove obstacle avoidance for novel tray geometry or recovery from collisions.

### Handoff interface

**Entry conditions**

The waypoint prior accepts the same acquisition entries but additionally assumes a collision-free vertical approach corridor above the active block and a feasible clearance corridor to the active goal. Demonstrated tcp starts include home z≈0.442 m and red-to-blue transfer z≈0.21-0.31 m with fingers open.

**Exit conditions**

Exit when the active block has passed a learned clearance waypoint and is on a descending/transport route to the goal: examples include red at z≈0.30 m near the red goal at index 250 in demos 10301/10302/10305/10311 and blue at z≈0.26-0.31 m near the blue goal at index 625 in demos 10300/10301/10302/10311. In faster demos the stop can be lower (blue z≈0.13 m in 10307/10310), which is deliberately shared with place_block.

**Failure signatures**

Predicted via-point below the tray rim/table clearance, tcp path grazing the tray walls, object z not increasing after the lift waypoint, or large deviation from p_obj-to-p_goal corridor. Premature handoff occurs if the place policy receives a block outside the funnel above the target.

**Overlap role**

Overlaps teach that waypoints are not hard cutpoints. Red place starts while acquisition is still carrying through the clearance corridor, acquire_blue starts while the previous place retreat is still clearing the tray, and blue place starts before acquisition has fully completed the goal approach. Both adjacent policies can therefore produce safe clear-and-carry motions.

**Successor readiness**

place_block needs a block that is either above the target funnel or entering it with adequate clearance. This prior can pass explicit predicted via-point progress s in [0,1] or the current active-object height/goal-relative vector; demonstrated support is along successful high arcs only.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
