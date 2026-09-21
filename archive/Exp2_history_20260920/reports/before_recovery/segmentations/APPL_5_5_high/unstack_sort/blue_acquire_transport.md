# Blue acquire, lift, and carry toward goal

With red already placed, acquire the blue block from the source, lift it safely, and carry it to a state near or above the blue goal suitable for final placement.

## Segmentation

Segments [330,590) begin with red already released at the red goal and the gripper open/high, include retreat from red, approach to the blue block, blue grasp and lift, and transport toward the blue goal until the block is goal-near or beginning descent. They overlap Skill 2 from 330-520 for the full object switch and overlap Skill 4 from 520-590 for the blue high-carry/final descent transition.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo10200`: [330, 590)
- `demo10201`: [330, 590)
- `demo10202`: [330, 590)
- `demo10203`: [330, 590)
- `demo10204`: [330, 590)
- `demo10205`: [330, 590)
- `demo10206`: [330, 590)
- `demo10207`: [330, 590)
- `demo10208`: [330, 590)
- `demo10209`: [330, 590)
- `demo10210`: [330, 590)
- `demo10211`: [330, 590)

## Heuristic 1

Identity-conditioned top-grasp reuse: treat blue pickup as the same geometric grasp-and-lift template as red pickup, selected by a blue target token and conditioned on red as a finished obstacle.

**Evidence inspected by the API**

- `demo10200` observations: 330, 460, 520, 590
- `demo10207` observations: 330, 460, 520, 590
- `demo10208` observations: 330, 520, 590

**Interpretation**

The blue acquisition mirrors the earlier red top-grasp: approach source, close, lift vertically, then carry. Demonstrations show the same finger aperture behavior and a stable tcp-object offset after lift. Sharing or encoding common pickup geometry should help learn blue pickup from few samples while still conditioning on color/goal identity. This differs from the transport prior below, which focuses on after-grasp motion to the goal.

**Applicability**

Transfers when red has already been placed and blue remains at the original source, with the same top-down gripper and block geometry. It uses object identity: blue is the target to pick and red is a placed obstacle/finished object. It is suitable for small source xy variation but not for side grasps or if blue has been tipped.

**Implications for a future training/inference pipeline**

Use a shared top-grasp encoder for target object o_target=blue with identity embedding e_blue and context red fixed. Inputs: qpos,qvel,tcp_pose,blue_pose,red_pose,goals. Features: tcp-blue, blue-red, blue_goal-blue, and target token. Diffusion action head is the same structure as red top-grasp but parameters may be partly shared or initialized from red-grasp prior. Training objective: action denoising + auxiliary next blue z and grasp-success label from future blue lift. Deployment sets target=blue once red_at_goal/open or directly within this skill; sample actions conditioned on blue-centric features.

**Assumptions and limitations**

Although geometry resembles red pickup, the dataset does not show independent randomization of object color, so identity-conditioned symmetry may be partially confounded with time. It may fail if red is not safely out of the way. Falsify by training a version without shared pickup geometry or object identity embeddings; benefit should appear in fewer missed blue grasps with varied source positions. Recovery after a failed red placement is unseen.

### Handoff interface

**Entry conditions**

Measured entry support ranges from high post-red-release states with gripper open (demo10200 index 330 qpos fingers about 0.04 m, red at goal) to low blue contact/early lift states in the overlap. The policy assumes blue_pose z about 0.020 m before grasp and red_pose already at red_goal.

**Exit conditions**

Exit when blue is grasped, lifted, and translated near the blue goal or beginning descent. Examples: demo10200 index 590 blue near [-0.239,0.254] at z 0.295 m; demo10201 index 590 already descending at z 0.221 m; demo10207 index 590 z 0.296 m.

**Failure signatures**

Gripper closes before centering over blue, blue_pose does not rise after closure, red_pose changes during blue approach, or tcp-blue offset deviates from the top-grasp relation. Such failures mean final placement will start without a secured blue block.

**Overlap role**

The [330,520) overlap with Skill 2 covers object switching and the blue grasp; the [520,590) overlap with Skill 4 covers blue high carry toward the goal. This prior accepts starts before contact and exits before final placement is complete.

**Successor readiness**

The final placer needs blue in hand, fingers closed near 0.018 m, and blue_pose high enough for controlled transport/descent. Demonstrated readiness includes blue z about 0.28 m at 520 and goal-near high/descending states at 590.


## Heuristic 2

Height-staged blue transport prior: after grasp, carry blue through a lift-to-safe-height, xy-translate-to-goal, then descend sequence driven by blue_pose-to-blue_goal error.

**Evidence inspected by the API**

- `demo10200` observations: 520, 560, 590
- `demo10207` observations: 520, 560, 590
- `demo10201` observations: 520, 590
- `demo10204` observations: 520, 590

**Interpretation**

Across inspected trajectories, blue first reaches a safe height near 0.28 m at the source, then moves in xy toward blue_goal, and only later descends. Encoding a staged vector field reduces the burden on diffusion to infer long-horizon transport from raw joints and should avoid unsafe table-level dragging. This is a temporal/spatial transport prior, distinct from target grasp reuse.

**Applicability**

Applies to moving a securely grasped blue block from source to blue_goal after the red block is already placed. It requires that the policy can observe blue_pose and blue_goal in the world frame and that lifting before large xy motion is feasible with the same arm workspace. It is not valid if the object must be dragged on the table.

**Implications for a future training/inference pipeline**

Implement a waypoint/phase-conditioned DP for blue transport. Causal features: h = blue_pose.z, e_xy = blue_goal.xy-blue_pose.xy, e_z_goal=blue_goal.z-blue_pose.z, tcp-blue. Train auxiliary waypoint labels from demonstration futures: stage=lift if h<0.25 and distance to source small, stage=translate if h high and e_xy large, stage=descend if e_xy small. The denoiser conditions on stage and object-goal error. Action is still absolute Panda joint targets plus gripper closed; optional predicted Cartesian waypoint w=[blue_goal.xy, z_safe] then [blue_goal.xyz]. Deployment estimates stage from current blue_pose and samples actions.

**Assumptions and limitations**

The staged lift-translate-descend structure is inferred from successful motions and not enforced by kinematics; it may be too rigid if obstacles require different arcs. Falsify by comparing against an unstructured diffusion policy on start-source variations; the prior should reduce low-altitude lateral dragging. The data cannot establish collision avoidance for obstacles other than red.

### Handoff interface

**Entry conditions**

Can take over from Skill 2 during or just after blue lift, e.g. demo10200 index 520 blue_pose z 0.280 m over the source and fingers closed, or from earlier overlap states where tcp is approaching blue and gripper is open. Red must be placed and static.

**Exit conditions**

Exit when blue is near blue_goal in xy and either still high or beginning descent; e.g. demo10200 index 590 blue at x -0.239,y 0.254,z 0.295, or demo10204 index 590 x -0.239,y 0.253,z 0.256.

**Failure signatures**

Blue is transported laterally while still near table height before being lifted, blue-goal xy error does not decrease, or tcp/blue slip during transport. Such states are unsafe for the final place policy.

**Overlap role**

The [520,590) overlap teaches both this skill and final placement to handle blue high carry and the first part of controlled descent; there is no requirement that handoff happen only after reaching exact goal height.

**Successor readiness**

Successor needs blue_goal-blue_pose xy error small or decreasing, blue in gripper, and enough height margin to descend. Demonstrated entry at 520 has high source carry; by 590 many trajectories are goal-near, but some are already descending, so successor is trained on both.


## Heuristic 3

Completed-object invariance prior: once red is placed, treat red_pose as a fixed success condition and bias blue acquisition/transport against disturbing it.

**Evidence inspected by the API**

- `demo10200` observations: 330, 520, 590
- `demo10202` observations: 330, 520, 590
- `demo10211` observations: 330, 520, 590

**Interpretation**

After index 330 red is at the red pad and remains essentially unchanged while the robot grasps and moves blue. The task success requires both red_at_goal and blue_at_goal, so blue policies should not ignore the completed red object. Explicitly preserving red can prevent policies from using shortcuts or paths that graze the red block. This differs from the grasp and transport priors by adding a non-target invariant.

**Applicability**

Applies once red has been placed at its goal and should remain undisturbed while the robot handles blue. It requires accurate red_pose and enough workspace clearance. It transfers to sorted-block tasks where completed objects become fixed constraints, but not to tasks requiring re-adjustment of the completed red block.

**Implications for a future training/inference pipeline**

Add a completed-object constraint to the blue skill. Inference inputs include red_pose, red_goal, blue_pose, blue_goal, tcp_pose. Training labels: future red displacement over horizon from demonstrations. Objective: L_DDPM(action)+lambda||red_pose_{t+H}-red_pose_t||^2+mu*I[red_at_goal] BCE. A learned dynamics/pose predictor estimates red disturbance from sampled action horizons; at deployment, sample K horizons and choose one with low predicted red displacement and good blue-goal progress. This encodes a constraint on non-target completed object, not a new scripted controller.

**Assumptions and limitations**

Because red never moves after successful placement, the auxiliary may be trivially satisfied and not causally responsible for performance. It cannot teach how to repair red if displaced. Falsify by ablating red-static loss and testing for red disturbances under perturbed trajectories; no difference would weaken the claim.

### Handoff interface

**Entry conditions**

Enter after red_at_goal, demonstrated by red_pose near [-0.241,-0.248,0.020] at index 330 with open gripper. The policy can also start during blue approach if red remains static and blue is still at source.

**Exit conditions**

Exit with red unchanged at goal and blue in hand near/above its goal, as seen at indices 520 and 590 where red_pose remains constant while blue_pose rises and translates.

**Failure signatures**

Any red_pose drift/tilt during blue acquisition, tcp path passing through the red pad area, or sampled actions that bring the gripper close to red after attention has switched to blue. This indicates completed-object constraints are violated.

**Overlap role**

Both adjacent overlaps require red invariance: Skill 2/3 overlap from 330-520 establishes red should stay fixed while blue is approached; Skill 3/4 overlap 520-590 maintains red fixed during blue transport.

**Successor readiness**

Final blue placement needs red to satisfy red_at_goal already; otherwise completing blue alone will not satisfy the task contract. Demonstrated red support is stable from 330 through final observations; recovery if red is off-goal is unseen.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
