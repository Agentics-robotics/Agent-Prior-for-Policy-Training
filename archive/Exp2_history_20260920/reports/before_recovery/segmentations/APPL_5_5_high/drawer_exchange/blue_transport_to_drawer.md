# blue_transport_to_drawer

Carry the grasped blue block from the table area to the open drawer cavity and begin insertion/release.

## Segmentation

Segments begin before blue is fully attached (index 810) to include the contact/lift overlap, pass through high carried transport (870-930), and stop at 1030 after the block has reached the drawer z band and release/retreat is underway. This groups long-range carried-object aiming into the open drawer with enough final overlap for local completion.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo1000`: [810, 1030)
- `demo1001`: [810, 1030)
- `demo1002`: [810, 1030)
- `demo1003`: [810, 1030)
- `demo1004`: [810, 1030)
- `demo1005`: [810, 1030)
- `demo1006`: [810, 1030)
- `demo1007`: [810, 1030)
- `demo1008`: [810, 1030)
- `demo1009`: [810, 1030)
- `demo1010`: [810, 1030)
- `demo1011`: [810, 1030)

## Heuristic 1

Drawer-frame blue insertion: represent transport target relative to the open drawer cavity to improve placement inside the drawer.

**Evidence inspected by the API**

- `demo1000` observations: 810, 870, 930, 1030
- `demo1002` observations: 810, 870, 930, 1030
- `demo1009` observations: 810, 870, 930, 1030

**Interpretation**

The insertion target moves with drawer_position, not with a fixed joint posture. Encoding drawer-frame blue error addresses the generalization difficulty of aiming a carried object into an articulated container.

**Applicability**

Transfers when drawer is open (drawer_position ~0.297-0.300 m), blue is grasped/lifted, and the goal is the cavity defined by drawer_center=[0.19-drawer_position,0,0.035] with allowed blue z 0.053-0.074. Requires blue_pose and drawer_position.

**Implications for a future training/inference pipeline**

Inputs causal qpos,qvel,tcp_pose,blue_pose,drawer_position,red_pose. Compute drawer frame origin d=[0.19-drawer_position,0,0.035] and blue error e=blue_pose.xyz-d. Condition diffusion decoder on e, tcp-blue attachment features, and current completion predicates. Training auxiliary predicts future blue_inside predicate from task contract; labels use future blue_pose/drawer_position.

**Assumptions and limitations**

Drawer target is fixed except for small drawer_position variation; no closed/partially open insertion failures are shown. Compare drawer-frame target encoding to absolute target replay; benefit should be reduced cavity miss rate under drawer_position variation.

### Handoff interface

**Entry conditions**

Supported entry at index 810/870: blue is being grasped/lifted, fingers close near 0.018 m by 870, blue_pose z around 0.27-0.28, drawer_position >0.26, red on pad.

**Exit conditions**

Blue is brought to/into drawer: by 1030 blue_pose is near x -0.172, y 0.073, z 0.063 and the gripper is open or opening, satisfying blue-inside z/xy when drawer remains open.

**Failure signatures**

blue is not attached at entry, blue xy path misses drawer cavity, drawer_position falls below threshold, or release occurs outside cavity/too high.

**Overlap role**

[810,870) shares attachment with predecessor; [930,1030) shares insertion/release with final skill so both learn late corrections rather than a one-step cut.

**Successor readiness**

blue_release_retreat_complete needs blue centered in/above the open drawer cavity, z descending toward 0.063, gripper ready to open, and no collision with drawer lip.


## Heuristic 2

Rigid blue-carry prior: maintain a stable tcp-to-blue transform during transport and use it to aim object motion into the drawer.

**Evidence inspected by the API**

- `demo1001` observations: 870, 930, 1030
- `demo1005` observations: 870, 930, 1030
- `demo1011` observations: 870, 930, 1030

**Interpretation**

From 870 to 930 blue remains high and moves with tcp toward the drawer, then descends to final z by 1030. Modeling this rigid relation helps convert desired object motion into arm joint actions rather than learning from joints alone.

**Applicability**

Useful once blue is grasped and co-moving with tcp during transport. Requires stable blue_pose and gripper state; assumes no deliberate in-hand reorientation.

**Implications for a future training/inference pipeline**

Train a carried-object dynamics model with latent offset r=T_tcp^{-1}blue while fingers closed. Diffusion samples are rescored by predicted future blue_pose following tcp motion until release. Inputs are causal state; labels use future blue_pose for supervised offset/prediction loss. Deployment updates r online from current tcp/blue poses.

**Assumptions and limitations**

Rigid-carry assumption fails under slip or collisions with drawer wall. A comparison without rigid-offset prediction should show worse blue future-pose prediction if this prior is useful.

### Handoff interface

**Entry conditions**

Enter when blue attachment latent is high: at 870 fingers are closed ~0.018 m and blue z is high; tcp-blue offset is stable. Red is already on pad and drawer open.

**Exit conditions**

Exit once blue is near the drawer target and release has begun; blue z near 0.063 by 1030 and gripper open near 0.04 m.

**Failure signatures**

tcp moves while blue_pose lags/slips, blue rotates excessively, or the policy treats blue as attached after gripper opens.

**Overlap role**

The initial lift and final release are both shared, teaching predecessor/successor to agree on attachment and detachment boundaries.

**Successor readiness**

Final skill should see blue no longer requiring long-range travel, with object pose close enough for local centering/release and fingers opening.


## Heuristic 3

High-arc transport then descent prior: separate clearance transport from local insertion into the drawer.

**Evidence inspected by the API**

- `demo1003` observations: 810, 870, 930, 1030
- `demo1008` observations: 810, 870, 930, 1030
- `demo1010` observations: 810, 870, 930, 1030

**Interpretation**

The object moves from table z to high z, across the workspace, then down into the drawer. A stage prior prevents blending lateral transport and descent, which could cause collisions or premature release.

**Applicability**

Applies when the path from table to drawer requires a high carry followed by controlled descent into an open cavity. Requires world z features and drawer geometry unchanged.

**Implications for a future training/inference pipeline**

Use a temporal latent s_t={lift_high, translate, descend_insert}. Labels from blue z and blue-drawer xy error: high while z>0.25, descend when xy error small. DP decoder conditioned on s_t; add auxiliary losses predicting future z stage and blue_inside. Inference filters s_t causally from current blue z and drawer-frame error.

**Assumptions and limitations**

Only successful high arcs are shown; no alternative shorter paths. Falsify by testing if a high-arc latent reduces drawer-edge contacts versus flat transport; if not, the constraint is unnecessary.

### Handoff interface

**Entry conditions**

Starts after or during blue lift with blue z rising from table to ~0.28; gripper closed, drawer open, tcp above table/drawer obstacles.

**Exit conditions**

Ends in local insertion/release overlap when blue has descended into the drawer z band around 0.063 and gripper is open/opening.

**Failure signatures**

low lateral travel clips the drawer edge, descent starts before blue xy aligns with drawer, or the object is dropped while still above the cavity.

**Overlap role**

[930,1030) is shared so the final policy can take over before release is complete and learn descent corrections while predecessor still controls carried blue.

**Successor readiness**

The successor receives a near-cavity, low-speed, near-release state instead of a distant carry state; blue_inside predicate should be almost satisfied.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
