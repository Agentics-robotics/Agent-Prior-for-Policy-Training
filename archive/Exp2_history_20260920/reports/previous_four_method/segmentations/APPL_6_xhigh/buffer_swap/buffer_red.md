# Evacuate red to the buffer and prepare blue pickup

Vacate the initially red region by moving red to a supported free buffer, release it and transition into stable blue acquisition. Preserve blue until its approach, then close and lift it far enough to support transfer to the blue-goal policy.

## Segmentation

Group the initial red evacuation in all demonstrations: identical home entry, red grasp/lift, red transfer to the free buffer, opening/support and retreat. Extend the useful end through blue approach, closure and lifting so the next policy can take control before, during or after acquisition. Ends differ individually and were inspected: demo10100:490 blue z=0.1865; 10101:487=0.1675; 10102:483=0.1717; 10103:488=0.2112; 10104:481=0.1279; 10105:480=0.1838; 10106:484=0.1954; 10107:493=0.1256; 10108:482=0.1278; 10109:486=0.2083; 10110:489=0.1870; 10111:485=0.2018 m. Red remains supported near [-0.181,0,0.02] at these ends. This is one evacuation problem with an intentionally broad next-object acquisition interface, not a claim that blue has reached its goal.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo10100`: [0, 490)
- `demo10101`: [0, 487)
- `demo10102`: [0, 483)
- `demo10103`: [0, 488)
- `demo10104`: [0, 481)
- `demo10105`: [0, 480)
- `demo10106`: [0, 484)
- `demo10107`: [0, 493)
- `demo10108`: [0, 482)
- `demo10109`: [0, 486)
- `demo10110`: [0, 489)
- `demo10111`: [0, 485)

## Heuristic 1

Occupancy-conditioned staging graph: represent which object blocks which goal and predict a reachable free intermediate site before generating motor actions. Hypothesis: a relational representation separates rearrangement dependency from absolute joint motion and generalizes better to small start-position changes than memorizing a world-coordinate trajectory.

**Evidence inspected by the API**

- `demo10100` observations: 0, 120, 180, 240, 275, 320, 455, 490
- `demo10111` observations: 0, 242, 276, 388, 448, 485
- `demo10103` observations: 0, 244, 488

**Interpretation**

Across demo10100, demo10103 and demo10111 the starting block positions change by centimetres but red is staged at approximately the same free central site and blue subsequently becomes active. The key dependency is that red_goal is initially occupied, not that a particular timestamp means 'go to center'. Encoding occupancy and roles should reduce the burden on a small dataset to infer that dependency from raw coordinates. This is a representation/subgoal hypothesis, distinct from temporal contact inference and trajectory hierarchy below. A reusable free-space planner is a scientific extrapolation, not an observed ability.

**Applicability**

Small planar rearrangements with two distinguishable, equal-sized blocks, an occupied destination, a free staging area and the same Panda/table geometry. Relative object/goal encoding may transfer to modest initial-position changes, but joint kinematics, world gravity, table height and reachable buffer geometry must remain represented. The catalog contains only red-first solutions and one buffer location.

**Implications for a future training/inference pipeline**

Candidate G: a role-conditioned graph encoder plus an absolute-joint action diffusion head. Inputs are qpos, qvel, tcp_pose, red_pose, blue_pose, red_goal and blue_goal; node features include world height/orientation, object identity and role; edge features include relative XYZ in metres, footprint-to-region margins and TCP-object offsets. Retain a robot/world branch because Panda joint actions are not translation-equivariant. A trainable staging head predicts b_world and uncertainty from the initial occupancy graph; training label is the supported red pose after first release, obtained from demonstration futures only. A role head predicts approach-red/carry-red/buffer-release/approach-blue/blue-lift from causal state/history, supervised by object-height and gripper-change labels. The denoiser conditions on graph, predicted b and role probabilities, predicts normalized absolute [q_target(7),g] chunks and decodes through the shared source normalizer and action bounds. Train denoising MSE plus staging-position likelihood and role cross-entropy; use predicted rather than oracle b for at least part of training. At inference compute the graph and b from observations, denoise a chunk and replan; no future supported pose is accessible. Test prediction: relative geometry and explicit occupancy reduce destination-confusion and initial-position sensitivity more than a flat encoder, without claiming global SE(2) equivariance.

**Assumptions and limitations**

The expert may simply use a fixed staging waypoint; this dataset cannot distinguish deliberate free-space reasoning from that simpler explanation. Learnable buffer prediction could collapse to a constant, or extrapolate to unreachable space. Compare against fixed-buffer and flat world-state DP with matched capacity, measuring held-out initial-position performance and occupancy violations. New buffer locations, color-order reversal and multi-object search are unestablished.

### Handoff interface

**Entry conditions**

Initial demonstrated entry has qpos[0:7]=[0,-0.3,0,-2.1,0,1.8,0.7854] rad, qpos[7:9]=0.04 m each, qvel=0, tcp_pose.xyz about [-0.384,0,0.442] m world, red near [-0.35,-0.20,0.02] and blue near [-0.35,+0.20,0.02]. Graph must identify blue occupying red_goal, not mistake proximity to a region for red success. Arbitrary initial arm poses are unobserved.

**Exit conditions**

Red is supported near [-0.181,0,0.020] m outside both regions; blue acquisition is under way or blue is lifting with gripper_command=-1 and fingers about 0.01824-0.01829 m. Segment endpoints show blue z 0.126-0.211 m and TCP roughly 0.009 m toward negative world x from blue. Continue closed grip during lifting; no home reset is required.

**Failure signatures**

Red follows TCP on retreat instead of remaining in the buffer, predicted staging point lies in a marked/occupied region, fingers close without blue co-motion, or blue remains at z=0.02 while TCP rises. A geometrically free predicted buffer alone is not evidence of support or reachability.

**Overlap role**

Both policies learn red descent while held, opening to +1, retreat to about TCP z=0.30 m, open approach to blue, closure to -1 and blue lift. In demo10100 this is 240:490; graph roles change from red-active to blue-active inside the overlap, rather than switching only at its endpoint.

**Successor readiness**

transfer_blue receives causal object/TCP poses, qpos and qvel and can infer whether it must finish red setdown or blue acquisition. The graph candidate should pass a predicted role/buffer estimate only as optional context, never as a substitute for observed contact geometry. Support covers the inspected overlap states; spatial/contact-error tolerance remains a hypothesis.


## Heuristic 2

Attachment is a latent, persistent state: infer grasp/support changes from finger aperture together with object-TCP co-motion, rather than equating a close command with successful grasp. Hypothesis: contact-belief-conditioned diffusion reduces premature lifting or release and tolerates modest timing variation.

**Evidence inspected by the API**

- `demo10100` observations: 95, 105, 115, 130, 150, 265, 275, 285, 320, 440, 455, 465, 490
- `demo10111` observations: 95, 105, 140, 276, 448, 485
- `demo10101` observations: 452, 487

**Interpretation**

demo10100:95 and 105 have almost identical TCP positions but very different finger configuration; 115-to-150 then changes object height while fingers remain near 0.018 m. Likewise 265-to-275 changes support/attachment before retreat, and 440-to-490 covers a second grasp. demo10111 reproduces these relations with slightly different object coordinates. Motor intent cannot be identified reliably by TCP pose or command sign alone; a learned latent attachment state is the proposed inductive bias. Unlike the graph candidate it changes temporal inference and gripper-event learning rather than waypoint selection.

**Applicability**

Same position-controlled Panda, parallel fingers and 0.04 m cubes, with causal object poses and finger positions. Transfer requires sufficient state accuracy to detect relative co-motion; neither force sensing nor direct contact labels are supplied.

**Implications for a future training/inference pipeline**

Candidate C: causal GRU contact-belief encoder conditioning a diffusion denoiser with separate arm and gripper output projections. Inputs are the shared-normalized state history, physical finger positions qpos[7:9] in m, TCP/object relative transforms, and backward finite differences at dt=0.05 s. Labels for empty, closing, red-attached, red-supported/releasing, blue-closing and blue-attached are computed for training using action gripper sign, subsequent object lifting and near-constant TCP-object transform. Learn emissions and transition logits rather than a scripted phase controller; soft transition-consistency regularization discourages implausible instantaneous attached-to-empty flips. Train denoising loss plus belief cross-entropy and short-horizon attachment/co-motion prediction. Output and bound all eight absolute commands under the original pd_joint_pos contract; gripper remains a learned dimension, with event-balanced sampling/loss to avoid dilution by long holds. Deployment updates the GRU from observations, conditions denoising on belief probabilities and executes short chunks. Futures supply supervision only, never belief input. Test prediction: fewer early-lift and early-open errors under varying finger-closing delays than a memoryless DP.

**Assumptions and limitations**

There are no failed grasps, deliberate slips or ambiguous occlusions, so a successful co-motion label is only a proxy for contact. Near-static fingers exhibit nonzero qvel; naive velocity thresholds will over-trigger. Ablate recurrent state and co-motion supervision separately against current-state DP; falsification is unchanged grip-event accuracy or more premature opening. Robust regrasp behavior cannot be inferred from these demonstrations.

### Handoff interface

**Entry conditions**

At the initial home state fingers are open at 0.04 m each and neither block is attached. Initialize the recurrent belief from observed poses/qpos/qvel, not from an assumed closed-grip command. Internal mode inference must accommodate the 95-to-105 transition where TCP remains near table height while fingers move from open to about 0.0183 m.

**Exit conditions**

A high posterior for red-supported and blue-attached accompanies the observed red buffer pose and blue rise; retain -1 gripper while transferring control. At demo10100:490 blue z=0.1865 m with nonzero arm qvel, so exit is not a zero-velocity arm condition.

**Failure signatures**

Closed command with fingers near the empty-grip limit, persistent blue at table height during TCP ascent, growing TCP-blue relative transform, or mode posterior oscillating between release and carry. Instantaneous finger qvel is noisy even in a successful held grasp and must not alone trigger failure.

**Overlap role**

The shared interval teaches the entire belief change held-red -> supported-red/open -> empty retreat -> blue approach -> closing-blue -> attached-blue. Both policies can complete opening or closing that began before transfer; they must not reset gripper state because a skill index changed.

**Successor readiness**

Provide the successor actual recent qpos/qvel and object/TCP poses so it can reconstruct attachment belief; a mode posterior is optional and should be recalibrated from those observations. A proposed multi-frame stable attachment check is an unvalidated tolerance, not a success-contract requirement.


## Heuristic 3

Separate free-space route geometry from contact-scale motor detail: model lift/transit/descent/retreat as a coarse task-space path and learn a conditional joint-space refinement. Hypothesis: the hierarchy reduces long-horizon coordination errors without replaying exact durations.

**Evidence inspected by the API**

- `demo10100` observations: 60, 95, 130, 150, 180, 240, 275, 320, 360, 390, 420, 455, 490
- `demo10111` observations: 95, 140, 180, 242, 276, 320, 388, 448, 485
- `demo10104` observations: 238, 481

**Interpretation**

TCP descends over red, lifts largely vertically before the lateral buffer transport, descends to the buffer, retreats to about 0.30 m and traverses toward blue before another descent. demo10111 follows the same geometry but not exactly the same joint/time trace; demo10104's overlap begins higher during descent. The reusable regularity is a multi-scale route/contact structure, not exact waypoint timestamps. This candidate changes the action-generation hierarchy and task-space training bottleneck; it does not use an explicit occupancy planner or persistent contact-belief GRU.

**Applicability**

Collision-free tabletop transfers with upright objects, downward-oriented TCP and a useful free-space height band around 0.27-0.30 m. Retain gravity direction, table height, Panda joint limits, PD lag and gripper actuation; cluttered routes and overhead fixtures are absent.

**Implications for a future training/inference pipeline**

Candidate H: hierarchical diffusion rather than the graph or contact-GRU architecture. Encode causal qpos/qvel and world TCP/red/blue poses/goals. A coarse diffuser predicts a short ordered sequence of task-space waypoints [TCP XYZ, rotation representation, gripper state, duration] relative to the currently active object and world vertical; a fine conditional joint diffuser maps the coarse path plus current qpos/qvel into 32 absolute joint-target/gripper commands. Train coarse labels from future tcp_pose/action gripper histories, with waypoint sampling concentrated around lift completion, high transit, descent, opening and closure; durations and phase labels are training targets, not inference inputs. Train both denoising losses and a differentiable learned TCP rollout head supervised by subsequent tcp_pose, adding consistency between coarse path and decoded motor trajectory. Use a soft penalty for low-height lateral sweep when neither approach nor supported placement is indicated; do not force constant height or require an unavailable IK tool. At deployment generate/revise coarse waypoints from current observations and execute only the first fine chunk. Shared complete-demo normalization applies to source fields and motor outputs. Test prediction: fewer table-level sweeps and better timing robustness with fewer demonstrated trajectories, at a possible cost in inference compute.

**Assumptions and limitations**

High arcs may reflect the source planner's fixed template rather than necessary collision clearance. A coarse bottleneck could erase fine contact corrections or over-constrain shorter valid routes. Compare flat joint DP, hierarchy without height/order features and this hierarchy at matched sampling compute; measure contact accuracy and low-clearance sweeps as well as success. Obstacle avoidance, safe exploration and disturbance recovery are not established.

### Handoff interface

**Entry conditions**

Initial open home pose is high above the table and stationary, as at index 0 of each segment. Geometry uses world z up and TCP orientation from wxyz; the learned hierarchy must still generate a feasible joint route from the observed qpos, not start at its first predicted waypoint.

**Exit conditions**

Red has been set down and released; blue is closed-grip lifting within the demonstrated height range, with upward motion that the successor can continue. At demo10111:485 TCP z=0.2019 m, blue z=0.2018 m and fingers about 0.01825 m; this is a moving intermediate waypoint, not a dwell target.

**Failure signatures**

Large lateral displacement near the table before lift, early descent while XY is still misaligned, decoded joints at bounds, waypoint/joint-decoder disagreement, or a cube failing to follow a nominal lift. These identify breakdown of the hierarchy, not proven recovery triggers.

**Overlap role**

Both hierarchies observe and learn low red setdown/open, vertical retreat, high lateral blue approach, low blue closure and its rising path. The 12-second overlap permits handing off between coarse waypoints rather than requiring completion of a full precomputed plan.

**Successor readiness**

The successor needs observed current TCP height, qpos/qvel and attachment evidence, plus an unfinished upward motion if blue is already held. Do not pass a stale open-loop waypoint schedule as ground truth. Any allowable deviation from these paths must be established experimentally.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
