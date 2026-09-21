# red_top_regrasp_lift

Reorient after drawer opening, top-grasp the red block and lift it clear for transport.

## Segmentation

This dataset begins in the S1 release/retreat overlap at index 240, when the drawer is open and red is still on the drawer surface, includes the high reorientation around index 330, top-down descent and closure near index 455, and ends after the red block is clearly lifted and moving toward the pad around index 530. This groups the learning problem of converting an opened-drawer red block into a carried object.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo1000`: [240, 530)
- `demo1001`: [240, 530)
- `demo1002`: [240, 530)
- `demo1003`: [240, 530)
- `demo1004`: [240, 530)
- `demo1005`: [240, 530)
- `demo1006`: [240, 530)
- `demo1007`: [240, 530)
- `demo1008`: [240, 530)
- `demo1009`: [240, 530)
- `demo1010`: [240, 530)
- `demo1011`: [240, 530)

## Heuristic 1

Red-centered top-grasp affordance: learn descent, closure and lift in a frame anchored on red_pose, improving tolerance to red placement left by the drawer-opening skill.

**Evidence inspected by the API**

- `demo1000` observations: 240, 330, 455, 530
- `demo1002` observations: 240, 330, 455, 530
- `demo1011` observations: 240, 330, 455, 530

**Interpretation**

The overlap starts while the first skill is still retreating and continues through top contact. Across demos, red remains at variable x/y but the top grasp geometry relative to red is stable. This prior targets the learning gap between global reorientation and local grasp closure.

**Applicability**

Transfers when the drawer is already open, red rests on the open drawer surface/front region at z about 0.063 m, and the gripper can approach vertically/top-down. Requires red_pose observation and enough clearance above the open drawer.

**Implications for a future training/inference pipeline**

Use causal qpos,qvel,tcp_pose,red_pose,drawer_position. Construct a red-centered vertical grasp frame with features [tcp_xy-red_xy, tcp_z-red_z, yaw/quat difference, finger gap]. A diffusion decoder is conditioned on this frame. Training labels for grasp success/attachment are computed from future red z increase and persistent tcp-red offset; inference only uses current observations.

**Assumptions and limitations**

The demonstrations do not show retrying a missed top grasp or avoiding collisions with a partially closed drawer. Ablate top-down affordance features against raw state DP; benefit is falsified if held-out red y variation is not improved.

### Handoff interface

**Entry conditions**

Measured starts include index 240: drawer_position around 0.300 m, gripper opening near 0.039 m, red_pose x about -0.195 to -0.210, z 0.063 m, tcp just after side release near the red/front of drawer. The policy also supports later overlap entries around index 330 with tcp at z about 0.37 m and gripper open.

**Exit conditions**

Exit when top grasp is established and red is being lifted/transported: at index 530 red_pose z is about 0.307-0.322 m, fingers closed near 0.018 m, tcp is co-moving with red above the drawer/pad path.

**Failure signatures**

tcp descends beside rather than over red, gripper closes before reaching low z, red z does not increase after close, or red_pose separates from tcp while fingers are closed.

**Overlap role**

[455,530) is shared with red place/transport so both skills learn the fragile attachment transition and initial acceleration of the carried block.

**Successor readiness**

red_transport_place_pad needs red attached in a top grasp, lifted clear of drawer lips (red z > about 0.25 m), drawer remains open, and the arm is moving toward the outside pad rather than still searching.


## Heuristic 2

Open-drawer clearance prior: maintain a safe vertical approach/lift envelope around the opened drawer while regrasping red.

**Evidence inspected by the API**

- `demo1001` observations: 240, 330, 455, 530
- `demo1004` observations: 240, 330, 455, 530
- `demo1008` observations: 240, 330, 455, 530

**Interpretation**

The robot rises high after release, then descends vertically to red and lifts again before traversing. This repeated clearance pattern likely protects against the open drawer lip and red/drawer contact; encoding it prevents a learner from shortcutting between poses.

**Applicability**

Applies when the drawer is open and forms an obstacle/clearance constraint around the red block. Assumes the top of the drawer and robot kinematics are similar, with tcp_pose z available in world metres.

**Implications for a future training/inference pipeline**

Candidate uses a learned cost-map/clearance auxiliary from state, not images: input [tcp_pose, red_pose, drawer_position] defines approximate drawer obstacle frame with drawer_center=[0.19-drawer_position,0,0.035]. Train decoder with L_diff plus hinge losses on tcp_z and red_z clearance during approach/lift, labels from demonstration positions. At deployment, compute clearance features causally and condition diffusion sampling or rescore samples by predicted collision risk.

**Assumptions and limitations**

No negative collision data exists; a learned clearance field is inferred only from successful paths. Falsify via a comparison without clearance auxiliary losses on perturbations near the drawer rim; if both collide equally, the prior adds no value.

### Handoff interface

**Entry conditions**

Can take over during S1/S2 overlap once drawer_position is >0.26 m and gripper is open. tcp may be low near x -0.44,z 0.128 at index 240 or high/reoriented near z 0.37 at index 330.

**Exit conditions**

Red lifted above clutter, typically red_pose z >0.30 at index 530, with closed fingers and tcp z approximately red z + 0.001 m; path has cleared the drawer lip.

**Failure signatures**

tcp path cuts through the drawer rim, red is dragged along the drawer surface after close, or lift height stagnates below 0.15 m.

**Overlap role**

The shared [240,330) region trains safe withdrawal from the drawer-open pose; [455,530) trains both this skill and successor to respect clearance while red becomes carried.

**Successor readiness**

Successor should see red high enough for lateral travel, not still inside the drawer frame; drawer_position must still exceed 0.26 m.


## Heuristic 3

Attachment-switch prior: explicitly represent free/contact/carry modes so the policy lifts only after a stable red grasp is likely.

**Evidence inspected by the API**

- `demo1003` observations: 455, 530
- `demo1005` observations: 455, 530
- `demo1010` observations: 455, 530

**Interpretation**

A key ambiguity is when to stop grasping and start lifting. Demonstrations show red is static at index 455, then high and co-moving by index 530. An attachment latent gives the policy a manipulable success signal not present in raw joint states.

**Applicability**

Useful for top grasps where object attachment can be recognized by stable tcp-red offset after finger closure. Depends on gripper aperture qpos and red_pose being observable.

**Implications for a future training/inference pipeline**

Train a latent binary attachment model b_t with labels from future red_pose displacement when qpos fingers <0.025 m and ||(tcp-red)_{t:t+H}|| variance is low. Inputs are causal qpos/qvel/tcp_pose/red_pose. Decoder D_theta(a|o,b) uses b to switch from descent/closure to lift/transport. At inference b is filtered online from current history, no future data.

**Assumptions and limitations**

Attachment is inferred, not force-sensed; future red motion labels may confound contact with incidental pushing. Test against a no-attachment-latent model; improvement should appear mainly in gripper close/lift timing.

### Handoff interface

**Entry conditions**

Supported entry includes open fingers descending near red at index 455 with red z about 0.063 and tcp z about 0.064. The policy may also enter from earlier retreat states in [240,330).

**Exit conditions**

End with latent attachment probability high: fingers about 0.018 m, tcp and red z co-increase, red_pose z around 0.307 by index 530.

**Failure signatures**

Attachment head remains low after close, tcp moves but red z stays near 0.063, or gripper gap is inconsistent/asymmetric indicating a poor pinch.

**Overlap role**

The [455,530) overlap makes attachment establishment a shared responsibility; the successor can begin when attachment is recognized rather than at a fixed timestep.

**Successor readiness**

Successor needs a carried red block with stable tcp-red offset and clearance; if attachment probability is low it should not take over.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
