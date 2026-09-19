# blue_approach_grasp_lift

Move from the red pad area to the blue block, top-grasp it and lift it clear of the table.

## Segmentation

This skill starts at index 650 while the robot is still retreating from the red placement, spans the global move toward blue, includes local descent/contact around index 810, and stops after blue is lifted by index 870. The broad start overlap supports entry before the predecessor's motion has settled; the stop overlap supports the successor taking over during initial blue lift.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo1000`: [650, 870)
- `demo1001`: [650, 870)
- `demo1002`: [650, 870)
- `demo1003`: [650, 870)
- `demo1004`: [650, 870)
- `demo1005`: [650, 870)
- `demo1006`: [650, 870)
- `demo1007`: [650, 870)
- `demo1008`: [650, 870)
- `demo1009`: [650, 870)
- `demo1010`: [650, 870)
- `demo1011`: [650, 870)

## Heuristic 1

Blue-relative top-grasp servo: approach, close and lift in a frame anchored on blue_pose while conditioning on red already released.

**Evidence inspected by the API**

- `demo1000` observations: 650, 760, 810, 870
- `demo1002` observations: 650, 760, 810, 870
- `demo1010` observations: 650, 760, 810, 870

**Interpretation**

After a global transit, the local grasp geometry is similar across demonstrations despite blue start variation. Anchoring on blue_pose addresses the generalization problem that the same joint target cannot exactly fit all blue locations.

**Applicability**

Transfers when blue starts on the table near world x -0.39 to -0.41, y 0.287-0.315, z 0.02 and can be top-grasped. Requires blue_pose observation and open gripper after red release.

**Implications for a future training/inference pipeline**

Causal inputs: qpos,qvel,tcp_pose,blue_pose,red_pose,drawer_position. Encode blue-centered features [tcp-blue pose, finger gap, red_on_pad current predicate]. Diffusion decoder outputs joint/gripper commands; train with auxiliary labels for blue attachment/lift from future blue z. No future labels at inference.

**Assumptions and limitations**

Does not demonstrate recovering if red is not on pad or if blue has been bumped. Compare blue-relative representation to absolute joint replay on held-out blue y/x; benefit should appear in approach and lift accuracy.

### Handoff interface

**Entry conditions**

Supported entry at index 650 has red on pad, gripper open near 0.04 m, tcp still over/near pad at z 0.20-0.26, drawer open. Later entry at 760 has tcp near the blue area but high (z ~0.27).

**Exit conditions**

Blue has been contacted, fingers are closing/closed and blue is lifted: by index 870 blue z is ~0.27-0.28 m in most demos, gripper gap ~0.018 m, tcp co-moving above blue.

**Failure signatures**

red not actually released at entry, gripper remains closed while approaching blue, tcp misses blue xy, or blue z does not increase after closure.

**Overlap role**

[650,760) shares red-retreat/global transit with predecessor; [810,870) shares blue contact/lift with transport successor.

**Successor readiness**

blue_transport_to_drawer needs blue attached or imminently attached, fingers closed, and tcp/blue lifted clear of the table for travel.


## Heuristic 2

Pad-to-blue transit waypoint prior: decompose the post-red retreat into high-clearance global motion followed by local blue approach.

**Evidence inspected by the API**

- `demo1001` observations: 650, 760, 810
- `demo1005` observations: 650, 760, 810
- `demo1008` observations: 650, 760, 810

**Interpretation**

The transition is long relative to contact phases and begins with the arm still leaving red. A waypoint/region prior prevents the learner from averaging between retreat and descent actions and supports multiple entry poses inside the [650,760) overlap.

**Applicability**

Useful for long transitions where the robot begins near the red pad and must move around the open drawer to the blue block. Assumes workspace layout fixed and drawer remains open.

**Implications for a future training/inference pipeline**

Implement a two-level policy: a recurrent waypoint latent w_t moves from pad-retreat to blue-approach. Inputs are causal full state; training labels for waypoint stage are inferred from tcp_pose regions (pad, midair, blue). DP decoder D(a|o,w). Add z-clearance auxiliary from current/future tcp_pose. Deployment filters w_t from current tcp region and red_on_pad predicate.

**Assumptions and limitations**

The learned global transit is tied to fixed table/drawer geometry and may not generalize to obstacles. Falsify by comparing with/without a waypoint/clearance latent; no benefit on starts from index 650 would reject it.

### Handoff interface

**Entry conditions**

Can start immediately after red release: index 650 states show gripper open, red z about 0.02 on pad, tcp not yet near blue. The policy includes retreat and global transit before local blue descent.

**Exit conditions**

Ready to hand off after the gripper has closed on/near blue and the object is lifting, or at least tcp is centered over blue with closing command in the shared [810,870) interval.

**Failure signatures**

Policy drives directly from pad to blue at low z, clipping table/drawer; red gets disturbed during retreat; or it arrives at blue with gripper not open.

**Overlap role**

The expanded overlap lets this skill learn to take control before predecessor has completed retreat, rather than requiring an exact arm pose at the blue side.

**Successor readiness**

Transport successor requires not just near-blue pose but a stable closed grasp and blue z increasing; otherwise this skill should continue through attachment.


## Heuristic 3

Blue attachment latent: represent the contact-to-carry switch explicitly to decide when blue transport can begin.

**Evidence inspected by the API**

- `demo1003` observations: 810, 870
- `demo1004` observations: 810, 870
- `demo1011` observations: 810, 870

**Interpretation**

The most important handoff variable is whether blue is actually grasped. The demos show transition from table z to high z between indices 810 and 870; encoding attachment avoids handing off on a mere near-pose.

**Applicability**

Applies when gripper aperture reliably indicates contact and blue can be lifted without complex reorientation. Requires blue_pose and finger qpos.

**Implications for a future training/inference pipeline**

Same attachment-switch mechanism as for red but using blue_pose. Labels b_t from future blue_z increase and stable tcp-blue transform while finger gap <0.025 m. Inputs qpos/qvel/tcp_pose/blue_pose. Decoder uses b_t to switch from centering/closing to lift. At inference b_t is filtered causally from recent observations.

**Assumptions and limitations**

No regrasp or slipped blue examples exist; attachment labels from future lift may be optimistic. Ablate attachment latent versus direct DP and test on slight initial blue offsets; expected benefit is better close/lift timing.

### Handoff interface

**Entry conditions**

Entry includes open gripper over/near blue at index 810; blue still on table z ~0.02; tcp z near 0.019-0.020 when closing starts in some demos.

**Exit conditions**

Exit with blue z raised to about 0.28 at index 870 or with stable closed fingers and low relative tcp-blue variance; gripper command closed/closing.

**Failure signatures**

fingers close with tcp not centered on blue, blue rotates/escapes, or gripper gap stays open while commanded closed.

**Overlap role**

[810,870) is intentionally assigned to both blue approach and blue transport so the contact-to-carry switch is learned by both candidates.

**Successor readiness**

Next policy needs attachment probability for blue high, blue off the table, and tcp positioned to begin travel toward the drawer cavity.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
