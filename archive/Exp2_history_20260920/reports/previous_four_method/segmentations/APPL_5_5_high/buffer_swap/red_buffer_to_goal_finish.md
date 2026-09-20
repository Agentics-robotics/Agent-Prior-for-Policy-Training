# red_buffer_to_goal_finish

After blue is placed at blue_goal, acquire the buffered red block, place it at red_goal, and retain final release/retreat context until the swap success predicates hold.

## Segmentation

Each segment starts at action index 700, well before red reacquisition, when blue has just been placed and the robot is high/open near the blue-goal side. It includes the full expanded transition through movement to the buffered red block, gripper closing/lifting, transport to red_goal, release, final settling, and retreat. Stops are each trajectory's action count so the final observations used by the task contract and release/retreat context are retained. The grouped trajectories share the final learning problem of completing the swap once blue is already at goal.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo10100`: [700, 1064)
- `demo10101`: [700, 1064)
- `demo10102`: [700, 1060)
- `demo10103`: [700, 1065)
- `demo10104`: [700, 1057)
- `demo10105`: [700, 1056)
- `demo10106`: [700, 1059)
- `demo10107`: [700, 1071)
- `demo10108`: [700, 1058)
- `demo10109`: [700, 1063)
- `demo10110`: [700, 1064)
- `demo10111`: [700, 1061)

## Heuristic 1

Active-red final-transfer prior: condition the final policy on red-relative and red-goal-relative geometry so it learns reacquisition from the buffer and placement at red_goal independent of absolute timing.

**Evidence inspected by the API**

- `demo10100` observations: 700, 780, 820, 860, 900, 950, 1000, 1064
- `demo10103` observations: 700, 860, 1065
- `demo10107` observations: 700, 860, 1071

**Interpretation**

At demo10100 index 700 blue is already placed and red is buffered; indices 780-860 show approach/close/lift of red from the buffer; indices 900-1000 show red carried to and lowered at the upper goal; final obs 1064 has both blocks in their goals and the TCP retreated. Demo10103 and demo10107 show the same endpoint structure. The learning problem is reacquiring a small buffered object and placing it accurately while ignoring the now-completed blue object. This prior assumes active-object/goal relative coordinates improve transfer from buffer pose variation. It differs from the terminal predicate prior by emphasizing continuous manipulation geometry.

**Applicability**

Transfers to final buffer-swap phases where blue has already been placed at blue_goal and red is available in the central buffer. It assumes the active object is red, a top-down pinch can reacquire it, and red_goal remains the upper marked region in world coordinates.

**Implications for a future training/inference pipeline**

Use an active-red goal-frame DP. Causal input features: red-relative [tcp_xyz-red_xyz, red_goal-red_xyz, blue_pose-red_xyz], goal-relative [red_pose-red_goal, blue_pose-blue_goal], qpos/qvel/finger aperture. The denoiser outputs absolute Panda joint targets plus gripper command, but intermediate latent actions are represented as desired TCP displacement in the red frame and then decoded through a learned joint-action head. Training includes action diffusion loss and auxiliary predictions of red future lift height and final red_goal error, with future observations used only as labels. Deployment selects this policy when blue_at_goal and red_buffered are observed.

**Assumptions and limitations**

This prior relies on blue already being correctly placed; the dataset lacks examples where final red placement must compensate for blue errors. It also assumes the central red buffer pose is consistent. Falsification: train without active-red frame and compare red-goal success under small buffer pose perturbations. Dataset cannot establish recovery if red is missing from the buffer or if blue blocks the path.

### Handoff interface

**Entry conditions**

Measured entry support begins at index 700 with blue_pose at blue_goal, red_pose in the central buffer around [-0.181,0,0.02], qpos fingers open near 0.04 m, and tcp_pose high near [-0.36,-0.20,0.30] over the blue-goal side. The overlap also supports later entry with TCP en route to red or red already lifted by index 860.

**Exit conditions**

Final useful end state is task success with red_pose at red_goal and blue_pose at blue_goal, plus the demonstrated release/retreat context. End observations show red near [-0.351,0.195,0.02], blue near [-0.352,-0.200,0.02], gripper open, and TCP retreated to z about 0.30 m over the red goal.

**Failure signatures**

Blue is not at blue_goal on entry, red not in central buffer, red does not lift with the gripper after closure, red is placed outside red_goal containment, or the TCP disturbs the already placed blue during retreat. These indicate premature final policy transfer or failed reacquisition.

**Overlap role**

The [700,860) overlap is the predecessor-to-successor handoff: it begins with blue release/retreat and includes movement to red, low approach, finger closure, and initial red lift. This final skill learns to take over before the predecessor has finished that transition.

**Successor readiness**

There is no downstream manipulation skill; the successor is task completion/termination. A completion monitor needs red_at_goal AND blue_at_goal from current poses and should tolerate the demonstrated gripper retreat/open state. It should not require gripper release by contract, although release/retreat are in the data.


## Heuristic 2

Task-predicate termination prior: train the final policy with explicit red_at_goal and blue_at_goal predictions so it can stop or retreat once the contract is satisfied rather than following a fixed-duration demonstration.

**Evidence inspected by the API**

- `demo10100` observations: 950, 1000, 1064
- `demo10102` observations: 860, 1060
- `demo10111` observations: 860, 1061

**Interpretation**

The final segment contains both partial and complete states: demo10100 index 950 has red near the goal but elevated at z 0.131, index 1000 has red at z 0.02 near [-0.351,0.196] and blue at its goal, and final index 1064 retains success with open gripper and TCP high. Demo10102 and demo10111 final observations show the same. The challenge is knowing when the task, not just a motion phase, is done. This prior assumes explicit success-predicate learning will reduce over-motion and improve handoff to termination. It differs from active-red geometry by representing symbolic completion and stopping.

**Applicability**

Useful for the last skill where task completion can be evaluated directly from red_pose, blue_pose, red_goal, and blue_goal. It transfers when the completion contract remains red_at_goal AND blue_at_goal with no drawer and no required sustained hold/release.

**Implications for a future training/inference pipeline**

Augment DP with differentiable task-predicate heads. Causal inputs are red_pose, blue_pose, red_goal, blue_goal, tcp_pose, qpos/qvel. Compute training labels red_at_goal, blue_at_goal from current/future observations using the contract: xy containment half-width 0.06 m, object half 0.02 m, z tolerance 0.011 m. Objective = action diffusion loss + BCE(red_at_goal) + BCE(blue_at_goal) + penalty for predicted actions after success (small joint velocity/open gripper target in final retreat states). Deployment evaluates predicate heads causally and can stop or hand control to a monitor when both high; labels from futures are not used at inference.

**Assumptions and limitations**

All demonstrations succeed, so negative examples for the predicate are mainly earlier in the same trajectories; there are no adversarial near-boundary failures. An ablation is a policy without predicate auxiliary loss but same data; if it terminates and places equally well under timing variation, the predicate prior is not beneficial. Dataset cannot prove robustness to external perturbations after success.

### Handoff interface

**Entry conditions**

Entry requires the predecessor overlap state with blue at blue_goal and red available/being lifted. This prior additionally checks that blue_at_goal is already true or nearly true before pursuing final completion; otherwise its predicate representation is outside demonstrated support.

**Exit conditions**

Exit is a learned stop/terminate condition when red_at_goal AND blue_at_goal are predicted from current poses. Demonstrated final observations have qpos fingers open near 0.04 m and tcp_pose z about 0.30 m after release/retreat, but the task contract does not require release; a controller may terminate earlier if predicates are confidently true.

**Failure signatures**

Predicate head says success while red z is high/in gripper, red outside the full rotated goal containment, or blue displaced. Another failure is continuing to move after both predicates are true and knocking a block out of goal.

**Overlap role**

The incoming overlap gives this predicate prior examples before final success: index 860 has blue_at_goal true but red not yet at red_goal, so the policy can distinguish partial from full completion. There is no outgoing overlap, but release/retreat to the final observation is retained.

**Successor readiness**

The successor is the task-level completion monitor. It needs current red_pose and blue_pose inside goal tolerances for at least the required one observation. If a separate retreat/settle controller is used, it can take over when predicates are true and qpos fingers are open or opening; such post-success recovery beyond demonstrated retreat is unseen.


## Heuristic 3

Waypoint-chain transport prior: constrain the final policy to infer progress through lift, high carry, descend, release, and retreat waypoints, improving long-horizon structure and reducing premature release or dragging.

**Evidence inspected by the API**

- `demo10100` observations: 700, 820, 860, 900, 950, 1000, 1064
- `demo10103` observations: 700, 860, 1065
- `demo10105` observations: 700, 860, 1056

**Interpretation**

Demo10100 illustrates a clean waypoint sequence: high/open at 700, low contact near 820, lifted red by 860, carried toward goal at 900, lowered near goal at 950, released by 1000, and retreated by 1064. Demo10103 and demo10105 final endpoints confirm similar release/retreat. The learning gap is long-horizon transport with accurate timing of descend/open/retreat. This prior assumes the motion is well described by a few task-space waypoints, unlike the predicate prior which focuses on termination, or the active-red prior which is a dense reactive representation.

**Applicability**

Transfers when final red motion can be approximated by a small number of geometric waypoints: lift from buffer, translate high above the table, descend over red_goal, open/release, and retreat. It requires similar obstacle-free workspace and top-down orientation.

**Implications for a future training/inference pipeline**

Implement a hierarchical waypoint-conditioned DP. A training preprocessor extracts future waypoints from demonstrations: p0=current, p1=red grasp/lift point when red z>0.05, p2=high carry midpoint, p3=above red_goal, p4=release/retreat. These labels use future observations only for supervision. Causal encoder predicts progress s_t and residual waypoints W_t from current qpos/qvel/tcp_pose/red_pose/blue_pose/goals. Low-level diffusion decoder conditions on (s_t,W_t) and outputs action chunks of absolute joint targets/gripper. Deployment repeatedly replans W_t from current observation and executes DP chunks; the prior is the waypoint bottleneck and progress monotonicity loss.

**Assumptions and limitations**

A low-dimensional waypoint prior may underfit fine joint-space details needed near contact and may fail with obstacles or altered kinematics. Compare against a standard chunk-based DP; if the waypoint bottleneck lowers grasp/place accuracy, the abstraction is too restrictive. The dataset cannot establish collision avoidance around new obstacles because the workspace is free.

### Handoff interface

**Entry conditions**

Entry can be early in the overlap at index 700 with open gripper/high TCP after blue release, or later with red already lifted at index 860. The waypoint policy assumes it can infer progress along the waypoint chain from red z, TCP pose, and finger aperture.

**Exit conditions**

End state is red released on red_goal and the TCP retreated to a safe height; final observations show tcp_pose z about 0.300 m, qpos[7:9] open, red/blue both at table height in goals. The useful end includes the demonstrated settling/retreat rather than only first contact with the goal.

**Failure signatures**

The policy skips the lift waypoint and drags red along the table, descends before reaching red_goal, opens the gripper while red z is high, or fails to retreat and contacts a placed block. These indicate the waypoint abstraction is mis-sequenced.

**Overlap role**

The incoming [700,860) overlap is the approach/lift prefix of the waypoint chain. Because both skill 2 and skill 3 include it, the waypoint progress variable can be initialized from open/high or already-lifted states rather than one exact cut.

**Successor readiness**

No manipulation successor is needed. A completion/monitor policy can take over after the release and retreat waypoint or immediately after predicates are true if allowed. The demonstrated final retreat gives support for safe post-release joint targets but not for alternate retreat directions.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
