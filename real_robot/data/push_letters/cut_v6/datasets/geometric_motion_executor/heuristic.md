# Supplied robot motion generation, guarded tracking and execution evidence

Generate and track safe robot motion between and through explicit bounded interaction objectives, return feedback/failures, and preserve all demonstrated motion evidence without training another control policy.

## Dataset rationale

Retain each original episode whole as an independent executor/context/outcome trace with zero policy supervision. This explicitly assigns waits, travel, lift/descent, robot posture changes, all actual contact motions, recontacts and final observations without learning trajectories the supplied components can generate. Full-sequence retention also supports later geometry/annotation review and guards against selective omission of awkward motions; it is not a claim that every recorded motion is safe to replay.

## Component responsibility: supplied_executor

execute_geometry(request) is a proposed adapter to the user-declared motion planner and low-level controllers. Inputs: current q/dq and T_base_flange/T_base_ee with timestamps; calibrated robot/tool collision geometry including the slender rigid installed pushing tool, validated T_ee_tool/contact surface, T_base_P and camera models; scene collision geometry with conservative uncertainty inflation; request mode prepare/push/withdraw/view/hold; a reachable geometric pose/path objective in explicit frames; allowed-contact patch and selected instance for push only; protected instances, tabletop/edge constraints, speed/acceleration/jerk/torque limits, timeout, interruption token and scene revision. Output: ticket with accepted/planning/running/arrived/stroke_complete/withdrawn or unreachable/collision_blocked/stale_scene/contact_lost/guard_stop/interrupted statuses, measured final state, tracking residual and reason. Planner owns joint path generation, collision checking and full connecting/reorientation motion, not spelling, contact choice or predicted letter dynamics. Adapter constructs standoff from boundary normal plus tool support geometry, safe hover from max obstacle height plus commissioned margin, and collision-checked descent; inner-loop contacts require validated opening clearance and an insertion path. During a stroke allow intended contact only with the selected actor patch, maintain contact-height/orientation constraints, track the local line, monitor scene/robot at bounded latency, and stop on unexpected motion, force/torque guard, contact loss or stale observations. Reobserve/replan after every short stroke. Contact-line tracking and guard calibration are integration obligations: the declaration of supplied motion generation does not verify compliance mode, wrench estimation, table model or a working controller adapter. If the chosen controller cannot safely implement the guarded primitive, this binding is a deployment blocker, not justification for cloning teleop dq. Measured tau_ext is not a calibrated force; commission model-based torque residual thresholds/tool loads separately. Interrupted execution cancels the ticket and invalidates precomputed continuation; Agent reobserves and requests a new objective. Completed travel never signals object or task success. View/withdraw objectives may be deterministic safe poses selected by Agent within collision constraints. Recorded commands are retained only for audit: metadata declares 20 Hz joint_velocity teleop, translation=base and rotation=ee, sampling30 Hz; verify actual units, w decoding, hold/lead timing, limits and controller frame conventions against deployment implementation before use. Null dq/target_pose are missing, never zeros; missing gripper feedback prevents a grasp contract. No robot execution or remote planner integration has been performed here.

### a_executor_trace

Source: `episode_2026091922002701` [0, 9015). Supervision kind: executor_only; bounds: None, None; decision indices: [].

**Boundary evidence indices**

[0, 9014]

**Condition evidence indices**

[0, 500, 800, 1800, 4800, 5100, 6900, 8500, 9014]

**Decision indices**

[]

**Deployment condition source**

High-level Agent or geometric policy supplies a live explicit robot objective, scene/tool model and constraints; the executor is not conditioned on a recorded index or inferred word.

**Label derivation**

No learned target or per-row imitation loss. Exact raw states/actions/media are execution/context/outcome evidence for adapter calibration checks, motion-domain audit, perception review and event/target derivation within learned cuts. Beginning null command dq is preserved. End is row9014, without fabricated terminal state.

**Merge check**

Do not concatenate with the other episode; independent initialization, elapsed time and physical arrangement. May be referenced by the learned cuts using original source keys only.

**Objective**

Retain complete evidence of motions the supplied executor must be able to synthesize/track from geometric objectives and of resulting object changes.

**Rationale**

Initial high tool pose and waits, low-table contacts, vertical/transverse reconnects, varied orientations and final held state are all assigned an execution role rather than densely learned.

**Split check**

No split needed for an executor-only responsibility trace; local learned target searches use their own bounded event cuts. Numeric motion bins did not define semantic cuts.

**Supervision exclusions**

[]

**Supervision kind**

executor_only

**Training condition**

Not a policy-training sequence. Used with original calibration/timing/missingness, no synthetic task-success or object labels.

**Uncertainty**

Recording complete but success unannotated. Calibration/execution binding and physical contact geometry remain unverified.

### b_executor_trace

Source: `episode_2026091922062101` [0, 7730). Supervision kind: executor_only; bounds: None, None; decision indices: [].

**Boundary evidence indices**

[0, 7729]

**Condition evidence indices**

[0, 300, 650, 1700, 3900, 4200, 5500, 6500, 7729]

**Decision indices**

[]

**Deployment condition source**

Live geometric objective from Agent/policy with current state, constraints and contact allowance, independent of this demonstrated order.

**Label derivation**

No policy targets. Preserve exact paired numeric records and original-media identities, including transient high joint rates near the legged-piece correction, startup missing action dq and terminal held observation. No interpretation of transient as a labeled failure and no7729+1 terminal observation.

**Merge check**

Keep episode identity and split grouping distinct from A. Reuse as executor evidence does not create independent samples for training or validation.

**Objective**

Retain alternate travel paths, inner-hole access, reorientation, reconnects, correction transients and outcomes in the second arrangement.

**Rationale**

Different initial layout and route expose the distinction between robot motion synthesis and selected object-interaction objectives. All observed behavior remains assigned even if later primitive-label validation rejects a sparse target.

**Split check**

Whole independent trace is appropriate for executor audit, not a reason to train an end-to-end controller. Event decisions are separately cut and sparse.

**Supervision exclusions**

[]

**Supervision kind**

executor_only

**Training condition**

Execution/context evidence only, no heuristics, no control-policy loss and no task-success target.

**Uncertainty**

Large joint-rate transient has no supplied cause label. Scene/force/pose ground truth, raw-depth scale/registration and deployable controller behavior are not verified.


All scientific text above is unchanged Runtime API output. Rendering and slicing are developer-owned. No policy code or training exists in this stage.
