# Known-tool acquisition and geometric execution evidence

Acquire the known spatula and execute explicit geometric objectives safely, without learning robot connecting motion or claiming autonomous contact strategy.

## Dataset rationale

Keep a separate executor-only prefix through an inspected held-tool handoff in every episode. These are responsibility cuts, not automatically detected contact times. Approach, alignment, closure, lift and early connecting motion are retained together to calibrate and verify the fixed grasp recipe and its outcome. No learned policy or control-loss rows belong to this group.

## Component responsibility: supplied_executor

Purpose and binding: geometry_execution is a designed adapter around the user-declared generic planner/tracker, with acquire_tool, connect, track_local, hold/cancel and retreat operations. None is a verified Flip egg API. Inputs are current q[7], dq[7], calibrated T_base_flange/T_base_ee, scene collision geometry in base metres, covariance/validity, tool mesh/landmarks, T_ee_blade after grasp, joint/workspace limits, speed/acceleration limits, an explicit SE(3) goal or short blade path, and permitted contact pairs. It returns request ID, status, actual state, tracking error, constraint violations and timestamps. Planning failure must return without silently changing the physical goal. For acquire_tool the Agent supplies a template T_tool_ee_grasp, approach axis, open width, calibrated closing setting and lift clearance. Estimate the template offline from confirmed pre-lift closure poses in the retained prefixes, transformed by annotated/fitted initial tool poses; use a robust fixed known-tool recipe, not a learned pickup policy. Verify it physically before deployment. Plan to a collision-free pregrasp, approach along its specified axis, command gripper closure through a separately verified I/O binding, lift only within validated limits, and test visible tool co-motion and stable attachment. Finite gripper feedback is necessary but insufficient; images and attachment residuals must agree. A failed test returns tool_not_retained and requires a fresh observation/Agent decision, not blind repeated closure. The prefixes sometimes include early free-space transfer; no precise recorded endpoint is replayed online. Handoff is any verified lifted-tool state within the observed approach envelope. For connect/retreat, the Agent supplies reachable geometry, with held-object/egg support constraints where applicable; only empty-tool free-space retreat is presumed safe. For track_local, convert the policy's blade target through T_ee_blade and the recorded/calibrated flange-to-EE transform, then solve and track feasible joint motion. Free-space paths may be planned; in intended contact the adapter must preserve the local path and timing or reject it, never route around the egg and report success. Required additional contact binding: bounded compliant Cartesian tracking or an equivalently validated force-limited implementation, explicit allowable tool-pan/egg contacts, pan-rim collision protection, measured tracking-error limits, external-torque baseline/stop logic and a workspace interlock. Generic planning alone does not supply these. Recorded impedance settings are clues, not verified deployable gains or safety limits. Determine safe bounds by integration testing; do not copy 25 N/6 Nm or teleop speed values as approved limits. Commands expire at 0.10 s; issue bounded stopping/hold on stale observations or interruption. Do not assume hold is safe under arbitrary contact: use the installed safety controller's stop semantics. Motion completion reports geometry only. Tool slip, significant pan displacement, unknown support, contact limit or infeasibility triggers reobservation and Agent arbitration. No automatic successful-flip status.

### exec_01

Source: `episode_2026091916095801` [0, 601). Supervision kind: executor_only; bounds: None, None; decision indices: [].

**Boundary evidence indices**

[0, 600]

**Condition evidence indices**

[0, 600]

**Decision indices**

[]

**Deployment condition source**

Agent-selected known spatula on support; online fitted tool pose and empty gripper, followed by visible retained-tool test.

**Label derivation**

No policy labels. Retain original state/command/media chain for grasp-template fitting, gripper calibration and executor outcome checks.

**Merge check**

Do not merge executor-only acquisition with the learned interaction loss.

**Objective**

Acquire and lift the spatula with explicit geometric goals.

**Rationale**

Tool rests on a small dish initially; at 600 it is lifted and held, before egg insertion.

**Split check**

Keep approach, close and visible lift outcome in one execution sequence.

**Supervision exclusions**

[]

**Supervision kind**

executor_only

**Training condition**

Execution evidence only; no control loss.

**Uncertainty**

Exact grasp contact and physical width calibration are unverified.

### exec_02

Source: `episode_2026091916151401` [0, 601). Supervision kind: executor_only; bounds: None, None; decision indices: [].

**Boundary evidence indices**

[0, 600]

**Condition evidence indices**

[0, 600]

**Decision indices**

[]

**Deployment condition source**

Known-tool pose, empty-hand state and retention test.

**Label derivation**

Original execution evidence; no policy labels.

**Merge check**

Acquisition is executor-owned, not interaction imitation.

**Objective**

Acquire tool and connect to a lifted observation state.

**Rationale**

At 600 the spatula is held above the pan region.

**Split check**

Preserve the complete pickup outcome chain.

**Supervision exclusions**

[]

**Supervision kind**

executor_only

**Training condition**

No policy loss.

**Uncertainty**

Visual retention inference is not contact ground truth.

### exec_03

Source: `episode_2026091916165401` [0, 601). Supervision kind: executor_only; bounds: None, None; decision indices: [].

**Boundary evidence indices**

[0, 600]

**Condition evidence indices**

[0, 600]

**Decision indices**

[]

**Deployment condition source**

Known-tool pose and measured/visual retention.

**Label derivation**

Executor and grasp-template evidence only.

**Merge check**

Keep independent of the held-tool control objective.

**Objective**

Acquire and lift tool.

**Rationale**

Tool held at 600; insertion adjustments occur later.

**Split check**

No benefit from separate approach and closure datasets.

**Supervision exclusions**

[]

**Supervision kind**

executor_only

**Training condition**

No policy loss.

**Uncertainty**

Width about 0.034 m does not prove force or secure grasp.

### exec_04

Source: `episode_2026091916183001` [0, 401). Supervision kind: executor_only; bounds: None, None; decision indices: [].

**Boundary evidence indices**

[0, 400]

**Condition evidence indices**

[0, 400]

**Decision indices**

[]

**Deployment condition source**

Observed free tool and gripper; verified lift.

**Label derivation**

Execution evidence only.

**Merge check**

Distinct from interaction supervision.

**Objective**

Acquire spatula.

**Rationale**

Earlier pickup than other episodes; 400 is already visibly lifted.

**Split check**

Keep the faster pickup intact rather than using fixed-duration cuts.

**Supervision exclusions**

[]

**Supervision kind**

executor_only

**Training condition**

No policy loss.

**Uncertainty**

Row 0 already contains a nonzero recorded command; not a fabricated reset.

### exec_05

Source: `episode_2026091916200101` [0, 651). Supervision kind: executor_only; bounds: None, None; decision indices: [].

**Boundary evidence indices**

[0, 650]

**Condition evidence indices**

[0, 650]

**Decision indices**

[]

**Deployment condition source**

Known tool geometry and online retention checks.

**Label derivation**

Execution evidence only.

**Merge check**

Later failed/adjusted insertion is learned-group evidence.

**Objective**

Acquire and lift tool.

**Rationale**

650 shows the held blade above the pan before prolonged adjustments.

**Split check**

Retain approach and lift context together.

**Supervision exclusions**

[]

**Supervision kind**

executor_only

**Training condition**

No policy loss.

**Uncertainty**

Endpoint is a handoff envelope example, not a certified collision-free pose.

### exec_06

Source: `episode_2026091916223301` [0, 601). Supervision kind: executor_only; bounds: None, None; decision indices: [].

**Boundary evidence indices**

[0, 600]

**Condition evidence indices**

[0, 600]

**Decision indices**

[]

**Deployment condition source**

Fitted tool pose, empty hand and retention test.

**Label derivation**

Execution evidence only.

**Merge check**

Do not train pickup velocities from this group.

**Objective**

Acquire and lift tool.

**Rationale**

Held tool at 600 with stable narrow gripper feedback.

**Split check**

Keep pickup chain intact.

**Supervision exclusions**

[]

**Supervision kind**

executor_only

**Training condition**

No policy loss.

**Uncertainty**

Retention requires visual co-motion, not width alone.

### exec_07

Source: `episode_2026091916244201` [0, 601). Supervision kind: executor_only; bounds: None, None; decision indices: [].

**Boundary evidence indices**

[0, 600]

**Condition evidence indices**

[0, 600]

**Decision indices**

[]

**Deployment condition source**

Agent object selection and calibrated grasp recipe.

**Label derivation**

Execution evidence only.

**Merge check**

Separate acquisition and interaction responsibilities.

**Objective**

Acquire and lift tool.

**Rationale**

Changed pan/egg placement does not require a separate pickup policy.

**Split check**

Keep through visible lift.

**Supervision exclusions**

[]

**Supervision kind**

executor_only

**Training condition**

No policy loss.

**Uncertainty**

No verified grasp success annotation is supplied.

### exec_08

Source: `episode_2026091916283401` [0, 401). Supervision kind: executor_only; bounds: None, None; decision indices: [].

**Boundary evidence indices**

[0, 400]

**Condition evidence indices**

[0, 400]

**Decision indices**

[]

**Deployment condition source**

Known tool geometry and visual lift confirmation.

**Label derivation**

Execution evidence only.

**Merge check**

Acquisition is not a learned motor skill here.

**Objective**

Acquire and lift tool.

**Rationale**

400 shows a lifted tool still near its rest, giving early handoff variation.

**Split check**

Keep before egg approach.

**Supervision exclusions**

[]

**Supervision kind**

executor_only

**Training condition**

No policy loss.

**Uncertainty**

Do not interpret subsequent egg appearance as verified material identity.

### exec_09

Source: `episode_2026091916295601` [0, 401). Supervision kind: executor_only; bounds: None, None; decision indices: [].

**Boundary evidence indices**

[0, 400]

**Condition evidence indices**

[0, 400]

**Decision indices**

[]

**Deployment condition source**

Known tool selection and retained-tool test.

**Label derivation**

Execution evidence only.

**Merge check**

Long later manipulation is not assigned to autonomous planner competence.

**Objective**

Acquire and lift tool.

**Rationale**

400 is a free lifted-tool state before extended interaction.

**Split check**

Keep pickup intact; later repeated adjustments stay together in the learned sequence.

**Supervision exclusions**

[]

**Supervision kind**

executor_only

**Training condition**

No policy loss.

**Uncertainty**

No exact contact boundary inferred from duration.

### exec_10

Source: `episode_2026091916320101` [0, 1001). Supervision kind: executor_only; bounds: None, None; decision indices: [].

**Boundary evidence indices**

[0, 1000]

**Condition evidence indices**

[0, 1000]

**Decision indices**

[]

**Deployment condition source**

Measured tool pose and lifted-tool validation.

**Label derivation**

Execution evidence only.

**Merge check**

Keep slow acquisition separate from learned contact decisions.

**Objective**

Acquire and connect with held tool.

**Rationale**

600 is still at the rest; 1000 is visibly lifted above the pan region.

**Split check**

Long acquisition timing is not a policy clock.

**Supervision exclusions**

[]

**Supervision kind**

executor_only

**Training condition**

No policy loss.

**Uncertainty**

Later human intervention is not executor success evidence.

### exec_11

Source: `episode_2026091916335001` [0, 601). Supervision kind: executor_only; bounds: None, None; decision indices: [].

**Boundary evidence indices**

[0, 600]

**Condition evidence indices**

[0, 600]

**Decision indices**

[]

**Deployment condition source**

Known-tool geometry and retention checks.

**Label derivation**

Execution evidence only.

**Merge check**

Later local adjustments belong to learned interaction.

**Objective**

Acquire tool.

**Rationale**

600 shows tool lifted from its dish.

**Split check**

Preserve pickup and lift outcome.

**Supervision exclusions**

[]

**Supervision kind**

executor_only

**Training condition**

No policy loss.

**Uncertainty**

Exact attachment must be fitted, not assumed from TCP metadata.

### exec_12

Source: `episode_2026091916411901` [0, 601). Supervision kind: executor_only; bounds: None, None; decision indices: [].

**Boundary evidence indices**

[0, 600]

**Condition evidence indices**

[0, 600]

**Decision indices**

[]

**Deployment condition source**

Tool geometry, Agent selection and online retention.

**Label derivation**

Execution evidence only.

**Merge check**

New pan heading does not create a new pickup policy.

**Objective**

Acquire and lift tool.

**Rationale**

Front-left pan handle layout; held tool observed at 600.

**Split check**

Keep the pickup observation chain.

**Supervision exclusions**

[]

**Supervision kind**

executor_only

**Training condition**

No policy loss.

**Uncertainty**

Final pan movement includes a human and is not robot recovery.

### exec_13

Source: `episode_2026091916445601` [0, 601). Supervision kind: executor_only; bounds: None, None; decision indices: [].

**Boundary evidence indices**

[0, 600]

**Condition evidence indices**

[0, 600]

**Decision indices**

[]

**Deployment condition source**

Known-tool pose and retention check.

**Label derivation**

Execution evidence only.

**Merge check**

Different pan location affects scene constraints, not policy count.

**Objective**

Acquire and lift tool.

**Rationale**

600 is a lifted spatula state before insertion.

**Split check**

Preserve approach/closure/lift.

**Supervision exclusions**

[]

**Supervision kind**

executor_only

**Training condition**

No policy loss.

**Uncertainty**

Calibrated tool and pan meshes remain required.

### exec_14

Source: `episode_2026091916463201` [0, 601). Supervision kind: executor_only; bounds: None, None; decision indices: [].

**Boundary evidence indices**

[0, 600]

**Condition evidence indices**

[0, 600]

**Decision indices**

[]

**Deployment condition source**

Online object fit and visually confirmed lift.

**Label derivation**

Execution evidence only; retain gripper command/feedback disagreement for adapter review.

**Merge check**

Do not learn gripper semantics from scalar commands.

**Objective**

Acquire and lift tool.

**Rationale**

600 shows a held tool; command gripper can differ from stable measured width.

**Split check**

Keep full pickup evidence to audit closing behavior.

**Supervision exclusions**

[]

**Supervision kind**

executor_only

**Training condition**

No policy loss.

**Uncertainty**

At 600 gripper command is 0.996078 while measured width remains about 0.03349 m; controller semantics require verification.

### exec_15

Source: `episode_2026091916480201` [0, 901). Supervision kind: executor_only; bounds: None, None; decision indices: [].

**Boundary evidence indices**

[0, 900]

**Condition evidence indices**

[0, 900]

**Decision indices**

[]

**Deployment condition source**

Known tool pose and retention test.

**Label derivation**

Execution evidence only.

**Merge check**

Do not use fixed 600-row pickup boundary.

**Objective**

Acquire and lift tool.

**Rationale**

600 remains near the rest; 900 is lifted and approaching the pan.

**Split check**

Keep the slower acquisition intact.

**Supervision exclusions**

[]

**Supervision kind**

executor_only

**Training condition**

No policy loss.

**Uncertainty**

Tool-rest contact timing is not annotated.

### exec_16

Source: `episode_2026091916505401` [0, 601). Supervision kind: executor_only; bounds: None, None; decision indices: [].

**Boundary evidence indices**

[0, 600]

**Condition evidence indices**

[0, 600]

**Decision indices**

[]

**Deployment condition source**

Tool recipe, observed pose and retention.

**Label derivation**

Execution evidence only.

**Merge check**

Planner handles connecting pickup motion.

**Objective**

Acquire and lift tool.

**Rationale**

600 shows lifted tool; pan is in a different central layout.

**Split check**

Keep through lift outcome.

**Supervision exclusions**

[]

**Supervision kind**

executor_only

**Training condition**

No policy loss.

**Uncertainty**

No collision-clearance certification follows from previews.

### exec_17

Source: `episode_2026091916522501` [0, 601). Supervision kind: executor_only; bounds: None, None; decision indices: [].

**Boundary evidence indices**

[0, 600]

**Condition evidence indices**

[0, 600]

**Decision indices**

[]

**Deployment condition source**

Fitted tool geometry and retention test.

**Label derivation**

Execution evidence only.

**Merge check**

Acquisition is deterministic geometry, not another learned policy.

**Objective**

Acquire and lift tool.

**Rationale**

Held spatula at 600 before pan interaction.

**Split check**

Keep acquisition chain intact.

**Supervision exclusions**

[]

**Supervision kind**

executor_only

**Training condition**

No policy loss.

**Uncertainty**

Physical grip force is unobserved.

### exec_18

Source: `episode_2026091916534901` [0, 1001). Supervision kind: executor_only; bounds: None, None; decision indices: [].

**Boundary evidence indices**

[0, 1000]

**Condition evidence indices**

[0, 1000]

**Decision indices**

[]

**Deployment condition source**

Known tool geometry and retained-tool check.

**Label derivation**

Execution evidence only.

**Merge check**

Initial idle time does not become learned task timing.

**Objective**

Acquire and lift tool.

**Rationale**

400 still shows no gripper at the tool; 1000 is lifted. This is not a fixed-duration stage.

**Split check**

Keep initial idle, approach and lift as context/execution evidence.

**Supervision exclusions**

[]

**Supervision kind**

executor_only

**Training condition**

No policy loss.

**Uncertainty**

Camera device timestamps later reset; retain pairings rather than shifting samples.

### exec_19

Source: `episode_2026091916553801` [0, 401). Supervision kind: executor_only; bounds: None, None; decision indices: [].

**Boundary evidence indices**

[0, 400]

**Condition evidence indices**

[0, 400]

**Decision indices**

[]

**Deployment condition source**

Tool recipe and visible lift/attachment confirmation.

**Label derivation**

Execution evidence only.

**Merge check**

Release dynamics later are learned local-motion evidence.

**Objective**

Acquire and start lifting tool.

**Rationale**

400 has narrow stable width and a visibly raised tool with upward motion.

**Split check**

Do not require a fixed lift height at this responsibility handoff.

**Supervision exclusions**

[]

**Supervision kind**

executor_only

**Training condition**

No policy loss.

**Uncertainty**

Runtime must finish a safe retained-tool test even if the demonstration handoff is early.

### exec_20

Source: `episode_2026091916583301` [0, 801). Supervision kind: executor_only; bounds: None, None; decision indices: [].

**Boundary evidence indices**

[0, 800]

**Condition evidence indices**

[0, 800]

**Decision indices**

[]

**Deployment condition source**

Observed tool pose and attachment validation.

**Label derivation**

Execution evidence only.

**Merge check**

Held-tool interaction begins after this acquisition envelope.

**Objective**

Acquire and lift tool.

**Rationale**

600 is still near the rest; 800 clearly shows tool held above the pan approach area.

**Split check**

Keep variable acquisition duration instead of inheriting another episode's boundary.

**Supervision exclusions**

[]

**Supervision kind**

executor_only

**Training condition**

No policy loss.

**Uncertainty**

Tool attachment and gripper calibration need deployment validation.


All scientific text above is unchanged Runtime API output. Rendering and slicing are developer-owned. No policy code or training exists in this stage.
