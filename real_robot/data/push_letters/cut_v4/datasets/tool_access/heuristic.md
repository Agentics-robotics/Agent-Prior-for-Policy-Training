# Tool access, disengagement and clearance positioning

Reach a caller-specified safe tool access/clearance pose without intentional piece displacement.

## Dataset rationale

Two long initial approaches and two irregular later transfers provide a common tool-waypoint learning problem. Buffers include the first object interaction only as context; parent object groups supervise contact and recontact. The tool-only calls deliberately do not need character identity.

### a_initial_access

Source: `episode_2026091922002701` [0, 615). Supervised interval: [0, 600).

**Boundary evidence indices**

[0, 599, 600, 614]

**Condition evidence indices**

[0, 599]

**Deployment condition source**

Caller supplies a reachable tool waypoint from the current selected piece's observed boundary and collision-checked approach plan.

**Label derivation**

Tool goal is recorded T_base_ee at599; auxiliary access-region referent is zigzag cutout below the rod in third599. D uses finite action.dq only; startup null dq is masked, finite v/w and scene history remain.

**Merge check**

Same tool endpoint/clearance objective as B initial and later transfers, despite longer startup waiting.

**Objective**

Approach the first zigzag piece from high initial pose without assigning a word.

**Rationale**

0 shows all pieces and high TCP;599/600 show rod poised over the zigzag, with base EE z about0.085m, preceding the push contact. Following15 rows are context only.

**Split check**

End before the object-moving cycle; retain startup pauses and approach under tool goal. Do not split the descent into fixed numerical bins.

**Supervision exclusions**

[]

**Training condition**

mode=approach; tool_goal=pose599; no semantic letter input; protect all physical pieces.

**Uncertainty**

Actual contact onset and true tip offset not measured; initial null joint commands are not zeros.

### b_initial_access

Source: `episode_2026091922062101` [0, 435). Supervised interval: [0, 420).

**Boundary evidence indices**

[0, 419, 420, 434]

**Condition evidence indices**

[0, 419]

**Deployment condition source**

Current target boundary and caller tool waypoint, not recorded scene/word.

**Label derivation**

Tool goal T_base_ee419; zigzag piece is front-right in third0 and beneath rod in wrist419. Null joint command components are masked under D.

**Merge check**

Same high-to-low access objective, different initial arrangement and approach path.

**Objective**

Bring tool from home-like start to access the front zigzag piece.

**Rationale**

Third419/420 is arm-occluded but paired wrist419/420 shows the piece and rod; state419 is downward command.434 follows contact-like torque increase and is context only.

**Split check**

Cut at access-to-object handoff; following context is not tool-only action supervision.

**Supervision exclusions**

[]

**Training condition**

mode=approach; tool_goal=pose419; current scene obstacle masks.

**Uncertainty**

Projected overlap is not verified force contact; wrist resolves target identity geometrically.

### a_transfer_to_l

Source: `episode_2026091922002701` [6135, 6355). Supervised interval: [6150, 6340).

**Boundary evidence indices**

[6135, 6150, 6199, 6200, 6339, 6354]

**Condition evidence indices**

[6150, 6339]

**Deployment condition source**

Agent chooses next access waypoint using current L-like boundary/obstacles.

**Label derivation**

Tool waypoint T_base_ee6339; no object goal; D original command labels. Protect the stationary D-like and R-like pieces along the route.

**Merge check**

Common transfer objective rather than D/L spelling-specific behavior.

**Objective**

Transfer away from D-like alignment toward the L-like piece, ending above its boundary.

**Rationale**

6150 shows departure near the right column;6339 shows rod above L-like piece without an observed relocation.15-row buffers bracket the corridor.

**Split check**

Reused subset spanning parent D-alignment/L-relocation handoff; not extend into L push at6400.

**Supervision exclusions**

[]

**Training condition**

mode=transfer; tool_goal=pose6339; protect all pieces; no intended object motion.

**Uncertainty**

Dense tracking must verify object stationarity; access loss is ineligible where actual displacement occurs, while parent object support remains.

### b_transfer_to_r

Source: `episode_2026091922062101` [2735, 2955). Supervised interval: [2750, 2940).

**Boundary evidence indices**

[2735, 2750, 2819, 2820, 2939, 2954]

**Condition evidence indices**

[2750, 2939]

**Deployment condition source**

Current scene and agent-selected next approach waypoint.

**Label derivation**

Tool waypoint T_base_ee2939; all pieces treated as stationary obstacles for this access example; original finite commands under D.

**Merge check**

Same depart/traverse/approach responsibility as A transfer, different direction and clutter.

**Objective**

Depart ring placement and approach R-like piece without moving either.

**Rationale**

2750 tip is released above ring;2939 is above R-like piece before contact at later3000. Parent goals explicitly condition their exit/access motions.

**Split check**

Use only visually supported free-transfer corridor, not entire R relocation.

**Supervision exclusions**

[]

**Training condition**

mode=transfer; tool_goal=pose2939; protect all pieces.

**Uncertainty**

No measured contact label; audit stationarity before using collision-free auxiliary targets.


## Heuristic 1

Learn a goal-relative, obstacle-aware tool-access policy with a phase/hysteresis prior and a kinematic command residual.

### Action decoding

Use shared D. Predict a seven-joint residual around a collision-checked, q-conditioned waypoint velocity proposal; supervise ORIGINAL finite action.dq and auxiliary recorded v/w separately. Output only the resulting bounded joint-velocity proposal to the verified controller adapter. Nominal path planning is a prior, not a replacement label. Missing dq rows cannot train the residual as zero. Treat all translation/rotation/controller units as commissioning gates.

### Applicability

Tool free or safely disengageable, reachable caller T_base_ee waypoint, reliable obstacle/tool tracking and commissioned robot collision geometry. Includes approaching a boundary/concavity from above, not deliberately displacing a piece. With no clear corridor return BLOCKED rather than sweeping through letters.

### Augmentation

Photometric illumination/texture and rod/scene occlusion augmentations with visibility masks; passive re-expression of all geometric positions, goal, tool pose and vector labels in a common planar frame. Do not move raw q or rotate joint command vectors as if Cartesian. Real spatial transformations require checked IK/collision recomputation and are not assumed available. Do not add an obstacle to an existing action trajectory without checking its swept path.

### Evidence

- `episode_2026091922002701`: [0, 599, 600, 6150, 6339]
- `episode_2026091922062101`: [0, 419, 420, 2750, 2939]
### Goal conditioning

Tool goal T_base_ee, clearance/contact-access mode, optional selected instance and intended boundary region, protected-instance set and safety limits. Training endpoint poses are explicit G/E-style hindsight arguments, not online inputs from future robot states.

### Handoff

**entry_conditions**

Valid calibration/controller, free or known disengageable tip, finite robot state, valid tool waypoint and current obstacle tracks.

**exit_conditions**

Waypoint position/orientation within uncertainty-aware tolerance with a safe approach corridor and low velocity, or report reached-waypoint-but-not-settled if the training boundary was in motion. Policy does not declare object placement.

**failure_signatures**

Unexpected component motion, rising torque residual, tip inside an unapproved occupied region, no safe path, stale tracking or repeated command clipping.

**overlap_role**

Initial context and 15-row buffers initialize scene and short history. Two transfers overlap object-policy release/access supervision under explicit tool goals, not object-motion goals.

**successor_readiness**

Return tip pose, target boundary accessibility, obstacle covariance and any remaining contact; relocate or align may start only after verifying entry. Otherwise agent selects another waypoint or safe stop.

### Heuristic id

clearance_phase_prior

### Input preprocessing

P/S plus current signed-distance obstacle raster from visible instance masks, rod projection and commissioned tool swept volume. Fuse q/T_base_ee with endpoint-relative pose, height relative to verified plane where available, current image-space obstacle distance and last 0.5 s of commands/state. Class labels are not inputs. Train a small recurrent phase network (depart, traverse, descend, settle) with q-conditioned command residual; latent phases may use weak height/motion cues, not purported contact truth. If metric obstacle conversion is unavailable, image clearance remains a cue but robot execution is blocked until a conservative collision envelope is verified.

### Limitations

Sparse examples, no independent active search or collision recovery demonstrations. A low tip projected onto a piece does not establish contact. Planning success cannot certify cable/arm clearance without acquired tool/robot geometry.

### Pipeline implications

Implement and validate swept-volume checker, waypoint proposer, residual target derivation and hysteretic entry/exit monitor. Count contact-contaminated access rows after dense tracking; keep their parent object supervision. No autonomous robot movement is authorized by this plan.

### Policy contract

**caller_arguments**

policy_id=access_clearance_v1; goal_tool_pose T_base_ee, mode=approach|depart|transfer, protected_instance_ids, optional target instance/boundary region, tolerance/time budget/safety profile.

**input_output_contract**

Group tool_access. Causal RGB/state/history through P/S; propose finite seven-joint velocity through D. Own obstacle avoidance, approach phase and stopping, not character assignment or letter displacement.

**memory_and_handoff**

Reset phase state on new call; initialize with up to 0.5 s available history and validity masks; preserve tracker IDs across calls. Return actual pose/clearance, never assume next object is contacted.

**selection_cues**

Choose before another policy when tip is high, on wrong side, obstructed, or a previous push finished in contact. Prefer this over arbitrary long motion from an alignment policy.

**status_and_progress**

Report RUNNING phase, waypoint error, clearance margin, uncertainty; READY only after stable safe entry. BLOCKED/LOST_TRACK/SAFETY_STOP include responsible obstacle/measurement.

### Policy id

access_clearance_v1

### Rationale

Both episodes approach from a similar high robot start and later traverse at low height between pieces. Tool-goal relative geometry plus clearance costs should transfer these motions without memorizing a particular next letter. This remains a hypothesis pending held-out starts and obstacle layouts.


All scientific text above is unchanged Runtime API output. Rendering and slicing are developer-owned. No policy code or training exists in this stage.
