# Proposed callable policies (not implemented or trained)

## shape_push_v1

Dataset: `shape_push_decisions`; heuristic: `contour_contact_prior`.

Learn one letter-identity-blind geometric contact/stroke selector. Use a small point-feature encoder (shared two-layer 64-unit MLP with permutation-invariant pooling) and a shared two-layer 64-unit candidate scorer conditioned on pooled actor/goal context, local curvature/normals, inner/outer-loop geometry, obstacle distance features, tool dimensions and recent measured motion. Softmax over feasible joint contact/direction/distance candidates; bounded features, weight decay and early stopping at episode-level validation. No large image encoder is trained from two episodes. Geometry canonicalization provides exact coordinate equivariance where valid; physical scale/friction are not assumed invariant. Learn supervised choice, not an uncalibrated object forward model. Stop/validity/progress and word-level decisions remain explicit deterministic or Agent responsibilities.

### Caller arguments

instance_id, target reference-contour transform G in P, metric position/yaw tolerances, scene_revision, plane/tool/robot calibration identifiers, protected tracks and workspace polygon, approach/stroke/force-guard limits, retry budget, baseline direction, optional last execution ticket/history token. The caller names a physical instance and objective, not an inferred demonstration label or command sequence.

### Input output contract

Causal Scene geometry/timing plus short history -> ranked joint contact/direction/distance specifications in P with validity flags, scene revision, uncertainty, and deterministic status. Point/length units are meters, angles radians; transforms use explicit source/destination frame tags. Contact points are on the piece boundary, not the flange/TCP. The adapter performs all metric transformations to executable robot objectives. Robot dq command vectors are never network output.

### Memory and handoff

Stateless network over up to three causal snapshots; external token records instance/reference-contour ID, scene/calibration revision, goal, last attempted candidate, ticket outcome and retry count. Initialize from within-segment buffers at training, from live observations at deployment; mask unavailable samples. Reset on new instance, unexpected track association, changed calibration, goal replacement or interruption. Retain previous measured response only when it concerns the same physical instance/goal and is causal.

### Selection cues

Invoke for a selected rigid planar piece not at its requested geometric pose when at least one safe contact is possible. Same API before travel, at standoff, in contact and after correction. It is not selected by letter category or visible word. Request new perception or a planner view instead if geometry is obscured/uncertain; choose another/staging target if blocked.

### Status and progress

proposed means a local execution objective exists. already_at_goal requires independently measured stable pose error within caller tolerance. needs_view, ambiguous_geometry, no_safe_candidate, unsupported_contact, stale_scene and executor_replan are actionable non-success statuses. After each stroke report measured translation/yaw/contour error change, contact retention estimate, protected-object motion, elapsed stroke budget and retry count. Do not report probability of task success from softmax score.

See the dataset heuristic document for applicability, limitations, and handoffs.
