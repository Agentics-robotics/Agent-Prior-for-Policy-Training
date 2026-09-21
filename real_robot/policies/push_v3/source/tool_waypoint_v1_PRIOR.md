# tool_waypoint_v1 — prior

Responsibility: noncontact TCP approach, reposition and guarded reset support on `tool_position`. Exact heuristic: `waypoint_clearance_residual`. See [shared science](PRIOR.md), [usage](tool_waypoint_v1_USAGE.md), [calling contract](CALLING.md), and [handoff](HANDOFF.json).

## Evidence

A=`episode_2026091922002701`, B=`episode_2026091922062101`:

| Frozen segment | Supervised [start,stop) | Materialized context |
|---|---|---|
| a_initial_tool | A[0,600) | [0,620) |
| b_initial_tool | B[0,380) | [0,400) |
| a_l_approach_tool | A[6120,6251) | [6100,6271) |
| b_r_reset_tool | B[4000,4251) | [3980,4271) |

The latter two intervals reuse piece evidence; they are not new demonstrations. Initial waiting, descent, lateral transit, lift/reapproach and real arm reconfiguration are retained. The B reset's large measured dq is not declared corrupt or a verified singularity. Endpoint moving commands are retained, not mislabeled zero. The installed tool is a rod; no grasp behavior is taught.

## Model and implemented bias

Approximately two-million-parameter independent model: two-layer robot/error encoder, compact 8-channel map CNN, 192-dimensional masked GRU, narrow task-relative representation, five-mode head, speed gate and bounded six-component native-command residual. Four causal snapshots span eleven previous paired rows plus current; unavailable history is masked. Native linear commands are represented in a reversible W-related basis, angular serialization is left intact. All q/dq/tau inputs are measured state, with validity masks; only actual previous command is an intent input.

A deterministic TCP-to-waypoint direction is the base proposal. Occupied planar corridors cause a proxy lift toward a fixed W TCP height of .10 m, followed by traversal/descent as geometry changes. It is a **limited proxy route generator**, not full arm/rod collision planning or verified extraction. The learned model controls phase/speed/lateral/posture corrections around that anchor, not a replayed seven-joint trajectory. The map is locally TCP-centered with coarse whole-scene occupancy.

Conditional action-sequence diffusion was considered for multiple free-space paths. Four waypoint invocations cannot constrain a broad trajectory distribution or redundancy policy. Geometry plus a bounded recurrent residual is better matched to this evidence. This choice is independent of the executor's one-step check.

## Goals, losses and eligibility

Training goal is last supervised `T_base_ee` and recorded terminal v,w. Online goal is caller `goal_tcp_base`, with stop/pass-through and native terminal intent; no endpoint/source index exists online. Huber native-action cloning, small heteroscedastic likelihood, weak mode cross-entropy, residual regularization and a command-residual auxiliary train learned outputs. No false object motion target is fabricated for tool-only rows. Finite v,w including zeros remain eligible despite null command dq. Mode labels derive from numeric commands and geometry, not a contact sensor. Cache coverage is reported per frozen interval and per loss after preprocessing.

## Perception/adapters and consequences

Uses shared original-RGB calibrated remap, contrast foreground, causal persistent masks and full recorded wrist/flange transforms. Without verified plane, outputs are board-plane/TCP proxies with 30-mm design margin. No depth metrology, physical rod-tip model or inverse-kinematic follower is asserted. Joint and tau guards are conditional on caller assets; singularity and reachability remain unchecked. Supplied audited intent conversion enables a physical translational speed cap and planar screen only; decoder still cannot output hardware joint commands.

Training settings: batch 32, AdamW learning rate 2e-4, weight decay 1e-4, gradient norm 1, prescribed 20k updates and EMA. No evaluation claims. A tool goal can be reached geometrically without intentionally pushing a piece, but noncontact intent/visual stability is only weak evidence, not contact-free certification. Unintended neighbor motion is a monitored failure. Arbitrary crowded navigation, jam extraction and unseen materials are unsupported.
