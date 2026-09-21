# piece_contact_graph_v1 — usage

`policy.act(graph_model, observation, call, memory, graph_spec)` returns an offline native [v,w] candidate and diagnostics. Use this policy's independent checkpoint. See [PRIOR](piece_contact_graph_v1_PRIOR.md), [CALLING.md](CALLING.md) for the complete exact shared schema, and [HANDOFF.json](HANDOFF.json).

Select for one trackable rigid flat piece when local outer/inner contours and tool/contact-side geometry are credible. It internally proposes boundaries and corrective modes; the agent supplies a physical objective, **not each push**. A hole/recess must be large enough for verified tool geometry before any later physical operation. Confidence/weak response is not proof of feasibility.

## Required call

* `policy_id='piece_contact_graph_v1'`, new `call_id`, current `now_s`.
* `instance_id`: persistent application ID; `instance_seed_W:[x,y]` and fresh `inventory_t` for initialization/reacquisition. W positions are meters; seed is the piece location, not contact selection. Missing/ambiguous binding fails explicitly.
* Exactly one `goal_SE2_W` (3x3 rigid homogeneous matrix applying to the **entry footprint's W points**) or `goal_footprint_W={outer:[rings], holes:[rings]}` with vertices in W meters. Goal remains fixed until a new call. No semantic label/word/episode/endpoint is accepted as a spatial objective.
* Current paired original dual RGB, robot transforms/state, calibration and timing, per CALLING; saved normalization/device in spec.

Optional `exit_tool_base` is a 4x4 TCP base pose; `terminal_twist` length-six native encoding and `arrival_mode` stop/pass_through describe its arrival. Absent exit uses entry TCP as an internal anchor but not a required completion waypoint. Set explicit exits for handoff-sensitive work. `tolerance_m` controls symmetric contour distance; `tolerance_rad` checks requested TCP orientation and an optional `semantic_orientation` directed live axis. It does not invent a canonical object yaw.

All shared options/defaults are implemented as listed in CALLING: timeout_s=30, max_age_s=.25, tolerance_m=.01, tolerance_rad=.08, speed_cap_m_s=.03, clearance_m=.02 (requested/unverified), native_abs_cap=3*saved scale, stall_s=8, retry_budget=8, neighbor_tolerance_m=.015; protected_instance_ids and current instance_bindings_W; optional verified top-plane workspace, fresh external union live_mask/live_mask_t, actual last_executed_action/last_command_verified, guards, audited intent_mapping, rod_geometry, semantic_orientation, abort. A physical speed cap/planar screen requires the audited matrix; full rod clearance and decoder readiness remain unavailable.

## Example: local temporary clearance goal

```python
scene = policy.inventory(obs, graph_spec)
obj = scene['instances'][0]  # HLA must verify/reacquire the intended physical identity
G = [[1.,0.,-.025],[0.,1.,.01],[0.,0.,1.]]
call = dict(policy_id='piece_contact_graph_v1', call_id='clear-corridor-3',
            instance_id='tracked-blocker-3', instance_seed_W=obj['seed_W'],
            inventory_t=scene['t'], goal_SE2_W=G, now_s=float(obs['t']),
            arrival_mode='stop', tolerance_m=.012, timeout_s=25.,
            speed_cap_m_s=.02, protected_instance_ids=[])
r = policy.act(graph_model, obs, call, None, graph_spec)
```

For rotation about entry centroid c, construct `R=[[cos(a),-sin(a)],[sin(a),cos(a)]]` and translation `desired_center - R@c`. Do not put desired centroid directly into the translation field unless that really is the intended homogeneous transform. Explicit goal silhouettes preserve holes and are checked for rigid compatibility.

## Results and continuation

Normal finite candidates return CONTROLLER_UNVERIFIED, meaningful progress/error, uncalibrated command std, track confidence, proximity, inferred phase, symmetry flag, neighbor changes and readiness. The one-step prediction hook is not actuation. `decode_action` remains disabled. REACHED_LOCAL_GOAL requires three fresh geometry/exit/arrival observations, caller-verified plane and optional semantic axis check; it is not word success. Null-action statuses are PERCEPTION_UNCERTAIN, BLOCKED, NEEDS_REPOSITION or ABORTED with reasons. RUNNING is not emitted as physical operation.

Continue only with fresh same-instance views and acceptable geometry, protection and budget diagnostics. Repeated no-progress, unexpected rotation/mask drift, uncertain neighbors, stalled contact, unsafe hole width and large measured reconfiguration require agent inspection. No calibrated force, success probability, singularity check or universal recovery is promised.

## Memory and switch

Memory persists masks and up to twelve causal feature records; the 192-dimensional recurrence is recomputed from four offsets. Reset on policy/goal/instance/frame change, tracking gap/loss, new call ID or interrupt. Never silently swap a lookalike barred piece. Preserve the desired world footprint outside memory before pausing. To use a tool reset or the field alternative, reacquire current identity, compare neighbor changes, retain goal_footprint_W or recompute SE(2) from the new entry footprint, and start memory=None with a new call_id. Do not reuse a previous candidate as last executed command. Physical readiness remains false even if local geometry is reached.
