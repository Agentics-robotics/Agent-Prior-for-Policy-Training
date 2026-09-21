# Calling the cut_v5 candidate policies

[PRIOR.md](PRIOR.md) describes science and adaptations; [POLICY_CATALOG.md](POLICY_CATALOG.md) links all three independent prior/usage documents. [HANDOFF.json](HANDOFF.json) is the selection/continuation/switch contract. This is an implemented Python library, not a hardware service.

## Hooks and loading

The executor trains separate checkpoints. `prepare_data(spec)` returns numeric cached arrays and provenance/eligibility/normalization metadata. `make_batch(policy_id, arrays, metadata, indices, spec)` gathers cached features without decoding histories again. `build_model(policy_id, spec)` creates that independent model on spec.device; load its own checkpoint and use evaluation mode for invocation. `compute_loss` and `predict` are training/reload hooks; predict returns finite native [v,w], not robot joint commands. Native scales include quantile estimates with magnitude-extrema floors so rare angular intent stays within bounded residual support.

```python
r = policy.act(model, observation, call, memory=None, spec=spec)
memory = r['memory']
# Use a fresh paired observation, updated now_s and ACTUAL previous-command feedback:
r = policy.act(model, next_observation, next_call, memory, spec)
```

Deployment needs saved `data_metadata.normalization`, policy configuration and device, plus live calibration. It does not need training episode IDs, source plans or endpoints. No optimizer, rollout runner or robot client is shipped.

`decode_action(action, controller_contract, spec)` always returns `enabled=False, command=None, reason=...`. A caller assertion cannot install/verify the missing recording follower. Valid act calls nevertheless run the learned model and return meaningful offline candidates. **Never send a candidate directly to a robot.**

## One causal observation

Required:

* `third_rgb`, `wrist_rgb`: original 1280x720 HWC uint8 **RGB**, not BGR or preview images.
* `T_base_ee`, `T_base_flange`: finite rigid 4x4 poses. TCP is the recorded ee, not an asserted physical rod tip.
* `q`, measured `dq`, `tau_ext`: length seven. Measured dq is not commanded dq; residual torque is not calibrated force.
* `t`, `sample_index`, `img_age_third`, `img_age_wrist`; provide `recv_time`, `img_t_third`, `img_t_wrist` when available. Live sample_index must advance; it is not a training index. t and call.now_s share a seconds clock. Image stamps detect repetition; absent stamps fall back to t, leaving detection of repeated image content to the caller. recv_time is accepted but freshness uses t and image ages, not timestamp re-pairing.
* `metadata.calibration`: K/distortion and third `T_base_cam`, calibrated wrist K/distortion/`T_flange_cam`, and `T_base_board`. Wrist transform is `T_base_flange @ T_flange_cam`; do not mix factory wrist K with calibrated extrinsics.
* `gripper_position`, `gripper_width_m` when available. -1/NaN are expected missing feedback and masked, never grasp/contact evidence.

No action_json, future image, source episode/segment, hidden object label, endpoint or target_pose training label is an online observation.

## Common call dictionary

Goals/metadata in the fixed call signature are JSON-compatible lists/dictionaries. RGB, optional live_mask and opaque memory are Python objects, not a remote JSON protocol.

| Field | Type/default | Implemented meaning |
|---|---|---|
| `policy_id` | required exact policy ID | Must match model |
| `call_id` | required nonempty string | New invocation identifier |
| `now_s` | required finite seconds | Same clock as observation t; update each call |
| `timeout_s`, `max_age_s` | 30 s, .25 s | Call lifetime; absolute now/t and camera-age bounds |
| `arrival_mode` | stop or pass_through; stop | Stop completion needs actual verified zero command and low measured motion |
| `terminal_twist` | six finite numbers; zeros | Desired terminal **native recorded encoding**, not automatically SI twist |
| `last_executed_action` | six finite numbers; zeros placeholder | Actual prior recorded-encoding command; never feed a candidate as if executed |
| `last_command_verified` | false | Stop arrival needs this AND explicit last_executed_action; default zero is not verified execution |
| `tolerance_m`, `tolerance_rad` | .01 m, .08 rad | Spatial acceptance enlarged to >=2*plane margin; tool rotation and optional semantic-axis tolerance |
| `speed_cap_m_s` | .03 | Enforced only through an audited intent matrix; otherwise explicitly unenforced |
| `clearance_m` | .02 | Requested physical clearance, reported unverified; route uses a fixed .10-m W TCP lift floor, not certified rod clearance |
| `native_abs_cap` | six positive finite numbers; 3*saved scales | Numerical native clipping, not a physical unit claim |
| `retry_budget`, `stall_s` | 8, 8 s | Inferred reset-mode entries; require .002-m proxy-error improvement within stall interval |
| `neighbor_tolerance_m` | .015 | Change threshold plus 2*plane margin |
| `protected_instance_ids` | list; [] | Named IDs need current bindings; all detected nonselected pieces are protected regardless of names |
| `instance_bindings_W` | name -> [x,y] | Fresh W seed positions to bind named neighbors at initialization |
| `abort` | false | Return ABORTED and reset memory, no candidate |

Positive finite values are required for the supplied age/time/speed/clearance/tolerance fields. Goal, identity, workspace, terminal twist and arrival mode are fixed during a memory lifetime. Start a new call to change them.

### Workspace and external perception

`workspace={T_base_W:4x4, plane_sigma_m:nonnegative float, verified:bool}` optionally supplies a measured relevant piece-top plane (z=0) and its uncertainty. Caller owns truthful verification. Without it, W is the **board-plane proxy**: board origin, board x, normal flipped toward base +z, y=z cross x. Its .03-m design margin is conservative, not calibrated sigma. The board calibration does not certify tabletop or letter-top height. T_base_W maps W column coordinates to base.

Fixed raster: W x in [-.2,.8), y in [-.35,.65), 256x256. Column u and row v map to `[-.2+u/256, -.35+v/256]` meters. Row increases W y. Model local crops are 64x64 native cells (0.25 m); coarse global occupancy supplements them.

`live_mask`: optional 256x256 uint8 current **union** foreground mask in that raster; `live_mask_t` within .05 s of observation t. It replaces HSV segmentation, not persistent tracking. It must originate from actual causal external perception, not a training endpoint or glyph template. The package does not assume an unavailable open-world segmenter exists. Inventory's default hook uses the board proxy; if the caller supplies a different W or external mask, obtain/convert seeds and goals in that actual frame before act.

### Optional audited numerical mapping and guards

`intent_mapping={verified:true, audit_reference:string, native_to_base_twist:6x6}` is an invertible audited current-configuration map from native six-vector to physical **base-frame TCP twist**, linear m/s and angular rad/s. It enables translational speed reduction and a .15-s planar TCP/protected-mask sweep screen, followed by inverse conversion. Out-of-map sweeps block. It does not implement angular/acceleration limits, a full rod/arm swept volume or actuation.

`rod_geometry={verified:bool, radius_m:positive float}` supplies a screen radius only, not an installed tip/tilt model. Missing radius defaults to .006 m as a **proxy screen margin**, not a measurement. Plane margin enlarges it. `guards={q_min:[7], q_max:[7], tau_abs_max:[7]}` enables .02-rad joint margins and residual-torque thresholds. Without supplied assets these checks are unknown, not passed. Singularity/reachability and physical clearance are never claimed verified; physical readiness stays false even with an intent mapping.

## Policy-specific physical goals

### tool_waypoint_v1

Required `goal_tcp_base`: finite rigid 4x4 TCP pose, base meters and rotation matrix. All detected pieces are obstacles. Choose it for noncontact approach/reposition, not intentional pushing or an unresolved jam. For moving handoff set arrival_mode=pass_through and choose terminal_twist in native encoding. For stop use zero terminal intent and later supply actual stop feedback.

### Both piece policies

Required `instance_id`: persistent nonempty application ID; `instance_seed_W:[x,y]` in current W meters; `inventory_t` within .5 seconds of initialization. This seed selects a piece, not a contact point. Acquisition requires distance <=.06 m and >=.008 m separation from the next alternative. Identity is then tracked; loss does not silently select another piece. The same margin is checked for named protected bindings.

Supply exactly one goal:

* `goal_SE2_W`: rigid 3x3 homogeneous matrix acting on entry-footprint W points, `p_goal=R@p_entry+translation`. Translation is **not necessarily the desired centroid**. For rotation about entry centroid c to desired center d, use translation `d-R@c`. Scaling/reflection and severe out-of-raster truncation are rejected. Goal mask stays fixed in W.
* `goal_footprint_W={outer:[ring,...], holes:[ring,...]}`: rings are W [x,y] vertex lists. Explicit holes remain holes. The rasterized goal must fit the selected observed shape by rigid overlap >=.5; it cannot be an unrelated alphabet template.

Optional `exit_tool_base` is a base-frame TCP 4x4 pose. terminal_twist/arrival_mode apply to its arrival. Without explicit exit, entry TCP becomes the model's default exit anchor but is not required as a completion waypoint. Use explicit exits for consequential handoffs. No caller goal is read from future data.

Optional `semantic_orientation={goal_axis_W:[dx,dy], observed_axis_W:[dx,dy], observed_t:seconds}` adds a live directed-axis completion check. A missing/stale (> .1 s) observed cue prevents semantic satisfaction; zero/nonfinite axes are rejected. The model consumes geometry, not character identity. A symmetric silhouette cannot establish semantic upright or correct spelling. tolerance_rad applies to this cue and requested tool orientation, not an invented canonical piece yaw.

## Outputs and status

`{action, memory, status, diagnostics}`. action is a finite length-six Python list in original recorded [v,w] encoding, or null. It is not an enabled physical twist or joint command. memory is opaque Python state, not trained weights.

* **CONTROLLER_UNVERIFIED**: normal usable candidate path. Learned model runs and returns progress/uncertainty; hardware readiness is false.
* **REACHED_LOCAL_GOAL**: three fresh observations satisfy geometry, requested exit/arrival and optional semantic cue, no uncertain neighbors, and caller-verified plane. Stop candidate becomes zero. This is not word success or physical safe-to-handoff.
* **PERCEPTION_UNCERTAIN**: stale/repeated/out-of-order frames, invalid input, lost/ambiguous track or >.5-s tracking gap. No candidate.
* **NEEDS_REPOSITION**: stall/reset budget. Preserve the goal and inspect/reacquire before another policy.
* **BLOCKED**: observed protected/nonselected motion, supplied robot guard violation or audited-mapping planar sweep collision. Not a complete arm-collision diagnosis.
* **ABORTED**: interruption, timeout, mismatched policy or changed fixed call signature.

RUNNING is in the broader design vocabulary but is deliberately not emitted as physical operation. A null or zero candidate is **not an executed stop**.

Diagnostics include contour/position and TCP rotation errors, actual TCP pose, heuristic track confidence, symmetry, phase, proxy proximity, reset count, dual-view disagreement, changed/uncertain neighbors, requested/enforced cap flags, last-command availability and explicit readiness. Native std is uncalibrated model uncertainty, not physical safety probability. Graph calls additionally return contact_candidate_index, mixture weights and selected_boundary_point_W (proxy geometry). `predicted_motion_proxy_unvalidated` exposes the raw auxiliary head: for piece models it predicts weak normalized local displacement (x/y divided by .01 m, angle by .1 rad), while **for the tool model this legacy diagnostic key contains normalized command residual, NOT object motion**. It is conditioned on the proposal anchor, not a validated physical rollout; never interpret it as contact or force.

Measured |dq|>.6 raises an advisory, not a corrupted-label or singularity diagnosis. Missing robot assets are visible. Every result's physical/hardware/safe-to-handoff readiness remains false in this package. Protection baselines are reacquired per call; compare old/new diagnostics externally across switches.

## Inventory, layout and examples

`inventory(observation,spec)` exposes snapshot-local handles, W seeds/rings, proxy areas, board-proxy transform/margin, view disagreement and **no semantic hypotheses**. Bind your own persistent instance_id only after checking the current physical identity.

```python
scene = policy.inventory(obs, field_spec)
x = scene['instances'][0]  # HLA verifies the intended actual physical instance
call = dict(policy_id='piece_goal_field_v1', call_id='stage-17',
            instance_id='physical-piece-42', instance_seed_W=x['seed_W'],
            inventory_t=scene['t'], now_s=float(obs['t']),
            goal_SE2_W=[[1.,0.,.02],[0.,1.,0.],[0.,0.,1.]],
            timeout_s=30., tolerance_m=.012, protected_instance_ids=[],
            speed_cap_m_s=.025)
r = policy.act(field_model, obs, call, None, field_spec)
```

A tool waypoint is chosen from the current scene, not a recorded endpoint:

```python
waypoint = np.asarray(obs['T_base_ee']).copy()
waypoint[2,3] += .04  # base-z objective; inspect corridor/table/rod implications
call = dict(policy_id='tool_waypoint_v1', call_id='inspect-18',
            goal_tcp_base=waypoint.tolist(), arrival_mode='stop',
            terminal_twist=[0.]*6, now_s=float(obs['t']), timeout_s=10.)
r = policy.act(tool_model, obs, call, None, tool_spec)
```

`plan_layout(word, scene, semantic_hypotheses, origin_W, reading_direction_W, gap_m=.015)` accepts actual caller probabilities `snapshot_handle -> {character:probability}`. It assigns distinct physical instances with threshold .8, then proposes footprint-dependent translation spacing. Too few distinct instances returns INFEASIBLE_REQUEST; missing/ambiguous semantic assignments return PERCEPTION_UNCERTAIN rather than asserting missing characters. PROPOSED_LAYOUT requires explicit orientation, table, reachability, obstacle and staging checks. It is not autonomous recognition, a complete word motion planner or a mandatory policy sequence.

## Memory and switching

Memory belongs to one model/policy/call/instance/frame/fixed objective. It holds acquisition-template tracker poses and at most twelve causal feature records. Four offsets [11,7,3,0] feed a recurrence recomputed at each act; no indefinitely carried neural state is transferred. Shape templates are re-rendered from acquisition through accumulated rigid transforms rather than repeatedly eroded; identity/pose can still drift under occlusion or merged masks.

Switch procedure: interrupt without pretending a candidate was executed; preserve desired **fixed world footprint** and last diagnostics externally; reacquire the same physical identity from a fresh observation; inspect neighbor changes and corridor; choose the next policy with new call_id and memory=None. Supply actual previous command feedback. Never transfer graph/field/tool hidden or mode state.

If the entry footprint changed, do not blindly reapply its old displacement. Pass retained goal_footprint_W or recompute entry-to-same-world-goal SE2. If W changes, re-express both physical goal and all seeds in the new frame. Reacquire after tracking loss or >.5-s gaps. Semantic identity, jams, tipping, obstacle clearance and full robot feasibility remain agent judgments wherever the executable diagnostics cannot verify them.
