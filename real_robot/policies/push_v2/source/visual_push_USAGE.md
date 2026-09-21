# visual_push usage

**Policy ID:** `visual_push`; **heuristic:** `dual_view_memory`. Use its independently trained EMA model, not a graph checkpoint or an untrained network. It serves the same one-instance/fixed-placement responsibility as contour_push, including demonstrated approach, contact changes, transport, corrections and withdrawal. It is neither a general recovery policy nor a mandatory second stage. [CALLING.md](CALLING.md) is the complete shared API/decoder entry; [visual_push_PRIOR.md](visual_push_PRIOR.md) and `HANDOFF.json.policies.visual_push` explain assumptions and handoffs.

## Exact interface and coordinate units

`policy.act(visual_model, observation, call, memory, visual_saved_spec)` returns `{action,memory,status,diagnostics}`. Saved spec supplies this model's normalization/device; weights and EMA are loaded by the executor/checkpoint owner.

One current observation requires original **RGB** uint8 `third_rgb` and `wrist_rgb` `[720,1280,3]`, finite q/measured dq/tau_ext seven-vectors, finite row-major `T_base_ee` and `T_base_flange` 4x4 transforms, gripper scalar fields (invalid -1/NaN/None retained as missing flags), `t`/`recv_time` seconds, increasing original `sample_index`, `img_t_third/wrist`, signed finite `img_age_third/wrist` seconds and recording-compatible calibrated metadata. Required camera blocks: third K/dist/T_base_cam and calibrated wrist K/dist/T_flange_cam, with TCP metadata. Image age magnitude >.5 s or observation gaps >.5 s refuse candidate continuation. Robot dt and image repeats enter features; clocks and pairings are never realigned. No previous demonstrated action or future image is required/accepted as a current feature. Recorded command dq is not measured dq; state remains in recording numerical coordinates.

All call values are ordinary JSON types. Required keys and optional controls are exactly:

* `call_id`: new invocation string; `scene_version`: current scene namespace; `instance_id`: arbitrary automatic inventory ID selected by actual HLA association; `template`: unchanged corresponding current inventory template (same ID/scene/chart, initialization age <=10 original rows).
* `spatial_goal`: `{frame:'third_undistorted_original_pixels', center_uv:[u,v], theta_deg_clockwise:angle, accept_similarity_approximation:True}`. Original undistorted chart 1280x720, top-left origin/u right/v down. Clockwise degrees relative to invocation template, not semantic upright, metric yaw or world pose. Fixed scale, max translation 520 original pixels; clipping/unsupported geometry refuses.
* `workspace_polygon_uv`: >=3 same-chart vertices; image constraint only, not verified robot workspace.
* `tolerances`: `position_px` (0,100], `angle_deg` (0,180], `foreground_iou` [.1,1], `stability_s` [.1,5], `neighbor_motion_px` >0. Every key required; values are caller-selected and not validated performance guarantees.
* `max_duration_s`: seconds (0,600] on robot time.
* Optional `obstacle_polygons_uv`: list of polygons, default []; `protected_templates`: fresh already-placed instance references, default []; `cancel`: bool, default False.

`observe_scene(observation, scene_version, scene_memory=None)` produces automatic class-free material proposals/templates, not letter names. HLA semantic word interpretation, repeated-instance allocation and goal design are external. Refuse ambiguous semantic identity upstream; do not use a mnemonic or demo order to index an action. `preview_goal` renders the supplied foreground automatically and returns half-raster foreground/RGB (one pixel = two original pixels). No hand-drawn per-move masks are required. Raw target-scene parsing, semantic upright recognition and metric/projective goal requests are not implemented by this motor call.

## Concrete invocation

```
scene = policy.observe_scene(obs, 'scene_31', None)
ref = next(p for p in scene['instances'] if p['instance_id'] == hla_selected_id)
call = {
 'call_id':'scene_31_visual_1', 'scene_version':'scene_31',
 'instance_id':ref['instance_id'], 'template':ref,
 'spatial_goal':{'frame':'third_undistorted_original_pixels',
    'center_uv':[650.,380.], 'theta_deg_clockwise':-10.,
    'accept_similarity_approximation':True},
 'workspace_polygon_uv':[[160,60],[900,60],[1040,700],[70,700]],
 'obstacle_polygons_uv':[], 'protected_templates':[],
 'tolerances':{'position_px':8., 'angle_deg':10., 'foreground_iou':.7,
               'stability_s':.5, 'neighbor_motion_px':20.},
 'max_duration_s':30.
}
goal = policy.preview_goal(ref,call['spatial_goal'],call['workspace_polygon_uv'])
out = policy.act(visual_model,obs,call,None,visual_saved_spec)
# Only the next actual paired observation, same instance/goal/call:
out2 = policy.act(visual_model,next_obs,call,out['memory'],visual_saved_spec)
nonexecuting = policy.decode_action(out2['action'],None,visual_saved_spec)
assert not nonexecuting['enabled']
```

This illustrates numerical/JSON structure, not a safe layout. HLA must provide a real associated returned ID, valid current sensors and allowable placement. The code checks template/goal/overlap/workspace validity but cannot certify complete path feasibility or safe robot travel.

## Causal history and memory lifetime

Third global 320x180 selected-marked RGB, target/tool detail 160x160, wrist 256x144 and a separate masked foreground/RGB goal 160x90 feed independent encoders and a two-layer gated temporal model. Eight history samples at nominal three-original-row intervals span about .7 seconds at original cadence. Missing prefixes/history are explicitly masked; online selection allows up to two earlier original rows of cadence slack, never future filling. No recurrent confidence can authorize a command with completely lost identity or a missing camera.

Motor memory is an opaque in-process numerical dict containing only this invocation's causal tracks, selected association, fixed rendered goal, cached feature window and monitors. Reset to None on a new instance/goal/template/chart/policy/control/tolerance change or after termination. Observation/contact-side changes within the same valid goal are normal continuation. A separate scene inventory memory may retain causal physical identities, but it is not motor memory. Do not transfer graph hidden state, action history, learned normalization or auxiliary state. Returned memory may contain numpy arrays and need not be JSON serializable.

## Output and validity behavior

Only status `running` returns a finite seven-element candidate in **recorded-native commanded dq** coordinates. It is not measured dq, an integrated position target or hardware output. To interpret rad/s/joint order/timing requires the independent decoder contract. No gripper action exists.

Visual entry/continuation requires current selected association/area proxy confidence >=.3, fresh paired views and finite mandatory state. This threshold allows partial observed markers that would fall below contour_push's .65 gate, but both depend on the same foreground pipeline. It does not establish superior occlusion recovery. Markers are never manufactured from the future or completed after full loss.

Statuses:

* `running`: learned non-executable candidate; uncertainty/progress diagnostics available.
* `uncertain`: lost/ambiguous selected/protected identity or >.5-s observation gap; null action, externally verified stop requested. Gap terminates; short tracking uncertainty can be re-observed while stopped.
* `unavailable`: missing/invalid/stale RGB/state/calibration/template, unsupported image/metric layout, changed fixed call, non-increasing sample/time or terminated call reuse. Reasons distinguish unsupported geometry/calibration_required.
* `blocked`: explicit goal overlap with supplied obstacles/protected foreground, selected foreground outside workspace, or protected-neighbor displacement beyond tolerance. Not a measured collision/contact state.
* `goal_observed`: center/foreground/rotation modulo symmetry agreement observed continuously for stability_s; null action and terminal. Not proof of clearance, safe successor or word success.
* `timeout` / `cancelled`: terminal, null action, no success implication.

Diagnostics include target-marker/track confidence, goal foreground IoU and center error in original pixels, 10-degree orientation hypotheses/residual, two-second discrepancy slope, error-direction changes, possible neighbor motion, projected TCP validity and native mixture dispersion. Cross-view consistency, contact, recontact count and tool clearance are unavailable/null. Dispersion and tracker confidence are uncalibrated. `successor_ready=false` always; HLA must inspect safety/clearance, scene identity, all placements, semantic readability/stability and next travel.

## Selection, switching and decoder prerequisites

The HLA may test this alternative when a still-unique selected marker is partially fragmented and wrist/current RGB plus causal response are informative. It cannot continue through fully lost identity merely because a neural history exists. Compare geometric cues with contour_push only as untested selection hypotheses; no ranking or mandatory sequence is encoded. Shared color/background/calibration errors can defeat both.

Before switching to/from contour_push: issue an independently verified stop/cancel, preserve the old intended rendered foreground, refresh inventory and externally confirm the same physical referent. `policy.reexpress_goal(new_ref, old_fixed_foreground)` searches relative angle against the new template, checks apparent scale/IoU and refuses invalid reexpression. Only goal_reexpressed permits constructing a new call; HLA resolves semantic symmetry and refreshes protected references. Keep the intended absolute spatial placement, not the old template-relative angle. Start the other trained policy with memory=None and its own checkpoint/spec; never blend outputs or transfer motor memory.

The shared `decode_action` only enables a non-actuating artifact with independently verified recorded rad/s semantics, q radians, matching joint names/order, timing/hold/latency/watchdog and stop, tool geometry, workspace and singularity checks, live q/previous actually issued command/dt, and positive velocity/acceleration/joint bounds. See CALLING §7 for every required field. Invalid inputs or bounds refuse rather than clip. The package performs no hardware I/O. Training/valid calls alone do not establish unseen-shape competence, recovery or robot success.
