# contour_push usage

**ID:** `contour_push`; **heuristic:** `boundary_graph`. This trained candidate generator covers one physical instance to one fixed image goal, including demonstrated approach/contact changes/corrections/withdrawal. It does not choose characters, ordering or a next goal. [CALLING.md](CALLING.md) is the complete shared API contract, including decoder fields; [contour_push_PRIOR.md](contour_push_PRIOR.md) explains the trained mechanism. Handoff fields are in `HANDOFF.json.policies.contour_push`.

## Entry point and exact inputs

`policy.act(contour_model, observation, call, memory, saved_spec)`; `contour_model` must be this policy's independently trained EMA checkpoint, loaded with its saved normalizers/spec/device. Never pass the visual checkpoint or an untrained initialization.

Observation: original paired RGB uint8 `third_rgb`, `wrist_rgb` `[720,1280,3]`; finite q/measured dq/tau_ext `[7]`; finite `T_base_ee`, `T_base_flange` `[4,4]` (or row-major 16); gripper_position/width scalars with -1/NaN/None treated as missing; robot `t`, `recv_time` seconds; original monotonically increasing `sample_index`; original `img_t_third/wrist` and finite signed `img_age_third/wrist` seconds; calibrated recording-compatible `metadata`, including third K/dist/T_base_cam, calibrated wrist K/dist/T_flange_cam and TCP. Both views are required even though the graph motor encoder consumes geometry rather than wrist RGB. Image age absolute magnitude must be <=.5 s. No clock realignment, future observation, action_json, class/segment ID or recorded object coordinates are legal online inputs.

Exact call fields:

| Field | Meaning and units |
|---|---|
| `call_id` | new string invocation ID |
| `scene_version` | current scene namespace string |
| `instance_id` | HLA-associated arbitrary ID returned by automatic inventory |
| `template` | unchanged current `observe_scene` template matching ID, scene, calibration chart; at initialization age <=10 original sample rows |
| `spatial_goal` | dict with `frame='third_undistorted_original_pixels'`, `center_uv=[u,v]`, `theta_deg_clockwise`, `accept_similarity_approximation=True` |
| `workspace_polygon_uv` | >=3 vertices in same original undistorted chart; image constraint, not certified robot workspace |
| `tolerances` | required position_px (0,100], angle_deg (0,180], foreground_iou [.1,1], stability_s [.1,5], neighbor_motion_px >0 |
| `max_duration_s` | seconds (0,600] on robot t |
| optional `obstacle_polygons_uv` | list of chart polygons; default [] |
| optional `protected_templates` | fresh references for already-placed objects; default [] |
| optional `cancel` | boolean, default false |

Call contents are JSON scalars/lists/dicts. Chart coordinates are original undistorted 1280x720 pixels: origin upper left, u right/v down. Angle is clockwise relative to the actual invocation foreground, **not metric yaw or semantic upright**. The fixed-scale similarity renderer preserves holes and flags approximation; >520-pixel translation, clipping, unsupported frame/metric goals and out-of-workspace goals refuse. Half-resolution masks/returned preview have 2-original-pixel raster spacing. `preview_goal` generates the silhouette, so callers do not manually label it.

## Concrete call example

```
scene = policy.observe_scene(obs, 'scene_21', None)
ref = next(p for p in scene['instances'] if p['instance_id'] == hla_selected_id)
call = {
 'call_id':'scene_21_relocate_1', 'scene_version':'scene_21',
 'instance_id':ref['instance_id'], 'template':ref,
 'spatial_goal':{'frame':'third_undistorted_original_pixels',
    'center_uv':[700.,420.], 'theta_deg_clockwise':20.,
    'accept_similarity_approximation':True},
 'workspace_polygon_uv':[[160,60],[900,60],[1040,700],[70,700]],
 'obstacle_polygons_uv':[], 'protected_templates':[],
 'tolerances':{'position_px':8., 'angle_deg':10., 'foreground_iou':.7,
               'stability_s':.5, 'neighbor_motion_px':20.},
 'max_duration_s':30.
}
goal = policy.preview_goal(ref, call['spatial_goal'], call['workspace_polygon_uv'])
out = policy.act(contour_model, obs, call, None, contour_saved_spec)
# Later, same exact call, latest actually paired observation:
out2 = policy.act(contour_model, next_obs, call, out['memory'], contour_saved_spec)
gate = policy.decode_action(out2['action'], None, contour_saved_spec)
# gate.enabled is false: controller semantics are NOT inferred from metadata.
```

`hla_selected_id` must come from real inventory association, not demonstration order or known-alphabet lookup. Polygon/goal/tolerances above illustrate argument shape only; they are not a commissioned safety setup or a verified achievable placement. If inventory or semantic association is ambiguous, do not call a motor model to resolve it by moving.

## Memory, outputs and statuses

Memory is None at invocation/segment start. It stores causal perception tracks, selected association, fixed goal and up to 30 original-row span of cached features/monitor values. Eight history positions at spacing three rows are masked if unavailable; online matching allows at most two earlier rows of observation cadence slack. The model recomputes masked two-layer temporal memory from this window. No hidden state/actions transfer from visual_push or previous targets. Keep call/instance/goal/template/chart/tolerances/controls fixed. A different goal, template, policy or referent requires stop, a new call ID and memory=None. A >.5-s observation gap terminates; uncertainty alone can retain the same call while stopped and re-observing the same track.

Return dict: `action` finite native command dq[7] for running, otherwise null; `memory`; `status`; `diagnostics`. Native means exactly the learned recorded-command coordinate system, not presumed rad/s until separately verified. There is no gripper command or hardware I/O.

* running: observed unique selected mask, proxy confidence >=.65, required sensors/goal checks valid; learned candidate only.
* uncertain: lost/ambiguous track, confidence below .65, missing protected object or long observation gap; request verified stop, never continue blind.
* unavailable: missing/malformed/stale data/calibration/template, unsupported geometry (reason includes calibration_required), changed fixed call, reused terminated call or non-increasing sample/time.
* blocked: explicit goal-obstacle/protected overlap, selected foreground outside image workspace, protected-neighbor displacement over tolerance.
* goal_observed: observed centroid, foreground IoU and rotation modulo shape-symmetry agree for stability_s; terminates with **no action**. Clearance and word success remain unknown.
* timeout/cancelled: terminated, null action, no success implication.

Diagnostics include contour/track confidence proxy, center error (original pixels), foreground IoU, registration/symmetry hypotheses (10-degree bins), error slope px/s, direction-change count, possible neighbor displacement px, projected TCP original pixels/in-image flag and mixture standard deviation in native action coordinates. Rod-tip visual verification is false; contact, recontact count, tool clearance and view consistency are unavailable/null. `successor_ready=false` is intentional: geometric agreement cannot establish safe handoff. Error-direction changes are not diagnosed re-contact events or forces.

## Selection, switching and decoder

The HLA may test this alternative for reliable visible boundaries/holes with stable association and commissioned tool/camera geometry. Only the association/relative-area proxy is automatically thresholded; actual contour completeness, reachable safe travel and hazard interpretation remain HLA/commissioning judgments. A TCP point does not establish rod clearance. Low contour proxy with still-valid RGB association can motivate considering visual_push, not a mandatory fallback.

Cancel/stop before switching, save the intended `goal['foreground_half']`, refresh automatic inventory, externally confirm the same physical instance and call `policy.reexpress_goal(new_ref, old_foreground)`. On goal_reexpressed, start visual_push with its own checkpoint, the returned relative goal and memory=None; refresh protected references and check symmetric orientation semantics. Reexpression refuses poor shape/scale/IoU rather than silently retargeting. Copying the old relative angle to a rotated new template would change the goal and is forbidden. Never transfer motor memory or an old candidate.

`decode_action(action, verified_contract, saved_spec)` requires all fields in CALLING §7: independently verified rad/s recording interpretation and q radians, matching seven joint names/order, timing/hold/watchdog/latency, current state/previous issued command/elapsed time, velocity/acceleration/joint bounds, and external stop/tool/workspace/singularity attestations. It refuses incomplete contracts and violations rather than clipping. Even enabled means only a non-actuating command artifact. No robot success, general collision avoidance or held-out shape competence follows from this call interface.
