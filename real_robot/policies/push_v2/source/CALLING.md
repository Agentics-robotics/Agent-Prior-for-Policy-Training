# Complete shared calling interface

Import `policy`. Model construction/loading belongs to the executor/checkpoint owner: `model = policy.build_model(policy_id, saved_spec)` followed by its supplied trained EMA state, placed on `saved_spec['device']` and set to eval. Do not use a freshly initialized model as a trained policy. The package itself does not read checkpoint files or actuate hardware. Both IDs are exact: `contour_push`, `visual_push`. See the two `_USAGE.md` documents and `HANDOFF.json` for selection/handoff details.

## 1. One current paired observation

`observation` is a dict, not a future window or demonstration record selector:

| Key | Required representation |
|---|---|
| `third_rgb`, `wrist_rgb` | original paired **RGB** numpy uint8 arrays `[720,1280,3]`, not BGR or resized input |
| `q`, `dq`, `tau_ext` | finite length-seven numerical arrays; `dq` is **measured** state, not command |
| `T_base_ee`, `T_base_flange` | finite 4x4 homogeneous transforms, or 16 entries reshapeable in row-major order; recorded base-frame transforms |
| `gripper_position`, `gripper_width_m` | scalar values if available; -1 position and NaN/None width are missing, not an open/closed state |
| `t`, `recv_time` | recorded robot timestamp and receive timestamp in seconds; only robot `t` differences enter features/monitor; recv_time retained as source, unused by network |
| `sample_index` | integer original paired index, strictly increasing within a call; never an episode/phase feature |
| `img_t_third`, `img_t_wrist` | recorded image timestamps, used for equality/repeat detection only; unavailable values do not create synthetic timestamps |
| `img_age_third`, `img_age_wrist` | original signed seconds, finite; act refuses absolute magnitude >0.5 s; negative recorded ages are not clock-realigned |
| `metadata` | recording-compatible dict with `calibration.third.{K,dist,T_base_cam}`, `calibration.wrist.{K,dist,T_flange_cam}`, calibration dimensions and `tcp`; use the calibrated wrist block, not factory substitute |

State uses source numerical coordinates, not fabricated physical interpretations. The transforms' recording calibration is meter-based, but there is no tabletop inverse projection or force/contact inference. Command units are only **recorded-native** until independently verified by the decoder. No `action_json`, source segment, letter/class token, training object coordinates, endpoint image or demonstration sequence index is accepted as an online action selector.

Deploy with the same installed rigid tool and commissioned camera geometry. Actual robot safety, clearance and word interpretation are external. Missing mandatory state/RGB/calibration/freshness causes `unavailable`, not NaN inputs or a fake zero action. Gripper feedback alone may be missing because the tool does not use a gripper action.

## 2. Automatic scene inventory (no per-move manual labeling)

```
scene = policy.observe_scene(observation, 'scene_12', scene_memory=None)
# scene['status']: inventory_available, uncertain, or unavailable
# scene['instances']: observation-derived template dicts
scene_memory = scene['memory']
```

`observe_scene` detects material foreground proposals and assigns arbitrary `scene_12/pN` IDs. Continuing a separate `scene_memory` maintains causal scene IDs across observations; changing scene version resets it. Refresh this inventory at the current observation before every new motor invocation, especially after motion/occlusion or a switch. `scene_memory` is NOT a motor hidden state and must not be passed to `act`.

The HLA associates the actual desired physical referent with one returned ID using its external semantics/task reasoning. This example does **not** authorize selecting by a known letter index or memorized demonstration order. Templates contain `scene_version`, `instance_id`, exact serialized `chart_id`, `sample_index`, original-chart `center_uv`, confidence, half-raster bbox/mask and masked RGB. They are produced by code from observations; do not hand-author masks per move. No character recognizer or upright-semantic direction is returned. Ambiguous semantic/physical association must stop upstream, even if a proposal exists.

The material detector is limited to separable brown-like pieces on bright support. Its confidence is an assignment-margin and relative-area proxy, not a calibrated probability of correct identity. It can fail on different colors, clutter, touching pieces and rod occlusion. It never completes a lost mask from a future observation.

## 3. Fixed spatial goal and exact call arguments

`call` must contain ordinary JSON-compatible types (lists, numbers, strings, booleans; no numpy arrays in call signatures). Required fields:

* `call_id`: unique string for this invocation; do not reuse it for another target/goal.
* `scene_version`: caller's current scene namespace string.
* `instance_id`: one current template's arbitrary physical-instance ID.
* `template`: the **unchanged current inventory template dict** for this referent. Initialization requires its index no more than 10 original samples before the current record. Same chart/scene/ID must match.
* `spatial_goal`: `{frame: 'third_undistorted_original_pixels', center_uv: [u,v], theta_deg_clockwise: angle, accept_similarity_approximation: true}`. Position is in original undistorted 1280x720 pixels, top-left origin, u right, v down. Angle is clockwise degrees **relative to the invocation template**, not a global yaw or recognized upright direction. Scale is fixed. The renderer preserves actual foreground holes and texture; it refuses clipping, translation >520 original pixels, malformed goals, or failure to acknowledge approximation. A semantic word string or unmeasured metric pose is not a legal goal.
* `workspace_polygon_uv`: at least three original-chart `[u,v]` vertices defining caller-allowed **image** workspace. Used for image goal/current-mask checks, not proof of collision-free 3D travel. Code does not use T_base_board as a plane or safety certificate.
* `tolerances`: all five keys required: `position_px` in (0,100], `angle_deg` in (0,180], `foreground_iou` in [.1,1], `stability_s` in [.1,5], `neighbor_motion_px` >0. These are caller hypotheses, not validated precision claims. Angle matching uses 10-degree search bins and symmetry ties; calibration RMS is approximately 4.42 original pixels.
* `max_duration_s`: positive seconds, at most 600, measured from current robot `t` at invocation start.

Optional fields: `obstacle_polygons_uv` (list of polygons in the same image chart, default []), `protected_templates` (fresh inventory references for already-placed pieces, default []), `cancel` (boolean, default false). Initialization checks goal overlap with supplied obstacle/protected masks and associates protected IDs. During continuation, protected disappearance returns uncertain and displacement beyond tolerance returns blocked. Static obstacle polygons are checked against the goal; a complete path or swept robot/tool collision check is **not** implemented. The graph's workspace-ring features are the image boundary, because no trustworthy per-demonstration table workspace label exists; caller workspace is a refusal/monitor constraint, not a hidden metric training label.

`preview_goal(template, spatial_goal, workspace_polygon_uv)` returns `foreground_half` `[360,640]` uint8 and `rgb_half`, plus approximation diagnostics, or raises `ValueError` for an invalid request. Every half-raster pixel corresponds to two original undistorted chart pixels. This executable renderer supplies the goal silhouette; the HLA need not draw it. The supported interface is explicit spatial goals. Raw user-target-scene images require external semantic instance association; direct target-scene parsing, metric layouts and calibrated projective rendering are not silently accepted.

## 4. Concrete candidate call

```
import policy
scene = policy.observe_scene(obs, 'scene_12', None)
# HLA's external association returns an ID from this actual inventory:
selected_id = hla_selected_instance_id
ref = next(x for x in scene['instances'] if x['instance_id'] == selected_id)
call = {
    'call_id': 'scene_12_move_4', 'scene_version': 'scene_12',
    'instance_id': ref['instance_id'], 'template': ref,
    'spatial_goal': {
        'frame': 'third_undistorted_original_pixels',
        'center_uv': [700.0,420.0], 'theta_deg_clockwise': 20.0,
        'accept_similarity_approximation': True},
    # Illustrative image polygon, NOT a commissioned robot safety workspace:
    'workspace_polygon_uv': [[160,60],[900,60],[1040,700],[70,700]],
    'obstacle_polygons_uv': [], 'protected_templates': [],
    'tolerances': {'position_px': 8.0, 'angle_deg': 10.0,
        'foreground_iou': 0.70, 'stability_s': 0.5,
        'neighbor_motion_px': 20.0},
    'max_duration_s': 30.0
}
rendered = policy.preview_goal(ref, call['spatial_goal'], call['workspace_polygon_uv'])
out = policy.act(trained_model, obs, call, None, saved_spec)
# On the next actual paired observation, with unchanged call:
if out['status'] in ('running','uncertain','blocked'):
    out_next = policy.act(trained_model, next_obs, call, out['memory'], saved_spec)
# Any uncertain/blocked output requests an externally verified stop before further motion.
formatted = policy.decode_action(out['action'], None, saved_spec)
assert formatted['enabled'] is False  # no verified controller contract, no execution
```

The numerical example is not an assertion that its layout/workspace is safe or achievable for a particular inventory. A valid current template and representable non-overlapping goal are checked at runtime. Complete candidate inference does not require controller verification; action decoding does. The HLA never supplies contact side, recorded object coordinates, motor phase, future images, or another policy's hidden state.

## 5. Lifetime, history and statuses

`act(model, observation, call, memory, spec)` returns `{action, memory, status, diagnostics}`. `action` is a finite list of seven recorded-native commanded-dq values only for `running`; otherwise null. It is never an issued action, pose increment, measured joint velocity or gripper command. Predictive dispersion is descriptive, not a safety-calibrated bound.

Memory is an opaque in-process dict containing numerical arrays, forward perception tracks, selected association, fixed goal, short cached features and monitor history; it need not be JSON serializable. Pass it only back to the same policy for the unchanged call. Up to eight observations are selected at nominal offsets 21,18,...,0 original sample rows; missing earlier history is masked. Online buffers choose the latest already-received row no later than each offset, allowing up to two rows of sampling slack; no interpolation or future filling occurs. At original 30-Hz observation cadence this matches training exactly. Recompute gated memory from this bounded causal window each call; no latent state is transferred. A 20-Hz external controller may consume latest available observations only after commissioning; dt/repeat/missing masks expose changed observation cadence, not clock repair.

* `running`: finite learned candidate, valid current selected association. Contour requires confidence >=.65; visual requires >=.3. Neither may move from a fully lost identity.
* `uncertain`: missing/ambiguous selected or protected track, lower contour confidence, or >0.5-s observation gap. Null action and stop request. A gap terminates the invocation; shorter tracking uncertainty can be re-observed under the same call while stopped.
* `unavailable`: malformed/missing data, calibration or call, stale invocation template, changed fixed inputs, non-increasing timestamp/index, terminated call reuse, or nonfinite candidate. Reasons distinguish `calibration_required` for unsupported geometry. Changing fixed arguments returns memory=None and requires a fresh invocation.
* `blocked`: initialization goal/protected-obstacle overlap, selected foreground outside supplied image workspace, or possible protected-neighbor displacement over the caller threshold. Null action; there is no invented collision/contact measurement.
* `goal_observed`: centroid/foreground/rotation-modulo-symmetry agreement continuously observed for the caller's stability duration. Terminates and returns null action. **Geometric advisory only**, not task success or safe handoff.
* `timeout`, `cancelled`: termination and null action; never success. `cancel` can change without changing the call signature.

All non-running statuses request a verified external stop. The package cannot perform that stop. Diagnostics expose track/contour proxy, foreground IoU, center error, orientation hypotheses/residual, two-second error trend, error-direction changes, unprotected/protected possible neighbor displacement, TCP projection/in-image validity, history count and mixture dispersion when available. Tool clearance, contact, re-contact count and cross-view consistency are null/unverified, not fabricated. The monitor does not automatically diagnose friction, contact loss, force anomalies or success from a plateau. `successor_ready` is always false; `controller_verified` in act is false.

Reset on instance, spatial goal, template, policy, chart, controls or tolerance change. Start a new call after goal/timeout/cancel/gap termination. Changing observation/contact side for the same instance/goal does not itself reset memory. After uncertain association, request fresh inventory if reacquisition is not unique; do not silently assign another component to the requested semantic identity.

## 6. Switching while preserving the goal

Save the old `preview_goal(... )['foreground_half']` or the fixed mask in old memory. Stop/cancel the old call using a verified external stop. Refresh current automatic inventory, externally confirm the same physical referent, and use its new template. Call:

```
conversion = policy.reexpress_goal(new_ref, old_fixed_foreground)
# Only continue if status == 'goal_reexpressed' and HLA accepts any semantic symmetry ambiguity.
new_call = dict(call)
new_call.update(call_id='scene_12_move_4_switch', template=new_ref,
                instance_id=new_ref['instance_id'],
                spatial_goal=conversion['spatial_goal'])
new_out = policy.act(other_trained_model, fresh_obs, new_call, None, other_saved_spec)
```

This computes a shape registration in the same chart and refuses poor IoU, incompatible apparent scale, or unsupported translation rather than silently replacing the goal with the current pose. Preserve absolute intended center/foreground; the angle must be re-expressed relative to the **new** template, not copied unchanged. Refresh protected references too. `reexpress_goal` returns `goal_reexpressed`, `uncertain`, or `calibration_required`, with 10-degree orientation hypotheses and achieved IoU; HLA must handle semantic symmetry. No motor history, hidden state, candidate action, auxiliary response state or normalization is transferable. A separate causal perception inventory may persist if its identity/uncertainty is still valid.

## 7. Decoder verification contract

`decode_action(action, controller_contract, spec)` is a pure formatter/refusal gate, returning `{enabled, command, reason}`. Absent/incomplete verification yields `enabled:false, command:null`. Even `enabled:true` only constructs an artifact and performs no hardware I/O.

An external commissioning system must supply ALL fields below from an actually verified controller and current live safety context, never copy a fictional example as verification:

```
{
 'verified': True, 'verification_id': verified_record_id,
 'mode': 'joint_velocity', 'units': 'rad/s',
 'recorded_dq_units': 'rad/s', 'q_units': 'rad',
 'joint_order': verified_seven_joint_names,
 'recorded_joint_order': verified_seven_joint_names,
 'timing_verified': True, 'limits_verified': True,
 'watchdog_verified': True, 'stop_verified': True,
 'workspace_check_passed': True, 'singularity_check_passed': True,
 'tool_geometry_verified': True,
 'update_hz': verified_update_hz, 'hold_s': verified_hold_seconds,
 'watchdog_s': verified_watchdog_seconds,
 'measured_latency_s': live_measured_latency,
 'max_latency_s': verified_latency_limit,
 'velocity_limits': verified_seven_positive_rad_per_s_limits,
 'acceleration_limits': verified_seven_positive_rad_per_s_squared_limits,
 'q_min': verified_seven_lower_rad_bounds, 'q_max': verified_seven_upper_rad_bounds,
 'current_q': current_verified_q_rad, 'previous_command': actually_issued_previous_rad_per_s,
 'elapsed_since_previous_s': measured_elapsed_seconds
}
```

Code checks seven unique identical recording/controller joint names, finite values, positive limits, update Hz in (0,1000], hold/watchdog in (0,1], latency <= max latency <= watchdog, dt within watchdog, velocity/acceleration bounds, current joint bounds and conservative one-hold joint bounds. It rejects, not clips, violations. It cannot verify the truth/freshness of external safety attestations or compute workspace/singularity safety from an image. Teleoperation metadata (20 Hz, joint_velocity) alone is insufficient. The output command remains the candidate dq plus verified order/units/timing; it does not integrate dq into a commanded pose, issue gripper control, fabricate a controller watchdog, or send a robot stop.

## Dependencies and evidence limits

Only interface-installed numpy/OpenCV/scipy/torch and read-only framework access during preparation. No downloaded perception/recognition assets or licenses are hidden. Data normalization buffers travel with the trained checkpoint/spec. Pixel goals and masks are not metric measurements; clearance and full-layout readability/stability remain HLA/commissioning judgments. These two episodes establish demonstration support for relocation/approach/re-contact, not validated deployment capability or a ranking between alternatives.
