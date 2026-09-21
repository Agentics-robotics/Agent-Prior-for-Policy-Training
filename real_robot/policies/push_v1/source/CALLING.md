# Calling the cut_v3 policy package

## Public functions and ownership

```python
from policy import inventory, act, decode_action
# Numerical executor hooks, not a user training loop:
from policy import prepare_data, make_batch, build_model, compute_loss, predict
```

The executor creates/trains/reloads each independent model and its saved `spec`.
For inference use that model's own saved normalization and `spec`, including
`policy_id` (`contour_push` or `visual_push`) and the actual tensor `device`.
Do not mix checkpoints, normalizers or memory across policies. Models return
learned **candidate recorded commands**, never motor IO. No controller contract
is needed for offline prediction or `act` candidate computation.

The HLA supplies **which current physical instance**, **one fixed spatial goal**,
one policy, and an invocation lifetime. It does not supply contacts or individual
move labels. Word/character interpretation, readable orientation, allocation of
distinct repeated-character pieces, layout, ordering and re-planning are external
HLA/recognizer responsibilities. This package has no open-alphabet recognizer.
A string such as `CAT` is not a legal motor call.

## One raw paired observation

Pass a dictionary for ONE actual current record, preserving original pairing:

| Field | Contract |
|---|---|
| `third_rgb`, `wrist_rgb` | uint8 RGB arrays `[720,1280,3]`, original distorted camera pixels, not previews/BGR/crops |
| `q` | finite seven-vector of recorded joint positions in recorded order |
| `dq` | measured joint velocity seven-vector; NOT the command; nonfinite/missing components get validity masks |
| `tau_ext` | seven-vector of measured residual values, not calibrated contact force; missingness permitted |
| `T_base_ee`, `T_base_flange` | finite `[4,4]` recorded transforms; base coordinates, recorded translation units (metadata uses meters); no extra TCP transform |
| `gripper_position`, `gripper_width_m` | original values, with negative position/nonfinite width invalid; null/missing is allowed and masked |
| `t` | finite robot-record time in seconds; strictly increasing within a call; used for causal dt, budget and stability, not episode-phase conditioning |
| `recv_time` | recorded receive timestamp, or missing with validity bit; no clock repair |
| `sample_index` | nonnegative integer original paired row index; strictly increasing for new call observations |
| `img_t_third`, `img_t_wrist` | original paired image timestamps; missing values masked; deltas/repeat flags only, not absolute clock input |
| `img_age_third`, `img_age_wrist` | finite signed original image ages in seconds; absolute age must be <=0.5; negative ages are not realigned |
| `metadata` | episode/live commissioning metadata with `calibration` described below; TCP/teleop fields should remain available for audit |

`metadata.calibration` must contain `generated` (stable chart-version string),
`third` with `K`, `dist`, `width=1280`, `height=720`, `T_base_cam`, and `wrist`
with its **calibrated** `K`, `dist`, width/height and `T_flange_cam`. Brown-Conrady
undistortion preserves K and original chart size. The wrist camera base pose is
`T_base_flange @ T_flange_cam`. The recorded EE already includes the tool TCP.
The package does not use depth or mistake `T_base_board` for a tabletop plane.

No action_json, goal anchor image, demo segment ID, object-coordinate table or
future frame is accepted as an online observation requirement. Images with
repeated exposure timestamps are retained as paired records with repeat flags.
Absolute sample indices only select causal history; they are never neural inputs.

The converter runs identically during preparation and online use. Robot numeric
values and masks, timestamp deltas, signed image ages, camera geometry and
selected/goal chart coordinates enter state features. Saved weighted training
statistics standardize them. Action mean/std are saved model buffers and invert
normalization to the native recorded seven-command output. Neither this numerical
scaling nor `joint_velocity` metadata verifies hardware units.

## Automatic inventory and observation templates

```python
inv = inventory(observation, scene_version="scene_12", memory=inventory_memory)
inventory_memory = inv["memory"]
```

`inv["instances"]` contains automatically proposed arbitrary scene-scoped IDs,
current pixel centroids, heuristic confidences, and executable `template` objects.
There are no letter names. The HLA/its external semantic module chooses among
these *actual observed* instances; it must not retrieve a training object by name.
Keep `inventory_memory` to preserve causal scene identity between calls. Use a
new `scene_version` after scene replacement, reset, camera change or loss of
identity. An ID that disappeared permanently is not silently reassigned.

Each returned template contains:

- `instance_id`, `scene_version`, `template_id`, `calibration_version`;
- `reference_t`, current observation robot time;
- binary **0/1** `[720,1280]` `mask` in the UNDISTORTED chart;
- uint8 RGB `[720,1280,3]` `rgb`, already masked to this foreground;
- `confidence`, an uncalibrated material/association heuristic.

These are observation-derived numerical assets, not per-move human clicks or
known-alphabet glyph templates. Pass them unchanged. A template for a new motor
invocation must be within one second of the current record and have confidence
at least .4. The mask/appearance is re-associated automatically on entry. It is
then only the initial reference, **not** the position used to choose later actions.
During the call a forward tracker follows the moving selected ID.

The automatic detector requires approximately recorded-scale, separated warm
material against contrasting neutral surroundings. It preserves holes, excludes
neutral robot/tool pixels by contrast and leaves occlusion missing. It can reject
empty/ambiguous components and may fail on touching objects or new materials.
There is no generic segmentation weight claim. `inventory` can return unavailable;
that is a real observability failure, not a request for manual per-step labeling.
Do not treat confidence as a calibrated probability or safety certificate.

## Call dictionary

Required fields:

```python
call = {
    "call_id": "scene_12/relocation_3",      # unique invocation label, not a motor feature
    "scene_version": "scene_12",
    "policy_id": "contour_push",           # optional, but must match the loaded spec if supplied
    "instance_id": chosen["instance_id"],
    "template": chosen["template"],
    "spatial_goal": {
        "center_uv": [goal_u, goal_v],
        "angle_deg": desired_relative_angle,
        "accept_similarity_approximation": True
    },
    "workspace_polygon": allowed_pixel_polygon,
    "obstacles": protected_instance_references,
    "tolerance_px": 8.0,
    "tolerance_deg": 8.0,
    "max_duration_s": 30.0,
    "neighbor_motion_threshold_px": 20.0
}
result = act(model, observation, call, memory=None, spec=saved_spec)
motor_memory = result["memory"]
```

`workspace_polygon` is a list of >=3 `[u,v]` image-chart vertices within
`[0,1279] x [0,719]`, nondegenerate. It is an allowed **image** region, not verified
3-D workspace geometry. Obtain it from the HLA's commissioned/observed scene
constraints, not a guessed metric rectangle. The package validates containment
of selected and rendered goal foreground. It cannot certify robot/tool-path
clearance from a 2-D polygon.

`obstacles` is optional, default `[]`. Each protected reference is
`{"instance_id": ..., "calibration_version": ..., "mask": binary_full_chart_mask}`,
normally drawn from current automatic inventory templates. The selected instance
is excluded. Overlapping protected goal/current masks cause refusal. Other live
observed neighbors are also monitored relative to invocation positions; >20 px
(default, configurable finite value >=5) displacement causes uncertainty and asks
the HLA to check the scene. Masks must be in the same undistorted chart. This
monitor is not a collision planner or a proof that unseen neighbors stayed still.

`policy_id`, `obstacles`, `neighbor_motion_threshold_px` and `cancel` are optional.
Required tolerances: 5..100 pixels, 5..180 degrees. Durations: .1..300 seconds.
Thresholds are explicit engineering settings, not measured control precision;
the supplied calibration reprojection RMS is approximately 4.42 pixels.

### Coordinates, orientation and rendering

Without a plane calibration, goal `center_uv` is the desired centroid in the
UNDISTORTED third-camera chart: u right, v down, original 1280x720 pixel units.
`angle_deg` is clockwise relative to the template's observed orientation.
No semantic upright angle is inferred. Symmetric foreground registration returns
multiple competing angle modes; it does not decide M versus W or other readings.

`render_goal` rotates/translates the actual observed foreground and masked RGB,
not a stored letter/font image. It preserves holes. A no-plane request needs
`accept_similarity_approximation: true`, has no caller scale parameter, and must
move the centroid <=160 pixels. Larger moves return
`calibration_required_for_large_perspective_change`. The limit is a conservative
implementation gate, not a perspective-error guarantee. Rendered area/clipping,
workspace bounds and protected-instance overlap are checked.

For a commissioned planar transformation, include instead:

```python
call["spatial_goal"]["plane_chart"] = {
    "verified": True,
    "commissioning_id": plane_calibration_id,
    "H_image_to_plane": commissioned_3x3_matrix
}
```

The homography maps undistorted chart coordinates to a right-handed chosen 2-D
plane chart, with consistent but otherwise arbitrary plane-length units. It must
have been commissioned for the relevant piece surface/table geometry, not copied
from `T_base_board`. In this mode `center_uv` still selects an image centroid,
while `angle_deg` rotates the piece in that plane chart; the transform is mapped
back projectively. The implementation performs this conversion, checks finite
invertibility/horizon and rendered area, and refuses inconsistent results. The
verification field is a caller commissioning attestation, not something this
package can verify from the two recordings. Metric layout generation upstream
must use that commissioned chart; the motor API does not invent inverse depth.

An optional `spatial_goal.foreground` is a full binary chart silhouette supplied
after instance association to a target. The deterministic rendered mask must
have IoU >=.7 with it, or the request is refused as not rigidly realizable by the
observed template. This is a consistency constraint; it does not invoke a target
scene recognizer. Raw word/target-scene interpretation is external. The rendered
mask/RGB, center and transform are available in memory/diagnostics and are fixed
for the call. Any changed template/instance/goal/controls starts a new invocation.

### Example of an HLA-generated local placement

```python
# `chosen` is selected by the HLA from inv['instances']; no fixed training order.
u, v = chosen['center_uv']
call['spatial_goal'] = {
    'center_uv': [u + 40.0, v],
    'angle_deg': 20.0,
    'accept_similarity_approximation': True
}
# Keep the actual commissioned/observed workspace and protected references.
# If the rendered shape falls outside them, act returns blocked/unavailable.
result = act(model, observation, call, None, saved_spec)
```

The numerical offset is only an invocation example, not a pushing controller or
a trained success claim. A different selected instance or layout requires a new
fresh template and new call. HLA policy switching loads the other trained model,
keeps the intended physical placement, and expresses relative orientation against
its new current template. Hidden state is never transferred across policies.

## Memory, history and statuses

`memory` is an opaque Python/numeric dictionary containing calibration maps,
forward tracker state, the fixed goal/template, recent converted observations,
monitor values and call identity. Do not mutate it, feed it future observations,
or serialize model hidden state from another invocation into it. Its snapshot of
the call is compared exactly; changes reset motor memory. `memory=None` always
starts a fresh call. Inventory memory and motor memory are separate.

Graph history is the last up-to-eight original paired rows. Visual history uses
original row offsets [14,8,4,2,0]. Missing prefixes or skipped online rows have
explicit masks, never nearest-timestamp replacement. A 20-Hz consumer of a 30-Hz
paired stream can therefore have sparse context without clock realignment. It
must preserve the source row index. The network recomputes a bounded call-local
GRU from the causal window, just as in training; no unbounded hidden memory is
carried between cuts/calls. Robot/image dt and repeat flags retain actual timing.
A duplicate/out-of-order record produces `waiting` and no new action.

Every result has `action`, `memory`, `status`, `diagnostics`:

- `running`: finite seven-element learned candidate in **native recorded command
  dq units/order**; candidate is NOT measured dq, integrated position or a gripper
  command. `execution_enabled` is false regardless of candidate availability.
- `uncertain`: no candidate; lost/fragmented or incompatible selected identity,
  insufficient contour observability, or possible neighbor movement. The visual
  alternative may use lower contour quality but still requires observed identity.
- `unavailable`: invalid raw inputs, calibration, template association, goal or
  shape-rendering arguments; diagnostic reason identifies the failed condition.
- `blocked`: observed workspace/protected-foreground conflict; no learned command
  is claimed to resolve the hazard.
- `waiting`: no strictly new paired observation; no synthetic repeated/zero action.
- `timeout`: caller duration exceeded; not success.
- `cancelled`: `cancel: true`; clears motor memory and requests verified stopping.
- `goal_geometry_observed`: centroid/orientation/foreground agreement (IoU >=.78)
  sustained for .5 seconds; **no candidate**, and no claim of safe withdrawal,
  tool clearance, word readability or task completion. Request verified stop and
  independent clearance/readability assessment before continuing.

When observable, diagnostics include selected confidence, current area/reference
ratio, centroid error, foreground IoU, rotation modes/ambiguity, recent discrepancy
trend, oscillation-count proxy, possible neighbor movement, projected tool point,
weak edge support and mixture dispersion in native command units. Registration
uses only current/reference/goal masks. Angular modes are silhouette similarities,
not calibrated metric yaw. Oscillation is not verified re-contact; recontact count,
contact truth, metric tool clearance and word success are returned as null.

No complete rod silhouette or width model is available. Nearby image edges support
a projected TCP confidence cue but cannot establish clearance or safe travel.
The input/goal pipeline can produce real learned candidates without that asset;
it cannot authorize hardware motion or certify safe successor readiness.

## Mandatory decoder gate (no actuation)

```python
gate = decode_action(result['action'], controller_contract=None, spec=saved_spec)
# gate == {'enabled': False, 'command': None, 'reason': ...}
```

Absent verification MUST NOT be replaced by a guessed 20-Hz/rad/s interface.
The recorded teleop metadata says joint_velocity at 20 Hz, paired recording 30 Hz;
that does not establish controller order, units, latency or hold/watchdog behavior.

To return a *serialized dry-run command*, the caller must supply all of:

- True flags `recording_verified`, `units_verified`, `joint_order_verified`,
  `timing_verified`, `limits_verified`, `watchdog_verified`, `stop_verified`,
  `tool_verified`, `workspace_verified`, `current_state_verified`,
  `singularity_check_passed`, `workspace_check_passed`.
- Nonempty `verification_id` for commissioning and `safety_check_id` for the
  current check; `source_field: "action_json.dq"`, `mode: "joint_velocity"`.
- `native_units` and identical `controller_units`, explicitly verified `rad/s`
  or `deg/s`. No unit conversion is silently applied. `joint_position_units`
  must respectively be `rad` or `deg`.
- `recorded_joint_order` and identical `joint_order`: seven unique joint labels
  in the actually verified recorded order. They are not guessed by this package.
- Positive finite `command_rate_hz`, `hold_s`, `watchdog_s`, `max_latency_s`,
  `dt_s`; finite nonnegative `command_age_s <= max_latency_s`; hold <= watchdog.
- Finite seven-vectors `velocity_limits`, `acceleration_limits`, `q`, `q_min`,
  `q_max`, `previous_command`, `checked_action`. Limits must be positive and
  position bounds correctly ordered. `checked_action` must exactly equal the
  candidate to bind external current workspace/singularity checks to THIS vector.

The gate checks current and constant-hold position bounds, velocity and
acceleration limits, command age and all assertions. The endpoint computation is
only a conservative envelope test; the emitted command remains joint velocity,
not an integrated position target. A violation returns `enabled: false`, null
command and a reason. It does not clip the candidate or invent a safe stop.

If every gate passes, `enabled: true` means **dry-run serialization only** with
mode, dq, units, joint order, rate, hold, watchdog and verification ID. These are
caller attestations; this package has no controller-verification infrastructure,
hardware authorization, network/client code or robot IO. Actual actuation remains
outside scope. Missing verifications never prevent offline training/prediction.

## Reproducibility and honest limits

Use the saved package metadata, model and normalizers together. Numeric caches
preserve source episode/index/segment provenance and report every filtering count.
No extra pretrained model/license or segmentation label asset is needed to run
this implementation. This adaptation is material-contrast-limited, not a general
open-world vision model. The two independent policies train for the fixed budget
on two episodes; their performance, unseen-shape/word generalization, collision
avoidance and physical precision are unvalidated. Do not read training loss,
`running`, accepted goal arguments or geometric agreement as robot success.
