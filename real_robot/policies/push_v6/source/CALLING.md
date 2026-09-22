# Calling Push v6

## Agent pipeline
1. Obtain fresh calibrated RGB and robot state. Raw invocation accepts `rgb_third` uint8 HWC, matching `camera={K,dist}`, `T_base_cam`, independently commissioned `top_z_base_m`, workspace_mask, revision, age, calibration flags and uncertainty. SAM automatic workspace-grid proposals return `associate_instances` plus 96-point metric candidate contours, NOT a motion. Proposal IDs are ephemeral: the Agent must associate physical instances from current/prior geometry and images, reject robot/background proposals, and request another view on merge/occlusion/uncertain association. No per-move human prompts are used. A cached associated Scene may instead be supplied directly.
2. Agent interprets current image crops/context into semantic identity/uprightness hypotheses, verifies distinct physical tracks for repeated characters, and rejects impossible or ambiguous requests. Unknown glyph conventions may need clarification. No closed recorded alphabet, recorded word or fixed ordering is used; semantic competence of the supplied Agent is not measured by these recordings.
3. User supplies per-instance target poses, or Agent proposes and obtains acceptance of an explicit baseline origin, reading/upright directions, spacing and layout region. Pack actual shapes without overlap, choose unblocked actors, and use staging if necessary. Convert each reference-contour target into `goal_delta=[dx,dy,dyaw]` about the CURRENT centroid in base XY. A word alone is not a geometric goal.
4. Invoke the selector, prepare with the supplied planner, reobserve/revalidate at arrival, execute at most one bounded stroke, then reobserve and measure object/neighbor response. Decide next contact/actor from current geometry, not recorded index or stage. Verify all object poses, semantic orientations and layout independently at the end.

## Associated Scene and call
```
observation = {
 'revision':11, 'age_s':0.02, 'calibration_valid':True,
 'calibration_id':'commissioned_camera_table_tool_v1',
 'uncertainty_m':0.006, 'stable':False,
 'tracks':[{'instance_id':'persistent_j', 'xy':[[x,y],...],
            'normal':[[nx,ny],...], 'loop':[0,0,...]}],
 'workspace':[[x0,y0],...], 'tool_radius_m':commissioned_radius,
 'contact_height_m':commissioned_tool_support_center_height
}
call = {'scene_revision':11, 'instance_id':'persistent_j',
        'reference_id':'causal_reference_j', 'goal_delta':[0.03,-0.01,0.2],
        'position_tolerance_m':0.008, 'yaw_tolerance_rad':0.08,
        'max_stroke_m':0.01, 'last_result':'progress'}
```
Contours have exactly 96 arc-length samples; outward normals point into free space, including holes. Coordinates/distances are meters, yaw radians. `geometry.shape_from_mask` implements conversion. Optional `causal_shape_references` is a list of previously clearly observed complete shapes in the same track dictionary format. `act` can repair an occluded hole bridge only by rigid registration with >85% bidirectional visible support; references MUST originate before the current observation and reset with changed instance/calibration. They are not stored letter templates or future masks. Raw proposals become associated tracks on the next call, at which time reference restoration is available.

The implementation assumes a commissioned approximately horizontal tabletop and uses base XY with positive base Z free space. Never silently relabel an arbitrary plane as base. Other tracks are protected. Complete trustworthy scene geometry is required: the network cannot certify that an unseen obstacle or occluded object is absent. Color is not a policy input; neutral-table hole refinement assumes a distinguishable table background, and camouflage/uncertain holes require another view. Deployment uses its own commissioned camera/table/tool geometry, not the weak offline reconstruction. No future image, achieved demonstration goal, episode/index or offline annotation is an online input.

`act` generates and filters joint candidates from geometry at invocation time. Contact means boundary point, not TCP. `proposed` returns scene revision, actor, contact, outward normal, inward unit heading, distance <=min(20 mm,0.2 diameter,caller limit), loop, contact height and uncertainty. Scores are not calibrated success probabilities or safety certificates. Disk-envelope filtering is only coarse: full robot/tool collision and insertion checks remain mandatory.

## Feedback, memory and handoff
The network is stateless. Memory holds instance/reference, objective, calibration and a nonprogress counter. Reset on instance/reference/goal/calibration change or interruption. `interrupted=True` clears memory and emits no decision. Send each execution result once; two consecutive `no_progress` outcomes emit `executor_replan`, requiring another view/contact/staging or eventual blocked report. Enforce a caller retry budget.

Statuses: `needs_view` for missing/uncertain geometry; `associate_instances` for new raw proposals; `ambiguous_geometry`; `calibration_required`; `stale_scene` for revision mismatch or age >0.2 s; `invalid_constraints`; `unsupported_precision` if positional tolerance is below scene uncertainty; `no_safe_candidate`; `executor_replan`; `interrupted`; `already_at_goal` only for an externally established stable measurement within supplied tolerances. Travel completion and recording termination never mean task success.

## Executor binding
`executor_request` serializes objectives only, with `execute=False`; it has no hardware IO. Contract requires `commissioned`, scene_revision, calibrated `T_flange_tool`, `R_base_tool`, `tool_collision_model_id`, nonempty commissioned guard_profile_id, speed and protected instances. It derives the disk support center from boundary+normal*radius, adds caller contact height, and composes tool-to-flange transforms to standoff/contact/end poses in base. The full supplied planner/controller must provide hover/descent, safe approach, hole insertion using the entire tool envelope, selected-object contact allowance, line tracking, subdivision to <=1 s, force/torque/tracking/staleness/contact guards and withdrawal. Fresh observation/revalidation is required AFTER prepare and BEFORE stroke. No queued stale continuation is authorized. This serializer is not a verified physical controller adapter; commissioning/guard and collision integration remain blockers to robot deployment.

## Executable check scope
Checks after trained-weight reload include: actual retained contour with synthetic workspace/timing/commissioned-tool/executor fixtures (must propose); missing Scene; stale revision; interruption/reset; absent calibration; stable geometric completion; unsupported precision; and original source RGB through generic SAM grid proposals with explicitly synthetic commissioning (must return association status). These establish tensor/serialization/status and causal preprocessing behavior, not perception precision, semantic recognition, persistent tracking correctness, safe physical execution or task success. Preparation additionally exercises SAM interior-point masks, geometry, causal hole recovery and loss/backward/predict with zero optimizer updates. Training diagnostics and interface checks are separate from unmeasured new-shape/word/robot claims.
