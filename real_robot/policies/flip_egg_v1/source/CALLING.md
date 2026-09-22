# Agent calling contract

## Scope and required binding
`egg_interaction_v1` is the sole callable learned motion policy. Its perception is learned internally, not an assumed pose/mask service. `executor_request` creates a proposed Cartesian contact segment and does no hardware I/O. This process has no robot connection. Generic planning alone is NOT compliant contact control. Do not deploy until the executor has validated collision geometry including tool and rim, retained-tool grasp/hold I/O, workspace interlock, torque/error stop logic and bounded contact tracking. Teleop speed and 25N/6Nm metadata are not approved safety settings. Missing geometry or a safe explicit grasp objective blocks acquisition; the library does not claim automatic pickup. Initial use may start from an independently verified retained-tool state.

## Online observation schema
`act(model, observation, call, memory, spec)` uses no recording/cache access. `spec.preprocessing` fixes `calibration_id=flip_20260919_recorded_rgb_tcp_v1` and crops. Each observation contains:
- `observation_id`: unique string for the current bundle; never reuse.
- `now`: current timestamp seconds in the **same clock** as robot samples.
- `samples`: one to five chronological causal frames, spaced .05-.20s. Usually supply one new frame per 10Hz call; on entry/resume may supply a recent five-frame bundle. Never resubmit frames already in memory.
Each sample has `t` seconds, `T_base_ee[4,4]` homogeneous SE(3) in base metres, `tau_ext[7]` joint external torque estimates Nm, `gripper_width_m` metres, `image_age_s[2]` ordered third/wrist, `third_rgb` and `wrist_rgb` original uint8 HWC [720,1280,3]. A camera may be null; two missing/unusable current cameras return invalid. Images are recorded paired frames, not independently realigned by timestamp. Age tolerance [-.02,.15]s accommodates documented negative host-age skew; does not certify exposure synchrony. Older than .1s robot state, repeated ID or nonincreasing times is rejected. Unexpected cadence clears history. Real runtime must separately monitor device resets and timestamp/transport faults, updating validity instead of silently changing the pairing.

Third crop original pixels [40,0,1040,720]; wrist full frame. Resize preserving aspect and center-letterbox to [144,192]. No intrinsics needed by this image-only actor; executor still needs verified transforms. No raw metric depth is used. A setup ID match is necessary but not proof of calibration.

## Call arguments and persistent goal
Required fields:
```
{
 session_id: 'new-task-id', reference_id: 'immutable-initial-reference-id',
 task: 'turn_over_in_same_pan', tool_template: 'demonstrated_spatula',
 calibration_id: 'flip_20260919_recorded_rgb_tcp_v1',
 initial_third_rgb: uint8[720,1280,3], reset: true,
 tool_retained: true, scene_valid: true, supported_context: true,
 authorized_for_proposal: true,
 max_predictive_spread: float[6], feedback: 'ok'
}
```
`initial_third_rgb` must have been captured at task start before interaction. This is the initial **scene** reference, not a segmented egg-face oracle. Keep the same reference and same-pan goal across retries. Required on reset; on continuation it is stored in memory. The Agent separately keeps original-resolution images for outcome comparison. Do not redefine the target from a later apparent face. No phase, original index, elapsed episode time, future pose, arbitrary landing point, mask or fitted blade transform is accepted as conditioning.

Boolean confirmations are CURRENT external observations/authorizations, not quantities inferred from width. `supported_context` means the egg is observed in the pan or visibly supported on the retained spatula during a continuing attempt, without off-pan/drop/unknown support needing intervention. A short observed airborne release must be followed by observation/arbitration, not a blind recovery. These checks do not replace a safety-rated interlock. `max_predictive_spread` specifies six positive commissioning limits in m/s (first3) and rad/s (last3); values must be chosen from validation and integration, not from the permissive calling fixtures. Predictive spread is not a sufficient safety detector.

Feedback interrupted/infeasible/contact_limit/tool_slip/tracking_error/no_progress, or `interrupt=true`, cancels and clears policy memory. Revalidate tool and scene, preserve the original reference in Agent memory, set reset=true, and optionally supply recent causal samples. A changed reference/session without reset is rejected. Agent must impose a no-progress timeout and manually authorize any retry. No learnt retry-after-drop or success detector is installed.

## Learned tensors
For each of five causal slots: two RGBs `[2,3,144,192]` in [0,1]. Feature `[28]` is position3 normalized (p-[.45,0,.15])/[.25,.35,.3], first two rotation columns transposed and flattened6, tau7/5, width/.14, clipped image ages2/.15, camera-valid2, previous dt/.1, previous achieved local six-rate/[.1,.1,.1,.5,.5,.5]. First valid slot has zero previous dt/rate. A slot mask disables missing history. Initial reference `[3,144,192]` is encoded separately. Training and online paths share this preprocessing; future target row i+3 and all target/outcome audits are absent online. Small appearance/camera dropout/history truncation apply only during training. Memory is a JSON-serializable rolling buffer of at most5 preprocessed observations plus initial reference, IDs; no unbounded recurrent state or episode clock.

## Decision and geometry decoding
Status active returns the JSON decision validated by package decision_schema, plus memory. It contains `velocity[6]` and `spread[6]`, `frame=current_recorded_ee`, current `T_base_ee`, observation/session IDs, timestamp, expires_at=timestamp+.1, horizon=.1, retain-grasp instruction, completion=unknown. First3 rates are current EE-origin linear velocity expressed in EE axes (m/s); last3 local rotation-vector rate (rad/s). They are **not blade-tip rates**, not joint velocity, not Cartesian force, and not a base-origin spatial twist.

For T=[R,p], u=[v,w], h=.1:
`p_goal=p+R*v*h`, `R_goal=R*Exp(w*h)`.
Executor inputs include current `now`, explicit validated `T_flange_ee[4,4]`, physical speed bounds, workspace bounds `[min_xyz,max_xyz]`, `scene_geometry_id`, `safety_profile_id`, and all four true validation flags: contact_tracking_validated, collision_geometry_validated, workspace_interlock_clear, retention_validated. Adapter returns an EE goal and `T_base_flange_goal=T_base_ee_goal*inverse(T_flange_ee)` plus a strict straight-origin/geodesic short-path contract. Actual controller must independently compare live start state, check swept robot/tool/rim collision and force/error limits, and reject if it cannot preserve local path/timing. No silent rerouting/retiming/clamping. At expiry stop via installed safe-stop semantics, not an assumed safe hold in arbitrary contact. No queuing stale proposals.

## Illustrative invocation sequence
1. Agent observes initial scene and creates session/reference. Executor or supervised setup uses explicit verified grasp/approach/lift geometry and gripper binding. If not installed, stop: there is no automatic pickup policy.
2. After visual co-motion/retention confirmation and workspace clearance, call policy with causal samples and immutable initial RGB. Warm-up of fewer than5 frames is supported with masked history.
3. If active, call executor_request with validated binding and current state. It may still reject. Otherwise request the <=.1s contact segment. Executor reports geometry arrival/error, NOT flip success.
4. Reobserve at10Hz, pass only new sample(s) and previous memory. If invalid/uncertain/interrupted, cancel; Agent reobserves and arbitrates. Preserve reference across a supported retry.
5. After observed release, wait for inspectable pan-resting egg and empty-tool clearance. Agent compares faces and support against initial reference. Unknown is allowed. Only Agent issues completed_flip. A safe empty-tool retreat is an explicit geometry request outside this policy.

## Executable checks and scope
`calling_cases` uses held-out recorded images/state (first retained test anchor) plus **synthetic permissive authorization/safety/geometry fixtures**. Tests full history, reset with single frame, one-view masking, both-view loss, stale state, missing reference, interruption, absent retention, unsupported goal and unbound executor rejection. It asserts pixel/feature parity with prepared cache, SE(3) decode/EE-flange roundtrip and rejection without a validated contact binding. These are deterministic software-interface tests, not measured calibration, contact safety or task success. The numerical preparation also checks target reconstruction against real i+3 EE transforms.
