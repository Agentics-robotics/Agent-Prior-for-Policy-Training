# piece_goal_field_v1 — usage

Invoke `policy.act(field_model, observation, call, memory, field_spec)` using this policy's independent checkpoint. See [PRIOR](piece_goal_field_v1_PRIOR.md), [CALLING.md](CALLING.md) for every shared argument/default/frame, and [HANDOFF.json](HANDOFF.json).

Select for one reliably segmented flat piece with an explicit geometric objective, especially if its canonical angle or a mechanics fit is ambiguous. This is not a fallback for missing masks or stale calibration. The policy chooses corrective mode and candidate motion internally; the HLA chooses the physical instance, world goal and protection constraints.

## Exact piece-specific arguments

Required `policy_id='piece_goal_field_v1'`, new string `call_id`, current `now_s`; persistent application `instance_id`; current `instance_seed_W` [x,y] meters plus `inventory_t` within .5 seconds. Seed selects an instance, not an action. Supply **exactly one** fixed objective:

* `goal_SE2_W`: 3x3 rigid homogeneous transform mapping entry-footprint W points to desired W points. W x/y meters, yaw radians embodied by its rotation. This is not desired centroid plus arbitrary glyph angle.
* `goal_footprint_W`: `{outer:[ring,...], holes:[ring,...]}` in W meters. Rings are vertex lists. Goal must be a compatible rigid placement of the observed selected shape; holes are explicit, not known-font templates. For raster information use the exact CALLING W mapping to create these rings.

Optional `exit_tool_base`: 4x4 TCP base pose. `terminal_twist` is length-six **native recorded encoding**, not automatically SI; `arrival_mode` is stop/pass_through. Without an explicit exit, entry TCP is an internal exit anchor but not a required completion waypoint. For a semantically directed glyph, supply `semantic_orientation={goal_axis_W:[2], observed_axis_W:[2], observed_t:seconds}` from a real causal semantic estimator; absent/stale observed axis prevents semantic completion. No such estimator or verified letter identity is hidden inside this model.

Causal observation: original third/wrist RGB uint8, q/measured dq/tau, full TCP/flange transforms, calibration metadata, live t/sample_index and image ages/stamps, per CALLING. Saved spec contains normalizers and device; deployment supplies no endpoint/episode/segment/action_json. W is the documented board-proxy frame unless caller provides verified top-plane workspace. Three-dimensional TCP exits remain in base frame.

Optional shared controls and defaults: timeout_s=30, max_age_s=.25, tolerance_m=.01, tolerance_rad=.08, speed_cap_m_s=.03, clearance_m=.02 requested/unverified, native_abs_cap=3*saved scale, retry_budget=8, stall_s=8, neighbor_tolerance_m=.015, protected_instance_ids/current instance_bindings_W, workspace, current live_mask/live_mask_t, actual last_executed_action/last_command_verified, guards, audited intent_mapping, rod_geometry and abort. Physical speed reduction/planar screen require the audited intent matrix; clearance, rod/arm sweep and decoder remain unverified. The field network uses proxy geometry, not pixel displacement misrepresented as velocity.

## Example: translate a selected current footprint

```python
scene = policy.inventory(obs, field_spec)
obj = scene['instances'][0]  # verify that this is the intended current instance
call = {
    'policy_id':'piece_goal_field_v1', 'call_id':'align-slot-9',
    'instance_id':'piece-physical-9', 'instance_seed_W':obj['seed_W'],
    'inventory_t':scene['t'], 'now_s':float(obs['t']),
    'goal_SE2_W':[[1.,0.,.015],[0.,1.,-.01],[0.,0.,1.]],
    'tolerance_m':.01, 'arrival_mode':'stop', 'timeout_s':25.,
    'speed_cap_m_s':.02, 'protected_instance_ids':[]
}
r = policy.act(field_model, obs, call, None, field_spec)
```

To preserve a goal across interruption, store its actual world rings and pass them as goal_footprint_W at the next acquisition. Reusing the old displacement transform after the entry footprint changed would move the objective.

## Returned state and handoff

Outputs: finite native `[vx,vy,vz,wx,wy,wz]` **candidate** or null; opaque memory; global status; diagnostics. Normal usable candidate status is CONTROLLER_UNVERIFIED. Geometric/contour error, tool error, symmetry flag, inferred mode, proxy proximity, uncalibrated uncertainty, neighbor motion and freshness/readiness distinguish progress from physical readiness. REACHED_LOCAL_GOAL needs three fresh observations, caller-verified plane, tolerance and requested exit/actual-stop checks, plus optional semantic axis. It is not word acceptance, a verified orientation classifier or enabled robot operation.

PERCEPTION_UNCERTAIN rejects stale/lost/swapped/ambiguous inputs; BLOCKED covers changed protected pieces and implemented guard/sweep checks; NEEDS_REPOSITION covers stall/reset limits; ABORTED covers timeout/interruption/changed fixed objective. Null action is not an executed stop. RUNNING is not emitted as hardware execution in this package. `decode_action` always remains disabled pending an actual verified follower implementation.

Memory keeps target identity, fixed goal, protection baseline and twelve causal feature records; four snapshots feed the recurrence. Start memory=None on new policy/call/goal/frame or reacquisition after loss/gap. Do not transfer learned recurrent state from the graph or waypoint policy. On switching, preserve desired world geometry, reacquire the same physical instance, transform the goal if W changed, compare old/new neighbor geometry, reset memory, and supply actual last-command feedback. Agent judgment is still required for jams, object tipping, semantics, corridor feasibility and collision models not present in the data.
