# tool_waypoint_v1 — usage

Entry point: `policy.act(model, observation, call, memory, spec)`. This model must have been loaded from the **tool_waypoint_v1** checkpoint; see [PRIOR](tool_waypoint_v1_PRIOR.md). [CALLING.md](CALLING.md) supplies the complete shared field table, frame/raster conventions, statuses and exact optional-adapter contracts.

Choose this policy for a **TCP waypoint without intentional object motion**, including inspect/recontact-side changes. Do not use it to push the target piece or assume a visually occluded rod is free of a hole. Physical corridor/escape feasibility is an HLA judgment and is not certified by this package.

## Required arguments

* `policy_id='tool_waypoint_v1'`, new string `call_id`, `now_s` in the observation clock.
* `goal_tcp_base`: finite 4x4 rigid TCP pose; translation meters, rotation matrix, **base** frame. It is not a rod-tip pose.
* Current causal `observation` with dual original RGB, calibration, flange/TCP transforms, q/measured dq/tau, t/live sample_index and image ages, as defined in CALLING.
* Saved deployment `spec` with native normalizers/device. No source IDs or endpoint images.

Set `arrival_mode='stop'` with zero native `terminal_twist=[0.]*6`, or `pass_through` with an explicitly chosen native terminal twist (same encoding as recording, not an unverified SI command). Optional shared controls: timeout_s, max_age_s, tolerance_m/rad, protected_instance_ids plus current instance_bindings_W, speed_cap_m_s, clearance_m, native_abs_cap, stall_s/retry_budget, neighbor_tolerance_m, guards, workspace, fresh live_mask/live_mask_t, intent_mapping, rod_geometry, abort. See the shared table for defaults and exact structures. `last_executed_action` and `last_command_verified` refer to actual execution feedback, never this policy's candidate. Completion of a stop requires verified actual zero intent and measured |dq|<.03.

The default board-plane proxy is unverified. Physical clearance_m is reported, not enforced as certified rod clearance; the analytic route uses a fixed .10-m W TCP lift floor. The physical speed cap is enforced only with a verified native-to-base-twist intent matrix; no hardware decoder is enabled even then.

## Concrete call

```python
waypoint = np.asarray(obs['T_base_ee']).copy()
waypoint[2,3] += 0.04  # base-z lift objective; inspect the actual escape corridor
call = {
    'policy_id':'tool_waypoint_v1', 'call_id':'inspect-before-contact',
    'now_s':float(obs['t']), 'goal_tcp_base':waypoint.tolist(),
    'arrival_mode':'stop', 'terminal_twist':[0.]*6,
    'tolerance_m':.012, 'tolerance_rad':.08, 'timeout_s':12.,
    'speed_cap_m_s':.02, 'protected_instance_ids':[]
}
r = policy.act(tool_model, obs, call, None, tool_spec)
# r['action'] is a numerical native-encoding candidate, never an enabled robot command.
```

All other detected pieces are protected even when names are not bound. Bind protected names with current W seeds if you need stable agent-visible neighbor names.

## Output, readiness and memory

A finite six-vector candidate normally comes with `CONTROLLER_UNVERIFIED`; null action comes with a reasoned uncertain/blocked/reposition/aborted status. REACHED_LOCAL_GOAL needs three fresh observations, verified plane, waypoint and orientation tolerance, requested arrival checks and no uncertain neighbors. It does not authorize actuation or word acceptance. Diagnostics include tool distance/angle, achieved base pose, lift/reset/approach/exit/hold mode, changed neighbors, native uncertainty, stale/check flags and readiness. `physical_ready`, `safe_to_handoff` and hardware-enabled remain false.

Memory caches at most twelve causal features plus instance tracks and current-call baselines. Four snapshots are recurrently encoded; no hidden state is transferred to another policy. Reset memory=None on new call/waypoint/frame, interruption, >.5-s tracking gap or policy switch. Preserve the intended physical piece goal externally while doing a tool reset. When resuming a piece policy, reacquire the same instance, preserve its fixed world goal or recompute entry-relative SE(2), and supply a fresh actual last-command observation. Do not blindly reuse an old relative displacement after the piece moved.

Failures: stale views, unexpected object movement, goal timeout/stall, joint/tau guard violation where assets exist, blocked planar screen, and uncertain tracking. Large measured joint reconfiguration is advisory, not automatically diagnosed as a singularity. Free-space success and autonomous jam recovery are untested, not promised.
