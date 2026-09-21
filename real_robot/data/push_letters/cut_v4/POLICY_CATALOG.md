# Proposed callable policies (not implemented or trained)

## access_clearance_v1

Dataset: `tool_access`; heuristic: `clearance_phase_prior`.

Learn a goal-relative, obstacle-aware tool-access policy with a phase/hysteresis prior and a kinematic command residual.

### Caller arguments

policy_id=access_clearance_v1; goal_tool_pose T_base_ee, mode=approach|depart|transfer, protected_instance_ids, optional target instance/boundary region, tolerance/time budget/safety profile.

### Input output contract

Group tool_access. Causal RGB/state/history through P/S; propose finite seven-joint velocity through D. Own obstacle avoidance, approach phase and stopping, not character assignment or letter displacement.

### Memory and handoff

Reset phase state on new call; initialize with up to 0.5 s available history and validity masks; preserve tracker IDs across calls. Return actual pose/clearance, never assume next object is contacted.

### Selection cues

Choose before another policy when tip is high, on wrong side, obstructed, or a previous push finished in contact. Prefer this over arbitrary long motion from an alignment policy.

### Status and progress

Report RUNNING phase, waypoint error, clearance margin, uncertainty; READY only after stable safe entry. BLOCKED/LOST_TRACK/SAFETY_STOP include responsible obstacle/measurement.

See the dataset heuristic document for applicability, limitations, and handoffs.

## push_boundary_v1

Dataset: `piece_relocation`; heuristic: `boundary_response_prior`.

Learn geometry-shared contact proposals and a short-horizon object-response ensemble for closed-loop single-piece relocation.

### Caller arguments

policy_id=push_boundary_v1; instance_id, fixed goal footprint and optional verified SE(2), orientation ambiguity set, protected IDs, E, tolerances/time/safety.

### Input output contract

Group piece_relocation. Current causal RGB/geometry/proprioception/history -> proposed seven-joint velocities through D plus predicted object response/mode confidence. Own local contact choice, same-piece recontact and departure.

### Memory and handoff

Reset goal-specific state on call; carry causal tracker state and past-response uncertainty only. Keep model adaptation local to current tool/material context; reset on shape/material change or track loss.

### Selection cues

Prefer when contours and table geometry are reliable, especially unseen shape with clear boundary, large translation/turn or deliberate staging. Switch to visual alternative for geometric ambiguity only if semantic identity and safety remain reliable.

### Status and progress

Expose contour/SE(2) residual, predicted/observed displacement, ensemble disagreement, active boundary region and stall count. Report LOCAL_GOAL_REACHED, NEED_REPOSITION or BLOCKED with evidence.

See the dataset heuristic document for applicability, limitations, and handoffs.

## push_visual_v1

Dataset: `piece_relocation`; heuristic: `dual_view_mask_goal_prior`.

Learn a selected-instance, fixed-goal-mask dual-view temporal command policy with short receding-horizon chunks.

### Caller arguments

policy_id=push_visual_v1; same instance_id/goal/protected_ids/E/tolerance/time/safety interface as boundary alternative; caller need not choose a contact point.

### Input output contract

Group piece_relocation. Causal dual RGB/proprio/history plus current masks and fixed goal raster -> short command proposals through D and diagnostic confidence. Own same-piece pushing/recontact, not word semantics.

### Memory and handoff

Initialize from bounded current history; reset action chunk on new goal, track uncertainty or stop; no hidden state copied from boundary policy. Transfer only shared explicit observations/track state.

### Selection cues

Choose for robust dual-view observations with unreliable hard contour pose, or as an independently validated alternative after a geometric proposal stalls. Do not select solely because a character is unseen.

### Status and progress

External geometric/image residual and internal ensemble/chunk uncertainty, camera availability and track confidence. Explicit BLOCKED/LOST_TRACK rather than action completion by elapsed time.

See the dataset heuristic document for applicability, limitations, and handoffs.

## align_local_v1

Dataset: `local_alignment`; heuristic: `local_response_trust_prior`.

Learn local goal-error correction with contact-mode memory, online observed-response adaptation and a conservative trust region.

### Caller arguments

policy_id=align_local_v1; selected instance_id, fixed goal footprint/optional verified SE(2), orientation alternatives, protected IDs, E, tolerances, maximum corrections/time and safety.

### Input output contract

Group local_alignment. Causal local geometry/dual RGB/proprio/history -> short joint velocity proposals through D plus local response/uncertainty diagnostics. Own correction/contact reset/release, not object reassignment.

### Memory and handoff

Initialize from current0.5s history; reset adaptation on instance/goal change or track loss; preserve only causal explicit response summaries when resuming same call. End status includes whether tip is still near contact.

### Selection cues

Choose for late slot/orientation error, revisit after neighbors move, or a coarse policy reaches near-goal but not desired tolerance. Choose relocation for a large move, access for wrong-side tool placement, stop for unreliable identity.

### Status and progress

Report local contour/angle error, predicted versus observed stroke response, protected displacement, contact mode, retry count and uncertainty. Terminal status based on current verification, not low imitation loss.

See the dataset heuristic document for applicability, limitations, and handoffs.
