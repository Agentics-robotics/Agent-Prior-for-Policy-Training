# Proposed callable policies (not implemented or trained)

## tool_waypoint_v1

Dataset: `tool_position`; heuristic: `waypoint_clearance_residual`.

Learn a compact clearance-aware waypoint residual policy around deterministic collision-checked tool motion.

### Caller arguments

policy_id=tool_waypoint_v1; goal_tcp_base pose, arrival_mode, optional terminal_twist, clearance_m, protected_instance_ids, tolerance_m/rad, speed_cap, timeout_s. Dataset responsibility: tool_position.

### Input output contract

Causal observations processed by C plus caller arguments -> one verified-interface command via D and a status object. Internally chooses feasible clearance path and speed residual; caller does not supply each low-level move.

### Memory and handoff

Reset on new call; initialize last-command memory from actual controller state and up to 12 live frames. Interrupted call returns current pose/velocity; do not carry unvalidated recurrent state into a piece call.

### Selection cues

Choose for initial descent, contact-side change or retract-to-inspect without intentionally moving a piece. Choose a piece policy instead if the goal requires object motion.

### Status and progress

Report waypoint distance, clearance minimum/uncertainty, phase lift/traverse/descend/hold, controller validity, object-motion alarm and READY/BLOCKED states using the global status vocabulary.

See the dataset heuristic document for applicability, limitations, and handoffs.

## piece_contact_graph_v1

Dataset: `piece_relocation`; heuristic: `boundary_mechanics_residual`.

Learn boundary-relative contact decisions and local response residuals inside a deterministic, receding-horizon piece-to-pose controller.

### Caller arguments

policy_id=piece_contact_graph_v1; instance_id; goal_SE2_W or goal_footprint_W; semantic_orientation_constraint if required; protected_instance_ids; tolerances; optional exit_tool_base pose/arrival twist; clearance/speed limits; timeout/retry budget. Dataset responsibility: piece_relocation.

### Input output contract

Current causal dual-view/robot scene representation and caller goal -> bounded command via D, progress/error, uncertainty and phase/contact candidate. Policy selects approach side and strokes internally; HLA supplies the physical objective, not human-selected pushes.

### Memory and handoff

New instance/goal resets mode and response memory; warm-start with12 valid observations and actual last command. Preserve only validated tracker identity/geometry across calls. A tool reset pauses the object policy and resumes with new observations, recomputed candidates and the same goal.

### Selection cues

Prefer when metric local contours and rod geometry are confident and contact-model uncertainty is low. For symmetric objects or unreliable canonical pose but good dense masks, consider piece_goal_field_v1. Neither alternative bypasses shared perception or controller safety gates.

### Status and progress

Report positional/angular or contour error with symmetry flag, active boundary ID, contact confidence, approach/contact/reset/exit phase, predicted versus observed progress, retry count, protected-piece motion and readiness. Never report a verified character/word solely from manipulation completion.

See the dataset heuristic document for applicability, limitations, and handoffs.

## piece_goal_field_v1

Dataset: `piece_relocation`; heuristic: `causal_goal_field_servo`.

Learn a causal, geometry-only goal-field servo with mode-aware resets and deterministic collision projection, avoiding alphabet-specific poses.

### Caller arguments

policy_id=piece_goal_field_v1; same physical instance/goal/protection/exit/tolerance/budget arguments as piece_contact_graph_v1. Dataset responsibility: piece_relocation. A goal mask may be supplied directly only with its calibrated W mapping and instance identity.

### Input output contract

Causal robot observations and current/goal metric maps -> one bounded command via D and status/progress maps. Internal responsibilities include mode inference, corrective servo and contact-reset proposal; HLA retains target choice and overall sequencing.

### Memory and handoff

Reset recurrent servo on new goal/instance; use up to12 prior live frames with validity masks. After a tool-policy call rebuild maps and initialize history from actual commands, never carry a memorized segment index.

### Selection cues

Prefer for a well-segmented shape with ambiguous orientation or lower confidence in the mechanics model, and for small contour alignment after gross placement. If masks/correspondence are unreliable, stop rather than choosing it as an ungrounded fallback.

### Status and progress

Return contour error, semantic-orientation validity, predicted local progress/uncertainty, phase, contact-proximity confidence, collision-projection magnitude, changed neighbors and termination/failure vocabulary from the global contract.

See the dataset heuristic document for applicability, limitations, and handoffs.
