# Proposed callable policies (not implemented or trained)

## egg_interaction_v1

Dataset: `held_spatula_interaction`; heuristic: `blade_relative_local_motion`.

Learn one causal, blade-relative short-motion policy for all held-spatula interaction progress; delegate robot motion generation, known-tool acquisition and safety enforcement to explicit executor adapters.

### Caller arguments

session_id; task=turn_over_in_same_pan; egg_track_id; pan_track_id; spatula_template_id; initial_face_reference; allowed_pan_interior; validated T_ee_B; geometry/calibration versions; safety envelope; fresh paired observations; reset/resume flag. No phase chosen from episode time and no supplied future success.

### Input output contract

Causal tensors/geometry are specified in input_preprocessing. Return u[6] in current blade axes (m/s and rad/s), predicted spread[6], validity/reason, support/visibility evidence references, proposed horizon<=0.10 s, and current observation ID. The selected Gaussian mode is one consistent motion proposal, not an average of modes. The executor must reject expired or infeasible proposals. No joints, gripper commands, force targets or guaranteed flip flag are returned.

### Memory and handoff

Maintain a 128-state GRU and short causal frame buffer within a session; the Agent separately retains the original-face goal and an observed release-attempt latch. Initialize from recent actual observations on entry; missing history is masked. Continuous adjustments reuse state. After interruption, slip, object reassignment or new grasp, clear motion memory and warm up from fresh causal observations, while preserving the original task goal unless the Agent explicitly starts a new task.

### Selection cues

Use when a known spatula is visibly retained and the egg/pan are observable in the demonstrated workspace; it handles the transition from lifted-tool approach through interaction progress. Use geometry_execution alone for calibrated acquisition and empty-tool free-space moves. Unknown geometry/support, a dropped tool or an off-pan object requests assistance rather than a different untrained phase policy.

### Status and progress

At each 10 Hz call provide active or invalid with reason, uncertainty, track visibility and observed support evidence. Executor feedback reports local goal progress. The Agent monitors mask/pose changes, support acquisition, release and settled appearance; no-progress timeout and repeated rejected goals stop the cycle. completed_flip is issued only by the Agent's explicit visual verification against the initial reference, with unknown allowed. The scene model is not a safety-rated human detector.

See the dataset heuristic document for applicability, limitations, and handoffs.
