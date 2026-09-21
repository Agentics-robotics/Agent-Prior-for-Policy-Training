# Proposed callable policies (not implemented or trained)

## contour_push

Dataset: `instance_relocation`; heuristic: `boundary_graph`.

Learn one reusable instance-to-placement policy with local boundary/topology sharing and closed-loop geometry feedback; never choose a spelling sequence or retrieve a known letter motion.

### Caller arguments

Common instance_id, observation template reference, explicit image placement/foreground goal, workspace/obstacle references, tolerance and budget. Caller controls WHAT instance/placement, when to stop, and when to switch; it does not choose latent boundary attention or individual contact points. No untrained contact-side argument is offered.

### Input output contract

Causal two-view observations and state enter the automatic converter; the graph policy consumes the derived geometry/history plus robot state and fixed goal. Outputs one candidate command dq[7] and shared status/progress fields, including contour completeness, registration ambiguity and contact-affinity uncertainty. Gripper feedback remains invalid. All output units/actuation are subject to the common controller gate.

### Memory and handoff

Maintain only within-call temporal graph state and causal object response. Reset at a new goal/instance or alternative-policy handoff; use available current perception tracks, not another policy's hidden state. Training histories are padded/masked at the original cut boundary.

### Selection cues

Prefer testing this alternative when boundary continuity, selected-instance association and projected tool/edge location are reliable, especially for unfamiliar visible shapes. Compare with visual_push when edges are fragmented but RGB/history is informative. This is a proposed selection rule, not an empirical ranking; if identity/goal is ambiguous neither policy is appropriate.

### Status and progress

Return centroid and foreground/rotation residual with ambiguity set, recent trend, geometry confidence, number of re-contact/oscillation events, possible neighbor displacement and blocked/uncertain/goal_observed flags. All use current/past estimates. Numerical safety/success thresholds require validation and cannot be inferred from episode endings.

See the dataset heuristic document for applicability, limitations, and handoffs.

## visual_push

Dataset: `instance_relocation`; heuristic: `dual_view_memory`.

Learn an independent two-view temporal visual relocation policy that retains near-contact appearance and causal response history lost by a contour-only representation.

### Caller arguments

Identical instance and image-goal interface, tolerance, workspace and budget as contour_push, enabling HLA replacement without an oracle conversion. Caller chooses the responsibility and call lifetime; the model internally chooses approach/contact/re-contact actions, not the next instance or staging destination.

### Input output contract

Causal RGB/state/history and target/goal markers to current command dq[7], status and uncertainty/progress. The goal stream is distinct from live observation. No known-alphabet token, demonstration number, verified contact label or depth metric is required. Return unavailable if mandatory scene identity, goal, calibration or backend verification is missing.

### Memory and handoff

Call-local recurrent memory, reset at goal/instance/policy change. A fresh call may warm up from permitted current causal observations using masks; no hidden state transfer or cross-cut training windows. A perception track may persist with its own uncertainty but supplies no future information.

### Selection cues

The HLA may test this alternative when the graph contour is unreliable but one or both RGB views still support stable identity and local tool tracking, or when recent visual response clarifies re-contact. If contour geometry is high-confidence, contour_push offers a more explicit shape bottleneck. Selection and tolerable occlusion duration must be validated; neither is claimed superior.

### Status and progress

Return shared geometric progress with uncertainty plus view consistency, target-marker confidence, temporal innovation and predictive action dispersion. During partial occlusion report degraded observability rather than apparent progress from hidden state alone. Use shared current-observation termination checks and reason-coded blocked/uncertain/timeout.

See the dataset heuristic document for applicability, limitations, and handoffs.
