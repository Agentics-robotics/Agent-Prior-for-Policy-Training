# Policy catalog

## shape_push_v1 — one learned continuation policy
Given a selected physical instance's visible support, explicit local geometric goal and current EE state, propose a short planar EE displacement. A deterministic initializer proposes an accessible contact when contact is not yet established. This is a deliberate revision of the prior's unreliable learned contact-label implementation; initial contact is not claimed learned.

- Learned: pooled point encoder and3-mode displacement mixture, trained from recorded robot displacement with inferred visible rigid-motion goals.
- Frozen optional perception: SAM2.1 Hiera-small, pinned Apache-2.0 tensor; automatic inventory yields proposals, not verified tracks.
- Deterministic: goal frame, sampling, initial contact access heuristic, limits/status/memory, executor objective conversion.
- Supplied: visual high-level Agent for identity/inventory/layout/staging/task monitoring; commissioned scene/robot calibration and motion planner/controllers.

No letter, word, travel, success or force classifiers. No new-shape success claim. Only push-letter data available. See PRIOR.md, CALLING.md, HANDOFF.json and actual preparation/training reports for scope and measured results. Previous blocked attempts remain in source history.
