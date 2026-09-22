# Policy catalog

| ID | Kind | Responsibility | Output |
|---|---|---|---|
| egg_interaction_v1 | Single independently callable learned policy, visual CNN+finite-history GRU+three-mode density | Held known spatula: choose local approach, insertion adjustment, support/lift, release-producing and clearance motion from causal two-view observations | Six achieved-motion EE-local rates for at most0.1s, predictive spread, validity and immutable observation/session references |

Joint visual encoder is trained end-to-end with the actor, not an independently callable geometry model. The original proposed `scene_geometry_aux` dependency is explicitly replaced by raw-image conditioning; no semantic geometry/landmarks are promised. Generic geometry_execution remains an executor responsibility, with a software proposal converter `executor_request`, not a learned pickup or verified hardware API. Autonomous acquisition remains gated by explicit grasp geometry and a validated gripper/retention binding. There are no separate scoop/flick/retry/stop policies.

Use only the demonstrated camera/TCP/tool/grasp setup. Selection and handoff are in HANDOFF.json and CALLING.md; do not select models by episode or phase. All20 recordings are retained, full episodes split together. Actual preparation coverage is in preparation_summary; actual training and reload/calling results are emitted by the framework, not inferred from source counts. No physical success rate is measured.
