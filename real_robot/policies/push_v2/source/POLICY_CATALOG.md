# Policy catalog — cut_v3

Two independently trained alternatives for **one physical instance to one fixed image placement**. Both include demonstrated approach, pushing/contact changes, correction, staging and withdrawal. Neither interprets a word, chooses a next piece/goal, performs hardware I/O, or proves safe readiness for a successor.

| ID / frozen heuristic | Learned mechanism and proposed selection cue | Documents |
|---|---|---|
| `contour_push` / `boundary_graph` | Variable-topology outer/inner contour message passing + causal gated memory + current-command Gaussian mixture. Consider when visible contours and association are reliable; code requires proxy confidence >=.65. | [Prior](contour_push_PRIOR.md), [Usage](contour_push_USAGE.md), `HANDOFF.json.policies.contour_push` |
| `visual_push` / `dual_view_memory` | Independent selected-marked third global/detail, wrist and goal RGB encoders + causal gated memory + current-command Gaussian mixture. Consider when partial boundary fragmentation remains uniquely identifiable and RGB/history is informative; code requires proxy confidence >=.3. | [Prior](visual_push_PRIOR.md), [Usage](visual_push_USAGE.md), `HANDOFF.json.policies.visual_push` |

These cues are hypotheses, **not a performance ranking**. There is no mandatory order/cascade, no policy ensemble and no shared motor weights. Both use the same fixed segments and common eligible data, so two networks do not add evidence or independent safety redundancy. Each policy independently chose a single-command mixture rather than diffusion; see the priors for justifications.

## Shared entry points

* [PRIOR.md](PRIOR.md): full scientific implementation/adaptation record, causal label rules, fixed training recipe and limits.
* [CALLING.md](CALLING.md): complete observation/call coordinates and units, inventory, goal rendering/reexpression, memory/statuses, concrete calls and decoder verification fields.
* [HANDOFF.json](HANDOFF.json): per-policy responsibility, selection, entry/continuation/exit, successor readiness, switching, failures, reset rules, dependencies, segment support and limitations, separating coded checks from HLA judgments.
* `policy.observe_scene` -> automatic material-specific inventory; HLA supplies semantic association.
* `policy.preview_goal` -> executable foreground goal from a current template and explicit placement.
* `policy.act` -> trained **candidate** native dq[7] or a reasoned null-action status.
* `policy.reexpress_goal` -> preserve an old fixed foreground against a refreshed template on switching, or refuse.
* `policy.decode_action` -> pure verified-contract formatter/refusal, never hardware I/O.

## Common dependencies and failures

Original paired RGB/state/calibration, the same rigid installed tool, trained checkpoint/spec and causal instance association are required. No weights/assets were downloaded. The assumed generic segmentation/flow models are replaced explicitly by executable brown-on-light-support color components and forward association; this is material-limited, not open-world segmentation or character recognition. Touching pieces, occlusion, color/background shifts, identity ambiguity and calibration errors can defeat **both** alternatives. HLA owns semantics, layout feasibility/order, staging goals, cancellation and re-planning. Metric/projective layouts and tool/table clearance need commissioned assets not present here.

Image chart is original undistorted third 1280x720 pixels, with a two-pixel internal raster; goals are bounded acknowledged similarity transforms, not metric poses. No contact/force/clearance measurement is fabricated. A stable geometric goal match returns `goal_observed` with `successor_ready=false`; HLA must separately verify clearance/safe stop, all placements, readability/stability and safe next travel. Native action units/order/timing remain unverified until an external controller contract passes the decoder. Actual robot actuation is disabled.

All original frozen cuts are retained; command nulls, edge exclusions and failed derived inputs are counted. Source support is only two familiar-scene demonstrations, not validated unseen-shape/word/layout competence. No preliminary performance study, rollout or performance-based checkpoint selection occurs; training losses cannot establish robot success.
