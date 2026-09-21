# Policy catalog

Three independent learned candidate priors implement the frozen cut_v5 portfolio. No performance ranking or mandatory execution order is established.

| Exact ID | Responsibility / selection cue | Science | Invocation |
|---|---|---|---|
| `tool_waypoint_v1` | Noncontact TCP approach/reposition; explicit free-space waypoint rather than intended piece motion | [Prior](tool_waypoint_v1_PRIOR.md) | [Usage](tool_waypoint_v1_USAGE.md) |
| `piece_contact_graph_v1` | One-instance relocation via outer/inner-boundary contact proposals; credible local contour geometry | [Prior](piece_contact_graph_v1_PRIOR.md) | [Usage](piece_contact_graph_v1_USAGE.md) |
| `piece_goal_field_v1` | Independent dense goal-field correction; reliable masks with uncertain canonical angle/mechanics | [Prior](piece_goal_field_v1_PRIOR.md) | [Usage](piece_goal_field_v1_USAGE.md) |

Shared entry documents: [PRIOR.md](PRIOR.md) covers executable pipelines, weak labels, dataset coverage, architectures and adaptations. [CALLING.md](CALLING.md) defines exact arguments, frames/units, statuses, examples and memory lifecycle. [HANDOFF.json](HANDOFF.json) is the machine-readable responsibility/selection/continuation/switch/failure/evidence contract for all three. [package.json](package.json) specifies independent fixed-budget training settings.

`inventory` exposes current geometric handles; `plan_layout` proposes distinct-instance layouts only from actual caller-supplied semantic hypotheses. Neither claims automatic open-world recognition. HLA owns semantics, physical instance assignment, spatial layout, staging and final word validation.

**Candidate-only safety boundary:** valid calls run the model and return finite recorded-encoding [v,w] proposals and diagnostics. Default status is CONTROLLER_UNVERIFIED. No verified controller/follower, rod/table calibration or full collision model is supplied, so decode_action stays disabled and physical/safe-to-handoff readiness stays false. Missing hardware assets do not prevent offline training/candidate calls, but must not be represented as verified metrology or successful generalization.
