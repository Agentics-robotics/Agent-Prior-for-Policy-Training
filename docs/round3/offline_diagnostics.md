# Offline rollout diagnostics

`src/round3/diagnostics.py` reads only already saved rollout trajectories. It does
not instantiate an environment, call a model/expert, read reset snapshots, or
alter success, termination, original failure categories, or learned inputs.

The rule is frozen at `round3/design_records/offline_diagnostic_rule.json` before
the first candidate N5 feedback packet. Four B0 development results already
existed at that time. These are descriptive analyses, not thresholds declared
before all model evaluations and not new scored endpoints.

The minimum distance from hand `[0:3]` to the interaction point `[4:7]` describes
approach. A distance at most 0.05 m is called proximity; it does not demonstrate
physical contact, a secure grasp, or task completion. Peak lift is the observed
object/tool z displacement from its first rollout observation; 0.03 m is a
descriptive threshold, with no inferred grasp condition.

Goal distances use the observed object point for pick-place-wall, ring body
center for assembly, peg head for peg-insert-side, pushed-object point for
stick-push, and handle for drawer/door. They are ordinary Euclidean distances;
assembly additionally reports xy distance and signed ring-minus-goal height.
These distances are not replacements for native success geometry, tolerances,
conjunctions, or hold requirements. Cabinet progress is taken only from the
existing logged trajectory `progress` array when available. Missing progress
remains missing; the simulator is never queried to fill it.

Timeouts retain their scored failure/termination metadata while also receiving
a descriptive approach/lift/progress label. This prevents `native_truncated`
from hiding every visible interaction difference. The labels do not establish
causal failure mechanisms. Feedback packets bind the diagnostic rule and
per-candidate diagnostics by hash; full reports retain per-episode values and
distribution summaries, including negative outcomes.

Data-quality reporting separately counts scripted-expert calibration successes
and failures, attempted/accepted demonstrations, and full replay checks. Debug
cost uses completed debug session ledgers plus the separately recorded recovery
reference, without double-counting summary audits of those same sessions.
Unaggregated CPU optimizer checks remain explicitly disclosed rather than
silently included in the formal 20,000-update budget.
