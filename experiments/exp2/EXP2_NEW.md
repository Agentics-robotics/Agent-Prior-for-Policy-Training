# Exp2_new — corrected-gripper APPL deployment

User authorization, 2026-09-19: use the new API credential for GPT-6 Astra/xhigh
APPL deployment, reduce each task/split to 15 layouts, and run only as far as the
authorized monetary budget permits. Currency must be confirmed before paid
generation. This supersedes the pause for APPL GPT-6 only; GPT-5.5 stays paused.

## Inputs and ownership

- Reuse the 36 frozen policies in `runs/exp2/M1_astra_xhigh`, their original
  segmentation, full-demo normalizers, API-authored source, priors and handoff
  documentation. No retraining, new priors, or manually changed API output.
- Select the first 15 seeds of each original ordered ID and position-OOD split:
  five tasks, 150 APPL episodes. Selection is independent of their outcomes.
- Reference the same selected layouts in the completed corrected naive-DP and
  SinglePrior_6_xhigh study. No baseline reruns. Its full 600 outcomes and original
  physical paths remain intact. Exp2_new is the new canonical comparison entry.
- Keep Exp1, the original studies, frozen scientific source and checkpoints intact.

## Deployment

Retain original prompts, tools, observations, context projection, policy-history
reset behavior, numerical stopping conditions, DDPM100, action chunk length,
seeds and geometric completion contracts. The physical cap remains 5,000 steps,
hidden from the API. The one controller intervention matches the new baselines:
clip arm targets as before; execute gripper +1 for a nonnegative prediction and
−1 otherwise. A process-local adapter applies this immediately before physics,
so the original invocation recorder saves the executed action alongside the
unmodified raw model output. Native gripper bounds are asserted to be [-1,1].

The Runtime API chooses policy, duration, conditions, notebook and finish actions.
Use only `https://api.openai.com/v1`, `gpt-6-astra`, reasoning effort `xhigh`, and
standard `service_tier=default`. Audit saved wire requests and returned identities.
The credential is outside the repository in a private file; it is never embedded
in prompts, source, reports or worker environments. Changing provider route is
declared and prevents interpreting this as an exclusively gripper-only API ablation.

## Budget and scheduling

Before every generation request, count the actual input via the Responses input
token-count endpoint. Reserve all input at the cache-write tariff plus the full
4,096-token output cap (including reasoning). Apply the documented long-context
multipliers. Settle against actual usage; absent cache-write detail retains the
conservative write tariff. Missing provider usage retains the reservation as an
unknown possible charge. Stop admission on insufficient budget, quota, transport,
identity or execution errors. Do not retry, substitute a model or use another key.
An input-count error sends no generation request and is retained explicitly.

Rates reviewed 2026-09-19: $10 ordinary input, $1 cached input, $12.50 cache write,
$50 output per million tokens; input/cache double and output increases by 1.5×
above 272,000 input tokens. Sources: [model](https://developers.openai.com/api/docs/models/gpt-6-astra),
[pricing](https://developers.openai.com/api/docs/pricing),
[token counting](https://developers.openai.com/api/docs/guides/token-counting).
Reported costs are usage estimates, not account invoices; the ledger explicitly
separates settled estimates and unknown-charge reservations.

Schedule in seed-index rounds, alternating ID/OOD across five tasks. One physical
episode runs at a time, within the five-GPU maximum. Inspect actual occupancy
before admission, preserve unrelated processes, use the existing isolated GPU
launcher and restricted policy workers. Reproduce the original nine-action
deployment checks for all 36 policies before paid evaluation, with no physics,
optimizer updates or Runtime API calls.

## Evidence and interpretation

Code: `experiments/exp2/exp2_new`. Artifacts: `runs/exp2/Exp2_new`.
Freeze policy/input/source hashes, selected layouts and original prompts before
launch. Save every raw API request/response, tool argument, invocation, action,
state, result, interruption, cost receipt, process exit and video.

Report completed successes/failures separately from interrupted and unstarted
episodes. Compare baselines on exactly the completed APPL subset as well as the
full preselected 15-layout subset. A partial budget-limited run is not a full
15-per-setting estimate and these already examined layouts are not untouched
confirmatory tests. Retain all earlier experimental results without relabeling.
