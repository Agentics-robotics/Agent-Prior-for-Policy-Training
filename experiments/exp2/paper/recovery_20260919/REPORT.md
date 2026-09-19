# Authorized recovery of the last unknown outcome

On 2026-09-19 the user authorized one reset of buffer_swap / ID / 20116. The original GPT-5.5/high episode stopped after 650 physical steps because the provider returned an upstream service error. It was neither a task failure nor a success.

The replacement succeeded after 945 steps. The main table now contains 1,200 complete cells and zero unknowns. The original interruption, all earlier frozen reports and the preceding ZIP remain retained. No completed task failure was retested.

The actual model/effort remains GPT-5.5/high to reproduce the historical comparison. New experiment defaults remain xhigh. Initial state, initial prompt and first request match the original exactly; all actions, API-selected policies and literal stop rules passed replay audit. No training or policy/prompt changes occurred. The derived source manifest records the previously audited gradient-disable correction in the zero-update reload check; the evaluation path is unchanged.

[Completion and API usage](completion.json) · [Authorization](authorization.json) · [Recorded video](replay.mp4) · [API invocations](invocations.json) · [Copy provenance](index.json)
