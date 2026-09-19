# M1_v2: API-owned stopping conditions and a private execution cap

Authorized by the user's 2026-09-16 follow-up. Configuration:
[m1_v2_5000.json](configs/m1_v2_5000.json). New artifacts:
`runs/exp2/M1_v2/inference_5000`. Previous 1500/3000 results, including the
interrupted 6202 attempt, remain immutable. No checkpoint, policy source,
segmentation, normalizer or training recipe is changed.

## Completed result (reviewed 2026-09-16)

Same-seed ID retests: **4/5 successes**; training-reset diagnostic: **1/1**.
Three ID successes occurred by step 1500, and the fourth at 1529. The failure
voluntarily finished at step 2070; no episode reached the private cap. All 94 API
requests completed. Validation covered 8294 physical steps, 30 early returns on
API conditions, hidden budget fields and exact retained context; 39 tests passed.
All 3747 historical files and all 18 policies were preserved, with no new training.

[Results](../../runs/exp2/M1_v2/inference_5000/REPORT.md) ·
[Analysis and remaining failure](../../runs/exp2/M1_v2/inference_5000/ANALYSIS.md) ·
[Completion receipt](../../runs/exp2/M1_v2/inference_5000/completion.json) ·
[Replays](../../runs/exp2/M1_v2/inference_5000/visualizations/README.md)

This is a multi-factor revision on already examined seeds. These observations
do not isolate the effect of hiding the budget or increasing the cap.

## Execution and ownership

- The executor caps an episode at 5000 physical steps. Neither the total cap nor
  remaining steps is supplied in API messages, tool schemas, observations or
  documents. Elapsed physical steps remain available as an observation timestamp.
- The API selects a frozen policy and duration (1–300 steps), reads its prior and
  handoff documents, and supplies zero or more stopping groups. Every group is a
  conjunction of numeric comparisons over documented measured metrics; any
  satisfied group returns control after that physical step. Thresholds, metric
  combinations, interpretation and next-policy decisions belong to the API.
- A condition-triggered return ends one invocation, not the episode. API may
  continue, switch policies, or explicitly finish. The executor also ends on the
  unchanged whole-task geometric success or its private cap. An already true API
  condition stops after the first action; it never fabricates a successful skill.
- Feedback reports current state, the latest invocation and metric minima/maxima.
  No skill-specific stopping threshold, collision controller or scripted recovery
  is added. World metrics and comparisons are implemented by the framework.
- Each invocation includes an API-authored notebook (up to 3000 characters) to
  retain important facts, failures and plans. It is recorded exactly as returned.
- The API request retains the original task/catalogue, latest complete read of
  every policy document, latest invocation/notebook, and six recent complete
  exchanges. All selected API output items, encrypted reasoning and tool outputs
  retain their original bytes/content; no repository-authored semantic summary
  replaces them. The full durable journal remains intact. read_invocation can
  retrieve an older numbered invocation with full boundary states.
- The private API request ceiling is 128 to accommodate additional feedback;
  this is separately accounted, never silently extended. No automatic provider
  retries or restart of interrupted physics. No prior results or test-seed-specific
  advice is inserted into the inference prompt.

## Evaluation

Reuse diagnostic seed 1000 and ID seeds 6200–6204 from reset with the same seeded
policy workers, DDPM100 and final EMA checkpoints. Use physical GPUs 4–7 after
checking actual occupancy and preserving unrelated jobs. Freeze the full library,
configuration and tested framework source before evaluation.

This changes total budget, budget visibility, prompt, tools, context selection
and API call allowance together. It is an informed system revision on previously
examined seeds, not a single-factor ablation or an untouched ID estimate. Report
success by 1500, 3000 and 5000 within the new trajectories separately, voluntary
finish, executor limits and infrastructure interruptions. Preserve all failed
attempts and missing provider-usage uncertainty.

Audit every condition-triggered stop against recorded states and literal API
arguments; check that no earlier step in that invocation satisfied the condition.
Verify every outgoing request for budget-field absence and whole-exchange context
provenance. Recheck goal predicates, checkpoint identities, initial-state equality,
worker shutdowns, costs and prior-artifact preservation.

```bash
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies freeze --config experiments/exp2/configs/m1_v2_5000.json
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies evaluate --config experiments/exp2/configs/m1_v2_5000.json --gpu 6 --seed 1000
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies report --config experiments/exp2/configs/m1_v2_5000.json --verify
```
