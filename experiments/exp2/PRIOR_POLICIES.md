# M1_v2: prior Diffusion Policies with explicit handoffs

This is the active Exp2 workflow authorized on 2026-09-16. M1_v1 is the completed
previous round (independent ID 0/5); its source, API journals, reports and traces
remain in [M1_v1](../../runs/exp2/M1_v1). The user authorized deleting its
checkpoints. The [exact receipt](../../runs/exp2/M1_v2/setup/v1_checkpoint_deletion_receipt.json)
records 31 files; historical checkpoint hashes remain. The previous
`prior_policies_20260916` path is an alias, so original evidence links resolve.

M1_v2 execution is complete: all 18 policies trained; independent ID 2/5 and
training-reset diagnostic 1/1. One ID success follows a fast corrective contact
and only establishes the agreed instantaneous geometric predicate. See the
[result report](../../runs/exp2/M1_v2/REPORT.md),
[analysis](../../runs/exp2/M1_v2/ANALYSIS.md) and
[completion receipt](../../runs/exp2/M1_v2/completion.json). These are measured
outcomes of the completed original round.

## Authorized 3000-step budget reevaluation (2026-09-16)

The later authorized inference revision is specified in [INFERENCE_5000.md](INFERENCE_5000.md).
It preserves the study described below and its context-limit interruption.

The user subsequently authorized increasing the physical evaluation limit from
1500 to 3000. [m1_v2_3000.json](configs/m1_v2_3000.json) reuses the exact frozen
18 policies, shared normalizer, data, prior/handoff documents and goal predicate.
Only the physical-step limit changes scientifically; API limits stay at 40
requests and 300 steps per invocation. Gym/ManiSkill's time limit is explicitly
set to the same 3000 steps without changing the shared native task or M0.

New receipts, API journals and reports live in
`runs/exp2/M1_v2/budget_3000`; the original run/configuration remain frozen.
The same diagnostic seed 1000 and ID seeds 6200–6204 are rerun from reset. This is
a same-seed budget reevaluation, not an untouched test set or a continuation
restored from the old physical state. The API sees the larger remaining budget
and may plan differently. Report both success within the first 1500 steps of
the new trajectories and success by 3000, alongside the original results.
No retraining, prior redesign, scripted recovery or automatic provider retry.

All six attempts have terminated: ID 2 successes, 2 task failures and 1 context-window
interruption; the diagnostic failed. Seed 6201 succeeded at step 1782, while 6204
succeeded at 1073. Seed 6202 stopped at 2740 with HTTP 502 (context capacity exceeded),
so its final outcome by 3000 is unknown and was not retried. See the
[budget report](../../runs/exp2/M1_v2/budget_3000/REPORT.md) and
[analysis](../../runs/exp2/M1_v2/budget_3000/ANALYSIS.md). The complete five-episode
final evaluation remains incomplete; 2/4 completed is not evidence of improvement
over the original 2/5.

```bash
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies freeze --config experiments/exp2/configs/m1_v2_3000.json
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies evaluate --config experiments/exp2/configs/m1_v2_3000.json --gpu 6 --seed 6200
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies report --config experiments/exp2/configs/m1_v2_3000.json --verify
```

## Changes authorized for M1_v2

1. Fit observation AND action ranges once on the 12 complete original training
   demonstrations (demo1000–1011). Every policy uses that identical immutable
   artifact. No per-skill range fitting, duplicated overlap weighting, validation
   observations or inference states enter normalization. Quaternion components
   retain fixed unit scales. The existing M0 implementation is not edited.
2. Runtime API chooses a new segmentation with substantially larger meaningful
   overlaps. The English prompt asks for complete shared transition coverage,
   evidence-based per-trajectory boundaries and explicit handoff information after
   every heuristic. The framework does not choose or rewrite boundaries/priors.
3. Each heuristic receives a fresh independent Runtime API implementation session,
   source package, training run, checkpoint, PRIOR.md and HANDOFF.json. API-owned
   semantics are distinguished from framework-measured boundary/overlap states.
4. Inference Runtime API reads both documents and observed training support to
   choose policies, invocation durations and transfers from the actual state.

This round changes several factors together. It is a system revision, not an
isolated causal ablation of normalization, overlap or handoff documentation.
Expanded temporal overlap does not establish recovery from unseen spatial/contact
errors. Whole-task success must be measured after actual training.

## Inputs, ownership and training

Configuration: [m1_v2.json](configs/m1_v2.json). Data: original
`data/exp2/demonstrations_v2`, API output `data/exp2/processed/M1_v2`.
Run: `runs/exp2/M1_v2`. All prompts, generated code and generated documents are
English. The developer writes framework, prompts and measured reports; actual
Runtime API writes segmentation, priors, model code and semantic handoff documents.
Submitted API bytes remain immutable. There is no hand-authored candidate, source
reuse from M1_v1, scripted action controller, silent fallback or transport retry.

Training remains seed 0, 20,000 updates per policy, batch 128, history 2, horizon
16, execution 8, DDPM 100 train/inference steps, epsilon prediction, clip_sample,
AdamW 1e-4/1e-6, EMA .999, cosine schedule and 500 warmup steps. Select final EMA
at that declared budget. The API chooses representation, learned structure and
objectives, with a 64-million-parameter cap. Future states are masked training
labels only. All priors in the submitted segmentation are trained.

Two-update interface checks must establish finite gradients, changed weights,
finite DDPM samples and exact EMA reload. API repairs its own failed interfaces
within the declared check budget; costs and failed attempts remain recorded.
Physical GPUs 4–7 only, through appl.gpu isolation; initial queues use currently
idle 6/7 and preserve unrelated jobs on 4/5. Inspect occupancy before scheduling.
Up to two independent training jobs may share a device after measured headroom.
Generated code executes in credential-free Landlock/seccomp workers.

Resource-only amendments are retained under `runs/exp2/M1_v2/setup`: after fresh
occupancy checks, shards 0/1/2/3 use GPUs 7/6/6/4. GPU 5 can execute already
declared tail policies ahead of the main queues; package locks and hashes prevent
duplicate design/training. Short API preparation checks may overlap training after
capacity checks. No existing process is stopped or moved; actual device receipts
take precedence over the initial scheduling plan.

## Policy package

`policies/<skill>/heuristic_<number>/` contains:

| Artifact | Owner and purpose |
| --- | --- |
| assignment.json | Framework: exact heuristic, data identity and shared normalizer hashes |
| design/, submission.json | API journal, versions, checks and immutable submitted hashes |
| source/policy.py, pipeline.json | API: learned model, objective and applicability metadata |
| source/PRIOR.md | API: mechanism, representation, loss, evidence and limitations |
| source/HANDOFF.json | API: entry/exit conditions, overlap role, successor readiness, failure/continuation cues and evidence |
| handoff_evidence.json | Framework: observed source segment starts, ends, overlaps and min/median/max states |
| training/handoff_evidence.json | The same measured support captured with the training run |
| training/policy_context.json | Shared normalizer identity and API handoff/evidence hashes |
| training/last.pt | Independent model/EMA/optimizer/RNG state, exact normalizer and policy context |
| training/result.json | Training validation and checkpoint hash |
| evaluation/handoffs.json | Actual API-selected entry/exit states; an explicit empty record for uninvoked policies |

Measured support is neither a hard applicability gate nor proof of real contact.
Semantic interpretation and transfer decisions belong to Runtime API. Skill
readiness may require manipulation state beyond its geometric subgoal; that does
not add a terminal task-success requirement.

## Evaluation and accounting

Freeze the entire submitted/trained library before any rollout. Predeclared trials:
training reset 1000 as a separately labelled diagnostic, then fresh independent
ID seeds 6200–6204. M1_v1 used 6100–6104; these are not repeated as a new untouched
test. Design API receives only training evidence, not evaluation states/results.
No performance-based model selection or extra training is scheduled in this round.

The selection API must read the chosen policy document (which also returns its
HANDOFF.json and measured support) before invocation. There is no preset skill
sequence or framework transfer controller. Original-run limits were 1,500 physical
steps, 40 API requests and at most 300 steps per invocation. The budget amendment
above changes only the first limit to 3,000. Each invocation records
entry/exit states, reason, model hash and handoff-document hash.

Success is the simultaneous conjunction of drawer open, red on pad and blue
inside drawer, using [demonstration_goals.json](configs/demonstration_goals.json).
No extra release, clearance, velocity or hold condition is imposed. Preserve
Exp1, frozen M0 code/evaluator/checkpoints and their historical results.
All segmentation, design, check, training and inference costs are declared. Unknown
API outcomes and interrupted physical trials require explicit reconciliation.

## Commands

All commands use the locked Exp2 environment. `initialize` is deterministic and
verifies an existing shared normalizer rather than replacing it. Do not duplicate
running API queues or physical trials.

```bash
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies initialize
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies segment
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies queue --gpu 7 --shard 0 --shards 4
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies queue --gpu 6 --shard 1 --shards 4
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies queue --gpu 6 --shard 2 --shards 4
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies queue --gpu 4 --shard 3 --shards 4
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies freeze
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies evaluate --gpu 6 --seed 6200
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies report --verify
```
