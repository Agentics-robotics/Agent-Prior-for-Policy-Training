# GPT-6 Astra/xhigh single-policy prior baseline

Preparation: 2026-09-18, following the user's request for an additional baseline
with Exp1-style API-authored inductive bias and no deployment agent. This is an
Exp2 experiment: reasoning effort stays **xhigh**, not Exp1's max.

## Scientific question and scope

Train one full-task diffusion policy per existing task. The Runtime API reads
the original training demonstrations and task contract, chooses its inductive
bias, and implements the learned representation, architecture and/or objective.
The repository supplies the interface, fixed training recipe and evaluator.
There is no trajectory segmentation, policy library, runtime language-model call,
handwritten phase controller, expert action access, or model selection on test
performance. One candidate is designed per task, with bounded API-owned interface
repair before immutable submission. This follows Exp1's offline API design
principle, not its separate multi-candidate development-selection protocol.

Five tasks, twelve original demonstrations each, one training seed (0), one
final-EMA checkpoint per task. Use the existing full-demonstration observation
and action normalizers, 60,000 optimizer updates per task, batch 128, history 2,
prediction horizon 16, execution prefix 8, DDPM100 and the existing optimizer,
learning-rate schedule, clipping and action representation. The update budget
matches naive DP. Parameter count and actual compute are recorded; equal update
counts do not imply equal FLOPs. The API can choose a learned architecture within
the existing 64-million-parameter interface limit.

The design session uses gpt-6-astra/xhigh, up to 36 API calls, 100 tool calls and
five two-update interface checks per task. The API owns all scientific policy
code and PRIOR.md. No earlier APPL priors, policy code, test trajectories or
performance results are supplied. Full-task manifest construction and numerical
training-data summaries are developer-owned preparation, not API segmentation.

## Evaluation

Freeze all five submitted/trained models and real B=1 deployment interface checks
before any test rollout. Reuse the original paired 30 ID and 30 position-OOD
layouts per task: 300 additional trials. These are previously examined paired
layouts, not fresh untouched test data. There is no performance-driven redesign
or early-checkpoint selection. Keep all formal results unchanged.

Each trial repeatedly calls only its fixed learned policy from observed state,
until the existing geometric success predicate is true or 5,000 physical steps
are reached. Use the same environment, reset, action bounds, diffusion seed and
goal definitions as the other methods. Preserve full traces and camera-frame
videos; report success by 1,500, 3,000 and 5,000 steps, latency, saturation,
parameter counts, training and API costs. Evaluation makes zero API requests.

Comparators are reused without retraining or reevaluation: naive DP, GPT-5.5/high
APPL, GPT-6 Astra/xhigh APPL. The original GPT-5.5 unknown remains unknown.
Versus naive DP, this tests offline API-designed inductive bias under the same
data/update budget. Versus APPL it changes decomposition, policy count, total
training compute and deployment agent together; it is not a clean ablation of
the runtime agent alone.

## Execution and preservation

New framework: `experiments/exp2/single_policy/`. Configurations:
`experiments/exp2/configs/single_policy_astra_xhigh/`. New runtime outputs:
`runs/exp2/single_policy_astra_xhigh/`. Frozen `src/appl`, environments, original
datasets, comparator sources and checkpoints, Exp1 and M0 remain unchanged.

All project commands use the locked Exp2 Pixi environment. Inspect actual GPU
occupancy before launching; use physical devices 0–7 with at most five active
simultaneously and preserve unrelated processes. API concurrency is at most four.
Retain every failed/interrupted attempt; no automatic transport retry or manual
rewrite of API output. Stop admission after an execution error and record it.
Completion requires all intended outcomes and independently validated artifacts.
