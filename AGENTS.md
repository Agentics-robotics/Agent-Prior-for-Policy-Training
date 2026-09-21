# Project execution

## Persistent Runtime API reasoning rule (user amendment, 2026-09-17)

For every future Exp2 run and every other new experiment, use reasoning effort
`xhigh` (Extra high). For Exp1-related experiments, use `max`. Check the actual
request field before launch and audit saved requests; never silently inherit
`high` from historical configurations. Preserve completed/frozen configurations
and journals as executed; this rule governs new work, not retroactive relabeling.
The currently authorized additional five-task APPL round uses `gpt-6-astra/xhigh`
for segmentation, prior implementation and deployment. See
[ASTRA_XHIGH.md](experiments/exp2/ASTRA_XHIGH.md). Keep original results intact.

## Repository-wide rules and navigation

Work on dev. Start with [README.md](README.md) for directory ownership and [PROGRESS.md](PROGRESS.md) for the single human-maintained progress summary. Update that summary with a review date and evidence links when a workstream changes; detailed measurements remain in its reports and receipts.

All Python and project commands use /home/users/oscar/.pixi/bin/pixi run ... with the appropriate locked environment. Exp1, relative_dp and experiment_interfaces use the root default MetaWorld environment. Exp2 uses /home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked. No Conda/venv or bare pip. Pixi changes are authorized with provenance retained.

Follow [agent.md](agent.md) and the applicable experiment rules below. Preserve frozen source, data, submissions, checkpoints, environment locks and provenance. Developer-authored framework/baseline code and Runtime API-authored designs must remain distinguishable. No ad hoc candidate design, silent fallback or automatic transport retries. Completion claims require validated artifacts; retain interrupted-call uncertainty and declared cost accounting.

Use the latest applicable experiment allocation, inspect actual GPU occupancy before scheduling, and preserve unrelated processes. The latest Exp2 resource amendment (2026-09-17) allows devices 0–7 with at most five simultaneously active; the active Astra study uses 1,2,3,5,7. Multiple jobs per device remain authorized. Earlier four-device allocations and their receipts are historical. Exp1-specific candidate order, check budgets and selection rules in this file and agent.md apply to Exp1.

## Exp2 scope

Latest scale-up amendment (2026-09-16): execute experiments/exp2/SCALEUP.md.
The subsequent resource amendment explicitly permits physical GPUs 0–7, multiple
jobs on each, with this experiment active on at most four physical GPUs at once.
Inspect actual occupancy and preserve existing processes; retain earlier receipts.
Task design and demonstration acquisition are explicitly developer-owned experiment
preparation. API-owned segmentation, priors, policy code and inference remain
unchanged in ownership. Four additional tasks, twelve demonstrations each, plus
the original drawer are compared at 30 ID and 30 position-OOD seeds per method
with an executor-only 5000-step cap. Initial 1500/3000 studies are archived;
M1_initial_test is the canonical retained initial study, with M1_v2 an alias.
Preserve all frozen original artifacts; new work belongs to M1_scaleup.

For all work concerning src/appl, experiments/exp2, environments/exp2, data/exp2 or runs/exp2, apply [experiments/exp2/AGENTS.md](experiments/exp2/AGENTS.md). Its project specification is [APPL_EXP2_REBUILD_AND_EXECUTE.md](APPL_EXP2_REBUILD_AND_EXECUTE.md), with current behavior documented in the [Exp2 entry](experiments/exp2/README.md) and [protocol](experiments/exp2/PROTOCOL.md). The rebuild specification is not an instruction to restart completed diagnostic attempts.

The latest user instruction (2026-09-16) authorizes a new M1_v2 round: fit one shared observation/action normalizer on the complete original training demonstrations; have Runtime API redo segmentation with substantially larger handoff overlap and explicit handoff information per heuristic; implement/train each API prior independently; then evaluate inference-time API selection using prior and handoff documents. The active specification is experiments/exp2/PRIOR_POLICIES.md, configuration is experiments/exp2/configs/m1_v2.json and implementation is src/appl/prior_policies. The previous run is M1_v1 (runs/exp2/M1_v1; its historical path remains an alias). The user explicitly authorized deleting M1_v1 checkpoints; exact inventories/receipts are in runs/exp2/M1_v2/setup. Preserve its source, submissions, API journals, metrics, reports and evaluation traces. Segmentation, heuristic content, policy source and semantic handoff documents remain API-owned. Preserve original inputs, Exp1, M0 and shared capabilities. Evaluation uses the agreed three geometric goals without extra terminal release/clearance/velocity/hold requirements. Frozen M0 results and evaluator remain unchanged. The retired older formal M1/M2 protocol is historical, not a gate to restart this newly authorized workflow.

The later 2026-09-16 budget amendment authorizes reevaluating the same frozen M1_v2 policies and seeds at 3000 physical steps using experiments/exp2/configs/m1_v2_3000.json. New evaluation artifacts live only in runs/exp2/M1_v2/budget_3000; retain the original 1500-step configuration and results. No retraining or policy/prompt changes accompany this amendment. Report success within the first 1500 steps of the new trajectories separately from later success; these are same-seed retests from reset, not continuations or fresh untouched ID trials.

Preserve Exp1 while working on Exp2. Active Exp2 imports use src/appl; never import archived Exp2 runners. Physical GPU processes follow the latest explicit allocation and appl.gpu isolation. Data/assets/calibration inputs live in data/exp2; runtime storage mappings and archive dependencies are described in [MIGRATION.md](experiments/exp2/MIGRATION.md).

The latest user follow-up authorizes a new inference revision addressing feedback/context failures with an executor-only 5000-step cap, hidden from API input. Use experiments/exp2/configs/m1_v2_5000.json and runs/exp2/M1_v2/inference_5000. API chooses policy, duration, literal numeric stop conditions and notebook; framework monitors conditions and retains raw journals without rewriting API output. Preserve existing policy packages and all previous evaluation artifacts. This separately recorded prompt/tool/context revision supersedes the no-prompt-change restriction only for this new study; it does not alter the completed budget_3000 study.

## Exp1 scope — completed experiment and preservation

Exp1's current report records completion of all 144 formal slots. Preserve its source, original protocol lock, data and results. The following rules retain the executed protocol for review and any explicitly requested reproduction/resumption; they are not pending work to restart merely because an agent opens this repository.

The specification is [EXPERIMENT1_CODEX_EXECUTION_HANDOFF.md](EXPERIMENT1_CODEX_EXECUTION_HANDOFF.md) with the latest user amendment: RuntimePriorAPI is a tool-using design agent. This scope includes src/experiment1, its dependencies src/relative_dp and src/experiment_interfaces, experiments/experiment1, the root locked environment and the original support data referenced from archive/legacy_rounds.

Experiment 1 is exactly six tasks × N=2/5/10/20 × B0/B1/A1/A2/A3/A4, one replicate, seed 0: 144 formal training slots. Latest user allocation (2026-09-13): all physical GPUs 0–7 are authorized. Preserve unrelated processes and inspect actual occupancy before scheduling. This supersedes the earlier 4–7 restriction; historical receipts keep their original GPU indices. The resource-only transition is recorded under experiments/experiment1/allocation_20260913 and execution_allocation.json; retain the original protocol.lock.json and scientific artifacts.

RepositoryAgent implements fixed tools, public framework, B0/B1, execution and reporting. RuntimePriorAPI actually designs candidate inductive biases and writes candidate code. Do not hand-author an A-candidate or reuse historical output under API provenance.

The API may read frozen public capability files and current task×N evidence, edit only its unsubmitted candidate files, call bounded zero-update interface checks, and explicitly submit hashes. No API key, shell, arbitrary execution, hidden test data or other sessions. Generated code executes under verified Landlock/seccomp restrictions.

A1/A2/A3 must all submit before any performance feedback. The outer runner alone trains/evaluates them, then permits A4. Submitted candidates remain immutable. No performance-driven debug/search, A5 or extra seeds. Tool/check/repair costs and training attempts are separate; repeated unsubmitted edits do not add model slots.

Freeze all 24 design/selection instances globally before hidden testing. Never use hidden results to design/select or claim completion without validated artifacts. Preserve full tool journals, code versions, submissions and interrupted-call uncertainty.

Round 1/2/3 are archived under archive/legacy_rounds; keep their contents immutable. Unfinished Round 4 was explicitly retired and its exclusive contents deleted by exact migration inventory. Keep shared diffusion/environment behavior and archived provenance. This latest authorization supersedes earlier Round 4 execution instructions.

Follow [agent.md](agent.md). No heuristic/ad hoc candidate design, silent fallback or automatic transport retries. The user explicitly authorizes the frozen B1 rule baseline, bounded interface diagnostics/repairs, terminal failure accounting and exact resumption; these are declared experimental operations, never concealed performance tuning.
