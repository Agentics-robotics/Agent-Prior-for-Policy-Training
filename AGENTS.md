# Project execution

## Repository-wide rules and navigation

Work on dev. Start with [README.md](README.md) for directory ownership and [PROGRESS.md](PROGRESS.md) for the single human-maintained progress summary. Update that summary with a review date and evidence links when a workstream changes; detailed measurements remain in its reports and receipts.

All Python and project commands use /home/users/oscar/.pixi/bin/pixi run ... with the appropriate locked environment. Exp1, relative_dp and experiment_interfaces use the root default MetaWorld environment. Exp2 uses /home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked. No Conda/venv or bare pip. Pixi changes are authorized with provenance retained.

Follow [agent.md](agent.md) and the applicable experiment rules below. Preserve frozen source, data, submissions, checkpoints, environment locks and provenance. Developer-authored framework/baseline code and Runtime API-authored designs must remain distinguishable. No ad hoc candidate design, silent fallback or automatic transport retries. Completion claims require validated artifacts; retain interrupted-call uncertainty and declared cost accounting.

Use the latest applicable experiment allocation, inspect actual GPU occupancy before scheduling, and preserve unrelated processes. Exp1's 0–7 authorization does not expand Exp2's current 4–7 configuration. Historical receipts retain their actual device indices. Exp1-specific candidate order, check budgets and selection rules in this file and agent.md apply to Exp1.

## Exp2 scope

For all work concerning src/appl, experiments/exp2, environments/exp2, data/exp2 or runs/exp2, apply [experiments/exp2/AGENTS.md](experiments/exp2/AGENTS.md). Its project specification is [APPL_EXP2_REBUILD_AND_EXECUTE.md](APPL_EXP2_REBUILD_AND_EXECUTE.md), with current behavior documented in the [Exp2 entry](experiments/exp2/README.md) and [protocol](experiments/exp2/PROTOCOL.md). The rebuild specification is not an instruction to restart completed diagnostic attempts.

The main experiment and independent demonstration-processing module are separate workflows. The latter publishes API-defined segment datasets and heuristic hypotheses; developer code validates/materializes the plan without substituting its own segmentation or heuristics. It currently has no training or simulator stage. Its completed output is not consumed by the main trainer. Integrating it into training requires a separately specified experiment change; documentation/layout work alone does not authorize that change.

Preserve Exp1 while working on Exp2. Active Exp2 imports use src/appl; never import archived Exp2 runners. Physical GPU processes follow the Exp2 4–7 configuration and appl.gpu isolation. Data/assets/calibration inputs live in data/exp2; runtime storage mappings and archive dependencies are described in [MIGRATION.md](experiments/exp2/MIGRATION.md).

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
