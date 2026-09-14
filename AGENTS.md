# Project execution

The active specification is EXPERIMENT1_CODEX_EXECUTION_HANDOFF.md with the latest user amendment: RuntimePriorAPI is a tool-using design agent. Work on dev.

All Python and project commands use /home/users/oscar/.pixi/bin/pixi run ... and the locked default MetaWorld environment. No Conda/venv or bare pip. Pixi changes are authorized with provenance retained.

Experiment 1 is exactly six tasks × N=2/5/10/20 × B0/B1/A1/A2/A3/A4, one replicate, seed 0: 144 formal training slots. Latest user allocation (2026-09-13): all physical GPUs 0–7 are authorized. Preserve unrelated processes and inspect actual occupancy before scheduling. This supersedes the earlier 4–7 restriction; historical receipts keep their original GPU indices. The resource-only transition is recorded under experiments/experiment1/allocation_20260913 and execution_allocation.json; retain the original protocol.lock.json and scientific artifacts.

RepositoryAgent implements fixed tools, public framework, B0/B1, execution and reporting. RuntimePriorAPI actually designs candidate inductive biases and writes candidate code. Do not hand-author an A-candidate or reuse historical output under API provenance.

The API may read frozen public capability files and current task×N evidence, edit only its unsubmitted candidate files, call bounded zero-update interface checks, and explicitly submit hashes. No API key, shell, arbitrary execution, hidden test data or other sessions. Generated code executes under verified Landlock/seccomp restrictions.

A1/A2/A3 must all submit before any performance feedback. The outer runner alone trains/evaluates them, then permits A4. Submitted candidates remain immutable. No performance-driven debug/search, A5 or extra seeds. Tool/check/repair costs and training attempts are separate; repeated unsubmitted edits do not add model slots.

Freeze all 24 design/selection instances globally before hidden testing. Never use hidden results to design/select or claim completion without validated artifacts. Preserve full tool journals, code versions, submissions and interrupted-call uncertainty.

Round 1/2/3 are archived under archive/legacy_rounds; keep their contents immutable. Unfinished Round 4 was explicitly retired and its exclusive contents deleted by exact migration inventory. Keep shared diffusion/environment behavior and archived provenance. This latest authorization supersedes earlier Round 4 execution instructions.

Follow [agent.md](agent.md). No heuristic/ad hoc candidate design, silent fallback or automatic transport retries. The user explicitly authorizes the frozen B1 rule baseline, bounded interface diagnostics/repairs, terminal failure accounting and exact resumption; these are declared experimental operations, never concealed performance tuning.
