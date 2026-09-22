# Project execution

All Python and project commands run through Pixi: `pixi run ...`.
Pixi binary on this host: `/home/users/oscar/.pixi/bin/pixi`.
Do not create Conda/venv environments or install with bare pip.
The current experiment specification is `round3/ROUND3_SPEC.txt`; historical Round 1/2 specifications remain immutable.
Preserve frozen data, split IDs, formal configurations, and prior run records.
Never report training or evaluation as complete without validated artifacts.
Round 3 latest user authorization (2026-09-08): use ONLY physical GPUs 0, 1, 2, 3. The user explicitly requested extra concurrent training processes on these underutilized cards, overriding the old one-process-per-GPU limit. Current allocation is two independent workers per GPU, eight workers total, using logical slot leases plus global scheduler and per-run locks. GPUs 4 and 5 were released at the user's request and must not receive new work. Preserve unrelated processes. Keep the fixed formal matrix, per-model budgets and independent child-process training unchanged.
The project lives directly at the repository root; do not recreate the redundant agent_training directory.
VLA / pi0.5 fine-tuning (branch dev-vla, 2026-09-20): user authorized physical GPUs 4, 5, 6, 7 for this work; it runs in the separate Pixi environment `vla` (`pixi run -e vla ...`, package `src/vla_pi05`) and must not touch GPUs 0-3 or the default environment.
