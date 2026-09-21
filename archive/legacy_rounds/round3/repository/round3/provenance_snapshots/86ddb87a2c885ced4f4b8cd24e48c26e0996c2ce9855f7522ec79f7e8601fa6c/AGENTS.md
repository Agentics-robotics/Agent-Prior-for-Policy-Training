# Project execution

All Python and project commands run through Pixi: `pixi run ...`.
Pixi binary on this host: `/home/users/oscar/.pixi/bin/pixi`.
Do not create Conda/venv environments or install with bare pip.
The current experiment specification is `round3/ROUND3_SPEC.txt`; historical Round 1/2 specifications remain immutable.
Preserve frozen data, split IDs, formal configurations, and prior run records.
Never report training or evaluation as complete without validated artifacts.
Round 3 latest user authorization (2026-09-08): use physical GPUs 0, 1, 2, 3, 4, 5 simultaneously, at most one formal training process per GPU. Preserve unrelated processes. This overrides historical GPU defaults, the migration handoff's two-GPU limit, and the initial four-GPU setting.
The project lives directly at the repository root; do not recreate the redundant agent_training directory.
