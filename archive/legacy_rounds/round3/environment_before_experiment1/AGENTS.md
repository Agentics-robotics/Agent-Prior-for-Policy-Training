# Project execution

新增接口开发同时遵循 [agent.md](agent.md)，历史实验行为保持不变。

All Python and project commands run through Pixi: `pixi run ...`.
Pixi binary on this host: `/home/users/oscar/.pixi/bin/pixi`.
Do not create Conda/venv environments or install with bare pip.
The current experiment specification is `ROUND4_SPEC.md`; historical Round 1/2/3 specifications and experiment source remain immutable.
Preserve frozen data, split IDs, formal configurations, and prior run records.
Never report training or evaluation as complete without validated artifacts.
Round 3 latest user authorization (2026-09-08): use ONLY physical GPUs 0, 1, 2, 3. The user explicitly requested extra concurrent training processes on these underutilized cards, overriding the old one-process-per-GPU limit. Current allocation is two independent workers per GPU, eight workers total, using logical slot leases plus global scheduler and per-run locks. GPUs 4 and 5 were released at the user's request and must not receive new work. Preserve unrelated processes. Keep the fixed formal matrix, per-model budgets and independent child-process training unchanged.
The project lives directly at the repository root; do not recreate the redundant agent_training directory.

# Round 4 execution (user authorization 2026-09-09)

Execute the full ManiSkill 3 + PartNet Mobility cross-cabinet drawer experiment, including feasibility, successful expert data and replay videos, model-ID-disjoint splits, priors, training, development selection, frozen test evaluation and ROUND4_REPORT.md. Do not stop after preparation or a single-seed pilot.
Use only physical GPUs 0, 1, 2, 3; preserve unrelated processes. Choose concurrency within measured memory and simulator limits. All project/Python commands use Pixi; ManiSkill uses `pixi run -e round4 ...` in this same workspace, and the default MetaWorld dependency set is preserved.
The user explicitly permits deleting Round 3 model weights to reclaim storage. Retain drawer N10 B0/P1/P2/P3, preserve deletion/retention receipts, and retain all datasets, configurations, logs, evaluations, reports and source. Prior acceptance audits describe the historical complete delivery; do not claim all deleted checkpoints remain loadable or rerun Round 3 training to recreate them.
Screen assets only on predeclared physical, reachability and expert-feasibility checks, never learned-policy scores. Keep train/dev/test disjoint by whole cabinet model ID, including all drawers. Test object details and outcomes must not influence Agent prior design or development selection. Fresh prior designers receive training-only evidence after feasibility is demonstrated.
Use identical raw information and per-cabinet nested demonstration subsets across methods. Main results use three training seeds. Freeze per-data-budget development selections before locked testing. Retain negative results and separate design/search/pilot costs from the final training matrix.
