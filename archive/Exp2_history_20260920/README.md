# Exp2 history archive — 2026-09-20

The active study is [Exp2_new](../../experiments/exp2/README.md): 225 complete OOD outcomes and 46 frozen models. This archive is historical evidence, not an execution queue. It includes both obsolete wrong-gripper experiments and other valid historical/supplementary results; archival does not imply that all archived data had the gripper bug.

| Directory | Contents |
| --- | --- |
| [runs](runs/) | Original M0/M1, GPT-5.5/APPL, GPT-6 old deployment, SinglePrior, extra corrected baseline seeds, partial ID, interrupted prefixes, budgets, original reports and raw journals |
| [experiments](experiments/) | Historical protocols and one-off diagnostic/recovery scripts, including failure reviews |
| [data](data/) | Earlier demonstrations/segmentation and GPT-5.5 processed skill data; preserved unchanged |
| [reports/previous_four_method](reports/previous_four_method/) | Earlier full four-method scientific report and its tables/figures/evidence |
| [reports/before_recovery](reports/before_recovery/) | Report revision recovered from the old ZIP before deleting it |
| [reports/current_before_navigation](reports/current_before_navigation/) | Exact current scientific report assets before link/index cleanup; includes the 255-row table with supplementary ID |
| [navigation_before](navigation_before/) | Previous README, PROGRESS and Exp2 agent instructions |
| [migration](migration/) | Exact move/delete inventory, original hashes, per-move journal and validation receipts |

The runtime archive physically lives at `/home/storage/oscar/appl_exp2/legacy/Exp2_history_20260920/runs`; the link above exposes it here. Selected episode/model leaves were moved to `runs/exp2/current`, and their old locations link back to them. Original top-level runtime names link into this archive so frozen absolute paths still resolve. Shared/Exp1 archives are untouched.

Scientific reports, original API output, checkpoint contents, trajectories and data are preserved. Only explicitly inventoried writing ZIPs, duplicate export directories and packaging utilities are deleted. Current reports are at [experiments/exp2/reports](../../experiments/exp2/reports/README.md).

Some older prose references the now-deleted ZIPs or original packaging layout. Those historical texts remain byte-preserved; use the active report and current manifest for working navigation. Historical scripts are retained as executed and are not supported run entry points.

Verification: [72 passed interface tests](migration/tests.json), [225 episodes / 46 models / 41 API submissions](migration/current_validation.json), [file preservation](migration/validation.json), [tracked-file relocation and no remaining ZIPs](migration/tracked_preservation.json). One-off migration sources are retained under `migration/tools/` as executed evidence; they are not active experiment runners.
