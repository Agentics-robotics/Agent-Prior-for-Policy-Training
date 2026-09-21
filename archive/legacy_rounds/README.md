# Legacy experiment archive

Round 1, 2 and 3 retain their original repository-relative paths under `roundN/repository/`. Existing checkpoints, reports, configuration, data, logs and historical source were moved, not copied. `path_mapping.json` maps all targets. The full original file hashes and sizes are in `experiments/experiment1/migration_inventory.json`; actual actions are in `migration_log.jsonl`.

Some legacy checkpoints were already absent or deleted. Only the inventoried existing files are preserved; this archive does not claim that every historical run is loadable. Historical deletion receipts and retention records remain with Round 3. Shared `src/relative_dp` remains in the active source tree because Experiment 1 reuses its frozen DP implementation.

Experiment 1 references only original successful Round 3 support episodes in `round3/repository/round3/data/<task>/train20/`; historical candidates, scores and conversations are excluded from RuntimePriorAPI evidence. Unfinished Round 4 was deleted according to the exact inventory; it is not represented as a completed experiment or renamed as Experiment 1.
