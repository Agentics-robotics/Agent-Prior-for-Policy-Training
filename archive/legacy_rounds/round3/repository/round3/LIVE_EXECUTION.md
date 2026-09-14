# Round 3 completed and accepted — 2026-09-08

This file records the final operational state. The preceding live-testing note is preserved byte-for-byte at [LIVE_EXECUTION_20260908T102704050474Z.md](provenance_snapshots/LIVE_EXECUTION_20260908T102704050474Z.md), SHA256 `4a49ca1f6517ae2a9caf22cce0c4a1d18f5dd9bf67d7cc5db29e276373f4aab1`. Earlier chronology remains at [LIVE_EXECUTION_20260908T091724Z.md](provenance_snapshots/LIVE_EXECUTION_20260908T091724Z.md).

## Completed experiment

All 98 formal models completed 20,000 updates with batch128 in independent child processes: 96 initial runs plus peg P4 at N5/N20, totaling 1,960,000 formal updates. All 4,900 development and 9,800 locked-test episodes are accepted. The six tasks are pick-place-wall, assembly, drawer, door, peg-insert-side and stick-push; nested fixed demonstration subsets are N=2/5/10/20 and each model uses training seed0.

All 18 initial designs and six sole feedback decisions are frozen: five `no_revision`, with peg the only P4 revision. The 24 development selections and global test gate were frozen before locked-test access. No further design, training or scored evaluation is pending. `round3/next_action.json` is absent; no agent design action remains.

## Accepted delivery and recovery evidence

- [Consolidated final-delivery acceptance](audits/final_delivery_acceptance.json) passed, binding 59 current final artifacts and combining the numerical, report, visual, Chinese-summary and runtime-reuse evidence. No owned project workers remain active.
- [Final numerical audit](audits/delivery_verifier_test_20260908T101614379062Z.json) passed at **2026-09-08T10:16:14.379029+00:00**. It checked 98 training/dev/test runs, 294 CPU checkpoints, 120 demonstrations, 1,020 reset snapshots, 24 frozen selection groups, six feedback decisions and all 4,900/9,800 episodes; invalid and pending counts are zero. This is CPU artifact verification, with no independent historical optimizer execution or simulator replay. Visual quality and cross-training-seed stability are outside its scope.
- [Actual video-render receipt](audits/video_render_2026-09-08T101934688812+0000/complete.json) passed at **2026-09-08T10:20:17.151399+00:00**. Four independent Pixi children on physical GPUs0–3 rendered 12 real paired MP4s covering all six tasks from cached actions, with zero policy calls and zero added scored episodes. The shared [video manifest](videos/manifest.json) was generated during the canonical final report.
- Canonical `/home/users/oscar/.pixi/bin/pixi run round3-resume` completed with **exit0**, `final=True`, `trained=98`, `evaluated=98`, `stage=test` (tool session64170). Log: [final_resume_idempotence.log](logs/final_resume_idempotence.log). This invocation generated the final report and reused the cached videos.
- [Runtime reuse audit](audits/final_resume_reuse.json) passed at **2026-09-08T10:25:13.836461+00:00**. All 33,034 accepted files, 98 completed runs, 4,900 development and 9,800 test episodes, and 12 video caches remained unchanged. Frozen source/gate/selection, ledger, training/session logs, events and MP4/sidecar identities were preserved. Added formal optimizer updates, scored episodes and video replays are all zero. The evidence combines unchanged file signatures, hashes, log counts and inventories with the canonical cache/empty-queue source review; it is not kernel-level profiling. The earlier `canonical_resume_reuse_review.json` remains a static preparation record, not the runtime acceptance.
- [Report integrity](reports/integrity.json) records `final=true`, `stage=test`, and validated completed results. Deliverables are [ROUND3_REPORT.md](ROUND3_REPORT.md), the [98-row CSV](reports/test_all_results.csv), [test summary](reports/test_summary.json), 14 PNGs in `figures/test/`, and the 12 paired videos. Accepted report SHA256: `527922ba3735a0b10ce83cee7ebb9f995652f7f92caa8c19f98004f2e321a322`.
- The Chinese-summary command actually succeeded after validating the final artifacts and produced [ROUND3_SUMMARY_ZH.md](ROUND3_SUMMARY_ZH.md), including the 24 frozen-selection comparisons.
- [Final visual and report review](audits/visual_delivery_review/final_review.json) passed at **2026-09-08T10:27:48.833036+00:00**: all 14 PNGs and 12 MP4s were inspected, with 98 report rows, 2,940 CSV fields and 73 report links checked. This review decoded existing videos with zero policy calls, simulator replay steps or GPU use. The regenerated Chinese overview also clarifies that completed-model session wall time includes historical interrupted and resumed sessions; see [cost clarification](audits/completed_model_session_cost_clarification.json) and [visual-review cost scope](audits/visual_delivery_review/cost_scope_note.json). No numerical result or report-renderer identity changed.

The final numerical run-ledger SHA256 is `012d6977fd5bd5b1c19075706242049b9a07507b0cb1115e2a9570ec259cfb57`. Selection SHA256 is `f1e4a4663c0cc9c3d08c73a2ac39cbc17f0460ecde75d8312864b974de762c89`; global-gate SHA256 is `debae7fbdc4e75d217cd5227bfb2fd4e82de5350d705b306ed2a42a7db842cba`, binding 1,310 files. All frozen experiment/model/evaluation sources, configurations, data, selections and prior records remain immutable. Report renderer source identity also binds the accepted video cache; do not waive identity mismatches.

## Results and limits

Across the 24 equally weighted task-by-N groups, the development-frozen selected system averages **73.59% test OOD**, versus **42.24% for B0** (+31.35 percentage points), where OOD=(C+E)/2. All 24 observed differences are positive. Final-system and initial-selected choices coincide; no test-based winner selection was used. All 72 initial candidate comparisons remain visible, including nine negative comparisons.

Peg P4 loses **15 and 1.25 percentage points** of test OOD at N5/N20 against frozen P1; its development losses were 15 and 5 points. P4 was not selected. It cost two additional from-scratch formal trainings (40,000 updates), 100 development episodes and 200 test episodes. Its 200-episode N5 feedback packet reused existing development evaluations; it did not add 200 feedback rollouts. These results do not demonstrate a general feedback benefit.

Scope remains six fixed tasks, numerical-state policies and one training seed. Coordinator historical/source exposure, absence of matched human-design/random-search controls and absence of component-isolating ablations limit autonomy and causal claims. Episode uncertainty is not training-seed uncertainty, and images used for design/diagnosis do not establish visual-policy generalization. Equal optimizer/chunk budgets do not imply equal FLOPs; summed concurrent worker times are not physical GPU occupancy. See [descriptive interpretation](audits/final_results_interpretation.json) and the complete report for negative, floor and ceiling outcomes.

## Repository and resource policy

Repository root: `/home/users/oscar/agent_learning/Agent-Prior-for-Policy-Training`. All `agent_training` content, including dotfiles, was migrated into the root and the redundant directory removed. Pixi locked installation, imports, CUDA and pinned dependencies were verified. All Python/project commands use `/home/users/oscar/.pixi/bin/pixi run ...`; no Conda/venv or bare pip. No commit or push was requested.

Latest GPU authorization remains **physical GPUs0–3 only**, with **two independent training/evaluation workers per card, eight total logical slots**, protected by global scheduler, slot and per-run locks. GPUs4/5 were withdrawn at07:03UTC; our jobs were stopped and the two interrupted models returned to the four-card queue. GPUs4/5 must receive no new work, and unrelated processes must be preserved. Each model's original budget and independent child-process training were maintained. The completed schedulers must not be mistaken for outstanding jobs.

Round1/2 specifications, historical reports/data and prior records remain preserved. Old-run checkpoints absent from the received migration remain disclosed rather than reconstructed. The original migration state is retained in [ROUND3_MIGRATION_HANDOFF.md](../ROUND3_MIGRATION_HANDOFF.md); it is historical. Exact operational AGENTS snapshots preserve earlier evidence identities after GPU authorization changed; no data, source or evidence hash waiver was used.

## Reuse commands

```bash
/home/users/oscar/.pixi/bin/pixi run round3-resume
/home/users/oscar/.pixi/bin/pixi run env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/verify_round3_delivery.py --stage test
/home/users/oscar/.pixi/bin/pixi run env CUDA_VISIBLE_DEVICES= python scripts/round3_chinese_summary.py
```

Completed models, data, frozen selections, scored results and video caches are verified before reuse. A passing numerical audit does not substitute for the separate report/video/recovery receipts above. Full stage commands remain documented in the repository [README](../README.md).
