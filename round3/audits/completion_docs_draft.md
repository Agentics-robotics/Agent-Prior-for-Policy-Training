# Completion documentation draft — review only

Prepared from existing documentation, `final_results_interpretation.json`, and the passing numerical audit `delivery_verifier_test_20260908T101614379062Z.json`. This file does not change operational state. Video, final report, Chinese overview, and runtime reuse completion must be confirmed by the root agent before publishing the conditional completion text below.

## Exact stale locations

| File/location at review | Required update |
| --- | --- |
| `README.md:3` | Replace “当前执行” and future locked-testing language with the actual 98 training / 4,900 dev / 9,800 test completion counts; link the final Chinese overview only after it exists and validates. |
| `README.md:95–131` | Rename the current-execution section; replace active eight-worker wording with the completed allocation and actual 96+2 matrix. Replace the prospective audit/render/report sequence with accepted artifact links and recovery commands. Preserve generic recovery behavior only as reference, not an outstanding work list. |
| `PROGRESS.md:1–5` | Replace the opening Round 3 status, especially 32/98 tests, 3,553 episodes, and actively advancing drawer. Explicitly delimit all following Round 1/2 and migration material as historical; preserve that history. |
| `round3/LIVE_EXECUTION.md:1–46` | Replace live test PID/session, 96/98 tests, audit waiter, “No realvideosyet”, development-only report, and required future steps. Preserve earlier operational bytes in a new timestamped provenance snapshot before replacement. Do not rewrite an existing snapshot. |
| `round3/next_action.json` | Absent at review. Do not create a stale request for more design or feedback. Absence is consistent with all six feedback decisions being frozen. |

The old Round 1 path and GPU1 statements are inside explicitly retained historical sections; they must not be presented as current Round 3 instructions. Do not modify Round 1/2 specifications, frozen source/data/configuration, or prior records to tidy history.

## Safe status text before full final-delivery confirmation

Round 3 的数值实验已完成并通过最终数值审计：六任务、N=2/5/10/20、每模型 seed=0，共98次正式训练（96次主矩阵加2次 peg P4），各20,000步；4,900个开发回合和9,800个锁定测试回合全部完成。审计核验了294个检查点、24组冻结选择及六次反馈记录，未发现无效或待完成run。真实视频、最终报告、中文概览和完整恢复复用验收的状态以各自实际产物为准，数值审计本身不覆盖这些交付项。

## README opening — publish only after final-delivery confirmation

**Round 3 已完成并验收**：六任务、N=2/5/10/20、每模型 seed=0，98次正式训练、4,900个开发回合和9,800个锁定测试回合；报告、图表与真实配对视频已生成并核验。项目已迁到仓库根目录，Pixi 环境修复并验证；最终训练与评测使用物理 GPU0–3、每卡两个独立工作进程。阅读 [中文结论](round3/ROUND3_SUMMARY_ZH.md)、[完整报告](round3/ROUND3_REPORT.md)、[逐模型测试结果](round3/reports/test_all_results.csv) 和 [执行记录](round3/LIVE_EXECUTION.md)。

## README Round 3 section — replacement after confirmation

### Round 3 已完成

规格见 `round3/ROUND3_SPEC.txt`。六任务为 pick-place-wall、assembly、drawer、door、peg-insert-side 和 stick-push；固定示范子集为 N=2/5/10/20，训练 seed 均为0。实际完成96次初始训练和2次 peg P4修订训练，每模型20,000步、batch128，总计1,960,000次正式更新。全部98组开发评测与98组锁定测试均完成，24组模型选择在测试前冻结。

以24个任务×N组等权平均，开发集冻结选择的模型 OOD 成功率为73.59%，B0为42.24%，相差31.35个百分点；此处 OOD=(C+E)/2。24组均观察到提升，最终系统选择与初始候选选择完全一致。候选并非普遍有效：全部72个初始候选对比中有9个下降。唯一实际反馈修订 peg P4在 N5/N20 的测试 OOD 相对冻结基线 P1 分别下降15和1.25个百分点，未进入最终系统。

这些结果限于固定六任务、单训练 seed 和数值状态输入。协调者接触过历史任务与源码，缺少匹配的人类设计及随机搜索对照，也没有独立机制消融；结果不能证明完全陌生环境下的自主发现、视觉策略泛化或普遍反馈收益。episode级区间不衡量跨训练 seed 稳定性。相同更新与样本预算不等于相同 FLOPs，并发工作进程耗时之和不等于物理GPU占用。

项目目录为 `/home/users/oscar/agent_learning/Agent-Prior-for-Policy-Training`，不再使用冗余 `agent_training` 子目录。全部 Python 和项目命令通过 `/home/users/oscar/.pixi/bin/pixi run ...`。最新授权仅允许物理 GPU0–3；训练与评测采用每卡两个独立进程、共八个逻辑槽位，并保留全局调度锁及每run锁。GPU4/5已撤回，不得分配新工作；每模型预算和独立子进程训练保持不变。

```bash
/home/users/oscar/.pixi/bin/pixi run round3-resume
/home/users/oscar/.pixi/bin/pixi run env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/verify_round3_delivery.py --stage test
```

数值验收见 [最终数值审计](round3/audits/delivery_verifier_test_20260908T101614379062Z.json)；最终产物见 [中文结论](round3/ROUND3_SUMMARY_ZH.md)、[完整报告](round3/ROUND3_REPORT.md)、[CSV](round3/reports/test_all_results.csv) 和 [视频清单](round3/videos/manifest.json)。数值审计覆盖模型、数据、逐回合记录和冻结身份；报告、图表、真实视频及恢复复用另行核验。

仅当运行时复用验收实际通过后追加：`round3-resume` 已验证退出码0，98次训练、4,900个开发回合、9,800个测试回合及冻结身份保持不变；缓存视频及sidecar保持原hash和文件状态，没有新增正式更新、计分回合或视频重放。链接到本次实际运行时验收文件，不能使用仅有静态源代码审查的 `canonical_resume_reuse_review.json` 代替。

## PROGRESS opening — replacement after confirmation

# Round 3 已完成并验收（2026-09-08）

项目已迁至仓库根目录，Pixi locked install、依赖导入和CUDA验证通过。按最新授权仅使用物理 GPU0–3，训练与评测每卡两个独立工作进程；GPU4/5已于07:03 UTC撤回。六任务、N=2/5/10/20、seed=0的正式实验全部完成：96次初始训练加2次 peg P4训练，每模型20,000步；4,900个开发回合和9,800个锁定测试回合。最终数值审计于10:16:14 UTC通过，核验98个run、294个检查点、24组冻结选择及六次反馈决定，无无效或待完成项。

开发集冻结选择在24组上等权平均的测试 OOD 为73.59%，B0为42.24%（+31.35个百分点）。peg P4在 N5/N20相对P1的测试 OOD 分别下降15/1.25个百分点，负结果保留，最终系统未选P4。报告、图表、真实配对视频和中文概览已生成并核验；完整恢复复用验收通过，未新增训练、计分回合或缓存视频重放。详见 [中文结论](round3/ROUND3_SUMMARY_ZH.md)、[完整报告](round3/ROUND3_REPORT.md)、[数值审计](round3/audits/delivery_verifier_test_20260908T101614379062Z.json) 和 [最终执行记录](round3/LIVE_EXECUTION.md)。

本轮仅一个训练 seed，采用数值状态输入；历史任务/源码接触、没有匹配人类或随机搜索对照、缺少机制消融均限制结论。下方 Round 1/2 与迁移内容为保留的历史记录，其中“当前进程”、旧路径、旧GPU分配和“暂停”不代表 Round 3 当前状态。

## LIVE_EXECUTION replacement outline — publish only after confirmation

# Round 3 completed and accepted — 2026-09-08

The actual experiment is complete: six tasks, nested N=2/5/10/20, training seed0, 96 initial runs plus peg P4 at N5/N20. Each of the 98 models completed20,000 updates with batch128 in an independent child process. All4,900 development and9,800 locked-test episodes are accepted. The24 development selections and global test gate remain frozen; five feedback decisions are no_revision and peg is the sole P4 revision. No further design, training or scored evaluation is pending or authorized by this frozen experiment.

The numerical audit `round3/audits/delivery_verifier_test_20260908T101614379062Z.json` passed at10:16:14UTC:98 training/dev/test runs,294 CPU checkpoints,120 demonstrations,1,020 reset snapshots,24 frozen selection groups, six feedback decisions and zero invalid/pending runs. Its run-ledger SHA is `012d6977fd5bd5b1c19075706242049b9a07507b0cb1115e2a9570ec259cfb57`; selection SHA `f1e4a4663c0cc9c3d08c73a2ac39cbc17f0460ecde75d8312864b974de762c89`; gate SHA `debae7fbdc4e75d217cd5227bfb2fd4e82de5350d705b306ed2a42a7db842cba`. Numerical acceptance is a CPU artifact audit, not independent historical optimizer replay, simulator replay, visual acceptance or cross-seed validation.

Add the actual validated renderer receipt, video/pair count, canonical-resume log and runtime audit path, final report/integrity/CSV/figure identities, Chinese overview path, and confirmed process exit status here. Do not substitute static preparation checks for these runtime receipts. State that reuse preserved the immutable run ledger, global gate, selection, accepted numerical artifacts, training/session logs, events and per-video MP4/sidecar identities only after comparing the actual before/after evidence.

On the24 task-by-N groups, frozen selected-system OOD=(C+E)/2 averages73.59% against B0's42.24% (+31.35percentage points); all24 differences are positive, descriptively. The frozen final-system and initial-selected choices coincide. Preserve all72 initial candidate results, including nine negative comparisons. Peg P4 loses15/1.25points of test OOD at N5/N20 against frozen P1; development losses were15/5points. It was not selected. Do not describe this as general feedback improvement.

Scope: numerical-state policies, fixed six tasks and one training seed; coordinator historical/source exposure, no matched human/random-search control, no component-isolating causal ablations. Episode uncertainty is not training-seed uncertainty. Equal optimizer/chunk budgets do not imply equal FLOPs; summed concurrent worker times are not physical GPU occupancy. Historical Round1/2 reports/data remain preserved, and missing migrated old checkpoints remain disclosed.

Repository root: `/home/users/oscar/agent_learning/Agent-Prior-for-Policy-Training`. The redundant `agent_training` directory was removed after migration; Pixi locked environment, imports, CUDA and pinned dependencies were verified. All Python/project commands use `/home/users/oscar/.pixi/bin/pixi run ...`, with no Conda/venv or bare pip. Latest GPU authorization remains physical0–3 only, two independent training/evaluation workers per card and eight total logical slots protected by scheduler/slot/run locks. GPUs4/5 were withdrawn at07:03UTC and must receive no new work. Unrelated processes are preserved. No commit or push was requested.

No `round3/next_action.json` exists at review; no agent design action remains. Preserve an exact timestamped snapshot of the superseded execution note and link it from this final record.
