# Agent Priors for Policy Training

**Round 3 已完成并验收**：六任务、N=2/5/10/20、每模型 seed=0，98次正式训练、4,900个开发回合和9,800个锁定测试回合；最终报告、14张测试图和12段真实配对视频均已生成。项目已迁到仓库根目录，Pixi 环境修复并验证；最终训练与评测使用物理 GPU0–3、每卡两个独立工作进程。阅读 [中文结论](round3/ROUND3_SUMMARY_ZH.md)、[完整报告](round3/ROUND3_REPORT.md)、[逐模型测试结果](round3/reports/test_all_results.csv) 和 [执行与验收记录](round3/LIVE_EXECUTION.md)。

以下保留前两轮的实验说明与历史结果；它们的设备和模型选择规则不作为Round3默认设置。

## 第一轮：Diffusion Policy 相对坐标实验

实施规格保存在 [ROUND1_SPEC.txt](ROUND1_SPEC.txt)。本项目比较 20 条相同专家示范下，原始状态与可逆相对坐标表示；每个任务只训练一个 seed。

本轮已实际完成：4×20,000 updates、240 个 dev episode、160 个正式测试 episode。四个模型的 IID/OOD 均为 20/20；成功率没有观察到差异。Drawer OOD 的成功平均完成步数为 raw 135.70、relative 93.55，这是需要多训练 seed 复验的次要观察。完整结果见 [ROUND1_REPORT.md](ROUND1_REPORT.md)，交付审计见 [artifacts/delivery_audit.json](artifacts/delivery_audit.json)。

## 环境与执行

本机 Pixi 位于 `/home/users/oscar/.pixi/bin/pixi`。若 PATH 尚未配置，可执行 `export PATH="/home/users/oscar/.pixi/bin:$PATH"`。所有项目运行命令必须通过 Pixi，不使用独立 venv、Conda 或裸 pip。项目目录为仓库根目录 `/home/users/oscar/agent_learning/Agent-Prior-for-Policy-Training`。

```bash
pixi install --locked
pixi run doctor
pixi run test
pixi run collect-round1
pixi run train-round1
pixi run evaluate-round1
pixi run report-round1
```

完整执行及恢复入口：

```bash
pixi run round1
```

单个 run 的恢复（会核对已有状态，并从最新完整 checkpoint 继续）：

```bash
pixi run train-round1 --run-id drawer_raw_n20_s0
```

主入口持有进程锁，避免同时启动两个正式优化器。训练每 1000 updates 以及 5k/10k/20k 保存原子 checkpoint；恢复 optimizer、EMA、loader/diffusion RNG 与采样位置。中断用正常的 SIGINT/SIGTERM，让程序在当前 update 后保存。`PROGRESS.md` 保存 PID，`logs/round1.log` 保存完整流水线输出；先检查原进程，避免重复启动。

每个正式 run 依次使用独立 Pixi 子进程，避免编译运行时状态从 debug inference 传入下一次训练。首次运行出现过速度异常，原始日志及 65 步逐位一致性恢复验证均保留，详见 [RUNTIME_RECOVERY.md](docs/RUNTIME_RECOVERY.md)。该调整未改变冻结模型、配置、数据或训练预算。

全部产物齐备后的独立审计命令：

```bash
pixi run python scripts/verify_delivery.py
```

Python 3.11、MetaWorld 3.1.1（commit `6e01ad7e2ffb2302e4dca04f796fcd8837df8540`）、MuJoCo 3.3.0、PyTorch 2.7.1、diffusers 0.35.1。全部传递依赖见 `pixi.lock`，运行时设备与版本见 `artifacts/environment.json`。本机默认使用物理 GPU 1、EGL，无自动购买算力或修改驱动行为。

## 四个固定实验

| run_id | 任务 | 表示 | 示范 | seed | updates |
|---|---|---|---:|---:|---:|
| drawer_raw_n20_s0 | drawer-open-v3 | raw | 20 | 0 | 20000 |
| drawer_relative_n20_s0 | drawer-open-v3 | relative | 20 | 0 | 20000 |
| door_raw_n20_s0 | door-open-v3 | raw | 20 | 0 | 20000 |
| door_relative_n20_s0 | door-open-v3 | relative | 20 | 0 | 20000 |

## 数据与模型语义

原生 observation 保留 39 维：当前 18 维、原生历史 18 维、当前目标 3 维。每块的手位置和交互点进行 `(hand, handle) -> (hand-handle, handle)`；当前目标替换为 `goal-current_handle`。其余字段不变。完整字段及任务特定四元数顺序见 `artifacts/observation_schema.json`。原生 reset 用当前状态填充历史；DP 在此基础上再使用相邻两步 observation。

在时刻 t，输入 `(o[t-1], o[t])`，预测 `a[t:t+16]`，执行前四个动作。起点重复 observation，尾部缺失动作以 mask 排除。动作标签是裁剪到原生边界后实际执行的四维输入，不再缩放或做示范 min/max normalization。每种表示的 mean/std 只使用固定 train20 的合法输入状态，std 下限 0.001；OOD 输入不裁剪。

每任务 100 条成功专家数据组成池，预先固定 RNG 选 20 条优化；其余 80 条不进入优化器或统计。dev、IID、OOD 各 20 个独立初始条件，OOD 左右各 10 个。采样通过原生 Task 设置和确定性 reset 适配层完成；真实位置、snapshot、恢复验证、失败专家尝试及文件 hash 均记录。

条件一维 U-Net 源于官方 [Diffusion Policy](https://github.com/real-stanford/diffusion_policy/tree/5ba07ac6661db573af695b419a7947ecb704690f)，许可证与修改说明随必要模块保存。训练及推理配置见 `src/relative_dp/config.py` 和冻结 artifact。

四个模型各 4,696,644 参数。资源执行设置统一采用 `torch.compile(mode="reduce-overhead")`，训练和推理都为 float32、禁用 TF32、使用确定性算法；有效 batch=128，无 gradient accumulation。设置在正式训练前根据吞吐实测固定，测量仅用于运行效率，不据开发/测试成功率调参。编译与 eager 的性能记录见 `artifacts/runtime_benchmark.json`，完整编译恢复测试随正确性测试执行。

## 选择、评测与产物

每个模型的 EMA 5k/10k/20k checkpoint 均用同一 dev20 评测，先比较成功率，再比较成功 episode 平均步数，最后选更早 checkpoint。四个选择全部冻结后再运行 IID/OOD 共 160 个测试 episode。每个 episode 固定独立 diffusion RNG，同任务表示配对。

- `data/dataset_manifest.json`：split、train20、实际位置、版本与 hash。
- `artifacts/`：硬件、状态 schema、正确性检查、冻结实验配置。
- `runs/<run_id>/`：正式配置、normalizer、训练日志、EMA 与完整恢复状态。
- `results/`：开发/测试逐 episode 数据、checkpoint 选择、summary、曲线与视频。
- `ROUND1_REPORT.md`：真实结果、计算成本、局限与下一步。
- `PROGRESS.md`：当前完成状态和精确恢复信息。

大文件已加入 `.gitignore`；源码、配置、lockfile 和小型结果可纳入版本控制。当前目录的原始文件将保留，不自动推送远端。


## Round 2 cabinet yaw experiment

Custom physically rotated MetaWorld tasks with paired world/frame diffusion policies. See [protocol](ROUND2_PROTOCOL.md), [geometry audit](docs/ROUND2_GEOMETRY.md), and the [completed report](ROUND2_REPORT.md).

Run the complete validated pipeline with `pixi run round2`; individual tasks are `round2-preflight`, `round2-calibrate`, `round2-collect`, `round2-train`, `round2-evaluate`, and `round2-report`. Round 2 artifacts are isolated in directories named `round2`. Four formal runs each use 20 demonstrations and 20,000 updates, followed by 240 dev and 480 final test episodes.

Round 1 frozen source, data, model and result files are preserved. Its original Pixi task file is archived at `artifacts/round2/round1_archive/pixi.toml`, because adding the new tasks necessarily changes the historical whole-workspace fingerprint. All other Round 1 frozen input hashes still match.


## Round 3 已完成

规格见 [ROUND3_SPEC.txt](round3/ROUND3_SPEC.txt)。六任务为 pick-place-wall、assembly、drawer、door、peg-insert-side 和 stick-push；使用固定嵌套示范子集 N=2/5/10/20，训练 seed 均为0。实际完成96次初始训练和2次 peg P4修订训练，每模型20,000步、batch128，总计1,960,000次正式更新。全部98组开发评测与98组锁定测试均完成，24组模型选择和全局测试门禁在读取锁定测试前冻结。六次反馈决定为五项 `no_revision` 和一次 peg P4；没有待完成的设计、训练或评测。

以24个任务×N组等权平均，开发集冻结选择的模型测试 OOD 成功率为73.59%，B0为42.24%，相差31.35个百分点；OOD=(C+E)/2。24组均观察到提升，最终系统选择与初始候选选择一致。全部72个初始候选对比仍有9个下降。唯一实际反馈修订 peg P4在 N5/N20 的测试 OOD 相对冻结基线 P1 分别下降15和1.25个百分点，未进入最终系统。

这些结果限于固定六任务、单训练 seed 和数值状态输入。协调者接触过历史任务与源码，缺少匹配的人类设计及随机搜索对照，也没有独立机制消融；结果不能证明完全陌生环境下的自主发现、视觉策略泛化或普遍反馈收益。episode级区间不衡量跨训练 seed 稳定性。相同更新与样本预算不等于相同 FLOPs，并发工作进程耗时之和不等于物理GPU占用。

实际交付与验收：

- [完整交付验收](round3/audits/final_delivery_acceptance.json) 已通过，绑定59个当前最终产物，并汇总数值、报告、图表、视频、中文概览和恢复复用证据；没有仍在运行的本项目工作进程。
- [最终数值审计](round3/audits/delivery_verifier_test_20260908T101614379062Z.json) 于2026-09-08 10:16:14.379029 UTC通过，核验98个run、294个检查点、4,900个开发回合、9,800个测试回合、24组冻结选择和六次反馈记录，无无效或待完成项。
- [渲染验收](round3/audits/video_render_2026-09-08T101934688812+0000/complete.json) 通过：物理GPU0–3上的四个独立进程完成六任务的12段真实配对视频；重放缓存动作，没有新增策略推理或计分回合。视频见 [清单](round3/videos/manifest.json)。
- `round3-resume` 已实际退出码0并生成最终报告、98行CSV和14张测试图；[报告完整性](round3/reports/integrity.json) 为 `final=true`。[运行时复用审计](round3/audits/final_resume_reuse.json) 确认33,034个已接受文件、98个run、4,900/9,800回合及12段视频保持不变，新增正式更新、计分回合和视频重放均为0；日志见 [final_resume_idempotence.log](round3/logs/final_resume_idempotence.log)。
- [中文结论与24组对照](round3/ROUND3_SUMMARY_ZH.md) 已通过最终产物验证并生成。数值审计与报告、图表、视频、恢复复用的验收分别记录；数值审计本身不覆盖视觉质量或跨训练seed稳定性。
- [最终视觉与报告复核](round3/audits/visual_delivery_review/final_review.json) 已通过，检查14张测试图、12段视频、98行报告结果、2,940个CSV字段及73个报告链接；仅解码现有视频，没有新增模拟器重放。

项目目录为 `/home/users/oscar/agent_learning/Agent-Prior-for-Policy-Training`，不再使用冗余 `agent_training` 子目录。Pixi locked install、依赖导入、CUDA与锁定版本已验证。全部 Python 和项目命令通过 `/home/users/oscar/.pixi/bin/pixi run ...`。最新授权仅允许物理 GPU0–3；训练与评测采用每卡两个独立工作进程、共八个逻辑槽位，保留全局调度锁、槽位锁及每run锁。GPU4/5已于07:03 UTC撤回，不得分配新工作；每模型预算与独立子进程训练保持不变。

完整入口及各阶段命令保留如下；已有完成产物先核验身份再复用：

```bash
/home/users/oscar/.pixi/bin/pixi run round3
/home/users/oscar/.pixi/bin/pixi run round3-prepare
/home/users/oscar/.pixi/bin/pixi run round3-validate-proposals
/home/users/oscar/.pixi/bin/pixi run round3-train
/home/users/oscar/.pixi/bin/pixi run round3-dev-eval
/home/users/oscar/.pixi/bin/pixi run round3-test
/home/users/oscar/.pixi/bin/pixi run round3-report
/home/users/oscar/.pixi/bin/pixi run round3-resume
```

只读数值核验、辅助标签清点及中文概览入口：

```bash
/home/users/oscar/.pixi/bin/pixi run env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/verify_round3_delivery.py --stage dev
/home/users/oscar/.pixi/bin/pixi run env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/verify_round3_delivery.py --stage test
/home/users/oscar/.pixi/bin/pixi run env CUDA_VISIBLE_DEVICES= python scripts/round3_dataset_accounting.py
/home/users/oscar/.pixi/bin/pixi run env CUDA_VISIBLE_DEVICES= python scripts/round3_chinese_summary.py
```

`round3-resume` 会校验已冻结的设计、模型、数据、选择、测试身份和视频缓存后复用。调度器检查存活进程与锁，避免重复执行；身份不匹配时拒绝复用。最终记录见 [LIVE_EXECUTION.md](round3/LIVE_EXECUTION.md)，迁移时的原始状态保留于 [ROUND3_MIGRATION_HANDOFF.md](ROUND3_MIGRATION_HANDOFF.md)。历史快报见 [FIRST_DEVELOPMENT_REPORT.md](round3/reports/FIRST_DEVELOPMENT_REPORT.md)。
