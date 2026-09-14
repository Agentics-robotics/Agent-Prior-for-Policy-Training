# Round 4：准备与可行性验证中（2026-09-09）

当前执行 [ROUND4_SPEC.md](ROUND4_SPEC.md)：ManiSkill 3 + PartNet Mobility 跨柜体抽屉控制实验。**尚无 Round 4 策略训练、学习策略评测或先验收益结果。** 当前状态与后续阶段持续记录于 [ROUND4_REPORT.md](ROUND4_REPORT.md)；环境准备不等于控制器、专家收集和视频重放可行性已通过。

已完成机器人空场景底盘/六轴末端/夹爪控制实测与视频，以及 20 项通用训练 CPU 检查。当前因官方 UCSD 柜体下载连接超时、本机无资产缓存而不能进入真实柜体可行性阶段；已核查官方替代来源，并询问用户已有资产目录或可用访问配置。没有在后台运行正式训练。

已归档迁移前的 Pixi 文件和操作文档，并在同一工作区增加独立 `round4` 环境（`pixi run -e round4 ...`）。默认 MetaWorld 环境（`pixi run ...`）的 98 个锁定包记录与归档完全一致；实际导入保持 PyTorch 2.7.1、NumPy 2.2.6、Gymnasium 1.2.0、MuJoCo 3.3.0 和固定 commit 的 MetaWorld 3.1.1，见 [环境保留证据](round4/audits/metaworld_environment_preservation.json)。仅物理 GPU **0–3** 获授权，保留无关进程；本轮并发按实际显存和模拟器约束确定。

已按用户授权删除 **418 个 Round 3 权重文件，释放 30.06 GiB**，保留 drawer N10 的 **B0/P1/P2/P3** 全部 16 个检查点/latest 文件，并完成逐文件 CPU 加载验证。全部非权重证据保留，正式/debug 目录内 2,509 个非权重文件的 SHA256 均未变化。见 [删除与保留回执](round4/audits/round3_checkpoint_retention.json) 和 [删除前完整清单](round4/audits/round3_checkpoint_inventory_before_cleanup.json)。

后续仍需完成训练候选柜体上的专家可行性与重放、柜体 ID 隔离划分和数据冻结、基线与 Agent 先验、单 seed 试验、三 seed 主实验、开发选择冻结及锁定测试。下方为保留的历史 Round 1/2/3 记录，其中“全部完成”“当前进程”和验收结论仅描述当时状态；Round 3 删除前的最终审计不表示现在仍持有全部权重。按用户选择，完整检查点验收和所有模型权重复评已无法原样重跑，不重新训练补回已删模型。

# Round 3 已完成并验收（2026-09-08）

项目已迁至仓库根目录，Pixi locked install、依赖导入和CUDA验证通过。按最新授权仅使用物理 GPU0–3，训练与评测每卡两个独立工作进程、共八个逻辑槽位；GPU4/5已于07:03 UTC撤回。六任务、N=2/5/10/20、每模型seed=0的正式实验全部完成：96次初始训练加2次 peg P4训练，每模型20,000步；4,900个开发回合和9,800个锁定测试回合。最终数值审计于10:16:14.379029 UTC通过，核验98个run、294个检查点、24组冻结选择及六次反馈决定，无无效或待完成项。

开发集冻结选择在24组上等权平均的测试 OOD 为73.59%，B0为42.24%（+31.35个百分点）。peg P4在 N5/N20相对P1的测试 OOD 分别下降15/1.25个百分点，负结果保留，最终系统未选P4。最终报告、98行CSV、14张测试图、12段真实配对视频和中文概览已生成。`round3-resume` 实际退出码0；运行时复用审计通过，33,034个已接受文件与全部数值结果、视频保持不变，未新增正式更新、计分回合或视频重放。

产物与证据：[中文结论](round3/ROUND3_SUMMARY_ZH.md)、[完整报告](round3/ROUND3_REPORT.md)、[数值审计](round3/audits/delivery_verifier_test_20260908T101614379062Z.json)、[视频渲染验收](round3/audits/video_render_2026-09-08T101934688812+0000/complete.json)、[恢复复用验收](round3/audits/final_resume_reuse.json) 和 [最终执行记录](round3/LIVE_EXECUTION.md)。没有待完成的设计、训练或评测。本轮仅一个训练 seed，采用数值状态输入；历史任务/源码接触、没有匹配人类或随机搜索对照、缺少机制消融均限制结论。

下方 Round 1/2 与迁移内容为保留的历史记录，其中“当前进程”、旧路径、旧GPU分配和“暂停”不代表 Round 3 当前状态。

# 第一轮实验：全部完成

截至 2026-09-07，代码、Pixi 环境、数据、正式训练、开发集选择、正式测试、视频与报告均已完成。当前没有运行中的项目训练、评测或资源监测进程。

| Run | 正式 updates | 选定 checkpoint | IID | 位置 OOD |
|---|---:|---:|---:|---:|
| drawer_raw_n20_s0 | 20000 | 10000 | 20/20 | 20/20 |
| drawer_relative_n20_s0 | 20000 | 10000 | 20/20 | 20/20 |
| door_raw_n20_s0 | 20000 | 5000 | 20/20 | 20/20 |
| door_relative_n20_s0 | 20000 | 20000 | 20/20 | 20/20 |

- 17 项 preflight 检查通过；每任务 100 条成功示范、同一份 train20 及独立 split 已冻结。
- 完整重放 320 个初始状态、200 条示范，共 16,444 步；数据与动作标签校验通过。
- 4 次正式训练各完成 20,000 updates；12 个规定 checkpoint、optimizer、EMA 与完整 RNG 恢复状态已保存。
- 240 个 dev episode 已完成，四个 checkpoint 选择冻结后才运行 160 个最终测试。
- 160 个正式测试全部成功；8 个成功视频重放及轨迹 hash 验证通过。没有失败 episode，故 8 个失败视频类型明确记为 absent。
- 配对公平性与交付审计见 `results/paired_integrity.json`、`artifacts/delivery_audit.json`。
- 再次运行 `pixi run round1` 已验证退出码 0：全部训练跳过，开发/测试/视频缓存按完整 hash 验证复用，无新增 optimizer update 或正式测试 episode。日志见 `logs/idempotence_check.log`。

完整结果：[ROUND1_REPORT.md](ROUND1_REPORT.md)；四行结果：[results/summary.csv](results/summary.csv)；阶段计时：[results/compute_cost.json](results/compute_cost.json)。成功率没有观察到表示间差异；Drawer OOD 平均完成步数的次要差异需多训练 seed 复验。

## 历史进程与运行恢复

- PID 804594 / tool session 1364：首个 raw 训练完成后，relative 的速度异常；在第 65 步安全保存后暂停。原始日志、耗时和逐位一致性诊断完整保留。
- PID 822128 / tool session 26467：采用独立 Pixi 子进程顺序恢复，完成剩余训练、全部评测及视频，退出码 0。
- PID 864337 / tool session 2304：完成后的缓存复用检查，退出码 0。
- 资源监测 tool sessions 68535、25291 已结束。
- 冻结模型、数据、配置及训练预算未改。细节见 [docs/RUNTIME_RECOVERY.md](docs/RUNTIME_RECOVERY.md)。

## 可复现与恢复命令

项目目录：`/home/users/oscar/Agent_Training/agent_training`。本机 Pixi：`/home/users/oscar/.pixi/bin/pixi`。

```bash
pixi run round1
pixi run train-round1 --run-id drawer_relative_n20_s0
pixi run evaluate-round1
pixi run report-round1
pixi run python scripts/verify_delivery.py
```

已完成的训练会校验后跳过；未完成训练会恢复 optimizer、EMA、独立 RNG 与采样位置。所有正式数值结果均已冻结。未完成项：无；失败视频缺席是实际全成功结果，阶段机制归因与跨训练 seed 稳定性属于研究限制。


## Round 2 已完成

- 环境：真实柜体 yaw，自定义 MetaWorld 扩展；两任务均通过 A 档校准。训练±10°，测试±20°/±35°/±50°并含±2°扰动。
- 数据：新示范20条/任务；40条完整重放、280个dev/test快照重置，以及正负far角度短重放通过。
- 调试：四配置各200步，共800次更新；100步跨进程恢复检查通过。
- 正式训练：四模型各20,000步，共80,000次更新；12个EMA checkpoint。配对初始权重、参数量和采样/噪声摘要一致。
- 评测：240次dev、480次冻结测试全部完成；四模型均依据dev选择20,000步EMA。另对全部240个测试状态执行冻结专家后置参考，未调整测试或模型。
- yaw主成功率：drawer world/frame = 0.0%/81.7%；door world/frame = 45.0%/80.0%。四模型IID均100%。两任务frame在正向far仍为0/10，限制已报告。
- 交付：5张图、12段实际动作重放视频（含2段配对视频）、逐回合轨迹/指标、成本、协议和报告。
- 独立验收：`artifacts/round2/delivery_audit.json`；模型与数据完整性检查通过。

主报告：[ROUND2_REPORT.md](ROUND2_REPORT.md)。结果：[summary.csv](results/round2/summary.csv)。视频：[manifest.json](artifacts/round2/videos/manifest.json)。

复现：`pixi run round2`。独立复核：`pixi run python scripts/round2/verify.py`。已完成项通过hash核验后复用，不额外训练。

未完成项：无。唯一下一步建议为补充独立训练seed；本轮未启动额外实验。Round 1 历史结果与冻结输入已保留，Pixi新增任务对应的原始配置存于 `artifacts/round2/round1_archive/`。

总入口复跑验收：`pixi run round2` 已验证复用全部完成产物，23个关键数值产物身份保持一致，未新增正式更新或模型评测回合；见 [idempotence_audit.json](artifacts/round2/idempotence_audit.json)。

## Round 3：用户要求暂停迁移（2026-09-08）

已停止本轮所有实验进程。pick-place-wall、assembly、drawer各完成60条专家校准、20条成功示范和完整重放、50个dev状态、100个封存test状态、D2证据16帧。door校准保存37/60，另两任务尚未开始正式准备。正式训练0次、模型评测0回合；pickwall仅有未冻结设计草稿。训练/评测/双GPU入口框架已写，但未完成端到端验收和报告模块。

完整迁移范围、恢复点及必须先修复的集成项见 [ROUND3_MIGRATION_HANDOFF.md](ROUND3_MIGRATION_HANDOFF.md)。停止系用户主动迁移决定，不是API key或算力阻塞。迁移后收到继续指令再执行。
