# 项目进度

## 当前摘要

核对日期：2026-09-15。以下根据现有报告、完成记录、配置与调用关系整理；本次文档整理未重新运行实验或校验全部产物。这是唯一的人工进度摘要，详细数字以链接的报告和原始记录为准。

| 工作线 | 当前状态 | 证据入口 | 待完成事项 |
| --- | --- | --- | --- |
| Exp1 正式实验 | 已完成；现有报告记录训练、dev、hidden test 均完成全部正式槽位 | [完整报告](experiments/experiment1/EXPERIMENT1_REPORT.md)、[模型结果](experiments/experiment1/reports/model_results.csv)、[完成记录](experiments/experiment1/reports/completion_manifest.json) | 保全冻结代码、数据、选择与结果；没有待补跑槽位 |
| Exp2 主实验 | ID 开发门槛未通过；正式 M0/M1/M2 矩阵未完成 | [执行报告](experiments/exp2/REPORT.md)、[DP 诊断](experiments/exp2/DP_DIAGNOSIS.md)、[运行汇总](runs/exp2/rebuild_20260915/report/summary.json) | 继续定位闭环失败并通过既定门槛，再完成正式技能库设计、冻结、配对训练与评估 |
| Exp2 示范拆分 | 六类子技能数据与假设说明已生成；尚未接入主训练 | [拆分报告](runs/exp2/demonstration_processing_20260915/REPORT.md)、[完成记录](runs/exp2/demonstration_processing_20260915/completed.json)、[数据 manifest](data/exp2/processed/drawer_exchange_20260915/manifest.json)、[验证记录](runs/exp2/demonstration_processing_20260915/validation.json) | 决定六类子技能与三类任务角色的映射、训练输入和实验协议；接入训练是单独的后续工作 |

示范拆分完成表示数据与文档已生成，未证明策略有效。主训练仍读取原始完整示范及候选指定的片段边界。两条 Exp2 流程的依赖与命令见 [Exp2 入口](experiments/exp2/README.md)。

当前资源范围：Exp1 物理 GPU 0–7，依据 [2026-09-13 资源修订](experiments/experiment1/execution_allocation.json)；Exp2 主实验按其[配置](experiments/exp2/configs/main.json)限制为物理 4–7。调度前检查实际占用。下方旧卡号、PID、运行中描述及恢复命令只代表历史时点。

## 历史记录：Exp1 执行过程

以下保留原进度原文，最后一条记录止于 2026-09-13 隐藏测试进行中。历史正文中的“当前”“最新”指各自记录时点；当前结论以本页顶部摘要及其证据为准。

---

# Experiment 1 进度入口

当前执行规范是 EXPERIMENT1_CODEX_EXECUTION_HANDOFF.md 和用户追加的 RuntimePriorAPI 工具 agent 要求。

用户最新 GPU 分配为物理卡 4、5、6、7，替代旧的 0–3 分配。候选 GPU worker 按 UUID 绑定，保留所有历史尝试的原始设备记录。

实时状态：`pixi run --locked experiment1-status`。
报告：`experiments/experiment1/EXPERIMENT1_REPORT.md`。

已完成实际精确迁移、六任务支持数据/控制审查、独立dev/test初态和24套真实图像证据。真实GPT-6 Astra/max工具冒烟已完成12次API调用、一次零更新接口检查和显式候选提交；它是合成基础设施测试，不属于144正式训练槽位。

公共GPU预检、源码/协议冻结与正式矩阵状态以可验证工件为准。不得把本进度说明当作训练已完成的证据。旧PROGRESS/README/AGENTS与Pixi环境快照保存在archive/legacy_rounds/round3/environment_before_experiment1。

2026-09-12 续推：修复 PyTorch CUDA UUID 与 NVML 的 `GPU-` 前缀差异，保留原始 UUID；`gpu-uuid-format-r4` 基础设施修订通过六任务公共拟合、物理 GPU 4–7 核验及精确恢复检查。旧失败尝试保留。完整测试 268 passed / 1 skipped，修复后相关测试 35 passed。实际 preflight 已通过，protocol.lock.json 已冻结。

早先执行 `pixi run --locked experiment1-run --resume --detach` 因启动环境缺少 `LOCAL_OPENAI_BEARER_TOKEN` 而停止，历史失败详情保留在 `experiments/experiment1/last_failure.json`。

2026-09-12 10:24 UTC：用户提供并授权持久化凭据后，已保存至仓库外 `~/.config/experiment1/runtime-api-credential.json`（0600）。专用启动入口 `~/.config/experiment1/launch.py` 自动加载凭据，通过 Pixi 调用冻结 runner；不修改实验源码或冻结模型/地址。恢复命令：`/home/users/oscar/.pixi/bin/pixi run --locked python /home/users/oscar/.config/experiment1/launch.py`。已有 runner 运行时不要重复启动。

后台 runner PID 853471，日志目录 `experiments/experiment1/logs/runner-1789208654505688207/`，物理 GPU 队列 4–7。首批四个正式 task×N 设计会话均已获得 HTTP 200 响应并执行工具循环。凭据阻塞已解除；启动时正式训练/dev/test 为 0/144，实时计数以 status 为准，不能据此声称实验完成。

2026-09-12 10:46 UTC 核查：pick-place-wall 四个 N 档的 A1/A2/A3 共 12 个正式候选已显式提交且接口检查有效，全部提交文件 hash 与冻结协议核验通过。正式 API 调用 153 次均为 HTTP 200 并已消费（另有基础设施 smoke 12 次）；正式接口检查 12 次。四个 B0 正在物理卡 4–7 训练，记录步数分别为 N2=11800、N5=12200、N10=11200、N20=14400，目标均为 20000；正式完成训练/dev/test 仍为 0/144，无新失败或阻塞。未向 API 返回开发反馈，隐藏测试尚未开启。报告已刷新，检查快照见 `experiments/experiment1/reports/live_execution_audit.json`。后台 runner 继续按冻结流程调度。

2026-09-13 01:25 UTC 最新资源变更：用户授权物理 GPU 0–7 全部使用，0–3 实测空闲。已准备仅涉及 CLI 卡号、隔离 worker 设备授权、runner 并行度与协议补充校验的精确改动；原文件和待应用文件保存在 `experiments/experiment1/allocation_20260913/{original,proposed}/`。原 `protocol.lock.json`、科学配置、候选代码和已有结果保持原 hash；补充记录为 `execution_allocation.json`，仅在切换时产生。八卡调度与修订校验的 6 项测试已通过。

为避免运行中源码变化，仅结束旧调度进程 853747，保留四个独立实例及全部训练/API worker。后台切换控制器 PID 321717 正在等待现有实例结束，然后自动应用资源修订、运行相关测试、对八张卡逐一进行零更新隔离/UUID 检查并启动八卡 runner。切换状态见 `experiments/experiment1/allocation_20260913/transition_status.json`，启动结果见同目录 `launch.log`；当前 `draining` 表示八卡尚未正式启用。不要同时手动启动另一个 runner。任何检查失败会明确写 blocked 并停止，不自动重试；失败槽位不因此重训。

2026-09-13 03:28 UTC：八卡资源修订已应用，55 项相关测试和八张卡零更新隔离检查通过，新 runner PID 512751 已实际启动。

2026-09-13 05:39 UTC：用户明确授权补上 assembly/N10/A4 的 OOM 失败槽位。全局隐藏测试尚未开始；仅停止外层调度进程 513020，保留 stick-push/N2 的独立运行实例，避免补跑期间越过全局测试门槛。恢复程序 PID 662683 在空闲物理 GPU 0 上加载原 step=1 checkpoint；实际训练账本确认 resumed_step=1，已推进至 step=100。候选代码/配置/seed/科学槽位不变，训练 attempt 与中断费用单独保留。

恢复原始 checkpoint、失败 slot、会话、q4 选择及开发反馈均备份至 `experiments/experiment1/recoveries/assembly_N10_A4_20260913/`。后台脚本完成训练/dev 后，仅将新开发结果返回同一 API 会话重新确定 q4，保留旧选择和全部工具日志，q1/q3 不变；所有现有实例结束后自动恢复八卡 runner，再全局冻结并隐藏测试。状态与错误见该目录 `status.json`、`stderr.txt`，恢复调度结果见 `restart.log`。本次是用户明确授权的一次恢复，没有自动重试；失败则写 blocked 并保留原因。

2026-09-13 06:03 UTC：assembly/N10/A4 恢复成功，完成原定 20000 步及开发评估。API 根据新开发结果按原规则将 q4 更新为 A4（开发 C/E 等权分数 0.725，原 A2 为 0.475）；原失败与选择记录保留。恢复程序已重新启动外层 runner PID 719175，24 个实例随后全部全局冻结，开始物理 GPU 0–7 八卡隐藏测试。06:31 UTC 状态：144 个系统均已完成训练/dev，8 个系统完成 test、8 个正在 test；这些是进度计数，不是最终隐藏测试结论。日志目录 `experiments/experiment1/logs/runner-1789279412036418660/`。
