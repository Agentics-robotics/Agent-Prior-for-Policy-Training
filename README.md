# Agent Prior for Policy Training

本仓库研究 API 设计的归纳偏置如何用于机器人策略学习。Exp1 研究少示范的 MetaWorld 单技能策略；Exp2 研究 Panda 长时序任务的技能拆分、策略学习与组合执行，当前比较抽屉及四个新增任务。

**先看 [当前进度](PROGRESS.md)**：这是唯一的人工进度摘要，链接各实验的报告和原始证据。目录职责见下表；Exp2 的使用方式见 [Exp2 入口](experiments/exp2/README.md)。

## 实验导航

2026-09-19 **Exp2 四方法实验和论文资料已更新**：五任务、各 12 条示范、每方法各 30 ID＋30 位置 OOD；最后一条旧 APPL 5.5 中断已按原设置补测，945 步成功，现为 **1,200 个完整结果、0 个 unknown**。旧 APPL ID 131/150、OOD 50/150。[补测回执](runs/exp2/M1_scaleup/evaluation_recovery_20260919/completion.json)。新增 single-policy prior baseline 为 **ID 110/150、OOD 16/150**，部署 API 调用为 0。[完整英文报告](experiments/exp2/paper/REPORT.md)、[论文资料 ZIP](experiments/exp2/Exp2_paper_bundle.zip)、[写作 AI 指引](experiments/exp2/paper/WRITING_GUIDE.md)、[资料核验](experiments/exp2/paper_bundle_validation.json)。

2026-09-18 额外的 **GPT-6 Astra / xhigh** 五任务 APPL 实验已完成并核验：36 个模型，**300/300 个完整结果、197 成功**；ID 124/150，位置 OOD 73/150。[最终结果](runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/REPORT.md)、[分析](runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/ANALYSIS.md)、[配对视频](runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/paired_examples.html)、[完成回执](runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/completion.json)。

原始尝试及历次中断/授权恢复记录全部保留；见[本轮规范与恢复沿革](experiments/exp2/ASTRA_XHIGH.md)。今后新实验使用 xhigh，Exp1 相关实验使用 max；历史记录保持原值。旧 GPT-5.5/high 对照原有的 1 个未知结果已在 2026-09-19 按授权补测解决；原中断记录保留。

原始五任务比较曾有 599 个完整结果和 1 次 HTTP 中断；授权补测后 600 个结果完整，旧报告保持原始口径，见 [scale-up 规范](experiments/exp2/SCALEUP.md)和[最终分析](runs/exp2/M1_scaleup/ANALYSIS.md)。
最新初始研究入口为 [M1_initial_test](runs/exp2/M1_initial_test/README.md)；
旧 1,500／3,000 步结果和设置已归入 [初始测试归档](archive/Exp2_M1_initial_tests/README.md)。
`M1_v2` 保留为历史路径别名。

| 工作线 | 源码 | 配置与说明 | 输入数据 | 运行产物与结果 |
| --- | --- | --- | --- | --- |
| Exp2 单策略 prior baseline | [single_policy](experiments/exp2/single_policy/) | [规范](experiments/exp2/SINGLE_POLICY.md)、[配置](experiments/exp2/configs/single_policy_astra_xhigh/) | 原五任务各 12 条完整示范，无切分 | [完成报告](runs/exp2/single_policy_astra_xhigh/REPORT.md)、[API 设计核验](runs/exp2/single_policy_astra_xhigh/design_completion.json)、[全部回放](runs/exp2/single_policy_astra_xhigh/replays.html) |
| Exp2 Astra/xhigh 追加实验 | [执行与审计](experiments/exp2/astra/)，复用 [prior_policies](src/appl/prior_policies/) | [规范](experiments/exp2/ASTRA_XHIGH.md)、[配置](experiments/exp2/configs/astra_xhigh/) | 原五任务示范与配对初态 | [最终报告](runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/REPORT.md)、[36 个 API policy](runs/exp2/M1_astra_xhigh/POLICIES.md)、[全部回放](runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/replays.html) |
| Exp2 五任务 scale-up | [scaleup](src/appl/scaleup/) 与 [prior_policies](src/appl/prior_policies/) | [实验规范](experiments/exp2/SCALEUP.md)、[五任务配置](experiments/exp2/configs/scaleup/) | [四任务各 12 条示范](data/exp2/scaleup/)，原抽屉输入复用 | [运行入口](runs/exp2/M1_scaleup/README.md)、[结果表](runs/exp2/M1_scaleup/REPORT.md)；当前选定结果见[论文报告](experiments/exp2/paper/REPORT.md) |
| Exp1 | [experiment1](src/experiment1/)，复用 [relative_dp](src/relative_dp/) 和 [公共接口](src/experiment_interfaces/) | [执行规范](EXPERIMENT1_CODEX_EXECUTION_HANDOFF.md)、[冻结协议](experiments/experiment1/protocol.lock.json) | [manifest](experiments/experiment1/manifests/) 指向 [Round 3 原始支持示范](archive/legacy_rounds/round3/repository/round3/data/) | [实验档案](experiments/experiment1/)、[完整报告](experiments/experiment1/EXPERIMENT1_REPORT.md) |
| Exp2 M0 基线 | [dp_baseline](src/appl/dp_baseline/) | [唯一配置](experiments/exp2/configs/m0.json)：12 条示范、DDPM100、执行 8 | [原始示范 v2](data/exp2/demonstrations_v2/) | [保留模型与评估](runs/exp2/m0/)、[最终 M0 报告](experiments/exp2/M0_REPORT.md) |
| Exp2 M1 初始测试 | [prior_policies](src/appl/prior_policies/) | [训练规范](experiments/exp2/PRIOR_POLICIES.md)、[5000 步推理修正](experiments/exp2/INFERENCE_5000.md) | API 切分与交接重叠；完整训练示范共享归一化 | [当前入口](runs/exp2/M1_initial_test/README.md)、[5000 步报告](runs/exp2/M1_initial_test/inference_5000/REPORT.md)、[分析](runs/exp2/M1_initial_test/inference_5000/ANALYSIS.md)、[独立 Policy 与模型](runs/exp2/M1_initial_test/policies/) |
| Exp2 M1_v1 | 运行目录中的冻结源码和 API 提交 | [历史配置](experiments/exp2/configs/m1_v1.json) | [上一轮切分](data/exp2/processed/drawer_exchange_20260916_overlap/) | [历史报告](runs/exp2/M1_v1/REPORT.md)、[版本与 checkpoint 删除说明](runs/exp2/M1_v1/VERSION.md) |
| Exp2 示范拆分 | [demonstrations](src/appl/demonstrations/) | [流程与入口](experiments/exp2/README.md#示范拆分入口)、[三个目标](experiments/exp2/configs/demonstration_goals.json) | 授权的原始训练示范 | [M1_v2 API 输出](data/exp2/processed/M1_v2/)、[切分运行](runs/exp2/M1_v2/segmentation/) |

Exp1 的数据和运行记录集中在 `experiments/experiment1`；Exp2 分别使用 `data/exp2` 和 `runs/exp2`。M1_v2 重新调用 API 切分和设计 prior，开发者不重写 API 边界、prior 或交接语义。

## 环境与常用命令

在仓库根目录执行。所有 Python 和项目命令使用 `/home/users/oscar/.pixi/bin/pixi run ... --locked`，开发分支为 `dev`。

| 范围 | 环境 | GPU 分配依据 |
| --- | --- | --- |
| Exp1、relative_dp、公共接口 | 根目录 [pixi.toml](pixi.toml) / [pixi.lock](pixi.lock)，MetaWorld | 物理 0–7；见 [资源修订](experiments/experiment1/execution_allocation.json)，历史卡号与原锁文件保留 |
| Exp2，包括示范拆分 | [environments/exp2](environments/exp2/)，ManiSkill/SAPIEN 与 MuJoCo | 最新授权为物理 0–7、同时最多五张、每卡可多任务；Astra 新轮次用 1/2/3/5/7；见 [局部规则](experiments/exp2/AGENTS.md) |

调度前检查实际 GPU 占用并保留无关进程。根包的安装名仍为 `relative-dp-round1`；它是历史打包名称，两套环境分别锁定依赖。

```bash
# Exp1：查询已有实验状态。
/home/users/oscar/.pixi/bin/pixi run --locked experiment1-status

# 根环境测试：排除需要独立环境的 Exp2 测试。
/home/users/oscar/.pixi/bin/pixi run --locked python -m pytest -q tests --ignore=tests/exp2

# Exp2：只运行其接口测试；实际实验结果另见报告。
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m pytest -q tests/exp2
```

Exp1 的执行/恢复语义见其[冻结执行规范](EXPERIMENT1_CODEX_EXECUTION_HANDOFF.md)；Exp2 的诊断、设计、训练、报告和查看器命令见 [Exp2 入口](experiments/exp2/README.md#命令)。

## 其他目录

| 目录 | 职责 |
| --- | --- |
| [tests](tests/) | 根层测试覆盖 relative_dp；`experiment1`、`interfaces`、`exp2` 分别覆盖对应模块 |
| [scripts](scripts/) | Exp1 GPU 分配变更与指定实例恢复脚本，属于有记录的实验操作 |
| [examples/interfaces](examples/interfaces/) / [outputs/interfaces](outputs/interfaces/) | 公共接口配置示例与历史 API 连通性回执；接口说明见 [INTERFACES.md](INTERFACES.md) |
| [docs](docs/) | 历史 Round 2 几何说明 |
| [archive](archive/) | [旧 Round 1–3](archive/legacy_rounds/README.md)、[旧 Exp2](archive/legacy_exp2/README.md)、[M0 DP 诊断与对照](archive/Exp2_M0DP/README.md)；部分原始示范仍被 Exp1 引用 |
| `configs/` / `artifacts/` | 目前为空的旧布局目录；当前实验配置见上方入口 |

存储链接、归档依赖及 Git 跟踪范围见 [存储说明](experiments/exp2/MIGRATION.md#存储与版本管理索引2026-09-15)。`.pixi`、`__pycache__`、`.pytest_cache`、`.ruff_cache` 和 `*.egg-info` 是环境、缓存或安装元数据。

开发规则见 [AGENTS.md](AGENTS.md) 和 [agent.md](agent.md)。当前进度只在 [PROGRESS.md](PROGRESS.md) 汇总；各报告保留详细结果，历史日志保留当时记录。
