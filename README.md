# Agent Prior for Policy Training

本仓库研究 API 设计的归纳偏置如何用于机器人策略学习。Exp1 研究少示范的 MetaWorld 单技能策略；Exp2 研究 Panda 抽屉任务的技能拆分、策略学习与组合执行。

**先看 [当前进度](PROGRESS.md)**：这是唯一的人工进度摘要，链接各实验的报告和原始证据。目录职责见下表；Exp2 的使用方式见 [Exp2 入口](experiments/exp2/README.md)。

## 实验导航

| 工作线 | 源码 | 配置与说明 | 输入数据 | 运行产物与结果 |
| --- | --- | --- | --- | --- |
| Exp1 | [experiment1](src/experiment1/)，复用 [relative_dp](src/relative_dp/) 和 [公共接口](src/experiment_interfaces/) | [执行规范](EXPERIMENT1_CODEX_EXECUTION_HANDOFF.md)、[冻结协议](experiments/experiment1/protocol.lock.json) | [manifest](experiments/experiment1/manifests/) 指向 [Round 3 原始支持示范](archive/legacy_rounds/round3/repository/round3/data/) | [实验档案](experiments/experiment1/)、[完整报告](experiments/experiment1/EXPERIMENT1_REPORT.md) |
| Exp2 主实验 | [appl](src/appl/) | [入口](experiments/exp2/README.md)、[协议](experiments/exp2/PROTOCOL.md)、[主配置](experiments/exp2/configs/main.json) | [原始示范 v2](data/exp2/demonstrations_v2/)、[资产](data/exp2/assets/) | [运行档案](runs/exp2/rebuild_20260915/)、[报告](experiments/exp2/REPORT.md) |
| Exp2 示范拆分 | [demonstrations](src/appl/demonstrations/) | [流程与入口](experiments/exp2/README.md#示范拆分入口) | 授权的原始训练示范 | [子技能数据](data/exp2/processed/drawer_exchange_20260915/)、[拆分报告](runs/exp2/demonstration_processing_20260915/REPORT.md) |

Exp1 的数据和运行记录集中在 `experiments/experiment1`；Exp2 分别使用 `data/exp2` 和 `runs/exp2`。示范拆分的输出尚未接入 Exp2 主训练读取器。

## 环境与常用命令

在仓库根目录执行。所有 Python 和项目命令使用 `/home/users/oscar/.pixi/bin/pixi run ... --locked`，开发分支为 `dev`。

| 范围 | 环境 | GPU 分配依据 |
| --- | --- | --- |
| Exp1、relative_dp、公共接口 | 根目录 [pixi.toml](pixi.toml) / [pixi.lock](pixi.lock)，MetaWorld | 物理 0–7；见 [资源修订](experiments/experiment1/execution_allocation.json)，历史卡号与原锁文件保留 |
| Exp2，包括示范拆分 | [environments/exp2](environments/exp2/)，ManiSkill/SAPIEN 与 MuJoCo | 主实验配置限制物理 4–7；见 [局部规则](experiments/exp2/AGENTS.md)；拆分本身不训练或运行模拟器 |

调度前检查实际 GPU 占用并保留无关进程。根包的安装名仍为 `relative-dp-round1`；它是历史打包名称，两套环境分别锁定依赖。

```bash
# Exp1：查询已有实验状态。
/home/users/oscar/.pixi/bin/pixi run --locked experiment1-status

# 根环境测试：排除需要独立环境的 Exp2 测试。
/home/users/oscar/.pixi/bin/pixi run --locked python -m pytest -q tests --ignore=tests/exp2

# Exp2：只运行其接口测试；实际实验结果另见报告。
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m pytest -q tests/exp2
```

Exp1 的执行/恢复语义见其[冻结执行规范](EXPERIMENT1_CODEX_EXECUTION_HANDOFF.md)；Exp2 的诊断、设计、训练、报告和查看器命令见 [Exp2 入口](experiments/exp2/README.md#命令与用途)。

## 其他目录

| 目录 | 职责 |
| --- | --- |
| [tests](tests/) | 根层测试覆盖 relative_dp；`experiment1`、`interfaces`、`exp2` 分别覆盖对应模块 |
| [scripts](scripts/) | Exp1 GPU 分配变更与指定实例恢复脚本，属于有记录的实验操作 |
| [examples/interfaces](examples/interfaces/) / [outputs/interfaces](outputs/interfaces/) | 公共接口配置示例与历史 API 连通性回执；接口说明见 [INTERFACES.md](INTERFACES.md) |
| [docs](docs/) | 历史 Round 2 几何说明 |
| [archive](archive/) | [旧 Round 1–3](archive/legacy_rounds/README.md) 与 [旧 Exp2](archive/legacy_exp2/README.md)；部分原始示范仍被 Exp1 引用 |
| `configs/` / `artifacts/` | 目前为空的旧布局目录；当前实验配置见上方入口 |

存储链接、归档依赖及 Git 跟踪范围见 [存储说明](experiments/exp2/MIGRATION.md#存储与版本管理索引2026-09-15)。`.pixi`、`__pycache__`、`.pytest_cache`、`.ruff_cache` 和 `*.egg-info` 是环境、缓存或安装元数据。

开发规则见 [AGENTS.md](AGENTS.md) 和 [agent.md](agent.md)。当前进度只在 [PROGRESS.md](PROGRESS.md) 汇总；各报告保留详细结果，历史日志保留当时记录。
