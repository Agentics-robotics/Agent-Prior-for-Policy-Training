# Exp2：目录、流程与使用入口

Exp2 使用 Panda 完成抽屉交换任务：打开抽屉、把红块移到外部垫子、把蓝块放入抽屉。当前进度统一见根目录 [PROGRESS.md](../../PROGRESS.md)；详细结果见 [REPORT.md](REPORT.md) 与 [DP_DIAGNOSIS.md](DP_DIAGNOSIS.md)。

执行依据是 [重建规范](../../APPL_EXP2_REBUILD_AND_EXECUTE.md)、[本实验规则](AGENTS.md)及[协议](PROTOCOL.md)。本页说明已有路径与调用关系，不改变技能定义、训练配方或正式门槛。

## 五个目录的职责

| 目录 | 职责与主要内容 |
| --- | --- |
| [src/appl](../../src/appl/) | 唯一有效源码：API 工具循环、数据读取、神经训练、环境、诊断、评估与示范拆分 |
| [experiments/exp2](./) | 本页、协议、报告、[主配置](configs/main.json)、[事故记录](incidents.json)和[迁移来源](migration_sources.json)；`prompts/` 当前为空，主要 prompt 在 Python 模块内 |
| [data/exp2](../../data/exp2/) | 原始示范、资产、重建公共输入和拆分数据；这是仓库内的真实目录 |
| [environments/exp2](../../environments/exp2/) | 独立 Pixi manifest 与 lock；实际依赖安装在外部存储 |
| [runs/exp2](../../runs/exp2/) | 训练、API 会话、诊断、校准、报告与迁移回执；目录映射见 [存储说明](MIGRATION.md#存储与版本管理索引2026-09-15) |

源码内部：`cli.py/config.py` 组织入口；`agent.py/journal.py` 管 API 会话与账本；`design.py/library.py` 管候选和技能库；`data.py/train.py/public.py/baseline.py/vendor/` 管数据与 DP；`gpu.py/worker.py/security.py/kernel.py` 管受限进程；`protocol.py/formal.py/deployment.py` 管正式流程和高层技能调用。独立拆分模块在 `demonstrations/`。

## 两条流程及其依赖

| | 主实验 | 独立示范拆分 |
| --- | --- | --- |
| 入口 | `python -m appl.cli`，Pixi task 名为 `exp2` | `appl.demonstrations.process_demonstrations`，已有运行脚本见下方 |
| 输入 | [demonstrations_v2](../../data/exp2/demonstrations_v2/) 中的完整示范；按原始轨迹划分训练/验证 | 明确列出的授权训练示范；当前运行使用同一批训练轨迹 |
| 技能定义 | `open_drawer`、`move_red`、`move_blue` 三类任务角色；正式候选自行确定片段边界 | API 自主定义子技能、边界、跨轨迹分组与假设说明 |
| 处理结果 | 候选代码、训练 checkpoint、开发验证和最终评估 | 原始状态/动作片段、图片、`dataset.json` 与 `heuristic.md` |
| 运行记录 | [rebuild_20260915](../../runs/exp2/rebuild_20260915/) | [demonstration_processing_20260915](../../runs/exp2/demonstration_processing_20260915/)；完整工具账本另在输出的 `_session/` |

主实验的数据读取器 [data.py](../../src/appl/data.py) 从主配置的原始示范目录读取数据，并可按候选提供的边界截取片段。它尚未读取 `data/exp2/processed`；主 CLI 也没有调用示范拆分模块。

当前拆分产出六类数据：开抽屉并退开、接近并抓住红块、搬红块到垫子并退开、接近并抓住蓝块、搬蓝块进抽屉、最后撤回并稳定。它们是三类任务角色的更细行为划分，但代码尚未建立二者的训练映射。输出说明中的 heuristic 是 API 提出的待验证假设。

模块本身通过参数接收输入路径、输出目录和 API client，不读取主实验配置；当前[运行脚本](../../runs/exp2/demonstration_processing_20260915/launch.py)借用 `main.json` 的训练轨迹清单与 API 设置，并传入公共控制接口。这是启动配置上的依赖，未把拆分结果接入训练。

后续接入需要单独确定：采用哪些技能及映射、如何读取片段、是否使用及如何验证假设、如何匹配 M1/M2 的数据曝光和预算。目录整理不改变这些实验选择。

## 主实验与模拟环境

| 方法 | 训练与执行方式 |
| --- | --- |
| M0 | 用完整轨迹训练一个 DP，直接执行低层策略 |
| M1 | 使用 API 最终切分和技能接口，以同主干/数据/预算训练，移除 API 数值 prior |
| M2 | 使用 API 设计的表示、loss 或结构机制；与 M1 共享高层调度权限和接口 |

`envs/native.py` 实现 ManiSkill/SAPIEN 原生环境，供回放、闭环诊断及最终 ID/OOD 评估使用。`surrogate.py` 管理 API 重建的 MuJoCo 模型，`model_contract.py` 校验模型，`verification.py` 用冻结模型执行设计阶段的受控验证。两类环境共用固定成功判定，校准通过并不保证 learned policy 成功。

既定流程是：数据/控制检查与 DP 诊断 → 达到开发门槛 → 完整技能库设计及一次反馈修订 → 冻结 → 配对训练与目标评估。具体门槛以 [protocol.py](../../src/appl/protocol.py)、[主配置](configs/main.json)和[协议说明](PROTOCOL.md)为准；当前完成情况只在 [PROGRESS.md](../../PROGRESS.md)汇总。

## 输入与运行档案

`data/exp2` 的 `demonstrations/` 保留第一版示范；`demonstrations_v2/` 是主配置的输入。`assets/` 保存机器人模型与来源/hash；`reconstruction_public/` 保存授权 URDF 和契约；`public/api_model.xml` 是重建流程使用的初始模型。`processed/` 保存独立拆分的输出。

[主运行目录](../../runs/exp2/rebuild_20260915/)按职责保存：

| 子目录 | 内容 |
| --- | --- |
| `diagnostics/` | D0–D4、采样、四元数与能力对照诊断 |
| `api_capability/` | API 神经候选试跑、提交版本、训练与验证 |
| `reconstruction/` | MuJoCo 模型设计、版本、提交与校准 |
| `reference/` | 独立参考控制器的物理可达性检查 |
| `checks/`、`incidents/` | 隔离/恢复等工程检查与中断或异常记录 |
| `source_versions/` | 运行源码快照 |
| `report/`、`viewer/` | 汇总 JSON/CSV/视频与查看器启动记录 |

正式流程通过门槛并冻结后才产生 `formal/` 下的对应产物。已有诊断和能力试跑不能作为正式主表。精确恢复、失败记账和候选不可变规则见协议。

## 命令与用途

以下命令均在仓库根目录运行，并使用独立锁定环境。主 CLI 默认读取 [configs/main.json](configs/main.json)，支持 `--config` 指定其他配置；同一输出目录会校验配置一致性。主 CLI 可能建立运行元数据，`audit` 和 `report` 会写入对应产物；直接查看现有报告无需运行它们。

```bash
# 仅执行 Exp2 接口测试。
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m pytest -q tests/exp2

# 记录数据、设备和源码审计；或根据既有记录重新生成报告。
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked exp2 audit
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked exp2 report
```

其他操作使用相同 Pixi 前缀：

| `exp2` 后的参数 | 用途 |
| --- | --- |
| `check` | 通过主 CLI 执行 `tests/exp2` |
| `diagnose-dp --stage D0 --gpu 5` | 原生示范回放；其他阶段包括 D1、D2、D3_open、D3_red、D3_blue、D4、D2_sampler、D2_quaternion、capability_m1、api_mujoco、reference |
| `design --stage capability` | 真实 API 神经候选能力试跑 |
| `design --stage reconstruct` | 真实 API MuJoCo 重建与校准 |
| `design --stage library` | 门槛后的正式技能库设计 |
| `train` / `evaluate` | 冻结后的正式配对训练 / 目标评估 |
| `viewer --port 8086` | 显示保存轨迹；不执行策略或推进物理 |

已完成的诊断按现有记录复用，失败结果保留；再次调用命令不构成重抽失败的授权。真实 API、训练和模拟操作遵循各自预算与恢复契约。

查看器示例：

```bash
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked exp2 viewer --port 8086
# 在本地终端执行，server-host 替换为实际 SSH 主机。
ssh -N -L 8086:127.0.0.1:8086 server-host
```

浏览器打开 [本机转发地址](http://127.0.0.1:8086)。可用 `--trace runs/exp2/rebuild_20260915/diagnostics/D3/move_red/rollouts/1000/trace.jsonl` 指定其他保存轨迹；状态日志为 20 Hz，原始 RGB 采样为 1 Hz。

## 示范拆分入口

独立模块见 [demonstrations/decouple.py](../../src/appl/demonstrations/decouple.py)。已有运行使用以下脚本；它记录的是指定输入与输出的同一会话，不是任意新数据集的通用 CLI：

```bash
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python runs/exp2/demonstration_processing_20260915/launch.py
```

匹配的已完成会话会校验并返回已有产物，不新增 API 请求；脚本仍需要原运行的 provider 配置与凭据，且会更新运行回执。只查看结果时，直接打开 [拆分报告](../../runs/exp2/demonstration_processing_20260915/REPORT.md)、[验证记录](../../runs/exp2/demonstration_processing_20260915/validation.json)与[输出 manifest](../../data/exp2/processed/drawer_exchange_20260915/manifest.json)。

输出的 `datasets/<skill_id>/` 内含数据索引、原始片段、`media/` 和假设说明；`_session/` 保存 API/工具账本与不可变方案版本。这一流程没有训练或模拟器调用，片段数量不代表独立示范数量。
