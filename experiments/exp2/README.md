# Exp2：当前唯一实验入口

**核对日期：2026-09-20。已完成并停止，无待执行实验。**

当前只展示 **Exp2_new：五任务 × 每任务 15 个位置 OOD 初态 × 三种方法，共 225 个完整结果**。三种方法使用相同初态和修正后的二值夹爪执行器，最多 5,000 个物理步。历史错误夹爪结果、其他轮次、额外 ID/OOD 结果和诊断均归档；原始记录保留。

| 方法 | 模型数量 | 部署时调用 API | 当前 OOD 成功 |
| --- | ---: | --- | ---: |
| Naive DP | 5，每任务一个 | 否 | 5/75 |
| Agent Prior DP / SinglePrior | 5，每任务一个；GPT-6 Astra/xhigh 编写 prior 与策略 | 否 | 10/75 |
| APPL / GPT-6 Astra/xhigh | 36 个子策略 | 是，依据 prior、状态和交接信息选择 | 30/75 |

## 先看这些

- [完整实验报告](reports/REPORT.md)、[失败分析](reports/FAILURE_ANALYSIS.md)、[恢复案例](reports/RECOVERY_CASES.md)
- [每个 API policy 的 prior 与源码](reports/PRIOR_CATALOG.md)
- [225 个视频](reports/VIDEOS.html)、[结果表](reports/tables/episodes.csv)
- [当前资产清单](../../runs/exp2/current/manifest.json)：唯一确定这 225 个回合和 46 个模型的清单
- [历史归档入口](../../archive/Exp2_history_20260920/README.md)、[迁移核验](../../archive/Exp2_history_20260920/migration/validation.json)

## 当前目录与依赖

```text
data/exp2/                          原始示范、环境资产、任务输入
  demonstrations_v2/              抽屉示范；冻结清单还保留原验证数据
  scaleup/<task>/demonstrations/   其余四任务的示范
  scaleup_astra_xhigh/             GPT-6 的切分数据

src/appl/                          冻结的数值/环境/工具框架
  scaleup/                        五任务定义、示范获取与任务协议
  envs/                           仿真环境、状态与动作接口、几何判定
  dp_baseline/                    完整任务 DP 的训练/推理
  demonstrations/                 提供给 API 的示范阅读与切分工具
  prior_policies/                 独立 prior policy 训练与部署工具
  vendor/                         Diffusion Policy 网络实现

experiments/exp2/                   三条方法的编排与实验说明
  current.py                      只读状态/完整性检查入口
  single_policy/                  API 设计完整任务 prior DP、训练接口
  astra/                          GPT-6 技能切分、prior 设计与训练编排
  binary_gripper/                 两个 baseline 的修正夹爪评估器
  exp2_new/                       APPL 修正夹爪部署、预算和审计
  analysis/                       当前结果核验及实际执行过的续跑代码
  configs/                        冻结配置；历史配置链接不代表新运行计划
  reports/                        完整报告、统计表、图、视频索引

runs/exp2/current/                 当前科学资产（实际位于实验存储盘）
  manifest.json                   225 回合、46 模型及原路径/哈希
  episodes/<method>/<task>/OOD/    15 回合：结果、轨迹、API 日志、视频
  models/<method>/<task>/<policy>/ 源码、prior、提交、训练和 checkpoint
  inputs/                         任务定义、API 切分、单策略数据、尺度与部署配置

archive/Exp2_history_20260920/      旧运行、旧诊断代码、旧数据和全部旧报告
```

训练流程：环境与示范 → 固定 DP / API 单策略设计 / API 技能切分和多 prior 设计 → 数值训练 → 保存 checkpoint。评估流程：同一组 75 个 OOD 初态 → 两个直接部署 baseline / API 选择 APPL 子策略 → 同一几何目标判定 → 结果与视频。

环境、任务、示范采集、工具、训练器和判定器由开发者实现；切分、heuristic/prior、API policy 源码与交接文档、APPL 部署决策由 Runtime API 产生。此次整理没有改动这些内容。

## 只读命令

在仓库根目录执行；不会训练、仿真或发送 API 请求。

```bash
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m experiments.exp2.current
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m experiments.exp2.current --verify
```

`--verify` 读取 46 个 checkpoint 并检查哈希，因此需要较多磁盘读取。接口测试：

```bash
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m pytest -q tests/exp2
```

## 冻结实现与兼容路径

旧 `M0`、`M1_*`、`Exp2_new`、`binary_gripper_v1_20260919` 等运行入口现在是指向归档或当前资产的兼容链接，不是并列的活跃实验。冻结配置、原始 API 日志仍能按当时路径解析。目录名中的历史版本不改变原实验的真实设置。

`src/appl`、三个当前评估/设计模块及环境锁被原实验的完整性校验覆盖，保持原字节；其中少量历史框架接口因此仍在。一次性诊断/恢复脚本已移入归档；当前代码不导入归档代码。请从本页和 `current.py` 进入，勿因历史脚本存在而重启旧实验。

[部署协议](EXP2_NEW.md)、[二值夹爪修正](BINARY_GRIPPER.md)、[单策略设计协议](SINGLE_POLICY.md)、[GPT-6 切分与训练来源](ASTRA_XHIGH.md)保留为执行时的原始规范；其中的历史任务数量和预算不替代当前 manifest。所有未来 Exp2/新实验使用 `xhigh`，Exp1 相关实验使用 `max`。Exp1 的代码、数据、模型与报告完全保留。
