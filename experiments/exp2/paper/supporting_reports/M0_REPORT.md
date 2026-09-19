# M0：12 条完整示范的 Diffusion Policy

整理日期：2026-09-16。这是当前 M0 的唯一结果入口；详细诊断与设置对照已归档到 [Exp2_M0DP](../../archive/Exp2_M0DP/README.md)。

## 保留的实验

**12 条原始示范 → 一个 DP → DDPM100、每次执行 8 步 → 原生环境完整任务评估。**

| 环节 | 固定设置与入口 |
| --- | --- |
| 输入数据 | [demonstrations_v2](../../data/exp2/demonstrations_v2/)；训练 demo1000–1011，验证 demo1012–1014；按完整轨迹划分 |
| 配置 | [m0.json](configs/m0.json)，保留实际运行时的原配置；实际训练 seed=0 |
| 观测 | 最近 2 帧，每帧 47 维：关节位置/速度、TCP 与两物块位姿、抽屉位置/速度、两个目标位置 |
| 缩放 | 训练集 min/max；近常量维度安全处理，四元数组件使用物理范围 [-1,1] |
| 网络与训练 | 原 U-Net [128,256,512]，16,998,408 参数；60,000 次更新，batch 128，余弦学习率；固定使用最后 EMA |
| 动作 | 预测 16 步；因果对齐后执行当前起的 8 步，再规划。每步输出 7 个绝对关节角（弧度）和 1 个夹爪命令 [-1,1] |
| 扩散 | 训练 100 个噪声时间点；推理 DDPM 100 步，clip_sample=true |
| 评估 | ManiSkill/SAPIEN 原生场景；20 Hz，最多 1,500 个动作；完整任务成功判定保持原样 |
| 产物 | [m0](../../runs/exp2/m0/)：训练、模型、开发评估；[confirmation](../../runs/exp2/m0/confirmation/)：同一模型的独立确认 |

`training.seeds`、`evaluation.conditions` 等继承的主实验元数据不代表本次跑过多训练种子或 OOD；本次只有 seed=0 和下表中的 ID 回合。

## 最终结果

| 评估集合 | 初态 seed | 完整任务成功率 | 原始证据 |
| --- | --- | --- | --- |
| 开发评估 | 6000–6004 | **2/5（40%）** | [evaluation.json](../../runs/exp2/m0/evaluation.json) |
| 后续独立确认 | 6005–6019 | **4/15（26.7%）** | [evaluation.json](../../runs/exp2/m0/confirmation/evaluation.json) |

确认成功初态为 6005、6009、6012、6018。这 15 个初态未用于本轮设置选择；开发结果和确认结果分别报告。仅一次训练、确认样本较少，4/15 的 Wilson 95% 区间约为 10.9%–52.0%。

**已得到非零成功的 M0，但尚不稳健，原正式开发门槛未通过。** 本报告不代表 M0/M1/M2 正式 ID/OOD 矩阵完成。

模型：[training/last.pt](../../runs/exp2/m0/training/last.pt)。SHA-256：

```text
cfc5159592886349eb9e9a078c44757d7f53b3ce6befdd5d4f52398fd6c2f0c4
```

[训练回执](../../runs/exp2/m0/training/result.json)记录了真实梯度更新、权重变化、EMA 重载动作误差为 0；全部评估保留初态、逐步轨迹、图像、设备和源码版本。

## 结论与边界

旧模型将近乎不变的四元数组件除以很小的经验标准差，曾把输入放大到约 700；当前使用物理尺度修复了这个数值问题。动作语义、因果窗口和原生示范回放已经核对。

当前主要失败发生在真实接触和抓取阶段。独立确认 15 例都打开抽屉，只有 6 例抬起红块，最后 4 例完成。成功示范没有充分覆盖策略偏离后的恢复状态，是现有证据支持的解释；它还不是唯一原因的因果证明。

设置研究没有找到可靠优于这条基线的方案：120 条数据模型独立确认为 5/15，相比 4/15 只净增一例；DDPM200、增量动作、执行 4 步等对照也未证明稳健改善。因此保留 **12 条 / DDPM100 / 执行 8**。全部正负结果、费用和限制见[归档研究报告](../../archive/Exp2_M0DP/experiments/exp2/M0_SETTING_STUDY.md)。

## 代码与运行

[dp_baseline](../../src/appl/dp_baseline/)只保留 `data.py`（观测、归一化、窗口）、`train.py`（训练、采样、加载）、`__main__.py`（train/evaluate）和包声明。网络、环境、评估器与 GPU 隔离继续复用 `src/appl` 公共实现；`appl.data`、`appl.train` 的兼容导入服务后续 M1/M2。

在仓库根目录执行，调度前确认所选 GPU 空闲；Exp2 仅允许物理 4–7：

```bash
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.dp_baseline train --gpu 7
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.dp_baseline evaluate --gpu 7
```

默认配置就是 `configs/m0.json`。已完成训练/回合会核验并复用；新训练须显式使用新输出目录。独立确认的原配置保存在 [confirmation/configuration.json](../../runs/exp2/m0/confirmation/configuration.json)，可通过 `evaluate --config` 指定；不是另一个训练模型。

`runs/exp2/m0` 是保留模型的便捷链接。原配置与回执中的历史路径保持可用，避免改写冻结记录。归档映射和本次零训练、零模拟器整理的验证见 [M0_LAYOUT.json](M0_LAYOUT.json)。
