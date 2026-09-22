# training_v3 神经网络输入可视化

核对日期：2026-09-21。开发者只读展示已冻结训练缓存，不改 API 设计、预处理、训练或模型。

交互视图显示三个模型的实际条件路径：四时刻 `maps [4,8,64,64]`、`state [4,112]`、`history_valid [4]`，以及当前 `base [6]`。接触图模型额外读取当前 `nodes [32,16]` 和 `score [32]`；`basis [3,3]` 用于线速度坐标转换。形状省略 batch 维。底层输入携带节点历史，但图分支只使用当前一帧。两个搬运模型使用相同几何缓存，各自独立训练。

RGB 是几何预处理的输入，不直接进入这三个网络。展示的 RGB 为原记录缩略图；地图为训练缓存原始 float16 数值，使用无损游程编码嵌入，逐帧验证字节一致。112 维状态中定义了 91 维特征，其余 21 维零填充。TCP 图和相对量沿用 API 实现中的 EE 位姿代理，不构成物理工具尖端或桌面标定验证。

样本选用指定片段监督范围的中点，不按动作预测或性能筛选：

| 样本 | 当前配对行 | 片段终点目标行 | 用途 |
| --- | ---: | ---: | --- |
| a_l_approach_tool | 6185 | 6250 | 工具策略输入 |
| a_l_move | 7134 | 8149 | 有目标位移的搬运输入 |
| a_d_stage | 4919 | 5199 | 缓存目标位移为零的搬运输入 |

每个样本展示记录行偏移 `[11,7,3,0]` 的真实四时刻历史。这是训练输入展示；训练目标来自片段终点，在线调用由 Agent 提供目标。训练与在线 TCP 目标的已有差异见[可学性审阅](../PUSH_V3_LEARNABILITY_REVIEW.md)。可视化未运行网络前向，也不报告成功率。

局部前六通道覆盖物体中心附近约 25×25 cm；全景占据覆盖 1×1 m。目标在局部窗口外时，目标 SDF 可能全部达到截断值；末端在窗口外时，TCP 热图可能为零。`a_d_stage` 当前目标流为零、几何退出标记为 1，这是原缓存数值，不足以单凭这一帧判断目标标注正确与否。

文件：

- [生成脚本](../visualize_input_conditions_v3.py)、[交互模板](template.html)、[数值数据](data.json)。
- [来源与哈希](receipt.json)、[交互核对范围](interaction_check.json)。
- [工具策略通道图](tool_channels.png)、[搬运通道图](piece_channels.png)、[零目标位移样本](zero_goal_channels.png)。
- 源码依据：[特征构造](../../policies/push_v3/source/geometry.py)、[网络](../../policies/push_v3/source/networks.py)。

复现（仓库根目录，使用现有锁定环境）：

```bash
MPLCONFIGDIR=/tmp/push-v3-viz-mpl /home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked --no-install python -m real_robot.reports.visualize_input_conditions_v3
```

模板仅将冻结数据渲染成图；输出的会话可视化路径由生成脚本中的 `VIZ` 指定。
