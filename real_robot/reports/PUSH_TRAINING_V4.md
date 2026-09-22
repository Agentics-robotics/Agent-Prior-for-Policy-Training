# Push training_v4：实现、预处理及训练记录

核对日期：2026-09-21。输入为冻结的 cut_v6；本轮完成 Runtime API 编写代码、实际数据转换、模型训练和本地调用检查。

**最主要的限制：47 个决策锚点最终只有 4 个留下有效弱标签，覆盖 4/12 个学习片段。** 这轮是可检查的训练原型，尚无证据说明已学会完整任务或获得泛化能力。原始 16,745 行及 executor-only 证据保留。

## 实际模型

唯一学习策略 `shape_push_v1` 对物体边界上的联合候选打分。原始第三视角 RGB 经颜色分割、相机/平面几何转换成轮廓；网络输入为 256×9 点特征，以及 2,688×29 个候选特征和有效性 mask。轮廓、目标和短历史在物体/目标相关坐标中表达，原始图像不直接进入可训练网络。

点编码器 9→64→64 加 max pooling；候选评分器 93→64→64→1；共 **15,041 个可训练参数**。候选组合来自 128 个边界点、7 个方向和3种短行程。输出为接触点二维坐标、推动方向、短行程长度及边界编号；`predict` 的六个数不是六轴 EE 指令。抬起、移动、下降由外部 planner/controller 执行。

API 选择小型候选评分网络，并说明其比完整轨迹 Diffusion 更符合少量几何决策监督。实现采用颜色/轮廓方案，未获取或运行 SAM/Qwen 权重；这与 cut 阶段的感知依赖设想不同，限制了颜色/材质泛化。

## 数据处理已经执行

API 从图像与 flange 变换推断工具偏移、有效水平顶面、物体轮廓和目标变换，再提取接触点及短平移前缀；检查接触关联、配准、抬升/旋转、短行程以及候选映射，最后审阅叠加图并排除遮挡/局部轮廓等不可信标签。标签和标定均为推断，没有人工接触真值。深度因尺度/配准未核实而未使用。

`prepared/metadata.json` 与 `evidence/final_anchor_audit.json` 是最终覆盖依据；早期 `A_*/B_*` 图和 JSON 是复核前诊断，不可当作最终保留清单。未来帧仅用于离线目标推导，模型观测窗口不越过锚点。

| 片段 | 有效决策点 |
|---|---:|
| `a_zigzag` | 1 |
| `a_ring_and_clearance` | 0 |
| `a_cluster_reorganization` | 0 |
| `a_legged_and_loop_finish` | 0 |
| `a_open_corner_retries` | 0 |
| `a_bar_finish` | 0 |
| `b_zigzag_and_clearance` | 0 |
| `b_loop_and_ring` | 1 |
| `b_legged_recontacts` | 1 |
| `b_loop_finish` | 0 |
| `b_open_corner` | 0 |
| `b_bar_finish` | 1 |
| `a_executor_trace` | 0 |
| `b_executor_trace` | 0 |

排除原因：

- 14 个：No supported first contact: RGB/tool/actor association or height gate failed
- 19 个：Goal topology mismatch or rigid residual exceeds 10 mm
- 1 个：Prefix motion below 4 mm; pause/tilt/lift unsupported
- 4 个：Actor/goal segmentation unavailable or occluded
- 1 个：Actor motion not distinguishable from registration noise: possible approach/occlusion motion, not a supported short interaction
- 2 个：Contact not corroborated on same actor at prefix endpoint
- 1 个：API overlay review: shadow-filled notches produce a rectangle rather than full bar; symmetry/goal yaw unreliable
- 1 个：API overlay review: tool occlusion truncates bar contour; reject despite numerical match

## 实际训练与验证

GPU 4、seed 0，按 API 预先声明的预算完成 **500 次 AdamW 更新**，使用最终 EMA。采样曝光总数 2,000 是重复抽样次数，不是独立样本数。权重 L2 变化 6.944989；最终 checkpoint 重载预测最大误差 0.0。无性能驱动选模、额外训练或隐藏测试。

执行过 6 次零更新预处理检查，其中 2 次失败保留了原始代码和错误，由 API 修复。框架有 18 项切分/因果/稀疏监督接口测试通过。训练后的独立进程加载检查返回真实几何提案，并核对缺失执行绑定、过期观测、反射目标和中断状态。

调用检查使用明确标识的共同钟、离线推断标定及工作区测试夹具，仅证明接口可调用。直接传入原录制 t/recv_time 会因约 3.2 秒钟差得到 stale_scene。没有更改录制时间，也没有把离线标定认作实机标定。

本轮 **77 次 gpt-6-astra/xhigh 调用**；请求与响应均核对；估算 API 成本 **$16.095949**，不是账单金额。18 个提交文件有对应 API write_file 来源。数据/模型科学内容由 API 编写，外层执行和本报告由开发者编写。

## Agent 接口与成品

上层 Agent 提供物体身份、参考轮廓、几何目标和实时标定；策略输出接触提案；执行转换器构造 planner 请求。附带的几何调度器支持 prepare→重观测→push→重观测；文字识别、布局规划、持续跟踪及真实 planner/controller 绑定仍由宿主提供。

- [本地调用入口](../policy_training_v4/README.md)
- [API 参数与调用说明](../runs/push_letters/training_v4/package_00/source/shape_push_v1_USAGE.md)
- [API 源码与目录](../runs/push_letters/training_v4/package_00/source/POLICY_CATALOG.md)
- [最终权重](../runs/push_letters/training_v4/package_00/training/shape_push_v1/last.pt)
- [模型来源与训练库](../runs/push_letters/training_v4/library.json)
- [最终逐锚点审计](../runs/push_letters/training_v4/package_00/prepared/evidence/final_anchor_audit.json)
- [独立调用检查](../runs/push_letters/training_v4/calling_check/result.json)
- [完成凭据](../runs/push_letters/training_v4/completion_receipt.json)

早期 cut_v3/v4/v5/v6 发布文件、training_v3 源码及三个 checkpoint 再次通过哈希检查。未进行机器人动作、提交/push 或独立部署打包。
