# Push training_v3 — cut_v5 三策略训练

核对日期：2026-09-21。按用户授权，由 gpt-6-astra/xhigh 编写策略，鼓励 Diffusion Policy并要求说明其他模型的选择理由。三个模型独立完成固定 20,000 步训练，保留最终 EMA。

| Policy | API 选择的模型 | 样本数 | 覆盖片段 | 参数数 | EMA 重载最大误差 |
| --- | --- | ---: | ---: | ---: | ---: |
| tool_waypoint_v1 | Compact recurrent mixture-of-mode residual behavior cloning with geometric route anchor and heteroscedastic command head. | 1,362 | 4/4 | 2,005,168 | 0.0 |
| piece_contact_graph_v1 | Recurrent contour graph, constrained differentiable contact mixture, learned weak response/uncertainty model and residual behavior cloning. | 15,765 | 22/22 | 2,147,601 | 0.0 |
| piece_goal_field_v1 | Compact convolutional goal-field encoder and GRU mode-conditioned inverse servo with bounded native-command residual and heteroscedastic losses. | 15,765 | 22/22 | 2,005,168 | 0.0 |

训练只采样各自原始数据组中的有效样本；完整逐片段覆盖和过滤原因保存在 prepared/result.json 与 metadata.json。输入预处理、模型、损失、目标/动作转换、prior 和调用文档均为 API 原文。

动作样本覆盖与可信的物体目标/几何/响应监督覆盖不同。以下为预处理缓存中 API 有效掩码的实际计数；有效仍只代表其自动标注规则通过，并非人工验证的真值。这三个物体掩码不适用于工具 waypoint policy，它使用独立的 TCP 目标。

- tool_waypoint_v1: 使用 TCP 目标，以上物体掩码不适用。
- piece_contact_graph_v1: {"goal_valid": 15165, "geometry_valid": 15431, "response_valid": 15381}。缺少可信物体目标的片段：[{"segment_id": "a_bar_stage", "action_rows": 600}]。这些行保留动作监督，以目标缺失掩码和终点 TCP 条件参与训练。
- piece_goal_field_v1: {"goal_valid": 15165, "geometry_valid": 15431, "response_valid": 15381}。缺少可信物体目标的片段：[{"segment_id": "a_bar_stage", "action_rows": 600}]。这些行保留动作监督，以目标缺失掩码和终点 TCP 条件参与训练。

两个搬运模型使用物体相对的局部几何与运动表示，但最终仍输出六维原始 v/w，包含受限残差；接触阶段没有硬性投影成纯二维动作。工具换位模型、轮廓接触图模型和目标场伺服模型各自拥有独立权重。


## API 模型选择理由

### tool_waypoint_v1

Conditional sequence diffusion could model multiple reconfiguration trajectories, but four waypoint examples do not identify that distribution. A fixed geometric route anchor and small gated residual are more data-efficient and inspectable. The one-step executor is not the reason for choosing a one-step residual.

### piece_contact_graph_v1

Diffusion over contact modes or short strokes is plausible, but would learn a poorly identified multimodal distribution from two episodes. Deterministic boundary candidates already express the important alternatives. This implementation learns scores, response corrections and low-dimensional residuals instead of a high-dimensional sampled trajectory.

### piece_goal_field_v1

Goal-conditioned action-sequence diffusion was considered for repeated L/R strokes. With no contact audit and only 22 hindsight invocations, causal closed-loop correction with a short masked recurrent history is a stronger constraint than sampling long multimodal futures. Reset modes preserve nonmonotonic behavior without a monotonic-progress loss.

## 调用与核验

入口为 [API 调用目录](../runs/push_letters/training_v3/package_00/source/POLICY_CATALOG.md)、[完整接口文档](../runs/push_letters/training_v3/package_00/source/CALLING.md) 和 [交接约定](../runs/push_letters/training_v3/package_00/source/HANDOFF.json)。训练后的清单见 [library.json](../runs/push_letters/training_v3/library.json)。

核验 34 次 API 调用和 15 个源码/文档文件的来源，费用估算 $8.676944（非账单）。检查有限梯度、权重更新、数据/权重哈希和固定随机种子的 EMA 重载。原 cut_v3/v4/v5 及旧训练框架保持不变。

这是实现与训练完成证据，不是任务成功率或泛化证据；本轮没有真机执行。实际调用接口检查见 completion_receipt.json 中单独记录。

通过仓库本地 `real_robot.policy_training_v3.inference.PolicyProcess` 加载，显式传入 `gpu=4`（配置允许 4/5/7）。每个模型的 `last.pt` 路径和哈希在 library.json 中。`inventory` 提供当前几何实例，`act` 接收 waypoint 或物体空间目标及独立调用状态；`close` 结束隔离进程。具体参数和交接语义以 API 文档为准。

| Policy | 有限六维候选动作 | 过期输入拒绝 | 非刚体目标拒绝 | 中断 | 解码状态 |
| --- | --- | --- | --- | --- | --- |
| tool_waypoint_v1 | True | True | True | True | CONTROLLER_UNVERIFIED |
| piece_contact_graph_v1 | True | True | True | True | CONTROLLER_UNVERIFIED |
| piece_goal_field_v1 | True | True | True | True | CONTROLLER_UNVERIFIED |

该检查仅使用一帧已有配对观测和当前 TCP/物体原位目标，确认实际入口可加载并处理输入；没有任务 rollout、性能评分或额外训练。动作解码仍禁用：API 未获得经核验的原始控制器/follower，返回的是候选 v/w，尚不能直接下发真机。

## API 声明的适配与限制

No pretrained segmenter, semantic recognizer, audited masks, tabletop/letter-top calibration, rod geometry or recording follower exists in the supplied assets. Implement shape-independent HSV foreground, calibrated dual-view local repair and causal rigid-mask tracking instead. Offline seven initial visual seeds come from inspected episode-zero images, never character embeddings. Workspace is a board-plane proxy with a conservative 30-mm design margin, not an estimated/calibrated standard deviation. Optional caller-supplied verified top-plane frame and fresh external union mask use the same conversion. Endpoint masks and motion are unreviewed weak labels with explicit per-loss confidence; all finite action labels remain, with missing object-goal masks and endpoint tool goals on ambiguous rows. Models use four causal snapshots within twelve records, a native-resolution 64x64 local field plus coarse whole-scene occupancy rather than twelve full 256-square RGB frames. Local graph and dense servo are independently weighted; graph planning is a one-step proxy, not a certified mechanics solver. No physical SE(2) augmentation is used without robot reachability and w-frame verification. Semantic matching/layout is executable only with caller-supplied current semantic hypotheses; absence is explicitly uncertain, never recognized by assertion.

All training data are two episodes in a narrow tool/material/color regime. No performance study, held-out validation, calibrated uncertainty, verified contact, success labels, or demonstrated unseen-shape transfer. Color foreground is not open-world perception and can fail under lighting, overlap, occlusion, new colors and merged components. Holes below the 3.9-mm raster and morphology scale are unreliable. Board-plane and TCP proxies are not tabletop/tip metrology; geometric screens are not arm/rod swept-volume safety or reachability. Semantic upright orientation needs an independent live cue. Command magnitudes/frames and physical speed caps are unverified unless an explicit audited intent mapping is supplied; even then the decoder remains disabled because the recording follower and collision model are absent. Valid act calls return learned offline candidates, not fabricated joint commands or physical readiness.
