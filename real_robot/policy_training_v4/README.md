# Push training_v4 的本地调用入口

这一版实现 cut_v6 的单个 `shape_push_v1` 几何决策策略。
模型、数据转换、标签推导和执行请求转换均由 Runtime API 编写；
本目录提供开发者编写的隔离运行、训练、来源检查和进程接口。

使用独立锁定环境：

```bash
/home/users/oscar/.pixi/bin/pixi run --manifest-path real_robot/training_v4_environment/pixi.toml --locked --no-install python -m real_robot.policy_training_v4 status
```

训练后的权重与 API 源码在 `real_robot/runs/push_letters/training_v4/package_00/`。
实际训练记录以 [library.json](../runs/push_letters/training_v4/library.json) 为准。
不要对同一目录再次启动设计或训练；保留已有尝试及其来源记录。

## 给调用 Agent 的文件

- [策略目录](../runs/push_letters/training_v4/package_00/source/POLICY_CATALOG.md)：选择哪个组件、谁负责什么。
- [完整调用合同](../runs/push_letters/training_v4/package_00/source/CALLING.md)：状态、时间、目标、planner 交接。
- [shape_push_v1 参数说明](../runs/push_letters/training_v4/package_00/source/shape_push_v1_USAGE.md)：输入字段和单位。
- [结构化交接合同](../runs/push_letters/training_v4/package_00/source/HANDOFF.json)。
- [实际训练与覆盖报告](../reports/PUSH_TRAINING_V4.md)。

## 进程接口

下面是集成示意；`observation`、`call` 和 `executor_contract` 由宿主按 API
文档构造，`output` 必须是尚不存在的日志目录。GPU 编号受本轮配置约束。

```python
from real_robot.policy_training_v4.inference import PolicyProcess

worker = PolicyProcess(
    "real_robot/configs/push_training_v4.json",
    "shape_push_v1",
    output="/tmp/push_v4_call_unique",
    gpu=4,
    revision=0,
)
try:
    inventory = worker.inventory(observation, scene_version="fresh_scene_1")
    result = worker.act(
        observation, call, reset=True,
        executor_contract=executor_contract,
    )
finally:
    worker.close()
```

`inventory` 返回当前图像里的几何对象，不提供持久身份或词义识别。
宿主提供选中实例、参考轮廓及其 SE(2) 目标、标定和工作区。
`act` 返回 `status`、`decision`、`diagnostics`；进程内部保存策略记忆。
当传入执行合同且有有效提案时，还返回 `executor_request`。
缺少已绑定的执行器时，几何提案仍可返回，执行请求为禁用状态。

决策的核心是接触点二维坐标、平面推动方向、短行程长度和边界编号。
数值预测虽然包含六个数，但含义是 `[x, y, dx, dy, length, loop_id]`。
升降、接近、换位的路径由外部 planner/controller 完成。
本接口只生成数据，不连接或驱动机器人。

`observation.t` 与 `recv_time` 必须来自共同的实时钟。
原录制数据中的两个时钟有偏差，直接原样回放会触发过期观测状态。
本轮调用检查中的合成时间仅是接口测试夹具；没有修改原始录制时间。

`source/agent.py` 另有 API 编写的几何调度器。上述进程包装器提供的是
`inventory/act`；上层语义 Agent 需负责物体身份、目标布局与 planner 反馈。
迁移到另一台电脑的独立部署包不属于本目录的训练产物。
