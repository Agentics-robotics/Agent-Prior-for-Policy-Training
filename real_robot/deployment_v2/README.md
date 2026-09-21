# training_v2：本地调用与 Agent 文档

这是已完成 `training_v2` 的独立接口，模型与实际训练覆盖见
[TRAINING.md](../policies/push_v2/TRAINING.md)。机器人侧相机采集、通信和动作执行
由使用方适配。

## 给部署 Agent 的入口

先读 [POLICY_CATALOG.md](../policies/push_v2/source/POLICY_CATALOG.md) 和
[本轮实际训练覆盖](../policies/push_v2/TRAINING.md)，再按 policy ID 读取：

| Policy | Prior / 模型机制 | 使用说明 |
| --- | --- | --- |
| `contour_push` | [contour_push_PRIOR.md](../policies/push_v2/source/contour_push_PRIOR.md) | [contour_push_USAGE.md](../policies/push_v2/source/contour_push_USAGE.md) |
| `visual_push` | [visual_push_PRIOR.md](../policies/push_v2/source/visual_push_PRIOR.md) | [visual_push_USAGE.md](../policies/push_v2/source/visual_push_USAGE.md) |

[HANDOFF.json](../policies/push_v2/source/HANDOFF.json) 的 `policies[policy_id]`
列出选择依据、进入/继续/退出条件、后继准备状态、切换步骤、失败信号、memory
重置规则和训练证据限制。[CALLING.md](../policies/push_v2/source/CALLING.md)
给出完整 observation/call 字段、坐标、单位和返回状态。
这些文件均为 API 提交原件；本机实际训练覆盖以训练报告为准。

两个 policy 是同一实例搬移任务的不同实现，没有固定调用顺序。
`goal_observed` 只表示图像目标匹配；`successor_ready=false` 明确表示工具撤离与
安全交接仍未得到确认。切换时保存目标 foreground，刷新物体关联，并通过
`reexpress_goal` 重新表达目标；不要直接把相对于旧模板的角度复制给新模板。

## 文件与环境

同步本轮代码后，模型布局为：

```text
real_robot/deployment_v2/           本地调用入口
real_robot/policies/push_v2/        manifest 和完整 API 源码/文档
real_robot/checkpoints/push_v2/     contour_push.pt、visual_push.pt
```

两个 checkpoint 在训练完成后由最终 EMA 无损导出，另行下载，不进入 Git。
导出保留原推理 loader 使用的 spec，并逐张量核对；没有量化或重新训练。
代码与权重必须匹配同一个 manifest，不能混用 v1 的 call 字段、memory 或权重。

已有本轮代码时，在接收电脑的仓库根目录下载权重；`SERVER` 换成服务器地址：

```bash
rsync -avP oscar@SERVER:/home/users/oscar/agent_learning/Agent-Prior-for-Policy-Training/real_robot/exports/push_v2/push_v2_weights.tar.gz .
tar -xzf push_v2_weights.tar.gz
```

也可下载同目录的 `push_v2_bundle.tar.gz`，独立解压到 `push_v2_bundle/`；完整包
包括接口、API 源码/文档、独立环境锁和两个权重，不依赖 Git 远端是否已更新。
同目录的 `SHA256SUMS` 和 `bundle_manifest.json` 提供文件校验信息。

使用现有独立推理环境，无需仿真环境：

```bash
pixi install --manifest-path real_robot/deployment/pixi.toml --locked
pixi run --manifest-path real_robot/deployment/pixi.toml --locked python your_host_script.py
```

需要 Linux x86_64、NVIDIA CUDA 驱动、`nvidia-smi`、`bubblewrap`、
`libseccomp.so.2`、Landlock ABI >= 3 和可用的非特权用户 namespace；
OpenCV 需要系统 GL/GLib 库。运行时无需 API key、原始数据、训练缓存或训练 runs。

## Python 使用

```python
from real_robot.deployment_v2 import PushPolicy

with PushPolicy("contour_push", gpu=0) as policy:
    scene = policy.observe_scene(observation, scene_version)
    # HLA 从 scene['instances'] 选择当前实例，按该 policy 的 USAGE.md 构造 call。
    goal = policy.preview_goal(
        call["template"], call["spatial_goal"], call["workspace_polygon_uv"]
    )
    result = policy.act(observation, call, reset=True)
    # 同一次调用的后续帧：保持 call 不变，reset 使用默认 False。
    # result = policy.act(next_observation, call)
    # 交接时保存 goal['foreground_half']；重新关联当前实例后调用：
    # transformed = policy.reexpress_goal(new_template, goal['foreground_half'])
    # 然后按返回状态检查，并以新 call、reset=True 启动目标 policy。
```

`observation`、`scene_version`、`call` 和 `new_template` 来自使用方真实观测与
Agent 的决定；上例只展示接口连接。`observe_scene` 的 scene memory 和 `act`
的 motor memory 由 worker 各自保存，不跨 worker 或 policy 共享。
同一对象上的调用按顺序进行；不同场景使用不同实例。

`policy.decode_action(action, controller_contract)` 只生成命令描述，仍不操作
硬件。缺少必要输入时应按 API 返回的状态处理，不将空 action 当成零速度命令。

本地 host 只转发 API 函数并核对源码、checkpoint、数据形状及 worker 生命周期；
没有新增 controller、感知规则或 policy 选择策略。训练完成与接口检查不等于
真机性能、实时性或未知字母泛化已经通过验证。
