# training_v3：跨电脑推理与 Agent 入口

本目录是 cut_v5 三个已训练模型的独立本地接口。科学源码和调用文档逐字节保留
Runtime API 提交；权重无损导出最终 EMA，不重新训练。可从任意解压位置运行，
使用当前 Python 环境及本机 GPU 编号，不使用训练服务器的 Pixi 路径或 GPU 4/5/7 限制。

## 接收电脑需要什么

支持当前锁文件的 Linux x86_64、NVIDIA GPU 和兼容 CUDA 驱动；需要 Pixi、
`nvidia-smi`、`bubblewrap`、`libseccomp.so.2`、Landlock ABI >= 3、可用的非特权
用户 namespace，以及 OpenCV 所需的系统 GL/GLib 库。默认使用 GPU 0，启动时要求
至少 2 GiB 空闲显存。Windows/macOS/CPU 推理没有在这个包中实现或验证。

无需 OpenAI API key、训练数据、训练缓存或训练 runs。环境首次安装需要下载锁定依赖；
随后模型推理不调用外部 API。当前包没有外部感知权重。

## 两种下载方式

**完整包现在即可直接使用，不依赖 Git 远端是否包含本轮代码。** 在接收电脑执行，
将 `SERVER` 替换成训练服务器地址。以下下载只读取服务器：

```bash
scp oscar@SERVER:/home/users/oscar/agent_learning/Agent-Prior-for-Policy-Training/real_robot/exports/push_v3/push_v3_bundle.tar.gz .
scp oscar@SERVER:/home/users/oscar/agent_learning/Agent-Prior-for-Policy-Training/real_robot/exports/push_v3/SHA256SUMS .
sha256sum --ignore-missing -c SHA256SUMS
tar -xzf push_v3_bundle.tar.gz
cd push_v3_bundle
pixi install --manifest-path real_robot/deployment/pixi.toml --locked
pixi run --manifest-path real_robot/deployment/pixi.toml --locked python -m real_robot.deployment_v3 check --gpu 0
```

完整包和仅权重包都是独立、不可覆盖的导出文件；不要解压覆盖旧版本目录。
`SHA256SUMS` 包含两个包的校验值，`--ignore-missing` 只校验实际下载的文件。

**如果 Git 已同步本轮 v3 代码**，在接收电脑仓库根目录下载权重即可：

```bash
scp oscar@SERVER:/home/users/oscar/agent_learning/Agent-Prior-for-Policy-Training/real_robot/exports/push_v3/push_v3_weights.tar.gz .
scp oscar@SERVER:/home/users/oscar/agent_learning/Agent-Prior-for-Policy-Training/real_robot/exports/push_v3/SHA256SUMS .
sha256sum --ignore-missing -c SHA256SUMS
tar -xzf push_v3_weights.tar.gz
pixi install --manifest-path real_robot/deployment/pixi.toml --locked
pixi run --manifest-path real_robot/deployment/pixi.toml --locked python -m real_robot.deployment_v3 check --gpu 0
```

Git 代码应包含 `real_robot/deployment_v3/`、`real_robot/policies/push_v3/`、
`real_robot/policy_training_v3/{__init__,public,security}.py` 及共享推理环境/工具。
截至这次打包，这些新增 v3 文件仍在本地工作区，未由本任务提交或推送；
仅执行 `git pull` 不会取得它们。权重在 `real_robot/checkpoints/push_v3/`，不进入 Git。

`check --files-only` 只核对源码/权重哈希；普通 `check --gpu 0` 还会依次加载三个
模型，输出 `ready` 和最终 `passed`。这是安装/加载检查，没有观测输入、任务执行或机器人动作。
同一环境中原有 `pixi run ... check` 快捷任务仍属于 v1，v3 必须使用上方完整模块命令。

## Agent 从哪里开始读

1. [POLICY_CATALOG.md](../policies/push_v3/source/POLICY_CATALOG.md)：选择和职责。
2. [TRAINING.md](../policies/push_v3/TRAINING.md)：本轮实际训练覆盖与缺失标签。
3. [CALLING.md](../policies/push_v3/source/CALLING.md)：完整 observation/call、坐标和状态。
4. [HANDOFF.json](../policies/push_v3/source/HANDOFF.json)：进入、继续、退出和切换约定。

| Policy ID | 机制 | 使用说明 |
| --- | --- | --- |
| tool_waypoint_v1 | [prior](../policies/push_v3/source/tool_waypoint_v1_PRIOR.md) | [usage](../policies/push_v3/source/tool_waypoint_v1_USAGE.md) |
| piece_contact_graph_v1 | [prior](../policies/push_v3/source/piece_contact_graph_v1_PRIOR.md) | [usage](../policies/push_v3/source/piece_contact_graph_v1_USAGE.md) |
| piece_goal_field_v1 | [prior](../policies/push_v3/source/piece_goal_field_v1_PRIOR.md) | [usage](../policies/push_v3/source/piece_goal_field_v1_USAGE.md) |

## 接入自己的相机和机器人状态

在上述 Pixi 环境运行自己的 host 脚本。`observation` 来自真实配对观测：原始
1280×720 RGB 第三视角/腕相机图像、当前相机标定、T_base_ee/T_base_flange、
q/实测 dq/tau_ext、时间戳和图像年龄等。所有精确字段以 CALLING.md 为准。
当前目标由调用 agent 给出；不要从示范终点或旧机器人标定中默默填入。

```python
from real_robot.deployment_v3 import PushPolicy

# observation 和 goal_tcp_base 分别由当前采集端和 agent 提供。
with PushPolicy("tool_waypoint_v1", gpu=0) as policy:
    scene = policy.inventory(observation, scene_version="current-scene-1")
    call = {
        "policy_id": "tool_waypoint_v1",
        "call_id": "tool-approach-1",
        "now_s": float(observation["t"]),
        "goal_tcp_base": goal_tcp_base,
        "arrival_mode": "stop",
    }
    result = policy.act(observation, call, reset=True)
    # 后续帧更新 observation、now_s 和实际已执行动作反馈；同一调用保留 memory。
    # 搬运 policy 改为 instance_id/instance_seed_W/inventory_t 加目标 SE2 或轮廓。
```

`inventory` 返回当前几何实例，`act` 返回六维候选 v/w、status、diagnostics；worker
内部保存该调用的 memory。切换 policy、物体、固定目标或重新获取跟踪时重新初始化；
保存原来的世界目标，勿在物体已移动后重复施加旧的相对位移。返回空 action 不等于实际停车。

`decode_action` 保留 API 原行为：始终禁用，不直接输出可下发机器人命令。
模型可在另一台电脑计算候选动作，实际运动还需接收端对接并核验录制时的 v/w
控制语义、控制器、标定及工具几何。这里没有机器人通信或执行模块。
本轮打包和接口检查不证明实时频率、任务成功率或新字母泛化。
