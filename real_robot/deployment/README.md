# Push 模型下载与本地调用

这份部署包包含两套已训练模型、API 原始模型/预处理/goal 转换代码、
归一化参数、独立 Pixi 锁文件及本地 Python 接口。机器人端的数据采集、通信和
控制器接入由使用方适配。这里没有网络服务、机器人驱动或 High-level Agent。

## 下载完整包（推荐）

在另一台电脑运行；将 `SERVER` 换成当前服务器的 SSH 地址或 SSH config 别名：

```bash
scp oscar@SERVER:/home/users/oscar/agent_learning/Agent-Prior-for-Policy-Training/real_robot/exports/push_v1/push_v1_bundle.tar.gz .
tar -xzf push_v1_bundle.tar.gz
cd push_v1_bundle
```

也可以把 `scp` 换成 `rsync -avP`，便于显示进度。完整包已经包括所有需要的代码和
两个权重，可独立解压使用，不依赖 Git 远端是否包含最新工作区修改。
服务器同目录的 `SHA256SUMS`、`bundle_manifest.json` 提供下载文件和内部文件校验值。

## 已有包含本部署接口的 Git checkout：只下载权重

确认 checkout 中已有 `real_robot/deployment/`、`real_robot/policies/push_v1/`
及对应 `manifest.json`。在**仓库根目录**执行：

```bash
scp oscar@SERVER:/home/users/oscar/agent_learning/Agent-Prior-for-Policy-Training/real_robot/exports/push_v1/push_v1_weights.tar.gz .
tar -xzf push_v1_weights.tar.gz
```

权重自动放到以下位置，也可以逐个复制这两个文件：

```text
real_robot/
├── deployment/                     # 本地调用器、环境与本说明，随代码分发
├── policies/push_v1/
│   ├── manifest.json               # 权重、API 源码哈希与训练来源
│   └── source/                     # API 原始代码与 CALLING.md，逐字节保留
└── checkpoints/push_v1/             # 不进入 Git，另行下载
    ├── contour_push.pt             # 约 14 MiB
    └── visual_push.pt              # 约 35 MiB
```

这是专供推理的 checkpoint，包含原训练最终 EMA 权重；导出时逐张量核对完全相同。
去掉了 optimizer、普通训练权重和训练专用元数据，没有压缩精度、重新训练或改变模型。
不要将旧 `last.pt` 改名冒充这两个文件；加载器按 manifest 核对内容。
**不需要原始示范、cut 数据、5 GB 预处理缓存、训练日志或 API key。**

## 独立推理环境

当前支持 Linux x86_64、NVIDIA CUDA GPU。需要 Pixi、可用的 NVIDIA 驱动、
`nvidia-smi`、`bubblewrap` 命令、`libseccomp.so.2`，以及支持 Landlock ABI >= 3
和非特权用户 namespace 的 Linux 内核/系统设置。OpenCV 还需要系统 GL/GLib 库。
Ubuntu/Debian 上相关系统包通常为 `bubblewrap libseccomp2 libgl1 libglib2.0-0`；
由该电脑管理员按系统版本安装。缺少隔离能力会明确失败，不自动取消隔离。

在 checkout 或完整包的根目录运行：

```bash
pixi install --manifest-path real_robot/deployment/pixi.toml --locked
pixi run --manifest-path real_robot/deployment/pixi.toml --locked python -m real_robot.deployment check --gpu 0
```

第一次安装需要下载 Python/PyTorch 等依赖；它们没有塞进模型压缩包。
此环境没有安装仿真器，也不依赖原服务器的 Pixi 路径或用户名。
检查只加载两个模型并退出，不采集图像、不调用 API、不训练、不操作机器人。
只核对文件可加 `--files-only`。选定 GPU 启动时至少要有 2 GiB 空闲显存；
这只是加载器门槛，不是实测的全场景显存需求或实时性保证。

一张 GPU 可供两个模型使用，均指定 `gpu=0`。每个 `PushPolicy` 保持自己独立的
调用历史，不能并发调用同一个实例；不同场景/会话使用各自的实例。

## Python 接口

把以下逻辑接入使用方程序，脚本通过上面的 Pixi 环境运行：

```python
from real_robot.deployment import PushPolicy

with PushPolicy("contour_push", gpu=0) as policy:
    inventory = policy.inventory(observation, scene_version="scene_001")
    # 使用方/HLA 根据 inventory 选择实例，按 CALLING.md 构造固定空间目标 call。
    result = policy.act(observation, call, reset=True)
    candidate_dq = result["action"]  # 可为 None；先读取 status/diagnostics

    # 同一调用后续帧：policy.act(next_observation, call)，默认保留历史。
    # 确认控制器契约后可调用 policy.decode_action(candidate_dq, contract)。
    # 该方法仅返回命令描述，不发送硬件命令。
```

换另一个模型只需 `PushPolicy("visual_push", gpu=0)`。如果权重放在别处，传
`weights="/your/path/push_v1"`；目录内仍须是以上两个文件名。
`observation`、`call`、`contract` 都由使用方提供，示例没有伪造这些数据。
`PushPolicy` 使用启动它的 Python 解释器；请勿拿其他环境的 Python 直接运行。
日志默认保留于系统临时目录，也可通过 `output="/new/log/directory"` 指定新目录。

详细的输入字段和调用返回值见
[API 原始 CALLING.md](../policies/push_v1/source/CALLING.md)：

- observation 需要成对的原始第三视角/腕部 RGB、机器人状态、采样索引和相机标定。
  使用原配对索引，不按两套原始时钟重新匹配。
- call 选择当前观测到的实例、固定目标位置/相对朝向、工作区和调用预算。
  字母识别、组词布局和 policy 选择由使用方/HLA 负责。
- 没有标定平面 homography 时，仅支持显式接受近似的小范围图像平面目标。
  感知依赖暖色材质、对比度和可跟踪性；部署整理没有改变这些模型限制。
- 返回的是候选记录命令 dq。单位、关节顺序、控制频率及真正硬件执行由使用方
  与实际控制器对接；下载包没有假定已接通机器人。

两套模型各完成 20,000 次更新，实际监督为 8,606 条、覆盖 18/22 段；4 段未获
有效输入。未知字母/形状/词汇的真实控制性能尚未评估。

## runs 为什么保留

`real_robot/runs/push_letters/train_v1/` 保存训练原件、约 5 GB 的输入缓存、
API 日志、来源和训练回执，供复现/排查。它不进入 Git，也不进入下载包。
部署代码以原字节副本进入 `policies/`，没有移动或删除原始训练产物。
如果之后需要释放服务器空间，应单独清理可重建缓存，并保留权重及来源记录。
