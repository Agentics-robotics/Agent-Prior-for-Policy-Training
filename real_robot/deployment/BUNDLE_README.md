# Push 本地推理部署包

这份完整包可独立使用，包含两个推理 checkpoint、原始 API 模型与预处理代码、
Python 调用器和锁定的依赖环境。机器人端由使用方适配。

在本目录执行（需要 Linux、NVIDIA 驱动、Pixi 及说明中的系统隔离依赖）：

```bash
pixi install --manifest-path real_robot/deployment/pixi.toml --locked
pixi run --manifest-path real_robot/deployment/pixi.toml --locked python -m real_robot.deployment check --gpu 0
```

[完整安装与调用说明](real_robot/deployment/README.md) ·
[API 输入/输出契约](real_robot/policies/push_v1/source/CALLING.md)

模型在 `real_robot/checkpoints/push_v1/contour_push.pt` 和 `visual_push.pt`。
不需要下载原始示范、训练缓存或 API key。此包不包含机器人驱动或网络服务。
