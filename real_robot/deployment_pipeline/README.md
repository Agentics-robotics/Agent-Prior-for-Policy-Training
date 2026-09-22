# Push v5 / Flip egg v1 跨电脑调用

这是2026-09-22完成的两个新模型的独立推理入口。科学源码与调用文档逐字节保留
API提交；导出训练选中的权重，不重新训练。Push选中第600步，Flip选中第2000步。
目录和GPU编号由接收电脑决定，不依赖训练服务器的Pixi绝对路径或GPU4配置。

## Git同步后下载什么

代码应包括 `real_robot/deployment_pipeline/`、`real_robot/policies/push_v5/`、
`real_robot/policies/flip_egg_v1/`、`real_robot/training_v4_environment/`，以及
已有共享 `deployment/{__init__,common,policy}.py`、
`training_pipeline/{__init__,public,security}.py` 和 `src/appl/{__init__,io,gpu,kernel}.py`。
本次没有替你执行commit/push；这些新文件需要先提交再同步到接收电脑。

两个学习模型的权重可以用Drive、SCP或其他文件传输方式下载，文件内容相同：

| 文件 | 放在接收电脑仓库中的位置 | 大小 |
| --- | --- | ---: |
| shape_push_v1.pt | real_robot/checkpoints/push_v5/shape_push_v1.pt | 47,909 bytes |
| egg_interaction_v1.pt | real_robot/checkpoints/flip_egg_v1/egg_interaction_v1.pt | 1,130,869 bytes |

也可下载 `real_robot/exports/pipeline_v1/pipeline_v1_weights.tar.gz`，在仓库根目录
解压，自动得到上述路径。**不要用旧v3/v4权重或直接把训练last.pt改名替代。**
导出文件只保留选中的推理参数与在线配置，加载器会校验其SHA-256。

Push的 `act` 接收上游提供的几何，不强制依赖SAM权重。若使用包里的 `inventory`
从RGB生成分割提案，还需下载 `pipeline_v1_sam_assets.tar.gz`，在仓库根目录
解压，得到 `real_robot/checkpoints/push_v5/assets/sam2.1_hiera_small.pt`，
184,416,285 bytes，约176MiB。资产来源/许可证回执保存在Push的 `asset_receipts/`。
Flip不需要额外感知权重。没有Qwen下载要求。

本次只生成本地文件，没有上传到Drive或生成Drive链接。无需下载原始示范、
训练缓存、优化器状态或API日志；单独推理也不需要OpenAI API key。

## 接收电脑安装与检查

使用Linux x86_64、NVIDIA GPU及兼容CUDA驱动，安装Pixi、`nvidia-smi`、
`bubblewrap`、`libseccomp.so.2`，并具备Landlock ABI >= 3、可用用户namespace
和OpenCV需要的GL/GLib系统库。启动要求指定GPU至少2GiB空闲显存。复用本轮
锁定环境；无需修改旧v3环境。Windows/macOS/CPU运行未在此包实现或验证。

在接收电脑仓库根目录执行（`pixi`指接收电脑自己的可执行文件）：

```bash
tar -xzf pipeline_v1_weights.tar.gz
# 只有使用Push inventory时需要：
tar -xzf pipeline_v1_sam_assets.tar.gz
pixi install --manifest-path real_robot/training_v4_environment/pixi.toml --locked
pixi run --manifest-path real_robot/training_v4_environment/pixi.toml --locked python -m real_robot.deployment_pipeline check --gpu 0
```

第一次安装会下载锁定的软件依赖；已有环境也应按此lock核对。
`check --files-only`只验源码/权重；`check --require-assets`额外要求SAM文件；
普通 `check` 会分别加载两个模型，无机器人动作。文件损坏或缺失会明确报错，
不会自动换模型或下载感知后端。

完整独立包 `pipeline_v1_bundle.tar.gz` 也包含代码、两个权重及SAM资产，
不依赖Git同步状态。解压进入 `pipeline_v1_bundle/` 后执行相同安装/检查命令。
下载归档旁的 `SHA256SUMS` 后可执行 `sha256sum --ignore-missing -c SHA256SUMS`。

## Agent调用文件和接入

| 模型 | Agent入口 | observation → decision |
| --- | --- | --- |
| push_v5 | [调用契约](../policies/push_v5/source/CALLING.md)、[handoff](../policies/push_v5/source/HANDOFF.json) | 可见物体几何、当前末端状态、SE(2)目标 → 接触后的二维末端位移；首次接触为API几何规则 |
| flip_egg_v1 | [调用契约](../policies/flip_egg_v1/source/CALLING.md)、[handoff](../policies/flip_egg_v1/source/HANDOFF.json) | 双视角RGB、机器人状态和因果历史、初始场景 → 末端局部六维速度及预测分布信息 |

```python
from real_robot.deployment_pipeline import Policy

with Policy("push_v5", gpu=0) as policy:
    # observation、call、executor_contract按对应CALLING.md由接收端提供。
    result = policy.act(observation, call, reset=True,
                        executor_contract=executor_contract)
    # result["decision"]为API定义的预测；executor_request为执行目标描述。
    # 后续调用保留memory，切换目标/物体或中断时遵循HANDOFF.json。

# Flip使用 Policy("flip_egg_v1", gpu=0)，但输入和输出契约不同。
```

模型接口不直接连接机器人。接收端仍需提供当前标定、观测、任务目标与对应的
planner/controller绑定，并解释返回的executor_request；不能把它直接当关节命令。
Push首次接触还需要完整几何与接触确认；SAM只给未验证分割提案。Flip记录的
EE坐标系/工具握持与历史输入必须与调用契约一致。无需从示范终点补部署输入。

训练完成与迁移检查不等于真机任务成功：Push仅74训练例，开发位移误差5.095mm，
高于目标平移基准4.764mm；Flip使用末端运动模仿，没有显式铲—蛋几何模块。
本次迁移保持上述模型与科学结果不变。每个模型目录保留TRAINING_RECEIPT.json
和独立API_USAGE.json。
