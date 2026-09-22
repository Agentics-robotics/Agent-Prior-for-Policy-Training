# Push v6 跨电脑调用

这是 Push training_v6 的独立推理入口。旧 `deployment_pipeline`、Push v5 和
Flip egg v1 导出不变。这里逐字节保留本轮 API 提交的科学源码，并无损导出其
选中权重；模型架构、数据处理和输出未由部署层重写。

## 下载与安装

完整包位于 `real_robot/exports/pipeline_v2/pipeline_v2_bundle.tar.gz`。
它包含代码、锁定环境文件、策略权重和冻结 SAM 资产；不包含示范、训练缓存、
优化器状态或 API 凭据。解压后进入 `pipeline_v2_bundle/` 即可安装和检查。

如果先通过 Git 同步代码，则下载下面两个包，在仓库根目录解压：

| 下载包 | 得到的文件 |
| --- | --- |
| `pipeline_v2_weights.tar.gz` | `real_robot/checkpoints/push_v6/shape_push_v1.pt` |
| `pipeline_v2_sam_assets.tar.gz` | `real_robot/checkpoints/push_v6/assets/sam2.1_hiera_small.pt` |

**本轮两个文件都需要。** API 的 `build_model` 会初始化冻结 SAM，即使只调用
几何输入分支，也会读取这个资产。SAM 没有参与梯度训练；策略权重文件也保留
原模型 state dict 中的冻结 SAM 张量，以保证与训练选中状态完全一致。
因此文件大小不能直接代表学习网络的参数量。不要替换成 v5 权重或原始 `last.pt`。

代码需要包括本目录、`real_robot/policies/push_v6/`、
`real_robot/training_v4_environment/`、共享 `deployment/{__init__,common,policy}.py`、
`training_pipeline/{__init__,public,security}.py` 和 `src/appl/{__init__,io,gpu,kernel}.py`。
完整包已按清单包含所需文件。下载归档旁的 `SHA256SUMS` 后可核对：

```bash
sha256sum --ignore-missing -c SHA256SUMS
tar -xzf pipeline_v2_weights.tar.gz
tar -xzf pipeline_v2_sam_assets.tar.gz
pixi install --manifest-path real_robot/training_v4_environment/pixi.toml --locked
pixi run --manifest-path real_robot/training_v4_environment/pixi.toml --locked python -m real_robot.deployment_pipeline_v2 check --gpu 0
```

上述 `pixi` 是接收电脑的可执行文件，`--gpu 0` 可改为当地可用 GPU。
需要 Linux x86_64、NVIDIA GPU/兼容驱动、`nvidia-smi`、bubblewrap、
`libseccomp.so.2`、Landlock ABI >= 3、可用用户 namespace 和 OpenCV 系统库；
启动检查要求指定 GPU 至少 2 GiB 空闲显存。Windows/macOS/CPU 运行未实现。
`check --files-only` 核对文件；普通 `check` 还实际加载模型，不发机器人命令。
推理无需 OpenAI API key，也不会自动下载替代模型。

本工作只生成本地文件；Git 提交/推送、Drive 上传需另行执行。

## 先运行一次真实权重示例

```bash
pixi run --manifest-path real_robot/training_v4_environment/pixi.toml --locked python -m real_robot.deployment_pipeline_v2 example --bundle push_v6 --gpu 0
```

这会读取 `policies/push_v6/examples/interface.json`，实际加载训练选中的权重，
输出 `decision` 的接触点/方向/距离，以及 `executor_request` 的准备位、接触位和
结束位。示例把从记录提取的单个物体轮廓放入合成场景，使用事后目标和
**合成标定/确认标志**，用于看懂
字段、验证调用链；不能拿这些示例坐标直接控制真机。之后按下方契约替换成实时
观测、真实目标和接收端标定，调用入口不变。
示例还记录本次加载和一次几何输入调用的耗时；这包含进程通信，但不包含原始
RGB感知或物理执行，不是延迟分布或实时性保证。

## 给调用 Agent 的文件

首先读取 [CALLING.md](../policies/push_v6/source/CALLING.md)、
[完整性检查补充契约](../policies/push_v6/source/FINAL_QUALITY.md)、
[HANDOFF.json](../policies/push_v6/source/HANDOFF.json) 和
[POLICY_CATALOG.md](../policies/push_v6/source/POLICY_CATALOG.md)。这些是 API 的
正式调用契约，包含状态、前置条件和上层任务分工。

```python
from real_robot.deployment_pipeline_v2 import Policy

with Policy("push_v6", gpu=0, timeout_s=180) as policy:
    result = policy.act(observation, call, reset=True,
                        executor_contract=executor_contract)
    # 检查 result["status"]，再解释 decision / executor_request。
    # 后续同一目标调用保留 memory；切换和中断遵循 HANDOFF.json。
```

接收端提供两种观测之一：

1. **已关联的几何 Scene**：每个物体有持久 ID、96×2 的 base XY 边界点、
   96×2 outward normals、96 个轮廓环编号，另有当前工作区、工具半径、接触高度、
   标定标识、观测时效与不确定度。内孔需保留轮廓；遮挡时可提供过去观测得到的
   `causal_shape_references`，不能用未来帧或预存的字母模板替代。
   确认场景几何完整后才设置 `geometry_complete_validated=True`；未确认时返回
   `needs_view`。调用包不会替接收端完成这个确认，不能为了通过接口直接置真。
2. **当前 RGB 与标定**：`act` 可用冻结 SAM 给出候选轮廓，返回
   `associate_instances`。Agent 需完成实例关联、排除背景/机器人提案并确认几何，
   再以已关联 Scene 调用。此包没有单独的 `inventory` 科学 hook，不能调用该操作。

`call` 指定 actor 的 `instance_id`、`scene_revision`、`reference_id`、
`goal_delta=[dx,dy,dyaw]` 和最大短推距离。位置单位米、角度弧度：绕当前轮廓
质心旋转 dyaw，再沿 base XY 平移 dx/dy。词语布局和目标位姿由上层 Agent 决定。

`proposed` 时，`decision` 给出物体边界上的 `contact_point_m`、
`direction_unit`、`stroke_length_m`，以及 outward normal、内外轮廓编号和时效信息。
**接触点不是 TCP 位姿。** 绑定好的 `executor_request` 用工具半径/标定变换得到
standoff、contact、end 三个法兰目标；单次 push 的高度和姿态保持一致。
approach、下降、撤离和路径避障由接收端 planner/controller 执行。

`executor_request` 始终是 `execute=False` 的目标描述，不直接控制机器人。
缺少 commissioning/标定/guard 参数时会返回 `calibration_required`。
需要接收端绑定：相机到 base 的标定、桌面/物体顶面、工具半径与接触高度、
`T_flange_tool`、`R_base_tool`、全工具碰撞模型、guard profile 及执行反馈。
文档和离线测试里的合成参数不能当成你的机器人标定。

## 真机调用顺序

新鲜观测和关联 → Agent 给当前物体目标 → policy 选接触/短推 → planner 到准备位 →
重新观测并核查接触 → 一次有限短推 → 观测物体和邻居的实际运动 → Agent 决定后续。
抬起/换侧由 planner 执行，不是该网络的预测维度。
调用者应设全局尝试次数限制；当前模型 memory 的目标 key 随 `goal_delta` 改变，
不能仅依靠内部 nonprogress 计数管理不断更新的目标。

迁移核验见 [回执](../reports/PUSH_V6_DEPLOYMENT_CHECK.json)，训练与数据覆盖见
[训练报告](../reports/PUSH_TRAINING_V6.md)。核验衡量加载、输出一致性与调用契约；
真实物理成功率、新字母泛化和标定精度需要在接收端测量。
