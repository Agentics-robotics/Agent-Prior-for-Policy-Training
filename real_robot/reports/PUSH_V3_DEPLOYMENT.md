# Push v3 portable deployment

核对日期：2026-09-21。三个模型的可迁移推理包已完成；实际最终压缩包在独立新目录解压、使用既有锁定推理环境通过检查。代码和文档已落地但未提交或推送 Git；完整包可独立下载。

[安装、下载及 agent 使用说明](../deployment_v3/README.md)。[API 策略目录](../policies/push_v3/source/POLICY_CATALOG.md)。[模型清单](../policies/push_v3/manifest.json)。

| Policy | 推理权重字节数 | 与原加载器最大动作误差 |
| --- | ---: | ---: |
| tool_waypoint_v1 | 8092111 | 0.0 |
| piece_contact_graph_v1 | 8666489 | 0.0 |
| piece_goal_field_v1 | 8092380 | 0.0 |

三个导出 checkpoint 的全部 state_dict 张量与训练最终 EMA 逐张量一致，保留原推理 spec；没有量化、重训、API 调用或修改科学代码。15 个 API 源码/文档文件逐字节复制。仅剔除原推理加载器同样剔除的三个顶层训练来源字段，未改模型输入或转换。

| 下载包 | 字节数 | SHA256 |
| --- | ---: | --- |
| push_v3_bundle.tar.gz | 22961077 | `17bf463973b66f1eff72d569908ac92fe9b221799b9bbdaba18f3db6aeb42063` |
| push_v3_weights.tar.gz | 22877331 | `ecde196aa82a01f97eb9c91a196fc18390fdbdb7efa5ddc1b626a5ef5086612b` |

解压后核验 42 个文件。没有 runs、原始数据读取器或训练观测；检查输入由外部 host 临时提供同一条已有配对观测。三个模型均返回有限六维候选，与原训练入口检查误差为 0，并通过过期输入、非法刚体目标、中断和禁用解码检查。

[可复现核验脚本](verify_deployment_v3.py)、[实际压缩包检查回执](PUSH_V3_DEPLOYMENT_CHECK.json)、[接口复用与原文件哈希](PUSH_V3_EXPORT_ORIGIN.json)。旧部署/框架/锁文件的 28 个来源文件和 cut_v3/v4/v5 的 71/93/85 个发布文件核验不变。

这是本机不同目录与独立推理环境的迁移检查，尚未实际登录另一台电脑，也没有完成目标电脑的驱动/实时性测试。支持范围为锁文件约定的 Linux x86_64 + NVIDIA GPU；加载器不再调用训练服务器的 Pixi 绝对路径或限定原物理 GPU 编号。

通过安装检查后，还需接收端供给真实同步观测、相机/工作平面标定和空间目标。当前 API decode_action 始终禁用；v/w 与实际控制器的接入由用户端完成，本包无机器人通信、硬件动作或任务成功率证据。
