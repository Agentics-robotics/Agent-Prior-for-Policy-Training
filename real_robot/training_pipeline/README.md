# 通用 API 实现与训练流程

2026-09-22。此目录为开发者编写的执行框架；模型、预处理、训练方案和
Agent 调用文档由实际 Runtime API 编写。沿用独立 training_v4_environment
锁定环境，旧 policy_training_v4 及历史科学结果保留。

本轮两个任务均已完成：[Push training_v5](../reports/PUSH_TRAINING_V5.md)、
[Flip egg training_v1](../reports/FLIP_EGG_TRAINING_V1.md)。各报告链接到 API 原始
调用文件、checkpoint library、覆盖检查和独立进程验证。Usage 分别见
[Push](../reports/PUSH_TRAINING_V5_API_USAGE.md)、[Flip](../reports/FLIP_EGG_TRAINING_V1_API_USAGE.md)。
已有目录不再运行；以下命令说明统一入口，用于新配置与新输出路径。

同一个 [实现 prompt](../prompts/implement_and_train.md) 读取不同任务事实、
已有 API cut/prior 和原始记录。训练阶段可以依据真实预处理结果更新衍生数据
计划，不受旧 cut 的决策点名单限制。policy 名称、数量、表示、监督掩码、
采样、优化器、学习率、EMA、停止与 checkpoint 选择均由 API 提交。

固定流程：API 检查数据与写代码 → 隔离预处理/零更新接口检查 → API 查看
覆盖与质量并修订 → API 对精确产物声明训练就绪 → 训练 → 重载与实际调用
检查 → 输出 library.json。实施错误携带诊断进入新的 API 修订，保留旧尝试；
传输错误不自动重试。科学数据不足时保留 API 给出的具体阻塞理由。

```bash
/home/users/oscar/.pixi/bin/pixi run --manifest-path real_robot/training_v4_environment/pixi.toml --locked --no-install python -m real_robot.execution.train_policies --config real_robot/configs/push_training_v5.json
/home/users/oscar/.pixi/bin/pixi run --manifest-path real_robot/training_v4_environment/pixi.toml --locked --no-install python -m real_robot.execution.train_policies --config real_robot/configs/flip_egg_training_v1.json
```

`preflight` 只检查本地配置/数据身份，`status` 查询进度；不要对已有运行重新
执行 `run`。后台 API 调用与训练继续时应查看运行日志/状态。所有新请求为
gpt-6-astra/xhigh。配置的费用、次数、时间和计算上限是执行保护范围，非
推荐模型大小或固定训练方案。GPU 4 的本流程作业采用文件锁排队，每次启动
前记录实际 GPU 占用，保留其他进程。

上面的统一入口适用于尚未启动的新运行配置，不能重跑本轮已有目录。它在
`training_pipeline` 的固定执行阶段外增加有界的数据/设计恢复：API首次声明
标签/表示问题时，将其诊断交回新的API修订，要求落实能用现有能力完成的
修复，区分训练依赖与真机接入条件。每次修订都保留旧包与费用，仍由API判断
修复方法及是否就绪；最终外部数据缺口不会被强行改成训练通过。此通用扩展
由本轮Push的实际阻塞触发；运行中的旧框架/请求保持冻结。

本轮最初通过底层入口启动；Push后续用同一通用
`execution.continue_training_pipeline` 创建了记录完整的设计修订。
新统一入口把这些固定步骤组合为一条命令，并留存驱动源码快照。

完成后每个包的 source/ 包含 API 的 CALLING.md、POLICY_CATALOG.md、
HANDOFF.json 和逐模型使用文件。调用器为
`real_robot.training_pipeline.inference.PolicyProcess`，输出由 API schema
定义，最终由部署侧连接真实 planner/controller；训练流程不执行机器人。

适用范围是本仓库的配对真机记录格式和 [PyTorch 接口](INTERFACE.md)，
可更换任务，不声称支持任意数据格式或任意训练引擎。当前命令承接已发布
cut，不运行新 cut。新版 [cut prompt](../prompts/general_cut_and_prior_v6.md)
仅保存待用，当前 Push cut_v6 和 Flip cut_v1 均源于旧 prompt。

验证包括带独立标识的开发者合成数据/模型测试，检查新采样、目标变体、
部分标签、多模型、API 选择 SGD/早停、隔离训练、重载和真实在线调用。
这些合成结果不属于正式科学模型或 Runtime API 产物。
