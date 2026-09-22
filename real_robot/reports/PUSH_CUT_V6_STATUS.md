# Push cut_v6 执行与来源记录

复核日期：2026-09-21。本文件记录开发者接口与执行过程；科学切分、prior、
模型输出和策略数量由 Runtime API 提交。完成状态以独立验证回执为准。

已完成：24 次 Astra/xhigh 调用，一次 write_plan、一次 check_plan、一次
submit_datasets；一个 learned policy、一个 supplied executor 组、14 段、
47 个候选决策锚点。51 个发布文件、所有原始切片及来源核验通过。参见
[中文成品](PUSH_CUT_V6_REVIEW.md)和
[最终完成回执](../runs/push_letters/cut_v6_with_example/completion_receipt.json)。

## 本轮实际条件

- [通用 prompt v5](../prompts/general_cut_and_prior_v5.md)保留小数据、
  物体/交互相对表示和强 prior 要求，先分配 planner/controller 与学习器责任，
  再确定残余学习问题、输出、决策时机与数据分组。
- [Push specific v2](../task_specifications/push_letters_v2.md)保留未知字母、
  未知形状和新单词泛化要求；加入用户声明的 planner/controller 可用能力。
  远端 planner、实际 controller binding、场景重建及标定未在本轮接入验证。
- 用户在初始运行开始后明确要求顺带举 contact point 例子。实际 prompt 增加的
  唯一示例短语是 `Spatial goals (e.g., a contact point)`，没有固定输出格式
  或 policy 数量。此运行属于带示例的设计引导，不能描述成完全未提示
  contact point 的自主发现。
- 原“积极考虑多个 alternative prior”的倾向已取消。数据 schema 同时允许
  learned_policy 和 supplied_executor；后者无 heuristic、无 learned policy。
  sparse 决策有显式索引，executor_only 的监督区间为 null。

## 执行范围与验证

实际入口是 `real_robot.execution.cut_v6`，显式传入
[push_cut_v6_with_example.json](../configs/push_cut_v6_with_example.json)。
会话目录为 `runs/push_letters/cut_v6_with_example`，成品目录为
`data/push_letters/cut_v6`。通用 prompt、specific、数据契约、配置和源码均冻结。

[启动前核验](../runs/push_letters/cut_v6_with_example/setup/preflight.json)
确认实际 AgentLoop 请求构造为 gpt-6-astra/xhigh；
[发送后核验](../runs/push_letters/cut_v6_with_example/setup/initial_wire_audit.json)
确认实际请求中只有一个该括号例子，并核对已返回的模型身份与档位。
21 项[接口测试](../runs/push_letters/cut_v6_with_example/setup/interface_tests.json)
通过，覆盖稀疏决策、执行器组、原始切片、证据、配对和传输约束。
这些检查不证明 learned policy 的任务性能。

16,745 个原始配对记录与 66,980 个原始媒体文件完成来源核验。
cut_v3/v4/v5 冻结接口和发布文件通过 hash 校验；
[training_v3 保留核验](../runs/push_letters/cut_v6_with_example/setup/training_v3_preserved.json)
确认三模型原始/部署权重及 API 科学源码不变。
本轮只做切分、prior、调用、目标推导和预处理/执行器适配设计，无训练或真机操作。

## 被用户修订中断的初始尝试

未含 contact-point 例子的初始会话原样保留于 `runs/push_letters/cut_v6`。
收到用户追加例子的指示后，精确中断该 Python 进程，退出码 130。
已完成 9 次请求，另有 1 次发送后未收到返回；未提交切分。
参见[中断回执](../runs/push_letters/cut_v6/interruption_receipt.json)。

初始尝试已知估算 USD 0.5360945，未决调用保留 USD 2.0495625 的执行预留；
预留不是实际账单，不能假定该调用零费用。原请求、已收到响应、状态和账本
均保留。新尝试另建会话，没有自动重放未决调用，也没有把初始设计输出传入。
最终报告应同时保留正式尝试和初始中断尝试的费用范围。

持续发送授权来自用户“允许，为什么最近总问我，之前也不问我啊？之后都允许”，
见[授权原文](../authorizations/openai_data_processing_20260921.json)。
本轮沿用该范围，没有再次向用户索取同类数据发送许可。
