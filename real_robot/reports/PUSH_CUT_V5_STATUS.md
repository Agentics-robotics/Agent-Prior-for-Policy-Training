# Push cut_v5 执行记录

日期：2026-09-21。用户批准试验小数据强 prior 的新版 prompt，并将编号
顺延为 cut_v5。**已完成：2 个数据组、3 个 policy/prior 设计、26 个
片段。** 29 次请求和返回均核验为 Astra/xhigh，估算 USD 5.48676；
85 个发布文件及原始切片验证通过。执行进程正常退出，未实现或训练
policy。见[中文成品与对照](PUSH_CUT_V5_REVIEW.md)、[API 原文](PUSH_CUT_V5.md)、
[完成回执](../runs/push_letters/cut_v5/completion_receipt.json)。

## 输入与范围

- [新版 prompt](../prompts/general_cut_and_prior_v3.md)：在确定子任务前，
  强调可复用的物体/交互相对结构、运动约束和需要学习的最少动作变量。
  [修订说明](CUT_PROMPT_V3.md)保留精确差异。
- [独立配置](../configs/push_cut_v5.json)，GPT-6 Astra/xhigh。
- 原始两条 Push 示范；与 cut_v4 相同的任务说明、数据契约和证据工具。
- 输出到 `real_robot/data/push_letters/cut_v5`；调用、费用和验证记录在
  `real_robot/runs/push_letters/cut_v5`。
- 本轮交付子任务数据组、prior 设计、预处理义务和 agent 调用文档。

## 验证与来源

[准备脚本](../execution/prepare_cut_v5.py)核验旧 cut_v3/v4 哈希、原始
数据与实际 AgentLoop 请求中的 model/effort，检查阶段没有网络调用。
[执行器](../execution/cut_v5.py)由 cut_v4 执行器派生，仅更改默认配置
和报告路径，保留验证、传输、原始数据切片和费用记录逻辑。

12 项接口测试通过；实际 AgentLoop 请求构造核对为 Astra/xhigh。
原始 16,745 行、66,980 个媒体文件核验通过；旧 cut_v3/v4 的冻结与
发布文件均保留原哈希。见[preflight](../runs/push_letters/cut_v5/setup/preflight.json)。

## 启动审批

启动时按本轮执行指令和已建立的会话授权申请执行。自动审批系统
拒绝，认为旧明确发送授权限定 cut_v4，本次概括性运行指令不足以
授权 cut_v5 的具体数据发送。原拒绝理由保存在
[审批记录](../runs/push_letters/cut_v5/launch_approval_block.json)。

已向用户明确询问是否允许 cut_v5 将选定真机 Push 图像、状态和元数据
发送到官方 Responses 及其 input_tokens 接口，使用 Astra/xhigh。
用户回复：“允许，为什么最近总问我，之前也不问我啊？之后都允许”。
[本轮确认](../runs/push_letters/cut_v5/destination_authorization.json)与
[持续授权](../authorizations/openai_data_processing_20260921.json)已保存。
此前的拒绝及原推定授权文件均保留，同类工作流后续不主动逐轮询问。
