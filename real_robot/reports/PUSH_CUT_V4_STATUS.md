# Push cut_v4 执行状态

核对日期：2026-09-21。**已完成：3 个子任务组、4 个 policy/prior 设计、
28 个片段。** 27 次 Astra/xhigh 调用，估算费用 USD 5.3269；93 个发布
文件与原始切片核验通过，执行进程正常退出。见
[中文成品审阅](PUSH_CUT_V4_REVIEW.md)、[API 原文报告](PUSH_CUT_V4.md)、
[验证](../runs/push_letters/cut_v4/validation.json)和
[独立统计](../runs/push_letters/cut_v4/review.json)。尚未实现或训练 policy。

用户已明确回复“允许本轮发送并继续”，解决启动前自动审批要求的数据
发送确认；授权和原拦截记录均保留如下。

## 本轮输入与范围

- [独立配置](../configs/push_cut_v4.json)：GPT-6 Astra/xhigh。
- [新版通用 prompt](../prompts/general_cut_and_prior_v2.md)、
  [Push 任务说明](../task_specifications/push_letters.md)、
  [原始数据契约](../DATA_CONTRACT.md)。
- 原始两条 Push 示范；不输入以前的科学设计或训练表现诊断。
- 输出目录：`real_robot/data/push_letters/cut_v4`；运行记录：
  `real_robot/runs/push_letters/cut_v4`。
- 本轮交付为子任务数据组、buffer/复用说明、独立 heuristic 设计、真机
  预处理义务和 agent 调用目录；后续实现、训练和真机执行不在本轮内。

## 已完成核验

[Preflight](../runs/push_letters/cut_v4/setup/preflight.json) 记录：
16,745 条原始配对记录、66,980 个媒体文件，约 8.31 GB；原始数据未改动。
12 项接口测试通过；零网络请求的实际 AgentLoop 请求构造检查确认
`model=gpt-6-astra`、`reasoning.effort=xhigh`。旧 cut_v3 的 14 个冻结
文件和 71 个发布文件哈希全部一致。

[独立执行入口](../execution/cut_v4.py) 保留原执行器的科学和验证逻辑，
仅更新导入路径、日期、默认配置及新报告路径，并冻结入口自身。
精确差异和来源保存在
[runner_derivation.diff](../runs/push_letters/cut_v4/setup/runner_derivation.diff)
及 [provenance](../runs/push_letters/cut_v4/setup/runner_provenance.json)。
完成报告将写入 `PUSH_CUT_V4.md`，不覆盖原 `PUSH_CUT.md`。

## 启动审批状态

[自动审批记录](../runs/push_letters/cut_v4/launch_approval_block.json) 保留
首次启动命令未执行的事实。审批拒绝理由：需要明确
授权本轮将选定的真机示范图像、机器人状态和相关元数据发送到
`https://api.openai.com/v1/responses`。旧 cut_v3 的发送授权限定于旧轮次。
用户已在后续确认中明确允许本轮发送并继续，完整问题与回复保存于
[发送授权](../runs/push_letters/cut_v4/destination_authorization.json)。获得
明确授权后的启动通过审批；原拦截记录保留，未发生传输失败自动重试。
