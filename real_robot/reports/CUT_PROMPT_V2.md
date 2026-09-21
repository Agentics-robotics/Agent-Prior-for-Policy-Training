# 子任务优先的 cut / prior prompt

设计核对日期：2026-09-21。以下保留执行前的 prompt 修订范围。
后续用户已授权执行，cut_v4 已完成，见[中文成品](PUSH_CUT_V4_REVIEW.md)。

用户本轮要求以 long-horizon 技能库为目标：先确定需要哪些子任务，使
subpolicy 完整覆盖示范中的各种行为；再自由选取、重组和复用 cut，形成
分别对应各子任务的训练数据组；最后为各组设计帮助泛化的 heuristic。

新版为 [general_cut_and_prior_v2.md](../prompts/general_cut_and_prior_v2.md)。
它继续配合独立的 [Push task specification](../task_specifications/push_letters.md)
和 [数据契约](../DATA_CONTRACT.md)，没有预设 Push 的技能名称、顺序或数量。

## 主要要求

- 行为覆盖：从示范行为映射到子任务、训练支持和可调用 policy，包括实际
  出现的过渡、调整、重复尝试等支持行为。
- 数据组织：每个 sub-dataset 对应一个子任务；可跨轨迹自由取片段，使用
  不规则边界、重叠和大量复用；每次使用保留来源、条件和监督含义。
- Buffer：保留有助于进入、完成和交接的前后片段；符合子任务和 prior 的
  动作用于监督，其余有用历史可作为上下文。设计 prior 后复核兼容性。
- 泛化：同一子任务组可以有多个不同 heuristic，各自独立训练为可调用
  policy；由 API 决定具体机制和数量。
- 真机输入：保留衍生/privileged input 的完整预处理、标定、权重和依赖
  说明；区分部署可计算输入和训练专用信息，要求后续提交可执行转换。
- 部署：明确参数、进入/终止条件、进度反馈、记忆和交接信息，并以实际
  示范支持的调用序列说明高层 agent 如何组合技能。
- 覆盖核验：后续预处理须逐子任务、逐行为统计有效监督和排除，避免仅凭
  原始记录保留量认定全部行为已进入训练。

## 范围和兼容性

此次修改只增加通用 prompt 新版并更新导航和项目指令。它沿用现有工具的
`skills`、`segments`、`heuristics`、coverage 和 handoff 字段；未修改工具
schema、runner、任务文件或数据契约，未新增实验配置或启动 API/训练。

已执行的 `general_cut_and_prior.md`、`push_cut_v3.json`、cut_v3 及其冻结
快照保持原样。现有 train_v1、training_v2、部署产物和 Exp1/Exp2 均不属于
本次编辑范围。后续新 cut 应使用独立配置和输出目录引用新版 prompt，
并按项目规则核验实际请求为 `gpt-6-astra/xhigh`。

核验：新增文档的本地链接及空白格式检查通过；cut_v3 冻结清单中的全部
14 个文件与已存 SHA-256 一致。此次为文档修改，未运行实验或训练测试。
