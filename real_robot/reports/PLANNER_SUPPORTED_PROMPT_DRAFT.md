# 规划器支持下的学习问题设计草案

日期：2026-09-21。仅整理提示与接口建议，尚未调用 Runtime API、修改训练框架或启动
新实验。保留全部已执行的 cut、prompt、模型、原数据与日志。

## 本次用户修正

用户意图是让学习器决定接触位置及后续运动，前往该处的运动由已有 motion planner
解决；不是进一步训练独立下降、抬升和换位网络。用户希望 Runtime API 根据给定
能力和小数据目标自主发现精简表示，API 提示不预先指定 contact-point 输出或模型数。

此前 controller-aware 草案包含具体接触点/平面技能示例，保留为讨论沿革，已由
[planner-supported 草案](../prompts/planner_supported_learning_draft.md)取代。该新文件
只有可供 API 阅读的通用指令；本报告属于开发者讨论记录，不应拼接到 API 输入。

## 为什么只补一句 planner available 不够

已执行的 general_cut_and_prior_v3 开头要求一个覆盖所有示范行为的 subpolicy library，
其后还要求各组和过渡行为有训练支持。implementation_v3 又锁定三个 policy、两个
组和原有分工。现有公开 act/predict 接口返回每次调用的六维 v/w。内部低维表示虽已
允许，但这些要求仍容易使设计停留在学习整段机器人运动的范围。

新的定义应保持完整行为覆盖，并将覆盖责任放到 policy、planner、controller 和执行
逻辑组成的整个系统。数据记录可以作为学习目标、规划器负责的运动证据、历史上下文
或结果监督；不需要每条运动记录都对应一个 learned action label。

## 接入新设计轮时的协调修改

1. 保留上一轮的小数据、物体/交互相对结构、强 motion prior 和可复用条件化目标。
2. 在新的总设计 prompt 中替换“行为覆盖全部由 learned library 承担”的相关条款，
   将新草案与原有证据、来源、保留及预处理规则协调合并；不能仅拼接冲突的段落。
3. 首先提供实际 planner/controller 能力契约：接受什么几何目标、约束和局部运动，
   控制何种工具点，使用何种 frame/unit/time，支持何种规划域，返回何种反馈。
   用户声明这些能力存在，但此轮尚未检查远程电脑上的具体实现或测量其性能。
4. 重新开放学习任务、取段/事件样本、输出结构、决策频率和模型数。实现阶段继承
   新设计的分工，不继承 training_v3 的固定三个模型，也不强制逐帧六维命令回归。
5. 对候选设计要求“learned output → executor inputs → feedback → next decision”
   的可执行闭环与标签来源。planner 负责其声明规划域内的连接运动；在物体交互中
   还需要什么控制或建模能力，由 API 据能力契约明确分工。
6. 当前仓库接触候选模块和目标场表示可作为另行声明的延续基础，但不应把人类
   期望的接触动作字段、论文答案或本报告隐藏混入声称自主发现的 API 输入。

## 草案评价标准

检查 API 是否先区分已有执行能力与剩余学习问题，是否减少数据需覆盖的机器人路径
变化，是否共享同类决策的训练支持，是否给出可实现的输出及目标标签；模型数量和
具体字段由设计结果决定。期望表示的出现和真正的泛化能力分别记录，不能用前者代替
后者的评估。

## 仅供用户/开发者理解的参考

- [OpenAI Docs: Reasoning best practices](https://developers.openai.com/api/docs/guides/reasoning-best-practices)：明确能力、目标及成功条件，保持指令直接。
- [HACMan](https://proceedings.mlr.press/v229/zhou23a.html)：相关研究确实采用接触位置加接触后运动参数的时间抽象表示；其 RL 结果不等于当前小示范 BC 的可学性证据。

上述具体表示参考不纳入 planner-supported API 草案。
