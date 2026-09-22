# 通用 prompt 与训练流程修正

核对日期：2026-09-22。

新版通用 cut prompt 保存为 `prompts/general_cut_and_prior_v6.md`，增加
“部署需要哪些变化 → 示范提供多少独立经验 → 交互需要什么状态和运动 →
Agent/planner 已承担什么 → 模型需要哪些输入输出”的设计要求。prior 可以
决定输入表示与输出形式，保留 `contact point movement` 通用例子。
不指定 Push 必须一个模型、二维输出，也不指定 Flip 的表示或 skill 数量。
复杂度应在保留交互所需几何、姿态和时间信息的前提下减少。

Push 新 specific v3 明确仅有两条示范；Flip 新 specific v2 增加用户说明的
铲起/翻转任务重点。这两个新版 specific 与新版通用 cut prompt 均未用于新
cut。用户随后明确要求先训练已有 Push cut_v6、Flip egg cut_v1，因此正式
训练输入承接这两次已有 API 设计及原任务事实。

旧 Push/Flip 的 prompt、specific、数据契约、配置、当时保存的 prompt.json
与 plan.json 已逐字节复制到
`prompts/archive/20260922_before_representation_revision/`，manifest.json 记录
SHA-256；旧路径仍保持原内容。旧 training_v4 框架、配置、实现 prompt 和
checkpoint 同时保留。新框架为独立 `training_pipeline/`。

cut 本身没有写死 47 点或 policy 数量。47 是 Push cut API 的选择；旧训练
实现 prompt/validator 将其冻结成了不可变的完整训练样本集合。这是本轮
修正的边界错误。cut 的稀疏调用时刻与训练数据密度也需要区分：部署可以
低频调用一个输出几何目标的模型，训练仍可能从多个有效状态/目标构造样本。
新 prompt 要求明确这个区别，后续实现可在保留原 cut 的情况下给出有依据的
衍生数据计划。

新通用流程固定调用/检查/训练/归档步骤，API 决定科学内容。移除了固定
policy ID、固定47点、强制完整标签、固定AdamW/采样/EMA等限制；新增全源
记录用途统计、独立锚点与衍生样本分开计数，以及 API 对实际预处理 hash 的
就绪评估。该评估不保证数据充分，作用是把真实覆盖问题交回设计 API，而
不是把“有一个可训练样本”当成科学成功。

输入必须因果、原始来源/边界真实、未来信息用途明确等约束仍保留。模型
使用可重复的隔离 PyTorch 单优化器接口；策略内部可以包含辅助学习模块。
不声称接口检查证明真机或泛化表现。

验证：11 项通用接口/调度测试通过，包含真正的隔离 CPU 合成预处理、
两个模型训练及在线调用；两个配置离线捕获的实际请求均为 Astra/xhigh，
没有网络请求。见 [检查回执](GENERIC_TRAINING_PIPELINE_CHECK.json)。正式
训练状态与结果分别记录在各自 runs 和训练报告，不能由本框架验证替代。

GPU 4 上另完成 1 项隔离端到端测试，两个开发者合成线性模型各训练2步，
重载及真实在线进程调用通过（33.74秒）。这些合成模型不属于科学候选。
用户明确授权后已启动 Push training_v5 和 Flip egg training_v1，均使用
同一个实现 prompt，承接旧 cut，不调用新版 cut prompt。实际完成状态以
各运行 workflow.json、library.json 及后续训练报告为准。

Push 首次实现经过实际检查后声明接触伪标签存在质量问题，未执行训练。
后续新增独立、通用的 `execution/continue_training_pipeline.py`，把完整诊断
交回 API，要求落实现有数据和能力允许的恢复，区分离线训练依赖与真机接入
条件。恢复方式、表示变化及最终是否就绪仍由 API 决定；旧提交、失败缓存与
费用完整保留。该反馈没有指定 Push 的目标、算法或样本名单。

未来新运行的统一入口为 `execution/train_policies.py`，组合初次实现与有界
恢复，并保存驱动源码快照；不对本轮目录重新执行。4 项控制器测试通过，
覆盖就绪后训练、再次阻塞、传输错误不重放、修订次数上限与已有运行保护，
测试不调用 API。此前已启动的冻结框架和 prompt 未因这次扩展而改变。

最终两任务训练均完成，详见[Push结果](PUSH_TRAINING_V5.md)与
[Flip结果](FLIP_EGG_TRAINING_V1.md)。两者各有一个可调用网络，但均包含API
作出的表示调整：Push改学接触后的二维末端位移、首次接触用几何规则；Flip
改用联合RGB编码和已记录的末端局部运动，未实现显式铲—蛋几何。旧cut保留，
衍生实现的变化在API提交内记录。完成训练不表示实现了原cut的全部表示设想，
也不表示泛化表现已验证；Push开发误差尚未优于简单目标平移基准。
