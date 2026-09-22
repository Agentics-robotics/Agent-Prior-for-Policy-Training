# 项目进度

## 当前摘要

**Push v6全新重跑、训练与迁移完成核对：2026-09-22。** 沿用旧cut_v6和原v5通用流程，仅新增尽量使用有效数据完成最终拟合的要求；未传入v5结果/源码/划分审查或取消轮修正。API最终选择一个接触点/方向/短行程候选评分器，18,497个可训练参数，冻结SAM；GPU4完成3,000步并选择最终权重。最终130例（A43/B87，9/12学习片段、115完整目标/15仅接触、11内孔）全部实际进入梯度，独立采样审计与checkpoint随机数状态一致。训练集接触标签差异16.403mm、行程差异2.101mm，尚不能称为精确模仿或物理有效；无独立泛化/真机成功率。[训练报告](real_robot/reports/PUSH_TRAINING_V6.md)、[完整逻辑](real_robot/reports/PUSH_V6_SYSTEM_WALKTHROUGH.md)、[梯度审计](real_robot/runs/push_letters/training_v6/gradient_coverage_audit.json)。92次Astra/xhigh API估算$19.6388715，无未知usage；八次预处理检查累计81.33分钟，正式优化约22.35秒，见[usage](real_robot/reports/PUSH_TRAINING_V6_API_USAGE.md)与[耗时核查](real_robot/reports/PUSH_TRAINING_V6_PREPROCESSING_COST.md)。重载误差0、9项API调用及独立worker检查通过；新[deployment_pipeline_v2](real_robot/deployment_pipeline_v2/README.md)完整包约327MiB，实际解压47文件核对、跨目录真实权重调用/CLI示例输出一致。[迁移报告](real_robot/reports/PUSH_V6_DEPLOYMENT.md)、[回执](real_robot/reports/PUSH_V6_DEPLOYMENT_CHECK.json)。接收端仍需绑定实时关联/完整场景、标定与planner/controller；未Git提交/推送或Drive上传。旧v5、Flip、cut/prompt/导出包核验保留，无新cut或额外性能修正训练。已取消的额外轮次仅留[删除和费用记录](real_robot/deletions/push_v5_feedback_continuation_20260922/deletion_receipt.json)：301文件删除，25次API/$4.699118，与本轮分账。

**Push v5训练划分追加审查：2026-09-22。** A/B整录像划分为API选择，非通用框架强制比例；最终168个有效例中仅A的74例/6组用于梯度，B的94例/6组用于开发及checkpoint选择。此前“覆盖12/12段”指准备数据，不是训练覆盖。原cut建议A→B再交换的诊断，当前仅实现A→B，没有A+B最终拟合；B有效数据未因质量被禁止训练。对于少示范部署目标，缺少开发诊断与最终拟合的区分会浪费可用于学习的经验，后续通用流程应明确此区别。见[审查与证据](real_robot/reports/PUSH_TRAINING_V5_SPLIT_REVIEW.md)。本次只审阅与澄清，已完成权重/导出、prompt和API科学源码保留，无新增API或训练。

**Real robot 两任务迁移包完成核对：2026-09-22。** 新 `deployment_pipeline` 将Push v5和Flip v1的API科学源码/调用说明逐字节导出到Git可管理的policies目录，移除训练调用器对本机路径/GPU4的依赖。两模型选中权重无损导出为47,909/1,130,869字节，另有Push可选SAM资产。已生成两权重包、SAM包、完整独立包与SHA-256清单；实际完整包在新目录通过62文件核对、双模型调用一致性检查，输出误差0，中断/执行器拒绝状态正确；SAM实际返回9个未验证提案。见[安装及下载说明](real_robot/deployment_pipeline/README.md)、[迁移报告](real_robot/reports/PIPELINE_DEPLOYMENT.md)与[回执](real_robot/reports/PIPELINE_DEPLOYMENT_CHECK.json)。Git提交/推送与Drive上传未执行；无新API、训练或机器人动作，原科学结果与旧部署版本保留。

**Real robot 通用流程修正与两任务训练完成：2026-09-22。** 新通用 `training_pipeline` 由 API 决定采样、表示、policy、优化与训练，修正固定47点等限制；15项接口/恢复控制器检查及GPU隔离合成训练通过。**Flip egg training_v1**承接旧cut_v1，API将原拟议铲具/语义几何改为联合视觉编码器＋末端局部运动：278,903参数，2,000步；准备10,931例，实际训练11条示范/5,364例。**Push training_v5**承接旧cut_v6，第一版142条接触弱标签因质量问题未训练；API随后自行修订为二维末端续推动作，最终168例覆盖原12个交互片段，A训练74/B开发94；9,641参数、800步、选中600步。B开发误差5.095mm，高于目标平移基准4.764mm，未显示相对此基准的优势；首次接触改由API几何规则提出。两个模型重载误差0，分别10/9个调用案例及独立进程检查通过。见[Push训练报告](real_robot/reports/PUSH_TRAINING_V5.md)、[Flip训练报告](real_robot/reports/FLIP_EGG_TRAINING_V1.md)、[Push核验](real_robot/runs/push_letters/training_v5/completion_receipt.json)、[Flip核验](real_robot/runs/flip_egg/training_v1/completion_receipt.json)。费用/tokens分任务留档：[Push100次/$16.283451](real_robot/reports/PUSH_TRAINING_V5_API_USAGE.md)，[Flip71次/$13.4936575](real_robot/reports/FLIP_EGG_TRAINING_V1_API_USAGE.md)，均Astra/xhigh，含失败与修订，无未知usage。未来新运行用[通用单命令入口](real_robot/training_pipeline/README.md)；[新版prompt及旧版留档](real_robot/reports/GENERIC_PROMPT_AND_TRAINING_REVISION.md)完成，新cut未执行。旧prompt/方案/模型、数据、环境锁保留，无真机执行或闭环成功率；本轮无待继续训练。

**Real robot training_v4 prompt/数据利用审查：2026-09-22。** 用户质疑仅 4 个有效样本。核对全部 77 个实际请求及代码：47 个稀疏点最初由 cut_v6 API 选择；将其冻结为实现阶段唯一合法训练索引的是开发者编写的 prompt/校验器，并非用户要求。实际流程为 47→7→4，43 个排除项反映当前标签处理失败，不能证明其余原始数据不可用；原始 16,745 行 EE/flange 位姿均完整有限。19 个目标几何失败中两个仅因轮廓环数不一致而排除；颜色分割、完整联合标签要求与仅 K>=1 的训练入口共同限制了数据利用。见[审查与建议的新合同](real_robot/reports/PUSH_TRAINING_V4_PROMPT_AUDIT.md)和[核验数值](real_robot/reports/PUSH_TRAINING_V4_PROMPT_AUDIT.json)。本次无新 Runtime API 调用、数据标签或训练，原训练包未改；恢复后可用样本数尚未验证。

**Real robot training_v4 完成核对：2026-09-21。** 从 cut_v6 出发，API 经 77 次 Astra/xhigh 调用提交 18 个模型、预处理、Agent/执行转换及调用文档文件，估算 $16.095949。唯一 `shape_push_v1` 为物体轮廓候选评分网络，输出接触点、平面方向和短行程；15,041 个参数，在 GPU 4 按 API 预定预算完成 500 步，最终 EMA 重载预测误差 0。**实际只有 4/47 个锚点保留弱标签，覆盖 4/12 个学习片段，不能据此声称泛化或完整行为覆盖。** 六次零更新预处理检查保留了两次接口错误及 API 修复；18 项框架测试和训练后真实进程调用检查通过。实现使用颜色分割/几何，未使用 SAM/Qwen 权重；原始 16,745 行、旧 cuts 和 training_v3 源码/三个权重均保留并核验。见[训练与覆盖报告](real_robot/reports/PUSH_TRAINING_V4.md)、[本地调用入口](real_robot/policy_training_v4/README.md)、[API 调用目录](real_robot/runs/push_letters/training_v4/package_00/source/POLICY_CATALOG.md)、[模型库](real_robot/runs/push_letters/training_v4/library.json)和[完成凭据](real_robot/runs/push_letters/training_v4/completion_receipt.json)。语义布局、真实 planner/controller 绑定仍由宿主提供；无真机执行或泛化评估。

**Real robot Flip egg 独立 cut_v1 完成核对：2026-09-21。** API 选择 2 个数据组、40 个片段、1 个控制 policy（egg_interaction_v1）与 1 个辅助感知模型；取铲交给固定工具配方及执行器，持铲交互共享带 GRU 的高斯混合策略，输出 6 维铲具局部运动。20 条完整轨迹、45,730 行全部保留，32,737 个控制监督候选锚点；20 个执行器前缀重用 12,570 行，三条人为干预尾部不参与控制监督。79 次请求/返回均核验 Astra/xhigh，129 个发布文件及原始切片通过检查，9 项接口测试通过，费用估算 $12.4355135。数值运动标签、人工感知标注、工具几何及接触跟踪适配尚未实现；无训练、性能评估或机器人执行。用户当前明确发送授权与原审批拦截回执均保留；Push 工作独立。见[中文结果及输入输出](real_robot/reports/FLIP_EGG_CUT_V1_REVIEW.md)、[时间线](real_robot/reports/flip_egg_cut_v1/timeline.png)、[完成回执](real_robot/runs/flip_egg/cut_v1/completion_receipt.json)及[执行来源](real_robot/reports/FLIP_EGG_CUT_V1_STATUS.md)。

**Real robot cut_v6 完成核对：2026-09-21。** API 在[通用 v5 prompt](real_robot/prompts/general_cut_and_prior_v5.md)＋[Push specific v2](real_robot/task_specifications/push_letters_v2.md)下选择 **1 个 learned policy（shape_push_v1）、1 个 prior、1 个 supplied-executor 组**，共 14 段。按用户追加要求，prompt 仅顺带举 contact point 例子，没有指定输出格式或数量。Policy 拟议输出物体边界接触点、平面推动方向与短距离，连接运动/高度/姿态由 planner/controller 承担；12 个学习片段选 47 个稀疏决策锚点，2 条完整执行器轨迹保留全部 16,745 行。数值接触标签、预处理与 planner 适配尚未实现，另提议 SAM 2.1 与 Qwen2.5-VL 两个冻结感知依赖，不能将一个控制 policy 说成整个系统只有一个模型。24 次请求/返回均核验 Astra/xhigh，51 个发布文件与原始切片通过验证，21 项接口测试通过。正式尝试估算 $3.75758；初始被用户修订中断的 9 次完成调用估算 $0.5360945，另一次未决调用保留 $2.0495625 预留而非实际账单。见[中文成品与审阅](real_robot/reports/PUSH_CUT_V6_REVIEW.md)、[Policy 调用目录](real_robot/data/push_letters/cut_v6/POLICY_CATALOG.md)、[执行器/Agent 契约](real_robot/data/push_letters/cut_v6/EXECUTION_CATALOG.md)、[完成回执](real_robot/runs/push_letters/cut_v6_with_example/completion_receipt.json)与[执行来源记录](real_robot/reports/PUSH_CUT_V6_STATUS.md)。旧 cut_v3/v4/v5 与 training_v3 源码/权重哈希不变；本轮无新训练、性能评估或机器人动作。

**Real robot planner 分工提示修正：2026-09-21。** 用户明确学习器选接触位置及后续运动、motion planner 负责连接运动，希望 API 自主决定精简输出而非预设 contact-point 答案或模型数。已形成[通用 API 草案](real_robot/prompts/planner_supported_learning_draft.md)及[修改依据与接入说明](real_robot/reports/PLANNER_SUPPORTED_PROMPT_DRAFT.md)：将完整行为覆盖归于整个系统，学习只覆盖已有执行能力之外的决策，允许稀疏目标/局部运动及共享模型。此前 controller-aware 草案已标为未执行且被取代；未启动新 API、训练或机器人动作，实际 planner 契约与新运行接口尚未接入。

**Real robot 任务空间输出设计草案：2026-09-21。** 用户报告另一台电脑观察到部分合理行为及不合时宜抬升，提出 contact-point motion 与底层 controller 分工。已核对当前六维残差及几何避障抬升路径，形成[下一轮设计 prompt 草案](real_robot/prompts/controller_aware_skill_design_draft.md)：联合重审 cut、任务空间输出、平面/姿态约束、controller 能力与标签转换，并允许无需学习的 controller 技能。此为用户反馈驱动的设计建议，尚无对应失败轨迹归因；草案未用于 API，controller 清单及新执行接口尚待承接，无新训练或机器人动作，已有科学产物保留。

**Real robot training_v3 输入容量与一致性追加核对：2026-09-21。** 每个模型的通用 core 含约 115 万参数，图编码器约 37.6 万；约 200 万总参数相对两条示范存在记忆风险，尚无独立评估证明过拟合。确认搬运默认在线 TCP 目标为入场位姿，训练为示范末端位姿；工具 4/4、搬运 20/22 片段的末行动作非零，而在线 terminal_twist 缺省零。历史按记录/调用次数取样，原记录约 30 Hz，需显式核对线上时间尺度及实际上一动作。见[追加审阅](real_robot/reports/PUSH_V3_LEARNABILITY_REVIEW.md#输入容量tcp-目标与历史条件追加审阅2026-09-21)与[只读数值证据](real_robot/reports/PUSH_V3_INPUT_CONSISTENCY_AUDIT.json)。无新训练、模型推理、API 或机器人动作，科学源码哈希不变。

**Real robot training_v3 输入可视化：2026-09-21。** 已从冻结缓存展示三个模型的四时刻几何图、112 维状态和接触图模型的当前 32 个节点，并对照原始 RGB 缩略图。涵盖工具接近、有目标位移及零目标位移的三个片段中点；无损地图编码逐帧核对，20 种模型/样本/历史组合、27 个状态组、32 个节点的显示交互通过 V8 核对。见[输入说明、数据及证据](real_robot/reports/input_conditions_v3/README.md)。未改科学源码或模型，无新 API、训练、网络推理或机器人动作；浏览器布局未验证。

**Real robot training_v3 可学性审阅：2026-09-21。** 已有日志显示三个模型的训练动作拟合改善，尚无独立性能或闭环成功证据。确认搬运模型训练时的末端目标为片段终点 TCP，而默认在线调用为入场 TCP；该输入条件差异尚未测量影响。弱阶段标注中 6,342/15,765 行为 exit，标签由轮廓误差阈值产生，不能视为已核验退出行为。仍有六维残差、混合动作阶段及代理接触平面等限制。见[具体审阅与验证优先级](real_robot/reports/PUSH_V3_LEARNABILITY_REVIEW.md)及[已有日志统计](real_robot/reports/PUSH_V3_LEARNABILITY_FACTS.json)。仅只读审阅，无新 API、训练、模型评估或机器人动作；冻结源码和模型保持不变。

**Real robot v3 跨电脑推理包完成核对：2026-09-21。** 为接收电脑准备独立 `deployment_v3`，15 个 API 源码/文档逐字节复制到 `policies/push_v3`，三个最终 EMA 无损导出到 `checkpoints/push_v3`。完整包约 22 MiB，包含独立环境锁和 Agent 调用文件；实际压缩包在新目录解压、用锁定推理环境通过三模型调用核验，与原加载器候选动作误差均为 0。无需训练数据、runs 或 API key；无新 API、训练或机器人动作。新代码尚未提交/推送，`git pull` 暂不能取得本轮新增文件；完整包可直接下载。见[迁移安装说明](real_robot/deployment_v3/README.md)、[打包报告](real_robot/reports/PUSH_V3_DEPLOYMENT.md)和[实际包检查](real_robot/reports/PUSH_V3_DEPLOYMENT_CHECK.json)。旧科学源码、权重、框架和环境锁保留。

**Real robot training_v3 三模型完成核对：2026-09-21。** 三个 cut_v5 prior 分别完成独立 20,000 步训练，最终 EMA 重载误差均为 0；实际调用均返回有限六维候选动作，并通过过期输入、非刚体目标和中断检查。API 用 34 次 Astra/xhigh 调用提交 15 个源码/文档文件，选择几何换位残差、轮廓接触图和目标场伺服模型，并解释未采用 Diffusion Policy 的原因；费用估算 $8.6769445。26/26 段、17,127 个动作锚点保留；工具模型 1,362 条，两个搬运模型各 15,765 条。搬运数据中 15,165 条有自动生成的物体目标，a_bar_stage 的 600 条缺少可信终点目标，仍参与动作监督；自动标注未经人工验证。所有缓存浮点值有限，8 项边界测试通过。模型并非硬约束的纯二维推策略；控制器解码禁用，无真机执行或泛化评估。见[完整报告](real_robot/reports/PUSH_TRAINING_V3.md)、[API 调用目录](real_robot/runs/push_letters/training_v3/package_00/source/POLICY_CATALOG.md)、[模型与权重清单](real_robot/runs/push_letters/training_v3/library.json)、[完成核验](real_robot/runs/push_letters/training_v3/completion_receipt.json)及[预处理核对](real_robot/runs/push_letters/training_v3/preparation_review_00.json)。旧 cut_v3/v4/v5 与旧训练框架哈希保持不变；训练和接口检查进程已退出。

**Real robot cut_v5 完成核对：2026-09-21。2 个子任务组、3 个 policy/prior、26 个片段。** 新增的小数据要求促使 API 明确物体/接触相对表示、平面运动与受限残差，但仍将接近/推/重接触放在完整物体调用内部，且把微调合并回搬运，未形成独立平推组。主要监督改为原始 v/w，关节命令由拟议 verified follower 转换并核验。16,745 行全部保留，v/w 均有限，16,417 行有有限七维 dq；衍生输入尚未预处理。29 次请求和返回均核验 Astra/xhigh，估算 $5.48676；85 个发布文件及原始切片通过验证，旧 cut_v3/v4 哈希保持不变。见[中文成品与对照](real_robot/reports/PUSH_CUT_V5_REVIEW.md)、[API 调用目录](real_robot/data/push_letters/cut_v5/POLICY_CATALOG.md)、[验证](real_robot/runs/push_letters/cut_v5/validation.json)与[执行记录](real_robot/reports/PUSH_CUT_V5_STATUS.md)。本轮未实现或训练 policy。用户的[持续发送授权](real_robot/authorizations/openai_data_processing_20260921.json)已记录，同类工作流不再主动逐轮询问。

**Real robot cut_v4 切分依据追加核对：2026-09-21。** 原始 16,745 行均保存有限的完整 EE 4×4 位姿，API 可以读取；不存在水平 EE 位置未保存的问题。实际运动概览仅整轨迹 24/20 bins（每 bin 约 12–13 秒），另查看 69/59 个状态索引。提交方案已识别平面推/抬起/重接触并采用物体局部几何，但仍按完整物体搬运组织主组。当前判断是切分粒度目标与运动证据呈现需要改进；桌面参考系/接触几何尚待核验。详见[追加分析](real_robot/reports/PUSH_CUT_V4_INTERFACE_AUDIT.md#7-追加核对为何没有单独形成平面推组)及[调用与原始位姿核验](real_robot/runs/push_letters/cut_v4/interface_audit/audit_facts.json)。未改动 prompt 或冻结设计，未启动新实验。

**Real robot cut_v4 子任务与接口审阅日期：2026-09-21。** 三组四 policy 的分工与 prior 差异有实质内容，调用契约仍需补齐目标/工具位姿生成、可计算的选择条件、统一停止/交接状态和可选退出位姿的缺省行为。工具组仅两段 approach、两段 transfer，共 1,072 条有限命令，四段末行均非零；独立 depart 的训练支持尚未单独建立。两个搬运 policy 仍包含重接触/离开，不是单独的纯平面推接口。旧训练入口固定两个 policy、共享样本集，后续承接三组四 policy 需要新接口。见[详细审阅](real_robot/reports/PUSH_CUT_V4_INTERFACE_AUDIT.md)、[可复现数值与文件核验](real_robot/runs/push_letters/cut_v4/interface_audit/audit_facts.json)。本轮仅审阅与确定性测量；93 个发布文件哈希不变，无新 API、训练或真机执行。

**Real robot cut_v4 完成核对日期：2026-09-21。3 个子任务组、4 个独立 policy/prior 设计、28 个片段。** API 分为工具接近/脱离/换位、物体搬运、局部对齐微调三组；搬运组提供几何响应与双视角时序两个 prior。前后各 15 行上下文 buffer，另有四段跨组动作监督复用；16,745 条原始记录全部纳入，16,417 条具有有限七维命令，尚未经过衍生输入预处理。27 次请求和返回均核验 Astra/xhigh，估算 $5.3269；12 项接口测试、93 个发布文件及原始切片核验通过，旧 cut_v3 文件保持原 hash。用户已明确授权本轮数据发送，原审批记录保留。本轮未实现或训练 policy。见[中文成品与时间线](real_robot/reports/PUSH_CUT_V4_REVIEW.md)、[API 调用目录](real_robot/data/push_letters/cut_v4/POLICY_CATALOG.md)、[完整原文报告](real_robot/reports/PUSH_CUT_V4.md)、[核验](real_robot/runs/push_letters/cut_v4/validation.json)及[执行记录](real_robot/reports/PUSH_CUT_V4_STATUS.md)。

**Real robot cut prompt 设计核对日期：2026-09-21。新版已用于上述 cut_v4。** 按用户修订，先确定完整覆盖示范行为的子任务，再通过自由取段、大量复用及与 prior 兼容的前后 buffer 组成各子任务训练组，最后设计独立 heuristic/policy 及高层调用、交接契约。真机衍生/privileged input 的完整预处理与部署转换要求保留；未预设具体技能或数量。见[新版 prompt](real_robot/prompts/general_cut_and_prior_v2.md)和[修订记录](real_robot/reports/CUT_PROMPT_V2.md)。已执行的原 prompt、cut_v3 配置和科学产物保持原样。

**Real robot training_v2 完成核对日期：2026-09-21。** 两个模型均完成独立 20,000 步，最终 EMA 重载误差为 0；prompt 明确允许 Diffusion 或其他模型，API 仍分别选择高斯混合模型。25 次 gpt-6-astra/xhigh 调用估算 $4.5841，提交 15 个源码/文档文件，包含逐策略 prior、使用指南和 HANDOFF.json。实际监督 6,314 条、覆盖 12/22 段，少于 v1；未进行性能比较或真机评估。[v2 完整包与调用入口](real_robot/deployment_v2/README.md)已完成，无损 EMA 导出和最终压缩包的独立解压接口检查通过；[Agent 目录](real_robot/policies/push_v2/source/POLICY_CATALOG.md)、[训练报告与证据](real_robot/reports/PUSH_TRAINING_V2.md)。cut_v3、train_v1、push_v1 与原框架哈希保持不变，训练和检查进程均已退出。

**Real robot 延迟核对日期：2026-09-21。部署实时性尚不满足记录中的 20 Hz 节奏。** 针对用户报告的目标电脑 0.6–0.7 秒，本机 RTX A5000 上用少量连续配对记录对原代码/权重计时：稳定 public `act` 中位数为 135.4 / 143.4 ms，纯网络前向 2.71 / 5.50 ms，首次 inventory 约 0.51–0.54 s；后者加一步 act 与用户报告量级相符，但目标电脑计时范围未知。主要可见开销为 CPU 转换及每步约 12.3 MB 的 JSON/Base64 请求；视觉策略也重复构造图特征。见[诊断与测量范围](real_robot/reports/PUSH_LATENCY.md)。原部署检查只验证加载/接口，没有做 20-Hz 实时验收。本轮仅诊断，无新 API、训练、机器人动作或模型/部署包修改。

**Real robot 部署核对日期：2026-09-21。可下载推理包已完成。** API 原始 10 个文件逐字节复制到 Git 可管理的 `real_robot/policies/push_v1/source/`；两个推理 checkpoint 位于 `real_robot/checkpoints/push_v1/`，共 51,025,976 字节，所有最终 EMA 张量与训练原件完全一致。完整 SCP 包约 45 MiB，另有仅权重包，见[下载与安装](real_robot/deployment/README.md)、[部署记录](real_robot/reports/PUSH_DEPLOYMENT.md)。独立锁定的 Pixi 推理环境和可迁移 Python 接口已在实际压缩包的新解压目录通过双模型加载/接口检查，该目录不含原始数据或 runs；见[回执](real_robot/reports/PUSH_DEPLOYMENT_CHECK.json)。用户明确由接收端适配机器人通信；本轮无新 API、训练、远程发布或真机动作，原训练缓存/权重/来源完整保留。完整包可直接 SCP 使用，不依赖 Git 远端是否更新。

**Real robot 核对日期：2026-09-21，train_v1 已完成。** GPT-6 Astra/xhigh 经 20 次调用提交 `contour_push`、`visual_push` 及完整 observation/goal 转换、动作解码等 10 个文件；请求和响应档位均已核对，本阶段 API 估算 **$4.1899**。两个模型在 GPU 0、1 各完成 **20,000 步**，分别为 3,671,118 / 9,067,629 个可训练参数；权重确有更新，最终 EMA checkpoint 重载动作误差均为 0，且均可由正式本地调用接口加载。预处理实际得到 **8,606 条监督样本，覆盖 18/22 段**；感知/跟踪排除 4,201 行、目标转换排除 2,410 行，4 段无有效监督，不能声称所有 motion 已覆盖。API 的暖色材质感知有材质/尺度/遮挡限制，未知字母/形状/词汇泛化尚未评估。开发者仅提供接口与执行，无额外前置性能验证、修复调用或重训。见[训练报告与覆盖表](real_robot/reports/PUSH_TRAIN.md)、[模型目录与来源](real_robot/runs/push_letters/train_v1/library.json)、[调用入口](real_robot/policy_training/README.md)、[完成记录](real_robot/runs/push_letters/train_v1/execution_status.json)。cut_v3、原始数据及 Exp1/Exp2 保持不变；未运行真机或 Flip egg。

**Real robot 核对日期：2026-09-20。Push cut_v3 已完成：22 个语义片段、1 个共享数据组、2 个独立 heuristic / policy 设计。** [通用主 prompt](real_robot/prompts/general_cut_and_prior.md)和[Push Task Specification](real_robot/task_specifications/push_letters.md)分别发送；25 次实际请求均核验为 GPT-6 Astra/xhigh，12 项接口测试及 71 个发布文件核验通过。两条轨迹分别切为 12 / 10 段，每段明确物理实例、局部空间目标、锚点像素框及监督排除项；临时挪动与后续再调整分开。`contour_push` 采用轮廓几何 prior，`visual_push` 采用双视角时序视觉 prior，分别拟议训练并由 High-level Agent 调用。16,745 条原始记录全部保留；1,200 行交接带屏蔽动作 loss，剩余 15,545 行候选中 15,217 行具有有限命令 dq，衍生输入有效性尚待实现。见[片段清单与审阅](real_robot/reports/PUSH_CUT_REVIEW.md)、[完整 API 报告](real_robot/reports/PUSH_CUT.md)、[调用目录](real_robot/data/push_letters/cut_v3/POLICY_CATALOG.md)、[验证](real_robot/runs/push_letters/cut_v3/validation.json)。本轮估算费用 **$4.4980**，独立于前两轮已删除运行的 $5.1827；[cut_v2 删除记录](real_robot/deletions/push_cut_v2_20260920/receipt.json)保留费用和文件清单。用户的[官方 API 地址授权](real_robot/runs/push_letters/cut_v3/destination_authorization.json)已解决启动前审批拦截。原始数据与 Exp1/Exp2 未改动；未写 policy/preprocessing code、未训练、未运行 Flip egg。感知权重、精确 mask、转换器、控制器语义核验及未知字母/形状/词汇的实际泛化验证仍属后续工作。

**核对日期：2026-09-20。Exp2 目录整理完成，实验仍已停止。** 当前唯一主线为 **Exp2_new：五任务 × 每任务 15 个位置 OOD × DP / Agent Prior DP / APPL GPT-6 Astra/xhigh，共 225 个完整结果**。成功分别为 **5/75、10/75、30/75**；无新增 API、训练或仿真。

- [当前入口与目录结构](experiments/exp2/README.md)、[完整报告](experiments/exp2/reports/REPORT.md)、[失败分析](experiments/exp2/reports/FAILURE_ANALYSIS.md)、[恢复案例](experiments/exp2/reports/RECOVERY_CASES.md)、[225 个视频](experiments/exp2/reports/VIDEOS.html)。
- [唯一资产清单](runs/exp2/current/manifest.json)：225 个 OOD 回合、46 个冻结模型及哈希。三个方法使用同初态与修正后的二值夹爪执行器；没有重新训练、选择模型或筛除失败。
- 旧错误夹爪实验、其余轮次、部分 ID 和诊断移入 [归档](archive/Exp2_history_20260920/README.md)。历史报告/API 输出/轨迹/模型保留；旧路径通过兼容链接可解析。Exp1 原有资产与冻结科学框架保持不变，见[迁移核验](archive/Exp2_history_20260920/migration/validation.json)。
- 已删除 Exp2 写作 ZIP、重复打包目录和打包脚本；其中独有报告、表格、图与分析已保留。详见[删除清单](archive/Exp2_history_20260920/migration/deletion_inventory.json)。
- GPT-6 Astra/xhigh 的示范切分、prior、源码、部署决策仍为 API 产物；开发者提供环境、示范、工具、训练与判定。此整理没有修改实验逻辑。未来 Exp2/其他新实验使用 `xhigh`；Exp1 相关使用 `max`。

完整 Exp2 执行沿革见[整理前进度记录](archive/Exp2_history_20260920/navigation_before/PROGRESS.md)；各阶段原报告均保留，不在当前摘要重复历史状态。

## 历史记录：Exp1 执行过程

以下保留原进度原文，最后一条记录止于 2026-09-13 隐藏测试进行中。历史正文中的“当前”“最新”指各自记录时点；当前结论以本页顶部摘要及其证据为准。

---

# Experiment 1 进度入口

当前执行规范是 EXPERIMENT1_CODEX_EXECUTION_HANDOFF.md 和用户追加的 RuntimePriorAPI 工具 agent 要求。

用户最新 GPU 分配为物理卡 4、5、6、7，替代旧的 0–3 分配。候选 GPU worker 按 UUID 绑定，保留所有历史尝试的原始设备记录。

实时状态：`pixi run --locked experiment1-status`。
报告：`experiments/experiment1/EXPERIMENT1_REPORT.md`。

已完成实际精确迁移、六任务支持数据/控制审查、独立dev/test初态和24套真实图像证据。真实GPT-6 Astra/max工具冒烟已完成12次API调用、一次零更新接口检查和显式候选提交；它是合成基础设施测试，不属于144正式训练槽位。

公共GPU预检、源码/协议冻结与正式矩阵状态以可验证工件为准。不得把本进度说明当作训练已完成的证据。旧PROGRESS/README/AGENTS与Pixi环境快照保存在archive/legacy_rounds/round3/environment_before_experiment1。

2026-09-12 续推：修复 PyTorch CUDA UUID 与 NVML 的 `GPU-` 前缀差异，保留原始 UUID；`gpu-uuid-format-r4` 基础设施修订通过六任务公共拟合、物理 GPU 4–7 核验及精确恢复检查。旧失败尝试保留。完整测试 268 passed / 1 skipped，修复后相关测试 35 passed。实际 preflight 已通过，protocol.lock.json 已冻结。

早先执行 `pixi run --locked experiment1-run --resume --detach` 因启动环境缺少 `LOCAL_OPENAI_BEARER_TOKEN` 而停止，历史失败详情保留在 `experiments/experiment1/last_failure.json`。

2026-09-12 10:24 UTC：用户提供并授权持久化凭据后，已保存至仓库外 `~/.config/experiment1/runtime-api-credential.json`（0600）。专用启动入口 `~/.config/experiment1/launch.py` 自动加载凭据，通过 Pixi 调用冻结 runner；不修改实验源码或冻结模型/地址。恢复命令：`/home/users/oscar/.pixi/bin/pixi run --locked python /home/users/oscar/.config/experiment1/launch.py`。已有 runner 运行时不要重复启动。

后台 runner PID 853471，日志目录 `experiments/experiment1/logs/runner-1789208654505688207/`，物理 GPU 队列 4–7。首批四个正式 task×N 设计会话均已获得 HTTP 200 响应并执行工具循环。凭据阻塞已解除；启动时正式训练/dev/test 为 0/144，实时计数以 status 为准，不能据此声称实验完成。

2026-09-12 10:46 UTC 核查：pick-place-wall 四个 N 档的 A1/A2/A3 共 12 个正式候选已显式提交且接口检查有效，全部提交文件 hash 与冻结协议核验通过。正式 API 调用 153 次均为 HTTP 200 并已消费（另有基础设施 smoke 12 次）；正式接口检查 12 次。四个 B0 正在物理卡 4–7 训练，记录步数分别为 N2=11800、N5=12200、N10=11200、N20=14400，目标均为 20000；正式完成训练/dev/test 仍为 0/144，无新失败或阻塞。未向 API 返回开发反馈，隐藏测试尚未开启。报告已刷新，检查快照见 `experiments/experiment1/reports/live_execution_audit.json`。后台 runner 继续按冻结流程调度。

2026-09-13 01:25 UTC 最新资源变更：用户授权物理 GPU 0–7 全部使用，0–3 实测空闲。已准备仅涉及 CLI 卡号、隔离 worker 设备授权、runner 并行度与协议补充校验的精确改动；原文件和待应用文件保存在 `experiments/experiment1/allocation_20260913/{original,proposed}/`。原 `protocol.lock.json`、科学配置、候选代码和已有结果保持原 hash；补充记录为 `execution_allocation.json`，仅在切换时产生。八卡调度与修订校验的 6 项测试已通过。

为避免运行中源码变化，仅结束旧调度进程 853747，保留四个独立实例及全部训练/API worker。后台切换控制器 PID 321717 正在等待现有实例结束，然后自动应用资源修订、运行相关测试、对八张卡逐一进行零更新隔离/UUID 检查并启动八卡 runner。切换状态见 `experiments/experiment1/allocation_20260913/transition_status.json`，启动结果见同目录 `launch.log`；当前 `draining` 表示八卡尚未正式启用。不要同时手动启动另一个 runner。任何检查失败会明确写 blocked 并停止，不自动重试；失败槽位不因此重训。

2026-09-13 03:28 UTC：八卡资源修订已应用，55 项相关测试和八张卡零更新隔离检查通过，新 runner PID 512751 已实际启动。

2026-09-13 05:39 UTC：用户明确授权补上 assembly/N10/A4 的 OOM 失败槽位。全局隐藏测试尚未开始；仅停止外层调度进程 513020，保留 stick-push/N2 的独立运行实例，避免补跑期间越过全局测试门槛。恢复程序 PID 662683 在空闲物理 GPU 0 上加载原 step=1 checkpoint；实际训练账本确认 resumed_step=1，已推进至 step=100。候选代码/配置/seed/科学槽位不变，训练 attempt 与中断费用单独保留。

恢复原始 checkpoint、失败 slot、会话、q4 选择及开发反馈均备份至 `experiments/experiment1/recoveries/assembly_N10_A4_20260913/`。后台脚本完成训练/dev 后，仅将新开发结果返回同一 API 会话重新确定 q4，保留旧选择和全部工具日志，q1/q3 不变；所有现有实例结束后自动恢复八卡 runner，再全局冻结并隐藏测试。状态与错误见该目录 `status.json`、`stderr.txt`，恢复调度结果见 `restart.log`。本次是用户明确授权的一次恢复，没有自动重试；失败则写 blocked 并保留原因。

2026-09-13 06:03 UTC：assembly/N10/A4 恢复成功，完成原定 20000 步及开发评估。API 根据新开发结果按原规则将 q4 更新为 A4（开发 C/E 等权分数 0.725，原 A2 为 0.475）；原失败与选择记录保留。恢复程序已重新启动外层 runner PID 719175，24 个实例随后全部全局冻结，开始物理 GPU 0–7 八卡隐藏测试。06:31 UTC 状态：144 个系统均已完成训练/dev，8 个系统完成 test、8 个正在 test；这些是进度计数，不是最终隐藏测试结论。日志目录 `experiments/experiment1/logs/runner-1789279412036418660/`。
