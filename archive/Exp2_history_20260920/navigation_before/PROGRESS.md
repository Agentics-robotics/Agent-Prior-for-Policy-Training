# 项目进度

## 当前摘要

**2026-09-20 Exp2_new 论文资料与失败复核完成：** 新建独立英文[完整报告](experiments/exp2/Exp2_new_paper/REPORT.md)、[写作指引](experiments/exp2/Exp2_new_paper/WRITING_GUIDE.md)、[失败分析](experiments/exp2/Exp2_new_paper/FAILURE_ANALYSIS.md)及[恢复案例](experiments/exp2/Exp2_new_paper/RECOVERY_CASES.md)。[完整证据 ZIP](experiments/exp2/Exp2_new_paper_bundle.zip)约814 MB，包含225个完整OOD结果、30个配对部分ID结果、255视频、60条完整数值示范、API原始日志/切分/41个API策略源码和完整轨迹；46个checkpoint二进制约17.18 GB仅列路径/哈希。[写作版 ZIP](experiments/exp2/Exp2_new_writing_bundle.zip)约60 MB，保留正文、结果、源码、示范、视频和案例证据，明确索引省略的全量日志/轨迹。两版逐文件哈希、ZIP CRC、主文档链接及结果一致性通过[完整包核验](experiments/exp2/Exp2_new_paper_bundle.validation.json)和[写作版核验](experiments/exp2/Exp2_new_writing_bundle.validation.json)。逐轨迹统计：抽屉15/15曾打开、5/15曾放好红块；双块分拣15/15曾放好红块、3/15放好蓝块；缓冲交换失败分布为首次红块4、蓝块3、最终红块4。全部45个APPL失败均由API主动finish，418–2663步，中位1049，非5000步耗尽。已确认两个原始API选策略后成功的恢复案例；因果解释仍待消融。此次0新API、0训练、0仿真，原实验/API输出/Exp1未修改。

**2026-09-20 Exp2_new OOD 全部完成并停止：** 五任务各15个位置OOD，**75/75完整结果、30成功（40.0%）、0未决结果**。同初态修正 naive DP **5/75（6.7%）**，SinglePrior **10/75（13.3%）**；分任务 APPL 为抽屉2/15、双块分拣3/15、缓冲交换4/15、拆堆11/15、托盘10/15。复用原7个完整OOD，追加68个最终结果；没有重测已完成失败。**未新增ID，仍为8/10；其余65个ID未调度。** 复用36个冻结policy，0重训/切分/设计，所有实际请求均Astra/xhigh。追加500 SGD范围内共1,197次生成请求，用量费用估算 **334.5613525 USD**，低于390 USD规划上限；Exp2_new连同原100 SGD轮次共1,460次生成请求、估算412.129084 USD，未知费用为0。短时GPU占用暴露过严调度保护，造成4个前缀中断；已修正资源调度并保全原前缀/费用，同初态补测首请求完全一致。所有进程退出、全部视频/动作/字面API规则核验通过，11,295个此前文件哈希不变，17项框架检查通过。见[最终三方法表](runs/exp2/Exp2_new/ood_resource_resume_20260920/REPORT.md)、[费用和执行分析](runs/exp2/Exp2_new/ood_resource_resume_20260920/ANALYSIS.md)、[完成核验](runs/exp2/Exp2_new/ood_resource_resume_20260920/completion.json)、[75个视频](runs/exp2/Exp2_new/ood_resource_resume_20260920/replays.html)。这是已查看初态上的配对事后比较，原实验与Exp1保留；按用户要求停止，不继续ID或其他实验。下方各轮次描述为执行历史。

**2026-09-20 OOD 资源中断后继续执行：** 追加预算第一阶段已核验 **45 个累计 OOD 结果、17 成功**（新增 38 个），另 4 个资源中断前缀保留。GPU 0 曾短时出现 PID 3207271 / 502 MiB，归属未确认即退出；调度器过严地把单卡占用升级为全局停止，导致另外四个回合中断。已保留原源码/账本/前缀并只修正资源调度：忙卡等待，其他回合继续。17 项框架测试通过。剩余 **30 个 OOD** 在 GPU 1/2/3/4/7 继续，包含4个中断项同初态补测；仍不调度 ID，不重测已完成失败。500 SGD 对应的 390 USD 总规划上限不变：第一阶段花费195.2687485 USD，续接阶段仅获剩余194.7312515 USD。最新[报告](runs/exp2/Exp2_new/ood_resource_resume_20260920/REPORT.md)、[状态](runs/exp2/Exp2_new/ood_resource_resume_20260920/results.json)、[资源事件](runs/exp2/Exp2_new/ood_resource_resume_20260920/resource_incident.json)。此前阶段均原址保留。

**2026-09-20 Exp2_new OOD 续跑已启动：** 用户追加 **500 SGD**，要求先完成全部 **75 个 OOD** 后停止。复用已完成 7 个 OOD，剩余 68 项已冻结为[独立清单](runs/exp2/Exp2_new/ood_continuation_20260920/plan.json)；其中仅上轮预算中断的 buffer_swap/OOD/30101 从同初态重测，原 755 步前缀保留，已完成失败不重跑。新增预算按原保守规划比例设 **390 USD**；旧账本及结果不改。GPU 0/1/2/3/7 并行，每卡一回合；15 项离线框架检查通过，原36模型检查直接复用。原 evaluator、模型、prompt、xhigh、夹爪修正和种子不变，仅变更存储/预算/调度。**不新增 ID、训练或 API 设计**。见[本轮报告](runs/exp2/Exp2_new/ood_continuation_20260920/REPORT.md)、[状态](runs/exp2/Exp2_new/ood_continuation_20260920/results.json)、[新账本](runs/exp2/Exp2_new/ood_continuation_20260920/budget/ledger.json)。下方为上一轮已结束的历史记录。

**2026-09-20 Exp2_new 预算内执行已结束并核验：** 150 回合计划中 **17 个完整结果、1 个预算中断、132 个未启动**；APPL Astra/xhigh 为 **ID 8/10、OOD 4/7**。相同已完成初态上，修正 naive DP 为 ID 6/10、OOD 0/7，修正 SinglePrior 为 ID 10/10、OOD 2/7；这是少量、已查看初态的配对结果，不是完整 15 ID＋15 OOD 结论。263 次生成请求均为 HTTP 200，报告输入 11,706,446 token、输出 123,015 token；按保存费率估算 **77.5677315 USD**，未知费用为 0。用户授权 100 SGD，账本上限保守取 78 USD；剩余 0.4322685 USD 不足下一请求的完整预留，故在发送前停止，未发生服务端拒绝。仅 GPU 7 串行推理，复用全部 36 个冻结 policy 和已有 baseline；0 重训、0 新切分/设计、0 传输重试。18 个回合/前缀及视频、实际 API 参数与原始输出、费用对账全部通过核验，67 个策略 worker 和全部调度进程退出。见[英文报告](runs/exp2/Exp2_new/REPORT.md)、[费用与配对分析](runs/exp2/Exp2_new/ANALYSIS.md)、[完成核验](runs/exp2/Exp2_new/completion.json)、[全部视频](runs/exp2/Exp2_new/replays.html)。保留首次零请求路径错误及一行修复的完整证据；12 项框架测试通过。剩余计划不自动重启，原实验、模型与 Exp1 保持原样。

以下为此前准备时的历史记录；币种已确认，执行状态以上方核验为准。

**2026-09-19 Exp2_new 准备完成，付费部署未启动：** 按用户新命名建立[独立入口](experiments/exp2/EXP2_NEW.md)、[冻结计划](runs/exp2/Exp2_new/plan.json)与[当前报告](runs/exp2/Exp2_new/REPORT.md)。复用 36 个 Astra/xhigh policy，每任务取原 ID/OOD 列表前 15 个初态，共计划 150 个新的 APPL 部署；引用已完成的 300 个同初态修正 baseline 结果，不重跑、不重训。新凭据仅用于官方 endpoint；实际生成固定 `gpt-6-astra/xhigh`，采用与 baseline 相同的夹爪符号解码。11 项费用/执行测试及全部 36 个模型离线复现检查通过：324 个动作最大差异为 0，全部检查进程已退出，见[完成核验](runs/exp2/Exp2_new/setup/offline_completion.json)。用户所说“一百块”的币种仍待回复，尚无新付费生成或物理测试。按 seed 轮转五任务/两 split；预算和服务中断保留 unknown，不算任务失败。旧 600 个 baseline、原 APPL 与 Exp1 全部保留。

**2026-09-19 夹爪修正完整补测完成：** 复用十个冻结模型，只将执行夹爪映射为非负值全开、负值全关；五任务 × 两方法 × 30 ID/30 OOD，共 **600/600** 回合及视频全部通过[最终核验](runs/exp2/binary_gripper_v1_20260919/final_validation.json)。Naive DP：ID **17→76/150（50.7%）**，OOD **0→7/150（4.7%）**；SinglePrior_6_xhigh：ID **110→139/150（92.7%）**，OOD **16→24/150（16.0%）**。SinglePrior 四个新增任务 ID 均 **30/30**，但抽屉 ID **22→19/30**；naive 抽屉保持 15/30，存在 6 得/6 失。本轮仅 4 个成功发生在 1500 步之后，无成功超过 3000 步。见[完整英文报告](runs/exp2/binary_gripper_v1_20260919/REPORT.md)、[结果解释和剩余失败](runs/exp2/binary_gripper_v1_20260919/ANALYSIS.md)、[全部视频](runs/exp2/binary_gripper_v1_20260919/replays.html)与[规范](experiments/exp2/BINARY_GRIPPER.md)。0 重训、0 Runtime API、0 自动重跑、0 unknown；1,967,663 物理步，全部进程正常退出，峰值五张 GPU。十个模型原始动作复现最大差异均为 0，冻结输入/源码/checkpoint/旧结果保持原 hash。两个 APPL 按用户要求暂停，其历史分数不可当作相同修正执行器下的四方法对照；本轮为已查看初态的配对事后消融。

**2026-09-19 四方法夹爪复核：** 已只读检查全部 1,200 条选定正式轨迹，三组 prior 方法均沿用连续夹爪接口，没有统一二值化。四个新增任务各 120 ID/120 OOD 回合中，“渐进空中闭合”保守筛查：naive DP **68/91**、single-policy prior **24/52**、旧 APPL **15/32**、新 APPL **0/10**；这些是现象计数，不是因果归因失败数。旧 APPL 检出的 15 个 ID 回合有 13 个最终成功，托盘 seed20301 的原始 API 日志明确记录调用释放策略重新张开空夹爪、换抓取 prior 后于 910 步完成。见[跨方法报告](experiments/exp2/analysis/cross_method_gripper_20260919/REPORT.md)、[逐回合证据](experiments/exp2/analysis/cross_method_gripper_20260919/episodes.json)和[源码/完成核验](experiments/exp2/analysis/cross_method_gripper_20260919/completion.json)。0 新仿真/训练/Runtime API；92 个 API 源码与论文归档一致。三组 prior 尚未做夹爪二值化因果消融。

**2026-09-19 DP 闭环诊断完成：** 冻结 DP 在原始 12 个训练初态上，抽屉 **9/12**，四个新增任务各 **0/12**（诊断上限 1,500 步）；60 次示范动作重放全部成功。定位到连续夹爪输出导致手指逐渐闭合、模型进一步预测闭合的反馈放大。仅将夹爪按正负执行成全开/全关，同三个训练初态上分拣 **0→3/3**、交换 **0→2/3**、拆堆 **0→3/3**，托盘仍 **0/3**，抽屉维持 **2/3**。示范前缀接管：抽屉/分拣/拆堆/托盘各 **3/3**，交换 **1/3**；托盘示范的红块放置 Y 范围仅约 1.3 mm，修正夹爪后两次合格放置偏离示范 28–33 mm 后无法接续。见[完整英文诊断](experiments/exp2/analysis/dp_diagnosis_20260919/REPORT.md)、[配对视频](experiments/exp2/analysis/dp_diagnosis_20260919/videos.html)和[完成核验](experiments/exp2/analysis/dp_diagnosis_20260919/completion.json)。这是独立训练初态与自适应诊断，不替换原 30 ID/OOD 主表；0 新训练、0 Runtime API，Exp1 和冻结科学输入未改。

**2026-09-19 实验合理性只读复核：** 确认 APPL 切换 Policy 时清空历史并复制当前帧作为两帧输入；旧/新 APPL 的选定 300 回合分别出现 1,867/1,385 次非初始策略切换。其对成功率的影响尚未做消融，不能把切换次数视作失败数。复核同时指出弱 DP baseline、训练量/时间信息差异、未验证的交接能力和单帧几何成功口径等限制。见[详细复核](experiments/exp2/analysis/experimental_review_20260919/REVIEW.md)及[原始计数与源码核验](experiments/exp2/analysis/experimental_review_20260919/evidence.json)。0 新 Runtime API 请求、0 训练、0 仿真；冻结代码和结果未改。报告有一处仍写补测前 180/299 的摘要句，当前有效结果为 181/300，已在复核中标注。

**2026-09-19 唯一 unknown 补测成功：** 旧 APPL GPT-5.5/high 的 `buffer_swap / ID / 20116` 按原设置从相同初态重跑，**945 步成功、10 次 API 请求**；初态、prompt、首请求、冻结策略、全部动作和字面停止规则核验通过。交换任务 ID 现为 **29/30**，旧 APPL 总 ID **131/150**、OOD **50/150**；主矩阵为 **1,200 个完整结果、0 个 unknown**。见[补测完成回执](runs/exp2/M1_scaleup/evaluation_recovery_20260919/completion.json)、[补测视频](runs/exp2/M1_scaleup/evaluation_recovery_20260919/buffer_swap/evaluation/APPL/ID/20116/replay.mp4)。原 650 步中断、冻结结果及[更新前 ZIP](archive/Exp2_paper_before_20260919_recovery/)全部保留，当前论文报告通过显式补测记录更新。GPU 3 进程已正常退出；没有训练或其他回合重跑。报告、图表和 ZIP 已同步更新，[最终核验](experiments/exp2/paper_bundle_validation.json)通过：893 个包内文件、379 个主文档链接、原 40 段示例加 1 段补测视频；1,292 次保留尝试包含 92 次历史中断。下方 2026-09-18 摘要描述补测前状态。

**Exp2 论文资料已完成（2026-09-18 复核）：** [完整英文报告](experiments/exp2/paper/REPORT.md)、[中文入口](experiments/exp2/paper/START_HERE.md)、[写作 AI 指引](experiments/exp2/paper/WRITING_GUIDE.md)和[资料 ZIP](experiments/exp2/Exp2_paper_bundle.zip)均已生成并核验。四方法主矩阵为 1,200 个预定结果、1,199 个完整结果和 1 个保留的旧 APPL 5.5 未知结果。资料包含方法/数学接口、M0 和初始 M1 历史、结果及失败分析、成本口径、表格/图、97 模型索引、92 个 API 原样策略包、10 组切分及实际 prompt、40 段配对原始视频。876 个包内文件的哈希、ZIP 完整性和 371 个主文档链接通过[资料核验](experiments/exp2/paper_bundle_validation.json)。历史中断/授权重跑另外导出为尝试清单，不当作新的独立初态或任务失败。

**新增 single-policy baseline 已完成（2026-09-18）：** 每任务一个 GPT-6 Astra/xhigh 候选，使用各 12 条完整示范、共享归一化和 60,000 更新；5 个模型、300 个新测试及视频全部通过核验。最终 **126/300 成功：ID 110/150（73.3%），位置 OOD 16/150（10.7%）**；部署 API 调用为 0。实际离线设计为 116 个已消费请求及 1 个保留的 HTTP 502；托盘按授权原样恢复同候选，原历史与未报告的中断用量保留。见[本轮报告](runs/exp2/single_policy_astra_xhigh/REPORT.md)、[完成回执](runs/exp2/single_policy_astra_xhigh/completion.json)、[恢复及最终计费口径](runs/exp2/single_policy_astra_xhigh/incidents/tray_design_http502/recovery_completed.json)。Exp1、M0 和原三组对照未改变；本轮与 APPL 的差异还包括切分、模型数量和总训练量，不是单独移除 runtime agent 的消融。

**执行已收尾：** 原协调器实际执行 120 回合，[补充分配](runs/exp2/single_policy_astra_xhigh/evaluation_supplement/completed.json)执行 150 回合，[提前调度](runs/exp2/single_policy_astra_xhigh/evaluation_tail/completed.json)执行 30 回合；原队列随后仅读取已有结果，物理总数仍为 300。[逐回合实际进程核验](experiments/exp2/paper/physical_execution_audit.json)已通过。三个协调器和 300 个策略 worker 均正常退出，使用 GPU 0/1/4/6/7，未超过五张；[收尾核验](experiments/exp2/paper/resource_release.json)未发现本实验遗留 GPU 进程，无关进程保留。报告整理没有新增 API、候选训练或仿真。

**最终核对：2026-09-18。额外的 GPT-6 Astra / xhigh 五任务 APPL 实验已完成。** 全部 36 个 API policy 已训练并通过部署检查；本轮取得 **300/300 个完整结果：197 成功、103 任务失败、0 未知**。ID **124/150（82.7%）**，位置 OOD **73/150（48.7%）**。[最终结果与三方法对比](runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/REPORT.md)、[证据分析](runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/ANALYSIS.md)、[完成回执](runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/completion.json)、[配对视频](runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/paired_examples.html)、[全部 300 段选定回放](runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/replays.html)。

结果由原 211 个完整回合、首次恢复的 68 个完整回合和最后授权续跑的 21 个回合组成。所有原尝试、HTTP 中断、API 原文和费用记录保留；32,209 个已有文件 hash 未变，新增轨迹及视频全部通过独立核验。补测没有新增训练或修改 policy/prompt；全轮次正式训练仍为 720,000 次更新。[执行核验](runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/evaluation_audit.json)、[视频核验](runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/videos.json)、[进程退出与 GPU 释放](runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/resource_release.json)。原 DP 和 GPT-5.5/high 对照直接复用，后者原有的 1 个未知结果仍单列。

双块分拣和托盘的 OOD 成功数相较旧 APPL 从 4/30 提升至 18/30、23/30；抽屉退步（ID 8/30，OOD 7/30），交换任务 OOD 也下降（8/30）。本次同时更换 API 模型、推理档位和 API 生成的切分/策略库，训练总量也不同，不能把差异单独归因于模型或 xhigh。103 个失败均由 API 在 437–2408 步主动结束；188 个成功发生在 1500 步内，另 9 个在 3000 步内，无成功超过 3000 步。Exp1、冻结 M0 和既有对照保持不变。今后新实验使用 xhigh，Exp1 相关实验使用 max。

**抽屉退步专项复核（2026-09-18）：** 对旧、新 APPL 的 120 个完整抽屉轨迹做只读分析。新 ID 30/30 都打开过抽屉，但只有 8/30 抬起蓝块；这 8 个全部完成。首次红块搬运选 h03 的 26 个回合中，16 个因抽屉回缩到 0.26 m 以下触发 API 停止条件，仅 1 个随后成功。定位到动作、交接和恢复的共同问题，尚未通过消融隔离因果；归一化 payload 和 DDPM100 一致。新增 [专项报告](experiments/exp2/analysis/drawer_astra_regression/REPORT.md)、[逐回合证据](experiments/exp2/analysis/drawer_astra_regression/evidence.json)、[配对状态曲线](experiments/exp2/analysis/drawer_astra_regression/seed6302_timeline.png)。0 新 API、0 训练、0 仿真，冻结结果保持不变。

### Astra 原始尝试与中断恢复历史

**2026-09-17 核对：** 额外的五任务 APPL `gpt-6-astra/xhigh` 轮次已完成全部新切分和 **36/36 个 API policy 提交、训练及部署检查**，共 720,000 次正式更新。原定 300 次评估尝试全部结束，**211 个完整结果（141 成功、70 任务失败），89 个 API 中断结果未知**，尚未取得 300 个完整结果。所有评估进程已退出，原始记录保留。[结果与比较](runs/exp2/M1_astra_xhigh/REPORT.md)、[证据分析](runs/exp2/M1_astra_xhigh/ANALYSIS.md)、[原轮次退出回执](runs/exp2/M1_astra_xhigh/evaluation_supervisor/process_result.json)。

评估中断为 82 次 HTTP 503、2 次 HTTP 429、5 次 HTTP 502；其中 79 次在 0 步发生，10 次保留物理前缀。211 个完整回合和全部 89 条中断前缀已核验原始 API 选择、字面停止条件、冻结模型、配对初态与隐藏预算；300 段原始帧视频通过核验，其中 79 段仅有初始帧。[完整执行核验](runs/exp2/M1_astra_xhigh/evaluation_audit.json)、[中断核验](runs/exp2/M1_astra_xhigh/incidents/evaluation_outage/audit.json)、[配对回放](runs/exp2/M1_astra_xhigh/paired_examples.html)、[全部回放](runs/exp2/M1_astra_xhigh/replays.html)。

**2026-09-17 已执行授权恢复，但服务仍不可用：** 用户回复“ok，把实验做完”后，按[89 项精确方案](runs/exp2/M1_astra_xhigh/incidents/evaluation_outage/retest_proposal.json)执行了首个补测 `buffer_swap / ID / 20122`。第 1 次实际 `gpt-6-astra/xhigh` 请求再次返回 HTTP 503，0 个物理步、0 个新增完整结果；首请求、prompt 与原初态逐字/逐项一致。调度按约定停止，另 88 项未启动，当前没有后台评估进程。24,318 个原文件 hash 未变，36 个模型与冻结科学设置复核通过。原矩阵仍为 211 个完整结果和 89 个未知，不能宣称实验已做完。[授权](runs/exp2/M1_astra_xhigh/incidents/evaluation_outage/retest_authorization.json)、[恢复报告](runs/exp2/M1_astra_xhigh/evaluation_recovery/REPORT.md)、[核验](runs/exp2/M1_astra_xhigh/evaluation_recovery/validation.json)、[未启动队列](runs/exp2/M1_astra_xhigh/evaluation_recovery/execution.json)。当前阻塞是 API 服务；88 个尚未启动项的授权保留，20122 已用掉本方案的一次补测，未自动再次尝试。此次仅增加 1 个明确失败的 API 请求，未返回用量，费用未知。

当前可确定：抽屉 ID 明显退步（6 次成功，即使 9 次未知全部成功仍低于旧轮次 29/30）；双块和托盘 OOD 的已确认成功数已超过旧轮次。70 个真正任务失败全部由 API 在 437–2408 步主动结束，没有触及 5000 步上限。完整配对计数、因果限制与计算量差异见[分析](runs/exp2/M1_astra_xhigh/ANALYSIS.md)，不能把 unknown 当成任务失败或声称已获得最终成功率。

四项授权设计恢复已全部完成，原失败回执及外层记录错误分别保留，没有额外候选或正式训练重跑。[恢复与模型完整核验](runs/exp2/M1_astra_xhigh/authorized_recovery/completion.json)、[模型与文档入口](runs/exp2/M1_astra_xhigh/POLICIES.md)。全部切分及设计共 663 次请求：660 次正常消费、3 次原 HTTP 中断；加上部署后共 3,566 次请求：3,474 次消费、92 次 HTTP 中断。未报告的中断成本保留为未知。[设计账本](runs/exp2/M1_astra_xhigh/design_completion.json)、[全轮次计费字段](runs/exp2/M1_astra_xhigh/results.json)。

[切分核验](runs/exp2/M1_astra_xhigh/segmentation_summary.json)与[恢复前调用账本](runs/exp2/M1_astra_xhigh/design_accounting_before_recovery.json)确认实际请求和正常响应均为 `gpt-6-astra/xhigh`；恢复前 621 次请求中 618 次正常消费、3 次 HTTP 中断，无待消费请求。完整覆盖原始示范，没有手工修改 API 输出。复用原示范、归一化、DP 对照和全部 300 个配对初态。[新轮次规范](experiments/exp2/ASTRA_XHIGH.md)、[输入核验](runs/exp2/M1_astra_xhigh/preparation.json)。长期规则：今后 Exp2 和其他新实验使用 `xhigh`，Exp1 相关实验使用 `max`；历史记录保持原值。

最新资源授权允许同时最多五张卡，当前使用池为 1/2/3/5/7，恢复前已重新检查占用；0/4/6 上无关进程保留。[恢复占用记录](runs/exp2/M1_astra_xhigh/incidents/recovery_gpu_occupancy.json)、[五卡修订](runs/exp2/M1_astra_xhigh/allocation_5gpu/amendment.json)、[多任务补位](runs/exp2/M1_astra_xhigh/ready_training_additions/admission_amendment.json)、[并行准备](runs/exp2/M1_astra_xhigh/preparation_overlap/schedule.json)、[并行训练](runs/exp2/M1_astra_xhigh/training_overlap/schedule.json)。资源调度使用原有 package/checkpoint 锁，没有增加候选、正式更新或改变训练配方。

三例原 API 设计中断发生时均无代码输出：托盘 `deliver_block__h02` 首请求 HTTP 429；双块分拣 `deliver_disengage__h01` 在六次读取后 HTTP 502；拆叠分拣 `acquire_lift__h02` 在七次读取后 HTTP 502。用户已[授权各恢复一次](runs/exp2/M1_astra_xhigh/incidents/design_recovery_authorization.json)，旧调用仍计入原预算，原尝试完整保留于 `retained_design_attempts`。[429 事件](runs/exp2/M1_astra_xhigh/incidents/tray_deliver_block_h02_429/incident.json)、[分拣 502](runs/exp2/M1_astra_xhigh/incidents/sort_deliver_disengage_h01_502/incident.json)、[拆叠 502](runs/exp2/M1_astra_xhigh/incidents/unstack_acquire_lift_h02_502/incident.json)。后续评估通过独立的四请求并发门限排队，无自动重试；8 进程竞争验证通过且未调用 API，[排队机制回执](runs/exp2/M1_astra_xhigh/api_capacity/amendment.json)。

托盘恢复已完成 API 提交和接口检查，但随后外层记录代码因重复 `status` 参数退出；这发生在成功提交之后，没有新 API 中断。原脚本、退出记录与修正前回执保留，经核验原 API 提交、源码归属及成功检查后，仅补正外层状态；托盘模型随后完成训练。其余两个授权恢复通过明确子集入口执行并完成，没有再次调用托盘 API。[记录错误与核验](runs/exp2/M1_astra_xhigh/incidents/tray_recovery_recording_error/incident.json)、[剩余恢复进程](runs/exp2/M1_astra_xhigh/authorized_recovery/network_remaining/)。

另定位并修复了 `buffer_swap/buffer_red__h02` 的重载检查误报：权重完全相同，但 EMA 与重载模型的 `requires_grad` 设置不同，DDPM100 输出相差约 0.000101；统一设置后差异为 0。原设计耗尽 36 次 API 调用和 5 次检查，其中四次检查遇到此误报。两次只读复现共 10 个 DDPM100 采样批次、0 次 policy 更新、0 次 API、0 步仿真。32 个无关模型全部完成后，已应用仅改重载验证一行的修正；训练、采样和部署行为保持原实现，原源码和回执保留。框架从 `93460e…` 变为 `6af981…`，严格检查只允许这一处差异。[诊断](runs/exp2/M1_astra_xhigh/incidents/buffer_red_h02_reload/incident.json)、[具体修正](runs/exp2/M1_astra_xhigh/incidents/buffer_red_h02_reload/proposed_fix.patch)、[版本回执](runs/exp2/M1_astra_xhigh/incidents/buffer_red_h02_reload/framework_revision.json)、[验证](runs/exp2/M1_astra_xhigh/incidents/buffer_red_h02_reload/correction_validation.json)。用户已[授权同会话有限延长](runs/exp2/M1_astra_xhigh/incidents/buffer_red_h02_reload/recovery_authorization.json)；实际用 2 次新调用、1 次接口检查完成 API 提交。原 36 次响应、工具输出和完整对话前缀逐项保持一致，开发者未改 policy 文件。[恢复回执](runs/exp2/M1_astra_xhigh/incidents/buffer_red_h02_reload/continuation.json)。

### 原始五任务 scale-up（GPT-5.5 / high）

**最终核对：2026-09-17。五任务 scale-up 的训练、600 次预定测试尝试与独立核验均已结束。** DP 有 300 个完整结果；APPL 有 299 个完整结果和 1 次临时 HTTP 502 中断，后者最终结果仍未知、未补测。全部 600 个进程已退出，599 个完整轨迹、中断前缀和 600 段视频通过核验，实验 GPU 已释放。[五任务结果](runs/exp2/M1_scaleup/REPORT.md)、[完整分析](runs/exp2/M1_scaleup/ANALYSIS.md)、[完成回执](runs/exp2/M1_scaleup/completion.json)。

新任务 DP 的 ID 成功为 2/120；APPL 的 ID 结果明显更好，但位置 OOD 仍有大量失败。所有 DP 成功均在 1500 步内；APPL 在 1500 步之后新增 22 次完成，其中 3 次超过 3000 步。这是同一条轨迹的前缀比较，完整的逐任务数字、不确定性、失败证据及计算量差异见上方分析。

- **范围与输入：** 原抽屉加分拣、临时区域交换、拆叠分拣、开放托盘装载，共五任务；每任务 12 条示范，DP/APPL 各 30 ID + 30 位置 OOD，共 600 次预定测试。[规范](experiments/exp2/SCALEUP.md)、[48 条新示范核验](runs/exp2/M1_scaleup/data_validation.json)。初版采集器腕部限位问题及统一修订保留在[采集回执](runs/exp2/M1_scaleup/collector_v1_incident.json)，未筛选训练 seeds。
- **训练与 API 所有权：** 新增 4 个 naive DP 各 60,000 更新、33 个 API prior 各 20,000 更新，合计 900,000 次正式更新；原抽屉模型直接复用。37 个真实部署检查通过，全部模型在正式测试前[全局冻结](runs/exp2/M1_scaleup/study_freeze.json)。切分、heuristic、prior 源码和交接语义由真实 API 生成，33 份新提交逐字来源核验通过。[训练回执](runs/exp2/M1_scaleup/training_completion.json)、[51 个 API policy 入口](runs/exp2/M1_scaleup/POLICIES.md)。
- **执行规则：** 双方使用同一配对初态、几何目标、DDPM100 / 执行 8 和 5000 物理步上限；API 看不到总预算或剩余步数。API 自己选择 policy、时长、数值停止条件和 notebook。共享尺度按每任务完整训练示范拟合。环境和 diffusion seeds 固定，API 生成未设 seed；APPL 与 DP 的总训练量不匹配。
- **API 模型复核（2026-09-17）：** 当前五任务的全部切分 148 次、51 个 policy 设计 708 次、正式部署 5049 次请求均为 `gpt-5.5 / high`；正常响应也均标识 `gpt-5.5`，唯一 HTTP 502 无模型响应。Exp1 的冻结配置及 1150 条调用报告为 `gpt-6-astra / max`，与 Exp2 不同。[全量调用字段审计](experiments/exp2/analysis/api_model_audit.json)另含历史轮次；这是请求/响应身份复核，不独立证明本地代理后端权重身份，未新增 API、训练或仿真。
- **中断与审计：** 交换任务 APPL／ID 20116 在第 650 步、第 11 次 API 请求遭遇上游 HTTP 502；最终结果及失败请求费用未知，未重试。[原始中断](experiments/exp2/analysis/transport_incident_20116.json)、[待授权的单次补测方案](experiments/exp2/analysis/transport_retest_proposal.json)。[独立执行审计](experiments/exp2/analysis/scaleup_execution_audit.json)已覆盖全部 599 个完整回合及中断前缀；冻结模型、API 原始选择、数值停止条件、配对初态和[600 段视频](runs/exp2/M1_scaleup/video_validation.json)全部通过核验。
- **实现与保全：** 四个新 DP 的训练窗口、尺度、动作变换和同 EMA 的 DDPM100 采样已与原 M0 做[只读等价核验](experiments/exp2/analysis/naive_equivalence_audit.json)。[85,576 个受保护文件](experiments/exp2/analysis/protected_preservation.json)及[120,298 个输入/切分/配置文件](experiments/exp2/analysis/frozen_inputs_audit.json)的 hash 核验通过。Exp1、原 M0 源码、模型、判定器和旧结果保留。
- **资源与观看：** 用户授权 0–7 中同时最多四张、每卡可多任务；本轮正式测试使用 1/2/3/7，现已释放。[资源分配](runs/exp2/M1_scaleup/allocation/amendment.json)和[DP 槽位交接](runs/exp2/M1_scaleup/allocation/dp_slot_handoff.json)仅改变调度，预定矩阵仍为 600 次。[五任务配对回放](runs/exp2/M1_scaleup/paired_examples.html)固定展示每任务首个 ID/OOD seed；[全部回放索引](runs/exp2/M1_scaleup/replays.html)已生成，并包含中断前的可用影像。

初始研究已整理为 [M1_initial_test](runs/exp2/M1_initial_test/README.md)，当前入口指向 [5000 步结果](runs/exp2/M1_initial_test/inference_5000/REPORT.md)。原 1500／3000 步研究和设置已迁入 [archive/Exp2_M1_initial_tests](archive/Exp2_M1_initial_tests/README.md)，4,615 个文件与原配置逐字保全，历史路径保留兼容链接。[迁移回执](experiments/exp2/M1_INITIAL_LAYOUT.json)。

## 已完成的初始研究与历史摘要

**最新完成核对（2026-09-16）：** [隐藏 5,000 步上限的反馈修正轮次](experiments/exp2/INFERENCE_5000.md)已完成，配置 [m1_v2_5000.json](experiments/exp2/configs/m1_v2_5000.json)。相同 ID 初态复测 **4/5 成功**，训练初态诊断 **1/1 成功**；ID 成功步数为 6200：1,529、6202：1,453、6203：1,197、6204：1,058，6201 在 2,070 步由 API 主动结束失败。新轨迹前 1,500 步成功 3 个、前 3,000 步成功 4 个，没有回合用到 5,000 步上限。复用全部 18 个模型，无新增训练。API 自己选择 policy、调用时长、数值停止条件及 notebook；框架逐步检查其原始条件，实际发生 30 次提前返回。94 次 API 请求均成功消费，逐条核验未向 API 暴露总预算或剩余步数；输入峰值为 56,969 token。39 项测试通过、8,294 个物理步完成核验，原 3,747 个文件 hash 保持不变。[报告](runs/exp2/M1_v2/inference_5000/REPORT.md)、[分析](runs/exp2/M1_v2/inference_5000/ANALYSIS.md)、[回放](runs/exp2/M1_v2/inference_5000/visualizations/README.md)、[完成回执](runs/exp2/M1_v2/inference_5000/completion.json)。

6201 与上一轮成功轨迹的前 400 步动作/状态完全一致，随后 API 将 h03 换成 h02；上一轮继续 h03 到第 600 步已经打开抽屉。这轮切换后恢复未成功，剩余 2,930 步未使用。因此问题包含继续/切换判断与偏移状态恢复，不能归因于总预算不足。本轮同时修改了反馈、prompt、上下文、预算可见性和 API 请求上限，且复用已分析过的 seeds；4/5 是开发复测结果，不是新的独立泛化估计，也不是单因素因果结论。成功定义仍为三个几何目标，部分成功终态蓝块仍可能被夹持，不代表已验证释放或持续稳定。[逐回合对照](runs/exp2/M1_v2/inference_5000/analysis_evidence.json)。

**此前 3,000 步追加轮次（核对 2026-09-16）：** M1_v2 采用 [3,000 物理步配置](experiments/exp2/configs/m1_v2_3000.json)，复用原 18 个模型、相同 6 个初态与原英文 prompt。5 个 ID 尝试为 **2 成功、2 任务失败、1 上下文超限中断**；示范初态诊断失败。新轨迹前 1,500 步成功 1 个，6201 在第 1,782 步新增成功，说明额外时间对这一轨迹有用，但尚不能声称总体成功率提高。6202 在 2,740 步的第 30 次 API 请求返回 HTTP 502（上下文超限）；未自动重试，其 3,000 步最终结果仍未知，不能把完成的 2/4 与原 2/5 直接比较。[报告](runs/exp2/M1_v2/budget_3000/REPORT.md)、[分析](runs/exp2/M1_v2/budget_3000/ANALYSIS.md)、[上下文诊断](runs/exp2/M1_v2/budget_3000/context_limit_incident.json)、[回放](runs/exp2/M1_v2/budget_3000/visualizations/README.md)。31 项测试通过、12,825 个记录动作逐步核验；2,545 个原轮次文件保持原 hash，训练更新为 0。追加 API 请求 131 次（130 消费、1 明确失败，失败请求费用未知）。[状态回执](runs/exp2/M1_v2/budget_3000/completion.json)明确记录全部尝试已终止，但完整五个最终评估结果尚未齐全。

2026-09-16 追加只读[失败与可复现性复核](runs/exp2/M1_v2/budget_3000/FAILURE_REVIEW.md)：六个回合的原始/实际动作及状态在前 300–360 步逐项一致，首次轨迹分叉均与 API 切换到不同 policy 对齐；两轮首条 API 消息仅 remaining_steps 不同，环境和 DP seed 相同，但 API 未设生成 seed。6200、6202 和诊断 1000 在单次调用内部曾成功抬高蓝块，调用结束前又降下，当前 API 只收到调用边界状态，可能错过交接时机。完整[对照数据](runs/exp2/M1_v2/budget_3000/failure_review.json)保留事实与未验证反事实的区分；本次复核无代码修改、新 API、训练或仿真。

### 原 1,500 步轮次

核对日期：2026-09-16。**M1_v2 已完成：18/18 个 API-authored policy 各训练 20,000 步，独立 ID 2/5、示范初态诊断 1/1 成功。** 完整训练示范统一归一化、约 7.9 倍的切分重叠、每条 heuristic/policy 的 API 交接文档均已落实；冻结后完成全部六个回合并核验 7,799 个物理步。其中 ID 6203 在一次高速度恢复动作后瞬时达到三个几何目标，按约定计成功，但不能视为稳定终态；详细限制见[分析](runs/exp2/M1_v2/ANALYSIS.md)。上一轮保留为 **M1_v1**（独立 ID 0/5），其 checkpoint 已按授权删除。Exp1 与 M0 保留。这是唯一的人工进度摘要，详细数字以链接的报告和原始记录为准。

| 工作线 | 当前状态 | 证据入口 | 待完成事项 |
| --- | --- | --- | --- |
| Exp1 正式实验 | 已完成；现有报告记录训练、dev、hidden test 均完成全部正式槽位 | [完整报告](experiments/experiment1/EXPERIMENT1_REPORT.md)、[模型结果](experiments/experiment1/reports/model_results.csv)、[完成记录](experiments/experiment1/reports/completion_manifest.json) | 保全冻结代码、数据、选择与结果；没有待补跑槽位 |
| Exp2 M0 | 保留 12 条示范 / DDPM100 / 执行 8：开发 2/5，独立确认 4/15 | [最终 M0 报告](experiments/exp2/M0_REPORT.md)、[保留模型与评估](runs/exp2/m0/)、[诊断归档](archive/Exp2_M0DP/README.md) | 本轮保持代码、模型与原判定；不将新任务定义追溯应用于旧指标 |
| Exp2 M1_v2 原轮次 | 已完成训练与 1,500 步测试；独立 ID 2/5，示范初态 1/1；尚不稳健 | [结果报告](runs/exp2/M1_v2/REPORT.md)、[分析](runs/exp2/M1_v2/ANALYSIS.md)、[逐步审计](runs/exp2/M1_v2/evaluation_audit.json)、[完成回执](runs/exp2/M1_v2/completion.json) | 保留冻结策略库与原评估证据；3,000 步追加测试的中断单独记录在上方摘要 |
| Exp2 M1_v1 | 15/15 policy 各 20,000 步；独立 ID 0/5、示范初态 0/1。旧 checkpoint 已授权删除，原核验结论为历史记录 | [报告](runs/exp2/M1_v1/REPORT.md)、[分析](runs/exp2/M1_v1/ANALYSIS.md)、[版本说明](runs/exp2/M1_v1/VERSION.md)、[删除回执](runs/exp2/M1_v2/setup/v1_checkpoint_deletion_receipt.json) | 保全源码、API 来源、训练记录和评估轨迹 |
| Exp2 示范拆分 | M1_v2：6 类技能、72 个片段、18 条 prior；完整覆盖、5,220 个重叠动作，30 次 API 请求全部消费 | [新报告](runs/exp2/M1_v2/segmentation/REPORT.md)、[API 核验](runs/exp2/M1_v2/segmentation/validation.json)、[manifest](data/exp2/processed/M1_v2/manifest.json) | 保全 API 提交，不手工修改边界或 prior；M1_v1 切分保留供追溯 |

M1_v2 的 28 项框架测试、18 个 checkpoint 契约核验和 18 个真实部署 worker 检查均通过。每个 policy 保存 API-authored HANDOFF.json、示范交接/结束状态、共享归一化 hash，以及实际测试调用前后状态；14/18 个 policy 在整任务测试中被 API 调用，其余明确标记为未调用。本轮 360,000 次正式更新、36 次接口检查更新；API 共 371 次请求（切分 30、设计 255、推理 86），全部消费，无自动重试或未决结果。[保全核验](runs/exp2/M1_v2/setup/preservation_check.json)确认 85,576 个受保护文件与 12,599 个 M1_v1 切分文件未变。

旧轨迹用新尺度重新编码后，红块合法落点的 38.36 降为 0.92，原关节速度峰值对应坐标的 58.86 降为 5.76；这些旧轨迹在新尺度下的全局最大值为 11.39，[完整记录](runs/exp2/M1_v2/setup/normalization_reencoding.json)。新轨迹仍出现恢复动作造成的速度尖峰，最高重新编码值 17.62；不能把共享尺度修复理解为消除了真实状态偏移。本轮同时改了切分、prior、归一化和交接信息，且使用不同的 ID seeds，不作为单因素因果消融。

2026-09-16 已将六个回合的原始相机帧整理为[可播放回放](runs/exp2/M1_v2/visualizations/README.md)，另有成功、空抓和预算耗尽的交互对照。原始分辨率 128×128，每 20 控制步一帧；所有源图 hash 保持不变，未新增 API、训练或物理仿真。[来源清单](runs/exp2/M1_v2/visualizations/manifest.json)记录帧与视频对应关系。

## M1_v1 已完成轮次与前期整理记录

以下训练、核验和 GPU 调度描述属于 M1_v1 的历史执行，不能视作 M1_v2 已完成。M1_v1 已迁入同名目录，旧路径保留为别名；31 个 checkpoint 共 11,075,462,809 字节已按清单删除。

新流程按用户最新指令直接消费保留切分，每条 heuristic 对应独立源码、API prior 文档、训练和 checkpoint 目录。开发者提供框架；API 自己实现科学机制并在推理时根据 prior 选 policy。初始使用 GPU 4/7；确认 GPU 6 原有进程已自行结束后，在授权的 4–7 范围内加入 GPU 6 执行最后一个模型，保留所有无关进程。推理测试采用三个几何目标，M0 判定不改。依赖与命令见 [Exp2 入口](experiments/exp2/README.md)。

本轮共 300,000 次正式训练更新、30 次接口检查更新；API 200 次设计请求、87 次推理请求，均有完整回执，无自动传输重试。五个独立 ID 回合中，三个未完成红块抓取交接，两个完成红块放置后未抓起蓝块；所有回合的蓝块均停留在桌面高度，失败不是额外末尾判定导致。逐动作判定与冻结模型调用核验见[测试审计](runs/exp2/prior_policies_20260916/evaluation_audit.json)。分技能归一化仍有窄范围敏感性：最大归一化值约 58.86；蓝块技能中红块合法落点偏移 2.08 cm 被编码为约 38.36。它与交接偏移共同构成下一步应检验的假设，尚未做因果消融；详见[失败分析](runs/exp2/prior_policies_20260916/ANALYSIS.md)。26 项框架测试、真实模拟器零动作检查及所有 API 源码逐字审计通过。

[最终保全核验](runs/exp2/prior_policies_20260916/preservation_final.json)确认 85,576 个受保护文件及 12,599 个保留切分文件未变；[完成回执](runs/exp2/prior_policies_20260916/completion.json)区分了实现/训练/测试完成与整任务成功尚未得到证明。

2026-09-16 最新示范处理：按用户澄清，以[三个几何目标](experiments/exp2/configs/demonstration_goals.json)判断本轮完成；英文 prompt 鼓励有依据的少量重叠、逐条选择边界。API 27 次请求、1 个方案版本，检查中补读证据后显式提交。5 类技能各 3 条 prior，覆盖全部 12,809 个原始动作、排除 0 个、重叠 660 个；新终态 12/12 满足三个目标。前四类技能仍采用共同时间点，末类按轨迹实际终点结束；不能把末尾长度差异当作内部边界全面自适应的证据。12,599 个新文件和两版旧产物 hash 核验通过；24 项 Exp2 测试及 8 个完成条件检查通过，训练更新/模拟器调用均为 0。

对前次审阅的澄清：上一版排除 569 个尾段动作，但在当时传入的“抽屉打开、红块在垫子、蓝块在抽屉”目标下，截断点同样 12/12 达标；先前额外使用释放/离物/持续 10 步要求，属于输入目标与审阅标准不一致，不能据此认定 API 未完成所收到的任务。新旧比较见[目标核验](runs/exp2/demonstration_processing_20260916_overlap/goal_validation.json)。该定义现按最新授权用于示范处理及新 prior policy 测试；已冻结 M0 判定和历史指标保持不变。

按用户最新清理授权，旧两版 processed 示范、对应旧运行目录和旧 M1/M2 专属产物已按精确清单删除，只保留当前 overlap 版本。旧 library/formal/deployment 三个独占编排模块已删除，共享功能保留；[删除回执](runs/exp2/prior_policies_20260916/cleanup/deletion_receipt.json)及[源码回执](runs/exp2/prior_policies_20260916/cleanup/retired_m1_m2_receipt.json)记录实际范围。自动审批曾拒绝较广共享源码清理，随后收窄为经过依赖核对的专属删除，未执行被拒绝的广泛修改。所有新 policy 内容和文档仍由 API 生成。

M0 活跃流程为 `src/appl/dp_baseline` → `experiments/exp2/configs/m0.json` → `runs/exp2/m0` → [最终报告](experiments/exp2/M0_REPORT.md)。保留原始 12 条数据、60,000 步最后 EMA、DDPM100 和执行 8；已实现非零成功，尚不稳健。

首次修复、120 条数据/增量动作/DDPM200 等设置研究均已完成并移入 [Exp2_M0DP](archive/Exp2_M0DP/README.md)。120 条模型独立确认 5/15，相比基线 4/15 未提供可靠提升证据，因此不替换当前基线。历史模型、失败轨迹、源码、额外示范和费用记录完整保留。2026-09-16 整理只改变入口与代码组织，未新增训练、API 请求或模拟器动作；校验记录见 [M0_LAYOUT.json](experiments/exp2/M0_LAYOUT.json)。

当前资源范围：Exp1 物理 GPU 0–7，依据 [2026-09-13 资源修订](experiments/experiment1/execution_allocation.json)；Exp2 主实验按其[配置](experiments/exp2/configs/main.json)限制为物理 4–7。调度前检查实际占用。下方旧卡号、PID、运行中描述及恢复命令只代表历史时点。

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
