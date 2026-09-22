# push_training_v5：API 实现与正式训练

核对日期：2026-09-22。

正式训练完成：9,641参数，800次更新，API按开发轨迹误差选择第600步权重，
实际训练及重载/调用检查耗时约6.50秒。独立 `PolicyProcess` 另通过真实记录
几何输入的调用、二维位移限幅、转换后高度/姿态保持、中断及未绑定执行器
拒绝执行检查，见[独立调用回执](../runs/push_letters/training_v5/host_worker_check/result.json)。

最终保留168例，A轨迹74例用于梯度训练，B轨迹94例用于开发与checkpoint选择；
准备数据覆盖原cut全部12个交互片段；**实际参数训练只覆盖A的6段，B的6段未用于梯度更新**。
当前没有A+B全量最终拟合，详见[划分审查](PUSH_TRAINING_V5_SPLIT_REVIEW.md)。
下表“2/2”指物化存储时合并的两条完整轨迹，
不表示原12个子区间只剩2个。每例目标窗口约0.167–1.002秒，中位0.532秒，
对应6–31行原始记录；邻近窗口相关，不能当作168次独立推。

学习目标已由原cut的接触点/方向/行程评分，改为**已接触时的二维末端位移**。
首次接触由API编写的确定性几何规则提出，planner负责到达；首次接触点不是
这个网络学出的。输入为当前可见物体占据点、目标运动及末端/历史状态，
网络不直接接收图像；图像仍用于前置几何提取。在线完整几何、接触确认与
planner/controller接入仍需部署侧提供。

第600步的B轨迹等权片段平均末端终点误差为**5.095 mm**；API计算的直接采用
目标平均平移基准为**4.764 mm**（位移截断18 mm），零运动基准为13.992 mm。
模型尚未优于这个简单目标基准。B也参与了设计与checkpoint选择，不是独立
泛化测试；该指标比较记录中的末端位移，不是物体到位或拼词成功率。
见[日志来源与数值摘要](../runs/push_letters/training_v5/training_summary.json)。

这里的单例误差是 `||预测的二维末端位移 − 记录的二维末端位移||₂`，再对B的
六个原交互组分别求均值、等权平均。例：示范向右移动10mm、预测向右5mm，
该例误差为5mm。实际单次示范位移约5–18mm，因此5.095mm对这个短推尺度并不小。
该值不测物体最终位姿；不同可行推法也可能偏离某次示范，因此不能据此换算
字母终点误差或任务成功率。当前结果仍未显示网络优于所记录的简单基准。

完成的实现修订：1；实际 API 调用 100 次，均为 Astra/xhigh。
API 费用估算 $16.283451，非实际账单。未知调用预留费用 $0.000000。
[独立 API usage 报告](PUSH_TRAINING_V5_API_USAGE.md)列出两次实现各自的调用数、
input/cache/output/reasoning tokens及费用；首次阻塞实现包含在内。

本轮承接原 cut/prior；没有运行新版 cut prompt。预处理、模型、训练和调用代码由 API 提交。

| Policy | 架构 | 可训练参数 | 准备的可用例 | 实际更新 | 优化器 | 重载误差 |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| shape_push_v1 | pooled visible-support encoder and three-component heavy-tailed displacement mixture | 9641 | 168 | 800 | AdamW | 0.0 |

准备 168 个衍生训练例；唯一原始锚点 168 个。两者不能混同为独立示范数量。
覆盖 2/2 个衍生片段。
表中可用例按 policy_weight>0 统计；API sampler 可能另划训练/验证用途，该数不等于实际用于梯度更新的独立样本数。实际划分见 API 数据与采样代码。

以下为 API 对实际数据的评估与限制：

**coverage_assessment**：TRAIN for the revised continuation problem, not the rejected physical-contact classifier. Actual preparation examined1093 source-key anchors including all47 original anchors and retained168 (74 A train,94 B development), including11 original anchors. All12 original interaction groups remain represented: A zigzag34, ring/clearance10, cluster8, legged/loop7, open-corner10, bars5; B zigzag/clearance17, loop/ring22, legged24, loop finish2, corner6, bars23. These are correlated overlapping windows in only two same-inventory episodes, not168 independent pushes. All16745 original paired rows remain context/executor evidence. I inspected all seven final contact sheets plus the preceding nine-sheet recovery review and detailed paired B1700/A8800/B5145 witnesses. The final early-A sheet no longer includes the observed merged stationary neighbor. Full-label physical contact coverage is NOT asserted: these labels are direct recorded EE displacements with inferred local visible-motion goals.

**data_recovery_assessment**：Carried out two actual new preparations rather than returning an unimplemented plan. Preserved the original cut/source_plan, completed blocked assessment, contact-recovery code/assets and all failed/journaled attempts. Revised the learned target to bypass unverified contact boundary/topology, table-clearance and physical tip-support conversion: measured T_base_ee planar differences are the action targets; bidirectional RGB corner tracking and rigid consensus infer achieved local SE2 goals. Current visible support is not called a complete actor or contact boundary, and future inliers do not replace causal input points. First recovery retained205 rows but image review caught merged moving/stationary support. A general85% rigid-hull coverage and<=12% discordant-track gate, plus explicit endpoint-age/one-frame timing uncertainty, produced the final168-row cache without an index whitelist. Final median rigid RMS0.790mm, range0.160-1.631mm; median support coverage99.22%, minimum85.16%; median70.5 rigid matches, minimum7. Differential uncertainty quantiles1.50/1.54/5.15mm include common-height sensitivity, fit disagreement and timing budget. Absolute chart uncertainty remains8mm and is not a safety certificate. Final exclusions: no resolved rigid support542, insufficient planar prefix262, high tool59, curved43, ambiguous support16, motion below uncertainty3. These exclusions describe eligibility for this primitive, not useless source evidence. Boundary ownership now follows the later original interaction group at shared starts.

**training_rationale**：The original142-row cache remains blocked and is not being trained. For the revised problem, the directly measured robot displacement supervision no longer depends on the false boundary/contact labels that caused that block. Reviewed visible rigid-motion goals with spatial/temporal consistency gates supply defensible albeit inferred hindsight conditioning. Retained examples cover all original interaction groups, including inner-loop visible motion, partial bars and open-corner cases, so a compact provisional continuation model is justified despite low independent diversity. One9641-parameter point-pooled three-mode heavy-tailed displacement model is sufficient; no diffusion trajectory model or recorded-alphabet specialists are warranted. Authorize the fixed800-update maximum AdamW schedule, original-group-balanced A sampler, B macro EE endpoint-error checkpoint selection and finite stopping rule already implemented. Report measured imitation errors and zero/goal-translation baselines separately from physical/generalization claims. Deterministic initializer and bounded executor-objective adapter complete the revised calling path, while commissioned physical execution remains intentionally blocked without bindings. Post-reload calling_cases include synthetic continuation/initialization, invalidity/reset/feedback cases and one real-source geometry fixture whose hindsight goal is explicitly a test call argument, not online future access.

**limitations**：Training and reload/calling results have not yet been measured at this readiness submission; preparation measured zero optimizer updates, finite nonzero gradients,9641 parameters and prediction[2,9]. No physical robot execution or success measurement is requested or claimed. Initial contact is now an explicit deterministic goal-displacement/accessibility heuristic, NOT learned contact selection. Full actor topology, persistent semantic association, table/tool full support/envelope calibration, contact confirmation and planner/controller guard commissioning remain deployment prerequisites; they are not needed to define the recorded EE displacement targets. Optional frozen SAM inventory yields unvalidated proposals, not an autonomous certified scene. Visual Agent owns semantic interpretation, distinct-instance matching and explicit layout/staging. Color-specific offline support extraction versus generic online perception is an unmeasured distribution shift. Fixed-orientation short planar continuation is the supported learned scope; arbitrary tilted/rotating/stacked/material-shift behavior is not established. B informed design and is development, never untouched generalization evidence. Completely unseen identities/shapes/words/layouts still require the independent tests in PRIOR.md; accepting their inputs is not success. Only push-letter assignment/data were provided; flip-egg training cannot be completed from absent recordings/task package.

实际训练权重已更新，checkpoint 重载和 API 提供的调用案例通过。调用案例是接口证据，不是真机成功或独立泛化评估。

- [API 调用目录](../runs/push_letters/training_v5/package_01/source/POLICY_CATALOG.md)
- [API 调用说明](../runs/push_letters/training_v5/package_01/source/CALLING.md)
- [API 模型/训练配置](../runs/push_letters/training_v5/package_01/source/package.json)
- [覆盖与质量记录](../runs/push_letters/training_v5/package_01/prepared/metadata.json)
- [训练库](../runs/push_letters/training_v5/library.json)
- [完成核验](../runs/push_letters/training_v5/completion_receipt.json)
