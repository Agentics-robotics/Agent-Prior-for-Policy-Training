# flip_egg_training_v1：API 实现与正式训练

核对日期：2026-09-22。

另已通过正式 `PolicyProcess` 独立进程调用：真实录制输入下连续两次返回
`active`，随后中断返回 `interrupted`；未绑定的执行器明确拒绝执行。
见[独立调用回执](../runs/flip_egg/training_v1/host_worker_check/result.json)。

本轮保留10,931条准备样本，实际训练划分为11条示范/5,364例，验证4条/2,781例，
留出5条/2,786例。每条轨迹抽取最多128个固定位置评估，留出集线速度分量平均绝对
误差约0.00468 m/s、角速度约0.01037 rad/s；零运动基准分别约0.01033和0.02016。
这些轨迹此前已供设计阶段检查，不是完全未接触的外部测试；这里只测短时实测
动作的离线拟合，没有闭环机器人或翻转成功评估。

原cut拟议的“铲具几何＋独立分割/关键点模型”在实现中改为联合视觉编码器及
实测末端局部运动目标，API说明缺少工具关键点/几何标注。没有声称学出了
铲—蛋显式几何状态或独立感知模型。最终checkpoint取第2,000步。

完成的实现修订：0；实际 API 调用 71 次，均为 Astra/xhigh。
API 费用估算 $13.493657，非实际账单。未知调用预留费用 $0.000000。
Input、缓存读取/写入、output 和 reasoning tokens 按任务单独记录在
[完整 API usage](FLIP_EGG_TRAINING_V1_API_USAGE.md)，不含此前 cut 的费用。

本轮承接原 cut/prior；没有运行新版 cut prompt。预处理、模型、训练和调用代码由 API 提交。

| Policy | 架构 | 可训练参数 | 准备的可用例 | 实际更新 | 优化器 | 重载误差 |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| egg_interaction_v1 | causal_visual_gru_mixture_behavior_cloning | 278903 | 10931 | 2000 | AdamW | 0.0 |

准备 10931 个衍生训练例；唯一原始锚点 10931 个。两者不能混同为独立示范数量。
覆盖 20/20 个衍生片段。
表中可用例按 policy_weight>0 统计；API sampler 可能另划训练/验证用途，该数不等于实际用于梯度更新的独立样本数。实际划分见 API 数据与采样代码。

以下为 API 对实际数据的评估与限制：

**coverage_assessment**：Train the exact checked package. Preparation produced 10,931 unique strided source anchors from all20 original interaction intervals, 368-982 per episode. Split remains whole-episode11/4/5: train5,364, validation2,781, test2,786. Static counts645/397/324 and high-angular-motion765/288/411 support motion-stratified evaluation and capped static sampling. All45,730 original rows accounted as retained context; original32,737 possible dense anchors reduced by documented stride3 plus last anchor, not success filtering or new cuts. No candidates rejected by pose, dt, age, width or motion audit. Acquisition and392 intervention-tail/context rows plus51 normal terminal-buffer rows remain loss-free as in the source prior. Inspected actual prepared RGB montages for16095801,16200101,16295601,16320101,16411901,16463201,16534901,16553801: third-view crops retain egg/pan/tool relationships and wrist supplies complementary but sometimes cropped/occluded support evidence. This is20 related same-setup recordings, not10,931 independent successful interactions.

**data_recovery_assessment**：Original source plan and recording metadata are retained verbatim in preparation artifacts, with original interval transcription in ORIGINAL_CUT.json. Recovered task-dependent motion supervision by replacing unsupported measured-blade/PnP dependency with recorded EE-local rates and jointly trained raw-RGB perception. No masks, tool landmarks, metric egg pose, force labels or success labels were fabricated. Model intentionally does not provide the originally proposed semantic geometry service. Negative third-image ages are common but all recorded ages were inside the declared[-.02,.15]s skew/age tolerance; no sample re-pairing, clock shifting or zero-motion substitution occurred. No one-view recovery was needed in this prepared dataset; explicit masks remain available online. Original paired samples and device-reset caveat retained. The32-probe-per-episode near-duplicate audit found no strict matches requiring whole-episode reassignment; this is not exhaustive perceptual deduplication. There is no extra tail evidence to recover for autonomous control without human-intervention leakage.

**training_rationale**：Actual target dt range .0971799-.1510472s, median .1000144s, is consistent with i+3 short achieved-motion conversion.129 reconstruction probes had max7.56e-10m and1.02e-9rad error, verifying numerical frame/units conversion only. The checked278,903-parameter joint visual-GRU three-mode model passed differentiable finite nonzero-gradient and predict[2,12] checks with zero optimizer updates. Enough independent episodes and variation remain for a bounded prescribed run with train-only episode-balanced sampling, static cap, correction/release-related angular weighting, joint perceptual training, validation-selected checkpoints and untouched test for final open-loop metrics. Select by equal-episode validation loss, at most2,000 updates with explicit early stopping. Do not equate lower validation loss or interface success with robot success; report physical-unit MAE/zero baselines and motion/progress strata separately. No training search or leave-block-out retraining claimed.

**limitations**：Training is justified only as known-setup held-tool achieved-motion imitation. Missing measured tool/pan geometry, grasp/gripper commissioning, retention/slip validation, exposure synchronization, swept-collision and force-limited contact-controller binding remain physical deployment gates. End-to-end EE-coordinate representation lacks original metric blade/pan invariance and does not provide executor semantic geometry; significant tool/grasp/camera changes are unsupported. Images and width are not force/contact truth. No verified task-success labels, material/heat diversity, physical execution or autonomous off-pan/drop recovery. Initial scene goal persists but trained model is not a verified face-change classifier. Predictive spread is not a safety/OOD certificate. Planned calling fixtures use real source inputs with explicitly synthetic permissive authorization/controller acknowledgements; full reload/calling outcomes and trained performance are still unmeasured at submission.

实际训练权重已更新，checkpoint 重载和 API 提供的调用案例通过。调用案例是接口证据，不是真机成功或独立泛化评估。

- [API 调用目录](../runs/flip_egg/training_v1/package_00/source/POLICY_CATALOG.md)
- [API 调用说明](../runs/flip_egg/training_v1/package_00/source/CALLING.md)
- [API 模型/训练配置](../runs/flip_egg/training_v1/package_00/source/package.json)
- [覆盖与质量记录](../runs/flip_egg/training_v1/package_00/prepared/metadata.json)
- [训练库](../runs/flip_egg/training_v1/library.json)
- [完成核验](../runs/flip_egg/training_v1/completion_receipt.json)
