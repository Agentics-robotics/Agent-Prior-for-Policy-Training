# push_training_v6：API 实现与正式训练

核对日期：2026-09-22。

完成的实现修订：0；实际 API 调用 92 次，均为 Astra/xhigh。
API 费用估算 $19.638872，非实际账单。未知调用预留费用 $0.000000。

本轮承接原 cut/prior；没有运行新版 cut prompt。预处理、模型、训练和调用代码由 API 提交。

| Policy | 架构 | 可训练参数 | 准备的可用例 | 实际更新 | 优化器 | 重载误差 |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| shape_push_v1 | shared_point_encoder_joint_candidate_scorer | 18497 | 130 | 3000 | AdamW | 0.0 |

准备 130 个衍生训练例；唯一原始锚点 130 个。两者不能混同为独立示范数量。
覆盖 9/12 个学习片段；另有两个全轨迹执行器上下文，因此框架总表有14个片段。
独立回放API采样器并匹配checkpoint中的随机数状态、实际样本曝光总数，确认130例全部
参与梯度（A43/B87），没有单独保留B为开发集。累计47,986次样本曝光，每例155–1,109次；
这些重复曝光不增加独立示范数量。115例有完整接触/方向/距离标签，15例仅监督接触，
共有11例内孔接触。见[梯度覆盖核验](../runs/push_letters/training_v6/gradient_coverage_audit.json)。

## 实际拟合诊断

API选择最后第3,000步的原始模型，不用EMA。优化及最终诊断约22.35秒，包含加载和
调用检查的训练子进程共43.49秒；八次离线预处理检查累计81.33分钟，API阅读/写码另计。

| API训练集诊断 | 数值 | 含义 |
| --- | ---: | --- |
| 接触点差异 | 16.403 mm | 最高分候选接触点到推断示范接触标签的平均欧氏距离 |
| 方向余弦差异 | 0.05668 | 有方向标签的样本上1−方向点积，按API批次汇总 |
| 行程差异 | 2.101 mm | 有长度标签的样本上绝对长度差，按API批次汇总 |

以上为训练集弱标签拟合，**不是物体到位误差、末端控制精度或新物体成功率**。
16.4mm的接触标签差异仍显著，不能据此称模型已精确学会示范接触；同一目标可能有
多个有效接触点，但本轮没有闭环对照证明这些不同选择有效。不据此自动开启新训练。
原始诊断见[final_fit_diagnostics](../runs/push_letters/training_v6/package_00/training/shape_push_v1/evidence/final_fit_diagnostics.json)。

以下为 API 对实际数据的评估与限制：

**coverage_assessment**：Final measured preparation retains 130 unique anchors (43 A, 87 B) from 1075 considered anchors, with 15 partial-target rows and 11 inner-loop labels. Nine of twelve learned cuts contribute: A zigzag19/ring8/cluster11/legged-loop5; B zigzag14/loop-ring30/legged21/open-corner9/bar13. A open-corner, A bar finish and B loop finish contribute zero final action rows. All 16745 original rows remain context/executor evidence; no source is called task-success ground truth. Two same-inventory episodes and correlated local states do not establish independent shape or word generalization. Representative final overlays were inspected for each surviving activity group, including A1030/A2325/A4335/A5250 and B555/B1615/B3270/B5100/B6395; residual silhouette/occlusion bias remains.

**data_recovery_assessment**：Kept original cuts/prior verbatim in immutable preparation reports and documented all implementation changes. Expanded beyond sparse reference anchors to within-cut 15-row sampling; resolved/pinned frozen SAM2; attempted direct/rectified stereo, retaining only a weak common-height hypothesis; replaced unreliable absolute robot contact projection with visible shaft localization; separated touching-object proposals; retained supported partial channels with true marginal likelihood; recovered hole geometry from causal references. Initial 341-row preparation was not accepted at face value: image review found merges, missing holes and up to24.5px tool projection residual. Subsequent passes addressed those problems, then all actors were subjected to a general complete-causal-reference test after half-bar errors were found. Final conversion retained332 before the completeness gate and rejected202 unresolved/partial actors, leaving130. Whole-component reference boxes were tried without relaxing quality criteria; they did not recover the three empty cuts. Other pre-gate losses were45 insufficient prefixes,126 unsupported near-tool associations,210 responses below uncertainty,30 registrations,130 height mismatches,12 direction inconsistencies,29 area inconsistencies and161 unresolved visual tool endpoints. These are conversion limitations or non-contact evidence, not proof the source motions are intrinsically useless. Raw/context evidence and all prior versions are retained.

**training_rationale**：The residual learned problem remains one shared identity-blind contact/heading/short-stroke decision, not motion planning or semantics. The 18497-parameter geometric scorer with frozen SAM and physical candidate constraints is a defensible small-data supervised prototype; a large image/motor diffusion model is not justified. Preparation passed finite nonzero-gradient and finite [2,5] prediction checks with zero optimizer updates. The submitted procedure guarantees an initial gradient sweep over every final valid row, then balances nonempty activity cuts for a fixed3000 AdamW warmup/cosine updates and selects the final model. Both episodes and all valid development-inspected evidence train the delivered weights; none is withheld as an independent test. Final in-sample pseudo-label fitting metrics are recorded separately from checkpoint reload and executable calling tests (including original RGB proposals and synthetic executor/commissioning fixtures). Those results must not be interpreted as measured robot success or novel-shape competence.

**limitations**：This decision authorizes fitting the supported geometric prototype, not deployment certification. Approximate sparse stereo height (six dominant-height correspondences across three poses), silhouette side bias, imperfect tool localization, missing small holes/obstacle contours and residual association uncertainty remain; pseudo-labels are not verified contact/force truth. Three activity cuts have no final gradient rows and fine finishing/corner diversity is limited, although related shape behaviors survive elsewhere. Supplied Agent semantic/uprightness/inventory/layout reasoning and complete causal scene validation are explicit upstream obligations, not measured learned abilities here. Commissioned camera/table/tool collision geometry and controller guards are missing deployment prerequisites; no robot is connected. No independent unseen physical shape, identity, word/layout or end-to-end test is available. Raw SAM proposals cannot be treated as validated scene truth; incomplete/uncommissioned/stale calls refuse action.

实际训练权重已更新，checkpoint 重载和 API 提供的调用案例通过。调用案例是接口证据，不是真机成功或独立泛化评估。

- [API 调用目录](../runs/push_letters/training_v6/package_00/source/POLICY_CATALOG.md)
- [API 调用说明](../runs/push_letters/training_v6/package_00/source/CALLING.md)
- [API 模型/训练配置](../runs/push_letters/training_v6/package_00/source/package.json)
- [覆盖与质量记录](../runs/push_letters/training_v6/package_00/prepared/metadata.json)
- [训练库](../runs/push_letters/training_v6/library.json)
- [完成核验](../runs/push_letters/training_v6/completion_receipt.json)

## 可下载与实际调用

新[推理入口](../deployment_pipeline_v2/README.md)已导出API原始源码、全部调用契约、
选中模型、必需SAM资产与真实权重示例。[完整归档](../exports/pipeline_v2/pipeline_v2_bundle.tar.gz)
在独立目录完成47文件核验，接触点/方向/长度及执行目标与原worker一致，CLI示例通过。
详见[迁移报告](PUSH_V6_DEPLOYMENT.md)和[实际回执](PUSH_V6_DEPLOYMENT_CHECK.json)。
Git提交/推送、Drive上传、接收端实时感知/标定/planner绑定与真机运行均未执行。
