# Exp2: five-task, four-method study

## Exp2_new writing package (reviewed 2026-09-20)

The corrected-executor three-method OOD comparison now has a separate English
[report](Exp2_new_paper/REPORT.md), [writing guide](Exp2_new_paper/WRITING_GUIDE.md),
[failure analysis](Exp2_new_paper/FAILURE_ANALYSIS.md), and
[API recovery cases](Exp2_new_paper/RECOVERY_CASES.md).
Use the [compact writing ZIP](Exp2_new_writing_bundle.zip) (~60 MB) for a writing
assistant, and the [full evidence ZIP](Exp2_new_paper_bundle.zip) (~814 MB) for
the complete action traces and API journals. Both retain the same 225 primary
OOD outcomes and 30 paired partial-ID outcomes; 65 planned APPL ID episodes remain
unstarted. Checkpoint binaries are indexed, not embedded. Each archive has its
own [full](Exp2_new_paper_bundle.validation.json) /
[compact](Exp2_new_writing_bundle.validation.json) validation receipt.
No new API, training, or simulation was used for this export. The earlier
four-method paper package remains separate and unchanged.

## Completed Exp2_new OOD comparison (2026-09-20)

All **75/75 OOD outcomes** are complete and audited: APPL **30/75 (40.0%)**,
corrected naive DP **5/75 (6.7%)**, and SinglePrior **10/75 (13.3%)** on identical
layouts. APPL per task: drawer 2/15, sort 3/15, buffer swap 4/15, unstack 11/15,
tray 10/15. All evaluation processes exited. The user requested stopping here;
no new ID trial was run, so ID remains 8/10 and the other 65 ID cells are unstarted.
All 36 policy packages, prompts and scientific settings were reused.

The added SGD 500 allowance incurred an estimated USD 334.5613525 across
1,197 generation requests, below the declared USD 390 cap. Total Exp2_new runtime
cost including the original round is USD 412.129084 across 1,460 requests; unknown
charges are zero. The preserved GPU resource incident and four reset retests are
documented; no completed failure or transport request was automatically retried.
See [results](../../runs/exp2/Exp2_new/ood_resource_resume_20260920/REPORT.md),
[accounting and interpretation](../../runs/exp2/Exp2_new/ood_resource_resume_20260920/ANALYSIS.md),
[completion audit](../../runs/exp2/Exp2_new/ood_resource_resume_20260920/completion.json),
and [all 75 videos](../../runs/exp2/Exp2_new/ood_resource_resume_20260920/replays.html).
The following sections retain the execution history.

## Earlier OOD-only continuation setup (historical)

The user added SGD 500 and requested completing all **75 OOD** outcomes, then
stopping. The seven completed OOD results are reused; 68 outstanding cells are
scheduled on GPUs 0/1/2/3/7. The prior budget-interrupted cell is explicitly
retested from the same reset, with its old prefix retained. No ID or completed
failure is rerun. Model, policies, prompts and evaluator remain frozen.
The additional budget cap is USD 390 at the previous conservative planning ratio.
See the [new report](../../runs/exp2/Exp2_new/ood_continuation_20260920/REPORT.md),
[plan](../../runs/exp2/Exp2_new/ood_continuation_20260920/plan.json),
[status](../../runs/exp2/Exp2_new/ood_continuation_20260920/results.json) and
[videos](../../runs/exp2/Exp2_new/ood_continuation_20260920/replays.html).
The original round below is retained unchanged.

## Exp2_new: corrected APPL deployment (2026-09-19)

The new comparison is named **Exp2_new**. Its [protocol](EXP2_NEW.md) reuses
all 36 frozen GPT-6 Astra/xhigh policies, with the corrected gripper executor,
the first 15 original ID and 15 position-OOD layouts per task, and the new official
API credential. No segmentation, prior generation or training is repeated.
The [frozen plan](../../runs/exp2/Exp2_new/plan.json) selects 150 new APPL trials
and references 300 already completed corrected baseline trials. Original physical
artifact paths remain intact. On 2026-09-20 the user confirmed a **100 SGD** budget
and authorized immediate deployment. The execution ledger uses a conservative
USD 78 ceiling; see [authorization](../../runs/exp2/Exp2_new/budget/authorization.json).
Execution ended on 2026-09-20: **17 completed outcomes (ID 8/10, OOD 4/7),
one budget interruption and 132 unstarted cells**. Recorded usage costs USD
77.5677315; the remaining USD 0.4322685 cannot reserve the complete next request.
The next generation was not sent. All 263 attempted generations returned HTTP
200 with reported usage; no provider rejection, unknown charges or retries occurred.
All 36 offline policy checks passed (324 actions, maximum difference zero),
and 12 framework tests passed after a preserved zero-request setup-path repair.
All evaluation processes and 67 policy workers exited; only GPU 7 was used.
See the [completion audit](../../runs/exp2/Exp2_new/completion.json).
Use the [new report](../../runs/exp2/Exp2_new/REPORT.md),
[matched comparison and costs](../../runs/exp2/Exp2_new/ANALYSIS.md),
[per-layout status](../../runs/exp2/Exp2_new/episodes.csv) and
[videos](../../runs/exp2/Exp2_new/replays.html).
The planned 15-per-task/split comparison is incomplete; do not restart pending
cells without a new funded-continuation instruction.
Historical scores below retain their original settings. GPT-5.5 stays paused.

## Completed inference correction (2026-09-19)

The [binary-gripper comparison](BINARY_GRIPPER.md) reruns the ten frozen naive
and single-prior models on all 600 original paired layouts, changing only final
gripper decoding. No retraining or Runtime API calls; both APPL versions are
paused. New evidence lives in
[binary_gripper_v1_20260919](../../runs/exp2/binary_gripper_v1_20260919/).
All 600 outcomes and videos passed [final validation](../../runs/exp2/binary_gripper_v1_20260919/final_validation.json);
all ten original-action reproduction checks have zero error. No workers remain.

| Method | Original ID → corrected ID / 150 | Original OOD → corrected OOD / 150 |
| --- | ---: | ---: |
| Naive DP | 17 → 76 (50.7%) | 0 → 7 (4.7%) |
| SinglePrior_6_xhigh | 110 → 139 (92.7%) | 16 → 24 (16.0%) |

The single prior succeeds on all 120 new-task ID layouts, while its drawer ID
drops from 22/30 to 19/30. This is a paired post-study decoder ablation, not a new
untouched test. Read the [report](../../runs/exp2/binary_gripper_v1_20260919/REPORT.md),
[interpretation](../../runs/exp2/binary_gripper_v1_20260919/ANALYSIS.md) and
[videos](../../runs/exp2/binary_gripper_v1_20260919/replays.html).
Preserve the historical matrix below; APPL has not been rerun with this decoder.

## Post-study DP diagnosis (2026-09-19)

The [frozen-model diagnostic report](analysis/dp_diagnosis_20260919/REPORT.md)
finds a gripper feedback instability and narrow intermediate-state coverage.
Original training-reset success is drawer 9/12 and each new task 0/12 within
1,500 steps; all 60 recorded expert action replays succeed. A paired binary-gripper
intervention improves sorting and unstack to 3/3 and buffer swap to 2/3;
tray remains 0/3. These are adaptive diagnostics, not replacement formal scores.
See [comparison videos](analysis/dp_diagnosis_20260919/videos.html) and
[validation](analysis/dp_diagnosis_20260919/completion.json). No retraining or new
Runtime API calls; frozen source, checkpoints, data and results are unchanged.
Read this report alongside the original paper evidence when assessing the baseline.

The [cross-method gripper review](analysis/cross_method_gripper_20260919/REPORT.md)
checks all 1,200 existing formal traces: the single prior and both APPL versions
share continuous gripper execution and exhibit progressive air-closure signatures,
with different incidence and recovery outcomes. This observational review adds no
rollouts and does not causally attribute all flagged failures to the same mechanism.

## Complete paper evidence

Final review 2026-09-19: the four-method comparison contains 1,200 planned cells
and 1,200 complete outcomes. The sole historical APPL 5.5 unknown was resolved
by one authorized same-setting reset: buffer_swap / ID / 20116 succeeded at
945 steps. APPL 5.5 now has ID 131/150 and OOD 50/150; the original interrupted
attempt and all frozen reports remain unchanged. See the [recovery audit](paper/recovery_20260919/REPORT.md). The English
report, methods, development history, raw tables, figures, exact API packages,
paired examples and provenance are ready for a paper-writing assistant.

- [Full English report](paper/REPORT.md), [Chinese entry guide](paper/START_HERE.md),
  and [writing instructions](paper/WRITING_GUIDE.md).
- [Portable ZIP](Exp2_paper_bundle.zip), [bundle validation](paper_bundle_validation.json),
  and [forty paired original videos](paper/execution_examples/index.html).
- [Model/source index](paper/POLICY_INDEX.csv) and [API prior catalog](paper/PRIOR_CATALOG.md).

## Single-policy prior baseline (complete)

Started 2026-09-18 after the user selected one candidate per task. GPT-6
Astra/xhigh designs one inductive-bias diffusion policy from each task's twelve
complete demonstrations. Each model trains for 60,000 updates, then runs directly
without a deployment API agent. The new matrix has 30 paired ID and 30 paired
position-OOD trials per task, with unchanged goals and the 5,000-step cap.

[Specification](SINGLE_POLICY.md), [framework](single_policy/),
[preparation evidence](../../runs/exp2/single_policy_astra_xhigh/preparation.json),
[final execution status](../../runs/exp2/single_policy_astra_xhigh/supervisor/status.json).
All five models and 300 new outcomes passed validation: **126 successes**, with
**110/150 ID** and **16/150 position-OOD** successes, and zero deployment API calls.
All experiment coordinators and policy workers have exited. Existing DP and both
APPL comparisons remain frozen.

[Baseline report](../../runs/exp2/single_policy_astra_xhigh/REPORT.md),
[completion receipt](../../runs/exp2/single_policy_astra_xhigh/completion.json),
[retained-incident accounting](../../runs/exp2/single_policy_astra_xhigh/incidents/tray_design_http502/recovery_completed.json),
and [worker/resource closure](paper/resource_release.json).

## Additional GPT-6 Astra / xhigh study

Final review 2026-09-18: all 36 API policy pipelines completed training and
deployment checks. The new study now has **300/300 complete outcomes: 197 successes,
103 task failures and no unknowns**, with **124/150 ID** and **73/150 position-OOD**
successes. All selected trajectories and 300 replay videos have validated evidence.

- [Final three-method results](../../runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/REPORT.md),
  [measured analysis](../../runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/ANALYSIS.md),
  and [completion receipt](../../runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/completion.json).
- [Fixed paired videos](../../runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/paired_examples.html)
  and [all 300 selected replays](../../runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/replays.html).
- [36 API-authored policy packages](../../runs/exp2/M1_astra_xhigh/POLICIES.md)
  and [study specification with recovery history](ASTRA_XHIGH.md).

The final view retains 211 original complete outcomes, 68 from the first resumed
batch and 21 from the last explicitly authorized continuation. Every original
attempt, interruption, API output and reported cost remains intact; no completed
task failure was retested, and recovery added no training updates. Earlier reports
remain under the parent study and its two preceding recovery directories.
The DP and GPT-5.5/high references are reused; the latter retains its one historical
unknown outcome. Both model/effort and the API-generated policy library changed,
so this is a comparison of complete pipelines rather than an isolated effort test.
The latest resource limit is five simultaneous physical GPUs; the final evaluation
used 1/2/3/5, and [all owned evaluation processes have exited](../../runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/resource_release.json).

## Original scale-up study

The original study compares naive DP and APPL on five tasks, with 12 training
demonstrations and 30 ID plus 30 position-OOD layouts per task and method.
Training and all 600 planned evaluation attempts have ended: 599 complete
outcomes and one retained HTTP interruption. All 600 available replay clips,
completed trajectories and the interrupted prefix passed independent validation.
The interrupted final outcome remains unknown. See the [completion receipt](../../runs/exp2/M1_scaleup/completion.json)
and [final analysis](../../runs/exp2/M1_scaleup/ANALYSIS.md).

- [Protocol](SCALEUP.md) and [five task configurations](configs/scaleup/).
- [Current progress](../../PROGRESS.md) and [combined live queue snapshot](analysis/live_status.json).
- [Five-task results](../../runs/exp2/M1_scaleup/REPORT.md) and [raw outcome table](../../runs/exp2/M1_scaleup/episodes.csv).
- [Paired examples](../../runs/exp2/M1_scaleup/paired_examples.html): the first declared ID/OOD seed per task.
- [API-authored policy packages](../../runs/exp2/M1_scaleup/POLICIES.md).

| Task | Data | Runtime artifacts |
| --- | --- | --- |
| Original drawer exchange | [Original 12 demonstrations](../../data/exp2/demonstrations_v2/) | [Drawer evaluation](../../runs/exp2/M1_scaleup/drawer_exchange/) |
| Two-block sorting | [12 demonstrations](../../data/exp2/scaleup/two_block_sort/demonstrations/) | [Task pipeline](../../runs/exp2/M1_scaleup/two_block_sort/) |
| Buffer-assisted swapping | [12 demonstrations](../../data/exp2/scaleup/buffer_swap/demonstrations/) | [Task pipeline](../../runs/exp2/M1_scaleup/buffer_swap/) |
| Unstacking and sorting | [12 demonstrations](../../data/exp2/scaleup/unstack_sort/demonstrations/) | [Task pipeline](../../runs/exp2/M1_scaleup/unstack_sort/) |
| Open-tray packing | [12 demonstrations](../../data/exp2/scaleup/tray_pack/demonstrations/) | [Task pipeline](../../runs/exp2/M1_scaleup/tray_pack/) |

Developer-owned task environments, collection, baseline and execution live in
[src/appl/scaleup](../../src/appl/scaleup/). Runtime API owns segmentation,
heuristics, prior implementations, handoff semantics and inference decisions.
The shared policy framework is [src/appl/prior_policies](../../src/appl/prior_policies/).
Each task fits its own shared normalizer from its complete original demonstrations.
Exp1 and the frozen original M0 remain unchanged.

Evaluation uses DDPM100, execution chunks of 8 and a private 5000-step cap.
The API does not receive the total or remaining budget. Both methods share each
layout and geometric completion contract. This original study used GPU pool
1, 2, 3, 7 with at most four active physical GPUs under its historical allocation.

## Initial study and archive

The active initial-test entry is [M1_initial_test](../../runs/exp2/M1_initial_test/README.md),
with its [5000-step report](../../runs/exp2/M1_initial_test/inference_5000/REPORT.md).
The original 1500/3000-step studies and settings are in
[archive/Exp2_M1_initial_tests](../../archive/Exp2_M1_initial_tests/README.md).
`M1_v2` paths remain compatibility aliases. Their five-seed development results
are distinct from the fresh 30 ID / 30 OOD scale-up evaluation.

## Shared framework and historical M1_v2 details

初始研究采用 **M1_v2：API 重新切分 → 每条 heuristic 的独立 Prior Diffusion Policy → API 根据 prior 与交接文档选择 policy**。状态统一见 [PROGRESS.md](../../PROGRESS.md)，执行规范见 [PRIOR_POLICIES.md](PRIOR_POLICIES.md)。Exp1 和 M0 保持不变。

最新推理修正已完成（核对 2026-09-16）：[m1_v2_5000.json](configs/m1_v2_5000.json) 下相同 ID 初态复测 **4/5 成功**、训练初态诊断 **1/1 成功**。执行器私下限制 5,000 步，API 看不到总上限或剩余步数；API 自己选择 policy、调用时长、逐步监测的数值停止条件和 notebook。30 次 API 条件提前返回均逐步核验，全部 94 次 API 请求完成。成功回合在 987–1,529 步结束，失败的 6201 在 2,070 步由 API 主动结束；没有回合用满上限。原 18 个模型不重训，原始历史保全，39 项测试通过。[报告](../../runs/exp2/M1_v2/inference_5000/REPORT.md)、[分析](../../runs/exp2/M1_v2/inference_5000/ANALYSIS.md)、[回放](../../runs/exp2/M1_v2/inference_5000/visualizations/README.md)、[核验](../../runs/exp2/M1_v2/inference_5000/completion.json)。细则见 [INFERENCE_5000.md](INFERENCE_5000.md)。这是预算、反馈和上下文一起改变的系统修正复测，不是独立泛化评估或单因素消融。

原 1,500 步轮次已完成：18 个 policy 各训练 20,000 步；独立 ID 2/5，示范初态诊断 1/1。其中一个 ID 成功发生在高速恢复动作后的瞬时达标，不代表终态稳定。完整[结果](../../runs/exp2/M1_v2/REPORT.md)、[分析](../../runs/exp2/M1_v2/ANALYSIS.md)和[完成核验](../../runs/exp2/M1_v2/completion.json)保留全部成功、失败与费用。

此前的 **3,000 物理步**追加测试使用 [m1_v2_3000.json](configs/m1_v2_3000.json)，复用同一批冻结模型与 6 个初态，结果单独保留在 [budget_3000](../../runs/exp2/M1_v2/budget_3000/)。无新增训练，API 请求上限为 40，每次 policy 调用最多 300 步；仿真器和评估器采用一致的 3,000 步上限。该轮从 reset 开始，分别统计前 1,500 步内和之后的成功。

追加测试的全部尝试已终止：ID 2 成功、2 任务失败、1 上下文超限中断；示范初态诊断失败。6201 在 1,782 步成功，6204 在 1,073 步成功；6202 的 3,000 步最终结果未知，未自动重试。完整[报告](../../runs/exp2/M1_v2/budget_3000/REPORT.md)、[分析](../../runs/exp2/M1_v2/budget_3000/ANALYSIS.md)、[回放](../../runs/exp2/M1_v2/budget_3000/visualizations/README.md)区分了额外步数的收益、物理交接问题和 API 历史重复累积的问题。

## 实验版本

| 工作线 | 代码与配置 | 数据和结果 |
| --- | --- | --- |
| M0 保留基线 | [dp_baseline](../../src/appl/dp_baseline/)、[m0.json](configs/m0.json) | [模型](../../runs/exp2/m0/)、[最终报告](M0_REPORT.md) |
| M1_v1 历史轮次 | [历史配置](configs/m1_v1.json)、运行目录内 source_versions | [原切分](../../data/exp2/processed/drawer_exchange_20260916_overlap/)、[报告](../../runs/exp2/M1_v1/REPORT.md)、[失败分析](../../runs/exp2/M1_v1/ANALYSIS.md) |
| M1_v2 初始研究（历史） | [prior_policies](../../src/appl/prior_policies/)、[m1_v2.json](configs/m1_v2.json) | [新切分](../../data/exp2/processed/M1_v2/)、[运行目录](../../runs/exp2/M1_v2/)、[共享归一化](../../runs/exp2/M1_v2/normalization.json) |

M1_v1 的 31 个正式训练及接口检查 checkpoint 已按用户授权删除，约 11.1 GB；源码、提交、API 日志、训练指标和评估证据保留。旧 `prior_policies_20260916` 路径仍是别名，历史引用继续可读。[版本说明](../../runs/exp2/M1_v1/VERSION.md)与[删除回执](../../runs/exp2/M1_v2/setup/v1_checkpoint_deletion_receipt.json)记录了不能再直接部署旧模型这一事实。

## 示范拆分入口

原始训练示范仍为 demo1000–1011，共 12 条。M1_v2 使用英文 prompt 要求 API 显著扩大有意义的交接重叠、依据每条轨迹的状态分别选择边界，并在每条 heuristic 后写明接管条件、结束条件、共享过渡的作用、后继技能需求和交接失败迹象。具体边界、重叠宽度、prior 和交接语义均由 API 生成，开发者不改写输出。

API 输入和执行判定沿用[三个空间目标](configs/demonstration_goals.json)。交接可能需要开爪、退让或持物等状态，但这些是技能适用信息，不是额外的整任务终态门槛。增加 overlap 扩大的是示范中的过渡覆盖，不能据此宣称已经覆盖偏离示范的恢复行为。

## 统一归一化

所有 policy 使用同一份 [normalization.json](../../runs/exp2/M1_v2/normalization.json)：从完整原始训练轨迹一次拟合观测和动作范围，包含 12,809 个因果动作输入观测；不按技能拟合，不因 overlap 重复计算，不读取验证或推理数据。四元数仍使用固定单位尺度。每个 assignment、训练记录和 checkpoint 保存该参数的身份及 hash。M0 数值实现不修改。

## 每个 policy 的目录

`runs/exp2/M1_v2/policies/<skill>/heuristic_<number>/`：

| 内容 | 作者与用途 |
| --- | --- |
| assignment.json | 框架：原始 heuristic、切分与共享归一化 hash |
| design/、submission.json | API 会话、版本和不可变提交 |
| source/policy.py、pipeline.json | API：模型结构、loss 和适用条件 |
| source/PRIOR.md | API：实际 prior、输入、机制与限制 |
| source/HANDOFF.json | API：接管/退出条件、后继需求、失败与继续执行提示 |
| handoff_evidence.json | 框架：示范起点、终点、重叠区状态及统计 |
| training/policy_context.json | 归一化与交接文档的身份、hash |
| training/handoff_evidence.json | 与训练一同保存的交接状态证据 |
| training/last.pt | 各自独立的模型、EMA、优化器、RNG、归一化及交接上下文 |
| evaluation/handoffs.json | 真实 API 调用前后状态；未被调用的 policy 明确记录为空，不宣称单技能闭环成功 |

每个 policy 训练 20,000 步、seed 0、DDPM100，使用最终 EMA。具体学习表示、网络和目标由 API 编写；框架只负责训练、数值接口、隔离和审计。每次接口检查的 2 个更新单独记账。

## 推理 API 与测试

推理 API 必须先读取 policy 的 prior、HANDOFF.json 和示范状态统计，再选择 policy 与调用时长。框架不预设技能顺序、不添加交接控制器。每次调用保存调用前后状态、API 理由及模型/交接文档 hash。

原轮次冻结完整策略库后，测试示范初态 1000 和独立 ID 初态 6200–6204，上限为 1,500 步。追加的 3,000 步测试复用这 6 个初态，标记为相同初态复测，不宣称是全新的独立测试。目标仍只有同一时刻抽屉打开、红块在垫上、蓝块在抽屉内。两轮均最多 40 次 API 请求，每次 policy 调用不超过 300 步。

## 命令

已完成的 M1_v2 回合可直接查看[保存的回放视频](../../runs/exp2/M1_v2/visualizations/README.md)。视频由原始 128×128 相机帧打包，每 20 个控制步采样一次，包含最终帧；无需重新运行 API、模型或仿真。完整状态轨迹仍保留在各回合的 trace.jsonl。

所有命令使用锁定的 Exp2 环境。最新 5,000 步反馈修正须显式指定 m1_v2_5000.json；原训练配置 m1_v2.json 保留。下列命令对应已完成的回合，返回保存结果；中断回合不会自动重试。以下初始研究命令保留其历史 GPU 4–7 配置。Scale-up 当前使用 1/2/3/7；最新授权允许从 0–7 中同时最多使用四张。调度前核对实际占用，勿重复启动。

```bash
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies evaluate --config experiments/exp2/configs/m1_v2_5000.json --gpu 6 --seed 6200
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies report --config experiments/exp2/configs/m1_v2_5000.json --verify
```

下列为已完成的原轮次命令，保留供追溯，不表示需要重新训练：

```bash
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies initialize
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies segment
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies queue --gpu 7 --shard 0 --shards 4
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies queue --gpu 6 --shard 1 --shards 4
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies queue --gpu 6 --shard 2 --shards 4
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies queue --gpu 4 --shard 3 --shards 4
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies freeze
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies evaluate --gpu 6 --seed 6200
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m appl.prior_policies report --verify
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m pytest -q tests/exp2
```

源数据、共享数值/环境/隔离能力和重建来源均保留。更早的旧 M1/M2 专属流程已退役，其删除回执在 M1_v1/cleanup。历史 [PROTOCOL.md](PROTOCOL.md) 不作为本轮的新门槛；存储映射见 [MIGRATION.md](MIGRATION.md)。
