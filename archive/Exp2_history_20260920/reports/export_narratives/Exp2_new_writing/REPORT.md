> **Compact writing companion.** The report below describes the full evidence package. This companion retains reports, results, all 60 numerical demonstrations, 255 videos, submitted policy source/priors, and the two recovery-case traces/responses. Complete per-step traces and full API design/deployment journals are in `Exp2_new_paper_bundle.zip`. Read `COMPANION_SCOPE.md` for exact scope. Scientific API files are unchanged.

# Exp2_new: API-designed diffusion priors and online policy selection under position shift

**Research evidence report, reviewed 2026-09-20.** This report describes the completed corrected-executor OOD comparison. It is not the earlier four-method Exp2 report and does not combine historical GPT-5.5 scores with the new executor.

## 1. Executive summary

We compare three learning-and-deployment pipelines on five state-based, long-horizon simulated Panda manipulation tasks. Each task supplies the same twelve successful training demonstrations. Naive DP learns one full-task diffusion policy. SinglePrior uses GPT-6 Astra/xhigh to design one full-task policy with an inductive bias, then deploys it without a language model. APPL uses API-defined overlapping skill datasets, separately trained prior-conditioned policy designs, and an inference-time API that selects among frozen learned controllers using current state, prior documents, and handoff evidence.

The primary evaluation is five tasks with fifteen matched position-OOD initializations per task. Under a shared corrected binary-gripper executor and a maximum of 5,000 physical steps, naive DP succeeds in **5/75 (6.7%)**, SinglePrior in **10/75 (13.3%)**, and APPL in **30/75 (40.0%)**. These are paired results on previously examined layouts, with one training seed and one selected complete outcome per cell. The improvement is heterogeneous: APPL solves 11/15 unstacking and 10/15 tray episodes, but only 2/15 drawer, 3/15 sorting, and 4/15 buffer-exchange episodes. SinglePrior exceeds APPL on sorting by one outcome.

The full trajectories provide examples of API-diagnosed acquisition/placement failures followed by successful policy switching. They also show a major limitation: changing among successful-demonstration controllers often cannot recover a missed grasp or displaced object. All 45 APPL failures terminate through the API's `finish` tool before the physical cap. This experiment does not isolate the causal effect of online reasoning from segmentation, policy multiplicity, or additional training compute.

Primary sources: [per-layout results](tables/episodes.csv), [summary](tables/ood_summary.csv), [final original completion audit](execution_history/ood_resource_resume_20260920/completion.json).

## 2. Research questions and method ownership

The comparison asks whether offline API-authored inductive biases and online selection among documented learned skills improve task completion under object-position shift. The method has two distinct computational levels: a language-model API chooses which frozen controller to run; the selected diffusion policy generates the low-level action sequence.

| Component | Naive DP | SinglePrior | APPL |
| --- | --- | --- | --- |
| Full demonstrations per task | 12 | Same 12 | Same 12, segmented with overlap |
| Scientific policy designer | Developer, fixed baseline | GPT-6 Astra/xhigh | GPT-6 Astra/xhigh |
| API-defined segmentation | No | No | Yes |
| Trained models across five tasks | 5 | 5 | 36 |
| Updates per model | 60,000 | 60,000 | 20,000 |
| Sum of formal updates | 300,000 | 300,000 | 720,000 |
| API at deployment | No | No | Yes |
| Low-level actions | Learned diffusion policy | Learned diffusion policy | Selected learned diffusion policy |

**Runtime API ownership:** skill boundaries and overlap, heuristic hypotheses, inductive-bias implementation, submitted policy source, semantic prior/handoff documents, and deployment policy/duration/stop-rule/finish decisions. The system does not silently replace these with developer-authored candidate designs or policy choices.

**Developer/framework ownership:** environment and task design, scripted demonstration acquisition, baseline, prompts and public interfaces, normalization recipe, numerical training and checkpointing, controlled execution, numerical feedback, geometric evaluation, resource scheduling, and accounting. The API does not directly update network weights, issue individual actuator actions, alter the environment, or redefine the success predicate.

All 41 API-authored policy packages have retained immutable submissions and authorship audits. The [catalog](PRIOR_CATALOG.md) links their exact source and prior text. A prior's written motivation is a hypothesis, not proof that the learned model implements the intended invariant or recovery behavior reliably.

## 3. Tasks, demonstrations, and OOD construction

### 3.1 Environment and observations

The tasks use a simulated Panda robot with a `pd_joint_pos` controller at **20 Hz**, implemented with the locked Exp2 ManiSkill/SAPIEN environment. Policies receive **47-dimensional structured state**: robot joint positions/velocities, TCP pose, red/blue object poses, drawer position/velocity, and target positions. Poses use world coordinates and `wxyz` quaternion order. Drawer channels are fixed zero compatibility channels in tasks without a drawer. Policy control is state-based; replay images are not the low-level observation input. The deployment agent receives structured state and measured feedback rather than camera-image understanding.

Actions contain seven **absolute arm-joint targets** and one gripper command. The corrected executor retains existing arm bounds and maps the sampled raw gripper coordinate to `+1` when nonnegative, otherwise `-1`. Raw and executed actions are both recorded. This does not turn the learned controller into a scripted grasp policy: it still has to predict the correct sign at the right state.

### 3.2 Tasks and data

| Task | Intended demonstrated sequence | Demonstrations | Total demonstration actions | Length range |
| --- | --- | ---: | ---: | ---: |
| Drawer exchange | Open drawer; move red out onto pad; move blue into drawer | 12 | 12,809 | 1,063–1,077 |
| Two-block sorting | Move red, then blue, to their separate pads | 12 | 9,237 | 756–777 |
| Buffer exchange | Stage red in buffer; move blue to red's old region; retrieve red to blue's old region | 12 | 12,742 | 1,056–1,071 |
| Unstack and sort | Remove upper red block; place red; place lower blue block | 12 | 8,729 | 717–739 |
| Tray packing | Place both table blocks in separate marked regions of an open tray | 12 | 8,897 | 730–752 |

The demonstrated ordering describes data collection, not a hard-coded API deployment sequence. These are successful scripted demonstrations with narrow initial-position variation. They are not teleoperation data, recovery demonstrations, or a broad distribution of failed-contact states. The package includes all sixty numerical trajectories and exact provenance in [training_trajectories.csv](tables/training_trajectories.csv), plus [task definitions](task_definitions) and [goal contracts](contracts).

### 3.3 Position OOD

Training/ID block-center XY perturbations have half-width **0.012 m**, except the drawer task's original blue ID range, whose half-width is **0.015 m**. OOD perturbations sample coordinates within **±0.040 m**, with one randomly selected axis forced to have magnitude **0.022–0.040 m**. Objects, robot, controller, target geometry, and task definition remain fixed. This tests object-position extrapolation, not new objects, dynamics, visual appearance, or robots.

In unstacking, both blocks share one XY displacement, preserving their relative stack geometry. In other tasks the two displacements are sampled separately. Therefore equal nominal displacement bounds do not make the tasks equally difficult OOD problems. OOD offsets are substantial relative to a block's **4 cm edge length** and to the narrow training support.

The fifteen primary seeds are the first fifteen entries of each original ordered OOD list: drawer 6400–6414, sorting 30000–30014, buffer 30100–30114, unstacking 30200–30214, tray 30300–30314. Selection was fixed without filtering on outcomes. The three methods have exactly matching saved initial states for every included layout. These seeds had been examined in earlier work; this is a paired post hoc evaluation, not an untouched confirmatory test set.

## 4. Training and API-designed policies

### 4.1 Shared diffusion interface

The recipe uses two causal observations, a prediction horizon of sixteen actions, and an eight-action execution prefix; diffusion training and sampling use **100 DDPM steps**. Actions are aligned to the fixed history/horizon convention, and the public epsilon-prediction interface is `[B,16,8]` conditioned on raw `[B,2,47]` history. Batch size is 128; AdamW learning rate is 1e-4, weight decay 1e-6, cosine scheduling with 500 warmup updates, gradient norm limit 1, and EMA decay 0.999. Training seed is zero, and the final EMA at the declared update budget is selected. The locked configurations and individual training receipts are authoritative.

Each task's shared observation/action scales are fit on its twelve **complete original demonstrations**, not separately on each cut skill. The configuration uses range/limit-based observation scaling with explicit unit-component quaternion bounds. Shared normalization prevents the earlier per-skill scale inconsistency; it does not eliminate distribution shift in contact, role, or intermediate state.

### 4.2 Naive DP and SinglePrior

Naive DP uses the fixed conditional diffusion U-Net, with 16,998,408 trainable parameters. SinglePrior gives the API one candidate design per task, complete demonstration access through the fixed tools, and the task contract. The API can implement learned representations, architectural conditioning, and auxiliary losses within the public interface. It receives no APPL candidate code or test-feedback design opportunity. Both full-task methods train for 60,000 updates per task. SinglePrior parameter counts range from 19,038,112 to 20,020,712. Equal update counts are not an assertion of equal FLOPs.

SinglePrior has no skill segmentation, runtime LLM, external phase controller, or online policy selection. Its five API-produced priors are directly deployed as frozen learned policies. See [model table](tables/models.csv) and [API catalog](PRIOR_CATALOG.md).

### 4.3 APPL segmentation and library

The API defines overlapping skill slices and three heuristic/prior variants for each skill. It writes independently submitted policy implementations and `PRIOR.md`/`HANDOFF.json` documents. Framework-generated handoff statistics describe training states; they do not choose semantic successors. The resulting skill counts are 3/2/3/2/2 for drawer/sorting/buffer/unstacking/tray, giving 9/6/9/6/6 policies and **36 models** overall.

The segmentation manifests account for all original actions without exclusions. Recorded overlap-action counts are 6,626 / 4,538 / 5,832 / 4,638 / 4,521 respectively; these are manifest audit quantities, not additional independent demonstrations. Two-skill object-reused tasks contain multiple slices per demonstration. Exact boundaries and API rationales are in [segmentations](segmentations).

Each APPL policy trains for 20,000 updates; parameter counts range from 4,505,681 to 20,186,659. APPL therefore uses 180,000 total updates in each three-skill task and 120,000 in each two-skill task. The sum of retained training elapsed times is approximately 14.36 worker-hours, versus 2.69 for naive DP and 3.43 for SinglePrior. These are sums of per-model training durations, not comparable end-to-end wall-clock runs or exact GPU-utilization/FLOP measures.

All policies in Exp2_new are reused from their earlier completed training runs. **Exp2_new itself performs no segmentation, new prior design, or training.** Checkpoint hashes were verified when building this package and remain indexed in [checkpoint_inventory.csv](tables/checkpoint_inventory.csv).

## 5. Deployment and stopping

APPL deployment requests use **GPT-6 Astra**, **`reasoning.effort=xhigh`**, and the official OpenAI route with standard service tier. Training-time API sessions used the previously recorded route; that provenance is retained rather than rewritten. The exact per-task English deployment prompts are in [execution_history/prompts](execution_history/prompts).

The API receives a policy catalog and current structured state, reads selected prior/handoff documents and measured training support, then calls `invoke_policy` with a policy identifier, requested duration, literal numeric stop conditions, reason, and notebook. The executor runs the selected policy and checks those exact conditions after each physical step. It returns current state, task-goal values, measured minima/maxima, and invocation outcome. The API may continue, change policy, revisit a skill, or call `finish`. There is no developer-authored failure-to-successor lookup table.

The physical cap is 5,000 steps, hidden from API input; elapsed steps are visible, total/remaining budget is not. Per invocation the maximum is 300 steps. The API generation limit is 128 per episode; requests permit at most 4,096 output tokens. A bounded context projection retains complete original exchanges, prior documentation and the API's notebook, while the full journal remains saved. Numeric stop groups are selected by the API; the framework neither rewrites them nor invents semantic handoff thresholds.

When switching policy, the controller clears its observation/action queues and pads the initial two-observation history with the current state. Continuing the same policy retains its stream. This behavior is preserved in Exp2_new. A possible effect on velocity-derived phase cues is a hypothesis requiring an ablation; this evaluation does not isolate that effect.

The two single-policy methods repeatedly execute their fixed controller without any deployment API call. All three methods use identical geometry-based success contracts and the corrected gripper decoder. APPL success is measured by the evaluator, not by the API's declaration.

## 6. Success definition

Drawer success requires simultaneous drawer displacement **strictly greater than 0.26 m**, red fully contained on its marked pad in XY with its specified height interval, and blue fully contained in the drawer cavity with its specified height interval. The targets and rotated block extents are evaluated in the recorded world frame.

The four other tasks require both blocks' rotated XY footprints to lie in their target rectangles, with center-height error strictly less than **0.011 m**. Pads have half-width 0.06 m in XY; block half-size is 0.02 m. Thus an upright centered block has roughly 4 cm of center-offset room per XY axis, reduced by rotation. Tray target center height is 0.036 m; table-pad target height is 0.020 m.

One observation satisfying the conjunction is sufficient. No extra release, TCP-clearance, velocity, or sustained-hold condition is imposed. Consequently success can occur while the final block is still held. Claims about stable release, contact robustness, or safety require additional evaluation. See the [exact contracts](contracts).

## 7. Primary OOD results

| Task | Naive DP | SinglePrior | APPL |
| --- | ---: | ---: | ---: |
| Drawer exchange | 0/15 | 0/15 | 2/15 |
| Two-block sorting | 0/15 | 4/15 | 3/15 |
| Buffer exchange | 0/15 | 1/15 | 4/15 |
| Unstack and sort | 5/15 | 3/15 | 11/15 |
| Tray packing | 0/15 | 2/15 | 10/15 |
| **All five tasks** | **5/75 (6.7%)** | **10/75 (13.3%)** | **30/75 (40.0%)** |

![Primary OOD results](figures/ood_results.png)

The aggregate absolute advantage of APPL is 33.3 percentage points over naive DP and 26.7 points over SinglePrior. APPL is not best in every task: sorting is 3/15 versus SinglePrior's 4/15. Each task has only fifteen outcomes, so differences of one or two outcomes should not be overinterpreted.

The exported Wilson 95% intervals are descriptive binomial intervals. Aggregate intervals are approximately 2.9–14.7% (DP), 7.4–22.8% (SinglePrior), and 29.7–51.3% (APPL). The aggregate mixes fixed heterogeneous tasks; it is not an estimate over a random population of robot tasks. These intervals do not capture uncertainty across training seeds or API-repeat runs. [Paired contingency counts](tables/paired_outcomes.csv) preserve within-layout matching; no multiple-comparison significance claim is made.

All five DP and all ten SinglePrior OOD successes occur within 1,500 steps. APPL has 29 successes within 1,500 and one between 1,500 and 3,000; none occur after 3,000. All 45 APPL failures end through API `finish`, at 418–2,663 steps, median 1,049. A 5,000-step allowance is therefore not equivalent to executing all trials for 5,000 steps. Increasing only the hidden cap would not change the stop condition that actually ended these observed failures; whether a changed continuation policy helps is untested.

## 8. Partial ID supplement

Only ten APPL ID layouts completed before the user prioritized finishing OOD: two per task. Matched results are DP **6/10**, SinglePrior **10/10**, and APPL **8/10**. These thirty method-layout outcomes and videos are included separately. The remaining 65 planned APPL ID layouts are unstarted, not failures. Neither a completed 75-layout ID score nor an ID/OOD significance comparison can be claimed from Exp2_new. See [partial_ID_summary.csv](tables/partial_ID_summary.csv).

## 9. Failure bottlenecks and recovery evidence

The [failure report](FAILURE_ANALYSIS.md) recomputes task progress from physical traces and separates observations from causal hypotheses. The low-scoring tasks have different bottlenecks:

- **Drawer:** all fifteen episodes open the drawer at least once, but only five ever place red and only two insert blue. Ten failures never reach red placement; three place red but fail blue insertion. Later contact often removes the previously achieved drawer-open condition.
- **Sorting:** all fifteen episodes reach red's goal. Eight failures never raise blue by 4 cm above its initial center height; four raise it but never reach its goal. The rise threshold is a descriptive motion proxy, not a contact detector.
- **Buffer:** four failures never raise initial red by 4 cm; three raise red but not blue; four reach blue's goal without completing final red placement. The final group includes one near-boundary placement and three failed red-retrieval cases.

The observations support a mismatch between narrow successful-demonstration support and the intermediate states encountered after contact errors. They do not by themselves distinguish insufficient data coverage, imperfect learned role/phase inference, policy-switch history padding, or policy-selection choices as exclusive causes. Valid prior documents and valid source interfaces establish authorship and contract compliance, not empirical generalization.

Two documented successful recovery examples are unstacking seed 30204 (switch to a separate gripper-event prior after repeated open-finger approaches, then success at 859 steps) and buffer seed 30102 (switch to a successor whose overlap includes the stalled predecessor's final lowering phase, then success at 1,245 steps). [Original API instructions, measured transitions, and videos](RECOVERY_CASES.md) are included. These establish observed recovery sequences, not a counterfactual proof that switching was necessary.

## 10. Execution history, accounting, and reproducibility

The selected 75 OOD results consist of seven original complete outcomes, 38 from the first continuation, and 30 from the resource-resumption phase. Completed failures were retained; no completed failure was rerun to replace its score. One original budget-interrupted prefix and four later resource-interrupted prefixes remain separately recorded. Those interrupted attempts are not independent test layouts or extra task failures; their costs remain included.

A transient unrelated GPU occupancy triggered an overly strict scheduler guard, interrupting four episodes. The subsequent resource-only change skipped busy slots without stopping other episodes. The four same-reset retests preserved exact first-request and initial-state equality. Earlier scientific source, policy identities, and completed results were audited as unchanged. All experiment processes exited before this writing export.

Exp2_new incremental deployment accounting covers **1,460 generation requests**, **65,589,946 input tokens**, and **670,859 output tokens**, including **177,980 reasoning tokens already counted inside output**. The saved tariff-based estimate is **USD 412.129084**: USD 77.5677315 for the first funded phase plus USD 334.5613525 across the two added-funding phases. Using the historical planning ratio 1 SGD = 0.78 USD gives approximately **SGD 528.37**. This is a usage estimate, not an invoice or a current currency quote.

This cost includes partial ID episodes and interrupted prefixes, not only the selected 75 OOD outcomes. The 75 selected complete OOD episodes themselves contain **1,293 deployment generations**. Input usage includes repeated/cached context; it is not unique text volume. Upstream segmentation, design, prior training and previous baseline evaluations are reused and are not included in the incremental Exp2_new dollar total. Unknown-charge reservations at final accounting are zero. See [final accounting](execution_history/ood_resource_resume_20260920/ANALYSIS.md) and the original ledgers in [execution history](execution_history).

All 75 selected OOD episodes passed action, identity, initial-state, fixed-evaluator, API-argument and video audits. This export rechecks source/result hashes, all selected request identities, and equality between original API invocation arguments and executed invocation records. Frozen source, environment locks, model hashes, original API journals and traces are supplied. The repository is not assumed clean: per-file hashes, not the Git commit alone, define the snapshot.

## 11. Limits and paper-safe conclusions

1. This is a five-task simulation study with structured state, twelve demonstrations per task, one training seed, and fifteen OOD layouts per task.
2. Layouts were previously examined; the execution correction and new deployment are declared post hoc. No fresh untouched holdout is claimed.
3. APPL changes segmentation, model count, total training budget and online control together. It is not a compute-matched or runtime-agent-only ablation.
4. The official API route and API sampling differ from historical gateway deployments. Cross-round score changes cannot be attributed exclusively to model reasoning or gripper correction.
5. Overlap of successful trajectories is not equivalent to demonstrated failure recovery. A documented prior is not a verified robustness guarantee.
6. Early `finish` is part of the evaluated agent behavior. Removing it or forcing all 5,000 steps would be a different experiment.
7. Success is geometric and can precede release. This study does not demonstrate stable terminal rest, real-robot transfer, vision robustness, or new-object generalization.
8. Recovery examples are selected qualitative case studies. No recovery success-rate denominator or causal necessity claim is established.

**Supported conclusion:** On this matched position-shift benchmark, the complete APPL pipeline achieves a higher aggregate success count than the two single-policy baselines, with strong task-dependent variation. Logged policy changes demonstrate that an inference-time API can use prior/handoff documents and measured failure feedback to recover in some episodes. Remaining failures show that online selection is constrained by the state coverage and capabilities of the learned controller library.
