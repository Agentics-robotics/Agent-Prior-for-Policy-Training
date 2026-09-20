# Concurrent-goal preserving lookahead — insert_blue__h03

## Assigned hypothesis and scope

The source heuristic is retained unchanged in the assignment. Its central statement is: **“Concurrent-goal preserving lookahead: rerank diffusion proposals by predicted containment progress and retention of red/drawer goals. Hypothesis: short-horizon checks reduce destructive terminal actions and improve simultaneous success beyond imitation alone.”** It proposes an action-conditioned transition ensemble, conjunctive geometry, support penalties and state-dependent progress rather than an encoder-only geometric prior. This document describes implementation choices and limitations separately; it does not amend that source heuristic.

The policy covers every action in the assigned **expanded** slices, including red placement/release, open-finger retreat and blue approach, closure, initial lift, transport, insertion and the retained terminal continuation. No phase is dropped, resampled away, or replaced with a scripted controller. The half-open source ranges are:

| Demonstration | Assigned actions | Shared actions with evacuate_red |
|---|---|---|
| demo1000 | [575,1065) | [575,845) |
| demo1001 | [581,1077) | [581,859) |
| demo1002 | [575,1063) | [575,847) |
| demo1003 | [575,1066) | [575,845) |
| demo1004 | [575,1065) | [575,846) |
| demo1005 | [582,1070) | [582,852) |
| demo1006 | [575,1064) | [575,846) |
| demo1007 | [575,1069) | [575,847) |
| demo1008 | [575,1071) | [575,853) |
| demo1009 | [575,1065) | [575,846) |
| demo1010 | [575,1069) | [575,849) |
| demo1011 | [575,1065) | [575,846) |

## Necessary interface adaptation

The available deployment hook predicts epsilon for **one** noisy action chunk. The frozen DDPM sampler provides no post-sampling candidate-selection hook and owns its noising/sampling schedule. It would be misleading to claim that this package samples K independent full diffusion chains and reranks them online.

Instead this is an **amortized lookahead-selection objective**:

1. Train the ordinary history-conditioned action diffusion proposal.
2. Independently train three action-conditioned short-horizon transition models, a causal future-work forecast, and a Gaussian behavior-support surrogate from demonstration labels.
3. During training, use the supplied alpha_bar to recover the denoiser's differentiable clean estimate. Construct five small, supported local alternatives around its detached value, predict their consequences, and select using the concurrent-goal score.
4. Distill a trusted selected alternative back into the differentiable clean estimate, and hence into the diffusion model. Detach the selection and all critic computations on counterfactual candidates. Do not update a critic to reward an invented outcome.
5. At deployment use the **same ordinary epsilon forward** and the fixed 100-step DDPM sampler. The causal forecast conditions the denoiser; the dynamics ensemble and behavior-support model do not run during deployment. Reobserve after the supplied eight-step execution prefix.

Thus selection is real and outcome-dependent in training, but its effect is amortized, not recomputed as runtime planning. The five alternatives are perturbed diffusion clean estimates, **not five independent full DDPM samples**. This is a testable objective-prior variant of the assigned research hypothesis, with less online adaptivity than the source proposal. It is not a claim of runtime ranking, calibrated online uncertainty, or trained recovery. A true online K-chain comparison requires an additional sampler API. No local replacement scheduler, action controller, IK or replay is introduced.

Only 16 future observations are supplied per training example. Consequently the progress head estimates **short-horizon remaining geometric work**, not full steps-to-go or an unobserved far-future return. This is another explicit adaptation of the remaining-progress proposal.

## Data, causality, normalization and alignment

Inputs are two causal 47-dimensional observations: qpos 0:9, qvel 9:18, world TCP pose 18:25, red pose 25:32, blue pose 32:39, drawer position/velocity 39:41, and red/blue aim points 41:47. Positions are metres, joint angles radians, and quaternions wxyz. Increasing gripper action opens the fingers. Finger aperture is observed through qpos[7:9], not inferred from the command alone.

A 153-dimensional feature vector concatenates the two shared-normalized states (94), their difference (47), and four three-vectors: TCP-minus-red, TCP-minus-blue, red-minus-pad goal, and blue-minus-observed aim point. Relative vectors use the corresponding shared translation half-ranges; a TCP/object difference uses the root-sum-square of their shared half-ranges. The history difference is a displacement, not a velocity estimate with an invented sampling period. The measured qvel and drawer velocity remain available.

All observation and absolute-action normalization is the supplied single fit over the complete original training demonstrations. The normalizer identifier is `e75d5e1c8d9862f1611bf577ea5fa6e6d3fd5a25a69627c1f55602c602a7193f`. The supplied `std` fields are range half-widths, not empirical standard deviations. No per-skill fit or narrow empirical increment scale is estimated. Quaternions retain unit-component scaling. The framework supplies shared-normalized eight-dimensional actions and decodes the final DDPM sample. Proposals are bounded in that common [-1,1] action encoding.

At current observation t, action slot 0 denotes action t-1 and its successor is the already observed state s_t. The transition ensemble therefore anchors slot 0 exactly to the current observation and predicts only slots 1 through 15. It does not apply the past action a second time. Dynamics losses, forecast losses and scoring exclude slot 0. Increment targets use the current observation for the first predicted transition, then adjacent valid future targets. Both action and future masks are respected; nothing crosses the assigned slices. The diffusion and behavior imitation losses still use the supplied action mask over all slots.

Future observations are used **only as masked training labels** for dynamics, progress and phase-relevance predictions. At deployment, every conditioning feature and forecast is computed solely from raw_history. There is no trajectory identity, source index, elapsed demonstration clock, future state, phase schedule, or stored action table in forward.

## Learned architecture

- A 153 -> 256 -> 256 history encoder (LayerNorm/SiLU) conditions the public Conditional U-Net.
- The progress MLP has two 192-wide hidden layers and predicts 16 sets of six values: four nonnegative future-work descriptors plus probabilities of red containment and near-placement relevance. Its 96 predicted values are concatenated with the 256-dimensional encoder output, giving 352 conditioning dimensions. This forecast is detached at the diffusion interface so epsilon training cannot repurpose its semantics.
- The U-Net uses the fixed 128/256/512 widths, 128-dimensional timestep embedding, kernel 5, groups 8, horizon 16 and epsilon prediction. It emits learned absolute-action diffusion noise, never direct scripted actions.
- Each of three independent transition members initializes a 192-dimensional recurrent state from the causal features. A GRUCell receives the previous predicted normalized state (41 coordinates) and the current proposed encoded action (8). A residual MLP predicts state increments with a fixed 0.1 multiplier in **shared** normalized coordinates. Rollouts are open-loop; ground-truth future states are not teacher-forced into them. TCP/red/blue quaternion predictions are unit-normalized after each step.
- The behavior model is a two-hidden-layer 256-wide MLP producing a tanh mean and log standard deviation for every encoded action. Log standard deviations are bounded to [-3,0]. It is a conservative behavior-support **surrogate**, not exact diffusion likelihood or a second runtime controller.

Coupled-object information is available as relative pose, two-frame co-motion, finger positions and robot velocity. There is no explicit measured contact state, object attachment constraint, or hard classifier declaring a grasp. The recurrent dynamics must learn action/object coupling from these signals. Relative features do not make absolute joint commands equivariant.

## Exact geometry and the learned progress forecast

For each cube, the predicted or observed unit quaternion gives its rotation R. World-axis XY extents are `e_i = 0.02 * sum_j abs(R_ij)`. The nine signed margins are:

- drawer: d - 0.26;
- red XY: (0.06,0.06) - abs(red_xy - (-0.18,-0.30)) - e_xy;
- red lower/upper center height: red_z - 0.014 and 0.031 - red_z;
- blue XY: (0.172,0.182) - abs(blue_xy - (0.19-d,0)) - e_xy;
- blue lower/upper center height: blue_z - 0.053 and 0.074 - blue_z.

Containment uses non-strict XY comparisons and strict height/drawer comparisons. The implementation does **not** use blue_goal as the cavity center. From observations, the demonstrated blue interior aim is (0.125-d,0.075,0.063), about 65 mm behind the cavity center in world x. That aim is used only in the future-work descriptor; it is not a required final pose or the success test. Observed red/blue goal fields remain in the causal feature encoder.

The four work labels at each valid future slot are: summed negative red margins, normalized TCP-blue distance, normalized blue-to-interior-aim distance, and summed negative blue margins. Margins use the corresponding shared physical translation scales. The forecast also learns red-contained probability and a coarse near-placement probability. The latter label requires red containment, blue XY within 0.10 m of the demonstrated interior aim and 0.04 < blue_z < 0.18 m. Those broad cutoffs define **score relevance only**, not admission, handoff or completion. These targets are derived from future training poses; deployment sees only their causal learned predictions.

Tracking predicted near-term work, rather than minimizing immediate blue distance everywhere, allows red lowering, open-finger blue approach and blue lift to remain legitimate progress. For example lifting blue may initially increase blue-to-goal height distance. The learned forecast can predict that increase. If all actual goals already hold, the teacher disables forecast-tracking and near-placement pressure and scores preservation plus uncertainty/support instead.

## Losses and gradient paths

The returned objective is

`L = L_epsilon + 0.30 L_dynamics + 0.10 L_progress + 0.02 L_support + 0.50 ramp L_rank`.

All losses are scalar tensors; `diffusion_loss` is the supplied masked epsilon MSE and `prior_loss` is the remaining weighted sum.

**Dynamics.** Each member uses an independent Bernoulli(0.8) per-example bootstrap mask. Its loss is shared-normalized multistep MSE on qpos, qvel, Cartesian positions and drawer state; plus three times a position/drawer-focused MSE; plus exact-margin regression; plus 0.2 times the summed orientation surrogate; plus 0.5 times valid adjacent physical-increment MSE. The orientation surrogate is `1 - dot(q_pred,q_target)^2 = sin^2(theta/2)`, not a full atan/acos geodesic loss. It is smooth and sign-invariant. All these terms depend on trainable predicted states. Differentiating only observed containment would be constant and is not used as a purported training loss.

**Progress.** Smooth-L1 regression of the four future-work predictions plus 0.25 times binary cross-entropy on the two future relevance labels. These losses train the progress head directly. Epsilon gradients do not flow into this head through its detached conditioning values.

**Behavior support.** Masked diagonal Gaussian negative log likelihood, `0.5*((a-mu)/sigma)^2 + log(sigma) + 3`, averaged over action coordinates. The constant shift has no gradient effect. This trains only the support head.

**Lookahead teacher and rank distillation.** Recover `x0 = (x_t - sqrt(1-alpha_bar)*epsilon_pred)/sqrt(alpha_bar)` using the supplied training alpha_bar, never an assumed sampler schedule. Around its detached bounded value construct five chunks: unchanged, plus/minus temporally smoothed random perturbations, and plus/minus a bounded step toward the behavior mean. The per-coordinate perturbation/step bound is 0.025 encoded units. Smooth random perturbations use an average-pooling kernel of 5. Each candidate is rolled through all three transition models.

At every valid predicted step the risk is:

- 12 times loss of already-achieved drawer/red/blue margins. Preserve the lesser of each observed positive margin and a modest interior buffer: 10 mm drawer, 5 mm XY, 1 mm height. Buffers do not become success thresholds and do not require larger margins than already observed.
- Smooth-L1 deviation from the causal forecast's four future-work values, disabled after observed simultaneous success.
- 0.25 times negative blue-margin squared deficit, weighted by predicted near-placement relevance and disabled after observed simultaneous success.

Aggregate temporally and combine half ensemble-mean risk with half worst-member risk. Add twice the shared-normalized ensemble variance of TCP/object positions and drawer opening, and 0.03 times squared action deviation under the behavior Gaussian. The variance is a disagreement heuristic, **not calibrated uncertainty**.

Select the minimum-score candidate with no autograd tracking. Distill only where alpha_bar >= 0.5, current bounded clean-estimate RMSE to the training demonstration is below 0.12 encoded units, selected ensemble variance is below 0.0025, and score improvement over the unchanged candidate exceeds 1e-5. The demonstration comparison is a **training trust-region label**, not a deployment input or action replay. The sample weight is alpha_bar times improvement/0.01 clipped to [0,1]. Rank loss is masked MSE between differentiable, unclipped x0 and the detached selected candidate, weighted per example. Its gradient flows through x0 to epsilon predictions and the causal encoder. It does not flow into the teacher or support model. The rank term is zero for the first 2,000 updates, ramps linearly over the next 4,000, then receives full weight. Teacher computations are exercised even during warmup so numerical interface checks cover scoring too. This schedule is only a training-critic warmup, not a task phase schedule. High-noise auxiliaries cannot be amplified without bound: ranking is disabled for alpha_bar below 0.5.

Detached argmin is not differentiable with respect to critic scores. The **distillation loss is differentiable with respect to the denoiser**, while supervised transition and progress losses are differentiable with respect to their predictions. This separation prevents counterfactual scoring from training a transition model to imagine success. It does not eliminate exploitation of prediction error.

## Handoff evidence and causal takeover

Actual observations were inspected in demo1000, demo1008 and demo1010; measured support summaries for all 12 trajectories were also read. These observations are evidence of demonstration support, not evidence of this policy's trained performance.

- At demo1000:575, red z=0.21093 m, TCP z=0.21180 m, d=0.300 m and finger sum approximately 0.03649 m. Red is close to the TCP while blue remains at its external source. The policy must start by lowering red, not pulling it toward blue.
- At 595, red z=0.03137 m is still just above its strict upper band. At 605 red z=0.03231 m remains outside that band while d has retreated to 0.29733 m. At 615 red z is approximately 0.020 m and fingers sum to 0.07952 m; TCP z is still only 0.03306 m. Geometric red success is not the whole-task endpoint and does not alone identify the next transition.
- At 640 the open TCP has risen to z=0.16246 m while red stays on its pad. At 760 the open TCP is above the blue source. At 815 blue is near the TCP at z=0.020 m with narrowed fingers. At 844-845 blue rises to z=0.1295-0.1401 m with the TCP. This co-motion supports, but does not prove, a blue grasp. The measured overlap covers this prerequisite sequence rather than only a static handoff pose.
- At 880 blue is near z=0.2804 m; at 960 it descends over the drawer at z=0.1652 m; at 980 its center is about (-0.17194,0.07365,0.06698) m with fingers still narrow. The actual conjunction can already hold before release. At 1000 blue has released/settled around z=0.063 m with open fingers. By 1064 TCP z is approximately 0.3232 m while goals remain satisfied.
- Demo1008 and demo1010 show different source blue XY and release yaw/translation. At their 985 observations blue is about z=0.067 m with narrow fingers. At 1005 it is around z=0.063 m with open fingers, but blue quaternion z components differ (about 0.0256 and 0.0279, versus demo1000's roughly 0.0529 after release). These values motivate full quaternion-dependent containment, not an exact terminal pose requirement.

The causal history exposes which object is close to and moving with the TCP, whether fingers are opening/closing, whether red is already supported on its pad and whether blue is near the source, in transit or in the cavity. The policy has no required starting source index and can take over within the supported shared transition. `HANDOFF.json` gives continuation and escalation cues. A lost-grasp state is not reclassified as a supported phase solely because fingers are closed.

## Completion, continuation and dependencies

The monitor must evaluate the supplied **actual simultaneous** conjunction at one observation: drawer open, red on pad, blue inside. No release, low velocity, clearance or sustained hold is required. There is no mandatory successor. The retained optional continuation has open fingers and a high TCP, but stopping earlier on actual valid conjunction is allowed. Continuing eight-step invocations after success is a task-level choice, not a terminal loop in this model.

Dependencies are only Torch and the provided `appl.public` numerical backbone/loss. There is no file, network, shell, simulator, force/torque, image or kinematics dependency in policy execution. All action proposals remain learned diffusion outputs. The fixed budget is seed 0, 20,000 AdamW updates, batch 128, learning rate 1e-4, weight decay 1e-6, gradient clipping 1, 500-step learning-rate warmup then cosine decay, EMA 0.999, and final-EMA selection. Horizon/history/execution lengths are 16/2/8, with 100 DDPM train and inference steps and clipped samples. No final-performance selection or feedback is available in this design session.

## Limitations and falsifiable expectations

The central empirical question is whether **amortized** local lookahead improves simultaneous goal preservation beyond imitation. It may not: action-conditioned success-only dynamics can fit a nominal path while ranking alternatives incorrectly. The ensemble may share correlated errors, especially on contact, released-object yaw, lost grasps and drawer disturbances. Pessimistic aggregation, trust regions, support likelihood and delayed distillation are conservative heuristics, not certified constraints. A Gaussian support surrogate may miss multimodality. The state-only forecast can also confuse near-identical observations requiring different timing. No exact rigidity, contact conservation, collision avoidance, calibrated uncertainty or grasp guarantees are enforced. Retention penalties operate on predictions, not the simulator.

The adaptation loses online candidate evaluation and therefore cannot promise the source proposal's local replanning benefits. Candidate sampling is training-only and local. Distillation can still favor unwanted pauses or actions that exploit the learned transition model. New failure recovery and rim collision handling are untested. Learned absolute joint actions are not invariant or equivariant merely because relative geometry is included. Unit-quaternion normalization enforces only that representation's unit norm.

Scientifically appropriate future comparisons are plain diffusion, forecast-conditioned diffusion without rank distillation, this amortized lookahead variant, and random local selection with matched training candidate budget. A true online K-chain reranker would be a separate interface-enabled experiment with equal inference sampling budget. Goal violations, large state prediction errors, unsupported actions or no improvement would falsify the claimed benefit. The available checks establish executable gradients and DDPM/EMA compatibility only, not task success, uncertainty calibration or superiority of this prior.
