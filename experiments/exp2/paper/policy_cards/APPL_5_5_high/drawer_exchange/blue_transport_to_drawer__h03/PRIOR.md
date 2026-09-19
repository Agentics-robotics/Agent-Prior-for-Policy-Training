# Prior for `blue_transport_to_drawer__h03`

## Original heuristic identity

This implementation keeps heuristic 3 for the `blue_transport_to_drawer` skill: **high-arc transport then descent prior**. The hypothesis is that the blue block should first be lifted to a safe clearance, then translated laterally over the workspace while high, and only then descended into the open drawer cavity for insertion and release. The assigned expanded slice is kept in full: actions 810 through 1030 for each training demonstration, including both shared transition phases.

## Executable adaptation

The source heuristic refers to a temporal latent state `s_t = {lift_high, translate, descend_insert}`. The available interface provides state histories and joint/gripper action labels, but no image encoder, IK solver, collision checker, or analytic forward-kinematics module. I therefore implement the heuristic as a **learned conditional action diffusion model** whose conditioning vector includes a trainable stage representation inferred causally from the two latest observations.

The causal stage labels used for supervision are:

- `descend_insert`: blue-to-blue-goal xy error is less than 0.060 m;
- `translate`: otherwise, blue z is greater than 0.245 m;
- `lift_high`: otherwise.

This rule is an implementation label for the prior, not a scripted controller. The diffusion model still predicts normalized native joint/gripper action trajectories and is trained by DDPM epsilon prediction.

## Architecture

`policy.py` defines `StageAwareHighArcPolicy`.

1. The two-observation state history is normalized with the assignment's shared M1_v2 normalizer fitted on complete original demonstrations. No per-skill or tiny-slice normalization is refit.
2. Causal engineered features are appended to the normalized history: blue-goal, TCP-blue, TCP-goal, blue-drawer and red-pad relative positions; two-step TCP and blue deltas; drawer open margin and velocity; finger widths; arm velocities; and soft indicators for high carry, xy alignment, lift, translate, descend, near insertion z, open gripper and closed gripper.
3. A trainable MLP encodes these features into a 256-dimensional condition. A separate trainable current-stage head predicts the three stage logits; its softmax is embedded and added to the condition, so the U-Net is explicitly conditioned on a learned high-arc stage representation.
4. The action diffusion backbone is the provided `DiffusionBackbone`, a conditional 1-D U-Net that predicts epsilon for `[B, 16, 8]` normalized action samples.
5. An auxiliary learned head receives the condition and the denoised clean-action estimate `x0` and predicts future blue-goal relative position, future stage logits, blue-inside probability, and gripper-open probability over the 16-step horizon.

## Losses and gradient paths

The main loss is the standard diffusion epsilon MSE from `appl.public.epsilon_loss`.

The prior loss is differentiable and contains only trainable predictions:

- future blue-to-goal relative position MSE from the auxiliary head;
- future stage cross entropy for `lift_high`, `translate`, and `descend_insert`;
- future blue-inside BCE using the task drawer-cavity z and xy containment rule as the label;
- future gripper-open BCE;
- a direct denoised gripper command loss encouraging open commands in the release overlap;
- current-stage cross entropy for the trainable stage head.

The future auxiliary terms depend on `x0 = (noisy_action - sqrt(1-alpha_bar) * predicted_epsilon) / sqrt(alpha_bar)`, so their gradients reach the denoising network as well as the auxiliary and conditioning modules. The clean estimate is clamped to a moderate range and weighted by `alpha_bar` confidence to avoid high-noise DDPM steps dominating the auxiliary objectives.

## Causal deployment inputs

Deployment `forward` receives only `raw_history [B, 2, 47]`, the current noisy action sample, and the diffusion timestep. It uses these causal observation fields:

- qpos/qvel including gripper finger width;
- TCP world pose;
- blue block world pose;
- red block pose and red goal for context;
- drawer position/velocity;
- blue goal position supplied in the observation.

No future observations are used at inference. Future observations are used only as training labels for the auxiliary losses.

## Handoff coverage

The policy is trained over the full expanded slice `[810, 1030)`. At the start, demonstrations show the blue block near table height with TCP nearby and the gripper closing or closed; by around index 870 the block is high, about 0.27-0.28 m. Around index 930 the block is laterally near the drawer insertion goal while still high, and by the end it has descended to approximately z = 0.063 m with fingers open/opening. This observation sequence is exactly the transition pattern encoded by the stage-conditioned prior.

The model can take over during the predecessor overlap because its causal inputs reveal whether the current state is still lift_high or already high translate: blue z, blue/TCP relative pose, finger width, and xy distance to the drawer goal. It can remain active into the successor overlap because those same inputs distinguish a merely aligned high block from a descended inserted block.

## Applicability and limitations

This prior is intended for the assigned state-based environment with unchanged drawer geometry and an already open drawer. It assumes the blue block is grasped or in the demonstrated lift/transport contact relation; the state does not directly prove contact. The implementation does not claim image invariance, analytic collision avoidance, IK, or scripted recovery. It learns from successful high-arc demonstrations only, so unusual disturbances, missed grasps, or alternative shorter paths are outside the documented support.
