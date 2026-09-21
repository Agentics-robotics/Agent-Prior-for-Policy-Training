# Prior for `side_grasp_red_pull_drawer_open__h02`

## Original heuristic identity

The assigned heuristic is a contact-phase automaton for the skill `side_grasp_red_pull_drawer_open`.  Its hypothesis is that the side grasp, drawer pull, release, and retreat behavior is easier to learn if the policy separates five latent phases:

1. approach the red drawer handle/block with the gripper open,
2. close on the red block,
3. pull while the gripper remains closed and the drawer opens,
4. release after the drawer saturates open,
5. retreat upward/away to prepare for the next top-grasp skill.

The assigned expanded slice is the full range 0:330 for each demonstration.  Indices 240:330 are intentionally shared with the successor `red_top_regrasp_lift`, so this policy learns both the end of opening and the release/retreat transition rather than stopping immediately at the geometric open-drawer subgoal.

## Executable adaptation

The handoff hypothesis mentions an HSMM-style latent state estimator.  The provided deployment interface, however, calls `forward(noisy_action, timestep, raw_history)` with only the two most recent observations and no persistent recurrent state.  I therefore implemented a stateless, learned, soft phase belief `p(z | o_{t-1:t})` rather than a persistent HSMM.  The belief is trained from heuristic labels derived from demonstration observations, but it is used only as differentiable conditioning for an action diffusion model.  There is no scripted phase controller and no action replay.

## Architecture

`policy.py` defines `ContactPhaseDiffusionPolicy`:

- Inputs: raw causal history `[B, 2, 47]` containing qpos, qvel, TCP pose, red pose, blue pose, drawer position/velocity, and goals.
- Normalization: the model uses the shared M1_v2 normalizer supplied in `spec['normalizer']`.  It does not fit or apply per-skill empirical scales.
- Causal feature prior: in addition to flattened normalized observations, the encoder receives broad-scale relative features: finger width/asymmetry, finger velocity, drawer opening and velocity, TCP-to-red displacement and change, red-to-drawer-center displacement, goal-relative vectors, TCP height, and remaining drawer opening.  These are observation features, not a controller.
- Phase module: a trainable MLP predicts logits over `{approach, close, pull, release, retreat}`.  The softmax probabilities form a weighted learned phase embedding.
- Auxiliary prediction heads: the same hidden state predicts future drawer opening at action slots 3, 7, and 15, and gripper command intentions at slots 0, 3, 7, and 15.
- Diffusion decoder: the hidden state, soft phase embedding, drawer-progress predictions, and gripper-intention predictions are projected to a 192-dimensional condition vector for the provided Conditional U-Net diffusion backbone.  The backbone predicts epsilon for normalized 16-step, 8-dimensional action sequences.

The model remains a learned diffusion policy: all executed actions are produced by the DDPM sampler using the learned epsilon model.  The phase predictor and auxiliary heads only condition and regularize the learned diffusion model.

## Training losses and gradient paths

`compute_loss` returns:

- `diffusion_loss`: standard epsilon MSE via `appl.public.epsilon_loss`, masked over valid action slots.
- `prior_loss`: a weighted sum of three differentiable auxiliary losses:
  - cross-entropy for the learned phase logits against heuristic phase labels,
  - masked MSE for predicted future drawer opening at slots 3, 7, and 15,
  - masked MSE for predicted gripper command intention at slots 0, 3, 7, and 15.

The total loss is `diffusion_loss + prior_loss`.  The auxiliary losses depend on trainable model outputs, so they provide gradients to the observation encoder, phase classifier, auxiliary heads, condition projector, and indirectly improve the conditioning used by the diffusion backbone.  No loss term is a constant penalty computed only from observed poses.

## Phase labels used during training

Phase labels are computed only for supervision during training:

- `approach`: default when the drawer is still closed and fingers remain open,
- `close`: drawer nearly closed, current fingers still partly open, and future finger observations indicate closing,
- `pull`: fingers are closed or drawer velocity is positive while drawer opening is below about 0.255 m,
- `release`: drawer is open and fingers are still closed or in the process of reopening,
- `retreat`: drawer is open and the fingers are open.

These labels are inferred from qpos finger gap, drawer_position, drawer_velocity, and future demonstration observations.  At deployment, future observations are unavailable; only the learned classifier's causal prediction from the two-frame history is used.

## Causal deployment cues

The model can take over at the beginning of the skill or partway through the transition because the causal observation history includes:

- total finger gap from qpos indices 7 and 8,
- drawer opening and drawer velocity in metres,
- TCP pose and red block pose in the world frame,
- short two-frame TCP/red motion cues,
- red-to-drawer geometric relation and remaining drawer opening.

These cues distinguish the demonstrated modes: open-finger approach, closing/contact, closed-finger pulling with increasing drawer_position, open-drawer release, and upward retreat with open fingers.

## Applicability and termination

This prior is intended for the demonstrated side-grasp opening sequence with an observable drawer_position and red block on the drawer front.  Continue it while the observation suggests the policy is still approaching, closing, pulling, or reopening.  It is ready to hand off when drawer_position is above the open threshold, the gripper is open, red is left on the open drawer surface, and the TCP has begun the demonstrated upward/away retreat.

The final task completion rule is not changed by this prior: the evaluator requires the drawer open, red on the outside pad, and blue inside the drawer at the same observation.  This policy only covers the assigned drawer-opening and release/retreat portion.

## Limitations

The implementation contains no IK, no force/contact sensor, no image encoder, no hard geometric constraint, and no guaranteed recovery behavior after a failed pinch, red slip, or drawer overshoot.  The contact phases are heuristic demonstration labels, not measured contact states.  The HSMM idea is approximated by a two-frame soft classifier due to the stateless forward API.
