# PRIOR: acquire_block__h02

## Assigned research hypothesis

The assigned heuristic is a contact-phase bottleneck for pickup. The acquisition motion is treated as a sequence of latent phases:

1. approach,
2. align,
3. close,
4. lift,
5. carry.

The motivation is that successful red and blue pickups contain a discrete contact/closure event. In the demonstrations, the gripper approaches open, closes near the block, the object rises with the TCP, and then the carried object enters an overlap region where a placement policy can also act. Making this bottleneck explicit should reduce gripper-command averaging and improve timing around contact.

This implementation keeps the assigned expanded acquire_block slices: red acquisition from indices 0-250 and blue acquisition/transition from indices 330-625 for the twelve demonstrations. It therefore includes the shared transition phases with place_block rather than creating a new boundary.

## Executable adaptation

The repository provides only state observations and joint/gripper actions, not force sensing, images, inverse kinematics or a contact detector. I therefore implement the hypothesis as a learned state-conditioned DDPM action denoiser with auxiliary latent-phase supervision derived from the demonstration states. The labels are training targets only; deployment uses only the two causal observations passed to `forward`.

The active object for auxiliary labels is inferred from the current causal state. If red is low and within roughly 9 cm XY of the red goal, blue is treated as the active remaining object; otherwise red is active. This rule matches the expanded slices: the first slice acquires/carries red, while the second begins after red placement and acquires/carries blue. It is used to build features and training labels, not to emit scripted actions.

Phase labels are heuristic state labels:

- approach: active block is on the table and the TCP is not yet locally aligned;
- align: TCP is close in XY and low above the active block while fingers remain open;
- close: fingers are closing or closed while the active block is still near the table;
- lift: active block has risen above the table but is not yet in the high carry band;
- carry: active block is lifted with closed fingers.

These labels are imperfect proxies for true contact, but they make the assigned bottleneck differentiable and trainable with the available data.

## Architecture

`policy.py` defines `ContactPhaseBottleneckPolicy`.

- The input to `forward` is `raw_history [B,2,47]`, the noisy action sequence, and the DDPM timestep.
- Observations are normalized with the single shared M1_v2 normalizer supplied by the assignment. No per-skill normalizer is fit.
- A causal feature encoder consumes the normalized two-step history plus world-frame relative features: TCP-to-red, TCP-to-blue, active-object-to-TCP, object-to-goal offsets, object heights, finger width and short-horizon state deltas.
- The encoder predicts current phase logits and a horizon of phase logits for the 16 action slots.
- Phase probabilities are concatenated with the learned state embedding and projected to a 384-dimensional global condition for the provided conditional 1D diffusion U-Net.
- A small learned residual epsilon decoder also receives the predicted per-step phase probabilities, noisy action, diffusion timestep embedding, state embedding and learned step embedding. This makes the denoising head explicitly phase-conditioned, including the gripper channel, while preserving a learned diffusion policy.

The model predicts DDPM epsilon for all eight action dimensions: seven absolute Panda joint targets and one gripper command. It does not compute IK, collision constraints, or a scripted action sequence.

## Losses and gradient paths

The total loss is the diffusion epsilon loss plus weighted prior losses.

1. **Diffusion loss**: `public.epsilon_loss(predicted_noise, noise, mask)` trains the complete action diffusion model.
2. **Current phase cross-entropy**: current phase logits from the causal encoder are trained against the state-derived phase label for the last observation.
3. **Horizon phase cross-entropy**: the 16 predicted phase logits are trained against phase labels computed from `future_obs`, masked by `future_mask`.
4. **Future active z and finger-width prediction**: a learned auxiliary head predicts normalized future active-object height and finger width. This is a trainable prediction; the loss is not a constant penalty on observed poses.
5. **Low-noise gripper x0 loss**: for sufficiently low DDPM noise levels, the predicted epsilon is converted to an estimated clean action, denormalized, and its gripper channel is weakly penalized against the demonstrated native gripper command. This gives the phase-conditioned denoising path an additional contact-timing gradient without replacing diffusion training.

The prior loss weights are intentionally small relative to the DDPM loss: 0.05 for current phase CE, 0.05 for horizon phase CE, 0.05 for z/finger auxiliary MSE, and 0.02 for the low-noise gripper x0 loss.

## Causal deployment inputs

At deployment, `forward` receives only:

- two causal observations containing qpos/qvel, TCP pose, red pose, blue pose, fixed zero drawer compatibility channels, and red/blue goal coordinates;
- a noisy action sample and DDPM timestep from the sampler.

No future observations, segment IDs, demonstration indices, success flags, or external schedules are used. Future observations are used only inside `compute_loss` as training labels for auxiliary predictions.

## Handoff and transition behavior

The policy is intended to take over when fingers are open and a target block remains on the table, or after red has been placed and the state indicates the blue acquisition transition. The observation fields that support this are finger qpos, TCP pose, object poses and the fixed goal positions in world coordinates. The learned active-object and phase features allow the same model to cover the red pickup and the red-to-blue transition without a preset time index.

A useful exit state is a carry/lift state: closed fingers, active object lifted, and small TCP-object relative translation. The demonstrations support overlap with place_block from lifted carries and early descent/carry states, so exact tray hover is not required before transfer. If the object does not rise or does not track the TCP, the model should be continued or recovery should be selected rather than handing off as a successful grasp.

## Limitations

This implementation does not claim true contact detection, force feedback, hard safety constraints, image invariance, IK, or geometric optimality. The phase labels are heuristic labels inferred from successful demonstrations and may be wrong in off-distribution states. The active-object rule assumes the red-before-blue order present in the assigned data. Recovery from slips, collisions, partial grasps, variable block sizes, or substantially different friction is untested.
