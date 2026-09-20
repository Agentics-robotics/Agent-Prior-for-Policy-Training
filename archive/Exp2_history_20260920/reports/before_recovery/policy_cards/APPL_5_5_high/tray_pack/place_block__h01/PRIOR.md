# PRIOR.md: place_block__h01

## Assigned heuristic

This policy implements heuristic 1 for `place_block`: **Active-goal placement funnel**. The research hypothesis is that red and blue placement share the same geometric structure: while a block is grasped or entering the placement funnel, the useful low-level behaviour is to reduce that block's residual position error to its own tray target, descend until the block is supported, open the fingers, and retreat without disturbing the other block. The assigned expanded M1_v2 slice includes the early carry-in overlap for red `[185,250)`, the red release/retreat and transition to acquiring blue `[330,420)`, and the corresponding blue carry-in and final placement `[580, stop)`.

## Executable adaptation

The handoff description mentions an active role selector. The provided policy interface does not pass a segment id or an external role label at deployment, so the implementation uses only causal state observations to infer role information. It computes a differentiable soft red/blue score from current TCP-to-object distance, object height above its goal, and current goal error. The network receives both colour-specific features and role-pooled features, so it can still learn from absolute colour context when the soft role is ambiguous during transition phases. This is an implementation adaptation, not a scripted controller.

No inverse kinematics, contact model, image encoder, or hard geometric constraint is implemented. The model remains a learned action diffusion policy that predicts DDPM epsilon for normalized 8-D native actions: seven Panda joint targets plus the gripper command.

## Architecture

`policy.py` defines `ActiveGoalFunnelPolicy`:

- A causal observation featurizer uses the two available raw observations `[B,2,47]`.
- It includes the shared M1_v2 observation normalization fitted on all complete demonstrations; no per-skill or per-slice normalizer is refit.
- It adds engineered goal-frame features: red/blue block residuals to their goals, TCP-to-red/blue offsets, red-blue and goal-goal offsets, object/TCP short-horizon deltas, quaternions, finger opening, causal soft role probabilities, active-object residuals, active TCP offset, inactive-object residuals, and release/funnel cues.
- A learned MLP maps these features to a 256-D global condition.
- The supplied conditional 1-D U-Net diffusion backbone maps `(noisy_action, timestep, condition)` to predicted noise.
- An auxiliary learned head receives the same condition plus the denoised action estimate `x0` and predicts horizon-wise active-object placement residual, TCP-to-active-object offset, and finger opening progress.

The action sampler, DDPM schedule, EMA, optimizer, and final decoding are supplied by the framework.

## Losses and gradient paths

The main loss is the standard masked epsilon-prediction loss against the sampled DDPM noise.

The prior loss is an auxiliary supervised loss on trainable predictions, not a constant geometric penalty. During training, `compute_loss` forms the denoised action estimate

`x0 = (noisy_action - sqrt(1-alpha_bar) * predicted_epsilon) / sqrt(alpha_bar)`

and passes `x0` with the learned condition through the auxiliary head. Targets are computed from `future_obs` for the active block selected by the causal soft role from the current observation. The head predicts:

1. active block residual to its goal, normalized by broad metre-scale constants,
2. TCP-to-active-block offset,
3. normalized future finger opening.

The auxiliary error is masked by `future_mask`, downweighted at high noise through an `alpha_bar` factor, and given extra weight near the placement funnel/release state where the future active block is close to its goal and the fingers are opening. Its configured weight is 0.05 relative to the diffusion loss. Gradients flow through the auxiliary head, the observation encoder, and the predicted epsilon via `x0`, so the denoising model is encouraged to represent actions that are consistent with goal-funnel convergence.

## Causal deployment inputs

At inference, `forward` uses only `raw_history` for the current and previous observation. It does not read future observations, labels, segment ids, demonstrations, or a script. Future observations are used only in `compute_loss` as training targets for the auxiliary head.

The state fields used are the standard world-frame M1_v2 fields: qpos/qvel, TCP pose, red and blue poses, drawer compatibility channels, and red/blue goals. Drawer channels are passed through the shared normalized observation but are constant zeros for this task.

## Handling expanded handoff slices

Because the expanded slice starts before final descent, the model sees lifted blocks near the TCP and can take over during carry-in. The early overlap examples teach the diffusion model to continue moving the held object toward the appropriate goal rather than assuming it starts exactly above the target. The red tail overlap `[330,420)` is intentionally retained: observations show red already at its goal, fingers open, and the TCP retreating or moving toward blue. The policy conditions on finger state, red goal residual, blue goal residual, and TCP/object geometry, which lets it learn that this part of the slice is a release/retreat/transition phase rather than another descent command.

## Applicability and limitations

The prior is applicable to the same state-based tray-packing setting with fixed/open tray geometry and observed red/blue target positions. It assumes the active block is grasped or at least in the demonstrated placement funnel and that the robot can execute similar absolute joint targets. It is not expected to recover reliably from gross failures such as opening far above the tray, losing the block outside the tray, or moving the inactive block significantly. It also does not prove transfer to clutter, unseen tray poses, or different object geometry.
