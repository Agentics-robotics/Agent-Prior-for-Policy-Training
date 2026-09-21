# Prior document: red_buffer_to_goal_finish__h02

## Assigned heuristic

The assigned hypothesis is the **task-predicate termination prior** for the final skill of the buffer-swap task.  The final policy should not merely replay a fixed-duration terminal motion; it should learn explicit representations of the task predicates `red_at_goal` and `blue_at_goal` so that the inference layer can stop or transfer to a task monitor when the contract is satisfied.

The policy covers the full expanded slice `red_buffer_to_goal_finish`, including the shared transition from the predecessor skill.  In the observed demonstrations the blue block is already at `blue_goal` at the start/overlap, while the red block is still at the central buffer.  The segment then includes red acquisition, red lifting, transport to `red_goal`, placement, gripper opening, and the demonstrated retreat context.  The final task success contract is only `red_at_goal AND blue_at_goal`.

## Executable adaptation

The repository supplies a DDPM action diffusion training and sampling loop.  This implementation keeps that interface unchanged: `forward(noisy_action, timestep, raw_history)` predicts action noise for normalized 8-D joint/gripper action sequences.  The prior is implemented as trainable state representation and auxiliary losses, not as a scripted controller.

The adaptation from the heuristic to this API is:

1. A causal encoder consumes the two available low-dimensional observations: qpos, qvel, TCP pose, red pose, blue pose, drawer compatibility channels, red goal, and blue goal.
2. The encoder uses the assignment's single shared M1_v2 normalizer fitted on complete original demonstrations.  It does not refit per-skill or per-segment scales.
3. Additional world-frame metric features are computed causally from the latest two observations: red-to-goal and blue-to-goal offsets, TCP-to-red and TCP-to-red-goal offsets, short-horizon motion deltas, predicate margins, finger width, and velocity summaries.  These features use fixed physical scales such as 0.05 m, 0.10 m, and 0.20 m, not empirical tiny slice statistics.
4. The encoded condition drives the standard conditional 1-D diffusion U-Net from `appl.public.DiffusionBackbone`.
5. Trainable predicate heads share the encoder.  Their labels are computed from observations using the completion contract: target half width 0.06 m, square block half width 0.02 m, full rotated xy containment, and z tolerance 0.011 m.

## Architecture

- Causal normalized history: `[B, 2, 47]` flattened after shared normalization.
- Causal metric feature vector: 39 dimensions in world units with fixed physical divisors.
- Encoder: MLP with LayerNorm and SiLU, output dimension 256.
- Diffusion action model: `DiffusionBackbone(condition_dimension=256)` with the assignment training configuration.
- Current predicate head: MLP from the encoder latent to two logits `[red_at_goal, blue_at_goal]`.
- Future predicate head: MLP from encoder latent, a learned per-step embedding, and the predicted clean action estimate for that DDPM step.  This head is used only in training as an auxiliary learned action/predicate coupling.

The model has no image encoder, no inverse kinematics, no forward kinematics provider, no hard constraint projection, and no rule-based action generation.

## Losses and gradient paths

The returned training objective is:

`loss = diffusion_loss + prior_loss`

where `diffusion_loss` is the standard epsilon MSE with the provided horizon mask.  The `prior_loss` is a weighted sum of:

1. **Current predicate BCE**: binary cross entropy from the causal encoder latent to current `red_at_goal` and `blue_at_goal` labels.  This trains the shared representation used by the diffusion condition.
2. **Future action-conditioned predicate BCE**: binary cross entropy from the predicted clean action estimate `x0`, the latent state, and step embedding to future predicate labels from `future_obs`.  This provides a differentiable path through the denoising prediction as well as through the auxiliary head.
3. **Success open-gripper loss**: when future labels indicate both predicates true, the predicted native gripper command is weakly encouraged toward open (`+1`).
4. **Settled success hold loss**: in demonstrated settled success states with open fingers, high TCP, and low qvel, predicted joint targets are weakly encouraged to stay near the observed settled qpos.

The future/action auxiliary terms use a clamped clean action estimate and are weighted by `sqrt(alpha_bar)` to reduce the influence of very noisy diffusion steps.  These terms are intentionally weak relative to the diffusion objective so the model remains a learned behavior cloned from the demonstrations rather than a hand-coded terminal controller.

## Causal deployment inputs

At inference, the model receives only the two most recent raw observations and the DDPM noisy action sample/timestep.  It can condition on:

- `red_pose` and `red_goal` to identify whether red is still buffered, lifted, descending, or placed.
- `blue_pose` and `blue_goal` to maintain awareness that the predecessor condition remains satisfied.
- `tcp_pose`, qpos, qvel, and finger width to distinguish approach, grasp/lift, place/release, and retreat phases.

Future observations are used only as training labels.  They are not available to `forward` at deployment.

## Handoff behavior over the expanded slice

The slice begins in a transition where blue is at its goal and red remains at the buffer.  This policy can take over there because the causal observations include both object poses and both goals: `blue_pose - blue_goal` is small, while `red_pose - red_goal` is large and red z is still near the table.  During the middle of the slice, red z and TCP-red proximity indicate grasp/lift/transport.  Near placement, red xy is close to `red_goal` but z may still be high, so the predicate labels remain negative until the block is down within z tolerance.  After success, the demonstrations show gripper opening and TCP retreat, but the task monitor can complete earlier when the contract is geometrically true.

## Applicability and limitations

This prior is appropriate for the assigned final buffer-swap skill with no drawer and with the same completion contract.  It relies on successful demonstrations; negative predicate examples are mainly earlier states in the same trajectories, not adversarial near-boundary failures.  The policy does not guarantee recovery if blue is moved away, if red starts far outside the demonstrated buffer/transport configurations, or if contacts differ substantially from training.  It supplies learned predicate cues and learned action diffusion outputs, while final success should still be checked by the external task monitor using the official contract.
