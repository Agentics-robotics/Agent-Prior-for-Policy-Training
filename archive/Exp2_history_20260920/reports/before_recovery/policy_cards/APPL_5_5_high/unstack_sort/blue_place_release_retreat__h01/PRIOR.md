# PRIOR: blue_place_release_retreat__h01

## Assigned heuristic identity

This policy implements heuristic index 1 for skill `blue_place_release_retreat`: the **blue goal-basin servo prior**.  The research hypothesis is that the final blue placement phase should be represented around the world-frame error from `blue_pose` to `blue_goal`, while preserving cues about TCP-to-blue geometry, gripper aperture and whether red is already at its goal.  The assigned slice is the full expanded segment for each demonstration, starting at index 520 and ending at the final release/retreat state.  Therefore the learned model covers not only the final descent, but also the shared transition from the preceding blue transport skill.

## Executable adaptation

The handoff text proposed an auxiliary final-success critic and terminal blue-goal error loss.  The provided low-level execution interface only calls `model.forward(noisy_action, timestep, raw_history)` and expects an epsilon prediction for the DDPM sampler; it does not provide an inference hook for ranking multiple sampled horizons.  I therefore implemented the critic as a trainable auxiliary representation and action-coupling loss during training, rather than as a separate deployment-time sampler selector.  At deployment the policy remains a standard learned action diffusion model; the auxiliary heads shape the encoder and denoiser during training but no scripted controller or external rejection logic is used.

No analytic IK, no forward-kinematics library, no hard geometric constraint and no image encoder are implemented.  Actions are the native Panda joint targets plus gripper command predicted by diffusion in the framework's normalized action space.

## Causal observations and representation

The model receives only the two causal state observations supplied by the framework.  It uses the shared normalizer fitted on complete original demonstrations through `normalize_observation`; it does not fit or apply a tiny per-skill normalizer.

The learned encoder concatenates:

- the two shared-normalized raw observations;
- world-frame `blue_goal - blue_pose` errors;
- `red_goal - red_pose` errors;
- `blue_pose - tcp_pose` and `blue_goal - tcp_pose` relations;
- gripper finger width, finger balance and finger velocity;
- blue and TCP height relative to `blue_goal.z`;
- continuous soft pad-proximity cues for red and blue;
- bounded quaternion components for TCP/object orientation cues;
- coarse last-minus-previous deltas for blue, TCP, robot joints and fingers.

The relative features use broad task/robot scale constants such as 0.50 m for table-plane goal errors and 0.30 m for vertical/TCP offsets.  These constants are not empirical ranges computed from the assigned slice.

These observations allow takeover during the overlap because the policy can distinguish: high source carry at index 520 (blue far from the blue goal in XY and high in Z, gripper closed), high goal carry near index 590 (small XY error but large positive height), descent/contact (small XY error and reducing height), release (blue at table height and fingers opening) and retreat (blue stationary at the goal while TCP height increases).

## Architecture

`policy.py` defines `GoalBasinDiffusionPolicy`:

1. `GoalBasinEncoder`: an MLP with Mish activations and LayerNorm that maps the two-observation causal state and engineered goal-basin features to a 256-dimensional condition vector.
2. `DiffusionBackbone`: the supplied conditional 1-D U-Net, conditioned on the encoder output, predicts epsilon for `[B, 16, 8]` noisy action horizons.
3. Auxiliary heads used only in training:
   - a state-only success value head from the condition vector;
   - an action-conditioned success value head from the condition vector and the DDPM clean-action estimate;
   - an action-conditioned terminal blue-goal error head.

The fixed framework DDPM sampler, optimizer, EMA and action denormalization are unchanged.

## Training losses and gradient paths

The total loss is

`diffusion_loss + prior_loss`.

- `diffusion_loss` is the standard epsilon MSE using `epsilon_loss(predicted_noise, noise, mask)`.  This is the main learned action diffusion objective.
- `prior_loss` is a weighted sum of:
  - BCE for a state-only learned future-success logit;
  - BCE for an action-conditioned learned future-success logit;
  - SmoothL1 loss for an action-conditioned predicted terminal blue-goal error.

The future-success label is computed from `future_obs` during training only: within the valid 16-step future horizon, both red and blue must be within the goal pad half-width 0.06 m in XY and within 0.011 m of the goal Z.  The terminal-error target is the last valid future blue position minus `blue_goal`, scaled by broad task coordinates `[0.12, 0.12, 0.30]`.

The action-conditioned heads consume the DDPM clean estimate

`x0 = (noisy_action - sqrt(1 - alpha_bar) * predicted_epsilon) / sqrt(alpha_bar)`,

clamped to a bounded range for numerical stability.  Therefore those auxiliary losses have a differentiable path into the denoiser's epsilon prediction as well as into the encoder and heads.  They are weighted by `sqrt(alpha_bar)` so very high-noise diffusion steps do not dominate the prior.  The state-only value head trains the causal goal-basin representation even when the sampled noising level is high.

## Handoff and termination behavior represented by the model

The model is trained on the full expanded slice from index 520 to the end of each demonstration.  In the early overlap it learns continued closed-gripper blue transport toward the goal.  Around the middle of the slice it learns the descent from high goal carry to the table while maintaining blue XY containment.  Later it learns opening the gripper after blue reaches table height and retreating upward without moving either block.

The inference agent may continue this policy while blue is still above or near the blue pad and the TCP/gripper relation indicates the robot is managing the block.  It may terminate when the current state shows red and blue at their goals and the TCP is no longer threatening to push blue.  Demonstrations commonly end with open fingers and TCP retreated upward, but the completion contract itself does not require gripper release or clearance.

## Dependencies and limitations

This prior depends on the state fields specified in the interface: joint positions/velocities, TCP pose, red/blue poses, and fixed red/blue goals.  The drawer compatibility channels are ignored except as zero delta features.  It is designed for the fixed-goal `unstack_sort` data and for starts similar to the observed expanded blue-placement basin.  If blue is not actually in the gripper at entry, if the grasp is slipping, or if blue is far outside the observed carry-to-goal region, the model has no explicit recovery mechanism beyond the learned diffusion distribution.
