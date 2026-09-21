# PRIOR: red_buffer_to_goal_finish__h01

## Assigned heuristic

This policy implements heuristic 1 for `red_buffer_to_goal_finish`: after the blue block has been placed at its lower goal, the final skill should reacquire the red block from the central buffer, move it to the upper red goal, and preserve the demonstrated release/retreat context. The assigned research hypothesis is an active-red final-transfer prior: represent the manipulation state in red-relative and red-goal-relative coordinates so the model can identify the approach, grasp/lift, carry, lowering, release, and retreat phases without relying only on absolute time.

The expanded M1_v2 slice is the full interval starting at index 700 and ending at the demonstrated terminal state for each trajectory. The shared [700, 860) interval is intentionally included. It contains the predecessor-to-successor transition: blue has just been released at blue_goal, the TCP retreats/travels toward red, descends to red, closes the fingers, and begins lifting red. This model is therefore trained to take over partway through that transition, not only after red is already grasped.

## Executable adaptation

The provided execution interface requires a learned DDPM epsilon predictor over normalized native actions: seven absolute Panda joint targets plus one gripper command. No analytic inverse kinematics or forward kinematics provider is available, and the sampler cannot diffuse a custom TCP action space directly. I therefore implement the heuristic as a learned representation prior rather than as a scripted TCP controller:

- The denoiser still outputs epsilon for the required normalized joint/gripper action trajectory.
- The global conditioning vector is produced by a learned encoder fed with causal active-red geometric features.
- Auxiliary trainable heads predict future red geometry labels during training. These predictions use future observations only as labels; deployment forward uses only the two causal observations supplied to `model.forward`.
- One auxiliary head also receives a differentiable estimate of the clean action sample and predicts future red lift/goal error. This provides an action/state coupling loss without claiming analytic dynamics or IK.

All observation normalization uses the single shared normalizer supplied for the complete original training demonstrations. The model does not fit per-skill or per-slice scales.

## Architecture

`policy.py` defines `RedFrameFinalTransferPolicy`.

1. **Causal feature map**
   - Inputs: `raw_history [B, 2, 47]` only.
   - Includes the shared normalized raw observations flattened across the two history steps.
   - Adds red-relative and goal-relative vectors for each history step:
     - `tcp_xyz - red_xyz`
     - `red_goal_xyz - red_xyz`
     - `blue_xyz - red_xyz`
     - `red_xyz - red_goal_xyz`
     - `blue_xyz - blue_goal_xyz`
     - `tcp_xyz - red_goal_xyz`
     - `tcp_xyz - blue_xyz`
     - `blue_goal_xyz - blue_xyz`
     - `red_goal_xyz - blue_goal_xyz`
   - Adds TCP/red/blue height, finger positions and aperture, qvel, two-step TCP/red/blue deltas, and scalar distance cues such as TCP-red distance and red-goal error.
   - Relative positions are scaled by a broad fixed metre scale from `pipeline.json` (`0.25 m`), not by a narrow final-skill empirical range.

2. **Condition encoder**
   - MLP with SiLU activations and LayerNorm maps the feature vector to a 256-dimensional global condition.

3. **Diffusion denoiser**
   - Uses the supplied `appl.public.DiffusionBackbone`, a conditional 1-D U-Net, with the 256-dimensional condition.
   - The model predicts epsilon for the fixed DDPM process over action sequences of horizon 16.

4. **Auxiliary learned heads**
   - `future_head(condition)` predicts a 16-step sequence containing red relative goal error `(red_xyz - red_goal_xyz)` and red lift height `(red_z - red_goal_z)`.
   - `final_error_head(condition)` predicts the last valid red goal error in the sampled horizon.
   - `action_effect_head(condition, estimated_clean_action, timestep)` predicts the same future red geometry from the denoised action estimate. Its loss is downweighted by `alpha_bar` so high-noise diffusion steps do not dominate through the clean-sample reconstruction factor.

## Losses and gradient paths

`compute_loss` returns `loss`, `diffusion_loss`, and `prior_loss`.

- `diffusion_loss`: standard masked epsilon MSE between the predicted and sampled DDPM noise. This trains the backbone, condition encoder, and action denoising path.
- `prior_loss`: weighted sum of differentiable auxiliary losses:
  - masked MSE for predicted future red goal-relative trajectory and lift height,
  - MSE for predicted final red goal error at the last valid future slot,
  - masked, alpha-weighted MSE for action-conditioned predicted red geometry.

These losses are not constants computed from observations alone. They are functions of trainable predictions from the condition encoder, auxiliary heads, and in the action-effect term the predicted diffusion epsilon through the estimated clean action. Future observations are used only as supervised labels.

## Causal deployment information

At deployment, `model.forward(noisy_action, timestep, raw_history)` receives only the current two-step observation history. The features that let the policy take over during the shared transition are:

- `blue_pose - blue_goal`: whether the predecessor has already placed blue at the lower target.
- `red_pose` relative to the central buffer and `red_goal - red_pose`: whether red is still buffered, lifted, or near the final goal.
- `tcp_pose - red_pose`: whether the TCP is still retreating from blue, approaching red, at grasp height, carrying red, lowering at the goal, or retreating.
- Finger aperture and finger velocities: open approach/release versus closed carry.
- Two-step TCP/red/blue deltas: motion phase and whether red is moving with the TCP.

The policy does not use the future labels or any non-causal timing signal during inference.

## Applicability, termination, and limitations

Applicable state support is blue already at blue_goal with red in the central buffer, or a later overlapping transition state in which the TCP is approaching, closing on, or lifting red. The intended terminal state is both task predicates true: red at red_goal and blue at blue_goal. Demonstrations include gripper opening and TCP retreat above the red-goal area, and the model learns this context, but the task contract does not require a gripper release or clearance predicate.

Limitations are inherited from the data and interface. This is not a recovery policy for an incorrectly placed blue block, a missing red block, or a heavily perturbed buffer pose. It has no image encoder, no explicit IK, no hard containment constraint, no collision checker, and no scripted fallback controller. Any robustness must arise from the learned diffusion model and the geometric conditioning.
