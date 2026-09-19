# PRIOR: red_place_to_blue_entry__h01

## Assigned heuristic

The assigned heuristic is the **red goal-relative carry/place prior** for the `red_place_to_blue_entry` expanded skill slice.  In the demonstrated task, the red block has already been lifted from the blue block and must be transported to the fixed red pad at `red_goal = [-0.24, -0.25, 0.02]` in world coordinates.  The handoff evidence shows red high in the gripper during the entry overlap, red descending to the red goal around indices 260--280, the gripper opening by the retreat phase, and the robot continuing toward the blue block during the successor overlap.

The hypothesis implemented here is that the action denoiser should receive an explicit learned representation of:

- `red_goal - red_pose.xyz`, the goal-relative placement error;
- `tcp_pose.xyz - red_pose.xyz`, the maintained grasp/carry offset;
- related blue-object and TCP-to-goal vectors needed for the shared transition into blue acquisition;
- causal gripper width and two-step motion cues that distinguish lift/carry, place/release, retreat, and blue approach.

## Executable adaptation

The source heuristic mentions future labels such as time-to-red-at-goal and future red goal error.  The deployment API only supplies two causal state observations and the diffusion sampler only executes action sequences, so I implement these future quantities as **training-only learned auxiliary predictions** rather than as noncausal inputs.  The forward pass at deployment receives only `raw_history [B,2,47]`, the noisy action sample, and the diffusion timestep.  No future observation, scripted servo, inverse kinematics, image encoder, or hard geometric controller is used.

All observation normalization uses the single shared M1_v2 normalizer supplied in the assignment.  The relative features use fixed metre-scale constants (`0.25 m` for object/TCP/goal vectors, `0.05 m` for two-step deltas, and `0.08 m` for finger width).  These constants are not fitted on the expanded skill slice and do not replace the shared normalizer.

## Model architecture

`policy.py` defines `RedGoalRelativeDiffusionPolicy`, a learned epsilon-prediction DDPM policy.

1. **Causal feature construction**
   - The two raw observations are normalized by the shared normalizer and flattened.
   - For each of the two observations, the encoder appends goal-relative and object-relative world-frame features: red goal error, blue goal error, TCP-red offset, TCP-blue offset, red-blue offset, TCP-to-red-goal, TCP-to-blue-goal, red/blue XY distances to goals, red/blue height errors, finger-width features, and smooth red-near-goal/red-high indicators.
   - Current-minus-previous deltas for red, blue, TCP, finger width, and red goal error provide causal phase/motion information.

2. **Learned condition encoder**
   - A small MLP with layer normalization maps the normalized/raw-relative feature vector to a 256-dimensional condition.

3. **Diffusion denoiser**
   - The condition is passed to the provided `appl.public.DiffusionBackbone`, a Conditional U-Net over the 16-step, 8-dimensional normalized action horizon.
   - The model predicts epsilon for absolute Panda joint targets and gripper command.  The fixed framework handles DDPM noising/sampling and action denormalization.

4. **Training-only auxiliary heads**
   - A condition head predicts, for every horizon slot, future red-goal error plus logits for `red_at_goal` and `gripper_open`.
   - An action-conditioned auxiliary head reads a bounded DDPM clean-action estimate `x0 = (x_t - sqrt(1-alpha_bar) * eps_pred) / sqrt(alpha_bar)` and predicts the same future quantities.  This creates a direct differentiable path from the prior loss to the epsilon prediction, while low-alpha noisy estimates are downweighted.

## Loss terms and gradient paths

The returned loss is:

`loss = diffusion_loss + prior_loss`.

- `diffusion_loss` is the standard masked epsilon MSE using `appl.public.epsilon_loss`.  It trains the diffusion backbone and condition encoder to denoise action sequences.
- `prior_loss` is a weighted sum of learned auxiliary losses:
  - Huber loss on predicted future `red_goal - red_pose.xyz`, scaled by `0.25 m`;
  - binary cross entropy for a contract-derived `red_at_goal` label (`|dx| <= 0.04 m`, `|dy| <= 0.04 m`, `|dz| <= 0.011 m`);
  - binary cross entropy for an open-gripper label from future finger width (`qpos[7] + qpos[8] >= 0.06 m`).

The condition-head losses train the shared causal encoder used by the denoiser.  The action-conditioned losses additionally depend on the model's predicted epsilon through the DDPM clean estimate, so the auxiliary objective is not a constant function of observed poses.  Future observations are used only as training targets and are masked by `future_mask`; they are never passed to `forward` during deployment.

## Full expanded slice and handoff behavior

This policy is trained over the full assigned segment `[130, 520)` for each demonstration, not only over the red release instant.  The causal observations let it take over during the predecessor overlap because red is already observable high above the stack with the TCP-red offset and closed fingers indicating a grasped carry.  After placement, the same representation includes red-at-goal features, open fingers, TCP-red separation, blue pose, and blue-goal/TCP-blue relations, allowing the diffusion model to continue through retreat and the start of blue acquisition in the successor overlap.

## Applicability and limitations

The policy is intended for the fixed `unstack_sort` state-based task with the demonstrated top-down grasp geometry and the supplied fixed red and blue goals.  It should not be treated as an object-invariant or goal-invariant placement controller beyond the training support.  It does not prove stable contact or enforce collision/containment constraints; it learns the joint/gripper action distribution from demonstrations.  If red is dropped far from the gripper, blue is displaced substantially, the goals change, or the robot enters with a very different grasp offset, the model may fail because those recovery regimes are not represented in the assigned data.
