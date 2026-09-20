# PRIOR: red_buffer_to_goal_finish__h03

## Assigned heuristic

The assigned research hypothesis is a waypoint-chain transport prior for the final red-block phase of the buffer-swap task.  The demonstrated sequence is: take over during the shared transition after blue placement, acquire the red block from the central buffer, lift it, carry it high, descend over the red goal, open/release, and keep the final retreat context until the swap predicates are satisfied.

This implementation keeps the full expanded slice, including the overlap from indices 700 to 860.  It is therefore trained not only on the red carry and placement, but also on the transition in which the TCP is open/high after blue release, approaches the red block, closes, and starts lifting.  The policy can be invoked at different progress points because its conditioning is inferred from the current causal observation history rather than from a fixed segment clock.

## Executable adaptation

The provided interface exposes two causal observations, a 16-step action chunk, and masked 16-step future states during training.  It does not provide an external IK solver, full-trajectory preprocessing hooks, images, or a controller interface.  I therefore implemented the heuristic as a learned representation bottleneck inside an action diffusion model:

- A causal encoder receives only `raw_history [B,2,47]` at deployment.
- The encoder predicts a continuous progress scalar, five phase logits, and five relative task-space waypoints.
- The waypoint chain is anchored to the current TCP, red pose and red goal in the world frame.  The five waypoints represent lift over the current red position, high carry midpoint, high pose above the red goal, release pose at the red goal, and high retreat pose near the demonstrated final TCP location.
- The predicted progress and waypoints condition a Conditional U-Net diffusion backbone that denoises normalized absolute Panda joint/gripper action chunks.

The waypoints are not executed directly.  They are trainable conditioning variables for the diffusion model.  All robot commands are still produced by the learned diffusion decoder as normalized joint targets and gripper commands.

## Architecture

`policy.py` defines `WaypointChainDiffusionPolicy`:

1. **Shared normalization.**  The model stores the supplied M1_v2 normalizer buffers for observations and actions.  It does not refit or use per-skill empirical scales.
2. **Causal features.**  The two raw observations are normalized and flattened.  Additional causal geometric features are computed from TCP, red pose, blue pose, goals, finger width and one-step deltas.  These features are in world metres and are scaled by broad fixed metre scales, not by the small expanded-slice range.
3. **Waypoint/progress encoder.**  An MLP predicts:
   - a progress scalar in `[0,1]`,
   - five phase logits for approach/open-to-red, grasp/initial lift, lifted carry, goal descent/closed-at-goal, and released/retreating,
   - residuals on a five-point relative waypoint chain.
4. **Diffusion decoder.**  The predicted progress, phase probabilities and waypoint vector are projected to a 256-dimensional global condition for `appl.public.DiffusionBackbone`.  The backbone predicts epsilon for `[B,16,8]` normalized action chunks.

## Losses and gradient paths

The total loss is `diffusion_loss + prior_loss`.

- `diffusion_loss` is the standard masked epsilon prediction loss from `appl.public.epsilon_loss`; it trains the U-Net and also backpropagates through the conditioning encoder because the denoiser depends on the predicted progress and waypoints.
- `phase_cross_entropy` trains phase logits against labels computed from the current observed red height, red-goal distance and finger aperture.
- `progress_mse` trains the scalar progress prediction to match the same causal phase ordering.
- `waypoint_mse` trains the predicted relative waypoint chain toward the geometric lift/carry/place/retreat chain generated from the current observation and goal.
- `x0_action_reconstruction` is a small auxiliary loss on the DDPM clean-action estimate, masked over valid action slots.  It is weighted by `alpha_bar` to avoid high-noise amplification.

All auxiliary terms include trainable predictions.  There is no constant-only penalty between observed poses, and no scripted action is substituted for diffusion output.

## Causal deployment inputs

At deployment `forward(noisy_action, timestep, raw_history)` receives only the last two observations.  It uses the following state fields:

- `qpos/qvel` for joint state and gripper aperture,
- `tcp_pose` for current end-effector position,
- `red_pose` and `blue_pose` for object locations,
- `red_goal` and `blue_goal` for target positions,
- drawer compatibility channels are ignored except as part of normalized observation input.

Future observations are used only as training labels supplied by the framework, never during inference.

## Handoff and termination cues

The policy can take over at the beginning of the assigned expanded slice when the gripper is open and the TCP is high after blue placement, or later when the red block has already been approached or lifted.  The observation cues that determine progress are red height, red-to-goal XY distance, TCP-to-red/TCP-to-goal relations and finger aperture.

A useful exit state is red at the red goal with the blue block still at the blue goal, gripper open, and the TCP retreated high near the demonstrated final height.  The completion contract only requires `red_at_goal AND blue_at_goal`; the learned retreat is retained because it appears in the assigned demonstrations and helps a successor or monitor receive a safe post-release state.

## Limitations

This prior is specialized to a free, top-down, state-based block transport setting.  It does not implement collision avoidance, analytical kinematics, image encoding, equivariance, or hard constraints.  The waypoint bottleneck may underfit detailed contact behavior if the training data require joint-space nuances that are not captured by the low-dimensional progress/waypoint representation.  Failures are expected if the red block is not reachable from the demonstrated approach family, if obstacles are introduced, or if a very different robot/object geometry is used.
