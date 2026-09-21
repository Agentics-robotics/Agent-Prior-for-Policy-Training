# PRIOR: blue_approach_grasp_lift__h02

## Assigned heuristic

The assigned hypothesis is the **pad-to-blue transit waypoint prior** for the `blue_approach_grasp_lift` skill. The expanded training slice is index `[650, 870)` in each demonstration. This slice deliberately includes shared transition phases: the robot may take over while it is still leaving the red pad, and it continues through the beginning of the successor phase after the blue grasp has formed and the blue block is being lifted.

The intended decomposition is:

1. retreat from the red pad without disturbing the placed red block,
2. make a high-clearance global transit around the open drawer/table geometry,
3. descend locally over the blue block with the gripper open,
4. close on or near the blue block and lift it into a successor-ready state.

## Executable adaptation

The provided interface supplies only two causal state observations and joint-space action labels. It does not provide images, an inverse-kinematics solver, a collision checker, or a forward-kinematics provider. I therefore implement the waypoint prior as **learned conditioning structure and differentiable auxiliary prediction**, not as a scripted Cartesian controller.

The model remains an epsilon-predicting DDPM over 16-step sequences of the eight native actions: seven absolute Panda joint targets and one gripper command. The fixed sampler, noising process, optimizer, and action decoding are provided by the framework.

## Architecture

`policy.py` builds `WaypointConditionedDiffusion`:

- A causal two-step recurrent encoder (`GRU`) consumes the two state observations.
- Each observation token contains the shared-normalized 47-dimensional state plus broad-scale geometric features: TCP-to-blue, TCP-to-red, object-to-goal offsets, TCP-to-blue-goal offset, object/TCP heights, gripper opening, drawer position/velocity, and TCP XY distances. These features use fixed metre-scale divisors such as 0.30 m or 0.50 m; no per-skill normalizer is fitted.
- A parallel MLP encodes the flattened two-token history. The GRU and MLP features are fused.
- The fused state predicts a four-way soft stage latent:
  - `pad_retreat`,
  - `high_global_transit`,
  - `blue_descent`,
  - `grasp_lift`.
- The same fused state predicts a normalized short-horizon TCP waypoint and a z-clearance summary.
- The stage probabilities, a learned weighted stage embedding, the predicted waypoint, and the predicted clearance summary are concatenated with the fused state and mapped to the global condition vector of a standard `DiffusionBackbone` U-Net.

Because the predicted stage, waypoint, and clearance are part of the U-Net condition, their trainable heads influence the denoising network directly rather than being merely logged diagnostics.

## Losses and gradient paths

`compute_loss` returns:

- `diffusion_loss`: the standard masked epsilon MSE between the U-Net prediction and the sampled DDPM noise.
- `prior_loss`: a weighted sum of trainable auxiliary losses:
  - cross entropy on broad causal stage labels inferred from the current observed TCP/object/gripper state,
  - Smooth L1 regression to the masked terminal future TCP position over the 16-step training horizon,
  - Smooth L1 regression to the masked maximum and terminal future TCP z over the horizon.

The future observations are used only as training labels for the waypoint and clearance summaries. Deployment uses only the two causal observations. The auxiliary penalties depend on trainable predictions; they are not constants computed only from observed poses.

The total loss is `diffusion_loss + prior_loss`. The auxiliary weights are intentionally small enough that the action diffusion objective remains primary while the encoder is biased toward the assigned retreat/transit/descent/lift decomposition.

## Causal deployment information

At inference, the model receives only the last two raw states. The relevant causal cues are:

- `tcp_pose[18:25]` in world coordinates for current TCP position/orientation,
- `blue_pose[32:39]` for the current blue block location,
- `red_pose[25:32]` and `red_goal[41:44]` for whether the robot is still near the placed red block,
- `qpos[7:9]` for gripper opening/closure,
- `drawer_position[39]` and `drawer_velocity[40]` for the already-open drawer context,
- `blue_goal[44:47]` for the fixed final drawer-side reference.

These observations let the policy take over partway through the transition: if the TCP is still over the red pad with the gripper open, the learned stage latent can represent pad retreat; if the TCP is already high and moving toward the blue side, it can represent global transit; if the TCP is above the blue block, it can represent descent; and after finger closure or blue lift it can represent grasp/lift continuation.

## Applicability and handoff

This policy is intended for the full expanded slice `[650, 870)`: it should be selected after red placement or during the overlap while the arm is leaving the red pad. It should usually continue until the gripper is closed on or near the blue block and the blue block/TCP are rising. A successor blue-transport policy benefits from receiving a state with a stable closed grasp rather than only a geometric TCP-over-blue condition.

## Limitations

The implementation does not impose hard obstacle avoidance, hard z-clearance constraints, hard grasp verification, IK, or invariance. It is a state-conditioned learned joint-action diffusion model trained from the fixed demonstrations and therefore is tied to the demonstrated workspace layout and the supplied shared normalizer. The stage labels are broad supervision for representation learning, not hard runtime gates or rejection thresholds.
