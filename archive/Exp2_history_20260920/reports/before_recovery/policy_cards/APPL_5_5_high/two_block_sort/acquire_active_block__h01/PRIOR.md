# PRIOR: acquire_active_block__h01

## Assigned heuristic

The assigned hypothesis is **active-object relative servoing** for the `acquire_active_block` skill in `two_block_sort`. During acquisition, the useful state variables are not primarily the block color or an absolute memorized arm path, but the relationship between the TCP and the block that is currently active. The demonstrations show the same structure for the first red acquisition and the later blue acquisition: move the open gripper above the selected block, descend, close around the block, and lift into an early carry state while maintaining a small TCP-to-object offset.

The assigned expanded slices are kept intact:

- red acquisition: observations/actions 0 through 219 in each demonstration;
- blue acquisition: observations/actions 340 through 599 in each demonstration;
- the shared transition/overlap portions with delivery are included, especially close, lift, and early carry.

## Executable adaptation

The implementation interface passes only causal observation history to `forward`; it does not pass a segment label or an explicit active-object string. To make the assigned scheduler-dependent heuristic executable inside this interface, I use a causal task-order selector:

- if the latest observed red block is already close to `red_goal` in world position, the active block is blue;
- otherwise the active block is red.

This matches the demonstrated order. At initial states both blocks may be away from their goals, so the selector chooses red. After the red delivery has placed red on the red pad, the selector chooses blue. This is an implementation adaptation of the handoff interface, not a general symbolic planner.

## Model architecture

`policy.py` defines `ActiveObjectRelativeDiffusion`, a learned epsilon-prediction DDPM denoiser over normalized absolute Panda joint/gripper action sequences. It uses the repository `appl.public.DiffusionBackbone` U-Net and therefore remains a learned diffusion action policy.

The learned conditioning encoder receives two causal observations and constructs a shared active-object representation per observation:

- normalized arm/finger `qpos` and `qvel` using the assignment's complete-demonstration normalizer;
- TCP world position and quaternion;
- `tcp_xyz - active_block_xyz`;
- `tcp_xyz - active_goal_xyz`;
- `active_block_xyz - active_goal_xyz`;
- `inactive_block_xyz - active_block_xyz`;
- selected active and inactive block absolute positions normalized with the shared complete-demonstration normalizer;
- selected active and inactive quaternions;
- active block height relative to its goal/table height;
- observed gripper opening;
- active goal and inactive-goal context.

Relative Cartesian features are scaled by shared position half-ranges from the complete trajectory normalizer. The model does **not** refit small per-skill scales.

The feature vector from the two observations is encoded by trainable MLP layers to a 192-dimensional condition vector. The conditional U-Net then predicts diffusion noise for the `[16, 8]` action horizon.

## Losses and gradient paths

The main loss is the standard differentiable diffusion epsilon MSE against the sampled DDPM noise, masked to remain inside each assigned skill slice.

An auxiliary prior loss is also used. From the predicted epsilon, the code forms the standard DDPM estimate of clean normalized action sequence `x0`. A trainable geometry head receives this predicted action sequence plus the learned condition vector and predicts future active-object geometry:

1. future `tcp_xyz - active_block_xyz`,
2. future gripper opening,
3. future active-block lift height,
4. future `active_block_xyz - active_goal_xyz`.

Targets come from `future_obs` within the same slice and are used only during training. The loss is masked and downweighted at high diffusion noise by `alpha_bar`; its weight in the total loss is `0.02`. This auxiliary loss has gradients through the predicted epsilon and the geometry head, so it is not a constant penalty computed only from observed poses.

## Causal deployment inputs

At deployment, `forward` uses only `raw_history [B, 2, 47]`, the noisy action sample, and the diffusion timestep. It does not use future observations, demonstrations, file access, IK, or scripted action generation. Future observations appear only in `compute_loss` as training labels for the auxiliary geometry head.

## Handoff and transition behavior

The model is trained over the full expanded acquisition slices, including the shared transition phases. It should be able to take over partway through acquisition because the condition vector includes causal cues for phase:

- open versus closing/closed fingers from `qpos[7:9]`;
- TCP-to-active-block lateral and vertical offset;
- active object table height versus lifted height;
- whether red is already at its goal, which selects blue acquisition;
- qpos/qvel and TCP pose indicating whether the arm is approaching, descending, closing, or lifting.

The delivery successor should receive a state where the selected block is grasped or beginning to be carried: fingers near the demonstrated closed width and active object rising above table height or already around the observed carry height. If the block is still on the table, the fingers remain open, or the TCP is not centered on the active block, acquisition should continue rather than handing off.

## Dependencies and limitations

This implementation depends on the fixed state schema and normalizer supplied by the assignment. It assumes the same red-then-blue task order. It has no image encoder, no analytic contact detector, no explicit collision constraints, no inverse kinematics, and no recovery policy for missed grasps or fallen objects. The learned model may still need data support to convert object-relative conditioning into correct absolute joint targets because actions are absolute joint/gripper commands.
