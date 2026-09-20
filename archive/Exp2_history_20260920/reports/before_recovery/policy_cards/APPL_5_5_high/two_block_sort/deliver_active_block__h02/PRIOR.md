# Prior for `deliver_active_block__h02`

## Original heuristic identity

This policy implements heuristic 2 for `deliver_active_block`: a vertical staging and clearance prior. The intended behavior is to deliver the currently active block by using a high carry clearance, then lowering nearly vertically at the goal, opening/releasing only when the block is table-supported, and retreating upward/open after release. The assigned expanded slices include both the close/lift overlap at the entry boundary and the open retreat/traverse overlap at the exit boundary.

## Executable adaptation

The repository provides state observations, action sequences, DDPM noising/sampling, optimizer, normalization, and execution. It does not provide inverse kinematics, forward kinematics, collision checking, contact state, images, or a symbolic controller. I therefore implement the heuristic as a learned state-conditioned action diffusion model rather than as hard geometric control.

The model receives only the two causal observations at deployment. During training, future observations are used only as labels for auxiliary losses. The learned deployment representation must infer the appropriate stage from the current TCP pose, red/blue block poses, goals, gripper finger positions, joint state, and short history deltas.

## Architecture

`policy.py` defines `VerticalStagingDiffusionPolicy`.

- The diffusion output is epsilon for a horizon of normalized 8D native actions: seven Panda joint targets plus one gripper command.
- Observations use the shared M1_v2 normalizer supplied in the spec. No per-skill or per-slice normalizer is refit.
- A learned encoder consumes:
  - the two normalized 47D observations;
  - world-frame relative features: each block relative to its goal, TCP relative to each block and goal, block/TCP heights, XY distances, finger opening, and two-step deltas.
- A trainable stage classifier predicts one of five latent stages: `lift_to_clearance`, `high_translate`, `descend_at_goal`, `release_low`, and `retreat_open`.
- The soft stage distribution is multiplied by a learned stage embedding. The observation embedding, stage probabilities, and soft stage embedding condition a standard `appl.public.DiffusionBackbone` U-Net.
- A learned auxiliary future-feature head is used only in training. It predicts normalized future TCP/block/finger features from the causal condition and the model's denoised action estimate, providing an action/state coupling loss with gradients into the denoiser.

The stage module is not a controller. It does not choose actions directly and it is not used to replay demonstrations. It is a learned conditioning variable for the diffusion denoiser.

## Losses and gradient paths

The total loss is:

`L = L_diffusion + 0.10 L_stage_CE + 0.02 L_gripper_stage + 0.05 L_future_feature`.

- `L_diffusion` is the standard masked epsilon prediction loss.
- `L_stage_CE` trains the causal stage classifier. Labels are heuristic labels derived from observed training states: high object height with large XY error gives high transport, small XY error with elevated object gives descent, low object at goal with low TCP gives release, and open fingers with a placed block and raised TCP gives retreat.
- `L_gripper_stage` is applied to the denoised action estimate `x0`, not to observations. It biases predicted gripper commands toward closed during lift/transport/descent and toward open during release/retreat.
- `L_future_feature` predicts normalized future TCP, red block, blue block, and finger features from the denoised action sequence and the encoded causal history. Because the prediction head consumes `x0`, the loss has a differentiable path into the diffusion prediction. Its contribution is reduced at very noisy diffusion timesteps.

No auxiliary term is a constant penalty computed only from observed poses.

## Handling the full expanded slice and transitions

The assigned data includes red delivery from indices 120-440 and blue delivery from 500 to the end of each trajectory. The policy is trained on the full expanded slice. The entry overlap contains states where the gripper may still be closing and the block may be low; these are represented as `lift_to_clearance`. The exit overlap after red release contains open-finger upward retreat and lateral traverse toward the blue side; these are represented as `retreat_open` when the placed block is at its goal and TCP height is increasing or already clear. The blue segment begins with the TCP low near the blue block and fingers open/closing, again mapped into `lift_to_clearance` before high transport.

The causal observation information that lets the policy take over partway through a transition is explicit in the state: block positions relative to their goal pads, TCP position relative to both blocks, TCP height, gripper finger opening, and two-step motion deltas. For example, an observation with red already on its pad, fingers open, and TCP high above the red goal or moving toward the blue side is compatible with continuing retreat/traverse; an observation with TCP low near the blue block and blue still off goal is compatible with close/lift overlap for blue.

## Applicability and limitations

This prior is intended for the demonstrated open-table two-block sorting setup with table-height blocks, target pads at the supplied world coordinates, and the same state/action interface. It is not a general obstacle-avoidance or manipulation planner. It does not enforce hard clearance, hard release timing, or exact vertical motion; these are learned biases from demonstrations. It may fail if blocks are much taller, obstacles appear, contact is lost unexpectedly, a block is already partly off the workspace, or the policy is invoked far outside the measured handoff support.
