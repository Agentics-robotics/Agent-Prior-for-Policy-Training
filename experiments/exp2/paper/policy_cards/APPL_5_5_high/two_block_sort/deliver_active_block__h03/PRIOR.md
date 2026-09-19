# PRIOR.md — deliver_active_block__h03

## Assigned heuristic

This policy implements heuristic 3 for `deliver_active_block`: the predicate-and-readiness prior. The research hypothesis is that delivery should not only put the active block inside its goal pad predicate, but should also learn the demonstrated manipulation interface: release the block, open the gripper, and retreat the TCP to a clear state. This matters because the task contract only requires `red_at_goal AND blue_at_goal`, while the demonstrations continue after placement to create stable handoff states. In the expanded M1_v2 slice, red delivery includes the shared transition window through the retreat toward blue acquisition, and blue delivery includes release and final retreat.

## Data slice and causal takeover information

The policy is trained on the full assigned expanded segments:

- red delivery and post-release retreat: indices `[120, 440)` in each demonstration;
- blue delivery and final release/retreat: indices `[500, episode_end)` in each demonstration.

The model can take over partway through these transitions because the causal state history contains the relevant world-frame variables: `tcp_pose`, `red_pose`, `blue_pose`, `qpos` finger positions, `qvel`, and the fixed red/blue goals. From these it can infer whether red or blue is already near its target, whether the gripper is still closed around a block, whether the block is lifted or lowered, and whether the TCP is in the low release phase or the retreat phase. No future observations are used during deployment.

## Implemented learned policy

The implementation remains a learned action diffusion policy. `forward(noisy_action, timestep, raw_history)` predicts DDPM epsilon for a 16-step sequence of normalized native actions. It uses the provided fixed diffusion sampler and the public `DiffusionBackbone` U-Net.

The conditioning network has three learned parts:

1. **Causal observation encoder.** The two raw observations are normalized with the single shared M1_v2 normalizer from the assignment. I add deterministic world-frame relation features such as object-to-goal, TCP-to-object, heights, finger opening, and two-step motion. These features use task/workspace-scale constants, not a per-skill refit.
2. **Predicate/readiness forecast head.** A learned auxiliary head predicts 12 logits: current, mid-horizon, and end-horizon values for `red_at_goal`, `blue_at_goal`, `red_successor_ready`, and `blue_terminal_ready`.
3. **Diffusion condition head.** The U-Net global condition is formed from the encoded observation and the model's own predicted predicate/readiness probabilities. These probabilities are not labels at inference time; they are trainable causal predictions.

There is no scripted controller, no inverse kinematics module, no image encoder, and no hard constraint projection. The output is still a denoised joint/gripper action sequence in the repository's native action space.

## Predicate and readiness labels used for training

Auxiliary labels are computed from observations using the task contract and the measured handoff semantics:

- object at goal: object center within 0.04 m in both x and y from the target center, and z within 0.011 m of the target z. The 0.04 m xy tolerance is the pad half-size 0.06 m minus the block half-size 0.02 m, matching full xy containment.
- open gripper: both finger qpos values exceed 0.035 m;
- clear TCP: `tcp_pose.z > 0.15 m`;
- red successor readiness: `red_at_goal AND open_gripper AND clear_TCP`;
- blue terminal readiness: `blue_at_goal AND open_gripper AND clear_TCP`.

The labels are training targets only. At deployment, the policy receives only the causal observation history and uses its own forecasts.

## Losses and gradient paths

`compute_loss` returns:

- `diffusion_loss`: standard masked epsilon prediction loss from the public helper;
- `prior_loss`: a weighted sum of two differentiable auxiliary terms.

The prior terms are:

1. **Auxiliary BCE forecast loss.** The auxiliary logits are supervised against current, mid-horizon, and end-horizon predicate/readiness labels. This loss updates the observation encoder and auxiliary head directly. Because the U-Net condition includes the predicted probabilities, diffusion gradients also pass through the readiness embedding.
2. **Terminal-phase clean-action consistency.** The predicted epsilon is converted to a DDPM clean-action estimate `x0`. In time steps where the demonstrated future observation is at a goal or ready state, an extra masked MSE encourages the denoiser to reconstruct the release/retreat action sequence. The term is down-weighted by `alpha_bar` so high-noise clean estimates do not dominate.

The readiness prior is therefore trainable: it is not a constant penalty computed only from observed poses.

## Handoff behavior represented by the model

For red, the model learns the continuum from carrying/lowering the red block, opening at the red goal, lifting the TCP, and retreating toward the blue side. Demonstrations show red is on the target and the fingers are open by approximately indices 300-340, while the slice continues to 440 with the TCP clear and positioned for the next acquisition. This expanded overlap is deliberately included so delivery can produce a successor-ready state rather than stopping while low or closed.

For blue, the model learns the analogous lowering/release/retreat sequence and final stable state. The task may already satisfy the formal goal predicate once both blocks are on pads, but the demonstrated interface has open fingers and TCP clearance near the end.

## Applicability and limitations

This prior is applicable to state-based two-block sorting demonstrations with the same observation schema and shared normalizer. It should be selected when delivery, release, or retreat is needed. It is not designed to recover from a missing grasp, a block knocked far away after release, or an unseen obstruction. The model has only successful demonstrations plus in-trajectory negative labels, so the predicate/readiness forecasts should be interpreted as learned cues for handoff, not certified safety checks.
