# PRIOR: deliver_active_block__h01

## Original heuristic identity

This policy implements heuristic index 1 for `deliver_active_block`: **goal-relative active-block delivery**.  The research hypothesis is that red delivery and blue delivery share the same structure when expressed relative to the selected block's own goal pad: grasped or near-grasped block, lift/carry, reduce object-goal error, lower, open/release, and retreat to a clear state.  The goal-relative representation is intended to reduce color/side memorization while still learning absolute Panda joint actions from demonstrations.

The assigned expanded slices are preserved in full:

- red delivery and transition: observations 120:440 for each demonstration;
- blue delivery and final retreat: observations 500:final for each demonstration.

The overlap portions are intentionally included.  In particular, the red slice after release also teaches the open, high retreat/traverse that prepares acquisition of the blue block.

## Executable adaptation of the heuristic

The deployment `forward` method receives only causal observation history, not a skill scheduler flag naming red or blue.  Therefore I adapted the heuristic into a learned candidate-selection representation:

1. Build a red candidate and a blue candidate from the same causal state history.
2. Encode each candidate with the **same** MLP using object-goal and TCP-object relative features.
3. Use a learned active-object gate to aggregate the red/blue candidate embeddings before conditioning the diffusion backbone.
4. Retain both candidate embeddings and normalized raw state in the condition so the model can still learn absolute joint-space requirements.

The gate is trained by an auxiliary geometric label.  Before red is placed, the red candidate is labeled active.  Once red is at its pad and blue is not, the expanded red-to-blue transition is represented as turning attention toward the blue candidate.  If both objects have the same completion status, TCP proximity to the objects disambiguates the label.  This is not a scripted controller: it only shapes the learned conditioning vector and the action sequence is still produced by DDPM denoising.

## Architecture

`policy.py` defines `GoalRelativeDeliveryPolicy`.

Inputs to `forward`:

- `noisy_action [B,16,8]`, normalized by the framework;
- DDPM timestep;
- causal `raw_history [B,2,47]` in the assignment schema.

Main components:

- Shared observation normalization using the assignment's single full-demonstration normalizer.  No per-skill scale fitting is performed.
- Candidate feature builder for red and blue.  Features include:
  - active object position minus its goal position;
  - TCP position minus active object position;
  - TCP position minus active goal position;
  - active object and TCP quaternions;
  - active object height, goal height, gripper width;
  - inactive object relation to the active candidate and inactive object's own goal error;
  - absolute goal position scaled by broad physical table scales.
- Shared candidate MLP for red and blue candidates.
- Learned two-way active gate.
- Raw normalized state encoder for qpos, qvel, TCP pose, both object poses, drawer compatibility channels and goals.
- `appl.public.DiffusionBackbone` Conditional U-Net with 256-dimensional global conditioning.

The output is predicted epsilon for the normalized action diffusion process.  The action decoder remains the fixed framework decoder for seven absolute Panda joint targets plus gripper command.

## Losses and gradient paths

The total loss is

`diffusion_loss + prior_loss`, where:

- `diffusion_loss` is the standard masked epsilon-prediction loss from `appl.public.epsilon_loss`.  It trains the candidate encoder, gate, condition MLP and diffusion backbone through the denoising prediction.
- `prior_loss = 0.05 * active_gate_cross_entropy + 0.05 * terminal_goal_state_mse`.

The active-gate loss trains the learned gate logits from causal observations.  The terminal-state auxiliary head predicts the future active object's goal-relative final xyz error and final gripper width from the learned condition vector.  Its labels come from `future_obs`, but the prediction is trainable and causal at inference; future observations are never input to `forward` during deployment.  The auxiliary loss therefore supplies gradients to the representation rather than adding a constant penalty computed only from observed poses.

No inverse kinematics, forward kinematics, image encoder, explicit waypoint generator, contact model or hard constraint solver is implemented.

## Causal deployment information

At deployment, the model can use only the last two observations:

- qpos/qvel for robot state and gripper opening;
- TCP pose in world frame;
- red and blue block poses in world frame;
- red and blue goal pad positions;
- zero drawer compatibility channels.

These fields let the model enter partway through a transition.  For example, if red is already on its goal and the gripper is open/high, the raw qpos/TCP state plus the gate's blue-candidate focus represent the observed traverse toward the blue side.  If blue is already placed and the TCP is high near the blue goal, the same representation supports the final open retreat/hold.

## Applicability and handoff cues

The policy is intended for the assigned two-block sorting task or very similar state-based tasks with the same block size, table height, Panda action interface and explicit red/blue goal pads.  It assumes that delivery has begun: the active object is near the TCP, grasped, being carried, being lowered/released, or the arm is in the demonstrated post-release transition.  It is not designed to recover arbitrary missed grasps.

Useful transfer/termination cues for the inference API are documented in `HANDOFF.json`.  In brief:

- keep this policy during carry, descent and release while object-goal error is being reduced;
- after red release, transfer when red is stable on the red pad, fingers are open, and TCP is clear or approaching the blue side;
- after blue release, task completion is available when both red and blue are on their respective pads.

## Limitations

The learned representation may still rely on fixed-goal regularities because the training demonstrations use fixed pads.  The auxiliary active-label rule can be ambiguous in out-of-distribution recovery states, especially if both blocks are displaced or neither is close to the TCP.  The model predicts joint targets from demonstrations and does not enforce geometric containment, collision avoidance or grasp force constraints beyond what is learned statistically.
