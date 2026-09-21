# PRIOR: blue_to_goal_handoff__h02

## Assigned heuristic

The assigned heuristic is the **goal-anchored blue placement prior** for the `blue_to_goal_handoff` skill.  Its hypothesis is that the important decision variable for this slice is not simply the middle of the demonstration, but the world-frame error between the blue block and `blue_goal`.  The demonstrations show blue being picked from the upper/red-goal region, transported through the center, lowered over the lower goal, released at table height around indices 650--700, and then the robot continuing into a transition toward reacquiring the buffered red block by index 860.

The expanded M1_v2 slice is preserved: each training segment runs from index 320 to 860.  This includes the shared predecessor transition into blue approach/grasp, the main blue transport and placement, and the shared successor transition after blue placement.

## Executable learned policy

The implemented model is still a learned epsilon-predicting DDPM action policy over 16-step sequences of the native Panda action representation: seven absolute joint targets plus one gripper command.  There is no scripted controller, replay table, IK module, collision module, or separate termination controller in `policy.py`.

The architecture has three learned parts:

1. **Causal goal-relative observation encoder.**  The encoder receives the two available causal observations.  It uses the shared observation normalizer from the full original demonstrations, then appends derived world-frame relative features for each history step:
   - `blue_pose.xyz - blue_goal`,
   - `tcp_pose.xyz - blue_goal`,
   - `tcp_pose.xyz - blue_pose.xyz`,
   - `red_pose.xyz - red_goal`,
   - `tcp_pose.xyz - red_pose.xyz`,
   - `blue_pose.xyz - red_pose.xyz`,
   - finger opening from the two finger joints,
   - TCP height relative to blue and to the goal plane,
   - contract-scaled blue containment errors and a current blue-at-goal indicator.

   These derived features use broad metric scales written in the code and metadata; they are not fitted to this skill's small expanded slice.

2. **Conditional diffusion backbone.**  The encoded causal condition is passed to the provided `appl.public.DiffusionBackbone`, which predicts DDPM epsilon for the normalized action sequence.

3. **Auxiliary trainable goal heads.**  During training only, an auxiliary trunk receives the learned condition together with the predicted clean action estimate reconstructed from the noisy action and predicted epsilon.  It predicts:
   - the final future blue-goal error over the training horizon,
   - the future minimum blue-goal distance over the current/future window,
   - a binary `blue_at_goal` logit.

Because these heads consume the predicted clean action estimate, their losses back-propagate through the diffusion prediction as well as through the shared condition encoder.  The prior loss is not a constant geometric penalty computed only from observations.

## Losses and gradient paths

`compute_loss` returns:

- `diffusion_loss`: standard masked epsilon MSE via `appl.public.epsilon_loss`.
- `prior_loss`: a weighted sum of auxiliary losses:
  - Smooth L1 loss between predicted final blue-goal error and the demonstrated final future blue-goal error in the masked horizon.
  - Smooth L1 loss between predicted future minimum distance and the label computed from current plus future blue poses.
  - BCE loss for whether current/future observations contain `blue_at_goal` under the task contract surrogate: center within 0.04 m in x and y and within 0.011 m in z.
  - A small additional terminal anchoring term that pulls predicted final error toward zero only for samples whose current/future window actually contains a blue-at-goal state.

The auxiliary loss is reliability-weighted by DDPM `alpha_bar` to reduce high-noise amplification when reconstructing the clean action estimate.

## Causal deployment inputs

At deployment, `model.forward` receives only the two-observation causal history.  It does not use future observations, future labels, or ground-truth success predicates.  The information that allows the policy to take over partway through the transition is present in the causal state: blue pose relative to blue goal, TCP pose relative to blue, gripper opening/closing state from qpos, red pose in the central buffer, and joint/qvel context.  These features distinguish early approach over the upper block, closed-finger transport, lowered release at the lower goal, post-release TCP lift, and the beginning of red reacquisition.

## Applicability and handoff behavior

The policy is intended for the expanded handoff slice where red has already been buffered and blue must be moved to the lower goal.  It can be invoked as early as the predecessor overlap because the observation shows blue still near the upper/red-goal region, open fingers, and the TCP approaching it.  It can also be invoked later during the grasp/transport transition because the blue-goal and TCP-blue relative features reveal whether blue is carried, elevated, near the goal, or already placed.

A useful exit for the successor is reached when blue is at `blue_goal` on the table and the gripper is no longer carrying it.  Demonstrations support transfer around index 700, where blue is at the lower goal, fingers are open, and the TCP has lifted high.  The expanded slice also trains continuation to about index 860, where the robot has moved toward and begun lifting the buffered red block, so the inference system may continue this policy briefly if it wants a later successor state.

## Dependencies and limitations

The implementation depends only on the provided state vector, shared normalizer, fixed DDPM machinery, and PyTorch.  It does not claim image invariance, physical contact detection, IK, or hard safety constraints.  The demonstrated data do not contain broad failed or marginal blue placements, so correction from severe overshoot is not established.  The gripper-release cue is learned from demonstrations and useful for handoff, but the task completion contract itself does not require release.
