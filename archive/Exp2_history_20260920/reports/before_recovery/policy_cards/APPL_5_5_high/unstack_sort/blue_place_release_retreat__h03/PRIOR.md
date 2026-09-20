# PRIOR: blue_place_release_retreat__h03

## Assigned heuristic

The assigned heuristic is the post-release verification and retreat prior for the `blue_place_release_retreat` skill.  The demonstrations continue after the blue cube is placed: the gripper opens, the TCP retreats upward, and the red and blue cubes remain at their goal poses.  The intent is not to redefine the task success condition, but to bias the terminal diffusion policy toward actions that avoid disturbing already successful object poses.

The expanded M1_v2 slice for this skill is the full range from index 520 to each trajectory end.  Therefore the model covers the shared transition with the preceding blue transport skill: it starts while the blue cube is still grasped high above the table, lowers it to the blue pad, releases it, and then retreats.  The observation history contains enough causal information for this takeover: current and previous TCP pose, blue pose, red pose, goals, qpos/qvel, and finger opening show whether the cube is still being carried, whether it is being lowered, whether it is already at table height on the goal, and whether the gripper is open.

## Executable adaptation

The original heuristic mentions terminal success prediction and future object displacement.  In this API, deployment `forward` receives only causal observation history and a noisy action sample, while future observations are available only as training labels.  I therefore implemented the hypothesis as a learned epsilon-predicting action diffusion model with training-only auxiliary heads:

- a terminal success head trained from the future observation at the last valid horizon step;
- a disturbance-risk head trained from future red/blue displacement relative to the current observation;
- a release/retreat phase head trained from causal current-state labels;
- an action-space open-gripper prior that is active only when the current observation already shows the blue cube at the goal and the fingers open.

The disturbance and success heads are conditioned on the model's own denoised action estimate, so their losses have a differentiable path through the predicted diffusion noise as well as through the learned auxiliary heads.  This avoids the ineffective alternative of computing only a constant penalty from observed future poses.  Future observations are not used by `forward` at deployment.

## Architecture

`policy.py` defines `TerminalRetreatDiffusionPolicy`.

- Input: `raw_history [B, 2, 47]` in the shared world-coordinate state schema.
- Observation normalization: the model uses the single shared normalizer supplied in the experiment spec.  It does not fit per-skill scales.
- Causal features: the two normalized observations are concatenated with fixed-scale relative geometric features: blue-to-goal, red-to-goal, TCP-to-blue, TCP-to-red, TCP-to-blue-goal, two-frame TCP/object deltas, joint deltas, finger width/balance/velocity, height clearances, goal-distance scalars, and current TCP/object quaternions.  The relative scales are fixed physical constants, not empirical per-slice statistics.
- Encoder: an MLP with LayerNorm and GELU maps the feature vector to a 256-dimensional condition.
- Diffusion backbone: the provided `appl.public.DiffusionBackbone` Conditional U-Net predicts epsilon for normalized 16-step action sequences.
- Auxiliary modules: a phase classifier uses the causal condition; a terminal head uses the causal condition plus a masked summary of the predicted clean action sequence.

The policy predicts the standard normalized action representation: seven absolute Panda joint targets and one gripper command.  There is no scripted controller, inverse kinematics, image encoder, or hard geometric constraint.

## Loss terms and gradient paths

The total loss is

`diffusion_loss + prior_loss`.

`diffusion_loss` is the standard masked epsilon MSE supplied by `appl.public.epsilon_loss`.

`prior_loss` is a weighted sum of:

1. `success_loss`: binary cross entropy between a learned action-conditioned success logit and whether the last valid future observation satisfies conservative red/blue goal predicates.
2. `disturbance_loss`: binary cross entropy between a learned action-conditioned disturbance logit and the normalized future red/blue displacement magnitude.
3. `phase_loss`: cross entropy for a four-phase causal label: high transport, descent, table/release, and high open-finger retreat.
4. `open_loss`: when the current observation already shows blue at the goal, table height, and open fingers, the predicted clean action sequence is softly encouraged to keep the gripper command open.

The clean action estimate is reconstructed from the noisy sample and predicted epsilon using the DDPM formula.  Auxiliary losses that depend on this estimate are weighted by `alpha_bar` to avoid unstable high-noise amplification while preserving gradients to the epsilon predictor.

## Causal deployment behavior

At deployment, only the two most recent observations condition the policy.  The learned condition tells the diffusion model whether it is in the shared transport/descent transition or in the terminal release/retreat phase:

- if the blue pose is high and near the TCP with closed fingers, the model learns continued transport toward the blue pad;
- if blue is above the blue goal but descending, it learns the lowering motion;
- if blue is at table height near the goal and the fingers are opening/open, it learns open-finger retreat;
- if both objects are already at their goals and the TCP is high, it learns the terminal hold/retreat convention.

The auxiliary heads are training mechanisms; the API may use the documentation and task predicates for policy selection and termination.  The model itself executes by sampling actions from the learned diffusion policy.

## Applicability and limitations

This prior is intended for the terminal part of `unstack_sort` after the red cube has been placed and the blue cube is being placed.  It can take over during the assigned overlap with blue transport because the observation contains the current blue pose, blue goal, TCP pose, and finger state.

Limitations:

- task success in the completion contract is only `red_at_goal AND blue_at_goal`; open gripper, clearance, velocity, and sustained hold are demonstration conventions rather than required predicates;
- the learned disturbance risk is supervised by the available short future horizon, not by a physics proof of stability;
- the policy has no recovery logic for a dropped cube far from the goal beyond what can be inferred from the demonstration slice;
- no external IK or collision checker is implemented.
