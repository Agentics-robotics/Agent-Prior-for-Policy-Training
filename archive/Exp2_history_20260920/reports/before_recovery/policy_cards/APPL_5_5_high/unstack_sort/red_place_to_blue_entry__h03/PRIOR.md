# PRIOR: red_place_to_blue_entry__h03

## Assigned heuristic

This policy implements heuristic 3 for `red_place_to_blue_entry`: a sequential object-attention prior for the ordered unstack-sort task.  The research hypothesis is that the long transition from releasing red to acquiring blue is easier to learn if the diffusion model has an explicit learned representation of which object is currently relevant.  The policy should attend to red while red is being carried to the red pad, then switch attention to blue once red is observed at the goal and the gripper has released, and continue through the demonstrated blue approach and lift-entry motion.

The assigned expanded slice is the full range `[130, 520)` for each training demonstration.  Therefore the implementation covers:

1. overlap with `red_unstack_lift` while red is grasped and lifted,
2. carrying red to the red goal at `[-0.24, -0.25, 0.02]`,
3. opening/releasing red and retreating upward,
4. traversing to the blue source region,
5. closing on blue and lifting blue to the successor-entry state.

## Executable adaptation

The original heuristic mentions target labels and attention logits.  The provided runtime only calls a diffusion denoiser with two causal observations, so I adapted the hypothesis into a trainable conditioning module inside an epsilon-prediction DDPM policy.  There is no scripted controller, no IK, no image encoder, and no hard action override.  The fixed sampler still produces normalized native Panda joint/gripper action sequences; this prior only changes the learned representation and adds auxiliary losses.

At deployment the model receives only `raw_history [B, 2, 47]`.  The observation fields used are qpos/qvel, tcp_pose, red_pose, blue_pose, red_goal and blue_goal in the world frame.  Drawer channels are ignored except through the normalized raw history because they are constant compatibility channels in this task.

## Architecture

`policy.py` defines `SequentialObjectAttentionPolicy`.

- Each causal history is normalized with the single shared experiment normalizer supplied in `spec`; no per-skill scale fitting is performed.
- Red and blue are encoded by a shared object MLP.  Inputs are broad-scale relative features: TCP-object, object-goal, TCP-goal, object and TCP deltas over the two available observations, object quaternion, smooth at-goal and lifted scores, finger opening/delta features, and an identity token.
- A history encoder embeds the two normalized 47-D observations.
- A learned attention MLP predicts two logits, `alpha_red` and `alpha_blue`, from the normalized history plus causal switch features such as red-goal error, finger opening, TCP-red distance, TCP-blue distance, red/blue lift scores and object deltas.
- The conditioning vector is the concatenation of the history embedding, red embedding, blue embedding, attention-weighted object embedding, attention probabilities and switch features, projected to a 256-D global condition.
- The final denoiser is the provided `appl.public.DiffusionBackbone`, a Conditional U-Net over 16 normalized 8-D actions.

This is still a learned action diffusion model: `forward(noisy_action, timestep, raw_history)` returns predicted DDPM noise for the action trajectory.

## Losses and gradient paths

The main loss is standard masked epsilon loss between predicted and sampled DDPM noise.

The prior loss is trainable and is not a constant geometric penalty:

1. **Soft attention cross-entropy** trains the attention logits.  The blue target is low while red is not yet placed/released, and high when the current or near-future training observations show red at its goal with an open gripper or blue lifting.  The labels may use future observations during training only; future observations are never passed to `forward`.
2. **Phase BCE** trains an auxiliary head to predict red-done-open, blue-target and blue-lift-soon indicators from the same causal representation.
3. **Consistency loss** encourages the learned attention probability on blue after the observed red-at-goal/open-release cue and on red before that cue.

All three terms backpropagate through learned logits/heads and the shared conditioning representation.  The included weights are 0.06, 0.03 and 0.02 respectively, so the auxiliary prior shapes representation without replacing the diffusion action objective.

## Handoff and causal takeover cues

The model can take over inside the shared transition because the causal observation includes the full state needed to infer phase: red pose relative to red_goal, blue pose relative to the TCP and blue_goal, gripper opening from qpos finger joints, TCP height, and two-step object/TCP deltas.  For example, at the start of the blue-overlap corridor the red block is already at the red pad, the gripper is open, and the TCP is retreating high above red; later the TCP is above or at blue and the gripper either remains open for approach or is closed while blue rises.  These are exactly the switch features and labels used by the learned attention prior.

A useful exit state for the successor has red fixed at the red pad, attention on blue, TCP close to or above blue, and blue lifted substantially above the table.  Demonstrated end states near index 520 show blue z around 0.278-0.280 m and fingers closed around 0.018 m per finger.

## Applicability and limitations

This policy is intended for the demonstrated ordered sorting problem: red must be placed before blue is acquired.  It may not generalize to reversed order, different object identities, recovery from a failed red placement, or starts outside the demonstrated red-carry/release/blue-entry corridor.  The attention switch can learn chronology correlated with red-at-goal, so it is not proof of autonomous symbolic planning.  The implementation does not enforce collision avoidance, containment predicates, stable grasp, or task completion checks beyond what the learned diffusion model infers from data.
