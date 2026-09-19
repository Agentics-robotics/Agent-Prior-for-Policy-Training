# PRIOR: red_top_regrasp_lift__h02

## Assigned heuristic

The assigned hypothesis is an **open-drawer clearance prior** for the `red_top_regrasp_lift` skill. After the drawer has been opened, the robot should withdraw and reorient without cutting through the drawer frame, descend from above to top-grasp the red block, close the fingers, and lift the red block high enough for the transport policy. The full expanded training slice is retained: indices 240-530 for each demonstration, including the shared transition with drawer opening at 240-330 and the shared transition with red transport at 455-530.

## Executable adaptation

The provided interface supplies state observations and joint/gripper actions, but no image model, analytic collision checker, inverse kinematics, or forward kinematics. Therefore I implemented the heuristic as a learned action diffusion model with an auxiliary learned future-geometry predictor. The prior does not script actions. Instead, the denoising network predicts epsilon for normalized joint/gripper action sequences, and the auxiliary predictor maps the DDPM clean-action estimate plus the same causal condition to predicted future TCP/red/finger states. Clearance and lift losses are applied to those trainable predictions.

At deployment the model receives only the two most recent raw observations. Future observations are used only as supervised training labels for the auxiliary head.

## Architecture

`policy.py` defines `OpenDrawerClearancePolicy`:

- A condition encoder consumes the two-step raw history after the shared M1_v2 observation normalization. It also adds engineered causal state features: TCP-red offsets, TCP/red/blue offsets from the drawer center `x = 0.19 - drawer_position`, current and previous heights, finger widths, drawer motion, and goal-relative object positions. These features are scaled by fixed metric constants, not by per-skill fitted statistics.
- A standard `appl.public.DiffusionBackbone` Conditional U-Net predicts DDPM epsilon for action sequences of shape `[B, 16, 8]`.
- A temporal auxiliary head receives the denoised clean-action estimate `x0`, the learned condition vector, and horizon position features. It predicts normalized future `tcp_xyz`, `red_xyz`, and the two finger widths for each action slot.

The action representation remains the repository representation: seven absolute Panda joint targets plus one gripper command, trained as an epsilon-prediction diffusion policy.

## Loss terms and gradient paths

The total loss is:

`diffusion_loss + prior_loss`

where `diffusion_loss` is the standard masked epsilon MSE.

`prior_loss` is a weighted sum of learned auxiliary terms:

1. **Auxiliary state prediction loss**: MSE from the auxiliary head's predicted future TCP position, red position, and finger widths to `future_obs` labels, using the shared observation normalization.
2. **Demonstrated clearance/lift undercut loss**: one-sided penalties when predicted future TCP height is below the demonstrated high-clearance TCP height during high phases, or predicted red height is below the demonstrated lifted red height during lift phases. Gates are derived from the future labels, but the penalized quantities are the trainable auxiliary predictions.
3. **Open-drawer geometric clearance loss**: when the current observation says the drawer is open and the predicted TCP is in the open drawer neighborhood but not directly over the predicted red block, a hinge discourages predicted TCP height below a fixed safe reference of 0.115 m.
4. **Finger phase loss**: one-sided penalties encourage predicted fingers to remain open in demonstrated open phases and closed in demonstrated grasp/lift phases.

The auxiliary head depends on `x0 = (noisy_action - sqrt(1-alpha_bar) * predicted_epsilon) / sqrt(alpha_bar)`. Thus these losses back-propagate through the predicted epsilon into the diffusion backbone and condition encoder. High-noise amplification is limited by clamping the clean estimate and weighting auxiliary losses by `alpha_bar`.

## Handoff and causal observability

The observation vector contains the necessary causal information for taking over partway through the transition: `tcp_pose`, `red_pose`, `drawer_position`, `drawer_velocity`, `qpos` finger widths, and the goals in a world coordinate frame. In the early overlap, open fingers and drawer_position above 0.26 m indicate the drawer-open handoff; TCP height and TCP-red offset distinguish whether the policy should keep withdrawing/reorienting or start the top approach. In the late overlap, red height, TCP-red proximity, and closed fingers indicate that red is being carried and transport can take over.

The model is trained on the full expanded slice rather than a narrowed subphase, so it learns the safe withdrawal, top approach, close, lift, and initial carried transition under one state-conditioned diffusion policy.

## Applicability and termination cues

Applicable states are those with an open drawer and red still requiring top regrasp/lift near the drawer area. A practical termination cue for the inference API is red carried near the TCP with closed fingers and red center height around or above 0.30 m, while drawer_position remains above the drawer-open threshold.

Task success itself is not redefined by this prior; the supplied completion contract remains drawer open, red on the outside pad, and blue inside the drawer in the same observation.

## Limitations

This implementation is not an analytic collision avoidance method. It learns clearance from successful demonstrations only and may fail under large perturbations, changed drawer geometry, missed grasps, or object poses far outside the demonstrations. It uses no images, no force/contact labels, no IK, and no hard constraints during sampling. All normalization uses the shared normalizer supplied with M1_v2; no per-skill refit is performed.
