# Prior for `red_transport_place_pad__h02`

## Assigned research heuristic

The assigned hypothesis is a **high-carry then soft-place temporal prior**. The intended behavior is to separate the red-block transport into stages: keep the red block high while moving across the workspace, descend only near the outside pad, open the fingers after the red block reaches the pad, and then retreat so the successor blue-block policy receives an uncluttered handoff state.

The assigned expanded slice is kept unchanged: demonstrations `demo1000` through `demo1011`, indices 455 to 760. This includes the overlap from the preceding red lift/regrasp phase (455-530) and the overlap into the blue approach phase (650-760). The model is therefore not just a pad-placement endpoint model; it is trained to take over during late lift, perform the transport/descent/release, and continue through the retreat transition.

## Executable adaptation

The repository supplies joint-space actions and state observations, but no analytic inverse kinematics, no contact estimator, no force signal, and no image input. I therefore implemented the heuristic as a learned state-conditioned action diffusion model rather than as a geometric controller. The high-carry/soft-place idea is represented by a trainable stage bottleneck and auxiliary losses that supervise predictions made by trainable heads.

The four learned stage classes are:

1. `lift_and_high_carry`: red grasped or being lifted, fingers closed, red not yet placed on the pad.
2. `pad_descent`: red moving near the pad and descending.
3. `soft_release`: red low on the pad while fingers are still effectively closed or the TCP remains close.
4. `retreat_to_blue_transition`: red on the pad, fingers open and/or TCP lifted away toward the successor blue approach.

Stage labels are computed from causal current observations during training using red pose, pad goal, TCP-red relation, and finger width. These labels are not used as an external controller at inference. At deployment the network predicts stage logits from the two-step causal observation history, and the soft stage distribution conditions the diffusion backbone.

## Architecture

`policy.py` defines `HighCarrySoftPlacePolicy`, a learned DDPM epsilon predictor.

Inputs at forward time are exactly the interface inputs:

- noisy normalized action sequence `[B, 16, 8]`,
- diffusion timestep,
- raw causal observation history `[B, 2, 47]`.

The model uses the assignment's shared normalizer buffers. It does not fit or assume any per-skill normalization. The conditioning encoder combines:

- the two shared-normalized observations flattened together,
- fixed-scale engineered state features in metres/radians, including red-to-pad, TCP-to-red, TCP-to-pad, TCP-to-blue, blue-to-goal, two-step red/TCP deltas, finger width and finger delta, drawer opening state, distances, heights, and qvel summaries.

These engineered features are scaled by broad physical constants such as 0.30 m, 0.40 m, 0.60 m, and 0.08 m rather than by narrow skill-specific empirical ranges.

The observation and engineered-feature encoders feed a learned trunk. A stage head predicts four stage logits. The softmax stage probabilities are multiplied by a learned stage embedding table and concatenated back into the global condition. A `appl.public.DiffusionBackbone` U-Net then predicts DDPM epsilon for the normalized action trajectory.

## Losses and gradient paths

The main loss is the standard masked epsilon prediction loss supplied by `appl.public.epsilon_loss`.

The prior loss is the weighted sum of three trainable auxiliary terms:

1. **Stage cross entropy** (`0.15`): the stage head predicts the current high-carry/descend/release/retreat phase from causal observations. Gradients update the observation encoder, stage bottleneck, and condition representation used by the diffusion backbone.
2. **Future feature prediction** (`0.10`): from the global condition and the model's denoised clean-action estimate, a learned head predicts short-horizon future deltas for red xyz, TCP xyz, and finger width. Targets come from `future_obs` during training only. The head input includes the clean-action estimate derived from the predicted epsilon, so this auxiliary term also provides gradients to the diffusion prediction, with alpha-bar weighting to reduce high-noise amplification.
3. **Stage-conditioned gripper hinge** (`0.02`): the denoised action estimate is mildly encouraged to keep the gripper closed during lift/carry/descent and open during placed/retreat states. This is a differentiable penalty on the model's predicted clean actions, not a scripted gripper command.

All auxiliary losses depend on trainable predictions. No loss term is a constant computed only from observed poses.

## Causal deployment behavior

At deployment the model receives only the current two observations. It can take over partway through the transition because the observation contains the necessary cues:

- red world pose and red goal identify whether the red block is still being carried, near the pad, placed low, or already stable on the pad;
- TCP pose relative to red pose identifies whether the gripper is still engaged with the red block or has retreated;
- finger qpos identifies closed versus open state;
- drawer position verifies that the drawer remains open during this skill;
- blue pose and TCP-to-blue relation give context for the retreat into the successor blue approach overlap.

The policy still outputs learned joint-space diffusion actions. It does not perform IK, collision checking, explicit grasp detection, or rule-based replay.

## Applicability and termination cues

Appropriate entry states are those with the drawer open and the red block carried or being lifted by a mostly closed gripper. The expanded slice supports starting around index 455, when red is close to the TCP and low but in the lift transition, as well as around index 530, when red is carried high near z = 0.31 m.

Useful transfer/termination cues are red on the pad at z about 0.02 m, fingers open near total width 0.08 m, and TCP moving upward/laterally away from the red block toward the blue block. These cues guide policy selection only; the task-level completion rule remains the supplied three-goal expression.

## Limitations

The training data show successful placements but not failed drops, collisions, or large recovery maneuvers. The learned future-feature head is not a dynamics guarantee. The stage labels are heuristic supervision derived from state geometry, so ambiguous states outside the demonstrations may be misclassified. There is no implemented image encoder, analytic IK, force control, or hard safety constraint beyond what the framework provides.
