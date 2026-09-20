# PRIOR: blue_release_retreat_complete__h01

## Assigned heuristic

The assigned hypothesis is a **completion-predicate local insertion prior** for the final blue-block phase.  The drawer is already open and the red block is already on the outside pad.  The policy should take over during the shared transition from blue transport to final placement, locally center and lower the carried blue block into the open drawer, open/release, and retreat upward while preserving the completed task state.

The expanded training slice is the full assigned range, not only the last static frames: every demonstration begins at index 930 and ends at its individual final index around 1063-1077.  The first part of the slice, indices 930-1030, overlaps the preceding transport skill.  In that overlap the blue block is still high and grasped near the drawer mouth; later frames include lowering to the drawer z band, finger opening, and TCP retreat.

## Executable adaptation

The original heuristic mentions predicate conditioning and future completion supervision.  The available interface is a state-based action diffusion model with no external IK, no image encoder, and no custom inference-time sampler scoring hook.  I therefore implemented the prior as:

1. a learned causal observation encoder that augments the shared normalized 47-D state history with evaluator-relevant geometric features;
2. a standard conditional DDPM action backbone that predicts action noise for the 16-step normalized joint/gripper action horizon;
3. a differentiable auxiliary head used during training that predicts future task completion and future local geometry from the model's own denoised action estimate plus the causal condition.

This keeps deployment as a normal learned diffusion policy.  At inference, `forward` receives only the two-observation causal history, the current noisy action sample, and the diffusion timestep.  Future observations are used only as training labels in `compute_loss`.

## Observation representation

The policy uses the single shared M1_v2 normalizer supplied with the assignment for the full observation vector.  It does not refit per-skill means or narrow late-slice scales.

The encoder concatenates:

- the two normalized full observations, including joints, velocities, TCP pose, red pose, blue pose, drawer position/velocity, and goals;
- for each history observation, relative features in world/drawer/task frames:
  - blue position error to the observed blue goal;
  - blue position relative to the drawer center `x = 0.19 - drawer_position, y = 0, z = 0.035`;
  - TCP relative to blue and to the blue goal;
  - red position error to the red goal/pad;
  - drawer-open margin, drawer velocity, and normalized finger opening;
  - signed margins for blue-in-drawer, red-on-pad, and drawer-open predicates, plus the corresponding boolean predicate indicators;
- short temporal deltas for TCP, blue, red, and drawer position between the two history observations.

The added metre features use task-level scales such as 0.15-0.25 m and the drawer/pad dimensions.  These are not tiny empirical scales fitted to this skill.

These causal features let the model enter partway through the transition: high blue z with closed fingers indicates the shared centering/descent phase, low blue z near the drawer z interval indicates release, and open fingers with TCP above the object indicates the retreat/settle phase.

## Learned architecture

`policy.py` defines `BlueReleaseRetreatDiffusionPolicy`:

- `PredicateInsertionEncoder`: an MLP with LayerNorm and SiLU activations maps the 160-D causal feature vector to a 256-D condition.
- `DiffusionBackbone`: the provided conditional 1-D U-Net predicts epsilon for 8-D normalized action sequences.
- `ActionConditionedCompletionHead`: a training-only MLP reads the 256-D condition and the DDPM clean-action estimate `x0` for each horizon slot.  It predicts a future completion logit and an 8-D future geometry vector per slot.

The policy remains an action diffusion model: the only value returned by `forward` is the predicted diffusion noise for the joint/gripper action trajectory.

## Losses and gradient paths

The total training loss is:

```text
loss = diffusion_epsilon_loss + prior_loss
prior_loss = 0.05 * completion_BCE + 0.02 * geometry_MSE
```

- `diffusion_epsilon_loss` is the standard masked epsilon prediction loss supplied by `appl.public.epsilon_loss`.
- `completion_BCE` trains an auxiliary predictor of the future evaluator predicate `drawer_open AND red_on_pad AND blue_inside` at each future observation slot.
- `geometry_MSE` trains an auxiliary predictor of future blue-goal error, TCP-blue relation, drawer-open margin, and finger opening.

The auxiliary inputs include the model's denoised action prediction
`x0 = (noisy_action - sqrt(1-alpha_bar) * predicted_epsilon) / sqrt(alpha_bar)`, clamped to avoid extreme high-noise values.  The auxiliary losses therefore have gradients through the trainable completion/geometry heads, through `x0`, and into the diffusion noise prediction and shared condition encoder.  They are not constant penalties computed only from observed poses.

Future observations are labels only.  They are masked by the provided segment mask and do not cross segment boundaries.  Very noisy diffusion timesteps are downweighted for the auxiliary x0-dependent losses using the provided `alpha_bar`.

## Applicability and termination

The intended invocation region is the assigned expanded slice: drawer open, red already on/near the outside pad, and blue/TCP in the late drawer-approach region.  Supported starts include the observed handoff at index 930, where the blue block is grasped high near `x = -0.17 to -0.18`, `y = 0.065 to 0.080`, `z ≈ 0.32`, fingers are closed, and the drawer position is about 0.297 m.  The same learned policy also covers later causal states in which the blue block is already low in the drawer and the gripper is opening or retreating.

Task success is exactly the supplied completion contract: drawer open, red on the outside pad, and blue inside the drawer at a single observation.  The demonstrations additionally retreat upward with the gripper open; that is learned as part of the local skill and is useful for handoff/safety, but it is not an extra task-success condition.

## Limitations

The model is trained from only twelve demonstrations of this local phase.  It should not be treated as a broad recovery controller if the drawer is not open, the red block is not on the pad, or the blue block has fallen outside the local drawer-mouth support.  The implementation does not claim analytic inverse kinematics, collision checking, image invariance, hard geometric constraints, or an inference-time optimizer over completion score.  All such structure is expressed through learned conditioning and differentiable training losses inside the provided diffusion policy interface.
