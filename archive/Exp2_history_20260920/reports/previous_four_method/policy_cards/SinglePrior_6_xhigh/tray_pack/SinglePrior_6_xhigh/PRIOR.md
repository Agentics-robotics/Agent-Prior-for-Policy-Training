# Shared Temporal Transport Relations

## Scope and scientific choice

This package is one full-task epsilon-predicting Diffusion Policy for `tray_pack`, policy `SinglePrior_6_xhigh`, identity `full_task`, heuristic index 1. It learns from all twelve complete assigned demonstration ranges. There are no phase-specific policies, trajectory cuts, action tables, fixed waypoints, external schedules, or controllers. The same learned forward function runs from reset through completion.

The selected prior is **shared temporal tool-object-goal representation learning**: both blocks should be described using the same geometric relations and the same feature encoder, while their identity, absolute robot configuration, and scene context remain available. A small auxiliary forecast of changes in those relations encourages the encoder to represent transport and contact transitions rather than only memorize absolute poses. This is a representation prior, not a hard physical constraint or an exact symmetry of robot actions.

The key distinctions are visible in state: an object can be far from the tool, move with the tool while far from its goal, or remain at its goal as the tool moves away. Relative position, relative motion, goal residual, and measured finger aperture expose these distinctions without assigning a phase label. No hand-authored grasp predicate, active-object rule, or red-first switch is implemented.

## Training evidence examined

I read the interface, numerical helpers and U-Net source, the full-task assignment, and whole-trajectory overviews for all twelve demonstrations. I inspected 24 evenly spaced states in `demo10300`, eight in `demo10301`, and six in each of `demo10302` through `demo10311`. I additionally read original transition samples in `demo10300` at 108, 112, 120, 276, 280, 286, 344, 352, 470, 478, 490, 650, 656, 662 and in `demo10310` at 96, 100, 104, 112, 268, 274, 334, 342, 462, 466, 636, 646. These are evidence samples, not training subset definitions.

### Detailed observations

- `demo10300:0`: both blocks rest at z = 0.020 m, the TCP is at z about 0.442 m, and each finger is at 0.040 m. The red and blue blocks are near (-0.4053, -0.2507) and (-0.3942, 0.2696), respectively. Goals are separate regions at (-0.14, -0.075, 0.036) and (-0.14, 0.075, 0.036).
- `demo10300:96` has the open tool at red-block height. At 108 and 112 the command is -1, each finger is about 0.0183 m, and the red object and TCP remain near z = 0.020 m. At 120 lifting starts. By 161, red and TCP heights are approximately 0.27568 and 0.27559 m, with an object-minus-TCP offset of about (0.00951, 0.00018, 0.00009) m. The shared near-contact features should make this very different from merely passing over a stationary block.
- At `demo10300:193` and 225, red is transported toward its goal while blue stays on the table. At 225 red is close in XY to its goal but is still at z about 0.310 m: XY goal proximity alone is not enough. At 276, 280, and 286 red is lowered to roughly 0.0490, 0.0457, and 0.0455 m. The gripper command is still -1 at 280 and is +1 at 286. At 290 red has settled at z about 0.0360 m.
- `demo10300:322,344,352,387,419` expose the full red-release/retreat/blue-approach transition. Red stays around (-0.1361, -0.0707, 0.0360), while the open TCP retreats to z about 0.316 m and then travels toward blue. These observations motivate conditioning on both objects and both goals, not cropping training into independent pick episodes.
- At `demo10300:470` the tool is open at blue height. At 478 the command is -1 and the fingers have closed. At 490 the lift begins, and at 516 blue and TCP heights are approximately 0.17708 and 0.17686 m, with an x offset around 0.00822 m. The blue transport repeats the red tool-object coupling with different absolute joints and a different goal. At 650 and 656 blue is near its goal but remains held around z = 0.0470 and 0.0455 m; at 662 it is at z about 0.0360 m and opening is underway.
- `demo10310:96,100,104,112` show a second grasp transition at a slightly different source pose: the tool is open near z = 0.020 m at 96 and 100, closed at 104 and 112. At 268 and 274, red is lowered toward the tray. At 334 and 342 red is already placed while the open tool finishes its retreat. At 462 and 466 the tool is again open at blue height. At 636 and 646 blue is being lowered. Repeated almost-stationary samples also reveal a limitation: a two-observation policy cannot reproduce an arbitrary hidden dwell timer exactly.

### Cross-demonstration coverage

The following source indices summarize the additional whole-trajectory evidence. In every listed middle sample, red has reached its region while blue is still at the table; in every final sample both blocks have settled near their own goals. The earlier and later overview samples show red and blue transport, respectively.

| Trajectory | Red transport sample | Red placed, blue pending | Blue transport sample | Final observation |
|---|---:|---:|---:|---:|
| demo10301 | 212 | 318 | 530 | 742 |
| demo10302 | 150 | 300 | 601 | 752 |
| demo10303 | 148 | 297 | 594 | 743 |
| demo10304 | 147 | 294 | 589 | 737 |
| demo10305 | 150 | 300 | 601 | 752 |
| demo10306 | 149 | 299 | 599 | 749 |
| demo10307 | 146 | 292 | 584 | 730 |
| demo10308 | 147 | 294 | 589 | 737 |
| demo10309 | 148 | 296 | 592 | 741 |
| demo10310 | 146 | 292 | 584 | 730 |
| demo10311 | 148 | 296 | 593 | 742 |

The inspected reset positions vary only narrowly: approximately x = -0.412 to -0.390 m and y = -0.270 to -0.251 m for red, and x = -0.412 to -0.392 m and y = 0.249 to 0.270 m for blue. Reset robot configuration and goals are unchanged in these samples. This supports sharing transport geometry but does not support claims of broad layout generalization. All observed demonstrations are red-first; the network can learn that preference through ordered context, but no order is hard-coded into action generation.

## Implemented architecture

### Causal inputs and normalization

`forward(noisy_action, timestep, raw_history)` accepts only the supplied [B,16,8] noisy normalized action, scalar or batched diffusion timestep, and [B,2,47] causal raw observations. There is no mutable cross-call state. Repeated DDPM forward calls do not advance a task clock.

All 94 raw history components are normalized with the unchanged shared mean and half-range buffers. They are retained directly in the final conditioning vector. This includes joint positions, velocities, both finger positions, all three poses, both goals, and the fixed-zero compatibility channels. There is no observation clipping or refitting.

Derived geometry is computed from raw physical units, using broad, fixed numerical scales rather than narrow empirical standard deviations. A tanh is used only to add local-resolution features; it does not replace or clip the broad-scale relative vectors or original observations.

### One shared block encoder

For each of the two observed times and each object i, the 38-component feature vector contains:

1. Object world position, its own goal world position, object minus TCP, goal minus object, other object minus object, and object minus TCP expressed in the TCP frame: six 3-vectors, scaled by 0.25 m.
2. Componentwise tanh of object-minus-TCP and goal-minus-object divided by 0.04 m: six additional local-resolution components. The scale is the known block side length, not an activation threshold or grasp command.
3. First two columns of the object's rotation matrix and of its rotation relative to the TCP: twelve components. Unit quaternion normalization and matrix conversion make these derived orientation features invariant to quaternion sign.
4. Both measured finger positions divided by 0.04 m: two components.

For each block, concatenate the earlier and later 38-component feature vectors, plus object velocity, TCP velocity, relative velocity, and two finger velocities from the observation difference. The difference interval is the task's 0.05 s control interval, not elapsed episode time. Linear velocities are divided by 0.5 m/s and finger velocities by 0.2 m/s. This gives 87 inputs.

A shared MLP, 87 -> 192 -> 192 -> 128 with SiLU hidden activations and final LayerNorm, is applied to both blocks. The two resulting 128-component embeddings are kept in named red/blue order, not pooled into an identity-free vector. There are no separate object-specific weights or separate action heads.

A scene MLP, 350 -> 256 -> 128 with SiLU and final LayerNorm, reads the 94 normalized raw components and both block embeddings. The U-Net condition concatenates raw history (94), both block embeddings (256), and scene embedding (128), totaling 478 components. This retains absolute robot information because outputs are seven absolute joint targets plus a gripper command: relational conditioning alone cannot yield joint-space translation or rotation equivariance.

### One diffusion model

The supplied `DiffusionBackbone` is instantiated exactly once, with condition dimension 478. It retains widths 128/256/512, kernel 5, groups 8, timestep embedding 128, and the supplied FiLM-conditioned temporal residual U-Net. Its only action output is the predicted epsilon tensor [B,16,8]. Sampling, action normalization/decoding, DDPM100, sample clipping, and execution prefix 8 are unchanged. There is no action override or Cartesian-to-joint controller.

## Loss and actual gradient paths

The primary loss is the supplied masked epsilon MSE. No phase weights, demonstration resampling, action relabeling, extra action penalties, or denoising schedule changes are introduced.

The auxiliary head reads each block's 128-dimensional embedding and the common 128-dimensional scene embedding. Its shared MLP is 256 -> 192 -> 96 with SiLU, where 96 = 16 slots times 6 relations. The last weights are initialized with standard deviation 0.01 and zero bias. The prediction is a scaled change in the following six-dimensional world-frame relation:

    r_i(s) = concatenate(p_i - p_TCP, g_i - p_i)
    target[j,i] = (r_i(future_obs[j]) - r_i(current_obs)) / 0.10 m

This predicts near-term changes rather than requiring the head to copy static absolute positions. A block that remains stationary as the TCP moves changes its tool relation but not its goal relation; a held, moving block changes its goal relation while approximately maintaining its tool relation. Both patterns appear in training.

The forecast uses the supplied future observation at exactly the same slot j, which is the state after action j. Action slots are t-1 through t+14, so the first future slot normally coincides with the latest causal state. There is no one-step shift or assumption that slot zero is t+1. The future mask, expanded over both objects and six relation components, excludes padded labels. The loss is componentwise smooth L1 with beta 0.1, averaged over valid slots and twelve total components. The beta corresponds to 1 cm after the 0.10 m target scale.

    loss = diffusion_loss + 0.05 * prior_loss

`prior_loss` is the unweighted forecast loss returned by `compute_loss`. Targets are constructed without gradients from training labels only.

The precise learning paths are:

- Epsilon loss -> U-Net action/noise parameters and FiLM condition projections -> scene encoder and shared object encoder. Thus the architectural geometry prior participates in learning the action distribution directly.
- Forecast loss -> shared forecast head -> scene encoder and shared object encoder. The forecast loss does NOT backpropagate through predicted epsilon into the U-Net; its contribution to action generation is through the shared causal representation. This limited claim is intentional. The head is not a verifier of a sampled action's physical consequences.
- Future states are labels only and never enter `encode_history` or deployment `forward`. The deployment forward does not evaluate the forecast head. The entire package remains one model/checkpoint; the auxiliary head is a training branch within that model, not another policy.

There is no reconstructed-x0 auxiliary loss and no division by sqrt(alpha_bar). The relation target and forecast inputs are independent of diffusion noise level, so the auxiliary gradient cannot acquire high-noise x0 amplification. The regularizer is a learned forecast, not an observed-state-only penalty with zero learning effect.

## Expected benefit and limitations

With twelve trajectories, a raw vector MLP must separately discover geometric subtraction, tool-local coordinates, velocity coupling, and repeated red/blue transport structure. Explicit relation inputs and tied encoder/head weights offer these features and share statistical strength across both transports. Ordered full-scene context lets the learned policy distinguish an already-placed block from a pending block, while short-horizon forecasting discourages a representation that discards transition-relevant motion. The broad plus bounded-local features retain long-range approach information and finer grasp/placement offsets simultaneously.

These are hypotheses, not measured performance improvements. No test layouts, rollout feedback, or performance-based selection have been used. The forecast is condition-only, not action-conditioned physics: at ambiguous dwell states it may average possible futures. It cannot certify attachment, containment, collision clearance, exact kinematics, or successful actions. The raw conditioning bypass also permits the model to rely on correlations instead of using the intended relations. Tied block encoding does not remove demonstrated order bias. Two observations and no memory cannot identify every hidden demonstration timing decision. Demonstrations contain no evidence of major failures, regrasping, large rotations, changed fixtures, or far-away starts, so recovery and extrapolation are not guaranteed. Quaternion-derived robustness does not establish rotation-equivariant joint control. Cartesian scale constants are numerical feature choices, not a motion planner or rigid grasp constraint.

## Training, deployment, and termination contract

Use the fixed seed 0, 60,000 updates, batch 128, AdamW learning rate 1e-4 and weight decay 1e-6, cosine schedule with 500 warmup updates, max gradient norm 1, and EMA decay 0.999. Select the last EMA at the predeclared budget. History length is 2 and action horizon is 16 throughout. The interface check's two training updates and reload checks are diagnostics only and do not initialize formal training or establish task success.

All observed releases, retreats, block-to-block transitions, and final tail states remain part of the complete-trajectory training contract. The unchanged evaluator terminates on simultaneous red_at_goal AND blue_at_goal geometry or the physical step limit. There is no extra release, TCP clearance, low-velocity, or hold requirement. The forecasts and metadata do not issue termination or action commands.
