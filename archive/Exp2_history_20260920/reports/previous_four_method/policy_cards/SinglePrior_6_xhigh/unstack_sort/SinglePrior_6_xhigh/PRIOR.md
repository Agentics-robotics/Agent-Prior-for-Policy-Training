# SinglePrior_6_xhigh: object-centric predictive relation conditioning

## Scientific choice

This is one full-task epsilon-predicting Diffusion Policy, not a sequence of policies. Its prior is that the two transfers can share a representation of **object–hand–goal relations and their short-horizon evolution**. The representation also retains the relation to the other object, since taking red off blue and approaching the subsequently exposed blue are not independent tasks.

A shared object encoder and shared relation forecaster provide structured conditioning to one action diffusion U-Net. A small supervised prediction objective encourages this conditioning to retain information about approach, joint object/hand motion, placement, opening, and leaving an already placed object stationary. It does not label these as discrete phases or enforce a particular order. The model learns their association with joint actions from all complete demonstrations. It does not use a phase classifier, skill router, planner, controller, IK solver, external schedule, or persistent deployment memory.

The expected benefit is sample efficiency: the network need not rediscover all object–TCP and object–goal differences through unrelated dense weights for the two colored objects, and the auxiliary task supplies dense temporal supervision for the conditioning representation. This is a hypothesis based on training evidence, not a claim of measured rollout improvement.

## Training evidence and inspection

The assignment contains twelve complete trajectories, 8,729 action samples in total. No trajectory was cut into skills or filtered by stage. I inspected a 24-frame whole-trajectory overview of `demo10200`, whole-trajectory four-frame overviews of each remaining trajectory, and denser original source observations/actions around transitions in `demo10200` and `demo10207`.

The following table records the four-frame overview indices for the other eleven demonstrations, their initial common red/blue XY coordinates rounded to millimetres, and their terminal source index. The two interior samples in each row show red being lowered toward its pad and then blue beginning its lift while red remains on its pad. Terminal observations have both blocks at their respective pads.

| Trajectory | Initial stack XY (m) | Interior indices | Terminal index |
|---|---:|---:|---:|
| demo10201 | (-0.351, -0.005) | 239, 478 | 717 |
| demo10202 | (-0.355, -0.008) | 243, 486 | 729 |
| demo10203 | (-0.358, -0.011) | 242, 485 | 728 |
| demo10204 | (-0.340, +0.006) | 240, 480 | 721 |
| demo10205 | (-0.339, +0.009) | 241, 482 | 723 |
| demo10206 | (-0.356, -0.008) | 243, 486 | 729 |
| demo10207 | (-0.354, +0.012) | 246, 492 | 739 |
| demo10208 | (-0.361, +0.010) | 245, 491 | 737 |
| demo10209 | (-0.339, +0.012) | 241, 482 | 723 |
| demo10210 | (-0.355, -0.007) | 243, 486 | 730 |
| demo10211 | (-0.342, -0.005) | 240, 480 | 721 |

Specific observations motivating the design:

- `demo10200:0`: red and blue have common XY approximately (-0.3533, -0.0053), with center heights 0.06 and 0.02 m. The TCP starts near height 0.442 m. Inter-object displacement exposes the stack dependency directly, unlike an episode clock.
- `demo10200:63,95,96,100,104,110`: the open gripper approaches red. At 95 its command is +1, at 96 it switches to -1 while the TCP remains near (-0.3624, -0.0053, 0.0614). By 100 the fingers are near 0.01825 m each rather than 0.04 m; 100 and 104 are nearly stationary closed-grasp observations, and 110 starts the lift. There is a systematic TCP/object offset of about 9 mm in X. The model should learn this offset, not force coincident centers or hard-code an ideal grasp pose.
- `demo10200:127,159,190,222,254`: red rises to roughly 0.319 m, traverses toward negative Y while blue stays at its initial support location, and descends toward the red pad. At 190 the red and TCP translations are coupled. Object–TCP and object–goal relation evolution distinguish carrying from approaching a stationary object.
- `demo10200:266,274,286,290,318,350,381`: red is near pad height before the command opens; fingers are already opening at 274, red stays at about (-0.2411, -0.2479, 0.0200), and the TCP retreats and then returns toward blue. Relative geometry plus finger configuration distinguishes release/retreat from another attempt to pick red. Both object tokens are retained through this transition.
- `demo10200:413,445,450,456,462,477,509`: the same broad approach/close/lift relation pattern occurs for blue, now from center height 0.02 m rather than 0.06 m. At 450 the gripper is closing, and at 456/462 it is nearly stationary around blue before lifting. This supports sharing a relation encoder and forecasting function rather than learning two entirely separate mechanisms.
- `demo10200:541,572,604,636,638,646,668,700,732`: blue transfers toward positive Y, descends, and reaches its pad while red stays placed. The demonstration continues through opening and retreat. Geometric completion does not require those last behaviors, but they remain in the full training data and the policy is not truncated at initial success.
- Dense comparison in `demo10207:95,103,112,270,285,450,465,478` shows analogous grasp/lift/place/open transitions. At 450 its command is still +1 and the TCP height is about 0.0296 m above blue's 0.0200 m center. In contrast, `demo10200:450` already commands -1 and has TCP height about 0.0204 m. Source index is therefore not a reliable phase variable. The network uses physical relations and causal motion, not these indices.

The inspected resets span only about 23 mm in X and 22 mm in Y, with essentially fixed goals and upright orientations. This is limited evidence for extrapolation. The stationary dwell examples also show genuine partial-observability issues: two causal observations cannot reconstruct an expert's unobserved pause timer.

## Implemented architecture

### Causal inputs and state scaling

`forward(noisy_action, timestep, raw_history)` receives exactly the noisy normalized action sequence, diffusion timestep, and the two supplied 47-dimensional observations. There is no previous-action input or episode clock. Forward has no mutable cross-call state. Both training epsilon prediction and deployment call the same encoder and backbone.

The robot/global path flattens the two observations after applying the supplied shared mean and half-range scales, without clipping or refitting. It uses a 94→256→128 Mish MLP and LayerNorm. Retaining joints, joint velocities, TCP world pose, and the other original channels is important because output actions are absolute Panda joint targets, not Cartesian displacements. The fixed-zero drawer channels receive no special dynamics or losses.

Each object's frame features contain 41 numbers:

1. Object minus TCP, goal minus object, peer object minus object, and goal minus TCP (four 3-vectors, scaled by 0.25 m).
2. Object and associated goal world positions (two 3-vectors, scaled by 0.5 m).
3. The first two rotation-matrix columns for the object, TCP, and peer (three 6-vectors, computed from wxyz quaternions).
4. Both finger positions divided by 0.04 m.
5. Squared norms of the three scaled object–TCP, object–goal, and object–peer displacements.

The two frames yield 82 numbers. Eleven additional motion features are the object, TCP, and peer position increments divided by 0.025 m, and both finger increments divided by 0.005 m. At 20 Hz the position scale corresponds to 0.5 m/s. A color coordinate (-1 for red, +1 for blue) gives a total of 94 inputs per object. These are fixed physical scales, not divisions by the tiny empirical orientation or layout variation in the demonstrations. The derived rotations do not depend on quaternion sign; the separate unmodified normalized state path is retained as specified.

A single shared 94→256→128 Mish MLP with LayerNorm encodes both objects. A shared 384→256→128 message MLP takes each node, its peer, and the robot latent. Its residual is added to the node and normalized. The message update is simultaneous; neither color receives a hard-coded priority. Color ordering is retained in the final conditioning so this is not a claim of complete object-permutation invariance.

### Predictive relation features

A shared 256→256→96 MLP reads a node and robot latent and predicts 16 slots of six relation changes. The six coordinates are the object's 3D displacement from the TCP and its 3D goal error. Predictions are in units of 0.1 m relative to the currently observed relation. A 256→128→16 MLP reads the robot latent and mean object latent and forecasts mean per-finger position changes, in units of 0.02 m. Forecast output weights start small (normal initialization with standard deviation 0.001 and zero bias).

Each flattened object forecast is embedded by a shared 96→128→128 MLP and added to its object latent with gain 0.25. The aperture forecast is embedded by a 16→64→128 MLP and added to the robot latent with the same gain. Concatenating robot, red, and blue latents yields a 384-dimensional condition. Direct residual paths preserve current-state information when the future is ambiguous. Forecasts are continuous learned features, never thresholds, switches, or native commands.

There is exactly one public `DiffusionBackbone(384, training)` with widths 128/256/512, timestep embedding 128, kernel size 5, groups 8, and its standard FiLM modulation. It predicts an epsilon tensor of shape [B,16,8]. All encoders, forecast heads, projections, and the U-Net are registered in one module/checkpoint. There are no pretrained components or additional models loaded at deployment.

## Losses and actual gradient paths

The primary loss is the supplied masked epsilon MSE, `epsilon_loss(predicted_noise, noise, mask)`. The DDPM target and sampler are unchanged.

For object i define the physical relation

`r_i(s) = concat(p_i - p_TCP, goal_i - p_i)`.

The relation target in slot j is

`delta_r[i,j] = (r_i(future_obs[j]) - r_i(current_obs)) / 0.1`.

The aperture target is

`delta_a[j] = (mean_finger(future_obs[j]) - mean_finger(current_obs)) / 0.02`.

These respect the interface alignment: action slots are t-1 through t+14, and future observations are states after each action. Slot zero normally corresponds to the currently observed state at t, rather than an assumed t+1 label. The loss compares predictions directly to the supplied labels; it never rolls a dynamics model forward.

Let `valid = future_mask * mask`. Both auxiliary errors use elementwise Smooth L1 with beta 0.25. The relation loss averages over valid slots, two objects, and six coordinates; the aperture loss averages over valid slots. Padded terminal/reset entries are not scored. The returned losses are

`prior_loss = relation_loss + 0.25 * aperture_loss`

`loss = diffusion_loss + 0.1 * prior_loss`.

This is a penalty on trainable forecasts, not an observed-state-only geometric penalty. Its direct gradients train the forecast decoders, the robot and shared object encoders, and the shared message function. The forecasts enter the U-Net condition, so epsilon-loss gradients also train both forecast heads and their projection MLPs, as well as the backbone and the direct latent paths. There is no stop-gradient between forecasts and denoising. Only supervised targets are detached.

The auxiliary objective has no direct derivative with respect to the epsilon output layer: it regularizes the causal conditioning representation, not generated-action feasibility. This distinction is intentional. It does not assert that the decoded joint action causes the predicted object motion. No inverse-kinematics or differentiable physics claim is made. It also does not reconstruct x0 from noisy actions, so there is no division by sqrt(alpha_bar) and no high-noise amplification in the auxiliary objective. Its weight is independent of the sampled diffusion timestep.

Unlike an always-on goal penalty, the auxiliary target is the expert's observed short-horizon continuation. It can encourage lifting away from the goal plane, moving an empty hand, or keeping a completed object stationary when that is demonstrated. It does not force both blocks to their terminal goals at every intermediate step.

## Fixed training and deployment contract

All twelve complete trajectories use the shared normalizer. The formal recipe remains seed 0, 60,000 updates, batch size 128, two observations, action horizon 16, execution prefix 8, epsilon prediction, 100 train/inference DDPM steps, sample clipping enabled, AdamW learning rate 1e-4 and weight decay 1e-6, cosine schedule with 500 warmup updates, gradient norm clipping at 1, and EMA decay 0.999. The final EMA at the predeclared budget is selected without performance-based checkpoint selection.

The policy emits normalized action noise predictions only. The supplied sampler performs DDPM and native-action decoding; the evaluator applies the same environment bounds as the baseline. There is no residual action controller or post-sampling overwrite. Seven native channels remain absolute joint targets and the eighth remains the gripper command increasing toward open.

At inference, forecasts are recomputed from the two actual observations on each model call. The future labels are never inputs to `forward`, and no ground-truth future, demonstration identifier, source index, or termination label is available to it. Repeated calls at different DDPM timesteps do not advance memory or progress.

The evaluator's existing simultaneous `red_at_goal AND blue_at_goal` test is authoritative, including rotated-footprint containment and height tolerance. There is no additional release, speed, TCP-clearance, or hold-time condition. Metadata provides documentation, not a runtime stopping rule. Training includes the demonstrated opening/retreat even where the evaluator may already have terminated.

## Limitations and falsifiable expectations

- The relation prior should make repeated approach/carry/place geometry easier to learn from twelve demonstrations, but no held-out or rollout performance is available. Interface checks are numerical/API diagnostics, not evidence of task success.
- Constant goals and a narrow reset distribution do not identify reliable behavior for widely shifted pads, arbitrary stack layouts, rotated cubes, falls, collisions, or failed grasps. The representation can encode these states, but training may not teach the right response.
- Absolute joint actions are not translation- or rotation-equivariant merely because some input features are relative. The robot/world-state path explicitly preserves this distinction.
- Deterministic future heads can average incompatible expert continuations near pauses and gripper switches. Low auxiliary weight, robust regression, residual conditioning, and stochastic diffusion reduce architectural pressure to collapse, but do not solve latent-timer ambiguity.
- Shared weights do not guarantee red-first behavior, successful unstacking, contact stability, completion, or collision avoidance. Those behaviors remain learned from demonstrations. Both colors and both goals remain visible throughout, including after the first placement.
- The forecasters are causal expert-continuation predictors, not action-conditioned consequence models. They cannot certify that a sampled action sequence realizes their predictions. Their contribution is representation learning and conditioning, not an exact mechanics constraint.
- No recovery policy or adaptive deployment agent is available. The same single checkpoint must handle the entire task and any encountered deviations.
