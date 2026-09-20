# PRIOR: side_grasp_red_pull_drawer_open__h03

## Original heuristic identity

The assigned heuristic is **Drawer-progress coordinate** for `side_grasp_red_pull_drawer_open`: organize the red-block side-grasp pull around the scalar drawer opening coordinate.  The research hypothesis is that the low-dimensional variable

`c = drawer_position / 0.30`

is a useful control and termination coordinate for this skill.  Until the drawer is safely past the task threshold (`drawer_position > 0.26 m`), actions should favor monotonic world-x drawer progress.  Once the drawer is open, the same expanded skill slice includes the shared transition in which the gripper opens and the arm retreats toward the successor top-grasp setup.

The implemented policy keeps this identity and is trained on the full assigned expanded slice, indices `0:330` of each demonstration, including the `240:330` overlap with the next policy.

## Executable learned diffusion model

This package implements a real learned action diffusion policy.  `model.forward(noisy_action, timestep, raw_history)` predicts DDPM epsilon for the 16-step sequence of eight native robot actions after affine action normalization by the framework.  It does not output scripted joint targets and it does not replay demonstrations.

The denoiser is the supplied `appl.public.DiffusionBackbone` Conditional U-Net.  The global condition is a learned 256-dimensional embedding produced from causal state history only.  Deployment receives only the two observations in `raw_history`; future observations are used only as training labels for auxiliary losses.

## Causal conditioning representation

The encoder uses the fixed shared M1_v2 normalizer supplied in the assignment.  It does not fit a per-skill normalizer or tiny range scales.  Its input features are:

- the two-step shared-normalized observation history;
- the shared-normalized difference between the newest and previous observation;
- world-frame relative vectors for TCP-to-red, TCP-to-drawer-center, red-to-drawer-center, red-to-red-goal, and blue-to-blue-goal;
- drawer progress terms: `drawer_position / 0.30`, previous progress, progress delta, `drawer_velocity / 0.30`, margins to `0.26 m` and `0.30 m`, and a threshold-margin feature;
- current finger openings and finger velocities.

The drawer center feature uses the assignment convention `drawer_origin_world + [-drawer_position, 0, 0]` with `drawer_origin_world = [0.19, 0, 0.035]`.

These causal features let the model take over partway through the transition: if `drawer_position` is already near 0.30 m, `drawer_velocity` is near zero, and the fingers are open, the condition identifies the release/retreat phase rather than the active pull phase.

## Prior losses and gradient paths

The primary loss is the standard epsilon diffusion loss against the sampled Gaussian noise.  The prior is implemented by trainable auxiliary heads, not by constant penalties computed only from observations.

From the denoising prediction, the code forms the DDPM clean-action estimate

`x0 = (noisy_action - sqrt(1-alpha_bar) * predicted_epsilon) / sqrt(alpha_bar)`.

A trainable action-effect network reads this predicted action sequence together with the causal condition and predicts, for every horizon slot:

1. future drawer progress `drawer_position / 0.30`;
2. scaled future drawer velocity;
3. a logit for crossing the open threshold `drawer_position > 0.26 m`;
4. future red-block y/z residuals relative to the current red pose.

The auxiliary loss is a weighted sum of progress MSE, drawer-velocity MSE, threshold BCE, red-y/z effect MSE, and a small monotonicity penalty on the predicted progress trajectory.  A causal phase classifier also predicts approach, active-pull, or open/retreat phase from the condition.  The effect losses backpropagate through the trainable effect head and through `x0` into the denoising U-Net, so they can shape the action diffusion model.  The auxiliary `x0` is clamped to `[-1.5, 1.5]` and SNR-weighted to avoid high-noise DDPM steps dominating early training.

## Adaptation from the source heuristic

The source heuristic suggested penalizing TCP velocity components orthogonal to world x via Cartesian reconstruction from FK.  The public interface provides joint states and TCP pose observations but no analytic forward-kinematics provider for converting candidate joint action sequences into TCP Cartesian velocities.  Therefore, this implementation adapts that part into a learned effect model: predicted denoised actions are required to support future drawer progress while matching the demonstrated small red-block lateral y/z drift.  This preserves the scientific intent of a progress-oriented pull without claiming unimplemented FK, IK, or hard Cartesian constraints.

## Full slice and transition behavior

The policy covers:

1. approach from the closed start with the gripper open;
2. side grasp/contact with fingers closing around the red block;
3. pulling red/drawer along world negative x while drawer progress rises toward 0.30 m;
4. release once the drawer has crossed the threshold;
5. retreat upward/backward toward the successor top-grasp configuration in the `240:330` overlap.

The model does not redefine task success.  The task completion contract remains the three-goal condition: drawer open, red block on the marked outside pad, and blue block inside the drawer at the same observation.  This policy is only the drawer-opening/red-pull skill and hands off before blue placement.

## Applicability and termination cues

Applicable observations are closed or partly-open drawers with reliable `drawer_position` and `drawer_velocity`, visible state of the red block, TCP pose, and fingers.  Continue the policy while drawer progress is below the open threshold or while it is still completing the learned release/retreat transition.  Prefer handoff when the causal state shows `drawer_position > 0.26 m`, close to 0.30 m, `drawer_velocity` near zero, open fingers, and a retreated TCP suitable for top regrasp.

## Limitations

The model has no image input, contact classifier, analytic IK/FK, model-predictive controller, or hard monotonic drawer constraint.  It may be overconfident if the red block slips sideways, if the side grasp is poor, or if the drawer state is corrupted.  The red and blue final task goals are present in the observation and relative features but this policy is not intended to place the blue block or finish the complete task by itself.
