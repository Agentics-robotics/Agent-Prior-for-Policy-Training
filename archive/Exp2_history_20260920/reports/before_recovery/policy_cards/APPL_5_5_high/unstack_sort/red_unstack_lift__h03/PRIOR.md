# Prior for `red_unstack_lift__h03`

## Assigned hypothesis

The assigned heuristic is the support-preservation prior for the `red_unstack_lift` skill.  In this task the red cube begins stacked on the blue cube.  The skill should grasp the top red cube, lift it clear of the blue support, and begin carrying red toward the red goal while keeping the blue cube approximately fixed at the source.  The expanded training slice is the full interval `[0, 190)` for each demonstration, including the shared transition `[130, 190)` where the red cube is already clear and lateral transport begins.

The original hypothesis says that generalization for unstacking depends on representing blue as a passive support/obstacle, increasing red-blue vertical clearance before significant lateral transport, and minimizing unintended blue motion.

## Executable learned diffusion implementation

The policy remains an action diffusion model.  Its `forward(noisy_action, timestep, raw_history)` returns epsilon predictions for the fixed DDPM sampler.  It does not run a scripted controller, IK solver, sampling rejection rule, or replay policy.

Architecture:

- A learned condition encoder receives the two causal raw observations and the shared global observation normalization supplied by the experiment.
- The condition encoder also receives causal world-frame relative features computed from the same observations: `tcp - red`, `tcp - blue`, `red - blue`, red/blue goal offsets, one-step TCP/red/blue motion, gripper aperture, and TCP/red/blue quaternions.  These features use broad metre scales such as 0.5 m and do not refit a small per-skill normalizer.
- The denoiser is the provided conditional 1-D U-Net backbone with a 256-dimensional learned condition vector.
- A trainable auxiliary GRU/MLP future decoder maps a candidate clean encoded action horizon plus the causal condition to predicted future deltas for red xyz, blue xyz, TCP xyz, and gripper aperture.  This decoder is used only for training losses; deployment sampling still uses the DDPM denoiser only.

## Losses and gradient paths

The total loss is

`L = L_DDPM + L_prior`.

`L_DDPM` is the normal epsilon MSE on the normalized action horizon with the provided action mask.

`L_prior` implements the support-preservation hypothesis with trainable predictions:

1. **Auxiliary future supervision.**  The auxiliary decoder predicts future red, blue, TCP and aperture deltas from the candidate action horizon and causal state.  It is supervised by `future_obs` labels inside the assigned slice.  The decoder is trained both from the clean demonstration action and from the DDPM clean-action estimate `x0` reconstructed from the predicted epsilon.  The latter path gives gradients from future-state consistency back to the denoiser.
2. **Blue preservation.**  The predicted future blue xy displacement is penalized toward zero, using the auxiliary decoder output.  This is not a constant penalty computed directly from observed blue poses; it depends on the learned action-conditioned future prediction.
3. **Clearance before lateral transport.**  From predicted red and blue future positions, the loss penalizes large predicted red xy displacement from the current red position while predicted red-blue z clearance is below about 0.10 m.  This encodes the desired vertical-lift-before-lateral-motion ordering in a differentiable way.

A smooth soft clip is applied to the reconstructed `x0` used by the auxiliary branch to avoid unstable gradients at very noisy diffusion timesteps.  The DDPM objective remains the primary action-learning objective.

## Causal deployment inputs

At deployment the model receives only the two most recent observations: joint state, TCP pose, red pose, blue pose, zero drawer compatibility channels, and the fixed red/blue goals.  Future observations are used only as labels in `compute_loss`.  The policy can take over during the overlap transition because the causal observation contains the current red-blue clearance, blue table-level pose, TCP-red relation, gripper aperture, and current object/TCP motion.  These cues distinguish approach, grasp/closure, vertical lift, and the beginning of red lateral transport.

## Handoff behavior

The model covers the complete expanded slice `[0, 190)`: initial approach from an open gripper, descent to red, gripper closure on red, vertical lift that separates red from blue, and the first lateral carry toward the red goal.  It should be invoked when red and blue are stacked and the blue cube should remain a passive support.  It is ready to hand off when red is high above blue, the gripper is closed around/near red, blue_pose remains near its original table-level pose, and red/TCP motion has begun toward the red goal.

## Dependencies and limitations

This implementation depends on accurate state observations for `red_pose`, `blue_pose`, `tcp_pose`, `qpos`, and goals.  It does not observe contact forces, so blue preservation is a learned kinematic bias rather than a certified non-contact constraint.  It does not implement hard containment, hard clearance constraints, action rejection, image encoders, equivariance, IK, or recovery from a displaced blue cube or failed red grasp.  The completion contract remains the task-level red/blue goal predicate supplied by the assignment; the handoff cues are guidance for policy selection, not a redefinition of task success.
