# PRIOR for `acquire_block__h01`

## Assigned heuristic

The assigned hypothesis is **active-object relative acquisition** for the `acquire_block` skill in `tray_pack`: represent pickup in the selected block's object/goal frame and share one manipulation law across the red-first and blue-second acquisitions.  The model must cover the full expanded slices:

- red acquisition and early carry: indices `[0, 250)` in each demonstration,
- blue acquisition and early carry: indices `[330, 625)` in each demonstration,
- including the measured overlaps with `place_block` around `[185,250)`, `[330,420)`, and `[580,625)`.

The intended causal stage rule is red first; after the red block is observed placed at its tray goal and the gripper has reopened, the active role changes to blue.  Future observations are used only as training labels by the framework; deployment receives only the two most recent observations.

## Executable adaptation

The repository actions are absolute Panda joint targets plus a gripper command, and no inverse kinematics or forward kinematics provider is available.  Therefore I did not implement a Cartesian controller.  Instead, this is a standard learned epsilon-prediction diffusion policy over the native normalized 8-D action sequence.  The prior is implemented in the learned conditioning representation and in trainable auxiliary losses.

The policy uses the single shared normalizer fitted on the complete original demonstrations, exactly as supplied in the assignment.  Relative geometric features are scaled only by broad fixed metre constants (`0.4 m` active-relative, `0.6 m` inactive-relative, `0.3 m` height), not by a per-skill or per-segment refit.

## Architecture

`build_model(spec)` returns `ActiveObjectRelativeDiffusionPolicy`.

Inputs to `forward` are:

- `noisy_action [B,16,8]`,
- diffusion timestep,
- `raw_history [B,2,47]` containing qpos, qvel, TCP pose, red pose, blue pose, drawer compatibility channels, and goals.

The condition encoder has four parts:

1. **Absolute state encoder.** The two raw observations are normalized with the shared full-task normalizer and encoded by an MLP.  This preserves robot joint configuration information needed to predict absolute joint targets.
2. **Shared active-candidate encoder.** Two candidate feature tensors are built: one assuming red is active and one assuming blue is active.  Both are passed through the same MLP.  Candidate features include qpos/qvel, TCP-active displacement, active-goal displacement, inactive-active displacement, TCP-goal displacement, active height, finger width, TCP quaternion, active quaternion, and active-frame TCP quaternion.
3. **Causal role module.** A learned MLP reads only the latest causal observation and geometric stage cues, then predicts a blue-active logit.  The red and blue candidate embeddings are mixed by the predicted probability.  The inactive embedding is also provided so the model can keep track of the other block without changing the active-object sharing assumption.
4. **Diffusion condition.** The absolute embedding, active embedding, inactive embedding, and role summary are fused to a 256-D condition vector for `appl.public.DiffusionBackbone`, the provided conditional U-Net.

The model remains a learned action diffusion model: all executed actions come from the DDPM sampler decoding learned epsilon predictions.  There is no replay table, scripted waypoint sequence, IK routine, or nonlearned action controller.

## Losses and gradient paths

`compute_loss` returns `loss`, `diffusion_loss`, and `prior_loss`.

- `diffusion_loss`: standard masked epsilon MSE from `appl.public.epsilon_loss`; gradients flow through the full condition encoder and U-Net.
- `role_binary_cross_entropy` with weight `0.05`: the trainable role logit is supervised by a causal red-placed predicate label.  The label is blue-active when the current red pose is near its red goal in XY and at low tray placement height; otherwise the label is red-active.  This uses no future state at deployment and matches the measured transition evidence.
- `phase_lift_and_finger_width_mse` with weight `0.02`: a trainable head predicts current active-object lift progress and gripper closed progress from the condition vector.  Targets are computed from current causal observations, but the loss is on model predictions and therefore is not a constant penalty.
- `active_tcp_and_goal_geometry_mse` with weight `0.01`: a trainable head predicts the current active TCP-relative vector and active goal-relative vector.  This keeps the fused condition sensitive to the role-relative geometry that defines the prior.

No auxiliary loss is a pure observation-only penalty; each auxiliary term depends on trainable predictions.

## Causal deployment information

At deployment the model receives only the two-observation history.  It can take over partway through a transition because the condition contains:

- finger positions and qvels, indicating open approach versus closed carry,
- TCP pose relative to each block,
- each block pose relative to its goal,
- red-at-goal evidence for switching to blue,
- active-object height for table approach versus lifted carry,
- the inactive block pose to avoid confusing the two objects.

This allows the same diffusion model to represent initial red approach, red lift/carry, red-place retreat/blue approach, blue closure, and blue lift/carry in the expanded slice.

## Applicability and termination cues

The policy is intended for the supplied `tray_pack` state schema with the same Panda/mimic gripper, cuboid blocks, and demonstrated table/tray geometry.  It is appropriate when an active block is on the table or being carried toward its tray goal and the gripper can execute the demonstrated top-down acquisition pattern.

A useful handoff to `place_block` occurs when the active block is close to the TCP, fingers are closed around the demonstrated width, and the active block is lifted and approaching or above its goal.  The measured endpoints may still have nonzero velocity, so the policy does not assume a static handoff state.

## Limitations

The training data contains successful demonstrations only.  The policy has no image encoder, explicit collision constraint, grasp verifier, recovery from missed grasp, or IK module.  The learned role selector may fail if both blocks are already moved, if red is not clearly placed but blue should be acquired, if a block is outside the demonstrated start region, or if the gripper state contradicts the normal red-then-blue order.
