# Prior for `side_grasp_red_pull_drawer_open__h01`

## Assigned heuristic

The assigned heuristic is **red-relative side-grasp servo** for the skill `side_grasp_red_pull_drawer_open`, heuristic index 1.  The original observation is that the demonstrations move from the home arm posture to a fixed side-grasp relation with the red block, close the fingers, pull the red/drawer system along the drawer x-axis until the drawer is open, then reopen and retreat upward/away.  The expanded M1_v2 slice covers indices `[0, 330)` in each of the twelve demonstrations, including the shared transition interval `[240, 330)` with the following top-regrasp skill.

## Executable adaptation

The repository provides state observations, joint/gripper action labels, DDPM noising/sampling, normalization, optimizer, and execution.  There is no inverse kinematics provider, no image encoder, and no hard contact solver available to this policy.  I therefore implement the heuristic as a learned conditional action diffusion model whose conditioning representation emphasizes the red/TCP/drawer geometry that the heuristic identifies, rather than as a scripted Cartesian servo.

At deployment, the model receives only the two causal observations supplied by the framework: qpos, qvel, TCP pose, red pose, blue pose, drawer position/velocity, and goal positions.  Future observations are used only as training targets for auxiliary prediction heads.

## Model architecture

`policy.py` builds a Torch module with these parts:

1. **Causal feature builder.**  The full two-step observation history is normalized with the single shared M1_v2 normalizer.  Additional broad-scale physical features are concatenated:
   - TCP minus red, TCP minus blue, object-to-goal, and object-to-drawer-center offsets.
   - Finger aperture and finger balance.
   - Drawer position and velocity.
   - The inferred invariant `red_x + drawer_position`, which is approximately constant while the red block co-moves with the drawer front.
   - TCP error to a side-grasp/pull family near `red + [-0.245, -red_y, +0.065]`, represented without issuing Cartesian commands.
   - Two-observation finite differences for TCP, red, blue, drawer, fingers, relative TCP-red offset, and qpos.
   - Smooth progress flags for early contact, open drawer, open/closed fingers, and high retreat TCP.

2. **Learned condition encoder.**  An MLP maps those features to a hidden vector.  A trainable phase head predicts five coarse phases: approach, closed pre-pull/contact, pull, release/low retreat, and high retreat.  The soft phase distribution selects a learned phase embedding that is fused back into the diffusion condition vector.  This is a learned phase representation, not a fixed schedule; at inference it is computed from the causal observation.

3. **Action diffusion decoder.**  The condition vector drives the provided `appl.public.DiffusionBackbone`, an epsilon-predicting conditional U-Net over the normalized 16-step action horizon.  The policy output remains the DDPM-predicted action noise; the fixed sampler decodes normalized absolute Panda joint targets and the gripper command.

4. **Auxiliary future head.**  A trainable head predicts short-horizon future qpos, TCP xyz, red xyz, drawer position/velocity, TCP-red offset, `red_x + drawer_position`, finger width, and TCP side-grasp error.  These predictions are used only for training the representation and are not an execution controller.

## Loss terms and gradient paths

The total loss is

`diffusion_loss + 0.05 * (future_geometry_loss + 0.10 * phase_cross_entropy)`.

- `diffusion_loss` is the standard masked epsilon MSE between the model's predicted DDPM noise and the sampled noise.
- `future_geometry_loss` is a masked MSE between trainable auxiliary predictions and future observation targets over the same horizon.  Absolute observation channels use the shared global normalizer; relative geometry uses broad metre-scale constants rather than fitted per-skill scales.
- `phase_cross_entropy` trains the phase logits from labels derived from the current causal state.  The phase logits also influence the diffusion condition through the learned phase embedding, so gradients from both diffusion and the phase loss shape the encoder.

The auxiliary losses depend on trainable model outputs.  They are not constant penalties computed only from observed poses.

## Full expanded slice and overlap behavior

The model is trained over the full `[0, 330)` slice.  It therefore learns:

- Initial approach from the home posture with open fingers.
- Low side approach to the red block.
- Finger closing and contact-supported pull.
- Drawer opening to approximately 0.30 m.
- Release and upward/away retreat during the `[240, 330)` overlap with `red_top_regrasp_lift`.

The observation information that lets the model take over partway through the transition is causal and state-based: drawer position/velocity indicate pull progress, finger aperture indicates whether the side grasp is closed or released, TCP pose and TCP-red relative offset indicate whether it is still at the low side grasp or already retreating, and red pose plus `red_x + drawer_position` indicate whether the red block has co-moved with the drawer front.  Thus an invocation inside the overlap can learn to continue opening/releasing/retreating without needing an absolute timestep.

## Applicability and limitations

This prior is intended for the measured demonstration support: red initially near the drawer front at z about 0.063 m, drawer initially closed, gripper open, blue untouched, and object/drawer variation similar to the twelve original trajectories.  It does not implement recovery from a missed side pinch, large red pose changes, altered drawer friction, or object rotations outside the demonstrated range.  It has no image perception, no IK, no contact constraint projection, and no scripted action replay.  All action choices are produced by the learned diffusion model conditioned on causal state history.
