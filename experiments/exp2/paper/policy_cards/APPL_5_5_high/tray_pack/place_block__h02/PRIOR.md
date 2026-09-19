# PRIOR for `place_block__h02`

## Assigned heuristic

The assigned heuristic is **support-gated release and retreat** for the `place_block` skill in the tray-pack task.  The demonstrations show a repeated sequence: carry a grasped block toward the tray target, lower it into the marked region, wait until the object is supported near the tray floor, open the mimic gripper, and retreat upward/away.  The heuristic specifically targets a common diffusion-policy failure mode: averaging closed-gripper descent with open-gripper retreat and therefore opening too early or dragging the block upward after release.

This M1_v2 policy covers the full expanded placement slices, not only the visually obvious release instant.  The red placement slice includes the overlap with the successor acquisition policy from roughly indices 330-420, where the red block is already released and the open gripper retreats/reaches toward the blue block.  The blue placement slice covers high carry, descent, release, and final retreat.

## Executable adaptation

The source heuristic refers to latent phases and support/contact, but the provided state data has no force/torque or contact labels.  I therefore implement support as a learned kinematic posterior trained from state-derived labels: an active block is considered supported for supervision when its world-frame z is close to the corresponding target z (about 0.036 m) and its xy position is inside the goal funnel.  Future observations are used only to create training labels for auxiliary losses.  Deployment forward passes receive only the two causal observations supplied by the API.

The active block is inferred for training labels from causal state: blue is treated as active when it is elevated, close to its goal after red is already placed, or grasped with closed fingers after red is placed; otherwise red is active.  This preserves the red retreat/transition overlap: an open gripper moving toward the still-tabletop blue block remains part of red retreat rather than being relabeled as blue placement.

## Model architecture

The model is still a learned action diffusion model.  It predicts DDPM epsilon for 16-step, 8-dimensional normalized action trajectories with the fixed sampler and optimizer supplied by the repository.

Implemented components in `policy.py`:

1. **Causal state encoder**
   - Inputs: raw two-step history `[B,2,47]` only.
   - Uses the shared M1_v2 full-demonstration normalizer from `spec`; no per-skill normalizer or scale refit is introduced.
   - Concatenates shared-normalized raw state history with fixed world-frame engineered features: TCP-to-object vectors, object-to-goal vectors, xy distances, z errors, finger width/velocity, one-step object/TCP deltas, and active-block cues.

2. **Trainable latent prior heads**
   - Five-way phase head for `carry_to_funnel`, `descend`, `supported_settle`, `open_release`, and `retreat`.
   - Support probability head.
   - Active-object future-height head over the diffusion horizon.
   - The soft phase posterior, support probability, and height prediction are concatenated with the encoder hidden state and projected to the global conditioning vector.

3. **Diffusion backbone**
   - A `DiffusionBackbone` Conditional U-Net receives the noisy action, DDPM timestep, and the learned condition vector and predicts epsilon.

## Losses and gradient paths

The total loss is diffusion loss plus soft prior losses:

- `diffusion_loss`: standard masked epsilon MSE from the public interface.  This trains the full encoder, conditioning projection, and U-Net because the U-Net condition depends on the learned latent posterior.
- Phase cross entropy: supervises the trainable phase head from causal current-state labels derived from the demonstrations.
- Support BCE: supervises the trainable support head.
- Future active-object height MSE: supervises the trainable height head using `future_obs` labels during training only.
- Clean-action gripper gate loss: reconstructs the predicted clean action `x0` from the model's epsilon prediction and penalizes the gripper command to be closed before support and open after observed release.  This term has gradients through the predicted epsilon and therefore trains the denoising network; it is down-weighted at very noisy DDPM steps to avoid excessive amplification.

No auxiliary term is a constant penalty on observed poses alone.  Every prior loss contains a trainable prediction: phase logits, support logits, future-height predictions, or predicted clean gripper action.

## Causal deployment inputs

At inference the model uses only:

- qpos/qvel including finger width and finger velocity,
- TCP pose,
- red and blue object poses,
- red and blue goals,
- two-step state differences computed from the causal history.

It does not use future observations, demonstration indices, images, force signals, inverse kinematics, or a scripted action controller.  The active-block and phase information used at inference is represented by the learned encoder and its trainable heads, not by an external rollout program.

## Handoff behavior over the expanded slice

For entry partway through a transition, the observation history contains enough state to identify the intended continuation: object height relative to its goal, finger width, TCP-object separation, and whether red and/or blue is already at a goal.  If the object is high and fingers are closed, the learned phase posterior can condition descent.  If the object is at z near 0.036 m with closed fingers, it can condition settle/release.  If the object is stable at the goal with open fingers and the TCP is high or moving away, it can condition retreat.

For red placement, the model also learns the open-gripper retreat/reach phase shared with `acquire_blue`.  The inference API may continue this policy through that overlap when it needs a safe retreat, or transfer to acquisition once red is placed and the gripper is open.  For blue placement, the useful exit is simply both blocks at their respective tray goals, optionally with the TCP already retreated.

## Limitations

Support is inferred from kinematics rather than measured contact.  The learned prior may fail for tilted blocks, unexpected tray geometry, externally disturbed grasps, or object poses outside the training distribution.  The gripper gate is an auxiliary learning bias, not a hard safety constraint; the sampled diffusion action can still violate it.  The policy has no implemented IK, collision checker, image encoder, or force feedback.
