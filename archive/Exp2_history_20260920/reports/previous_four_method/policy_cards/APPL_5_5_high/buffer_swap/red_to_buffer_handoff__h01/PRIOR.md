# PRIOR: red_to_buffer_handoff__h01

## Assigned heuristic identity

This policy implements heuristic index 1 for skill `red_to_buffer_handoff`: an **object-relative contact-funnel prior** for the first transfer in the buffer-swap task. The assigned hypothesis is that the red block transfer is easier to learn if the policy represents the motion in object-relative frames and infers a contact phase, rather than treating the demonstration as a single absolute joint-time trace.

The full expanded training slice is used: source indices 0 through 460 for each demonstration. This includes the red approach, pinch, lift, carry to the central buffer, lower/release, gripper retreat, motion toward the blue block, descent over blue, and the beginning of blue closure. The overlap with the successor is therefore deliberately modeled rather than removed.

## Executable adaptation

The repository provides state observations and joint/gripper actions, but it does not provide inverse kinematics, forward kinematics, images, contact labels, or hard constraint solvers. I therefore implement the heuristic as a learned conditional diffusion model over the required native action representation: seven absolute Panda joint targets plus one gripper command. The prior is expressed in the learned conditioning representation and auxiliary trainable losses, not as a scripted controller.

Deployment receives only the two causal observations supplied by the interface. Future observations are used only during training to form auxiliary targets for learned prediction heads.

## Architecture

`policy.py` defines `ContactFunnelDiffusionPolicy`.

1. **Shared-normalized state input**
   - The model stores the single shared M1_v2 normalizer from the spec.
   - Both causal observations are normalized with this shared normalizer; no per-skill or per-phase scale is refit.

2. **Object-relative feature encoder**
   - For each of the two observation steps, the model concatenates the shared-normalized 47-D observation with relative world-frame vectors:
     - TCP minus red position,
     - TCP minus blue position,
     - blue minus red,
     - red/blue goals relative to red and blue,
     - central buffer point `[-0.181, 0.0, 0.02]` relative to red.
   - It also includes broad physical-scale geometry scalars such as TCP-red XY distance, TCP-blue XY distance, red-buffer distance, object height, TCP height, finger width/opening cues, qvel, and finite differences across the two causal observations.
   - Position-relative quantities use a broad 0.25 m scale rather than narrow slice-specific standard deviations.

3. **Learned soft phase representation**
   - An MLP maps the causal features to a hidden vector.
   - A trainable phase head predicts logits for five contact-funnel phases: approach, close/lift, carry, lower/release, and blue-handoff transition.
   - The softmax probabilities weight a trainable phase embedding table. This soft phase embedding is concatenated with the hidden state and fed to a condition head.
   - At inference the phase is not externally scripted; it is inferred by this learned network from causal observations.

4. **Diffusion action model**
   - The condition vector is passed to the provided `appl.public.DiffusionBackbone`, which predicts DDPM epsilon for normalized action chunks of horizon 16.
   - The fixed framework handles noising, sampling, EMA, optimizer, and conversion between normalized and native action spaces.

## Losses and gradient paths

`compute_loss` returns the required `loss`, `diffusion_loss`, and `prior_loss`.

- **Diffusion loss**: standard masked epsilon prediction loss between the backbone output and sampled DDPM noise. Gradients update the object-relative encoder, phase embedding path, condition head, and U-Net backbone.

- **Phase auxiliary loss**: cross-entropy from the learned phase logits to causal geometric phase labels derived from current observed state cues such as red height, red-buffer distance, TCP-object distance, and finger width. The labels are approximate training targets, not runtime rules. The loss depends on trainable phase logits and therefore provides gradients to the encoder.

- **Object displacement auxiliary loss**: the condition vector feeds a trainable head predicting short-horizon red and blue XYZ displacements at offsets `[3, 7, 11, 15]`. Targets come from `future_obs` during training only. This loss trains the same causal representation to distinguish whether red should remain still, lift/carry/lower, settle at the buffer, and whether blue is still stationary or entering handoff. It is masked using `future_mask`.

The prior loss is a weighted sum of the phase and object-displacement losses. There is no constant-only penalty computed solely from observed poses.

## Causal deployment inputs

At deployment the model uses only:

- `qpos`, including finger positions,
- `qvel`,
- `tcp_pose`,
- `red_pose`,
- `blue_pose`,
- `red_goal`,
- `blue_goal`,
- the fixed zero drawer compatibility channels as part of the normalized state.

The model can take over partway through the expanded transition because these observations reveal the local phase: whether red is still at the lower start, lifted with the TCP, near or resting at the buffer, whether fingers are open or closed, and whether the TCP is in the high retreat/blue-approach corridor or low over the blue block. Those causal cues are the basis for the learned soft phase and action denoising condition.

## Applicability and handoff

The policy is intended for successful top-down red manipulation demonstrations with similar Panda/table/block geometry. It is most applicable when red begins near the lower region, blue remains near the upper region, and the TCP/finger state lies within the demonstrated approach, carry, release, or blue-handoff corridors.

A useful successor handoff state is red resting near the central buffer, fingers open or closing according to the blue approach phase, blue still in its upper region, and TCP either high in the approach corridor to blue or low and centered over blue with closure underway. The policy includes the overlap [320, 460), so the inference agent may continue it after red reaches the buffer until the successor has a coherent entry state.

## Limitations

This implementation does not claim or implement inverse kinematics, collision avoidance, image encoding, exact contact estimation, object containment constraints, or recovery behaviors. It learns from the provided state-action demonstrations only. Large object perturbations, missed grasps, moving blocks, clutter, and off-corridor recoveries are outside the demonstrated support. The buffer location is encoded as a fixed broad prior point because the assigned demonstrations use a single central temporary buffer.
