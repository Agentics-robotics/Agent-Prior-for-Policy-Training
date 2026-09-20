# Prior document: blue_approach_grasp_lift__h01

## Original heuristic identity

The assigned heuristic is **Blue-relative top-grasp servo** for the `blue_approach_grasp_lift` skill.  Its role is to take over after the red block has been placed on the outside pad, move from the red-pad area toward the blue block, top-grasp the blue block, close the gripper, and lift the blue block high enough for the transport-to-drawer successor.

The expanded training slice is retained exactly: indices 650 through 870 for each demonstration.  This includes the shared transition from the predecessor during indices 650-760 and the shared contact/lift transition to the successor during indices 810-870.  The implementation therefore learns the whole motion from red-pad retreat/global transit, through blue-centered approach and grasp closure, to lifted blue-block readiness.

## Executable adaptation of the heuristic

The source heuristic describes a blue-relative servo, but the available policy interface outputs normalized absolute Panda joint targets plus one gripper command.  No inverse kinematics, forward kinematics provider, image encoder, or external geometric controller is available.  I therefore implement the servo idea as a **learned action diffusion model** whose conditioning representation is anchored on the observed blue pose and whose training loss encourages the denoised action sequence to predict contact and lift readiness.

At deployment the model receives only the two most recent causal state observations.  It does not receive future observations, trajectory indices, or scripted phase commands.  Phase information is inferred from observed TCP-blue offset, gripper opening, blue height, red-pad proximity, drawer position, and recent state deltas.

## Model architecture

`policy.py` defines `BlueRelativeDiffusionPolicy`.

- The diffusion backbone is `appl.public.DiffusionBackbone`, the provided conditional 1-D U-Net for epsilon prediction over action sequences of shape `[B, 16, 8]`.
- The learned condition encoder is an MLP with SiLU activations and layer normalization.  It outputs a 256-dimensional global condition vector.
- The encoder input contains:
  - the two raw observations normalized by the shared M1_v2 normalizer supplied in `spec`;
  - per-history-step TCP-minus-blue, TCP-minus-red, blue-minus-blue-goal, red-minus-red-goal features using broad fixed metre-scale denominators;
  - summed finger opening and finger velocity cues;
  - red-on-pad and drawer-open causal cues;
  - recent TCP, blue, and joint deltas between the two observations;
  - last-frame phase cues such as TCP-blue XY distance, TCP-blue vertical offset, blue height above table, finger opening, red-pad distance, and drawer-open margin.

All learned actions remain the framework's normalized absolute joint/gripper actions.  The relative representation changes the learned conditioning structure; it does not claim exact equivariance of joint commands.

## Loss terms and gradient paths

The primary loss is the standard DDPM epsilon MSE supplied by `epsilon_loss(predicted_noise, noise, mask)`.  This trains the model to denoise normalized native action sequences.

The prior loss is an auxiliary differentiable future-readiness loss.  From the predicted epsilon, the code forms a DDPM clean-action estimate

`x0 = (noisy_action - sqrt(1-alpha_bar) * predicted_epsilon) / sqrt(alpha_bar)`.

A trainable auxiliary head receives this `x0` estimate, the horizon coordinate, and the same causal condition embedding.  It predicts:

1. future blue z relative to current blue z;
2. future TCP-minus-blue xyz;
3. future summed finger opening;
4. a contact label derived from future TCP-blue proximity and closed fingers;
5. a lift label derived from future blue height increase.

The labels are computed from `future_obs` during training only.  They are not available to the deployment forward pass.  Because the auxiliary head consumes the trainable clean-action estimate, the auxiliary loss has a gradient path through the denoising network as well as through the auxiliary head and condition encoder.  It is not a constant penalty computed only from observed poses.  Low-signal high-noise diffusion steps are down-weighted by `alpha_bar` in this auxiliary term.  The auxiliary term is intentionally small (`aux_weight = 0.05`) so that the model remains primarily a diffusion action learner while biasing the representation toward the grasp/lift hypothesis.

## Causal deployment inputs

The deployment forward call uses only `raw_history [B,2,47]`, `noisy_action`, and the DDPM timestep.  The observation fields used are:

- `qpos`, `qvel` for arm and finger state;
- `tcp_pose` for current end-effector position and orientation;
- `red_pose` and `red_goal` for the causal red-on-pad cue;
- `blue_pose` and `blue_goal` for blue-relative approach and lift geometry;
- `drawer_position` for the causal drawer-open cue.

No future observations, completion labels, hidden simulator state, files, network access, or scripted control laws are used at inference.

## Handoff and transition behavior

At the start of the expanded slice the red block is already on the outside pad, the drawer is open, and the gripper is open near the red-pad area.  The model learns to move away from the pad and toward the blue block.  During the middle of the slice it learns the blue-centered top approach with an open gripper.  Around the contact portion it learns to close the gripper and lift the blue block.  At the end of the slice the supported successor state is a closed grasp with the blue block lifted roughly to 0.24-0.28 m and co-located with the TCP.

The observation information that lets this policy take over partway through a transition is the current TCP-blue offset, current finger opening, recent TCP/blue deltas, blue height, red-pad proximity, and drawer-open margin.  These cues distinguish red-retreat/global transit, high approach above blue, low contact, and lifted states without relying on a time index.

## Applicability and limitations

This policy is appropriate when the scene resembles the demonstrated handoff: red already released, drawer open, blue visible near the demonstrated tabletop region, and the gripper not obstructed.  It is not designed to solve earlier red-placement failures, to search for a missing object, or to recover from large blue-block perturbations.  The implementation does not enforce hard constraints or exact grasp mechanics; contact and lift readiness are learned from demonstrations via the auxiliary prediction loss.
