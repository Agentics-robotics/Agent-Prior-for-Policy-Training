# PRIOR: red_top_regrasp_lift__h01

## Assigned heuristic

The assigned hypothesis is a **red-centered top-grasp affordance** for the `red_top_regrasp_lift` skill.  The skill starts while the drawer-opening policy is finishing its release/retreat and spans the transition into the red transport policy.  In the demonstrations the drawer is already open, the red block rests on the open drawer/front surface at about z = 0.063 m, and the robot reorients from the side-release posture into a top-down grasp, closes the fingers, lifts the red block, and begins moving toward the outside pad.

The full expanded training slice is indices 240--530 for each demonstration.  This implementation intentionally covers the shared transition phases: the early overlap with drawer opening (roughly 240--330), the central regrasp and lift, and the late overlap with transport/place (roughly 455--530).

## Executable adaptation of the heuristic

The repository provides state observations and joint/gripper action labels, but no image encoder, IK solver, contact model, or forward kinematics service.  Therefore the prior is implemented as a learned representation and learned auxiliary prediction inside an action diffusion model rather than as a geometric controller.

At deployment the model receives only the causal two-step observation history.  It uses:

- normalized complete raw state history using the single shared M1_v2 normalizer fitted on the original full demonstrations;
- current and previous TCP pose;
- current and previous red pose;
- gripper finger positions and finger velocities;
- drawer position and drawer velocity;
- red goal position;
- relative TCP-to-red position and relative quaternion features.

No future observations are used by `forward`.  Future observations are used only as training labels for the auxiliary learned loss.

## Architecture

`build_model` returns `RedTopRegraspLiftPolicy`, a learned DDPM epsilon predictor.

1. **Raw-state encoder.** The two normalized 47-D observations are flattened and passed through an MLP.
2. **Red-centered affordance encoder.** A second MLP encodes engineered causal features in a red-centered frame: TCP minus red position, previous relative position, relative motion, red/TCP motion, red-to-goal vector, red-to-drawer vector, drawer opening margin, finger gap, finger asymmetry, finger velocities, TCP/red quaternions, relative quaternions, TCP-red XY distance and top-down quaternion cue.
3. **Condition fusion.** The two encodings are fused into a 256-D global condition.
4. **Diffusion backbone.** The fused condition drives the provided conditional 1-D U-Net backbone, which predicts epsilon for normalized 16-step native joint/gripper action trajectories.
5. **Auxiliary learned dynamics head.** During training only, the predicted clean action estimate is encoded per horizon step and combined with the same condition to predict future red-centered relative geometry and a lifted/top-grasp attachment logit.

The auxiliary head is not a controller and is not queried by the sampler.  It supplies gradients that encourage the shared condition and action denoising pathway to represent the red-centered lift consequences of candidate action sequences.

## Losses and gradient paths

The total loss is:

`loss = diffusion_loss + prior_loss`

- `diffusion_loss` is the standard masked epsilon-prediction MSE against the DDPM noise target.
- `prior_loss` is a weighted sum of:
  - masked MSE for predicted future red-centered geometry: future TCP-minus-red, future red displacement from the current red pose, and future finger gap;
  - masked BCE for a learned lifted/top-grasped state label defined from future red height, gripper closure, and TCP-red proximity.

The clean action estimate is reconstructed from the model's predicted epsilon and the noisy action.  The auxiliary loss therefore has gradients through the auxiliary head, the condition encoder, and the diffusion epsilon output.  To avoid high-noise amplification, this auxiliary path is weighted by the DDPM alpha-bar and the clean action estimate is bounded to a conservative normalized range.

The future labels are not used at inference.  They only shape trainable predictions during training, satisfying the prior without adding scripted actions.

## Causal handoff information

The policy can take over partway through the transition because the causal state contains the necessary cues: drawer opening, TCP pose relative to the red block, finger aperture, red pose, and recent TCP/red motion.  At the early boundary the gripper is open and the TCP is retreating from the drawer-opening side grasp; the model learns the demonstrated upward retreat and reorientation.  At later overlap entries the TCP is already high above the red block, so the same red-centered frame indicates that the next useful behavior is centering, descending, closing, and lifting.

A useful exit state for the successor is a closed top grasp with red and TCP co-moving, red z above the drawer lips, and motion starting toward the outside pad.  The implemented policy does not define final task success; task success remains the supplied three-goal completion contract.

## Applicability and limitations

This policy is intended for states similar to the demonstrations: drawer open, red block on the open drawer/front surface, and sufficient clearance for a vertical top-down approach.  It does not implement explicit retry, collision checking, object search, IK, image perception, or a hard state-machine.  It is a learned state-conditioned action diffusion model trained on successful examples over the assigned full expanded slice.
