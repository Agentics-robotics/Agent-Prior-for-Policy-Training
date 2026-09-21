# Prior for `blue_acquire_transport__h02`

## Original heuristic identity

The assigned heuristic is the height-staged blue transport prior for `blue_acquire_transport`: after the red block has already been placed, the robot should acquire the blue block, lift it to a safe height, translate it in xy toward the blue goal, and hand off to the final placing policy while the block is near or above the blue goal. The motivating observation is that successful demonstrations lift the grasped blue block before large lateral motion, which should reduce table-level dragging.

M1_v2 assigns the full expanded slice `[330, 590)` for this policy. This includes two shared transition portions: the transition from red placement toward the blue block (`330-520`) and the beginning of goal-near carry/descent shared with the final blue placing policy (`520-590`). Therefore the executable policy is not only a high-level carry model. It also learns the causal transition from an open gripper at the red-place exit, to blue approach, pregrasp descent, grasp/settle, lift, high transport, and goal-near handoff.

## Implemented diffusion policy

The model remains an epsilon-predicting action diffusion policy over the provided native action representation: seven absolute Panda joint targets plus one gripper command. It uses the repository DDPM sampler and the provided Conditional U-Net backbone. The model does not execute a scripted waypoint controller; all deployed actions are sampled by the diffusion model.

The denoiser condition is produced by a learned state encoder. Inputs are the two causal raw observations only. The encoder uses:

- the shared normalized 47-dimensional observations from the two-step history;
- world-frame relative features for `blue_goal - blue_pose`, `red_goal - red_pose`, `tcp_pose - blue_pose`, `tcp_pose - red_pose`, and `blue_pose - red_pose`;
- two-step TCP and blue motion differences;
- gripper finger width and finger-velocity features;
- broad-scale distance and height features;
- a heuristic phase one-hot computed from the current causal state.

The broad scales are task/workspace-level constants, not a refit per-skill normalizer. Observation normalization uses the single M1_v2 shared normalizer supplied in the assignment.

The learned auxiliary heads predict:

1. a six-class phase distribution: red-to-blue transition/open, pregrasp descent/open, grasp-and-settle, lift at source, high xy transport, and goal-near high handoff;
2. a staged waypoint in normalized world coordinates;
3. blue-block displacement at future action slots 3, 7, and 15;
4. an open/closed gripper phase value.

The U-Net condition includes the encoder latent and these trainable predictions. Thus diffusion loss gradients also pass through the representation used to express the prior.

## Losses and gradient paths

`compute_loss` returns the standard diffusion epsilon loss plus a weighted prior loss. The prior loss is differentiable with respect to trainable predictions:

- phase cross-entropy trains the phase head from causal observations;
- waypoint MSE trains a predicted staged waypoint: above-blue for transition/lift, grasp height for pregrasp descent, and above-goal for high transport and goal-near handoff;
- future blue displacement MSE trains the motion-prediction head using `future_obs` labels at slots 3, 7, and 15;
- gripper phase MSE trains a learned open/closed prediction;
- a small low-noise gripper action prior is applied to the DDPM clean-action estimate, encouraging open commands in the early transition and closed commands during lift/transport phases.

The future observations are used only as training labels for auxiliary losses. Deployment forward receives only the causal two-step observation history, the noised action sample, and the DDPM timestep.

## Causal takeover and handoff information

The policy can take over partway through the transition because the observation contains the world-frame TCP pose, blue pose, red pose, red and blue goals, joint/finger positions, and a two-step history. These fields distinguish the relevant states: open gripper leaving the red goal, TCP aligned above the blue block, TCP low near the blue block, fingers closing/closed, blue height increasing, large blue-goal xy error during carry, and small blue-goal xy error near the handoff.

The intended exit state is a blue block held by the gripper near the blue goal in xy with enough height margin for the successor to descend. Exact placement height or release is not required for this policy. In the demonstrations, some trajectories at index 590 remain near 0.29 m while others have already started descending; the model is trained on this overlap so the final placing policy can receive either form of goal-near high carry.

## Limitations

This implementation does not include inverse kinematics, forward kinematics beyond observed TCP pose, force/contact reasoning, collision checking, image encoders, or hard safety constraints. Stage thresholds are used only to form features and training labels; they are not hard runtime rejection rules. The demonstrations do not establish recovery after a missed grasp or object slip, nor behavior with obstacles beyond the already placed red block.
