# PRIOR: blue_to_goal_handoff__h01

## Assigned hypothesis

The assigned heuristic is an active-object reflection prior for the middle phase of the buffer-swap task.  The segment starts at index 320 and ends at index 860 for every training demonstration.  It includes both shared transition phases: the entry overlap from the previous red-to-buffer skill, the blue pick/carry/place phase, and the exit overlap toward the red-buffer-to-goal skill.

The hypothesis is that the blue transfer can be learned more sample-efficiently when represented in a frame tied to the active object, the active goal, and an approximate reflection of the repeated pick-carry-place schema.  In the demonstrated slice, red is already buffered near the central free region, blue begins near the upper/red-goal region, blue_goal is the lower marked region, and the robot must place blue at blue_goal before moving back toward red.

## Executable adaptation

The repository exposes state observations and absolute Panda joint/gripper actions, but it does not expose inverse kinematics, forward kinematics, a robot mirror map, images, or a controller interface.  Therefore the prior is implemented as a learned conditioning and auxiliary-loss structure inside a standard action diffusion model, not as scripted motion.

The model receives only the two causal raw observations supplied at deployment.  Future observations are used only in `compute_loss` as training labels for an auxiliary prediction head.  At inference, `forward(noisy_action, timestep, raw_history)` predicts DDPM epsilon for the normalized action sequence from causal history only.

## Architecture

`policy.py` builds `ReflectionBlueTransferPolicy`:

1. A raw-state branch normalizes the two 47-dimensional observations with the shared M1_v2 full-demonstration normalizer and encodes the flattened 94-vector.
2. A shared relational branch builds active-object features for each causal observation.  The active object is blue and the passive object is red.  Features include TCP-to-blue, blue_goal-to-blue, red-to-blue, red_goal-to-blue_goal, TCP-to-red, red_goal-to-red, broad metric-normalized absolute positions, top-down quaternions, finger positions/velocities, and relevant distances.
3. The same relational branch is applied to a y-reflected, red/blue-swapped version of the spatial observation.  This gives architectural weight sharing for the reflected manipulation schema.  Joint positions and velocities are not transformed because no valid kinematic mirror map is available; absolute joint actions remain learned targets.
4. A phase branch encodes gripper opening/closing, heights, distances to blue and red, goal offsets, and recent causal motion of TCP/objects.  This is what lets the same policy take over at index-320-like high/open entry states, index-460-like low/closed blue contact states, index-650/700 blue release states, or index-860-like red reacquisition states.
5. The branches are fused into a 256-dimensional condition vector and passed to the provided `DiffusionBackbone`, which predicts epsilon for horizon-16 action diffusion over the 8 native action dimensions after the framework's action normalization.

All observation/action normalization for the diffusion task uses the single assigned shared normalizer.  The additional relative features use broad metric scales, not per-skill refits.

## Training losses and gradient paths

The returned loss is:

- `diffusion_loss`: masked epsilon-prediction MSE from the provided DDPM batch.  This is the primary learned action diffusion objective and trains the encoder, fusion network, and U-Net.
- `prior_loss`: `0.03 * supervised_future_task_space_MSE + 0.01 * reflection_consistency_MSE`.

The auxiliary head predicts, for each of the 16 future slots, normalized TCP xyz, blue xyz, red xyz, and finger width from the same fused causal condition used by the diffusion model.  The supervised term compares these trainable predictions to `future_obs` under `future_mask`.  The consistency term runs the same predictor on the reflected/swapped causal observation and penalizes disagreement with the reflected/swapped version of the original prediction.  These penalties are not constants computed only from observed poses; they depend on trainable predictions and backpropagate through the shared encoder and auxiliary head.

The auxiliary head is not used as a controller and does not replace the diffusion action model.  It exists to shape the learned representation toward task-space prediction and approximate y-reflection consistency.

## Causal deployment inputs

The policy uses the following fields from the two most recent observations:

- `qpos`, `qvel`, especially the finger joints and velocities;
- `tcp_pose` position and quaternion;
- `red_pose` and `blue_pose` positions/quaternions;
- fixed `red_goal` and `blue_goal` positions;
- drawer compatibility channels are present but not semantically used.

No future observations, success predicates, schedule index, images, IK, or scripted object targets are used in `forward`.

## Applicability and handoff behavior

The intended entry is after red has been released into the buffer, with blue still near the upper/red-goal region and the gripper open, or later within the same transition when the TCP is already descending/closing over blue.  The expanded slice also trains the model after blue placement: the gripper opens above blue, retreats, then approaches and lifts the buffered red block to make the successor policy ready.

The model learns the full slice [320, 860), so a policy selector may invoke it during either overlap.  Causal cues that indicate phase include blue-to-blue_goal distance, TCP-to-blue distance, blue height, finger width, red-to-buffer/goal relation, TCP-to-red distance, and recent observed motion.  The useful exit state for a successor is blue stable at blue_goal, red still in the buffer or being approached/lifted in the demonstrated corridor, and the gripper/TCP in a state consistent with red reacquisition.

## Limitations

The reflection prior is approximate.  It does not guarantee physical equivariance of Panda joints, object contact, grasp closure, or placement containment.  Quaternion reflection is not treated as a rigorous SE(3) operation; the demonstrations are near top-down and the primary prior is on relative positions and phase.  Recovery from missed blue grasp, red leaving the buffer, or placing blue outside the marked region is not separately demonstrated.  The model remains a learned stochastic diffusion policy selected and timed by the external inference API.
