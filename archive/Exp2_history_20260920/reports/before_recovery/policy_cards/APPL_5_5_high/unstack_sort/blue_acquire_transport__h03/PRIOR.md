# PRIOR: blue_acquire_transport__h03

## Assigned heuristic

The assigned heuristic is **Completed-object invariance** for the `blue_acquire_transport` skill.  The task is `unstack_sort`: red must finish on the red pad at `[-0.24, -0.25, 0.02]` m and blue must finish on the blue pad at `[-0.24, 0.25, 0.02]` m.  This skill covers the expanded demonstrated slice from index 330 to 590 in each training trajectory.  At the start of this slice red has already been placed at its goal and the gripper is open above/near the red pad; during the slice the robot moves to the blue block, closes on it, lifts it, and transports it near or above the blue goal.  The red block remains static throughout the demonstrated transition overlaps.

The research hypothesis is that once red is a completed object, the blue policy should treat red_pose as a success condition that must be preserved, rather than ignoring red while manipulating blue.

## Executable adaptation

The provided policy interface exposes only state histories and an action diffusion model.  It does not provide images, inverse kinematics, forward kinematics, collision checking, or sampler-time candidate rejection hooks.  Therefore I implement the heuristic as a learned diffusion prior rather than a scripted path filter:

1. The denoiser is a standard learned epsilon-predicting action diffusion model.
2. The causal observation encoder includes full shared-normalized observations plus world-frame relative features involving `red_pose`, `red_goal`, `blue_pose`, `blue_goal`, and `tcp_pose`.
3. A separate invariant encoder branch is gated by a smooth red-at-goal cue.  When red is already on its target pad, this branch injects red-goal, TCP-red and blue-red relations into the diffusion condition.
4. A trainable action-conditioned auxiliary predictor maps a candidate 16-step action horizon and the causal observation history to predicted future red displacement, TCP displacement, blue-to-goal vector, and red-at-goal logits.
5. The auxiliary predictor is supervised with future observations from training.  The prior then applies differentiable penalties to the predictor outputs produced from the denoiser's current clean-action estimate.  This gives a gradient path from the completed-object invariance objective through the auxiliary predictor to the action denoiser.

This adaptation preserves the original heuristic identity while fitting the available API.  It is not a hard safety constraint and it is not an external controller.

## Model architecture

`build_model(spec)` returns `CompletedObjectInvariantDiffusion`.

Inputs at training and deployment are causal only:

- `raw_history [B, 2, 47]`, with the fixed M1_v2 shared normalizer fitted on the complete original demonstrations.
- `noisy_action [B, 16, 8]` and the diffusion timestep.

The observation features are:

- the two full observations normalized by the shared normalizer;
- relative world-frame vectors for red-goal, blue-goal, TCP-red, TCP-blue, blue-red and TCP-blue_goal at the previous and current observations;
- current-minus-previous TCP, red and blue xyz deltas;
- distances, gripper opening/balance, and smooth red/blue at-goal cues.

Relative xyz features use a fixed workspace scale `[0.50, 0.50, 0.30]` m.  This is intentionally not a per-skill refit and does not replace the shared observation/action normalizer.

The condition MLP outputs a 256-dimensional global condition for `appl.public.DiffusionBackbone`.  The auxiliary head uses the same condition plus an encoded 16-step candidate action sequence and predicts, for every future step:

- `(future red xyz - current red xyz) / workspace_scale`,
- `(future tcp xyz - current tcp xyz) / workspace_scale`,
- `(future blue xyz - blue_goal xyz) / workspace_scale`,
- a logit that future red remains at its red goal.

## Losses and gradient paths

`compute_loss` returns `loss`, `diffusion_loss`, and `prior_loss`.

The diffusion term is the standard masked epsilon MSE against the DDPM noise target.

The prior term contains:

- supervised auxiliary prediction losses for red displacement, TCP displacement, blue-to-goal vector and red-at-goal logits, using `future_obs` labels;
- a predicted red-static penalty evaluated on the denoiser clean-action estimate;
- a predicted red-at-goal BCE penalty evaluated on the denoiser clean-action estimate;
- soft penalties for predicted low TCP or blue paths inside an XY margin around the current red pose.

The action-dependent penalties are computed from trainable predictions, not directly from constant observed poses.  Thus they can change the denoiser through the path:

`predicted epsilon -> clean action estimate -> auxiliary predictor -> prior penalty`.

The fixed observed red pose and goals are used only as conditioning and label/reference information.

## Causal deployment behavior

At deployment the model receives only the last two observations.  It can take over partway through the transition because the state contains enough causal information to disambiguate the phase:

- `qpos[7:9]` indicates gripper opening/closing;
- `tcp_pose` relative to `blue_pose` indicates approach, grasp or transport;
- `blue_pose` height and its distance to `blue_goal` indicate whether blue is still at the source, lifted, or near the goal;
- `red_pose` relative to `red_goal` indicates whether the completed-object assumption is valid;
- the previous-to-current object and TCP deltas indicate whether red is static and whether blue is moving with the gripper.

The learned action diffusion model covers the entire expanded slice, including the shared transition phases from red placement into blue approach and from blue transport into final placement readiness.

## Applicability, handoff and limitations

This policy is intended for states where red is already placed and should remain undisturbed while the robot handles blue.  It should terminate or transfer when blue is held near/above the blue target and red remains at its goal.

Limitations:

- The training data shows red remaining static after placement; it does not show recovery from a displaced red block.
- The auxiliary red disturbance model is therefore trained mostly on no-disturbance labels and should be viewed as a bias, not proof of causal red protection.
- No inverse kinematics, image encoder, geometric collision checker, hard constraint, or scripted action controller is implemented.
- The outer sampler is the fixed DDPM sampler; this implementation does not perform explicit K-sample ranking at inference.
