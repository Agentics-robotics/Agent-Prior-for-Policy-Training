# Prior document: red_unstack_lift__h01

## Assigned heuristic

This policy implements heuristic 1 for `red_unstack_lift`: a red-relative top-grasp prior.  The source hypothesis is that, while the absolute Panda joint path changes with the small initial stack translation, the useful local structure is mostly invariant when expressed around the red cube: the TCP descends to the red top while the gripper is open, closes on the red cube, then the TCP and red cube rise together while the blue cube remains on the table.  The expanded M1_v2 slice is the full interval `[0, 190)` for each demonstration, including the shared transition `[130, 190)` in which the red cube is already lifted and begins moving laterally toward the red goal.

## Executable adaptation

The repository action space is not Cartesian control.  Actions are absolute Panda joint targets plus one gripper command, and no analytic IK or forward-kinematics API is available.  Therefore the implemented prior does not command a scripted top grasp.  It is a learned epsilon-predicting DDPM over the provided normalized 8-D action horizon.  The prior is implemented as the representation and training objective used by that diffusion model:

1. The model receives only the two causal state observations supplied to `forward(raw_history)`.  It normalizes the raw 47-D observations with the shared task normalizer supplied by the framework.
2. It augments the normalized raw history with metric red-relative features: `tcp - red`, `tcp - blue`, `red - blue`, `red_goal - red`, `blue_goal - blue`, and `red_goal - tcp`, computed for both history observations.  It also includes causal deltas of TCP, red, blue, and qpos, plus scalar cues for finger opening, TCP height above red, red clearance above blue, TCP/red xy distance, red/goal xy distance, and recent blue xy motion.
3. These features are encoded by trainable MLP layers and a small unsupervised phase-gating head.  The resulting 256-D condition vector drives the standard conditional 1-D U-Net diffusion backbone from `appl.public`.
4. The output of `forward` is still only the predicted DDPM noise for the action sample.

The relative feature scales are fixed metric scales such as 0.25 m for local object/TCP relations and 0.40 m for goal xy relations.  They are not per-skill fitted normalizers.  The raw observation and action normalization remains the single shared normalizer fit on all complete original demonstrations.

## Architecture

`build_model(spec)` constructs `RedRelativeLiftDiffusion`:

- shared-normalized raw observation history, flattened to 94 values;
- red-relative and goal-relative causal features, giving a 154-D hand-designed input;
- trainable phase head producing a 5-way soft phase code from the same causal input;
- trainable condition encoder producing a 256-D global condition;
- `appl.public.DiffusionBackbone(condition_dimension=256, spec['training'])`, i.e. the fixed conditional U-Net DDPM backbone with the assigned horizon and action dimension;
- auxiliary future-descriptor MLP used only during training.

There is no image encoder, analytic IK, collision checker, invariant action transform, or hard constraint module.

## Losses and gradient paths

The main loss is the standard masked DDPM epsilon MSE:

`diffusion_loss = epsilon_loss(predicted_noise, sampled_noise, mask)`.

The prior loss is differentiable and depends on trainable predictions.  During training, the predicted noise is converted to a denoised action estimate

`x0 = (noisy_action - sqrt(1-alpha_bar) * predicted_noise) / sqrt(alpha_bar)`.

A learned auxiliary head takes the condition embedding and a bounded version of this predicted clean action horizon.  It predicts future state descriptors: `tcp - red`, `red - blue`, `red_goal - red`, blue displacement relative to the current blue pose, and total finger opening.  These descriptors are supervised from `future_obs` with `future_mask`.  The prior loss is the masked MSE between the learned predictions and these future descriptors.  Gradients flow through the auxiliary head into the condition encoder and into the diffusion prediction through `x0`; it is not a constant penalty computed only from observed poses.  The total training loss is

`loss = diffusion_loss + 0.05 * prior_loss`.

The auxiliary head is not used to control the robot at deployment.  Its purpose is to bias the learned denoising representation toward action sequences that explain the demonstrated causal lift and transition: maintaining TCP/red alignment, increasing red/blue vertical separation, limiting blue drift, closing the gripper, and beginning red-goal-directed transport.

## Causal deployment inputs

At inference, `forward` uses only `raw_history [B, 2, 47]`, the noisy action sample, and the DDPM timestep.  The relevant causal fields are qpos/qvel, tcp_pose, red_pose, blue_pose, red_goal, and blue_goal in world metres/quaternion coordinates.  Future observations appear only as labels inside `compute_loss` during training.

## Applicability and handoff

The policy is intended for the initial unstacking phase with red stacked on blue and for the overlap transition in which the successor may take over before a fixed final carry pose.  Observations that support invocation include open fingers around 0.04 m each, the red pose about 0.04 m above the blue pose in z, and the TCP above the workspace.  Observations that support transfer to the red-placement successor include closed fingers around the demonstrated pinch aperture, red_pose lifted well above blue_pose, blue_pose remaining near table height, and the TCP/red pair moving laterally toward the red goal.  Because the model conditions on instantaneous TCP/red/blue/goal relations and recent motion rather than on an absolute time index, it can take over within the demonstrated transition when the red cube is already partly lifted.

## Limitations

The demonstrations only cover successful centered top grasps over a small initial stack-xy range.  This implementation does not prove grasp contact and does not recover deliberately from a missed grasp, side contact, tilted block, or significant blue displacement.  It does not redefine the task success predicates; task completion is still evaluated only by the supplied completion contract.
