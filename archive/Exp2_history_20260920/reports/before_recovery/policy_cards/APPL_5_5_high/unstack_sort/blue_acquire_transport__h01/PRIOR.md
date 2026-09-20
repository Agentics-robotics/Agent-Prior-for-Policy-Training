# PRIOR: blue_acquire_transport__h01

## Original heuristic identity

This policy implements heuristic 1 for `blue_acquire_transport`: **identity-conditioned top-grasp reuse**. The assigned hypothesis is that the blue pickup and lift should reuse the same geometric top-grasp structure as the earlier red pickup, while selecting the blue block as the active target and treating red as a finished/context object. The policy covers the full expanded training slice `[330, 590)` for each demonstration, including both transition overlaps: object switching and blue grasp acquisition in `[330, 520)`, then high carry toward the blue goal in `[520, 590)`.

## Executable adaptation

The repository supplies state histories, joint actions, the DDPM noising/sampling procedure, optimization, and the shared normalizer. No inverse kinematics, forward kinematics, camera encoder, hard collision checker, or scripted controller is available. Therefore the prior is implemented as a learned conditional diffusion model over the provided 8-D native action sequence: seven absolute Panda joint targets plus the gripper command.

The heuristic is adapted into the following trainable components:

1. A shared block object encoder processes either block from the same feature template: object pose, goal, TCP-object offset, goal-object offset, short-horizon object displacement, and finger state.
2. Learned role and identity embeddings select `blue` as the target token and `red` as the context/finished token. This is the executable form of identity-conditioned grasp reuse.
3. A condition encoder combines the shared normalizer output with world-frame relative features, phase cues, and the two object tokens.
4. A standard learned conditional U-Net diffusion backbone predicts epsilon for action denoising.
5. Auxiliary learned heads predict future blue lift/goal approach/contact-related labels from the same condition embedding. These predictions are trainable and are compared to future training observations only during training.

All observations and actions use the single shared M1_v2 normalizer fitted on the complete original demonstrations. The model does not fit or assume narrow per-skill normalization ranges.

## Architecture

`build_model(spec)` returns `IdentityConditionedTopGraspPolicy`.

Causal inputs to `forward` are only:

- `raw_history [B, 2, 47]`, with fields qpos, qvel, TCP pose, red pose, blue pose, zero drawer compatibility channels, red goal, and blue goal;
- `noisy_action [B, 16, 8]`;
- DDPM timestep.

The condition pathway includes:

- flattened normalized two-step observation history using the shared normalizer;
- normalized observation finite difference between the two causal states;
- object-relative features such as `tcp - blue`, `blue_goal - blue`, `blue - red`, `red_goal - red`, `tcp - blue_goal`, and short pose deltas;
- finger width/open-closed summaries;
- smooth phase cues for TCP-blue XY centering, blue lift, red-at-goal support, blue-near-goal support, and closed/high-carry state;
- a shared block encoder plus learned target/context and blue/red embeddings.

The resulting 256-D condition vector modulates `appl.public.DiffusionBackbone`, which predicts DDPM epsilon for the normalized action sequence. The policy remains a learned diffusion model; it never directly outputs scripted Cartesian moves.

## Losses and gradient paths

`compute_loss` returns:

- `diffusion_loss`: standard masked epsilon MSE between the backbone prediction and DDPM noise;
- `prior_loss`: a weighted auxiliary loss from trainable predictions;
- `loss = diffusion_loss + prior_loss`.

The auxiliary targets are computed from training labels in `future_obs` and masked by `future_mask`:

- future blue height;
- future blue XY error to the blue goal;
- future blue lift delta from the current state;
- future TCP-blue offset;
- a lifted-and-closed binary cue;
- a gripper-closed binary cue.

The auxiliary heads share the condition encoder with the diffusion backbone, so their gradients train the object-relative representation used for denoising. No auxiliary term is a constant penalty on observed poses alone.

## Causal deployment information and handoff support

At deployment the model receives only the two most recent state observations. It can take over partway through the transition because the current blue pose, TCP pose, qpos finger width, qvel, and short pose deltas identify whether the system is still moving from red release, centered above blue, closing on blue, lifting, or carrying toward the blue goal. Red pose relative to red_goal tells the model whether red is already in its finished context location. Blue pose relative to blue_goal and TCP-blue offset identify when the motion has entered the high-carry overlap and is suitable for transfer to final placement.

The policy was designed for the observed support where index 330 is typically post-red-release with fingers open and red at the red goal; indices near 460 are low/contact or gripper-closing around blue; indices near 520 show blue lifted high with closed fingers; and index 590 shows blue near or above the blue goal, ready for final descent or already beginning descent in some demonstrations.

## Applicability, termination, and limitations

Applicable when red is placed, blue is upright and reachable from above, and the task remains within the demonstrated top-down geometry. Continue while the blue block is not yet securely lifted and transported near the goal. Transfer when blue is held high with closed fingers and is close to the blue goal, giving a successor placement policy a useful state for descent and release.

Limitations:

- no explicit IK or Cartesian controller;
- no hard guarantee of gripper contact, collision avoidance, or red immobility;
- no visual perception or invariance beyond the provided state features;
- limited recovery data after missed grasps, disturbed red placement, or tipped blue;
- object identity is learned from the available demonstrations and may be partially confounded with time because the dataset does not independently randomize color roles.
