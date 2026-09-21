# PRIOR.md: red_unstack_lift__h02

## Assigned research heuristic

This policy implements heuristic 2 for `red_unstack_lift`: an event-gated grasp-lift prior. The original hypothesis is that the red unstacking slice is not a single stationary behavior. It contains an ordered sequence of events: open approach to the stacked red block, finger closure/contact, vertical lift of the grasped red block, and the beginning of lateral transport toward the red goal. The prior therefore represents the behavior with a small ordered latent state machine whose modes separate the different action distributions and especially the sharp gripper timing.

The assigned expanded training slice is the full interval `[0, 190)` for all twelve demonstrations `demo10200` through `demo10211`. This includes the shared transition overlap `[130, 190)` with the red placing successor. The implementation keeps that whole range and does not refit any per-skill normalization statistics.

## Executable adaptation of the heuristic

The repository provides a state-based action diffusion interface, not a contact sensor or an external controller. I adapted the heuristic as a learned conditional DDPM with three trainable latent modes:

1. `approach_open`
2. `close_contact`
3. `lift_carry`

The modes are inferred from the two causal observations available at deployment. During training only, future observations are used to create pseudo-labels for phase and transition heads. These labels are supervision for trainable predictions; they are not available to `forward()` and are not a scripted action source.

The label rule is deliberately broad and semantic rather than a tiny per-skill fit: open approach is assigned while the fingers are wide and red remains stacked; close/contact is assigned when total finger width is below about `0.055 m` before lift; lift/carry is assigned when current red height or red-blue vertical separation indicates lift, or when a closed-finger state has a near-future red height above `0.135 m`. These thresholds use world metres and gripper aperture and are documented as training labels, not hard deployment gates.

## Architecture

`policy.py` defines `EventGatedGraspLiftPolicy`.

- Input to deployment `forward()` is only `raw_history [B, 2, 47]`, the noisy action sample, and the DDPM timestep.
- Raw observations are normalized with the shared assignment normalizer through `appl.public.normalize_observation`.
- A small set of causal engineered features is concatenated to the normalized state: finger aperture/asymmetry, finger velocities, TCP-red vector, red-blue vector, red/blue goal vectors, and one-step causal finite differences for TCP, red, blue, and finger width. These are observations in the same world frame; no IK or forward kinematics is used.
- The per-history-step features are embedded by an MLP and a GRU.
- The recurrent state predicts current phase logits, transition logits, soft phase probabilities, and a soft phase embedding.
- The diffusion condition vector is an MLP of the recurrent state, current phase distribution, learned mode embedding, predicted transition distribution, and an ordered scalar expectation over modes.
- The action denoiser is the supplied `DiffusionBackbone`, a conditional 1-D U-Net, and predicts epsilon for normalized eight-dimensional action horizons.
- A learned prototype head predicts an encoded action horizon as an auxiliary training target. It is used only as a differentiable regularizer on the latent representation; inference actions still come from DDPM sampling.

## Loss terms and gradient paths

`compute_loss` returns the required `loss`, `diffusion_loss`, and `prior_loss`.

- `diffusion_loss`: standard masked epsilon MSE between the U-Net output and sampled DDPM noise. This trains the learned diffusion action model.
- `phase_loss`: cross-entropy from trainable phase logits to pseudo phase labels inferred from current observation and training futures.
- `transition_loss`: cross-entropy from trainable transition logits to the pseudo phase of a future state near execution slot 7, encouraging the encoder to know whether it is still approaching, closing, or ready to continue lift/carry.
- `prototype_loss`: masked MSE from the trainable auxiliary action prototype to the normalized demonstrated action horizon, giving the phase-gated condition a direct action-distribution gradient.
- `order_loss`: differentiable penalty on predicted backward mode transitions, implementing the ordered state-machine prior without imposing a scripted controller.

All auxiliary terms depend on trainable network outputs. No prior loss is a constant computed only from observed poses.

## Causal deployment behavior

At deployment the model observes only the last two raw states. Useful causal cues are:

- `qpos[7:9]` finger openings and `qvel[7:9]` finger velocities,
- `tcp_pose[0:3]` relative to `red_pose[0:3]`,
- `red_pose[2]` and red-blue vertical separation,
- one-step history motion of TCP, red, blue, and finger width,
- fixed red and blue goal positions.

These cues allow the policy to take over from the initial state, an open pre-grasp state, a just-closed state, or the overlap where the red block is already lifted. If invoked partway through the transition, closed fingers plus elevated red and TCP/red co-motion push the learned phase toward `lift_carry`, so the denoiser is conditioned to continue lifting/carrying rather than replaying approach or closure.

## Applicability and handoff

The policy is intended for the assigned unstack-sort setting with red initially above blue. It is appropriate when contact/grasp progress is visible through finger aperture and red object motion. It should hand off to a placing policy after the red block is high above blue and lateral transport toward the red goal is underway. In the measured training support this ranges from approximately red `z = 0.13-0.16 m` at index 130 to red `z ~= 0.30 m` near index 190 with closed fingers around total width `0.0365 m`.

## Limitations

The model does not implement inverse kinematics, collision constraints, image perception, or a verified contact estimator. The phase labels infer grasp/contact from aperture and object pose; if the gripper closes on empty space, the network may still predict a lift phase unless the observation history shows lack of red motion. The learned policy is still a diffusion model trained from the provided demonstrations and may not recover from large off-distribution disturbances.
