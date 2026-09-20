# PRIOR for `place_block__h03`

## Original research heuristic

The assigned heuristic is **tray packing noninterference** for `place_block`: placement should be conditioned on both object poses and the tray geometry, optimizing the active block placement while preserving the inactive block. This is important because task success is conjunctive: red must be in the red tray region and blue must be in the blue tray region at the same time. The expanded M1_v2 slice includes not only lowering and releasing the active block, but also shared transition phases: red placement and retreat toward blue in `[185,420)`, and blue placement and final retreat in `[580,stop)`.

Observed demonstrations support the hypothesis. In demo10300, red is transported from a grasped elevated state at index 185 toward the red target, is on the tray near index 300, remains at the red goal through the transition at 330 and 420, and is still stable while blue is transported, placed and retreated at 580, 625, 700 and 742. The blue block remains at its table pose during red placement and transition, then becomes the active block in the second segment.

## Executable adaptation

The repository provides state histories, fixed DDPM noising/sampling, action normalization and optimizer mechanics. It does not provide images, inverse kinematics, forward kinematics, a collision checker, or a hard constraint interface. Therefore I implemented the heuristic as a **learned action diffusion model** with a structured scene encoder and differentiable auxiliary learned predictions, not as a scripted controller.

Deployment `forward` receives only the causal two-observation history `[B,2,47]`, the noisy action sample and the DDPM timestep. Future observations are used only during training losses. The policy outputs epsilon for the normalized 16-step sequence of 8 native actions: seven Panda joint targets plus gripper command.

## Architecture

`policy.py` builds `TrayNoninterferenceDiffusion`:

- A `TraySceneEncoder` encodes the two causal observations. It uses the shared assignment normalizer for all 47 raw channels, preserving the M1_v2 requirement that all policies use the same normalization fitted on complete demonstrations.
- The encoder also appends analytic metric features scaled by broad task constants, not by a tiny per-skill refit: red-goal and blue-goal vectors, TCP-object vectors, object-object separation, cross-object-to-goal vectors, gripper opening/velocity, conservative signed distances to the tray object-center bounds, and signed distances to each 12 cm by 12 cm goal box.
- The encoded scene conditions the provided `DiffusionBackbone` U-Net. The policy remains an epsilon-predicting DDPM action model.
- A trainable auxiliary object-future head receives the scene condition and a differentiably estimated clean action sequence `x0` from the denoising equation. It predicts future red and blue xyz deltas for all horizon slots.
- A trainable role head predicts whether the current window is red-active, blue-active, or a no-active-block transition/retreat window.

The engineered features are representation priors only. They do not implement IK or hard collision avoidance.

## Loss terms and gradient paths

The total loss is `diffusion_loss + prior_loss`.

`diffusion_loss` is the standard masked epsilon MSE supplied by `appl.public.epsilon_loss`.

`prior_loss` is a weighted sum of differentiable losses whose operands include trainable predictions:

1. **Future object dynamics loss**: the auxiliary head predicts scaled future red/blue xyz deltas from the causal state condition and denoised action estimate. It is trained against `future_obs`. This gives the model a learned action-state coupling instead of a constant observation-only penalty.
2. **Role classification loss**: a pseudo-label derived from current gripper/object geometry and masked future object motion trains the role head to distinguish red placement, blue placement and transition/retreat phases.
3. **Inactive constancy loss**: according to the role label, the predicted inactive object delta is penalized toward zero; in no-active transition windows both predicted deltas are encouraged to remain zero. This is the direct implementation of noninterference.
4. **Active goal loss**: when future labels show the active object is within the goal funnel, the predicted active position is penalized toward the assigned goal xy and, when low, the goal z.
5. **Tray containment loss**: predicted active positions near the goal are penalized if outside conservative object-center tray bounds derived from the fixture dimensions and object half-size.
6. **Keep-out loss**: when the active object is predicted/labelled low near placement, its predicted xy path is penalized for approaching the inactive object's current xy closer than 5.5 cm.

The auxiliary head receives a differentiable `x0` estimate computed from the predicted epsilon, so these terms can send gradients through the auxiliary modules, encoder and denoising backbone. The clean-action estimate is bounded by a differentiable `tanh` before the auxiliary head to limit high-noise DDPM amplification.

## Causal deployment information and handoff behavior

The policy can take over partway through the expanded transition because the observation vector directly contains the necessary world-frame cues:

- TCP pose and object poses identify whether the gripper is near/over a held block, over a released block, or traveling toward the next block.
- Finger opening in `qpos[7:9]` identifies closed transport versus open release/retreat phases.
- Red and blue xyz poses show which object is already at its goal and which object remains on the table or is being carried.
- Fixed red and blue goal channels and analytic tray distances identify the target side of the tray and containment margins.

For the `[330,420)` transition after red placement, these inputs show red low and stable near the red goal, blue still on the table, gripper open, and the TCP retreating/toward blue. For the `[580,625)` overlap into blue placement, the same inputs show red already placed and blue held/elevated near the TCP. The model therefore learns a coherent behavior over the full assigned slice, including transition actions, from causal state rather than from a preset schedule.

## Dependencies and limitations

This policy depends only on Torch and `appl.public`. It assumes reliable state estimates for both block poses, TCP pose, gripper joints and the fixed goal channels. It does not use images or external geometry libraries. It does not enforce hard safety constraints during sampling; the noninterference/containment behavior is learned through conditioning and auxiliary training. The dataset has positive demonstrations but no examples of severe inactive-block displacement or near-wall recovery, so recovery claims would be untested extrapolation.
