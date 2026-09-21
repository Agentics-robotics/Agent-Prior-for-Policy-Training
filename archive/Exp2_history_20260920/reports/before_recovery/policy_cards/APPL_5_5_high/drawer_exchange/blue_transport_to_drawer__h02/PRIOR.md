# PRIOR: blue_transport_to_drawer__h02

## Assigned heuristic

The assigned hypothesis is the **rigid blue-carry prior**: once the blue block is grasped, the block should remain at an approximately stable transform relative to the robot TCP while the arm transports it toward the open drawer cavity. The policy is responsible for the full expanded slice `810:1030`, not only the central carry phase. This slice includes the shared closing/lift transition from the predecessor, the main high carry toward the drawer, the descent/insertion phase, the beginning of release, and retreat context shared with the successor.

The task-level completion rule remains the supplied three-goal contract: drawer open, red block on the marked outside pad, and blue block inside the drawer at the same observation. This policy does not redefine completion with extra release, velocity, clearance, or hold conditions.

## Executable adaptation of the research prior

The interface provides state history and actions but no analytic forward kinematics, inverse kinematics, contact solver, collision checker, or image observations. Therefore I implemented the heuristic as a learned diffusion action model with a differentiable learned dynamics prior rather than as a scripted Cartesian controller.

The adaptation is:

1. A standard conditional diffusion model predicts normalized Panda joint/gripper actions.
2. The conditioning encoder uses the shared normalized observation history and causal geometric features: TCP pose, blue pose, TCP-blue offset, drawer position, blue goal, red goal, finger widths, and one-step TCP/object finite differences.
3. During training only, an auxiliary dynamics head receives the model's denoised clean-action estimate and predicts future TCP and blue-block positions.
4. The blue-block prediction has a structured rigid branch `predicted_tcp_position + current_blue_minus_tcp_offset`, blended by a learned attachment gate with a learned free-object branch for pre-grasp and release portions.
5. Future observations are used only as supervised training labels for this auxiliary head and for the prior loss. Deployment `forward()` receives only the two causal observations required by the API.

This keeps the policy a learned action diffusion model while making the rigid carry relation a trainable representation and loss term.

## Architecture

`build_model` constructs `RigidCarryDiffusionPolicy`.

- **Condition encoder:** an MLP maps 148 causal features to a 256-dimensional condition vector. Features include the two-step shared-normalized observation, world TCP/blue/red/goal deltas, drawer-center delta, finger widths, distances, and the current TCP-local blue offset computed from the observed TCP quaternion. All engineered metric features use broad fixed metre scales; no per-skill or tiny expanded-slice normalizer is refit.
- **Diffusion backbone:** `appl.public.DiffusionBackbone` with the fixed training recipe predicts epsilon for 16 normalized 8-D actions.
- **Auxiliary dynamics prior:** a GRU conditioned on the encoded state and on the predicted clean action sequence predicts future TCP positions, a free blue-block trajectory, and an attachment logit. The rigid branch keeps the observed current world offset between blue and TCP. The final blue prediction is a learned gate between rigid and free branches.

## Losses and gradient paths

`compute_loss` returns:

- `diffusion_loss`: the standard masked epsilon-prediction loss from `appl.public.epsilon_loss`.
- `prior_loss`: a weighted auxiliary loss containing:
  - future blue position prediction loss,
  - future TCP position prediction loss,
  - binary attachment-gate supervision from future finger width and blue height labels,
  - rigid offset consistency when the current and future labels indicate carried-object motion.

The prior is not a constant pose penalty. It depends on the predicted epsilon through the DDPM clean estimate
`x0 = (noisy_action - sqrt(1-alpha_bar) * predicted_epsilon) / sqrt(alpha_bar)`.
This clean estimate is smoothly bounded before the auxiliary head to reduce high-noise amplification while preserving gradients into the epsilon predictor. A teacher dynamics term using the ground-truth encoded action sequence also trains the auxiliary dynamics representation, but the model-dependent branch is included in the optimized prior loss so the diffusion network receives action-prior gradients.

## Causal deployment inputs

At inference, the model sees only `raw_history [B, 2, 47]` and the DDPM noisy action sample/timestep. It uses:

- qpos/qvel,
- TCP pose in world coordinates,
- red and blue poses in world coordinates,
- drawer position and velocity,
- red and blue goal positions.

No future state, future action label, image, IK output, hand-authored action, or replayed demonstration action is used by `forward()`.

## Expanded-slice behavior and handoff

The model is trained on the complete `810:1030` expanded slice. The initial portion teaches closing/lift agreement with the predecessor; the central portion teaches high rigid carry; the final portion teaches descent, release onset, and retreat agreement with the successor. The observation information that lets this model take over mid-transition is the causal TCP-blue offset, finger width, blue height, and one-step motion. If it starts after the blue is already high and co-moving with the TCP, these cues allow the conditioning encoder and attachment gate to represent the carry mode. If it starts earlier, open fingers and table-height blue observations allow the same policy to represent approach/closure/lift within the overlap.

Useful exit state for the successor is a blue block near the drawer target with the TCP close to the block, or the fingers opening and the TCP beginning retreat after insertion. The policy documentation describes these as invocation/termination cues for the inference API, not as hard task success conditions.

## Dependencies and limitations

The policy depends only on `torch`, `numpy`, `math`, and `appl.public`. It uses the shared M1_v2 normalizer from the assignment. It does not implement analytic IK, contact recovery, collision avoidance, rotational containment checking, or an invariant action representation. The rigid prior assumes limited slip and limited in-hand reorientation; if the blue block lags the TCP, rotates unexpectedly, or collides with the drawer wall, the learned model may continue to predict an attached carry mode incorrectly.
