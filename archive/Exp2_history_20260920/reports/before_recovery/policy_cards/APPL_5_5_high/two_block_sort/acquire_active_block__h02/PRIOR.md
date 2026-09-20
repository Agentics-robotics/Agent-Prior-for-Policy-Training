# Prior for `acquire_active_block__h02`

## Original research heuristic

The assigned heuristic is the **monotone acquisition phase prior** for the `acquire_active_block` skill in `two_block_sort`.  It proposes that acquisition is not a single homogeneous control mode: demonstrations move from open free-space approach, to aligned descent, to low closure around the active block, to attachment/lift, and finally to an early carry state that can be handed to delivery.  The intended benefit is to reduce multimodality in the diffusion action distribution and to support takeover at several overlap states rather than only at a single fixed endpoint.

The full expanded M1_v2 slice is implemented: red acquisition over the first segment (`0:220`) and blue acquisition after red placement over the second segment (`340:600`) for every training demonstration.  The shared transition phases are included, especially close/lift/carry overlap with `deliver_active_block` (`120:220` and `500:600`) and the red-delivery-to-blue-acquisition transition (`340:440`).

## Executable implementation

The policy remains a learned epsilon-predicting DDPM action policy.  It does not output scripted Cartesian targets and it does not implement IK.  The trainable model has three parts:

1. **Causal phase/condition encoder**
   - Input: the two causal raw observations `[B, 2, 47]` available at deployment.
   - It uses the provided shared observation normalizer, then appends broad physical relative features: TCP-to-red, TCP-to-blue, TCP-to-soft-active-block, active-block-to-goal, object heights, finger width, finger velocity, and a soft feature indicating that blue is the active block after red is on the red goal at table height.
   - A per-step MLP and one-layer GRU produce a hidden state.
   - A trainable phase head predicts a posterior over five ordered acquisition phases: `approach_above`, `descend_align`, `close_contact`, `lift_attach`, and `early_carry`.

2. **Phase-conditioned denoiser**
   - The phase probabilities, a learned probability-weighted phase embedding, and compact motion cues are passed through a condition MLP.
   - A standard `appl.public.DiffusionBackbone` U-Net denoiser receives this condition and predicts action noise for normalized 8D Panda joint/gripper action sequences.
   - Thus the phase prior affects the learned score model at every DDPM training and sampling step.

3. **Auxiliary phase losses**
   - The main loss is the standard masked epsilon loss on the framework-provided noised action batch.
   - A current-state phase cross-entropy trains the phase head from state-derived labels.
   - A future-state phase cross-entropy applies the same causal two-frame encoder to valid future training histories.  These future observations are labels only; they are never used by deployment forward.
   - A monotonic regularizer penalizes decreases in the expected ordered phase across valid future slots.

The returned `prior_loss` is the weighted sum of these auxiliary terms.  It is differentiable with respect to trainable phase encoder parameters.  It is not a constant computed only from observed poses.

## Phase labels

Labels are generated from observable state variables, not from future success annotations:

- The active block is red unless red is close to the red goal in xy and z, in which case blue is treated as active.
- `approach_above`: default while far from the active block or in high free space.
- `descend_align`: TCP near the active block in xy, active object still on the table, TCP above it.
- `close_contact`: low and aligned while fingers are closed or closing.
- `lift_attach`: active object height above the table.
- `early_carry`: active object high enough to be in the demonstrated carry/overlap regime.

These rules are deliberately broad physical rules in world metres and finger widths.  They are not hard execution gates; during deployment the learned posterior is used through the denoiser condition.

## Causal deployment information

Deployment `forward` receives only `raw_history [B,2,47]`, the current DDPM noisy action sample and diffusion timestep.  It uses:

- `qpos` and `qvel`, including finger qpos/qvel,
- `tcp_pose` position,
- `red_pose` and `blue_pose` positions,
- fixed red and blue goal positions,
- the shared observation/action normalizer supplied in the assignment.

No images, future observations, demonstration indices, external contact flags, forward kinematics or IK are used.

## Handoff adaptation

The handoff hypothesis is executable because the phase posterior can be inferred from causal observations: distance from TCP to active object, TCP height over object, finger width/velocity, and object height.  At the start of the second acquisition segment, the observation shows red down at the red goal and the gripper open while TCP retreats/crosses toward the blue block, so the soft active-block feature and normalized object states allow the policy to take over mid-transition rather than assuming a home start.

Delivery readiness is represented by high probability on `lift_attach` or `early_carry`; however the low-level model does not expose a separate API value, so the inference agent should recompute the same cues from observations or use its own policy selection logic.  The denoiser has been trained on the overlap states so it can either continue lifting/carrying or be replaced by delivery without requiring a singular endpoint.

## Limitations

- The auxiliary labels are heuristic and may be noisy near actual contact.
- Finger closure plus object lift is only an inferred grasp/contact cue; no force sensor or contact truth is available.
- The model has no hard safety constraints and no scripted recovery if it misses the block, closes early, or lifts without attachment.
- The relative features are representation aids only; they do not provide exact spatial invariance or an IK controller.
- All normalization is the assigned shared complete-demonstration normalizer.  No per-skill scale fitting is performed.
