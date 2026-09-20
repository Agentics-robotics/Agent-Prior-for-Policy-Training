# acquire_active_block__h03 PRIOR

## Original research heuristic

This policy implements heuristic 3 for `acquire_active_block`: successful acquisition should be treated as a verifiable gripper-object coupling, not merely as executing a close command.  The handoff evidence shows red acquisition around observations 120-220 and blue acquisition around 500-600.  At the early entries the TCP is near the block and the fingers are closing; later observations show the object lifted with the TCP and the finger width near the demonstrated closed value.  The prior therefore asks the learner to predict an attachment state from recent TCP, object, and finger motion, and to continue closure/lift until that attachment state is likely.

## Executable adaptation

The available interface provides two causal state observations, future state labels during training, absolute joint/gripper action targets, and a fixed DDPM action sampler.  No force sensor, images, forward kinematics, IK, or hard contact constraint is available.  I adapted the hypothesis as follows:

1. The main policy remains a learned diffusion model over the 16-step sequence of 8-dimensional normalized native actions.
2. A causal observation encoder receives the two raw state observations after the shared full-demonstration normalizer, plus broad-scale engineered metric features: TCP/object relative positions, last-step TCP/object relative motion, finger width and finger velocity, distances to goals, object heights, and co-motion errors.
3. The encoder predicts two learned latent quantities: which block is active (`red` or `blue`) and per-block attachment logits.
4. The U-Net denoiser is conditioned on the encoder hidden state together with the learned active-block and attachment predictions.  Thus the action diffusion model can represent different actions for open/closing, uncertain-contact, and attached carry states.
5. During training only, future observations define soft attachment targets.  A block is labelled attached when the current fingers are closed, the object is near the TCP, future relative TCP-object offset is stable, and the object is currently lifted or is lifted with the TCP over the future training horizon.  These labels are not available at deployment.
6. A learned action-conditioned future relative-offset head predicts future red/blue TCP-object offsets from the denoiser's clean-action estimate and the causal condition.  This is the differentiable version of the proposed coupling loss; it does not use analytic kinematics.  The loss is weighted on attached-labelled windows and sends a small, clipped gradient through the clean-action estimate to avoid high-noise DDPM amplification.

## Architecture

`policy.py` defines `AttachmentAwareAcquirePolicy`.

- Encoder: MLP with LayerNorm and Mish activations.  Input is the flattened two-step normalized observation history plus engineered causal relative features in world units.
- Heads: a 2-way active-block classifier and a 2-logit attachment classifier.
- Conditioning: the encoder hidden vector, active logits/probabilities, attachment logits/probabilities, and active attachment probability are projected to a 256-dimensional global condition.
- Denoiser: the provided `appl.public.DiffusionBackbone`, a conditional 1-D U-Net, predicts DDPM epsilon for normalized actions.
- Auxiliary future head: an MLP embeds the denoised clean action estimate and predicts a `[horizon, 2 objects, 3 xyz]` trajectory of future TCP-object relative offsets.

The model uses the assignment's shared normalizer.  The engineered features use fixed broad metric scales such as 0.5 m for positions and 0.05 m for small motions; it does not refit per-skill statistics.

## Losses and gradient paths

The total loss is:

`L = L_diffusion + L_prior`

where `L_diffusion` is the standard masked epsilon-prediction loss supplied by `appl.public.epsilon_loss`.

`L_prior` is a weighted sum of:

- Active-block cross entropy.  The label is red until red is placed on its table-height goal pad, then blue.
- Per-block and active-block attachment BCE.  Soft labels are computed from future training observations as described above.  The prediction is made from causal history only.
- Future relative-offset MSE.  The target is the future observed object-minus-TCP position for the active object.  The prediction is trainable and action-conditioned, so this term is not a constant pose penalty.  It is weighted by attachment confidence labels, future masks, and a bounded DDPM alpha weight.

The auxiliary future-offset loss gradients flow into the future head and weakly into the denoiser through the clean-action estimate.  The active and attachment heads also affect the denoiser because their logits/probabilities are part of the global condition.

## Causal deployment inputs

Deployment `forward(noisy_action, timestep, raw_history)` receives only the last two raw observations.  It can use:

- qpos/qvel, including finger width and finger velocity;
- TCP pose in world coordinates;
- red and blue object poses in world coordinates;
- fixed red and blue goal positions;
- last-step changes between the two observations.

It does not read future observations, segment indices, trajectory IDs, or a scripted schedule.

## Full expanded slice and transition behavior

The policy covers the full assigned expanded slices: approach to red, red close/lift and the shared acquisition-delivery overlap, retreat/approach to blue, and blue close/lift with its overlap.  The active-block classifier lets the same network take over from either the beginning of acquisition or from a transition state: red remains active until placed on its pad at table height, after which blue becomes active.  In the overlap, the attachment predictor gives a causal cue for whether to keep lifting/settling or hand off to delivery.

If invoked partway through the transition, the observation history still contains the essential handoff information: TCP-object offset, whether the fingers are still open or at the demonstrated closed width, whether the object has started moving with the TCP, and whether the red goal has already been reached so that blue is the next active block.

## Applicability and limitations

This prior is appropriate for the provided two-block sorting task with visible object poses and Panda finger state.  It assumes contact mechanics like the demonstrations: a 4 cm block, closing finger width around 0.018 m per finger when grasped, and sufficient friction for a parallel-jaw lift.

Limitations:

- Failed grasps are not demonstrated; the model learns negative attachment mostly from pre-contact/open-finger windows.
- Attachment is a learned probabilistic state, not a verified physical constraint.
- The model has no IK, no analytic forward kinematics, no force/torque sensing, no image encoder, and no hard invariant controller.
- The future-offset auxiliary head predicts offsets from demonstrated actions; it is not a simulator and should not be interpreted as a certified contact model.
