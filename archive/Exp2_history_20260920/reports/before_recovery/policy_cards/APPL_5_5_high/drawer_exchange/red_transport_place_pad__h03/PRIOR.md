# Prior for `red_transport_place_pad__h03`

## Assigned heuristic

The assigned hypothesis is a **carry-release object dynamics prior** for the `red_transport_place_pad` skill.  During this skill the red block is expected to move approximately as a rigid object attached to the TCP while the gripper remains closed; after the gripper opens on the marked outside pad, the red block should remain stationary while the TCP retreats and begins the transition toward the blue-block skill.  The expanded M1_v2 slice is indices 455:760, so this implementation intentionally includes the shared lift-to-carry entry transition and the release-to-blue-approach exit transition.

## Executable adaptation

The provided interface supplies state histories and joint/gripper target actions, but it does not provide inverse kinematics, forward kinematics, contact labels, images, or an online physics model.  Therefore the prior is implemented as learned representation and learned differentiable losses rather than as a scripted controller or analytic rigid-body constraint.

The model receives only two causal observations at deployment.  Future observations are used only as training labels inside `compute_loss`.

## Architecture

`policy.py` defines a learned epsilon-predicting diffusion policy:

1. **Causal object-centric encoder.**  The encoder consumes the two raw observations normalized with the single shared complete-demonstration normalizer.  It also computes causal relative features in world units: TCP-to-red offsets, one-step TCP/red displacement, red-to-pad vector, TCP-to-pad vector, TCP/red-to-blue vectors, finger width, finger-width change, and drawer state.  These are fixed feature transformations, not a controller.
2. **Attachment and offset latent.**  Trainable heads predict an initial attachment logit, a TCP-to-red offset in metres, and a release-readiness latent.  These latents are concatenated with the learned history embedding and projected into the global conditioning vector for `appl.public.DiffusionBackbone`.
3. **Diffusion action model.**  The backbone predicts DDPM epsilon for the full 16-step horizon over the 8-dimensional normalized action sequence.  Inference remains standard action diffusion using the repository sampler.
4. **Learned red-object rollout auxiliary.**  During training only, a GRU rollout consumes the causal latent and a clean-action estimate.  It predicts future red positions, attachment logits, carry deltas, free deltas, and small residual deltas.  There is no analytic FK; the action-to-object dynamics are learned from demonstrations.

## Losses and gradient paths

The returned total loss is

`diffusion_loss + prior_weight * prior_loss`.

`diffusion_loss` is the standard masked epsilon MSE.  `prior_loss` includes:

- red future-position prediction loss against `future_obs[:, :, red_pose position]`, normalized by the shared normalizer's red-position scales;
- attachment BCE, with labels derived from demonstration future finger width (`sum(qpos[7:9]) < 0.055 m`);
- current attachment BCE and current offset regression for the causal initial state;
- carry-delta supervision encouraging the learned carry branch to match observed red displacement while the future gripper is closed;
- free-branch stationarity and small residual regularizers after opening.

To keep the auxiliary stable at high diffusion noise, the action input to the rollout is a blend between the demonstration clean action and the DDPM clean-action estimate.  The blend factor is nonzero and depends on `alpha_bar`, so the auxiliary losses still backpropagate through the predicted epsilon to the diffusion denoiser while avoiding high-noise amplification.

## Causal deployment inputs and handoff behavior

At deployment, the model uses only the current two observations: qpos/qvel, TCP pose, red pose, blue pose, drawer position/velocity, and the goal markers.  The observation cues that let it take over partway through the transition are:

- the gripper width indicates whether the hand is still in the closed carrying phase or has opened;
- the TCP-red relative position indicates whether the red block is likely attached and what offset should be carried;
- red-to-pad and TCP-to-pad vectors locate the remaining transport/release motion;
- after release, the stationary red pose and open fingers indicate that the policy should retreat toward the blue-block approach region.

The policy learns the complete expanded slice.  Starts near index 455 have closed fingers and a stable red/TCP offset; starts later in the slice can be interpreted from the same causal cues, e.g. low red height near the pad with closed fingers before release, or open fingers with the red block stationary after release.

## Dependencies and limitations

The implementation depends only on `torch` and `appl.public`.  It uses the assigned shared normalizer and does not refit per-skill statistics.  It does not implement IK, image encoders, analytic contact detection, rigid-body collision checking, or hard geometric constraints.  The learned attachment latent is supported by the demonstrations but may not recover from unseen red slips, unstable grasps, or large rotations that break the demonstrated TCP-red relation.
