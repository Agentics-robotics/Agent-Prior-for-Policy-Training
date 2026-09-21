# Prior for `red_place_to_blue_entry__h02`

## Assigned hypothesis

The assigned heuristic is a **release-reset prior** for the `red_place_to_blue_entry` skill.  The full expanded slice runs from red carry after unstacking through red descent, release, retreat, blue approach, blue closure, and blue lift entry.  The scientific hypothesis is that this transition should be represented as an explicit manipulation-state change: after red reaches the red pad, the policy should switch from closed carry to open retreat, leaving red stable and making the gripper available for blue.

## Adaptation to the executable diffusion-policy interface

The provided interface supplies state histories and joint-space actions, but no force/contact sensor, no IK solver, no images, and no non-diffusion control hook.  I therefore implemented the prior as a learned epsilon-predicting DDPM policy whose conditioning includes a trainable release/reset latent inferred causally from the two available observations.  The latent is used only inside the neural policy; it is not a scripted gate and it does not overwrite sampled actions.

The deployed `forward(noisy_action, timestep, raw_history)` receives only the causal raw history `[B,2,47]`.  Future observations are used only during training for auxiliary labels and losses.

## Architecture

`policy.py` builds a `ReleaseResetDiffusionPolicy` with these learned components:

1. **Shared-normalized observation encoder.**  The two raw observations are normalized with the assignment's shared full-demonstration normalizer.  No per-skill or slice-specific scale is fit.  The encoder also receives world-frame relative features computed from the same observations: red-to-red-goal, blue-to-blue-goal, TCP-to-red, TCP-to-blue, red-to-blue, two-step object/TCP displacements, finger aperture, finger aperture change, and smooth release cues such as red near goal/table and TCP retreat height.

2. **Four-state release/reset latent.**  A learned stage head predicts logits for `carry`, `descend_contact`, `open_release`, and `retreat_switch`.  The softmax probabilities mix learned phase embeddings.  The phase probabilities and embedding mixture are concatenated with the observation embedding to produce the 256-dimensional global condition for the diffusion U-Net.

3. **Diffusion action decoder.**  The action model is the supplied conditional 1-D U-Net backbone.  It predicts DDPM epsilon for normalized joint/gripper action horizons of length 16.  Actions remain seven absolute Panda joint targets plus one gripper command.

4. **Auxiliary trainable heads.**  A gripper-intent head predicts the demonstrated open/close sign across the horizon from the causal representation.  A red-future head predicts normalized future red XYZ deltas from the condition and the diffusion model's current clean-action estimate.  These heads provide gradients to learned representations and, for the action-conditioned red head and clean-gripper BCE, to the diffusion epsilon prediction.

## Loss terms and gradient paths

The returned `loss` is the standard diffusion epsilon loss plus a weighted prior loss:

- `diffusion_loss`: MSE between predicted and sampled noise using the framework mask.
- Stage cross entropy, weight `0.05`: labels are computed from causal current/previous observations using red-goal proximity, red height, finger aperture/motion, TCP-red geometry, and blue lift/approach cues.  This trains the latent phase predictor.
- Gripper-intent BCE, weight `0.03`: the causal representation predicts the demonstrated horizon open/close command sign.
- Clean-action gripper BCE, weight `0.03`: the DDPM clean estimate `x0=(x_t-sqrt(1-alpha_bar)*epsilon_pred)/sqrt(alpha_bar)` is used as a gripper open/close logit, so this term directly affects the diffusion decoder.
- Action-conditioned red future loss, weight `0.05`: a learned head predicts future red XYZ deltas from the condition and the clean action estimate.  It is weighted more strongly once future labels show red near the pad/table, encouraging the model to represent no-drag release.  This is a trainable prediction; it is not a constant penalty computed only from observed poses.

Auxiliary losses are alpha-weighted or clipped in the clean-action estimate to avoid excessive gradients at high DDPM noise.  Padding and slice boundaries use the provided masks.

## Causal cues for taking over inside the transition

The model can take over at several points in the expanded handoff because the observation history contains the relevant state:

- During carry/descent, red is elevated and close to the TCP while fingers are closed.
- Near contact, red XY is close to the red goal and red Z approaches the tabletop goal height while fingers may still be closed.
- During release, finger aperture/velocity changes and red remains near the pad instead of following the TCP.
- During retreat/switch, red stays at the pad while TCP-red distance and TCP height increase, then TCP moves toward blue; later blue height and closed fingers indicate blue pickup.

These cues are all in the world-frame state vector: qpos/qvel fingers, TCP pose, red pose, blue pose, and goal coordinates.

## Applicability and termination

This prior is applicable when the red block is the currently manipulated object or has just been placed, and the next useful state is an available gripper for blue.  The policy should be continued through the shared overlap when the API wants the complete red-release-to-blue-entry transition.  Termination/handoff is appropriate once red is stable on the red pad, the hand has opened and retreated, and either the gripper is ready to approach blue or blue is already grasped/lifted for the successor.

## Dependencies and limitations

The implementation depends only on the 47-dimensional low-dimensional state and the shared normalizer supplied by the assignment.  It does not implement IK, force control, contact detection, collision constraints, image encoders, object-frame equivariance, or a scripted action controller.  Contact and release are inferred from finger aperture and relative TCP/object motion.  The dataset does not demonstrate recovery from dropping red outside the pad or from unusual friction/wedging, so this model should not be claimed to solve those cases.
