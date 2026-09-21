# PRIOR: blue_approach_grasp_lift__h03

## Assigned hypothesis

The assigned heuristic is **Blue attachment latent: represent the contact-to-carry switch explicitly to decide when blue transport can begin**.  The intended skill is the expanded `blue_approach_grasp_lift` slice, indices 650--870 in each M1_v2 demonstration.  This slice includes shared transition context from the red pad area, the move to the blue block, the top grasp, finger closure, and the initial lift.  The overlap with the following blue-transport policy is intentional for indices 810--870.

## Executable adaptation

The repository supplies a fixed DDPM training and sampling loop over normalized 8-D Panda joint/gripper actions.  I implemented the heuristic as a learned conditional action diffusion model rather than as a scripted grasp controller.  The attachment variable is a trainable latent predicted from the two causal observations available to `forward`; future observations are used only in `compute_loss` to make training labels for that latent.

The model does not perform IK, does not use images, and does not impose a hard contact constraint.  It learns joint target denoising from the demonstration actions while using the attachment latent to bias the conditioning representation through the approach/close/lift switch.

## Architecture

`policy.py` defines `BlueAttachmentLatentDiffusionPolicy`:

- Input to deployment/training `forward`: `raw_history [B,2,47]`, `noisy_action [B,16,8]`, and DDPM timestep.
- Observation processing uses the single shared M1_v2 normalizer from the spec.  No per-skill normalization is fitted.
- The causal encoder receives:
  - the two normalized 47-D observations,
  - world-frame relative features: TCP-blue, previous TCP-blue, relative change, TCP-red, red-goal, blue-goal, approximate drawer-center relation, TCP-to-blue-goal,
  - short-history TCP/blue/red motion,
  - finger width and finger-width change,
  - TCP/blue/red heights, distances and drawer state.
- A small MLP produces a 256-D condition vector.
- A learned attachment head predicts a scalar blue-attachment logit from that condition.
- The scalar probability mixes learned `free` and `attached` phase tokens.  The mixed token is added to the condition before a standard `appl.public.DiffusionBackbone` Conditional U-Net predicts epsilon for the action sequence.

Thus the model remains a standard learned action diffusion model: inference samples actions with the supplied DDPM sampler, and the latent only changes the learned conditioning vector.

## Loss terms and gradient paths

The returned total loss is:

`diffusion_loss + prior_loss`

where:

1. `diffusion_loss` is the normal epsilon-prediction loss from `appl.public.epsilon_loss`.  It trains the U-Net, encoder and latent pathway because the latent-conditioned representation is used in every denoising call.
2. `attachment_loss` is binary cross entropy on the trainable attachment logit.  The target is computed only during training from the current state plus masked future observations: closed fingers, TCP-blue proximity, future blue height increase, and a stable future TCP-blue transform.  This follows the assigned idea of using future lift/contact evidence as a training label while keeping deployment causal.
3. `gripper_phase_loss` reconstructs the clean DDPM gripper channel from the predicted epsilon and applies a small smooth-L1 penalty to the demonstrated normalized gripper command, weighted more strongly near contact/carry states.  This auxiliary term depends on the model's clean-action estimate and provides a gradient to the denoiser around the open-to-close switch.

No auxiliary loss is a constant penalty on observed poses alone; each prior component depends on either the predicted latent or the predicted denoised action.

## Causal deployment information

At deployment the policy sees only the most recent two state observations.  The observations contain enough information for this handoff because they include:

- finger positions and velocities (`qpos[7:9]`, `qvel[7:9]`), which distinguish open, closing and closed gripper states,
- TCP pose in world coordinates,
- blue block pose in world coordinates,
- short-history TCP/blue relative motion,
- red block/pad context and drawer state for the expanded slice.

Therefore the same model can take over partway through the shared transition: if invoked near index 650 it observes TCP over the red-pad area with open fingers and learns the move toward blue; if invoked near 760--810 it observes open fingers above/near blue and learns descent/closure; if invoked after contact it observes closed fingers and a rising or coupled blue block and continues the lift toward the transport-ready exit.

## Applicability, termination and limitations

The prior is applicable when the state estimator provides reliable TCP, blue pose and finger qpos, and the blue block can be top-grasped without reorientation.  A useful exit for the successor is closed fingers with blue lifted clearly off the table, typically around z 0.20--0.28 m in the demonstrations, with small TCP-blue relative motion.

Limitations: the model has no explicit slip detector, no regrasp behavior, no visual correction, and no hard guarantee that the learned attachment latent equals physical contact.  The full task completion condition remains the external three-goal rule: drawer open, red on the outside pad, and blue inside the drawer.
