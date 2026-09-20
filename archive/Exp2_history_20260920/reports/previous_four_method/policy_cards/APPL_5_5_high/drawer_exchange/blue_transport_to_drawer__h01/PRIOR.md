# Prior for `blue_transport_to_drawer__h01`

## Original heuristic identity

The assigned heuristic is **drawer-frame blue insertion** for the `blue_transport_to_drawer` skill.  Its hypothesis is that the transport target should be represented relative to the articulated drawer cavity rather than as a fixed absolute waypoint.  The drawer cavity center used by the assignment is

`drawer_center_world = [0.19 - drawer_position, 0, 0.035]`.

The implemented policy keeps the full assigned expanded slice, actions 810 through 1030 of each demonstration, including both shared transition regions: attachment/lift from the predecessor and insertion/release overlap with the successor.

## Executable diffusion-policy implementation

The model is a learned DDPM action diffusion policy.  At deployment, `forward(noisy_action, timestep, raw_history)` receives only the two causal state observations and predicts epsilon for the normalized 16-step action sequence.  The fixed sampler outside this file performs DDPM denoising and decodes the sampled normalized actions.

The architecture uses:

1. The shared M1_v2 observation range normalizer supplied in the spec, fitted on the complete original demonstrations.
2. Causal drawer-frame geometric features computed from the two available observations:
   - blue position relative to the drawer center;
   - blue position relative to the observed blue goal;
   - TCP-to-blue and TCP-to-drawer errors;
   - blue-goal-to-drawer offset;
   - finger width/balance;
   - drawer open margin and drawer velocity;
   - current inside-cavity signed margins using the cube's quaternion-derived axis-aligned XY extents;
   - red-pad margins as task context;
   - two-step deltas for TCP, blue, drawer, fingers, and arm qpos.
3. A trainable MLP condition encoder that maps the normalized observation history plus these prior features to a 256-dimensional global condition.
4. The provided conditional U-Net diffusion backbone, conditioned on that learned vector, to predict action noise.

The geometric feature scales are broad task/workspace scales or contract geometry values, not per-skill empirical re-normalization.  No observation scale is refit on the slice.

## Auxiliary prior loss and gradient path

In addition to the standard epsilon-prediction diffusion loss, training uses a small auxiliary prior loss.  From the predicted epsilon and the DDPM `alpha_bar`, the code forms a differentiable estimate of the clean normalized action sequence.  A trainable auxiliary predictor consumes:

- the current learned condition;
- each denoised action slot;
- a cumulative action latent over the horizon;
- the action-slot fraction.

It predicts future blue position in the drawer frame and a future `blue_inside` logit.  Labels come from `future_obs` during training only.  The target inside predicate uses the assignment's drawer center, blue z interval, cavity half-widths, and quaternion-based rotated block XY extents.  The auxiliary loss is therefore not a constant penalty computed only from observed poses: it is a supervised loss on trainable predictions that depend on the model's denoised action estimate, so gradients flow through the auxiliary head into the diffusion denoiser.  The auxiliary term is lightly weighted (`0.05`, with inside BCE subweight `0.15`) so that action imitation remains the primary objective.

To avoid high-noise numerical amplification, the clean-action estimate used by the auxiliary head is passed through a bounded `tanh` transform and the auxiliary mask is weighted by `alpha_bar`.

## Causal deployment information

At runtime the policy observes only the current two-step state history: qpos, qvel, TCP pose, red pose, blue pose, drawer position/velocity, and goal fields.  It does not observe future labels.  The drawer-frame features let the policy take over partway through the transition because the state reveals where the blue block and TCP are relative to the open drawer cavity, whether the fingers are open/closing/closed, and whether the blue block is still on the table, lifted high, descending, or already near the cavity.

Typical causal phases within the slice are:

- near index 810: blue is on or near the table while the TCP is aligned with it and fingers are closing/open-to-closing;
- by about index 870: blue is lifted around z 0.27-0.28 m with close TCP-blue proximity and closed fingers;
- by about index 930: blue is above the drawer area, still held high;
- by about index 990 and later: blue is near x -0.172, y 0.074, z 0.063 and the gripper command opens.

The model learns all of these phases as a single diffusion policy over the full expanded segment rather than using a scripted phase switch.

## Applicability and termination cues

This policy is most applicable when the drawer is open beyond the task threshold, the red block is already on the pad, and the blue block is being grasped, lifted, carried, or inserted into the open drawer.  It should continue while the TCP/blue pair is moving from the grasp/lift state toward the drawer-frame target or while the blue block is descending into the cavity.

A useful exit state for the successor is blue centered inside or immediately above the open drawer cavity, blue z descending toward approximately 0.063 m, and the gripper opening or open.  Under the task contract, completion can already be true once the drawer is open, red is on the pad, and blue is inside; no additional release, speed, clearance, or sustained hold condition is added here.

## Dependencies and limitations

The implementation uses only state inputs and the allowed `appl.public` diffusion backbone.  It does not implement inverse kinematics, forward kinematics, image encoders, external geometric planning, hard constraints, or a replay controller.  The data show an open drawer with limited variation in `drawer_position`; recovery from a closed drawer, a dropped blue block, or a large perturbation outside the observed handoff distribution is not established.
