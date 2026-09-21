# Prior document: blue_release_retreat_complete__h03

## Assigned heuristic

This policy implements heuristic 3 for `blue_release_retreat_complete`: a near-goal residual stabilizer for the final phase of the task. The assigned expanded segments are the full slices starting at index 930 and ending near indices 1063--1077 for demonstrations demo1000--demo1011. The overlap interval [930, 1030) is shared with the previous blue transport skill and contains the transition in which the gripper is still closed around the blue block, the drawer is open, the red block is already on the outside pad, and the TCP/blue pair descends from above the drawer toward the cavity. The later part of the slice contains release and retreat, with open fingers and the TCP moving upward while the blue block remains at the drawer goal.

The original heuristic suggested a residual local policy around a nominal drawer-frame controller. The available interface provides no analytic inverse kinematics, no forward kinematics, no contact model, and no image observations. Therefore the executable prior adapts the hypothesis as a learned residual action diffusion model: it learns the nominal local action trajectory from causal state features and uses that learned nominal as conditioning and regularization for a DDPM noise predictor. No scripted action controller or replayed trajectory is used at deployment.

## Causal inputs used at deployment

`model.forward` receives only the current two-step causal state history `raw_history [B, 2, 47]`. The model uses the provided shared M1_v2 normalizer fitted on the complete original demonstrations; it does not refit per-skill or per-slice scales. In addition to the normalized raw state, the encoder constructs differentiable state features in world units:

- blue position relative to the observed `blue_goal`, scaled by 0.15 m;
- TCP position relative to the blue block and to `blue_goal`;
- red position relative to `red_goal`, so the model can condition on the already-completed red subgoal;
- blue/TCP position relative to the open drawer reference `x = 0.19 - drawer_position`;
- drawer position and drawer velocity;
- finger opening from the two finger joint positions;
- quaternion components for TCP, red, and blue as observed state inputs;
- two-step motion deltas for blue, TCP, qpos, fingers, and drawer.

These features let the model take over partway through the transition: at the beginning of the assigned slice the causal observation shows an open drawer, a closed gripper, the blue block and TCP nearly coincident above the drawer, and downward motion. Around release it observes the blue block reaching the goal height and the gripper opening. During retreat it observes open fingers, fixed blue pose, rising TCP height, and small velocities.

## Architecture

The model remains a learned action diffusion policy. It predicts DDPM epsilon for the normalized 16-step, 8-dimensional action sequence. The fixed sampler and optimizer supplied by the repository are unchanged.

Implemented modules in `policy.py`:

1. **Local state feature encoder.** A small MLP maps the normalized causal state history and drawer-frame relative features to a 256-dimensional latent vector.
2. **Learned nominal stabilizer head.** From the latent vector, a trainable head predicts a nominal 16-step encoded action trajectory. This is the executable replacement for the heuristic's unavailable analytic nominal controller/IK approximation.
3. **Conditional diffusion backbone.** The nominal trajectory is flattened and combined with the latent state to form a 256-dimensional global condition for the supplied Conditional U-Net DDPM backbone. The backbone predicts epsilon from the noisy action sample, timestep, and condition.
4. **Learned future descriptor head.** For auxiliary training only, a trainable head maps the denoised action estimate and causal latent state to future local descriptors. This provides a differentiable coupling between denoised actions and the desired release/retreat state evolution without claiming physical simulation or hard constraints.

## Losses and gradient paths

The main loss is the standard masked epsilon prediction loss against the sampled diffusion noise. The prior loss is deliberately small relative to the diffusion loss and contains only trainable predictions:

- **Nominal action loss** (`0.10` weight): masked MSE between the learned nominal encoded action sequence and the demonstration encoded actions, with a mild extra weight on the gripper command. This trains the nominal local stabilizer head.
- **Residual clean-action loss** (`0.005` weight): from the predicted epsilon, a DDPM clean action estimate is reconstructed. At low-noise timesteps it is encouraged to remain close to the learned nominal trajectory. This gives gradients to the epsilon predictor and nominal head and implements the residual-around-nominal bias.
- **Future descriptor loss** (`0.01` weight): a learned descriptor head predicts future blue-goal error, TCP-blue relation, finger opening, joint-speed norm, drawer position, and TCP height from the denoised action estimate and latent state. Targets come from `future_obs`; the loss is masked and downweighted at high diffusion noise. Because the descriptor prediction depends on trainable outputs, it is not a constant observation-only penalty.

The auxiliary losses do not redefine task success and do not impose hard geometry. They bias training toward the observed sequence: final centering/lowering, gripper opening, and upward retreat.

## Applicability and handoff interpretation

This policy should be invoked only near the assigned support. Appropriate entry observations include an already-open drawer (`drawer_position` about 0.297 m in the measured starts), the red block near the outside pad, the blue block above or near the drawer goal in XY, the TCP close to the blue block, and a mostly closed gripper. The overlap [930, 1030) is intentionally included: the model learns to continue the late descent before the completion predicate is true, not merely to hold a completed state.

The useful exit state is a completed or near-completed final task state: blue block centered in the drawer at the observed goal height, drawer still open, red still on the pad, gripper open, TCP high/clear of the block, and low qvel. No manipulation successor is required by this skill; a higher-level inference API should check the supplied three-goal completion contract.

## Dependencies and limitations

The implementation depends only on `torch` and `appl.public`. It uses the supplied shared normalization and supplied DDPM sampler/backbone. It does not use images, external files, inverse kinematics, forward kinematics, analytic collision/contact reasoning, equivariant action representations, or a hard safety filter.

The policy is intentionally local. If invoked while the blue block is still at the table start or far outside the drawer neighborhood, the learned nominal prior is outside its support and should not be expected to transport the block. The model can learn recovery only within the variation present in the assigned slices and overlap; larger blue-drawer errors, lost grasps, closed drawer states, or disturbed red placement are untested.
