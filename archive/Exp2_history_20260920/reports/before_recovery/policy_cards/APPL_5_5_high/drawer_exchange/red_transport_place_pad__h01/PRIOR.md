# PRIOR: red_transport_place_pad__h01

## Assigned heuristic

The assigned heuristic is **Pad-frame red placement** for the `red_transport_place_pad` skill.  The expanded training slice is indices 455:760 in every M1_v2 demonstration.  The slice starts during the predecessor overlap while the red block is already grasped and being lifted/carried, passes through the red placement and gripper opening on the fixed outside pad, and ends during the successor overlap while the TCP is retreating toward the blue block.

The research hypothesis is that action selection for this skill should be easier when the red object is represented relative to the fixed red pad goal `[-0.18, -0.30, 0.02]` rather than only as an absolute world pose.  The demonstrations show the red block moving from variable drawer-side poses to a tightly clustered final pad pose, with the gripper opening after the block is down and the arm then moving toward blue.

## Executable learned diffusion implementation

The implementation remains a learned action diffusion policy.  It predicts epsilon for the fixed DDPM sampler over the normalized 16-step action sequence of 7 Panda joint targets plus one gripper command.  There is no scripted controller, no replay, no inverse kinematics, no hand-coded action replacement, and no image encoder.

The model has three learned parts:

1. **Causal pad-frame condition encoder**
   - Input: only the two current causal observations `[B, 2, 47]`.
   - The original 47-D observations are normalized using the single shared M1_v2 normalizer fitted on the complete demonstrations.
   - Additional causal relative features are computed in world coordinates:
     - red-to-red_goal error,
     - TCP-to-red offset,
     - TCP-to-red_goal offset,
     - TCP-to-blue and blue-to-blue_goal transition cues,
     - drawer-open margin,
     - finger width,
     - short two-frame TCP/red/joint/finger deltas,
     - current soft margins for the red-on-pad geometry.
   - These features are passed through an MLP to a 256-D conditioning vector.

2. **Conditional diffusion backbone**
   - The policy uses `appl.public.DiffusionBackbone` with the assigned training configuration.
   - The backbone receives the noisy action sequence, DDPM timestep, and learned condition vector, and returns predicted epsilon with shape `[B, 16, 8]`.

3. **Auxiliary learned placement predictor**
   - During training only, the predicted epsilon is converted into a differentiable clean-action estimate `x0` using the DDPM formula supplied by the interface.
   - A learned auxiliary head receives the causal condition and the estimated action plan and predicts future red placement quantities over the same horizon:
     - red center error relative to `red_goal`,
     - a `red_on_pad` logit computed against future labels from the task geometry,
     - a gripper-open logit from future finger-width labels.
   - The auxiliary loss is differentiable through `x0` and therefore through the diffusion model's predicted epsilon.  It is not a constant penalty computed only from observed poses.

The total training loss is:

`diffusion epsilon MSE + 0.05 * (red-error MSE + 0.25 * red-on-pad BCE + 0.10 * gripper-open BCE)`.

The auxiliary losses use `future_obs` labels only during training.  Deployment forward passes receive only the causal observation history and do not access future states.

## Adaptation from heuristic to data interface

The heuristic mentions controlling the carried object relative to the pad.  The provided action space, however, is absolute joint and gripper targets, and the interface supplies no kinematics solver or contact model.  Therefore the implementation adapts the heuristic into:

- a learned pad-frame representation in the conditioning features, and
- a learned action-to-future-placement auxiliary predictor that biases the diffusion model toward action sequences associated with successful red placement and release.

This is a representational and loss prior, not a hard geometric constraint.

## Expanded-slice handoff behavior

The policy is trained over the full 455:760 expanded slice.

- **Entry overlap 455:530:** observations show the drawer open, fingers closed around red, TCP nearly co-located with the red block, and red rising from about 0.063 m toward the high carry path.  The policy can take over here because the causal state includes red pose, TCP pose, finger width, drawer position, and their recent two-frame deltas.
- **Core placement:** red is transported in world frame toward the fixed pad.  The pad-frame features expose the remaining red-to-goal error and TCP-to-red offset while the network still learns joint actions from demonstrations.
- **Exit overlap 650:760:** red is already on the pad near z 0.02 m and the gripper is open.  The TCP is retreating upward and then toward the blue block.  The condition includes red-on-pad margins, finger width, TCP pose, blue pose, and TCP-to-blue cues so the learned policy can cover the shared transition to the successor.

## Applicability and limitations

This prior is intended for the fixed-pad M1_v2 task with the red pad at `[-0.18, -0.30, 0.02]`.  It assumes the red block is already grasped or at least very near the TCP at entry, and that the drawer remains open.  It does not implement recovery if the grasp is lost, if the pad moves to an unseen location, if the object is far outside the demonstrated transition manifold, or if contact dynamics differ substantially from the demonstrations.

The final task completion rule remains the supplied contract: drawer open, red on the outside pad, and blue inside the drawer at the same observation.  The release and retreat cues documented here are skill-handoff guidance, not additional task-success requirements.
