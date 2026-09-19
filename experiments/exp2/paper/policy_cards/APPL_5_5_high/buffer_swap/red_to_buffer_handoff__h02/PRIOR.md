# PRIOR.md: red_to_buffer_handoff__h02

## Original research hypothesis

The assigned heuristic is the swap-topology buffer prior for the first phase of the buffer-swap task.  The red block initially occupies the lower region that will later be the blue goal, while the blue block occupies the upper region that is the red goal.  Directly placing red at its final red goal is topologically blocked or undesirable because blue is there.  The useful first subgoal is therefore to move red to the free central buffer around world position `[-0.181, 0.0, 0.02]`, freeing the lower/blue-goal region while leaving blue available for the next skill.

The assigned expanded slice is the full range `[0, 460)` for each demonstration, not only the red placement.  It includes the overlap `[320, 460)` with the following blue-acquisition skill.  Demonstrations show that by about index 300-320 red has settled near the central buffer and the gripper is open/high; by index 459 the TCP is low near blue and fingers are closing.  This policy therefore learns both red-to-buffer and the demonstrated transition toward blue acquisition.

## Executable implementation

The policy is still a learned action diffusion model.  `policy.py` builds an epsilon-predicting conditional DDPM with the repository `DiffusionBackbone` 1D U-Net.  It predicts normalized absolute Panda joint/gripper action sequences.  There is no scripted controller, no IK, no image encoder, and no replay of recorded actions.

The prior is implemented in the learned conditioning path:

1. The two causal raw observations are normalized with the single assignment shared normalizer fitted on complete original demonstrations.
2. Additional world-frame geometric/topological features are computed from the current observation:
   - TCP-to-red, TCP-to-blue, TCP-to-buffer, red-to-buffer, red-to-blue-goal, blue-to-red-goal and red-to-blue relative vectors;
   - corresponding XY distances;
   - soft scores for `red_in_initial_blue_goal`, `blue_in_red_goal` and `red_buffered`;
   - object/TCP heights relative to the buffer tabletop height;
   - finger-width/opening features;
   - red-goal to blue-goal separation.
3. A trainable MLP encoder maps normalized history plus these prior features to a latent state.
4. A trainable subgoal head predicts a red buffer placement in metres.  It is parameterized as a bounded residual around the heuristic buffer center: `x=-0.181`, `y=(red_goal_y+blue_goal_y)/2`, `z=(red_goal_z+blue_goal_z)/2`, with residual bounds `[0.08, 0.08, 0.035]` m.
5. A trainable predicate head predicts the current `red_buffered` logit.
6. The latent state, predicted subgoal, predicted buffered probability and prior features condition the diffusion U-Net.

This makes the topological buffer variable part of the differentiable policy representation.  The diffusion loss also backpropagates through this conditioning path because the U-Net receives the learned subgoal and predicate predictions.

## Losses and gradient paths

`compute_loss` returns:

- `diffusion_loss`: standard masked epsilon MSE against the sampled DDPM noise for action sequences.
- `prior_loss`: `0.20 * subgoal_l2 + 0.05 * red_buffered_bce`.

The subgoal loss supervises the trainable subgoal head.  The target is the current red pose if red is already buffered; otherwise it uses the first short-horizon future red pose in the training batch that lies on the table near the central buffer, falling back to the heuristic buffer center when release is outside the available 16-step future window.  This adaptation is necessary because the public batch interface does not provide the whole future slice or a precomputed release index.  The fallback remains a supervised target for a trainable prediction and is documented as a prior, not as a deployment-time controller.

The buffered BCE supervises the trainable predicate head from the current causal observation: red is labelled buffered when its current pose is near the central buffer on the tabletop.  Both auxiliary terms have gradients with respect to model outputs.  No loss term is a constant-only penalty computed from observed poses.

## Causal deployment inputs

At deployment the model receives only the two-observation state history supplied to `forward`: qpos, qvel, tcp_pose, red_pose, blue_pose, zero drawer compatibility channels, red_goal and blue_goal.  It computes the same prior features and learned subgoal/predicate predictions from current observations only.  Training future observations are used only to create auxiliary labels in `compute_loss`.

Because the observation includes actual red_pose, blue_pose, tcp_pose and finger qpos, the model can be invoked partway through the shared transition.  If red is still lower/not buffered, the prior features indicate the need to continue red placement.  If red is already buffered and the TCP is high/open or moving toward blue, the same model conditions the diffusion backbone to generate the overlap behavior toward blue acquisition.

## Applicability and termination cues

Applicable support is the demonstrated buffer-swap topology: red starts in or near the lower blue-goal region, blue occupies the upper red-goal region, and the central buffer is free.  The useful exit state is red released on the central buffer, blue still available for the next skill, and the robot either open/high after release or in the later overlap approaching/closing on blue.

Termination guidance does not redefine task success.  The task-level completion contract still requires both red and blue at their final goals.  This policy is only the red-to-buffer handoff component and transition.

## Dependencies and limitations

The implementation relies on state observations in world metres and quaternion components and on the supplied shared normalizer.  Relative feature scales are fixed physical scales, not refitted per-skill empirical scales.  The encoded buffer x-coordinate is the demonstrated world-frame prior; the model does not search for arbitrary free cells and has no explicit obstacle or occupancy reasoning.  Robustness to an occupied buffer, failed grasp recovery, large unseen fixture changes, or a different intended temporary cell is untested.
