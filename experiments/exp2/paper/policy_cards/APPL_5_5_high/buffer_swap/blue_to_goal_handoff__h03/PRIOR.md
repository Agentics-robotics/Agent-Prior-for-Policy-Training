# PRIOR: blue_to_goal_handoff__h03

## Assigned research hypothesis

The assigned heuristic is a contact-mode temporal prior for the long middle slice of the buffer-swap task. The policy must cover the expanded range `320:860` for every training demonstration, not only the central blue transport. This includes the predecessor overlap, where the robot is open/high and then approaches or closes on the blue block, and the successor overlap, where blue has been placed and the robot retreats toward, approaches, closes on, or lifts the buffered red block.

The heuristic proposes six latent modes:

1. `entry_approach_blue`
2. `blue_grasp_lift`
3. `blue_carry`
4. `blue_place_release`
5. `retreat`
6. `red_handoff`

The purpose is not to hand-code these behaviours, but to give a learned diffusion model a trainable temporal representation that can disambiguate the same skill at different contact and handoff phases from causal observations.

## Implemented learned diffusion policy

`policy.py` implements `ContactModeDiffusionPolicy`, a real DDPM epsilon-prediction action model. Its deployment `forward(noisy_action, timestep, raw_history)` receives only the two causal observations provided by the framework and returns predicted diffusion noise for the 16-step action chunk. It never reads future observations at deployment.

The model contains:

- a shared-normalized state encoder using the assignment normalizer fitted on complete demonstrations;
- fixed-scale physical relative features, such as TCP-to-blue, TCP-to-red, object-to-goal vectors, object heights above the table, xy distances, and finger aperture;
- a GRU over the two causal observation steps;
- a six-way trainable mode posterior head;
- a mode embedding and condition MLP;
- one standard learned `appl.public.DiffusionBackbone` Conditional U-Net DDPM denoiser;
- six small learned residual denoising experts gated by the mode probabilities.

The denoiser output is:

`epsilon = shared_unet(noisy_action, timestep, condition) + 0.25 * sum_m p(m | history) * residual_expert_m(noisy_action, timestep, condition)`.

This is still a learned action diffusion policy. The residual experts are not controllers; they only add learned epsilon residuals inside the diffusion denoising network.

## Weak mode supervision and losses

The auxiliary mode targets are computed from observed demonstration states during training. They use finger aperture, TCP/object distances, blue and red object heights, and blue-goal proximity to assign weak labels for the six modes. These labels are event proxies from successful demonstrations, not hard contact truth and not runtime rejection rules.

The total training loss is:

- `diffusion_loss`: standard masked epsilon MSE between predicted and sampled DDPM noise;
- `prior_loss`: trainable mode-structure auxiliaries:
  - current-history mode cross entropy, weight `0.10`;
  - teacher-forced future-observation mode cross entropy, weight `0.05`;
  - adjacent mode-probability smoothness on current plus future teacher-forced classifications, weight `0.005`.

All auxiliary terms depend on trainable predictions from the mode head. The future observations are used only as training labels for mode classification and smoothness; they are not inputs to deployment action prediction. No penalty is merely a constant function of observed poses.

## Causal information used for handoff takeover

The model can take over partway through either transition because the two causal observations contain the necessary mode cues:

- qpos/qvel and finger width distinguish open approach, closing, closed carry, release, and renewed closing on red;
- TCP pose and object poses in world metres distinguish high retreat from low grasping;
- blue pose relative to `blue_goal` distinguishes carry/place/release from later retreat;
- red pose height and TCP-red distance distinguish open approach to red from the red handoff/lift overlap.

At an early entry state like index 320, blue is on the upper/red-goal region, the TCP is high, and fingers are open, so the posterior should favour `entry_approach_blue`. Around index 460, the TCP is low at the blue block and the fingers are closing/closed, so it should favour `blue_grasp_lift`. During indices like 500-600, blue is lifted and transported; near 650 it is placed and released at `blue_goal`. Around 700-740 the model observes blue placed and an open/high retreat. Around 780-860 the TCP is near red and the fingers close or red is lifted, which supports the `red_handoff` exit mode.

## Adaptation from hypothesis to executable model

The original handoff description suggested a mixture-of-experts denoiser. To keep the model compact and stable under the supplied training interface, the implementation uses one shared diffusion U-Net plus six small mode-gated residual denoising experts. This preserves the switching prior while avoiding multiple full-size U-Nets. The mode posterior is learned and differentiably gates the residuals and condition embedding.

No inverse kinematics, image encoding, invariant action representation, hard geometric containment constraint, or scripted replay controller is implemented. The output remains the normalized native 8D action sequence expected by the fixed DDPM sampler.

## Applicability and termination cues

This policy is intended for the full expanded slice of `blue_to_goal_handoff`: predecessor overlap, blue approach/grasp/lift/carry/place/release, retreat, and successor overlap toward red. It is best invoked when observations resemble that successful sequence and the normalizer statistics match the supplied assignment.

Useful termination or transfer cues are observational, not task success definitions: blue should be placed at the supplied `blue_goal` on the table, and the arm should either be open/high after release or already in the red handoff phase with TCP near red, gripper closing, or red lifted. The completion contract remains the task-level condition `red_at_goal AND blue_at_goal`.

## Limitations

The weak modes may be wrong if a grasp misses, if an object is displaced outside the demonstrated support, or if contact differs from the successful demonstrations. The recurrent structure can infer phase from two observations but does not provide explicit recovery or branch planning. The policy relies on the shared full-demonstration normalizer provided by the assignment and does not refit scales to the small skill slice.
