# Prior: observation-correctable progress memory for finish_red

## Identity and unchanged research hypothesis

Policy `finish_red__h02`, skill `finish_red`, heuristic index 2, M1_v2.

The assigned heuristic states: **"Remember completed manipulation while inferring the current contact mode: use a soft, observation-correctable progress memory to disambiguate approach, release and terminal retreat. Hypothesis: recurrent diffusion initialized from expanded overlaps avoids repeating completed actions at handoff."** This document records implementation choices separately; it does not revise the source heuristic or claim to have verified its prediction.

The motivating ambiguity is temporal, not an optimization of placement margins. A low, downward-facing TCP can accompany either pickup closure or setdown opening. Completed blue placement should normally remain remembered during red acquisition, but observation disagreement must be visible instead of being hidden by an irreversible latch.

## Executable adaptation to the available interface

The mandatory forward interface supplies **two causal observations**, not a carried recurrent hidden vector or an entire observation stream. Repeated calls during the 100 DDPM steps must not advance task memory. Accordingly, this implementation is a **bounded-memory, state-initialized recurrent DP**, not a claim of persistent long-range deployment memory. Each forward call deterministically reconstructs the belief from the two observations. There is no cache, trajectory identifier, absolute source index, clock, preset blue-done bit, or external state-machine schedule.

The learned initializer infers likely earlier progress from the current scene: blue supported at its goal versus co-located with the moving TCP; red still at the central buffer versus elevated or already at its goal; finger aperture and velocities; and causal TCP/object displacement. It is trained at the dataset's random anchors throughout the unchanged expanded slices. A duplicated-current-state initialization branch learns to approximate the full two-frame belief, reducing dependence on having a predecessor's hidden state. This is a soft training preference, not a guarantee that two observations resolve every alias.

To obtain multi-step progress supervision without feeding future observations into a policy encoder, a separate learned GRU forecasts future contact/progress belief from the current causal hidden state and an action chunk. It is trained with demonstrated normalized actions, and also with a differentiable DDPM clean-action estimate. **Future observations are labels only**: they provide weak mode targets, goal flags, and margin targets. They never enter the observation encoder, initial memory, or diffusion conditioning. The forecast branch is not used as an inference-time motor controller and does not provide additional measured deployment history. This is a necessary, explicitly limited adaptation of the original longer-memory proposal to the fixed interface.

## Full expanded training support and inspected evidence

All twelve assigned start/stop ranges are retained exactly in `pipeline.json`; the outer data loader owns sampling. No phase is removed or replayed. This includes blue descent and release, empty retreat, red approach and closure, red lift/carry, red setdown/opening, and the terminal retreat/tail. In particular, overlaps with `transfer_blue` are approximately 239-250 actions, not one exact handoff cut: demo10100 [600,850), demo10111 [603,844), and demo10107 [613,855) exemplify the supplied measured ranges.

Actual inspected observations establish the following support, not learned-policy success:

* demo10100:600 has blue at z=0.2093 m and TCP z=0.2095 m, red at the buffer near (-0.181,-0.0012,0.020) m, and per-finger positions about 0.0183 m. Blue being near its goal in XY does not mean it is placed. Other measured starts have blue height down to roughly 0.137 m.
* At 630, blue is at z=0.0219 m, TCP z=0.0221 m, and the action still requests g=-1. At 640, blue is at z=0.0200 m and fingers are approximately 0.0382 m each with g=+1, while TCP remains low. At 650 fingers are approximately 0.0400 m, still near the supported blue block. At 690 the empty TCP has risen to z=0.2911 m while blue remains supported.
* At 755 the open TCP is over the red buffer. At 800 it is low with an open command; 810 begins a negative gripper command while the measured fingers are still open, and 820 has approximately 0.0183 m per finger. At 850 the red block and TCP have lifted together to about 0.2127 m. Joint velocity, aperture, object location, and motion all matter; aperture alone is not a contact sensor.
* At 960 red is near its goal in XY but still at z=0.0512 m. At 980 red is supported at z=0.0200 m, fingers are opening around 0.0352 m, and TCP is at z=0.0234 m. At 1020 TCP is retreating at z=0.2183 m. At 1063 both objects remain at their goals, TCP z=0.3003 m, fingers are open, and joint velocities are nearly zero.
* demo10111:603,643,694,755,812,844,941,966,978,997,1060 and demo10107:613,654,821,855,1070 show the same qualitative ordering at different source indices. Their observations were inspected directly; indices are documentation evidence only, not model inputs or label rules.

## Representation and learned modules

The model outputs epsilon for a [B,16,8] action tensor. Actions remain the repository's affinely normalized **seven absolute Panda joint targets plus gripper command**, not task-space commands, residual waypoints, or copied demonstration actions. The fixed DDPM sampler and outer safety/action bounds remain unchanged.

Each frame has 77 features:

1. All 47 raw state channels normalized by the supplied **single full-demonstration range normalizer**. Quaternion components keep the supplied unit component scales, and drawer compatibility channels are retained as zeros.
2. Twelve world-frame relative-position features: red-TCP, blue-TCP, red-goal, and blue-goal.
3. Eighteen backward differences: nine joint/finger positions and the three positions of TCP, red, and blue. The first frame's differences are zero because its predecessor is unavailable. These are differences per observation interval, not claimed metre/second velocities. Measured qvel is already included in the raw state.

Relative positions and positional differences use the axiswise maximum of the supplied TCP/red/blue positional half-ranges. Joint differences use the supplied joint half-ranges. No per-skill scale is fitted; no narrow overlap statistics determine normalization. World geometry is intentional, with no claim of rotation/translation equivariance for joint actions.

A 77-to-192-to-128 MLP with LayerNorm and SiLU encodes each frame. A learned 128-dimensional initializer and a 128-dimensional GRUCell process the two causal frames. Two learned heads predict six mode logits and eight goal outputs (six signed margins plus two goal logits). The U-Net receives 238 conditioning values: current hidden state (128), both normalized observations (94), mode probabilities (6), predicted margins (6), predicted goal probabilities (2), and signed predicted-versus-measured goal disagreement (2). There is no argmax mode selection or mode-specific scripted command.

The diffusion network is the supplied Conditional U-Net, widths 128/256/512, kernel 5, groups 8, diffusion embedding 128. Its FiLM conditioning learns how progress affects denoising. A separate 8-to-128 action encoder and 128-dimensional GRUCell forecast progress for the 15 genuinely future slots. The shared mode/goal heads decode this forecast; it is a coarse learned progress model, **not** analytic FK, IK, or a full dynamics model.

## Mode and geometry supervision

The six modes are finishing-blue-placement/release, empty-retreat, red-approach/closing, red-attached-carry, red-place/open, and final-retreat/terminal. `semantic_targets` implements weak labels using object/TCP relations, aperture, height, goal geometry and backward red-height difference. Per-finger aperture above 0.030 m is an opening cue and below 0.028 m a held/closing cue. These values define training proxies only; they do not gate native actions.

A supported blue block and open/departing TCP indicate the empty-retreat transition; becoming closer to the buffered red indicates red acquisition. A closed hand near a lifted red gives attached-carry evidence. Red near its goal and lowering or below 0.250 m gives place/open evidence. Both measured goals plus open fingers and TCP above the red block give final-retreat evidence. A masked six-step lookahead can confirm future blue separation or red lift around label transitions. End-of-window labels have less hindsight and can be imperfect. There are no force/contact annotations, retry demonstrations, or time-index labels.

Goal supervision uses the signed world-axis XY containment margins of the rotated cube: extent along an axis is 0.02 times the sum of absolute entries in that rotation-matrix row; margin is 0.06 minus extent minus absolute centre error. Height margin is 0.011 minus absolute object-goal centre-height error. Quaternions are normalized for this geometric calculation. Three nonnegative margins define each observed goal. Regression targets are divided by the same shared positional scales; classification targets come from the unscaled signs. A near-goal airborne block has negative height margin even if its internal mode is late. Goal geometry is supervision and an observation-disagreement feature, not a constraint projected onto generated actions.

## Losses, masking, and gradient paths

The total loss is the weighted epsilon loss plus these auxiliary terms:

| Term | Weight | Trainable predictions and gradient path |
|---|---:|---|
| Six-mode cross entropy | 0.12 | Causal history beliefs and demonstrated-action progress forecasts; trains mode head, observation memory, initializer and forecast modules |
| Signed goal-margin Smooth L1 | 0.05 | Six predicted margins against geometric targets; trains shared goal head and latent representations |
| Observed-goal binary cross entropy | 0.06 | Two predicted goal logits; separate geometry belief from progress label |
| Soft ordered-tail loss | 0.025 | Squared decrease beyond 0.02 in P(mode greater than k), across adjacent predicted beliefs |
| Same-mode persistence | 0.01 | Squared probability change on adjacent same-label states |
| State-initializer consistency | 0.04 | Duplicated-current-history CE + margin loss + 0.5 goal BCE + KL to stopped-gradient two-frame belief + 0.1 normalized-hidden agreement |
| Denoised-action/progress coupling | 0.03 | Forecast CE + margin loss + 0.5 goal BCE from reconstructed clean actions; gradients reach the denoiser through x0 as well as the memory/forecast heads |

Mode/goal sequence losses have weight 1 plus 0.5 at label changes and 0.35 in the terminal class. Epsilon training has weight 1 plus 0.75 at gripper-command changes, 0.5 at mode changes, and 0.35 in the terminal class, normalized over valid elements. This emphasizes transitions without discarding any portion of the slice.

Ordering and persistence act on trainable probabilities, not observed poses. They are disabled when the relevant labels regress or observed achieved-goal geometry is lost (persistence additionally requires unchanged labels). They are **soft regularizers**, not a triangular transition matrix or an irreversible latch. All modes remain possible. Their benefit for actual disturbances is untested.

Slot zero is action t-1 and its post-action state is current t. The forecast starts with action slot one, targets future_obs slot one onward, and avoids double-counting the current state. All future targets and lookahead confirmations are masked; no target crosses a slice. Forecast supervision never consumes future observations as inputs. The clean estimate is x0=(noisy-sqrt(1-alpha_bar)*epsilon)/sqrt(alpha_bar). Only its auxiliary forecast input is clamped to [-1.25,1.25]. That coupling loss is multiplied by alpha_bar in the numerator to damp high-noise amplification; this does not change inference clipping or action decoding.

Every nonzero auxiliary acts on a trainable prediction. Measured margins and mode rules alone cannot update the model. Denoising gradients reach the causal encoder and both conditioning heads, and the extra clean-action coupling also directly reaches epsilon prediction. None of these losses is an observed-pose-only penalty masquerading as training.

## Handoff, continuation, and task completion

See `HANDOFF.json` for the API-facing interface. The policy can be invoked before blue support, during opening, during empty retreat, while approaching/closing on red, or after red lift, insofar as those observations resemble the shared training envelope. Full-scene conditioning and a learned initializer avoid the incorrect assumption that the final skill starts after blue completion. The latent vector is not passed from the predecessor and is not required by a successor.

`progress_diagnostics` optionally exposes mode probabilities, predicted goal probabilities, measured margins/flags, and their disagreement. The mandatory forward returns only epsilon; the framework is not assumed to call this diagnostic helper. These probabilities have not been calibrated, and no termination action is generated by this helper.

The supplied task succeeds when **red_at_goal AND blue_at_goal**, with full rotated XY containment and the stated height tolerance. It requires neither release nor a TCP clearance, speed threshold, or sustained hold. If commands continue, the learned data distribution includes g=+1, object settling, and upward retreat to approximately TCP z=0.300 m with nearly stationary joints. These are useful end-behavior cues, not a redefinition of success. Unexpected geometry should override confidence in a late mode at the selection/evaluation layer; this policy does not implement a recovery controller or an automatic stopping latch.

## Dependencies, budget, and limitations

Only Torch and the supplied `appl.public` numerical backbone/loss are imported. All neural modules belong to the model state_dict, including training-only forecast modules, and participate in checkpoint/EMA handling. The outer recipe remains seed 0, 20,000 updates, batch 128, horizon 16, history 2, execution 8, 100-step DDPM training/sampling, AdamW 1e-4/1e-6, cosine schedule with 500 warmup steps, gradient norm 1, and EMA 0.999, selected at the final predeclared budget.

The principal limitation is the short causal context: unlike the original aspirational long-memory model, this executable cannot remember a completed manipulation if the scene and the two supplied frames cease to reveal it. Stationary dwell/command-change instants can remain genuinely aliased. Weak phase thresholds can mislabel unfamiliar scenes; the learned forecast may exploit correlations rather than physical causation. There is no calibrated contact estimate, guaranteed invariance, hard geometric safety constraint, image encoder, IK, full FK, collision checker, universal monotone graph, retry/reset strategy, or validated recovery after a bumped object. Interface checks establish executable finite training/sampling/reload behavior, not policy success or superiority to a current-state DP. Comparisons against memoryless DP and recurrent DP without ordering, especially randomized handoff times, remain scientific evaluation work.
