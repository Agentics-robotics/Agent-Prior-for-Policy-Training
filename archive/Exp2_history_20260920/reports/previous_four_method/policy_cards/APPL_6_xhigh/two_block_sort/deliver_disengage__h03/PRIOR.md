# Prior: event progress with elastic duration

Policy: `deliver_disengage__h03`. Skill: `deliver_disengage`. Assigned heuristic: 3, M1_v2.

## Original hypothesis and scope

The assigned hypothesis is that manipulation event order is more stable than event timestamps, and that similar stationary-looking poses can have different intended continuations. An event-progress diffusion policy with a learned duration model should continue more consistently across supported handoff timings than a fixed-clock policy. The original heuristic is not modified by this implementation document. The choices and adaptations below are an executable approximation of that hypothesis, not additional claims about the demonstrations.

Training retains **all 24 expanded slices exactly as assigned**, including open low approach, closing, lifting, early transport, lowering, release lag, dwell, retreat, high next approach, and terminal settling. It does not extract only a central delivery interval. For example, demo10000 includes [80,425) and [470,773); demo10001 includes [78,417) and [463,765); demo10011 includes [84,435) and [480,777). Other assigned trajectories retain their supplied ranges. End-state observations are context, not invented additional actions. No source index, trajectory identity, episode clock, or fixed skill invocation schedule enters the network.

## Observed support and handoff interpretation

Actual observations were read from demo10000, demo10001 and demo10011, including incoming transitions and outgoing endpoints. They support an ordered-transition prior, not a proof of contact or learned success:

- demo10000/80 is an open descending approach: TCP z approximately 0.066 m, red center z approximately 0.020 m, each finger approximately 0.040 m, and nonzero arm velocity. At /106 the low arm is almost settled, with open fingers. At /107 the command changes to -1 while fingers are still open; by /110 fingers are approximately 0.0183 m. A finger command and a measured grasp aperture are different signals.
- At /150 red and TCP rise together to approximately 0.191 m. At /170 they are around 0.279 m and nearly stationary before transport; /185 has early horizontal motion. This is why invocation cannot force the transport phase or demand zero qvel.
- At /280 red is still grasped and descending near the goal, z approximately 0.052 m. At /290 it is around 0.0226 m with closed fingers. The /296 opening command precedes the actual widening; /305 has fingers approximately 0.03976 m and red resting at approximately 0.020 m while the TCP is still low. At /315 retreat has begun; /350 is high empty retreat. Goal membership can precede physical disengagement.
- The second occurrence includes open blue approach /470, command closure /504, aperture response /505-/506, and an elevated dwell /570. Opening command /691 has fingers still approximately 0.0183 m, /692 approximately 0.0269 m, /693 approximately 0.0320 m, and /700 approximately 0.03976 m. /720 is open retreat. This is not instantaneous binary contact.
- demo10001/417 is a moving high next approach, TCP z approximately 0.268 m with red settled and fingers approximately 0.040 m; nonzero arm velocities include about 0.179 rad/s. demo10011/435 is also high next-approach context, around 0.266 m. Neither is an empty stationary prerequisite for takeover.
- demo10011/777 has both blocks at approximately 0.020 m, TCP z approximately 0.299 m, open fingers and almost stationary joints. This supports terminal settling context but does not add a settling requirement to task success.

The measured start-width range is about 0.0799993 to 0.0799997 m total aperture and the expanded endpoint widths are about 0.0799991 to 0.0799999 m. Incoming overlaps extend across approach-close-lift and early horizontal motion. The red outgoing overlap begins while descending and still holding the block, then includes release and next approach. These measurements are evidence, not hard runtime applicability gates.

## Causal inputs and explicit interface adaptations

The only deployment inputs are noisy normalized actions, diffusion timestep, and two raw observations `[B,2,47]`. State is world-frame metres and wxyz quaternions, with seven arm positions and two finger positions in qpos, qvel, TCP pose, red/blue poses, and their goals. Drawer channels are compatibility zeros. There are no images, executed command history, explicit carried/next-role IDs, long history, or persistent per-environment recurrent state in this interface.

Accordingly:

1. A two-step GRU learns an observation-based initializer on every call. Its hidden state is not retained across calls or DDPM iterations. Normalized observation differences and raw normalized levels expose motion and aperture changes; measured qvel remains available. Fifteen percent of training examples duplicate the latest state to train a truncated-history initializer. This does not invent physics or rescale qvel. It does not guarantee calibrated uncertainty under history loss.
2. A learned two-way role head is supervised by training-only geometry. It preserves the red role through open high next approach, distinguishes low blue approach and blue carrying through TCP/blue proximity and aperture, and uses both placed objects for terminal blue context. It does not switch to blue merely when the still-grasped red block first satisfies a placement proxy. The model consequently supports the observed red-then-blue order; it cannot honor an arbitrary external role instruction absent from its input. The caller should still transfer role IDs to the successor and retain them in orchestration.
3. A learned initial open-command probability and command-switch hazards replace unavailable past executed commands. Commands in the action labels supervise these predictions; they are never deployment inputs or teacher-forced conditioning.
4. The batch interface supplies only local masked futures, not a full-trajectory offline preprocessing hook. We therefore do not claim complete offline phase segmentation, complete-trajectory resampling, or exact physical full-phase duration labels. Local training-only annotations use command sign, actual finger aperture, TCP/object heights and displacement, and object-goal proximity. Adjacent valid labels receive 0.8/0.1/0.1 temporal smoothing. Events are approach, closing, lifting, transport, lowering, releasing, empty retreat, and next approach/terminal settling. The last event absorbs terminal continuation; it is not task-success detection.
5. Within-event progress labels are **geometric/aperture proxies**, not elapsed-time fractions: vertical approach gap, aperture closure/opening, lifted height, horizontal goal distance, lowering height, retreat height, or high approach distance to blue. Characteristic annotation lengths (roughly 0.25-0.28 m) and aperture endpoints (0.0183 and 0.040 m) are physically motivated label definitions supported by the read observations, not refitted action/observation normalizers. Annotation placement proxies do not implement the rotated-containment task predicate.
6. Remaining-time supervision is exact only relative to the local annotated advancing event: its first observed future boundary supplies seconds at 0.05 s per sample. Otherwise the label is right-censored at the last valid local observation. Nominal duration targets use positive within-event progress rates, clipped to 0.15-12 s for numerical stability. They are local rate-derived effective durations, not recovered expert phase clocks. Long dwell duration is weakly identifiable; there is no hard-coded dwell timer.

## Learned architecture and progress/time mapping

Every learned action remains an eight-channel epsilon-prediction DDPM output. The framework's original-time noising, 100-step DDPM sampler, action decoding and EMA are unchanged.

The encoder normalizes with the **single supplied complete-original-data normalizer** stored as buffers. It consumes normalized levels and differences using a 128-dimensional step MLP and GRU, then a 128-dimensional context MLP with a flattened-history skip. Quaternions keep their common unit-component bounds. There is no per-skill scale fit, clipping of observations, geometric invariance, or frame conversion.

The belief head predicts phase probabilities, role probabilities, eight progress values, positive nominal durations, positive remaining durations, and bounded progress/log-remaining uncertainty scales. Phase entropy also represents ambiguity. Positivity uses softplus; progress lies in [0.005,0.995]. Remaining time is separate from geometric progress, so an almost fully closed gripper can still predict a nonzero closing dwell.

For each possible starting phase k, a differentiable ordered clock places its end at predicted remaining time R_k and its start at `-u_k R_k/(1-u_k)`. Other events use learned positive nominal durations. Soft boundaries have 0.035 s width. The resulting eight paths are mixed by the phase posterior rather than choosing an event by argmax. This produces per-slot phase probabilities and progress at offsets -0.05 to +0.70 s, respecting the interface's slots t-1 through t+14. Subsequent calls infer fresh clocks from feedback, not from an episode counter.

A context-conditioned head generates six arm-target knots for each of eight events. These are learned offsets around the measured current joints, expressed in the common encoded absolute-action coordinates. Linear within-event interpolation and posterior mixing yield an original-time clean-action reference. Knots are generated anew from the observations; they are not stored demonstrations or scripted trajectories.

For diffusion, a local coordinate is the expected event index plus progress. It is normalized over the horizon and blended with a 10 percent physical-time coordinate to avoid grid collapse at a dwell. The noisy action sequence is differentiably resampled onto a uniform local-progress grid. The public temporal U-Net (128/256/512 widths) predicts grid features/epsilon, conditioned on the belief, decoded action reference, phase path and coordinates (450 dimensions). Its output is pulled back to original timestamps. Searchsorted chooses interpolation intervals without gradients; interpolation weights and coordinates retain gradients. An original-time convolutional residual sees raw noisy actions, pulled-back predictions, reference, coordinates, context and diffusion timestep. It preserves information that nonuniform resampling can discard.

This is a **progress-domain denoiser representation**, not a new DDPM on a falsely assumed independent progress-grid noise distribution. Resampling creates correlated internal noise; only the final original-time epsilon is scored against the original sampled noise. The raw-time residual prevents pretending this interpolation is invertible.

Gripper behavior is separate: a learned two-state persistence recurrence uses closing/opening hazards and an initial open probability. It produces a command reference at physical timestamps. A separate direct-time convolutional epsilon head predicts the final gripper score; the final score is not interpolated across the phase grid and is not overwritten with a binary script. The auxiliary command targets are -1/+1, but final DDPM outputs can be continuous within [-1,1]; exact binary persistence is not a hard constraint. Conditional switch supervision gives observed switches sixfold weight and trains only the hazard eligible under the previous labeled command. The previous label is used to weight the loss, not fed to inference.

## Objective and gradient paths

Total loss is masked original-time epsilon MSE plus the following trainable prior terms. Weights are fixed in pipeline.json and code:

| Term | Weight | Prediction receiving gradients |
|---|---:|---|
| Current soft phase cross-entropy | 0.08 | belief and encoder |
| Decoded phase-path cross-entropy | 0.04 | phase, remaining/nominal durations, progress and encoder |
| Role cross-entropy | 0.04 | role head and encoder |
| Progress heteroscedastic Gaussian NLL, shifted by log minimum scale | 0.015 | progress and uncertainty heads |
| Decoded progress MSE | 0.08 | progress/time decoder and belief |
| Log remaining-time NLL at observed boundary or one-sided log censoring penalty | 0.02 | remaining-duration and uncertainty heads; uncertainty trained on observed boundaries |
| Nominal log-duration rate regression | 0.01 | nominal duration head |
| Time-decoded seven-joint action-reference MSE | 0.35 | knot generator, progress, phase, durations, encoder |
| Persistent open-command BCE | 0.12 | initial-command and hazard heads plus belief |
| Eligible closing/opening switch BCE | 0.035 | hazard head and belief |
| Phase CE on future causal pairs ending at slots 8 and 15 | 0.025 | same belief encoder, not a deployed future input |
| Soft ordered-transition penalty | 0.005 | independently predicted expected phases at current and two later causal pairs |
| Within-event knot second-difference MSE | 0.0005 | trainable arm knots |

The order penalty allows 0.15 expected-event regression and up to two events of advancement over each short supervised interval. It is a soft regularizer on predictions, not a penalty on fixed observed poses and not a hard runtime prohibition on regression. There is no phase reset across a slice's current-object transition, and no next-pickup training data is fabricated beyond an endpoint.

Action supervision uses batch.mask. Annotation, progress, duration and future-pair supervision also require batch.future_mask; paired terms require both endpoints valid. Padding never becomes an observed event or an assumed final boundary. Native action labels use radians, and encoded targets use the same common range transform as DDPM. No high-noise x0 inversion is used for auxiliary training, avoiding its amplification. Diffusion gradients also reach the belief, clocks, knots and persistence head through denoiser conditioning and resampling. All prior terms involve learned predictions; none is a constant observation-only penalty.

## Applicability, continuation and completion

Use this model when recent observed motion plausibly follows the demonstrated event order. Entry can be open and low, just before command closure, partially lifted, already elevated, descending, opening or retreating. It is not initialized to transport merely because its name is deliver_disengage. Two causal observations, actual fingers and TCP/object relation are the information that supports takeover partway through an overlap. There is irreducible ambiguity between identical dwell states when command history is absent.

During nonterminal outgoing overlap, continue for smooth predecessor completion or transfer to an acquire model that supports the still-grasped descent/release portion. Transfer live state history and orchestration roles, not just a phase index. Useful later handoff is an open empty retreat/high next approach with the delivered block remaining on its pad, even if arm motion persists. Terminal continuation learns open retreat and settling; stopping may occur earlier if the supplied contract is satisfied.

The task contract remains exactly: red_at_goal AND blue_at_goal, world coordinates, each 0.04 m block fully contained in its separate 0.12 m square pad under rotated XY containment, target center z 0.020 m within 0.011 m, one observation. Red goal is (-0.18,-0.25,0.02) m and blue goal is (-0.18,+0.25,0.02) m. There is **no additional gripper-release, clearance, velocity or sustained-hold requirement**. This model does not compute a certified contract result; the caller evaluates the supplied contract separately. Release/retreat are skill continuation and handoff cues only.

## Dependencies, limitations and scientific tests

Dependencies are torch, math and the supplied appl.public numerical backbone/helpers. No FK, IK, contact simulator, image encoder, file access, replay, external controller or hard geometric constraint is used. Role assignment, geometry labels, and ordered clocks encode the demonstrated red-then-blue support, not general recovery after a missed grasp. Bounds in the annotation are soft-supervision design choices, not deployment safety checks.

A two-observation recurrent model may infer expert timing from tiny velocity transients; successful demonstrations cannot establish robustness to stalls, retry sequences, altered controller rates, arbitrary temporal warps, or calibrated timeout decisions. Local censoring supplies limited information about long durations. Soft order can hide failure rather than repair it, and duration/progress uncertainties are not asserted calibrated. No claim is made that task geometry is guaranteed by joint-space actions.

The distinctive hypothesis needs trajectory-held-out tests against a fixed-clock DP, a history-only unaligned encoder, and an unwarped phase-conditioned DP. Evaluate multiple measured overlap takeover times, including moving entries and release lag. Benign temporal-resampling tests must rescale qvel and duration labels; this implementation does not fabricate such training data. These are proposed tests, not executed ablations or measured performance claims. Interface validation consists only of the framework's two-update, sampling and EMA reload checks. Final training uses the predeclared 20,000-update budget, seed 0, batch 128, AdamW 1e-4/1e-6, 500-step warmup/cosine, EMA 0.999 and last EMA selection; no design-time rollout performance is available.
