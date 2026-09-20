# open_access__h03: successor-aware short-horizon diffusion selection

## Assigned hypothesis and scope

The assigned heuristic is unchanged: **"Successor-aware short-horizon diffusion selection: favor demonstrated-like chunks predicted to preserve opening and establish red pickup. Hypothesis: consequence prediction improves handoff quality beyond marginal action imitation."** Its proposed mechanism is a diffusion action proposal, a short action-conditioned dynamics ensemble, successor-readiness prediction, and consequence-sensitive candidate selection. This document describes implementation decisions and limitations, not a revision of the source heuristic.

The policy learns **all** of the assigned expanded open_access slices, not just the near-open portion: initial rest, approach and orientation adjustment, handle acquisition, pull, handle release, rising retreat, red approach/descent, finger closure and initial lift. Stops remain 475 for demo1000/1002/1003/1004/1006/1009/1010/1011, 476 for demo1001, and 474 for demo1005/1007/1008, with every start at 0. The overlaps with evacuate_red begin at original action 195 and continue to those stops. Training masks retain padding semantics and do not cross these slices. No source indices or demonstration identifiers enter the learned policy.

## Evidence read and the transition interpretation

I inspected actual original states/actions, including demo1000 at 0, 80, 150, 195, 215, 230, 240, 260, 300, 380, 420, 440, 450, 465 and 474; demo1001 at 195, 240, 440, 450 and 475; and demo1006 at 195, 240, 450 and 474. In demo1000:

- At 0, the drawer is closed, fingers are 0.040 m each and the TCP is at approximately (-0.384, 0, 0.442) m. At 80 the TCP is approaching; at 150 it is near z=0.128 m, with fingers about 0.00705 m each and close command -1.
- At 195, d=0.18630 m and vd=0.10710 m/s. TCP x is -0.32631 m, red x is -0.08354 m, and red z is 0.063 m. The robot is still pulling, not merely translating its TCP in free space.
- At 215, d=0.29700 m and vd=0.03498 m/s. At 230 d is 0.300 m, but the fingers are still around 0.0071 m each. The opening subgoal does not establish release.
- At 240, fingers are about 0.03939 m each, command is +1, d remains about 0.300 m, and TCP z is still about 0.128 m. At 260 TCP z has risen to 0.177 m; at 300 it is about 0.369 m. These are different, supported moving handoff states, not one static boundary.
- At 380/420 the open TCP moves toward and descends onto red while d stays open. At 440, TCP and red centers differ by roughly 9 mm horizontally and 1 mm vertically, but fingers remain open. At 450 the command is -1, fingers are about 0.01824 m each and red has not appreciably lifted. This is evidence of grasp establishment, not a contact sensor.
- At 465, red z=0.07290 m and TCP z=0.07379 m. At 474, red z=0.13331 m and TCP z=0.13420 m; their relative translation is nearly preserved while the arm is moving. The supplied post-slice boundary statistics put red z around 0.142-0.143 m and total finger width around 0.03649 m. The action-state at 474 and the post-action endpoint must not be conflated.

The other inspected demonstrations support the same transitions with different red/blue locations and slightly different final timing. These successful trajectories motivate local coupling models, not a safe/unsafe classifier. Fingers, geometry and correlated motion support an inferred grasp, not proof of attachment.

## Interface adaptation: what is and is not selected

The fixed interface exports only an epsilon-prediction forward call. It has no post-DDPM candidate-reranking or variable-prefix execution hook. I therefore implement **soft selection among three learned denoising refinements within each DDPM step**, rather than pretending that the framework samples and ranks K completed independent chunks.

A common U-Net proposes epsilon. Three small learned temporal heads produce paired outputs: an epsilon refinement, bounded to +/-0.25 per component, and a clean-action companion prediction in [-1,1]. Each sees the noisy chunk, common epsilon, causal condition, timestep embedding and horizon position. Training supervises both outputs and enforces the noise/clean reconstruction relation using the actual batch alpha_bar. At deployment, the companion is an approximation to the corresponding expert's denoised chunk; the forward method needs no assumed scheduler coefficients or future states. The companion is **not guaranteed to be the exact x0 recovered from that expert by the fixed scheduler**. Pairing error is an explicit limitation.

For each companion, the ensemble predicts consequences of action slots 1 through 8. Costs combine a learned demonstration-density proxy, ensemble disagreement, opening preservation, over-pull, stage-dependent relative-motion penalties and advisory readiness. A softmax with temperature 0.15 determines preferences. The selection strength is at most 0.5, decreases with ensemble variance, and is multiplied by (1-t/99)^2. The remaining weight is uniform over experts. The returned value is a convex combination of **epsilon predictions**, not a scripted or averaged native-action controller. At high noise it is nearly the uniform diffusion proposal; the learned consequence preference becomes stronger during late denoising. The fixed external DDPM process, decoding, clipping and eight-action execution remain untouched.

A second, training-time adaptation directly teaches the proposal to have demonstrated consequences through an action-conditioned model. Together these mechanisms retain the consequence-prediction/selection identity. They do not reproduce external beam search, independent complete-chain reranking, formal model-predictive control, or exact diffusion-likelihood computation. Candidates may collapse to similar refinements; no artificial action diversity is forced on this narrow successful dataset.

## Representation and learned modules

Deployment input is exactly two causal raw observations, shape [B,2,47], plus the current noisy normalized action chunk and diffusion timestep. There are no images, hidden simulator contacts, future observations, previous model predictions treated as facts, or noncausal episode phase indices.

The complete-demonstration shared normalizer from spec is registered as buffers, unchanged. Its std fields are half ranges, not empirical standard deviations. Quaternion components retain unit-component bounds. All original absolute joint/gripper targets are kept in their original shared affine encoding; the framework decodes seven Panda joint targets in radians plus one open-increasing gripper command. No per-skill or transition-specific fit is performed. Relative translation errors divide by the corresponding shared positional ranges (or the componentwise maximum of the two shared ranges). The metre-valued sigmoid widths below are semantic softness parameters for advisory phase labels, not fitted observation scales.

- Causal encoder: concatenated two normalized states and their difference (141 numbers), MLP 141 -> 256 -> 128 with SiLU and LayerNorm.
- Epsilon proposal: public Conditional U-Net with the prescribed widths 128/256/512, kernel 5, groups 8, timestep width 128 and global condition 128.
- Three candidate heads: temporal convolution 224 -> 128 -> 16, using noisy action, base epsilon, causal condition, separate sinusoidal timestep features and fixed within-chunk positions. Eight outputs refine epsilon; eight predict the paired clean companion. These positions are local chunk coordinates, not demonstration time.
- Causal imitation anchor: condition -> 256 -> 16*8, with tanh output. It is supervised on original normalized demonstration chunks. Squared distance to this mean is a unit-variance Gaussian **proxy** for demonstration likelihood, not a learned safe-set or true DDPM density.
- Three independently initialized recurrent dynamics members: two-state initializer, hidden size 128, GRUCell taking the 41-dimensional normalized physical state and 8-dimensional normalized action, residual increment head, quaternion renormalization, and a three-output readiness head. Members are trained on independently Bernoulli-bootstrapped minibatch rows (probability 0.8), not separate datasets. The rollout is autoregressive for eight actions from the actual current observation; no future state is teacher-forced into deployment.

Physical predictions cover qpos/qvel, TCP/red/blue poses, and drawer displacement/velocity. Goals are available through the initial two-state context but are not physical prediction targets. Quaternions use wxyz. Quaternion renormalization makes the model's pose representation valid but is not an action-orientation constraint, IK, kinematic consistency guarantee or invariance claim. A normalized residual increment maps back to a physical increment under the shared affine state scales. Learned TCP effects come from observed action/state pairs, not an unavailable FK provider.

## Dynamics labels, alignment and gradient paths

At observation t, action slot zero is t-1, whose successor is the current observation. Rollouts start from raw_obs[:, -1] and consume encoded_action[:, 1:9]; their labels are future_obs[:, 1:9]. Multiplying future_mask by action mask excludes unavailable endpoints and padding. This avoids predicting the past slot a second time. Dynamics learn from original demonstration actions, not reconstructed diffusion targets alone.

For each member, the state error is 5 times mean squared error on the 29 non-quaternion shared-normalized state components plus 0.2 times mean squared geodesic rotation angle divided by pi squared over TCP/red/blue. Quaternion signs are aligned; the angle is computed as 4*atan2(norm(q-r), norm(q+r)) for unit, sign-aligned quaternions, with a numerical epsilon. Losses are averaged over valid prefix positions and valid bootstrapped sequences.

Training-only future-state soft readiness labels are:

1. Opening: sigmoid((d-0.26)/0.015).
2. Release/rising retreat: opening times sigmoid((total finger width-0.065)/0.006) times sigmoid((TCP z-0.15)/0.03).
3. Red lift: opening times sigmoid((0.05-total finger width)/0.008), sigmoid((0.04-TCP/red center distance)/0.012) and sigmoid((red z-0.075)/0.012).

The action-conditioned recurrent readiness head learns these with binary cross entropy against soft labels. These labels describe local geometric readiness, not oracle contact or final whole-task completion. Predicting them throughout each short prefix allows transitional velocities and does not require stopping.

Let D be selected epsilon MSE, E the average expert epsilon MSE, C companion clean-action MSE, P paired reconstruction MSE, A anchor action MSE, S ensemble demonstrated state loss, R ensemble demonstrated readiness BCE, F generated-action future match, and G generated-action energy. The exact total is:

`loss = D + 0.10 E + 0.20 C + 0.20 P + 0.10 A + S + 0.25 R + 0.10 F + 0.02 G`.

- D and E train the backbone, encoder and epsilon refinements against actual sampled noise. All full-horizon action losses use the original action mask.
- C trains the clean companions against encoded original actions. P penalizes `sqrt(alpha)*companion + sqrt(1-alpha)*expert_epsilon - noisy_action`; this ties outcome-scored companions to denoising candidates without inventing an inference scheduler.
- A trains the causal likelihood proxy. S and R train the recurrent forward models and readiness heads on actual future labels. There is no observed-pose-only penalty mislabeled as a trainable prior.
- For F and G, recover actual training x0 from the selected epsilon using batch alpha_bar and clamp to normalized bounds. Roll this x0 prefix through **parameter-detached** ensemble members via functional_call, retaining derivatives with respect to actions. F matches future state labels plus 0.25 readiness BCE. G evaluates the same selection energy as inference, with the anchor detached. Both are weighted per sample by alpha_bar squared to suppress high-noise amplification. Gradients flow through predicted consequences -> generated actions -> selected epsilon -> proposal parameters; these losses cannot train the forward models to invent desirable effects. Input clipping can suppress gradients outside action bounds, which is deliberate.
- Selection weights inside forward are detached. Main diffusion gradients optimize the selected experts but cannot directly manipulate the scoring ensemble through the selector. Ensemble and anchor parameters receive their supervised losses. All heads and ensemble parameters are included in the same optimizer and EMA checkpoint.

## Stage-conditioned consequence energy

The scoring horizon is the next eight actions, consistent with the fixed executed prefix. No long-range planning or prefix-length choice is claimed.

A soft pull gate requires closed fingers, TCP near z=0.128 m and y=0, and TCP/red separation above roughly 0.12 m. It corresponds to the measured handle geometry. The penalty for changes in red x + d and red y/z models red travelling with the drawer during pulling. It is multiplied by both current and predicted pull gates, deactivating as fingers open or the TCP leaves that geometry. Closed-handle lateral motion is separately discouraged. This does not reward carrying red while still attached to the handle.

A different soft pickup gate requires closed fingers, small TCP/red separation and red height around/above the initial lift region. It penalizes changes in red-minus-TCP translation after likely pickup. Opening loss is emphasized once d is around/above 0.18 m; d beyond 0.300 m receives an over-pull cost. These are predicted-outcome penalties, not hard physical feasibility constraints. The small readiness deficit cost is stage-conditioned: opening during pulling, retreat after full opening when far from red, and lift when near red. Imitation provides the full initial approach, waiting/release and intermediate route rather than replacing them with a geometric controller. The local readiness weight is deliberately small because demonstrated closure/settling can precede observable lift by multiple prefixes.

Energy weights are 2 for normalized action-anchor distance, 2 for selected-state ensemble variance, 2 for opening loss, 2 for over-pull, 4 for pull coupling, 2 for premature lateral movement, 4 for pickup coupling and 0.05 for readiness deficits. Uncertainty uses normalized fingers, TCP/red translations, d and vd. These predeclared weights have not been performance-tuned. The gates and advisory costs use task geometry; their use neither certifies contact nor changes task success criteria.

## Handoff and causal takeover

Causal qpos/qvel and the two observations distinguish stationary closure from rising retreat; drawer position AND velocity distinguish articulation from free TCP motion; finger aperture and TCP/red relation distinguish handle release, red approach and likely red attachment. Orientation is also retained. This information is present when entering halfway through the overlap, even without an elapsed phase clock or a hidden recurrent state carried over from another policy. Every invocation initializes its models from actual history.

Continue locally while the requested portion of opening/release/acquisition remains supported. Transfer to evacuate_red with the **actual most recent observation/history**, not predicted rollout endpoints or predicted readiness as fact. Supported exits include d around 0.186 m and still increasing if the successor will finish opening, release near d=0.300 m, upward retreat, approach, or initial red lift near the expanded endpoint. It is not necessary to drive velocity to zero. Red lift and small TCP/red relative motion are stronger evidence of an acquisition handoff than a close command alone. The full expanded endpoint is useful but not a mandatory switching threshold; the API selects invocation duration.

HANDOFF.json details continuation, evidence and failure signatures. The three whole-task goals remain the supplied simultaneous drawer-open/red-on-pad/blue-inside contract. This opening skill is not claimed to finish the whole task. There is no extra release, speed, sustained-hold or TCP-clearance condition added to that contract, and no omitted segment is justified by an invented completion condition.

## Applicability, limitations and testable claim

The model is local learned diffusion with an outcome-selection prior. It is applicable to accurate state observations and demonstrated dynamics around successful continuations, including the full training transitions. Jammed drawers, missed handles, failed red grasps, unexpected contacts and action/state counterfactuals far from demonstrations are untested. The dynamics can ignore actions or share systematic bias; bootstrap disagreement is not calibrated epistemic uncertainty and low disagreement is not safety evidence. Learned rotations need not be kinematically reachable. There are no force, collision, contact, image, IK or analytic robot-kinematics modules. Nothing guarantees drawer preservation, grasp stability, action equivariance or a safe state.

Other limitations are companion/epsilon mismatch, unit-variance likelihood approximation, candidate collapse, soft averaging of refinements, high-noise x0 clipping, small-dataset predictive bias and repeated within-step rollout cost. Forward scoring cannot know a slice-end future mask at deployment; the API should hand off instead of expecting extrapolation beyond initial red lift. Readiness supervision is geometric and successful-only, not a calibrated switching or failure classifier.

Training is the supplied seed-0, 20,000-update, batch-128 recipe, AdamW 1e-4/1e-6, cosine decay with 500 warmup updates, gradient norm 1, EMA 0.999, horizon 16, history 2, execution 8 and fixed 100-step DDPM. Select the last EMA at budget. Interface checking is not evidence of task performance. The prediction is fewer over-pulls and better red-ready transitions than the same proposal without consequence preference. A scientifically informative comparison should disable ranking and generated-action consequence auxiliaries with the same data and budget. Degraded pickup, low-likelihood selections, unnecessary pauses, or no advantage over the proposal would weaken the hypothesis. No such evaluation or benefit is claimed during this design session.
