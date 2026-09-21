# Event-aligned diffusion prior: transfer_blue__h03

## Assigned research hypothesis (unchanged)

Heuristic 3 states: "Align actions to observed manipulation events rather than a trajectory clock: learn persistent phases and transition hazards driven by release, approach, closure and lift evidence. Hypothesis: event-conditioned diffusion tolerates different dwell and contact timings without shifting the whole action sequence."

This implementation preserves that event-alignment identity, not a rigid-body dynamics or passive-object disturbance prior. The original heuristic is not modified. The engineering adaptations below are separate from that hypothesis. The model is a learned action diffusion policy throughout; no annotated event chooses an action at deployment.

## Scope and full expanded segmentation

The policy learns the entire supplied slice, not just blue carrying. It completes incoming red descent/setdown/release when necessary, retreats and acquires blue, lifts/transports/sets down/releases blue at the initially red region, then retreats and approaches/closes/lifts buffered red. In particular, the outgoing overlap is part of this policy's training objective rather than an early termination exclusion.

The unchanged original-source ranges are half-open:

| Demonstration | Expanded slice | Incoming overlap with buffer_red | Outgoing overlap with finish_red |
|---|---|---|---|
| demo10100 | [240,850) | [240,490) | [600,850) |
| demo10101 | [242,847) | [242,487) | [605,847) |
| demo10102 | [240,843) | [240,483) | [602,843) |
| demo10103 | [244,848) | [244,488) | [606,848) |
| demo10104 | [238,839) | [238,481) | [598,839) |
| demo10105 | [240,838) | [240,480) | [599,838) |
| demo10106 | [241,842) | [241,484) | [601,842) |
| demo10107 | [246,855) | [246,493) | [613,855) |
| demo10108 | [239,841) | [239,482) | [600,841) |
| demo10109 | [243,846) | [243,486) | [604,846) |
| demo10110 | [245,849) | [245,489) | [607,849) |
| demo10111 | [242,844) | [242,485) | [603,844) |

Nothing in the source uses these indices to produce actions or event labels. They document scope only. Training uses the repository's random windows and padding masks across all these phases.

## Causal representation and learned architecture

Inputs are exactly two causal 47-dimensional world-frame states. The model uses qpos, measured qvel, finger aperture, TCP pose, red/blue pose and both goals. The drawer compatibility channels remain zero. There are no images, trajectory IDs, global action indices, invocation counters or time-to-go inputs. The diffusion noise timestep is used only by the DDPM backbone; it is not a manipulation clock.

All raw observations use the supplied shared full-demonstration mean and half-range buffers. Absolute seven-joint-plus-gripper action normalization and decoding remain the repository's shared affine transform. No per-skill fit or empirical standard deviation is introduced. Four world-frame relative vectors (red minus TCP, blue minus TCP, red goal minus red, blue goal minus blue) are divided by the per-axis maximum of the shared TCP/red/blue position half ranges. Goals retain their original channels as well. This is a relative feature representation, not rigid-motion invariance or action equivariance.

A frame contains 47 normalized observations, 12 relative coordinates and 18 backward derivative features. The latter are normalized qpos and TCP/red/blue position differences divided by the true 0.05-second interval and passed through tanh. The older frame has zero derivative features because no earlier state is supplied. This saturation is confined to engineered derivative features; original qvel is preserved under its shared normalization. No action or velocity time warping occurs.

A 77-to-192-to-128 MLP with LayerNorm/Mish encodes both frames. The older embedding initializes a 128-dimensional GRUCell, which updates with the newer embedding. The older state has a learned nine-mode posterior, not a hard-coded phase-zero initialization. An emission head, nine by fourteen hazard head and nine readiness heads are shared across windows. The nine modes are:

0. Red held descent and setdown.
1. Red opening, retreat and transfer toward blue.
2. Blue open approach.
3. Blue closure and initial lift.
4. Blue elevated transport.
5. Blue goal alignment, descent and held setdown.
6. Blue opening, retreat and transfer toward red.
7. Red open approach.
8. Red closure and initial lift.

Opening/retreat and early lateral approach are merged where their distinction is not reliably observed. Closure and initial lift are also merged rather than assuming exact contact timestamps.

Let p_old be the older posterior and h_old(k,1) its one-step exit hazard. Probability flows from k to k+1 according to p_old(k) h_old(k,1), with the last phase absorbing within this slice. The propagated distribution is mixed with 10% uniform reset mass. Current belief is softmax(current emission logits + 0.5 log propagated prior). This is a soft ordered prior, not a hard automaton; new observations can correct a mistaken phase or initialize partway through a transition.

The current hazard head predicts fourteen distinct conditional exit probabilities per event, rather than a constant geometric dwell parameter. Survival is the product of one minus these hazards. Its posterior-weighted cumulative exit probabilities at 1, 4, 8 and 14 real steps, posterior entropy, learned readiness, mode probabilities and a 32-dimensional expected event embedding join the GRU state and the flattened normalized observation history. A 269-to-256-to-256 conditioner supplies FiLM context to the public Conditional U-Net. The U-Net predicts epsilon for the original 16 by 8 absolute-action chunk using the fixed 128/256/512 widths. Direct observation conditioning prevents phase uncertainty from becoming a hard action gate.

Forward is deterministic given its inputs and has no mutable recurrent cache. Repeated DDPM evaluations do not advance the event state. The optional event_belief(raw_history) diagnostic returns mode probability, normalized entropy, residual exit CDF and readiness. The required forward returns only epsilon; the fixed sampler does not automatically transport that diagnostic to another policy. A successor can always infer its own belief from the causal observations.

## Training annotations, alignment and losses

Weak event annotations are made only inside compute_loss from measured geometry and demonstrated gripper commands. Positive command means opening. Closed red versus closed blue is assigned by which object is closer to TCP. Blue being near its own goal distinguishes outgoing red acquisition from incoming red setdown. A 0.045 m planar blue-goal neighborhood and 0.02 m vertical support neighborhood are annotation conventions, NOT the task success predicate. Open approach is distinguished from preceding open retreat by a 0.06 m planar TCP-object neighborhood. Elevated blue transport begins above goal height plus 0.22 m or once blue y is below red-goal y minus 0.045 m; goal alignment takes precedence. These coarse separators reflect the inspected event order. Neither this function nor any thresholds appear in deployment forward or action sampling.

Action/state indexing is explicit: at current state t, action slot 0 is t-1 and slot 1 is t. future_obs slot j is state t+j. Hence future state j is paired with action j+1 for intention labels, and only future slots 0 through 14 have a corresponding next-command label. The future endpoint at slot 15 is not incorrectly paired with action 15. All future labels and shifted-history auxiliaries are masked with both state and command validity; slice boundaries are never crossed.

The differentiable objective is:

L = L_epsilon + 0.15 L_mode + 0.08 L_hazard + 0.05 L_proximity + 0.02 L_order.

* L_epsilon: masked mean squared epsilon error on all eight action channels and all valid real-time slots. It trains the U-Net, condition projection, recurrent encoder, posterior and hazard/readiness conditioning paths.
* L_mode: label-smoothed cross entropy (smoothing 0.03), comprising 0.5 current posterior classification, 0.25 older emission classification and 0.25 next-state posterior classification. For the latter, a separate training-only encoder call uses [current state, future state t+1]. Future information never enters the current denoising condition. This trains emissions, initialization, recurrence and the learned transition prior.
* L_hazard: discrete survival negative log likelihood. At each lag 1 through 14 the target is whether the future label has advanced beyond the current event. The risk set includes slots only through the first observed advance, and valid earlier slots of censored windows. A sample with no visible transition is a right-censored negative sequence, not a claim about the whole event duration. Hazards are selected by the training event label for supervised loss only; conditioning always uses predicted belief. This loss trains the hazard head and encoder.
* L_proximity: squared error of the selected trainable readiness prediction against max over visible advancing lags l of (1-l/15), or zero if a complete fourteen-step window contains no advance. Truncated negative windows are excluded. This is local boundary proximity, not exact elapsed-event progress. Its gradients train the readiness head and encoder.
* L_order: predicted probability mass assigned to backward or multi-event jumps across adjacent observations, averaged for older/current and current/next beliefs. Staying and advancing by one are allowed. This is a soft penalty on trainable probabilities, not a constant computed from observed poses. It is intentionally weak and does not forbid a reset.

Current minibatch mode counts define inverse-frequency sample weights B_valid/(9 count), clamped to [0.33,3]. These weights rebalance diffusion and auxiliary losses, with each reduction normalized by its valid weight sum. This gives rare release/acquisition phases more influence without discarding the long approach/carry portions. It is a bounded approximation to event-balanced sampling, not an exact uniform sampler or a whole-dataset frequency estimate.

All labels and balancing weights are detached. Every nonzero auxiliary term nevertheless depends on a trainable prediction and supplies model gradients. No pose-only penalty is misrepresented as a learning signal. There is no auxiliary IK, dynamics constraint, passive-object penalty or scripted gripper controller.

## Explicit interface adaptations

A full semi-Markov model with elapsed event age and normalized progress s=(t-start)/(end-start) would require complete event boundaries or persistent long histories. Neither is supplied to this compute_loss/forward interface. This package therefore implements a finite-context approximation: learned current-state initialization, one observed belief transition, state-dependent nongeometric residual-duration hazards, and future-derived local boundary proximity. It does NOT maintain persistent belief across invocations, infer exact normalized whole-event progress, or implement an exact duration-augmented semi-Markov likelihood. The event-alignment hypothesis remains testable, but this adaptation has less temporal information than the original suggested encoder. Identical two-frame dwell observations can remain ambiguous; phase uncertainty cannot manufacture missing event age. The ordered regularizer may also discourage legitimate recovery despite the reset floor.

The repository controls sampling, so no custom uniform-across-events sampler is claimed. Loss weighting is the executable substitute. Every window retains actual 20 Hz timing, horizon 16, execution 8 and measured qvel units. Fixed training uses 20,000 updates, batch 128, AdamW 1e-4/1e-6, cosine schedule with 500 warmup updates, gradient clipping 1, EMA 0.999, and 100-step training/inference DDPM with clipped clean samples. Selection is last EMA at the declared budget, seed 0; no success-based tuning is available.

## Handoff evidence and use

Measured starts show held red descending, per-finger positions near 0.0183/0.0182 m and combined aperture about 0.0365 m, not an initially open blue approach. demo10100:240 has red/TCP heights about 0.163 m and command -1; at 265 red is about 0.0228 m and still commanded closed; at 275 both fingers are about 0.0396 m and red is supported near z=0.020 m. At 320 TCP is about 0.301 m while red remains down, demonstrating learned retreat rather than continued red carrying.

At demo10100:455 blue is already held with fingers near 0.0183 m and command -1. At demo10107:456 TCP is similarly low at blue but fingers are about 0.0400 m and command is still +1; closure is evident at 460. demo10111:448 is already closed. Thus universal timestamp cuts are inappropriate. Aperture, measured qvel, backward changes, TCP-object geometry and object support/lift evidence allow the learned posterior to enter mid-transition. They are evidence of a likely grasp, not tactile confirmation.

In demo10100:600, blue/TCP heights are about 0.209 m during goal descent; at 630 blue is about 0.0219 m with command -1. At 640 fingers are opening (about 0.0382 m) with blue at z=0.020 m; at 690 TCP has retreated to about 0.291 m. Blue location alone would have suggested completion too early for the nominal red-lift handoff. At 800 red approach is still open; at 810 the command has just changed to -1 while fingers remain physically open; at 820 fingers are near 0.0183 m with red still low. At 849 red/TCP are rising near 0.203 m while blue remains supported at its goal. demo10107:854 shows red height about 0.181 m; demo10111:843 about 0.170 m. The measured endpoint range roughly 0.17-0.21 m is moving, not a settled joint target.

Continue through release/retreat/red reacquisition if those phases are unfinished and the observed sequence remains supported. A useful nominal exit carries closed red upward with blue undisturbed at its negative-y goal; pass the actual causal state history and velocity, not a reset-to-rest command. A predecessor/successor may share the overlap earlier by re-estimating event state, but no model automatically calls another model. HANDOFF.json states the API guidance and failure signatures.

Task success remains solely the supplied completion contract: red_at_goal AND blue_at_goal, using rotated full-XY containment and z tolerance. Goal centers are red [-0.35,0.20,0.02] and blue [-0.35,-0.20,0.02] m. There is no extra release, TCP clearance, velocity or sustained-hold success requirement. The nominal red-lift exit is a skill-continuity cue, not a replacement for that contract.

## Dependencies, uncertainty and scientific limits

Runtime dependencies are Torch, math and appl.public.DiffusionBackbone. There is no external simulator, kinematics library, image encoder, network access, file reading, action replay or stored demonstration lookup. Only weights, shared normalizer buffers and causal observations enter deployment.

All demonstrations share an ordered planner-generated sequence with modest timing variation. Pseudo-label boundaries and short histories may not resolve dwell intention. Goal-relative phase labels are not a proof of contact, rotated containment or safe grasp. Recovery from slips, repeated failed grasps, phase-skipping shortcuts and large layout changes is unestablished. The reset floor allows belief correction but is not evidence of recovery competence. Blue disturbance is observed as state, not explicitly constrained. Event posterior entropy is an uncertainty summary, not a calibrated safety certificate.

The scientific prediction is better transition timing robustness than clock-conditioned or memoryless diffusion. Relevant later comparisons are the same model without mode/hazard supervision and an encoder-capacity-matched model without event conditioning. No performance or invariance result is claimed from this implementation session; interface smoke testing checks executable gradients, sampling and reload, not task success.
