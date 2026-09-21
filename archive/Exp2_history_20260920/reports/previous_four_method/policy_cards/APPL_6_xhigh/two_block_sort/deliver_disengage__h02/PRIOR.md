# PRIOR: deliver_disengage__h02

## Identity and research hypothesis

Assigned skill: **deliver_disengage**, heuristic **2**, M1_v2. The original heuristic is not modified. Its statement is: "Optimize the actual containment contract, not an exact replay point: separate geometric task completion from physical disengagement and preserve previously placed blocks. Hypothesis: outcome-aware diffusion favors actions that satisfy the rotated-XY and height tolerances while retaining the release/retreat interface taught by the demonstrations."

This package implements a learned, state-conditioned action diffusion policy with contract-aware training. It is neither joint replay nor a scripted geometric controller. Native outputs remain seven absolute Panda joint targets and one gripper command. All 24 assigned expanded slices are used without trimming acquisition or outgoing overlaps, relabelling boundaries, or changing the training budget. Geometry is in world metres, with unit wxyz quaternions. No images, IK, external forward kinematics, simulator, contact labels or new action-outcome data are used.

## Explicit adaptation of candidate D2

The proposed research pipeline includes online candidate reranking or bounded rollout-energy descent during sampling. The supplied interface fixes the DDPM sampler and exposes an epsilon-prediction forward hook, not a post-sample selection hook. This implementation deliberately uses **training-time amortized outcome guidance**, not an undocumented replacement sampler:

1. Train a causal state representation, per-slot placement-window probabilities and separate current/future readiness probabilities.
2. Train an action-conditioned recurrent forecast on actual synchronized future poses and apertures.
3. Differentiate analytic containment and protected-object energies through a detached-parameter copy of that forecast into the diffusion model's estimated clean actions, with low-noise, locality, calibration and event gates.
4. At deployment use the learned denoiser directly, explicitly conditioned on causal placement-window and current readiness probabilities as well as history features. There is no online autograd, reranking, rollout optimization, or scheduler reconstruction in `forward`.

Thus this tests the contract-aware action-learning part of D2, not its full online sampler intervention. Predicted outcome margins never decide observed task success. No persistent live-versus-forecast disagreement monitor is implemented. The trust term is toward demonstrated actions during training, rather than toward an unguided sampled chunk at inference. These are implementation choices and limitations, not changes to the source heuristic or the completion contract.

## Observation representation and roles

`forward(noisy_action, timestep, raw_history)` receives only two causal 47-dimensional observations. The representation contains:

- Both complete observations normalized with the supplied shared original-demonstration mean and half-range, plus their normalized difference (141 features).
- Current red/blue goal-minus-object and TCP-minus-object positions (12 features), and the six analytic live contract margins.
- Two observed protection-support flags, a two-component nominated-object one-hot, and normalized mean finger aperture (5 features).

The resulting 164 features enter a 256-wide MLP with LayerNorm and SiLU. Relative spatial features use the per-axis maximum of the supplied TCP/red/blue position half-ranges. This is a deterministic combination of the shared scales, not a new data fit. Quaternion component normalization remains the shared unit-component convention. Drawer compatibility channels remain part of the raw input but have no drawer-specific interpretation.

There are no explicit role IDs in the supplied observations. The model derives a nomination from TCP/object proximity, preferring an object not supported as released and protected. Both protected gives ID -1; red and blue IDs are 0 and 1. This is a causal role descriptor, not a fixed red-then-blue clock, an action selector, or a claim to measure grasp contact. While red is contained but held, its small TCP separation and narrow fingers leave it unprotected; after opening/separation, blue can become the nomination. The full raw state and motion features remain available so a nomination switch does not force an immediate low approach.

Protection support requires the LIVE full geometric predicate, less than 0.003 m object displacement between the two observations, and either total finger width above 0.065 m or TCP/object distance above 0.08 m. The separation alternative protects red even when the fingers have subsequently closed on blue. These thresholds label a conservative demonstrated release context; they are not rejection thresholds, contact proof, or success requirements.

## Learned modules and alignment

The action denoiser is the supplied Conditional U-Net, widths 128/256/512, kernel 5, groups 8 and 128-dimensional diffusion-step embedding. Its global condition is 256 context features, 32 placement-window probabilities, and two current readiness probabilities (290 total). All actions use the shared eight-channel affine action normalizer. The outer framework owns DDPM noising, 100-step sampling, decoding, sample clipping and controller execution. This model does not add a new action coordinate frame or claim equivariant joint control.

Two causal MLP heads predict 16-by-2 placement-window logits and 17-by-2 readiness logits (current plus 16 future slots). The placement-window label means the corresponding object is geometrically eligible in that slot and is not already protected in the current history. It is a placement/containment-window predictor, not a precise first-contact event detector or a learned task-success classifier. A currently held contained block remains eligible until observed release support. A negative window label means no demonstrated placement at that offset, not a fabricated failed attempt.

The recurrent forecast has a 128-dimensional GRU hidden state initialized from causal context. At each action slot it consumes the normalized absolute action, its difference from a current-state joint/aperture reference, a slot coordinate, and 64 context features. It predicts TCP/red/blue positions and normalized residual quaternions plus both finger apertures, relative to the latest observed state. The position and aperture residual units use only the supplied shared scales. A small nonzero output initialization starts near persistence without deleting the action gradient path. This is a recurrent supervised regression model, not a robot or rigid-body dynamics solver. It can ignore actions if observational shortcuts suffice; action sensitivity and calibration need independent evaluation.

Slot j is aligned to the supplied state AFTER action t-1+j. In particular slot zero predicts the already causally available state t. The residual forecast anchor is state t, and the recurrent head learns this offset from labels; it is not wrongly labelled as a t+1-only predictor. All future losses use `future_mask * mask`; pairwise displacement losses use both neighboring valid masks. No target crosses a retained segment. Action imitation covers the entire expanded slice, including its known first slot and padded tail under the supplied action mask.

## Exact analytic contract and readiness

For each cube, form all eight corners `p + R(q) v`, with `v` in `{+/-0.02}^3`. The two XY margins are `0.06 - max_corner(abs(corner_xy - goal_xy))`. The height margin is `0.011 - abs(p_z - goal_z)`. An object's live predicate holds iff all three margins are nonnegative. Task completion is the conjunction for red and blue at one observation. Rotated corners are implemented explicitly, not replaced by centre distance or an axis-aligned cube assumption. Height is a centre-height test, not corner-height containment.

Readiness is different. Its supervised proxy requires the per-object geometric predicate, both fingers above 0.035 m, TCP/object distance above 0.06 m, and TCP at least 0.05 m above the object. It is a per-object current empty-hand/separation context, not a latched release state, grasp sensor, or success constraint. A released object can stay protected even when its readiness proxy changes during the next acquisition. Current and future readiness come from trainable logits; opening and retreat actions themselves come from diffusion imitation, not a readiness controller.

`model.diagnostics(raw_history)` optionally returns live margins, live per-object predicates, their task conjunction, observed protection support, the nominated ID, predicted placement-window/readiness probabilities and observed readiness proxies. The required deployment forward still returns only epsilon. The framework is not assumed to automatically call this extra diagnostic method. The invocation agent can compute the documented live predicate from its observations. It should pass original qpos/qvel, TCP/object poses, goals and two-observation motion context to any successor along with the role/protection interpretation, rather than substituting predicted states.

## Losses and gradient paths

Let D be masked epsilon MSE. Training returns:

`L = D + 0.5 R + 0.05 E + 0.025 U + 0.01 C_demo + 0.05 P_demo + 0.02 C_action + 0.05 P_action + 0.02 T`.

- **R, forecast regression:** shared-scale position MSE, 0.25 times sign-invariant quaternion loss `max(0,1-dot(q_pred,q_true)^2)`, 0.25 times shared-scale aperture MSE, and 0.5 times shared-scale adjacent position-change MSE. Gradients train the GRU, pose/aperture heads and causal encoder. Futures are labels only.
- **E, causal placement-window BCE:** future-mask-aware BCE with positive weight 3, supervised by analytic geometric eligibility at real future observations, excluding objects already protected in live history. No classifier replaces the analytic completion predicate. Its initial bias is -2. The event head also receives gradients through denoiser conditioning.
- **U, separate readiness BCE:** average of current readiness BCE and masked future readiness BCE. The same causal representation predicts both; future labels are never forward inputs.
- **C_demo:** squared negative margins of forecast object poses divided by the physical contract tolerances `[0.06,0.06,0.011]`, only at future-labelled placement slots. It differentiates through predicted poses and normalized rotations into the forecast and encoder. This is not a penalty on observed poses alone.
- **P_demo:** forecast drift of live protected objects, plus 0.1 times quaternion drift and 0.1 times negative-margin energy. Position drift uses shared object scales. This teaches preservation, not exact placement-centre replay for the carried object.
- **C_action/P_action:** form `x0=(noisy-sqrt(1-alpha_bar)*epsilon)/sqrt(alpha_bar)`, clip the rollout input to [-1,1] in the supplied normalized action coordinates, and forecast with detached parameters and detached history context using `torch.func.functional_call`. Analytic energies now have the gradient path `energy -> predicted object poses -> clean actions -> epsilon -> U-Net/encoder/conditioning heads`. Forecast parameters cannot reduce this action loss by simply moving their predictions. These are meaningful differentiable action regularizers, not constant observed-geometry costs.
- **T, trust:** masked clean-action MSE to demonstrated normalized actions, weighted by `alpha_bar^2` and enabled only for `alpha_bar >= 0.5`. It complements the main epsilon loss and restrains out-of-support action-outcome optimization.

Action-outcome losses use detached reliability weight `1[alpha_bar>=0.5] * alpha_bar^2 * exp(-encoded_action_MSE/0.04) * exp(-forecast_position_MSE/0.02)`. Errors are computed per slot against actual demonstrated actions/futures during training. They cannot be reduced by differentiating through the weight. Placement terms additionally use `max(0,2*p_event-1)` and the future-labelled placement mask. Consequently a large negative height margin during airborne transport alone never demands immediate descent. Protected-object terms are live-history-gated, not gated by future success of red. Masked means include the appropriate label mask in their denominator, while the reliability/event probabilities remain multiplicative attenuation factors in the numerator. These checks restrict guidance to local, comparatively reliable training predictions; they are not proof of dynamics calibration at deployment.

## Expanded handoff evidence and invocation

Actual states were read in demo10000, demo10003 and demo10011, including their slice starts, overlap transitions and last retained actions. The measured aggregate start width is about 0.08 m total: assuming every call starts with an already closed grip would contradict the assignment. The encoder receives finger positions/velocities, TCP proximity, object motion and height to support mid-transition takeover without an elapsed-time index.

- demo10000/80 has open fingers and TCP z=0.0661 m; /120 has each finger near 0.0183 m and TCP/object z near 0.02 m; /180 and /200 have a closed grasp context and object height about 0.28 m. These phases remain imitation training targets.
- /265 has lowering at z=0.2085 m, /290 has red z=0.02262 m and closed command -1. Red is geometrically eligible before release. At /296 the command turns +1; /305 has fingers near 0.03976 m and settled red z=0.02 m. /350 has TCP z=0.2984 m. /424 is a high approach toward blue, not a terminal red-pad hold.
- /470 is an open descending blue approach, /520 a partial grasp/lift context, /584 and /630 airborne blue transport with red unchanged. /690 has blue z=0.02284 m and command -1 while red is already contained: the task contract is satisfied without disengagement. /692 and /700 show opening/settling, /720 retreat, and /750-/772 terminal TCP near 0.299 m with each finger near 0.04 m.
- demo10003/285 similarly contains red while closed; /679 has released blue; /716 is still retreating. demo10011/480 begins very low with open fingers, /610 is airborne, /710 is open but TCP still low, and /776 is the useful high terminal context.

The assignment's long red outgoing overlap, for example demo10000 [265,425), includes lowering, opening, settling, retreat and high next approach. Incoming overlaps [80,185) and [470,585) include acquisition through lift. None is removed or forced into a carry-only policy. For another acquire invocation, distinguish contained-but-held, released-but-low, and released-and-separated with current observations. The acquire successor shares the bridge and may finish disengagement; transfer does not require waiting for a hard timer. With no successor, one observed task conjunction suffices, and continuing to the demonstrated open/high state is optional.

## Applicability, limitations and evaluation scope

Use the demonstrated robot/controller, known 0.04 m cubes, separate fixed pad geometry and measurable world poses. The supplied shared normalizer fitted to all 9,237 original samples is used unchanged; no per-skill or per-occurrence scale is fitted. Geometry tolerance division is a physical loss unit, not empirical normalization. Fixed training remains seed 0, 20,000 updates, batch 128, horizon 16/history 2, execution 8, AdamW 1e-4/1e-6, cosine schedule with 500 warmup updates, gradient norm 1, DDPM 100 and last EMA with decay 0.999.

Only successful near-centre outcomes are supplied. Neither readiness probabilities nor counterfactual outcomes are validated as calibrated. The short horizon cannot optimize a far-away pad event; event supervision and imitation deliberately preserve grasp/lift/carry instead. Grasp inference is ambiguous during opening/contact, qvel can contain contact oscillations, and two observations do not supply a persistent latch. Corner geometry is exact for the stated rigid cube, but it does not make joint actions invariant or guarantee containment. No fallen-block recovery, moving-pad tolerance, collision avoidance, novel rotations, disturbed-object rescue or unseen dynamics is demonstrated. Action clipping and regularization are not safety guarantees.

Proposed comparisons remain unrun: identical diffusion without outcome losses, centre-only versus full-corner/height losses, with/without protection, forecast action-sensitivity and held-out-original-trajectory calibration, and a future properly integrated online guided sampler. Interface checks are not policy performance evidence. This package makes no rollout success claim before the outer framework's declared-budget training and evaluation.
