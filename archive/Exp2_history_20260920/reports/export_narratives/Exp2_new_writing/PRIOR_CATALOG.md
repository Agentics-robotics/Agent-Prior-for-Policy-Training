> **Compact writing companion.** The report below describes the full evidence package. This companion retains reports, results, all 60 numerical demonstrations, 255 videos, submitted policy source/priors, and the two recovery-case traces/responses. Complete per-step traces and full API design/deployment journals are in `Exp2_new_paper_bundle.zip`. Read `COMPANION_SCOPE.md` for exact scope. Scientific API files are unchanged.

# API-authored policy catalog

The prior summaries below are copied verbatim from submitted API metadata. They describe intended mechanisms, not experimentally isolated causal effects. Naive DP is developer-authored. No scientific API output was edited for this bundle.

## naive_DP

### drawer_exchange / naive_DP

Fixed conditional diffusion U-Net; no Runtime API design.

16,998,408 trainable parameters; 60,000 formal updates.

### two_block_sort / naive_DP

Fixed conditional diffusion U-Net; no Runtime API design.

16,998,408 trainable parameters; 60,000 formal updates.

### buffer_swap / naive_DP

Fixed conditional diffusion U-Net; no Runtime API design.

16,998,408 trainable parameters; 60,000 formal updates.

### unstack_sort / naive_DP

Fixed conditional diffusion U-Net; no Runtime API design.

16,998,408 trainable parameters; 60,000 formal updates.

### tray_pack / naive_DP

Fixed conditional diffusion U-Net; no Runtime API design.

16,998,408 trainable parameters; 60,000 formal updates.

## SinglePrior_6_xhigh

### drawer_exchange / SinglePrior_6_xhigh

Object/drawer-relative anticipation: shared object encoders and a causally predicted short-horizon relational future condition one learned action diffusion U-Net. A masked future-relation regression loss supervises the same anticipation features used by the denoiser.

19,038,112 trainable parameters; 60,000 formal updates.

[Unchanged API prior](policies/SinglePrior_6_xhigh/drawer_exchange/SinglePrior_6_xhigh/source/PRIOR.md) · [Unchanged API implementation](policies/SinglePrior_6_xhigh/drawer_exchange/SinglePrior_6_xhigh/source/policy.py) · [Authorship audit](policies/SinglePrior_6_xhigh/drawer_exchange/SinglePrior_6_xhigh/authorship_audit.json)

### two_block_sort / SinglePrior_6_xhigh

A goal-relative predictive object-factorization prior: a shared encoder represents each block's two-frame relationship to the TCP and its own pad, and supervised causal relational forecasts condition one absolute-joint epsilon diffusion U-Net alongside the complete normalized observations.

19,379,044 trainable parameters; 60,000 formal updates.

[Unchanged API prior](policies/SinglePrior_6_xhigh/two_block_sort/SinglePrior_6_xhigh/source/PRIOR.md) · [Unchanged API implementation](policies/SinglePrior_6_xhigh/two_block_sort/SinglePrior_6_xhigh/source/policy.py) · [Authorship audit](policies/SinglePrior_6_xhigh/two_block_sort/SinglePrior_6_xhigh/authorship_audit.json)

### buffer_swap / SinglePrior_6_xhigh

Occupancy-aware object relational conditioning with supervised causal interaction forecasts. A shared two-object encoder represents gripper contact geometry, each object's assigned and opposite goals, and the other object's occupancy of its goal. Learned short-horizon TCP, object and finger forecasts condition one epsilon-prediction diffusion U-Net alongside the full normalized observation history.

19,504,329 trainable parameters; 60,000 formal updates.

[Unchanged API prior](policies/SinglePrior_6_xhigh/buffer_swap/SinglePrior_6_xhigh/source/PRIOR.md) · [Unchanged API implementation](policies/SinglePrior_6_xhigh/buffer_swap/SinglePrior_6_xhigh/source/policy.py) · [Authorship audit](policies/SinglePrior_6_xhigh/buffer_swap/SinglePrior_6_xhigh/authorship_audit.json)

### unstack_sort / SinglePrior_6_xhigh

Object-centric predictive relation conditioning: a shared two-object encoder and shared short-horizon relation forecaster expose object-to-TCP, object-to-goal, and inter-object structure to one learned action diffusion U-Net. Forecasts are supervised by demonstration futures during training and computed only from the two causal observations at deployment.

19,487,544 trainable parameters; 60,000 formal updates.

[Unchanged API prior](policies/SinglePrior_6_xhigh/unstack_sort/SinglePrior_6_xhigh/source/PRIOR.md) · [Unchanged API implementation](policies/SinglePrior_6_xhigh/unstack_sort/SinglePrior_6_xhigh/source/policy.py) · [Authorship audit](policies/SinglePrior_6_xhigh/unstack_sort/SinglePrior_6_xhigh/authorship_audit.json)

### tray_pack / SinglePrior_6_xhigh

A shared temporal tool-object-goal relation encoder conditions one action diffusion U-Net. A training-only, shared short-horizon relation-displacement forecast regularizes this representation without imposing an object order or a controller.

20,020,712 trainable parameters; 60,000 formal updates.

[Unchanged API prior](policies/SinglePrior_6_xhigh/tray_pack/SinglePrior_6_xhigh/source/PRIOR.md) · [Unchanged API implementation](policies/SinglePrior_6_xhigh/tray_pack/SinglePrior_6_xhigh/source/policy.py) · [Authorship audit](policies/SinglePrior_6_xhigh/tray_pack/SinglePrior_6_xhigh/authorship_audit.json)

## APPL_6_xhigh

### drawer_exchange / open_access__h01

Prismatic relational conditioning for a learned action DDPM: typed robot, drawer, red and blue nodes exchange messages using moving-drawer coordinates, with separate world-X axial and YZ transverse channels on drawer edges. Robot joints and world positions preserve reachability and gravity information. A masked drawer-displacement prediction auxiliary trains the shared causal encoder.

19,489,208 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/drawer_exchange/open_access__h01/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/drawer_exchange/open_access__h01/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/drawer_exchange/open_access__h01/authorship_audit.json)

### drawer_exchange / open_access__h02

Contact-event memory rather than an episode clock. A two-observation GRU feeds a learned eight-phase, duration-augmented ordered filter. Its soft posterior and censored dwell statistic gate phase-specific conditioning adapters of a shared learned action diffusion denoiser. Soft contact-motion phase supervision, ordered future-phase prediction, anti-chatter and transition-consistency losses train the latent representation over the entire expanded slice.

19,131,216 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/drawer_exchange/open_access__h02/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/drawer_exchange/open_access__h02/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/drawer_exchange/open_access__h02/authorship_audit.json)

### drawer_exchange / open_access__h03

Successor-aware short-horizon diffusion selection. A full-state diffusion proposal has three learned paired epsilon/clean-action refinements. A separately supervised three-member recurrent dynamics ensemble predicts eight-action consequences and opening, release/retreat and red-lift readiness. Within-denoising soft selection prefers demonstration-like predicted consequences with less disagreement, opening loss and stage-inappropriate coupling. A differentiable training consequence loss also teaches the epsilon proposal through frozen dynamics. This is an explicit adaptation of completed-chunk reranking to the fixed epsilon-only deployment interface, not an external planner.

18,041,340 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/drawer_exchange/open_access__h03/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/drawer_exchange/open_access__h03/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/drawer_exchange/open_access__h03/authorship_audit.json)

### drawer_exchange / evacuate_red__h01

Object-role relational attention conditions a learned joint-action DDPM on causal robot, drawer, red, blue and fixed-pad tokens. Explicit tool/object, object/target and object/drawer relations coexist with world anchors and sign-invariant rotation matrices. Learned robot-query attention changes focus without discarding background goals. A masked future-object displacement predictor trains the same attended representation.

20,082,059 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/drawer_exchange/evacuate_red__h01/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/drawer_exchange/evacuate_red__h01/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/drawer_exchange/evacuate_red__h01/authorship_audit.json)

### drawer_exchange / evacuate_red__h02

Clearance waypoint hierarchy: a causal full-state encoder predicts event and manipulated-entity probabilities and conditions a learned spatial waypoint diffusion denoiser. A separate conditional action DDPM consumes the resulting robot/object intent and current state. Local future waypoints, event changes, capped duration and ongoing motion are supervised; teacher and predicted intents are mixed during action training.

19,812,790 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/drawer_exchange/evacuate_red__h02/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/drawer_exchange/evacuate_red__h02/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/drawer_exchange/evacuate_red__h02/authorship_audit.json)

### drawer_exchange / evacuate_red__h03

Attachment-consistent action diffusion: a causal learned none/handle/red/blue posterior and per-block continuous tool-relative transform refinement condition an absolute-joint DDPM. A learned action-conditioned rollout predicts TCP, block, drawer and finger states. Masked predicted block-relative translation and rotation changes are penalized only on confident co-motion windows; the handle receives an X-only prismatic coupling penalty. Release observations and commands deactivate the training constraint, not the denoiser.

18,602,102 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/drawer_exchange/evacuate_red__h03/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/drawer_exchange/evacuate_red__h03/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/drawer_exchange/evacuate_red__h03/authorship_audit.json)

### drawer_exchange / insert_blue__h01

Articulated containment margins condition a learned action diffusion transformer. World-frame wxyz poses yield rotated block extents, full XY containment margins, strict Z-interval margins and the drawer-opening margin. Both objects remain represented throughout red placement, blue acquisition, insertion and optional terminal retreat. A differentiable denoiser-latent head predicts masked demonstration future margins at four horizons; it does not impose premature goal satisfaction.

4,505,681 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/drawer_exchange/insert_blue__h01/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/drawer_exchange/insert_blue__h01/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/drawer_exchange/insert_blue__h01/authorship_audit.json)

### drawer_exchange / insert_blue__h02

Contact-belief adaptive diffusion: a causal two-frame GRU infers an eight-mode soft contact belief and a continuous relative-position latent. One learned epsilon diffusion decoder is conditioned on that belief and predicted contact changes. A learned temporal residual blends current and forecast contact embeddings with entropy/event-dependent retention. Future co-motion supplies weak training labels, and an action-conditioned learned motion predictor couples an auxiliary loss to denoised actions. The fixed interface does not permit changing the executed prefix; adaptive commitment is implemented inside the denoiser, not as an adaptive execution scheduler.

18,107,916 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/drawer_exchange/insert_blue__h02/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/drawer_exchange/insert_blue__h02/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/drawer_exchange/insert_blue__h02/authorship_audit.json)

### drawer_exchange / insert_blue__h03

Concurrent-goal preserving lookahead, implemented as an amortized selection objective for a learned action diffusion policy. Three bootstrapped action-conditioned transition models score five local alternatives to training-time DDPM clean estimates using exact world-pose containment margins, learned short-horizon remaining-work forecasts, retention of observed red/drawer/blue goals, ensemble disagreement and a learned behavior-support penalty. Detached selections supervise the denoiser. Deployment uses the fixed DDPM sampler, not an unsupported external controller or runtime K-chunk reranker.

20,186,659 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/drawer_exchange/insert_blue__h03/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/drawer_exchange/insert_blue__h03/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/drawer_exchange/insert_blue__h03/authorship_audit.json)

### two_block_sort / acquire_transport__h01

Role-relative object-goal binding with a shared two-object graph encoder, learned requested/predecessor role probabilities, and a FiLM-conditioned learned action-chunk epsilon diffusion model. Both object nodes remain available during predecessor disengagement. Because the fixed forward interface has no caller role-ID fields, a causal learned role resolver is supervised for the demonstrated red-then-blue order; there is no scripted action controller.

20,116,875 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/two_block_sort/acquire_transport__h01/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/two_block_sort/acquire_transport__h01/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/two_block_sort/acquire_transport__h01/authorship_audit.json)

### two_block_sort / acquire_transport__h02

Contact belief rather than command state. A learned two-observation recurrent contact filter, soft mode transitions, and a persistent probabilistic gripper decoder condition a joint eight-channel action diffusion model. Masked finger-response and TCP-relative object-transform forecasting provide additional trainable contact/co-motion supervision. The complete expanded acquisition slices include predecessor red descent, opening, retreat, blue approach, closure, lift and initial transport.

18,742,319 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/two_block_sort/acquire_transport__h02/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/two_block_sort/acquire_transport__h02/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/two_block_sort/acquire_transport__h02/authorship_audit.json)

### two_block_sort / acquire_transport__h03

Height-separated waypoint hierarchy: an eight-step auxiliary pose/duration diffusion with learned phase-conditioned progress produces four local role-relative TCP/requested-object knots. Smooth bounded interpolation conditions the standard learned actuator-action DDPM. Soft prediction-dependent geometry losses favor separated approach, empty retreat/travel, and loaded lift regimes while exempting predecessor lowering.

19,629,388 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/two_block_sort/acquire_transport__h03/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/two_block_sort/acquire_transport__h03/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/two_block_sort/acquire_transport__h03/authorship_audit.json)

### two_block_sort / deliver_disengage__h01

Grasp-frame transport with explicit detachment, implemented as an action-indexed geometric reference denoiser conditioning a learned joint-action DDPM. Separate red and blue future object poses, a free TCP reference, causal attachment probabilities and a learned two-observation grasp-transform filter provide offset-compensated TCP references. Masked prediction losses teach rigidity only across inferred held intervals, not during empty retreat.

19,859,748 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/two_block_sort/deliver_disengage__h01/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/two_block_sort/deliver_disengage__h01/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/two_block_sort/deliver_disengage__h01/authorship_audit.json)

### two_block_sort / deliver_disengage__h02

Contract-aware learned action diffusion with a causal placement-window predictor, separate disengagement-readiness predictions, and a recurrent action-conditioned pose/aperture forecast. Analytic rotated-corner containment and centre-height losses train the forecast and, through a detached-parameter forecast, regularize denoised action chunks. Already released contained objects receive a preservation loss. Outcome guidance is amortized during training; this implementation does not change the fixed DDPM sampler or perform online rollout-energy optimization.

18,752,545 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/two_block_sort/deliver_disengage__h02/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/two_block_sort/deliver_disengage__h02/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/two_block_sort/deliver_disengage__h02/authorship_audit.json)

### two_block_sort / deliver_disengage__h03

Event progress with elastic duration: a two-observation recurrent belief predicts ordered manipulation events, within-event progress, positive nominal and remaining durations, and uncertainty. A learned progress-knot action reference is time-decoded, and a progress-resampled temporal diffusion denoiser is pulled back to physical action timestamps. A separate persistent gripper head conditions a direct-time gripper diffusion score.

20,075,093 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/two_block_sort/deliver_disengage__h03/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/two_block_sort/deliver_disengage__h03/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/two_block_sort/deliver_disengage__h03/authorship_audit.json)

### buffer_swap / buffer_red__h01

Occupancy-conditioned staging graph with learned role probabilities and a probabilistic staging-site prediction conditioning an absolute-joint action diffusion model. A full five-node causal scene graph distinguishes red, blue, their different goal regions, and TCP. A blue-plus-goals layout subgraph proposes a free red buffer; the robot/world branch retains absolute Panda configuration. Training combines masked epsilon MSE, locally available supported-buffer likelihood, five-role cross-entropy, and soft predicted-site footprint/table penalties. All deployment conditioning uses predictions, never future labels or a scripted controller.

19,402,517 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/buffer_swap/buffer_red__h01/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/buffer_swap/buffer_red__h01/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/buffer_swap/buffer_red__h01/authorship_audit.json)

### buffer_swap / buffer_red__h02

Learned two-frame causal GRU contact belief conditions an action DDPM. Finger aperture, TCP-object relative rotations/translations and backward co-motion support six soft manipulation modes plus independent red/blue support predictions. Learned transitions encourage persistence without a scripted controller. Event-weighted gripper denoising and trainable short-horizon action-conditioned attachment/co-motion forecasts supervise the full red-buffer-to-blue-acquisition slice.

18,201,960 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/buffer_swap/buffer_red__h02/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/buffer_swap/buffer_red__h02/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/buffer_swap/buffer_red__h02/authorship_audit.json)

### buffer_swap / buffer_red__h03

Hierarchical learned diffusion: an internally denoised, ordered four-waypoint TCP route with orientation, gripper and within-window timing conditions the fixed joint-action epsilon diffuser. A supervised action-conditioned TCP trajectory predictor couples task geometry to decoded motor trajectories, with weak low-height lateral-sweep regularization.

19,691,731 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/buffer_swap/buffer_red__h03/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/buffer_swap/buffer_red__h03/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/buffer_swap/buffer_red__h03/authorship_audit.json)

### buffer_swap / transfer_blue__h01

Learned attachment-conditioned action diffusion with a causal two-object TCP-relative transform estimator, none/red/blue attachment probabilities, and an action-conditioned recurrent future-state predictor. Masked future-state supervision and attachment-gated relative-pose consistency train both the representation and, through reconstructed clean actions, the denoiser. The full expanded slice includes initial red descent/release, blue acquisition/carry/setdown/release, and buffered-red regrasp/lift.

19,473,719 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/buffer_swap/transfer_blue__h01/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/buffer_swap/transfer_blue__h01/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/buffer_swap/transfer_blue__h01/authorship_audit.json)

### buffer_swap / transfer_blue__h02

Role-conditioned noninterference in a learned action diffusion model: a TCP/red/blue interaction graph estimates passive versus active roles, and an action-conditioned ensemble provides masked preservation and conservative point-to-cube clearance training losses. The fixed sampler has no candidate-ranking hook; ranking is not implemented and preservation is amortized into the epsilon predictor during training.

19,219,835 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/buffer_swap/transfer_blue__h02/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/buffer_swap/transfer_blue__h02/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/buffer_swap/transfer_blue__h02/authorship_audit.json)

### buffer_swap / transfer_blue__h03

A learned finite-context event posterior and state-dependent residual-duration hazards condition an epsilon-predicting joint-action diffusion model. Soft stay/advance belief propagation encourages event persistence without a trajectory clock. Training-only command/geometry annotations, right-censored future event targets and a soft ordered-transition loss align the full red-release to blue-transfer to red-reacquisition sequence.

18,468,920 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/buffer_swap/transfer_blue__h03/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/buffer_swap/transfer_blue__h03/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/buffer_swap/transfer_blue__h03/authorship_audit.json)

### buffer_swap / finish_red__h01

Rotated-footprint contract-aware action diffusion. Exact observed margins condition a learned epsilon U-Net. An action-conditioned future-pose/support surrogate supplies masked supervision and low-noise differentiable containment and blue-preservation guidance to denoised action estimates during training. Online actions remain learned DDPM samples, not a controller.

18,608,842 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/buffer_swap/finish_red__h01/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/buffer_swap/finish_red__h01/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/buffer_swap/finish_red__h01/authorship_audit.json)

### buffer_swap / finish_red__h02

Soft, observation-correctable progress-memory diffusion policy. A learned state initializer and two-observation GRU infer six contact/progress modes; their belief, predicted goal geometry, and observation disagreement condition a learned action DDPM. Action-conditioned recurrent progress forecasts, supervised only by training future labels, teach ordering and persistence. This is a bounded-memory adaptation, not a persistent hidden-state controller.

18,336,918 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/buffer_swap/finish_red__h02/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/buffer_swap/finish_red__h02/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/buffer_swap/finish_red__h02/authorship_audit.json)

### buffer_swap / finish_red__h03

Grasp-frame-conditioned joint-action diffusion. A learned two-observation blue/empty/red attachment encoder gates measured object-to-TCP transforms and grasp-compensated goal TCP errors. A state-conditioned learned joint-noise decoder and masked action-conditioned future-pose losses learn local action/motion coupling without IK. Every action is sampled by the supplied joint-action DDPM.

18,620,517 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/buffer_swap/finish_red__h03/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/buffer_swap/finish_red__h03/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/buffer_swap/finish_red__h03/authorship_audit.json)

### unstack_sort / acquire_lift__h01

Role-relative geometry with support context, implemented as a learned two-object message-passing encoder conditioning an action DDPM. Shared object and edge networks combine TCP/object/goal differences, table and inter-object height context, sign-invariant rotation matrices, and causal motion with a global robot branch. Three learned latent attention slots replace unavailable invocation tags; both objects also remain in an ungated pooled path.

18,883,662 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/unstack_sort/acquire_lift__h01/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/unstack_sort/acquire_lift__h01/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/unstack_sort/acquire_lift__h01/authorship_audit.json)

### unstack_sort / acquire_lift__h02

A causal recurrent contact-belief prior conditions a learned absolute-joint action DDPM. Six soft object-indexed modes distinguish empty approach/retreat, red closing, red held, red releasing, blue closing and blue held. Future finger positions, coupled object/TCP displacement and relative orientation changes provide training-only outcome supervision; no measured contact is asserted.

18,697,570 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/unstack_sort/acquire_lift__h02/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/unstack_sort/acquire_lift__h02/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/unstack_sort/acquire_lift__h02/authorship_audit.json)

### unstack_sort / acquire_lift__h03

Clear support before lateral loading. A history-conditioned action diffusion U-Net is trained with an action-conditioned response ensemble, frozen-response matching, and a soft world-z clearance energy. Small late-denoising energy-gradient steps are accepted only against an unguided candidate with bounded action changes and low ensemble disagreement. Explicit causal selected/previous-object role features separate red pickup, prior-red completion, and blue pickup.

18,853,648 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/unstack_sort/acquire_lift__h03/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/unstack_sort/acquire_lift__h03/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/unstack_sort/acquire_lift__h03/authorship_audit.json)

### unstack_sort / carry_place_clear__h01

Transport the object rather than the TCP: a learned attachment-confidence gate exposes causal TCP-to-object rigid transforms to a shared object-token encoder, while both object-to-pad errors remain visible after placement. A learned action-conditioned future-pose predictor supplies masked held-object transform losses to the epsilon diffusion model.

18,976,841 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/unstack_sort/carry_place_clear__h01/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/unstack_sort/carry_place_clear__h01/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/unstack_sort/carry_place_clear__h01/authorship_audit.json)

### unstack_sort / carry_place_clear__h02

History-informed, phase-conditioned action diffusion with separate arm and gripper epsilon heads. A binary clean-gripper auxiliary, weak ordered event supervision, masked persistence losses and an action-conditioned measured-response predictor distinguish persistent grip events from smooth absolute joint setpoints. The fixed Gaussian DDPM is retained; categorical gripper diffusion is adapted to a Bernoulli auxiliary and binary-support consistency, not represented as an implemented categorical sampler.

19,939,226 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/unstack_sort/carry_place_clear__h02/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/unstack_sort/carry_place_clear__h02/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/unstack_sort/carry_place_clear__h02/authorship_audit.json)

### unstack_sort / carry_place_clear__h03

Contract-aware learned action diffusion with a three-member action-conditioned outcome ensemble, causal horizon-progress conditioning, phase-gated rotated-corner containment shaping and preservation of already achieved goals. Goal energy has a zero-cost feasible region rather than a pad-center target. Opening, retreat and the red-to-blue transition retain ordinary demonstration supervision.

18,812,877 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/unstack_sort/carry_place_clear__h03/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/unstack_sort/carry_place_clear__h03/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/unstack_sort/carry_place_clear__h03/authorship_audit.json)

### tray_pack / acquire_block__h01

Role-factored relative geometry: a shared encoder processes both pose-and-goal bundles using world and TCP-relative translations, relative rotations, goal errors and causal differences. Learned latent-role attention pools object interactions; a separate fixed-base robot stream conditions a learned action DDPM. This is a representation-sharing prior, not a phase controller.

19,874,888 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/tray_pack/acquire_block__h01/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/tray_pack/acquire_block__h01/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/tray_pack/acquire_block__h01/authorship_audit.json)

### tray_pack / acquire_block__h02

Contact-progress-conditioned hidden semi-Markov latent diffusion: a two-frame recurrent belief encoder, six soft manipulation modes, uncertain run-age initialization and learned progress-dependent stay/exit hazards condition an absolute-joint/gripper DDPM. Training-only action, aperture, height and masked future co-motion cues supervise phase, pending-role, duration likelihood and readiness. Finite transition preferences favor demonstrated dependencies without an irreversible automaton.

18,598,171 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/tray_pack/acquire_block__h02/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/tray_pack/acquire_block__h02/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/tray_pack/acquire_block__h02/authorship_audit.json)

### tray_pack / acquire_block__h03

Attachment as an action-conditioned invariant: an original joint/gripper action diffusion model is conditioned on learned exclusive none/red/blue attachment beliefs and structured TCP/object forecasts. Attached forecasts compose a learned slowly varying TCP-to-object transform; independent forecasts have separate support-motion branches. Future pose, regime, rigidity, stationary-object and low-noise denoising-consequence losses train this distinction over the complete expanded slice.

20,021,364 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/tray_pack/acquire_block__h03/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/tray_pack/acquire_block__h03/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/tray_pack/acquire_block__h03/authorship_audit.json)

### tray_pack / deliver_block__h01

Move the payload rather than only the TCP. A learned goal-relative TCP path latent with continuous rotations and gripper commands is conditioned on causal TCP-to-object transforms and attachment confidence. A local learned inverse-dynamics decoder supplies bounded normalized joint proposals to the joint-action DDPM denoiser. Future-path supervision, attached-payload consistency and a differentiable denoised-action/path cycle train the geometry. The mandatory sampler remains in joint-action space; this is an explicitly documented geometric-latent adaptation, not a separate Cartesian DDPM or exact IK.

19,583,897 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/tray_pack/deliver_block__h01/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/tray_pack/deliver_block__h01/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/tray_pack/deliver_block__h01/authorship_audit.json)

### tray_pack / deliver_block__h02

Rim-aware transport topology implemented as an absolute-joint diffusion policy with fixture-relative causal conditioning, learned contact/motion-mode forecasts, and a supervised action-chunk-conditioned pose ensemble. A differentiable, confidence-weighted geometric preference compares a denoised chunk with the demonstrated chunk through frozen rollout weights. This amortizes the assigned predictive-ranking hypothesis into diffusion training; the fixed inference interface does not perform completed-sample ranking.

19,661,368 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/tray_pack/deliver_block__h02/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/tray_pack/deliver_block__h02/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/tray_pack/deliver_block__h02/authorship_audit.json)

### tray_pack / deliver_block__h03

Containment is not disengagement. A learned two-frame recurrent contact belief with probabilistic phase persistence conditions an original-action diffusion policy. Separate containment, attachment and readiness predictions, future motion supervision, release-event survival likelihood and a detached-branch anti-reclosing loss distinguish closed-held goal entry from reusable empty-hand transitions.

18,730,377 trainable parameters; 20,000 formal updates.

[Unchanged API prior](policies/APPL_6_xhigh/tray_pack/deliver_block__h03/source/PRIOR.md) · [Unchanged API implementation](policies/APPL_6_xhigh/tray_pack/deliver_block__h03/source/policy.py) · [Authorship audit](policies/APPL_6_xhigh/tray_pack/deliver_block__h03/authorship_audit.json)

