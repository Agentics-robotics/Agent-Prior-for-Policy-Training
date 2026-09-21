# Contact-belief-conditioned acquisition diffusion policy

Policy: `acquire_transport__h02`  
Skill: `acquire_transport`  
Assigned heuristic: 2, **Contact belief rather than command state**  
Experiment: M1_v2

## Scientific hypothesis and evidence

The original hypothesis is that acquisition depends on the history linking commanded gripper motion, measured fingers, and object/TCP co-motion, not on treating an issued close command as established contact. It proposes a persistent contact belief to improve asynchronous handoff across closing, lifting and predecessor release. The source heuristic is unchanged. This document describes our executable adaptation, not a replacement for that heuristic or a claim that it has been experimentally confirmed.

Actual observations read during implementation include:

- `demo10000/106,107`: an open grip near the red block precedes and persists at the first -1 command. At /110 the fingers are around 0.01828 and 0.01825 m, but the red block remains near its 0.02 m support height. At /125 lift is just beginning; /150 has red z about 0.191 m, /170 about 0.279 m, and /184 begins goalward transport with the grasp maintained.
- `demo10000/265`: the recurrent slice starts with **red still held**, red z about 0.2085 m above its pad, fingers about 0.01829 m each. At /296 the command opens while the measured fingers remain narrow; /300 is about 0.03707 m per finger; /305 is about 0.03976 m. At /350 red is stationary near its goal at z=0.02 m while the open TCP has retreated to z about 0.298 m. /424 and /470 show empty blue approach/descent.
- `demo10000/503,504,505,506`: blue close is commanded at /504 with fingers still about 0.04 m. /505 has fingers about 0.0205 m and velocities about -0.341/-0.368 m/s. /506 has fingers about 0.01827/0.01824 m but no substantive lift yet. /525 shows initial lift, and /570 has blue z about 0.2804 m with TCP z about 0.2800 m.
- `demo10011/115,116,117`: successful closed contact has a slightly negative red quaternion x component, unlike some other grasps. /272 again starts recurrent acquisition with red loaded above its goal. /309 and /359 support open release/retreat. /519,520,521,560,583 support blue co-motion, lift and initial transport. Finger velocity signs are not a contact truth label.

Measured boundary evidence is also used: start total finger aperture spans about 0.03653 to 0.08 m, while expanded-slice ends have total aperture about 0.03653 m and the carried object near the demonstrated high transport region. These are data support summaries, not learned success rates or hard admission thresholds. In this document per-finger opening and summed aperture are distinguished.

## Full expanded slice coverage and handoff scope

All assigned segments are retained, including both kinds of overlap. There is no phase filter, narrower resegmentation, terminal-state controller, demonstration lookup, or preset skill schedule. Source ranges below are half-open:

| Trajectory | Initial slice | Recurrent slice | Shared intervals with delivery |
|---|---|---|---|
| demo10000 | [0,185) | [265,585) | [80,185), [265,425), [470,585) |
| demo10001 | [0,185) | [260,580) | [78,185), [260,417), [463,580) |
| demo10002 | [0,185) | [263,584) | [79,185), [263,423), [468,584) |
| demo10003 | [0,181) | [255,571) | [76,181), [255,409), [455,571) |
| demo10004 | [0,184) | [267,585) | [81,184), [267,428), [472,585) |
| demo10005 | [0,190) | [265,575) | [80,190), [265,425), [470,575) |
| demo10006 | [0,190) | [268,586) | [82,190), [268,430), [474,586) |
| demo10007 | [0,182) | [258,572) | [77,182), [258,413), [459,572) |
| demo10008 | [0,190) | [270,587) | [83,190), [270,432), [477,587) |
| demo10009 | [0,187) | [266,586) | [80,187), [266,427), [473,586) |
| demo10010 | [0,189) | [267,583) | [81,189), [267,429), [475,583) |
| demo10011 | [0,192) | [272,584) | [84,192), [272,435), [480,584) |

The long recurrent slice teaches red loaded descent, goal-aligned lowering, release, open retreat, empty transfer to blue, approach, closure, lift, and the beginning of grasp-maintaining transport. The initial slice teaches approach to red through red lift and initial transport. A partway invocation can use the observed red-goal relation, red/TCP separation and motion, blue source pose, actual fingers and joint motion to distinguish these phases. No assumption that entry means empty/open is built in.

The fixed data loader supplies windows over these expanded slices and masks padding. We do not replace its sampler or claim a special overlap-balanced sampler. Randomly truncating history during training is our explicit cold-start augmentation; ordinary windows inside shared ranges expose partway entries without future-history leakage.

## Required adaptation to the provided interface

The full research proposal requests role IDs, previously **executed** commands and persistent causal history. The actual forward API provides only `[B,2,47]` observations, a noisy eight-channel action chunk and diffusion timestep. It provides no role IDs, executed commands, reset callback or persistent hidden-state protocol. It fixes horizon 16, execution prefix 8 and an eight-channel epsilon-prediction DDPM.

Accordingly:

1. We implement a **two-sample causal recurrent filter**, recomputed on every forward call. It does not retain hidden state across replans or DDPM iterations. Its learned initial distribution is observation-conditioned, not a fixed empty-gripper state. With two consecutive 20 Hz observations it sees one measured finite difference; this is not several samples of grasp verification.
2. Role information is inferred from the named red/blue channels and goals in the demonstrated red-first, blue-second context. We do not feed source indices, segment identity, a clock or a hidden requested-object ID. Arbitrary reordered requests or ambiguous predecessor identity are not supported merely by the word 'requested' in the skill description.
3. Previously executed commands are not reconstructed as if measured. The gripper decoder estimates its initial command probability from observations. Original gripper commands, including action[t], are used only as training labels. Actual expert action chunks additionally train an auxiliary action-response predictor; they never enter the deployment encoder.
4. A binary-state probabilistic gripper decoder supplies persistent **conditioning**, rather than emitting an independent hard +/-1 command. All eight executed channels remain outputs of the fixed learned DDPM. There is no post-sampler gripper threshold hook, separate controller, automatic open pulse or command override. Thus exact binary execution is not guaranteed. Learned binary-command supervision encourages appropriate bimodal behavior.
5. Fifteen percent of training histories repeat the latest frame in place of the older frame. This removes measured finite-difference evidence and trains observation-conditioned initialization. There is no fabricated pre-slice history, long-history curriculum or simulated negative rollout.

These are material limitations on testing the original stronger long-memory hypothesis. The implemented hypothesis is specifically whether short-history contact filtering, response supervision and learned gripper persistence improve a fixed-interface learned diffusion policy.

## Causal representation and trainable architecture

Inputs use qpos `[0:9]`, qvel `[9:18]`, TCP `[18:25]`, red pose `[25:32]`, blue pose `[32:39]`, compatibility drawer zeros `[39:41]`, and goals `[41:47]`. Poses are world metres with unit wxyz quaternions; arm joints are radians. All raw observation channels use the supplied shared means/half-ranges. We do not estimate new scales. Action targets are the framework's shared affine-normalized seven absolute Panda targets plus dimensionless gripper command. The native mapping remains per-finger target `0.025 * command + 0.015` m, subject to the simulator's existing dynamics/contact, not our own gripper mechanics.

Each observation has 91 features:

- 47 shared-normalized raw channels;
- two seven-dimensional TCP-relative object poses: inverse-TCP-rotated translation and relative quaternion;
- six world object-minus-goal coordinates;
- nine world TCP/red/blue finite-difference velocities;
- fourteen relative-pose finite-difference rates;
- one indication that the two supplied frames differ rather than being repeated padding.

Derived translations/goals/heights use the maximum shared half-range over the nine TCP/red/blue position components (about 0.2489 m), including after rotation. Derivatives divide by the known 0.05 s observation interval. Quaternions keep unit-component scale, not tiny local empirical ranges. Relative quaternions are normalized and signed so the largest-magnitude component is positive; this avoids a scalar-sign discontinuity at the demonstrated approximately 180-degree wrist/object rotation. This convention is not globally continuous for arbitrary orientations. No new dataset statistic is fit.

A 91-to-128 MLP and 128-wide GRUCell encode the two frames. A learned initial hidden state uses the first frame. The first frame emits six mode probabilities. The second predicts a learned 6-by-6 transition matrix, applies it to the first distribution, and combines its log prior with learned current evidence. Transition diagonal logits initialize to 2.5; every transition remains learnable. Modes, in order, are:

1. empty approach or retreat;
2. red closing/unverified;
3. red loaded/lifting;
4. blue closing/unverified;
5. blue loaded/lifting;
6. predecessor red loaded near its pad or releasing.

This expands the original four explanatory modes by color because explicit role IDs are unavailable. Modes are soft internal variables, not switches that select scripted motions. The source hypothesis's missed-grasp/slip classes are not invented without negative evidence.

Both normalized observations (94 values), final recurrent features (128), probabilities (6) and their learned embedding (32) form a 260-dimensional base condition. A 96-wide GRU predicts an initial open probability and horizon-indexed close-to-open/open-to-close hazards. Initial hazard biases are -3, a learnable persistence initialization. Its marginal probabilities evolve as `p_next = (1-p)*hazard_open + p*(1-hazard_close)` without teacher-forced command inputs. A 32-dimensional embedding of all probabilities and hazards yields the 292-dimensional diffusion condition.

The supplied Conditional U-Net uses widths 128/256/512, timestep embedding 128, kernel 5 and 8 groups, with FiLM conditioning. It jointly predicts epsilon for all eight channels. A separate 128-wide action-conditioned GRU predicts an 18-dimensional physical-response vector over the chunk for auxiliary training. That vector contains deltas from the current measured state of both normalized finger positions, both TCP-relative object poses, and both normalized world object heights. It is not a calibrated simulator or a hard contact/kinematic constraint.

Absolute joint actions remain in the robot frame. Relative features neither make the overall policy invariant nor make its joint outputs equivariant. There is no IK, FK library, image encoder or external geometry planner.

## Losses, alignment, and gradient paths

`loss = diffusion_loss + prior_loss`, with

`prior_loss = 0.08 L_mode + 0.15 L_grip + 0.10 L_transition + 0.20 L_response + 0.05 L_coupled`.

- **Diffusion:** the public masked epsilon MSE over all eight channels. Gradients train the U-Net, history filter, mode embeddings and gripper-conditioning modules.
- **Mode:** soft cross-entropy between learned current posterior and a weak training-only explanatory target. The target uses TCP-object distance, actual mean per-finger width, object elevation, two-frame relative displacement, red-to-goal XY distance, demonstrated current close/open label, and masked future lift confirmation. Bandwidths are hypotheses: proximity 0.05 m, width sigmoid center 0.029 m/width 0.003 m, elevation center 0.040 m/width 0.012 m, co-motion residual bandwidth 0.008 m, red-goal XY bandwidth 0.07 m, release width center 0.037 m/width 0.002 m. Future lift beyond 0.05 m adjusts **closing confidence** by a factor between 0.7 and 1; it does not turn a supported object into current loaded contact. Scores get 0.015 smoothing and normalization. Future slots are masked. The target is detached; cross-entropy trains posterior emission, transitions and the recurrent encoder. It is not a loss evaluated only on observed poses.
- **Gripper marginal:** masked binary cross-entropy of predicted open marginals against original +/-1 command labels. This trains the persistent gripper decoder and shared history/mode conditioning.
- **Gripper transitions:** conditional command-label BCE using the learned two hazards and previous command as a **loss label only**. Adjacent action masks must both be valid. Observed switches have weight 5, other pairs weight 1, to learn rare legitimate changes rather than imposing universal smoothing. This trains hazards and context, without feeding the previous ground-truth command to forward.
- **Response:** Smooth L1 between learned physical deltas under expert encoded actions and actual future-state deltas. The 18 dimension weights are `[1,1,5,5,5,1,1,1,1,5,5,5,1,1,1,1,2,2]`, normalized by their sum. Gradients train the response predictor and causal conditioning. It cannot simply be a constant observed co-motion penalty because every term compares a trainable prediction to a label.
- **Coupled response:** construct `x0 = (noisy - sqrt(1-alpha_bar)*epsilon)/sqrt(alpha_bar)`, feed `clip(x0,-2,2)` into the same learned response predictor, and compare against the same physical labels. Weight each error by alpha_bar, averaged over the unweighted valid mask, to reduce high-noise amplification. Gradients pass through the response predictor and unclipped parts of x0 to epsilon and conditioning; clipped extremes deliberately have no direct x0 gradient. The expert-action response loss anchors this auxiliary model. Both branches train it jointly, so this is a soft learned regularizer, not guaranteed dynamics consistency.

At observation t, action slots are t-1 through t+14; future slot zero is state t. Physical targets subtract state t and **exclude slot zero** to avoid rewarding reconstruction of an already observed state. Physical losses require both action and future masks; future confirmation uses those same masks excluding zero. Categorical losses mask individual actions and transition losses mask adjacent pairs. No label or target crosses a skill boundary. Future poses, current expert commands and response targets are absent from model.forward. No auxiliary relies on privileged deployment state.

## Training, dependencies, and deployment

Dependencies are Torch and `appl.public` only. The model has no filesystem, network, shell or demonstration access. Shared normalizer buffers, all neural parameters and learnable initial distributions serialize in the state dict. Forward has no stochastic history augmentation or mutable recurrent cache, so repeated DDPM calls and EMA reloads have the same semantics. History truncation occurs only in compute_loss during training.

The framework retains seed 0, 20,000 updates, batch 128, AdamW learning rate 1e-4/weight decay 1e-6, 500 warmup steps, cosine schedule, gradient clipping 1, EMA 0.999, 100-step training and inference DDPM, horizon 16, history 2, execution 8 and last-EMA selection. No checkpoint or budget is selected from feedback. The prescribed checker tests interface execution, two real updates, sampling and reload; it is not a manipulation performance experiment.

The inference API selects invocation duration and successor. Keep the two most recent actual causal observations available at handoff. A successor does not need our private latents: world object/TCP motion, fingers, qvel and goals suffice to recompute approximate readiness. No latent interchange or exported force estimate is implemented. Caller-side role/command context may aid policy selection but cannot be passed through this forward signature.

## Applicability, continuation, and exit

Use in the observed Panda/cube setting, particularly uncertain closure, loaded lift, or predecessor release in the shared intervals. Widths near 0.04 m (open), 0.0205 m (transient closure), and 0.0183 m (stable loaded finger positions) are evidence, not proof of contact or rigid thresholds. TCP and block centers are nearly level in these demonstrations during grasp, with a small roughly 8-10 mm lateral offset; do not assume an unobserved fixed vertical grasp offset.

For recurrent entry, red near its pad, narrow fingers and red/TCP co-motion indicate that descent/release is still necessary before blue acquisition. Opening fingers plus a supported, stationary red while TCP rises indicate the reverse loaded-to-empty transition. The policy learns those actions from the expanded slice. Continue while these phases remain unfinished rather than transferring simply because red is geometrically over its goal.

A useful acquisition exit has the intended block lifted near the demonstrated z=0.28 m, fingers maintaining closure, small stable TCP-object relative displacement, and grasp-preserving initial motion toward its goal. An external caller should accumulate several actual observations when possible; this model sees only two at a time. Delivery may also take over earlier inside shared pickup overlap if it continues closing verification and lift. Do not inject a release/reset motion at transfer.

Signs to reassess include an inferred loaded condition with the block still supported, growing TCP-object separation, repeated open/close toggles, or persistent disagreement between predicted belief and observed lift. Successful finger qvel has opposing nonzero components, so that alone is not failure evidence. Failure detection and recovery are not hardcoded.

These readiness cues do not redefine task completion: the supplied contract is red_at_goal AND blue_at_goal with full rotated XY containment and the specified z tolerance. It does not require gripper release, TCP clearance, zero velocity, or sustained hold. Acquisition's high transport exit is a subgoal, not task success.

## Limitations and falsifiable claims

All inspected grasps succeeded. Contact remains inferred; finger compliance, lag and object settling can mimic discrete state. Weak labels can be wrong, can over-credit stationary near-contact, and are not posterior calibration data. Future confirmation within a short chunk cannot validate a failed-grasp class. Only two frames may be unable to disambiguate a long stationary pre-close dwell from immediate closure; history truncation makes this ambiguity explicit, not solved. Learned Markov persistence is within the supplied history/chunk, not memory across the task. No guarantees prevent diffusion gripper toggles, off-manifold physical forecasts, collisions or drops.

The policy is trained on the red-then-blue support, not an arbitrary object-role interface or a complete standalone task controller. The response predictor might learn context shortcuts and ignore some action dependence. Dominant-component quaternion signing is local to the observed orientation neighborhood. Physics losses are soft and do not enforce contact, joint limits beyond the framework, or geometric containment.

A proper future test would compare matched causal-history DP with and without filtered modes, remove response losses and persistent gripper conditioning separately, and vary handoff points throughout all shared ranges. Relevant outcomes are lift reliability, physical opening/closure timing, unintended command toggles and successor readiness, not merely mode-label or timestep prediction accuracy. No such outcome measurements are available in this design session.
