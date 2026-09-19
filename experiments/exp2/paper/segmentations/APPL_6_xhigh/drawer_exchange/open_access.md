# Open drawer and establish red pickup

Open the drawer beyond 0.26 m, relinquish the handle without losing opening, and establish red pickup through initial lift so red evacuation can take over across a broad range of manipulation states.

## Segmentation

All 12 demonstrations begin from the same robot rest configuration with red in a closed drawer and blue outside. The primary manipulation is handle approach, closure and prismatic pulling. Retain handle release, full retreat, red approach/descent/closure and measurable initial red lift as a transition suffix. Individually inspected endpoints have red z approximately 0.142-0.143 m, fingers approximately 0.0182 m and continued upward motion: 475 for most, 476 for demo1001 and 474 for demo1005/1007/1008. Thus the outgoing interface covers acquisition, not only a released-handle pose. Images demo1000:0/240/480 confirm layout, extended drawer and red pickup; state/action samples determine exact boundaries. This option does not learn complete red placement beyond its shared suffix.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo1000`: [0, 475)
- `demo1001`: [0, 476)
- `demo1002`: [0, 475)
- `demo1003`: [0, 475)
- `demo1004`: [0, 475)
- `demo1005`: [0, 474)
- `demo1006`: [0, 475)
- `demo1007`: [0, 474)
- `demo1008`: [0, 474)
- `demo1009`: [0, 475)
- `demo1010`: [0, 475)
- `demo1011`: [0, 475)

## Heuristic 1

Prismatic relational coordinates: separate articulation along the drawer axis from object offsets while retaining robot reachability. Hypothesis: this representation improves generalization to placement and opening variation rather than memorizing an absolute TCP path.

**Evidence inspected by the API**

- `demo1000` observations: 0, 140, 150, 160, 195, 215, 230, 240, 300, 390, 440, 450, 465
- `demo1001` observations: 0, 195, 240, 300, 390, 440, 450, 475
- `demo1009` observations: 0, 195, 240, 450, 475

**Interpretation**

In demo1000 red x changes from 0.1055 at 0 to -0.0835 at 195 and -0.1972 at 240 while blue is stationary; demo1001/1009 preserve different red placements on nearly identical drawer motion. After release the TCP rises and rotates while red stays still until pickup. Absolute coordinates entangle articulation and object placement. Axis/relative encoding addresses this without pretending joint actions are translation invariant. This is a representation hypothesis, distinct from phase memory or action-candidate lookahead.

**Applicability**

Same Panda and pd_joint_pos controller, drawer translating along world -X, gravity direction and drawer geometry retained. Most plausible transfer is modest variation of red placement and opening amount. A changed cabinet pose requires a newly supplied cabinet frame, not assumed global SE(3) invariance. State poses and drawer_position must be available causally.

**Implications for a future training/inference pipeline**

Inputs: causal qpos/qvel, tcp_pose/red_pose/blue_pose (world metres, wxyz), drawer_position d and drawer_velocity. Construct c=[0.19-d,0,0.035] m, tcp-c, red-c, TCP-to-red and blue-to-TCP relations; use quaternion-sign-invariant rotation matrices. A typed relational encoder has robot/drawer/red/blue nodes and separate axial-X and transverse-YZ channels on drawer edges. Keep qpos and world-height/base-relative features for reachability and gravity. Condition an action-chunk diffusion denoiser on these tokens; predict eight original-normalized absolute controller targets, inverse-normalize and bound at output. Train noise-prediction loss plus auxiliary next-drawer-displacement regression from the relational latent. Next d is a future training target, never an inference input. Deploy by recomputing c and denoising a short receding-horizon prefix; no IK is needed. The prior is moving-frame/axis-structured conditioning, not projection of joint actions. Prediction: better sample efficiency for separating loaded pull from red approach than an unstructured world-state MLP.

**Assumptions and limitations**

Drawer motion and robot starts are virtually identical across demonstrations; benefit over a time-indexed memorizer is therefore hypothetical. Initial red slip means red_x+d is only approximately conserved. Compare a relational encoder to a capacity-matched world-state encoder on held-out trajectories and placement/opening variation; no gain or systematic d-dependent errors falsify the claim. Successful data do not establish blocked-drawer recovery, arbitrary cabinet rotation or noisy-pose robustness.

### Handoff interface

**Entry conditions**

Demonstrated initial entry: drawer_position=0 m, drawer_velocity=0 m/s, qpos arm=[0,-0.3,0,-2.1,0,1.8,0.7854] rad, fingers=0.04 m each, tcp_pose xyz approximately [-0.384,0,0.442] m world, red z=0.063 m inside, blue z=0.020 m outside. Later entries along the observed approach/contact manifold can be encoded by drawer-relative features; arbitrary open-drawer starts are not established.

**Exit conditions**

Primary exit has drawer_position approximately 0.300 m after release, then upward retreat and red approach. Expanded endpoint: red z approximately 0.142-0.143 m, TCP about 0.009 m to red's -X side, fingers approximately 0.0182 m, g=-1 and positive red/TCP lift motion. Preserve the red grasp when transferring this late; opening completion is not a reason to open fingers again.

**Failure signatures**

d does not rise while TCP moves -X, red fails to translate with drawer before pickup, d falls below 0.26 m, fingers close without red lifting, or red approach begins while the tool remains attached to the handle. These are alarms, not demonstrated recovery cases.

**Overlap role**

Both policies learn [195,474-476): continue a pull from d approximately 0.1863 m and drawer_velocity approximately 0.1071 m/s with fingers approximately 0.0071 m, release at full opening, retreat to TCP z approximately 0.369 m, approach red, close and lift. Retain both handle-relative and red-relative features during the change.

**Successor readiness**

Red-evacuation receives qpos/qvel, tcp/red/blue poses and drawer states. Early takeover still requires pulling/releasing and upward retreat; late takeover must continue lifting held red before translation. The measured endpoints define supported examples, not a validated millimetre tolerance.


## Heuristic 2

Contact-event memory rather than episode clock: an ordered latent should disambiguate similar pre-contact, attached and released poses, improving gripper timing and takeover consistency.

**Evidence inspected by the API**

- `demo1000` observations: 125, 140, 150, 160, 195, 230, 240, 260, 300, 390, 440, 450, 465
- `demo1001` observations: 195, 240, 390, 440, 450, 476
- `demo1008` observations: 450, 474

**Interpretation**

At 140 TCP is at handle height with open fingers; at 150 nearly the same pose has fingers near 0.007 m and drawer displacement. At 230/240 the TCP stays near z=0.128 while fingers release. At 440/450 similarly close TCP poses precede/follow red closure, then red rises by 465-476. Static pose can be ambiguous. Recurrent contact-event state is the proposed bias; different red-lift timing in demo1001 and demo1008 argues against a global clock even though drawer timing is nearly identical.

**Applicability**

Contact is unobserved but causal pose, proprioception and available command history exist at 20 Hz. Retain observed handle/block widths, compliance and task ordering. Timing transfer is more plausible than transfer to new contact mechanics.

**Implications for a future training/inference pipeline**

Use a causal recurrent encoder over approximately 0.5-1 s of qpos/qvel, tcp/red/blue poses, d/vd and previous available commands. Train an ordered semi-Markov latent over handle approach/closure/pull/release, retreat, red approach/closure/lift. Soft phase targets use finger changes, drawer displacement, TCP height and red/TCP co-motion; future motion may label training phases but is not an input. Condition phase-specific adapters of a shared diffusion denoiser on the filtered posterior and time-since-inferred-event, not episode timestep. Loss = noise prediction + soft phase supervision + dwell/transition regularization. Output original-normalized absolute joint/gripper chunks; re-infer phase at each observation and execute short prefixes with standard bounds. Prediction: fewer premature grip/release actions and less sensitivity to dwell duration than a purely geometric encoder.

**Assumptions and limitations**

Geometry/future-co-motion phase labels are proxies, not measured contact forces. Ordered phase bias may obstruct retries after missed grasps. Compare to memoryless and clock-conditioned denoisers; ablate dwell regularization and test truncated histories/timing-shifted handoffs. Successful-only data cannot calibrate uncertainty on failures or validate reversed-phase recovery.

### Handoff interface

**Entry conditions**

Initial entry uses the demonstrated open-finger rest state and a masked history buffer. Later entry initializes a phase posterior from qpos[7:9] in metres, available command history, TCP-object distances in world metres, drawer_velocity in m/s and recent pose changes, not elapsed episode time.

**Exit conditions**

Progress from pull through released-handle retreat and red approach to attached-red initial lift. Supported late exit: d approximately 0.300 m, g=-1, fingers approximately 0.0182 m, red/TCP rising near z=0.143 m. Handle release at TCP z approximately 0.128 m with fingers opening toward 0.04 m is an intermediate handoff, not a stop-all-motion state.

**Failure signatures**

Phase chatter, predicted free fingers while width remains near 0.007 m and drawer moves, or predicted red attachment without correlated lift. Near-zero TCP speed is ambiguous because pauses occur before closure and after contact. Posterior certainty outside the demonstrated history manifold is not established.

**Overlap role**

The 279-281-action shared interval supervises handle attachment through free travel to red attachment, including fingers changing from approximately 0.04 to 0.0182 m while TCP pose initially barely changes. Both policies learn the contact evolution instead of an instantaneous skill switch.

**Successor readiness**

Pass causal observations and optionally a recent action/history buffer, not a required private latent. The successor must infer whether to finish pulling, release/retreat, approach or continue red lift. Do not reset g to open at option switches. Test timing tolerance by switching throughout the overlap.


## Heuristic 3

Successor-aware short-horizon diffusion selection: favor demonstrated-like chunks predicted to preserve opening and establish red pickup. Hypothesis: consequence prediction improves handoff quality beyond marginal action imitation.

**Evidence inspected by the API**

- `demo1000` observations: 195, 215, 230, 240, 260, 300, 440, 450, 465
- `demo1001` observations: 195, 240, 440, 450, 476
- `demo1006` observations: 195, 240, 450, 475

**Interpretation**

At 195 d=0.1863 m is moving; at 215 d=0.2970 it slows; at 230 it reaches 0.300 before fingers release. The robot retreats while preserving opening, then establishes another grasp. Action matching alone need not preserve these consequences under its own sampled actions. Short action-conditioned consequence prediction is an optimization/inference prior, distinct from coordinates or phase memory. The dependencies are observed; counterfactual prediction accuracy is an assumption to test.

**Applicability**

Same dynamics and accurate object/drawer state observations. Most useful for short local continuation near full opening and red acquisition. It is not long-horizon planning through unseen collisions.

**Implications for a future training/inference pipeline**

Train a full-state action diffusion proposal using original normalized targets. Separately train an ensemble action-conditioned recurrent dynamics model on qpos/qvel, TCP/object poses, d/vd to predict physical increments for short observed prefixes, with geodesic rotation loss. A progress/readiness head uses training-only future labels for opening, release/retreat and red lift. At deployment sample K chunks, roll out short predictions and rank by demonstration likelihood minus uncertainty and stage-conditioned penalties for loss of opening, implausible red/drawer coupling during pulling, and loss of red/TCP coupling after pickup. Infer stage from current geometry/history; deactivate pull coupling after release. Replan after a short executed prefix, using original absolute action decoding and bounds. No unavailable kinematics or force model is assumed. Scoring weights require validation. Prediction: fewer over-pulls and more consistent red-ready handoffs than proposal-only imitation.

**Assumptions and limitations**

Only successful action neighborhoods support the learned dynamics. This does not justify a reliable safe/unsafe discriminator. Keep short horizons and demonstration-likelihood penalties. Compare to the identical diffusion proposal without lookahead; low-likelihood selections, worsened pickup or merely longer pauses falsify benefit. Jam recovery and impact response remain untested.

### Handoff interface

**Entry conditions**

Accept the demonstrated initial rest state or an in-distribution partial pull/approach with qpos/qvel and poses known. Evaluate forward-model uncertainty before relying on candidate scores. Current d/vd distinguishes useful articulation from mere TCP travel; failed entry states are not in training.

**Exit conditions**

Choose prefixes retaining d>0.26 m and a feasible next manipulation state. Expanded endpoint has d approximately 0.300 m, red z approximately 0.143 m, fingers approximately 0.0182 m and g=-1. Earlier supported exits include release and rising retreat without requiring zero velocity.

**Failure signatures**

Predicted/observed d diverge, selection proposes continued over-pull or lateral travel before release, predicted red lift is absent, or d decreases during red approach. Low ensemble disagreement alone is not safety evidence.

**Overlap role**

Both policies train on late pull through red lift. This candidate additionally predicts successor-relevant outcomes across release and reattachment. It must not reward transporting red while still attached to the handle or force every transition into a static pose.

**Successor readiness**

Transfer actual observed state/history after the prefix, not predicted state as fact. Red policy supports takeover while d approximately 0.186 m is still increasing, after release near 0.300 m, or during initial red lift. A learned readiness score is advisory, not a scripted switching predicate.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
