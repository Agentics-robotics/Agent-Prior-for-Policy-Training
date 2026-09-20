# Transfer blue to its goal and prepare buffered red retrieval

Finish red buffer release if necessary, acquire and carry blue into the now-vacant initially red region, set blue down and release it, then retreat and initiate buffered red pickup/lift while preserving blue's achieved placement.

## Segmentation

All segments begin before red has finished setdown at the buffer and end after red has been regrasped and lifted, enclosing the entire blue transfer as their core. Starts a are 240,242,240,244,238,240,241,246,239,243,245,242 for demos10100-10111; inspected red heights at a range 0.094-0.239 m, with closed fingers and blue stationary. Ends d are 850,847,843,848,839,838,842,855,841,846,849,844; inspected red heights range 0.170-0.213 m while blue is supported at its goal. Red release/retreat/blue grasp is the incoming interface, and blue release/retreat/red grasp is the outgoing interface. They are intentionally substantial additions around the common blue-carry learning problem. Sparse timing differences are not treated as different skills.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo10100`: [240, 850)
- `demo10101`: [242, 847)
- `demo10102`: [240, 843)
- `demo10103`: [244, 848)
- `demo10104`: [238, 839)
- `demo10105`: [240, 838)
- `demo10106`: [241, 842)
- `demo10107`: [246, 855)
- `demo10108`: [239, 841)
- `demo10109`: [243, 846)
- `demo10110`: [245, 849)
- `demo10111`: [242, 844)

## Heuristic 1

Rigid attachment should make object and TCP motion predictable together: learn an attachment-conditioned invariant transform and penalize inconsistent predicted transport. Hypothesis: object-centric auxiliary dynamics improve carry stability and grasp-offset robustness.

**Evidence inspected by the API**

- `demo10100` observations: 240, 275, 455, 480, 490, 540, 600, 610, 630, 640, 690, 810, 820, 850
- `demo10111` observations: 448, 485, 603, 643, 812, 844
- `demo10103` observations: 488, 606, 848

**Interpretation**

During blue carry in demo10100:480,490,540,600 the world blue-minus-TCP displacement stays approximately [0.0086,-0.0002,-0.0002] m despite substantial motion; after opening at 640 it no longer follows TCP. demo10111 and demo10103 show the same relation with slightly different offsets/heights. A learner that predicts only arm targets may miss whether an object is actually moving with it. The proposed bias ties action learning to rigid attachment and commanded-versus-realized motion. It differs from passive-object protection and event-time alignment below.

**Applicability**

Rigid cube transport by the same parallel-jaw gripper with accurate TCP/object poses; modest grasp offsets and PD lag are retained. This prior is meaningful only during attached modes; buffer release, empty travel and next-red acquisition require separate learned gating.

**Implications for a future training/inference pipeline**

Candidate R: attachment-aware DP with a differentiable learned state-transition head. Inputs: causal qpos/qvel, world tcp_pose/red_pose/blue_pose and goals, finger aperture and relative transforms D_i=T_tcp^-1 T_i. A trainable causal encoder estimates attachment probabilities and a smoothed D for each object. The joint-action denoiser outputs the original eight absolute commands. A rollout head F(state,action_chunk) predicts future qpos, TCP and object poses, trained on synchronized subsequent observations, explicitly accounting for action-target versus measured-qpos lag. During training only, use future co-motion and gripper event labels to supervise attachment. For attached intervals add L_rigid=sum_k ||log((T_tcp_pred[k] D)^-1 T_object_pred[k])||^2 with separate metre/radian scaling; downweight this loss at release transitions using supervised gating. Denoising remains the primary objective so empty transit and changing active objects are learned. At inference estimate D from observations, condition diffusion on attachment/relative transform and optionally score candidate chunks by rollout consistency; use no future object pose and no unprovided force field. Decode through the common full-demo normalizer, bound actions and replan. Prediction: object-level transport errors and accidental opening decrease compared with plain joint imitation even when current TCP is slightly offset.

**Assumptions and limitations**

Constant relative pose is only approximate; compliant fingertips and discrete simulation cause small deviations. The candidate may incorrectly enforce rigidity during release or crush a slip correction. Remove the attachment auxiliary loss and compare with identical denoising architecture; evaluate predicted offset residual, grip continuity and task success. Unseen slipping, nonrigid objects and larger offsets have no demonstrated recovery support.

### Handoff interface

**Entry conditions**

Earliest supported takeover is red still held descending at the buffer, not an empty gripper: e.g. demo10100:240 red z=0.1628 m with fingers near 0.01825 m and action g=-1. Later entries include red-supported/open, open approach, blue-closing and blue-lifting. Initialize the transform estimator from causal object/TCP geometry with an attachment confidence, rather than assuming blue is already held.

**Exit conditions**

Blue is supported inside blue_goal near [-0.352,-0.200,0.020] m; red is regrasped and lifting at the buffer, z about 0.170-0.213 m, g=-1 and fingers about 0.01825 m. Track red's new attachment transform after regrasp; do not retain blue's offset.

**Failure signatures**

The inferred attached object's relative transform drifts, object height does not rise with TCP, g becomes +1 during unsupported carry, or blue remains suspended despite a supported-state estimate. No measured force is available to resolve all ambiguities.

**Overlap role**

The incoming overlap includes the entire red release and blue pickup; the outgoing overlap includes the entire blue setdown/open/retreat and red pickup/lift. The object whose rigid offset is meaningful changes twice in this segment, and both adjacent skills learn those same changes.

**Successor readiness**

finish_red must receive actual qpos/qvel, red_pose and tcp_pose history from which its new red grasp offset and motion can be estimated; it may take over earlier and finish blue release itself. Optional offset/confidence outputs are advisory. The measured 8-9 mm world-x offset is not a universal calibration constant.


## Heuristic 2

Protect completed and buffered objects through role-conditioned noninterference: model active and passive objects differently and discourage predicted passive-object disturbance. Hypothesis: explicit preservation improves chained manipulation where collateral motion would invalidate earlier progress.

**Evidence inspected by the API**

- `demo10100` observations: 240, 275, 320, 390, 455, 490, 540, 600, 640, 690, 755, 810, 820, 850
- `demo10111` observations: 242, 276, 388, 448, 485, 603, 643, 694, 755, 812, 844
- `demo10107` observations: 246, 282, 493, 613, 654, 821, 855

**Interpretation**

Red remains unchanged at the buffer through blue pickup/carry/release, then deliberately moves during retrieval; blue similarly stays at its completed goal while red is picked up. This pattern recurs in demo10100, demo10107 and demo10111. It suggests an asymmetric interaction structure: only the currently manipulated object should normally move. This is a preservation-objective and inference-ranking prior, not the held-object rigidity relation or an elapsed-phase representation. The successful data support stationarity on demonstrated routes, not responses to actual collisions.

**Applicability**

Two-object tabletop rearrangements with one passive supported object while the other is manipulated, and fixed goal regions. Requires full object poses, world table height and the same action dynamics; transfer to dense clutter or additional objects is speculative.

**Implications for a future training/inference pipeline**

Candidate P: role-aware interaction graph DP plus conservative candidate ranking. Use qpos/qvel, TCP/object poses, world goals and backward object velocities as causal input. A trainable role encoder estimates active-object/support probabilities from aperture, relative positions and history; node interactions represent TCP-to-object and object-to-object geometry. Train a diffusion head on absolute joint/gripper chunks and an ensemble forward head on future object poses from demonstrations. Add L_passive=sum_i,k w_passive(i,t,k)*||p_i_pred[k]-p_i(t)||^2 and orientation analog, with weights labelled from actual supported/non-manipulated intervals during training, then predicted at inference. Weight is OFF for red during buffer descent and retrieval and for blue while actively transported. At deployment denoise a small fixed number of candidate chunks; rank by imitation likelihood, predicted passive displacement, ensemble disagreement and a soft predicted TCP-to-passive-cube clearance cost based on the 0.02 m half-size, retaining a no-guidance baseline. Geometry is in world metres, rotations from wxyz; action decoding uses the common complete-demo normalizer and original bounds. Future passive labels/poses are training-only. Prediction: under modest route deviations this candidate better preserves already arranged objects than equally sized flat DP, conditional on forward-model calibration.

**Assumptions and limitations**

A dynamics model trained only on successful noninterference may predict every passive object remains still even for unsafe actions. Geometry penalties only screen obvious proximity and cannot certify collision safety. Ablate passive-motion loss and candidate scoring independently against the same graph DP; count collateral motion and selection regret. Unseen collisions, pushed buffers and recovery are not covered.

### Handoff interface

**Entry conditions**

At early takeover red is active and descending at the buffer while blue is still on its original region; the policy must not apply a passive-red constraint until red is actually released. In subsequent supported-red states, its observed pose is near [-0.181,0,0.02] m and blue becomes active. Current aperture and TCP association determine the role, not color alone.

**Exit conditions**

Blue is now passive at blue_goal, fingers have closed on red and red is rising from the buffer. At demo10107:855 red z=0.1924 m while blue remains z=0.020 m. Exit actions preserve closed red grip and do not pull the arm back toward blue.

**Failure signatures**

Previously stationary red changes pose during blue transit; blue leaves its goal while the empty hand retreats or red is lifted; role inference freezes red when it should be picked up; predicted noninterference disagrees with observed movement. A static object prediction is not a safety guarantee.

**Overlap role**

Incoming overlap learns red's active-to-passive switch and blue's passive-to-active switch; outgoing overlap learns the reverse roles after blue placement. Both neighbors learn opening/retreat before the new grasp. Protection must switch with manipulation state rather than penalize all object motion.

**Successor readiness**

The next policy needs blue actually supported at its goal and the measured red/TCP grasp state, not merely a high passive-object score. A causal baseline pose for each supported object may be passed with uncertainty; finish_red can also reconstruct it from recent observations. Supported spatial variation is limited to the observed sites.


## Heuristic 3

Align actions to observed manipulation events rather than a trajectory clock: learn persistent phases and transition hazards driven by release, approach, closure and lift evidence. Hypothesis: event-conditioned diffusion tolerates different dwell and contact timings without shifting the whole action sequence.

**Evidence inspected by the API**

- `demo10100` observations: 240, 265, 275, 285, 320, 390, 440, 455, 465, 490, 600, 630, 640, 650, 690, 755, 800, 810, 820, 850
- `demo10101` observations: 242, 277, 452, 487, 605, 645, 815, 847
- `demo10107` observations: 246, 282, 456, 493, 613, 654, 821, 855
- `demo10111` observations: 276, 388, 448, 485, 643, 694, 755, 812, 844

**Interpretation**

Near-identical TCP poses occur before closure, during closure and before lifting, and low setdown/open precedes high retreat. Across trajectories these events occur at different indices: blue is open at demo10107:456 but closed at demo10100:455 and demo10111:448. qvel and finger history distinguish intent better than timestamp. Event-relative organization should make temporal alignment easier with little data. Unlike candidate R, this does not enforce rigid dynamics; unlike candidate P, it does not score passive-object disturbances.

**Applicability**

Same ordered transfer with variable approach, opening and closing durations, sufficient causal history and no branches requiring repeated failed grasps. Retain true 20 Hz action timing and qvel units; event alignment is not permission to replay demonstrations at arbitrary speed.

**Implications for a future training/inference pipeline**

Candidate E: semi-Markov latent-event encoder conditioning a joint-space diffusion model. Causal inputs are qpos/qvel, finger aperture, TCP/object/goal relative poses and backward finite differences; do NOT input trajectory ID, global timestep or time-to-go. Learn mode emissions and transition hazards for red-setdown/open, retreat, blue-approach/close/lift/carry/setdown/open, retreat, red-approach/close/lift; phases can be merged when observationally indistinguishable. Training labels come from gripper-command changes plus future evidence of support/lift; normalized within-event progress s=(t-event_start)/(event_end-event_start) is an auxiliary future-derived target only. Train denoising loss, mode/progress prediction and a soft ordered-transition regularizer; sample uniformly across events and random causal windows so long dwells do not dominate. Keep every target chunk and qvel at its actual 20 Hz, rather than performing inconsistent action time warping. At inference update learned event belief/hazard from measured changes, denoise original absolute seven-joint-plus-gripper commands and replan; phase is not a scripted controller. Use the one full-demo normalizer. Test prediction: later/earlier contact transitions disturb motor performance less than in clock-based imitation.

**Assumptions and limitations**

All demonstrations share the same event order, so a monotone prior could lock out legitimate recovery or shortcut behavior. Timing diversity is small and mostly planner-generated. Compare to absolute-time-conditioned and memoryless DPs, and ablate phase supervision while retaining recurrent capacity. The prediction is falsified if alignment does not improve event accuracy or timing perturbation performance; failed-grasp retry competence is unestablished.

### Handoff interface

**Entry conditions**

May enter anywhere in the full incoming overlap, from held red descent (finger positions about 0.018 m) through open retreat (about 0.040 m) to blue closure/lift. The event-state encoder must initialize from causal history or a learned current-state posterior; it cannot assume that its first call means phase zero.

**Exit conditions**

A red-lift event has occurred after confirmed blue setdown/release. Red height is about 0.17-0.21 m at stored endpoints and rising, blue remains supported at goal, g=-1. Phase probability and its uncertainty should accompany but not override these measurements.

**Failure signatures**

Phase advances solely with elapsed calls while object pose stalls, close/open predictions chatter at a dwell, state initialization jumps past a needed release, or the policy repeatedly approaches already grasped blue. A real slip may require a backward transition not demonstrated here.

**Overlap role**

Both adjacent policies learn the entire release-to-next-lift event sequence and the variable dwell lengths inside it. For example demo10107:456 is still open at blue, whereas demo10100:455 is already closed, so a universal timestamp cut would be wrong.

**Successor readiness**

The successor can consume a causal history buffer and re-estimate its own event state, optionally using the predecessor's phase posterior as a soft prior. The nominal useful endpoint is moving, not settled qvel=0; it needs continuity of red grip and upward movement rather than a prescribed joint vector. Random-window training must include all overlap phases.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
