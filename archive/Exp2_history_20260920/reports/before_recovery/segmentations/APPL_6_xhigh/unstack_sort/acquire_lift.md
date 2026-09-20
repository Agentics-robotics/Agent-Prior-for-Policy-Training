# Acquire and lift a selected block, with previous-placement transition context

Acquire the selected block without disturbing its support, lift it into free transport space and maintain the pinch during initial goal-directed motion. When invoked for blue before red release, first finish red placement, open and clear the gripper, then approach and acquire blue. The useful end state is a selected block moving with the TCP and fingers closed, not a particular joint pose or elapsed time.

## Segmentation

Each trajectory contributes two segments: red acquisition from the original open home state, and blue acquisition beginning while the previously held red is descending to its pad. These share alignment, pinch acquisition and vertical separation; red's support is blue and blue's support is the table. Red endpoints are early transport after a full lift (world red z about 0.308-0.318 m); blue endpoints are early transport following a full lift (world blue z about 0.280-0.284 m). For example demo10201 red stop 170 is only beginning lateral departure, whereas demo10207 stop 184 is further into transport; both retain a pinched block and nonzero qvel. Blue starts range from red z about 0.175 m at demo10210 238 to 0.281 m at demo10211 229, before release. This intentionally adds a long previous-placement lead-in, not a separate instruction to place blue prematurely. The caller supplies the acquisition target (red for the first segment, blue for the second) and previous-object role; the policy must complete old-object release before changing the manipulated role. The repeated acquisition core justifies grouping despite different support heights and lead-in duration. Images inspected at demo10200 0/100/280 and demo10204 460 corroborate stack, grasp and release geometry, but numerical state is the primary evidence and no contact sensor is inferred from images. All endpoints in these listed ranges were read. Observe/action normalization is shared across the complete original training set. Each heuristic is an alternative separately trained Diffusion Policy candidate; none is a claim of measured generalization.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo10200`: [0, 180)
- `demo10200`: [235, 540)
- `demo10201`: [0, 170)
- `demo10201`: [225, 525)
- `demo10202`: [0, 178)
- `demo10202`: [234, 538)
- `demo10203`: [0, 176)
- `demo10203`: [232, 536)
- `demo10204`: [0, 174)
- `demo10204`: [230, 532)
- `demo10205`: [0, 177)
- `demo10205`: [233, 535)
- `demo10206`: [0, 179)
- `demo10206`: [236, 540)
- `demo10207`: [0, 184)
- `demo10207`: [242, 550)
- `demo10208`: [0, 182)
- `demo10208`: [240, 548)
- `demo10209`: [0, 175)
- `demo10209`: [231, 533)
- `demo10210`: [0, 180)
- `demo10210`: [238, 542)
- `demo10211`: [0, 173)
- `demo10211`: [229, 531)

## Heuristic 1

Role-relative geometry with support context: share the pickup mapping across object identities and source translations using local TCP/object/support relations, while retaining global robot configuration. The expected benefit is learning one height-aware pickup geometry rather than memorizing two color-specific joint trajectories.

**Evidence inspected by the API**

- `demo10200` observations: 0, 80, 100, 150, 170, 280, 390, 440, 450, 500, 520
- `demo10204` observations: 0, 77, 100, 130, 160, 280, 350, 385, 434, 480, 505
- `demo10208` observations: 0, 82, 286, 446, 472

**Interpretation**

At demo10200 100->150 the red z rises from 0.060 to 0.308 m while blue stays at 0.020 m. At demo10204 100->160 the same local lifting problem occurs from a different XY position; 480->505 repeats it for blue from the table. TCP is consistently about 9 mm in negative world x from the grasped block center, not exactly coincident. This supports object-relative representation and explicit support height, but not a claim of exact world/robot translation symmetry. The previous-object context is necessary: blue acquisition starts while red is still held. Unlike A2, this hypothesis changes spatial sharing rather than latent timing; unlike A3 it does not add motion constraints.

**Applicability**

Same Panda joint-position controller, parallel fingers and roughly cubic 0.04 m blocks on a horizontal table, with causal world-frame object/TCP poses and goals. Intended transfer is small source translations and reuse between elevated red and tabletop blue, not arbitrary robot-base rotations. A causal invocation supplies selected-object identity and previous-object identity, if any; those are task context, not measured observation fields. Deployment selection requirements are otherwise unspecified.

**Implications for a future training/inference pipeline**

Candidate A1: relational-encoder Diffusion Policy. Inputs are causal qpos (7 rad + 2 finger m), qvel, tcp_pose/red_pose/blue_pose (world m, wxyz), red_goal/blue_goal, and invocation tags selected/previous. Drop the fixed-zero drawer channels after fitting the prescribed common normalizer. Build shared object tokens containing pose, goal minus object position, TCP minus object position, table-relative height and selected/previous/other role; edge features include relative pose and support relation (red above blue initially). Use sign-invariant quaternion/rotation-matrix features. A shared message-passing encoder plus a robot-global branch for qpos, qvel and absolute TCP conditions a temporal diffusion denoiser over the original 8D action chunks. Decode seven absolute joint targets in rad and the dimensionless gripper command, clipped to the declared limits at 20 Hz; do not interpret joint targets as deltas. Train ordinary noise-prediction/action reconstruction losses, sampling red and blue invocations comparably. Role assignments come from segment identity in training and the caller in deployment, not future goal completion. All candidates use the single observation/action normalizer fitted on the complete original demonstrations; geometric differences are formed in physical units and scaled using those fixed dataset-wide scales, never fitted per skill. At inference use only current/past states, graph encoding and receding-horizon denoising. Hypothesis: sharing local geometry reduces data needed for the two pickup heights while retaining a global branch prevents false whole-arm translational equivariance.

**Assumptions and limitations**

The initial stack translations span only about 2 cm and orientations are nearly fixed. Relative features cannot prove translation generalization, color interchangeability or collision avoidance. Global Panda reachability must not be discarded. Failure prediction: object-relative-only encoding chooses correct local motion but an unreachable arm solution. Compare this graph encoder against an equal-capacity flat-state diffusion encoder, and separately remove support-height and previous-object edges. Benefits should appear as reduced grasp error across source position and red/blue height, not just lower training loss. No failed grasps, toppled stacks or wrong-order attempts establish recovery.

### Handoff interface

**Entry conditions**

Accept either the demonstrated initial state (qpos arm [0,-0.3,0,-2.1,0,1.8,0.7854] rad, qpos[7:9] about 0.040 m, TCP world z=0.442 m, red above blue at z=0.060/0.020 m) or the blue invocation with red still grasped above its pad. In the latter observed starts red z=0.175-0.281 m, fingers about 0.0183 m each and downward arm motion. The role graph must retain red as the previous manipulated object until placed; do not simply aim at blue immediately.

**Exit conditions**

The selected block follows the TCP, fingers remain about 0.0182-0.0183 m with gripper command -1, and the block has reached the observed lift band: red about 0.308-0.318 m or blue about 0.280-0.284 m at selected endpoints. Early lateral motion and nonzero arm qvel are supported; exit is not a requirement to stop at one joint pose. Continue closed-finger transport actions during transfer.

**Failure signatures**

Selected object fails to rise with TCP, the other block moves during red separation, or red is still carried away from its pad when the blue role becomes active. A low relational distance alone is not proof of a grasp. Source translations outside the observed roughly x=-0.361 to -0.339 m, y=-0.011 to +0.012 m band and altered orientations are unvalidated.

**Overlap role**

Both this candidate and carry/place learn aligned open descent, closing, vertical lift and early transport. On blue entry, it also learns completing red descent, opening and retreat before approaching blue; red carry/place learns these same actions. Role tags differ across that overlap but raw actions must agree.

**Successor readiness**

Carry/place receives the same selected block and its goal plus current qpos/qvel/tcp_pose and both object poses. It may take over earlier in the shared pickup band with fingers still open, or later with the block held; the graph must communicate which object is currently held versus merely selected. Measured phase bands are support examples, not validated distance/velocity tolerances.


## Heuristic 2

Infer manipulation state from causal coupling, not from the last gripper command or elapsed time. A history-conditioned latent-contact prior should distinguish closing, holding and releasing, reducing premature lift and previous-to-next-object switching.

**Evidence inspected by the API**

- `demo10200` observations: 80, 90, 100, 150, 265, 280, 300, 390, 440, 450, 465, 500, 520
- `demo10204` observations: 77, 100, 130, 160, 260, 270, 280, 290, 310, 350, 434, 460, 480, 505
- `demo10201` observations: 75, 270, 425, 455

**Interpretation**

The same g=-1 command does not specify whether fingers have closed or the block is moving: demo10204 100 has qpos fingers near 0.0204 m and fast closure, 130 has near 0.01825 m and rising red, while demo10200 100 is already near 0.01825 m but red is still on blue. At red release, demo10204 270 is still pinched, 280 is opening, 290 is nearly fully open, and 310 retreats without red following. A causal latent attachment/role belief addresses this aliasing. The observed sequence motivates the labels; robustness to delayed contact is a hypothesis that requires intervention tests later.

**Applicability**

Applicable when short causal histories of the declared states are available at 20 Hz and blocks can be approximately rigidly pinched. No force, tactile or contact flags are assumed. The same model must accommodate an initial empty hand and a blue invocation that still holds the previous red object. It is a learned belief model, not a claim of measured contact.

**Implications for a future training/inference pipeline**

Candidate A2: causal latent-contact Diffusion Policy. Feed a masked rolling history (initial design 12 observations = 0.6 s, tune later) of qpos/qvel, world TCP and object poses, goals and role tags to a recurrent encoder. Add learned latent states previous-held/previous-releasing/empty-approaching/selected-closing/selected-attached, with soft duration-aware transition regularization rather than absolute timestep input. Condition a standard joint-action diffusion denoiser on the belief. Train with denoising loss plus auxiliary prediction of future finger separation, object displacement, TCP-object relative-transform persistence and soft phase labels. For training only, derive attachment/phase targets from command changes, future qpos[7:9], upward object displacement and bounded relative-pose change over several demonstration frames; use uncertain soft labels around transitions, not asserted contact ground truth. Future outcomes supervise the encoder but are never inputs. At deployment update belief from past/current states and prior commands, denoise original normalized 8D absolute targets, execute a short prefix at 20 Hz and reobserve. All scaling is the common full-demonstration normalizer. Candidate advantage should be temporal alignment and action consistency when the PD response lags, rather than A1's spatial equivariance or A3's geometric penalty.

**Assumptions and limitations**

Rigid following is only an attachment proxy: common commanded motion and actual grip can be confounded, and successful demonstrations provide no slip/disengagement calibration. A semi-Markov structure might merely recover the expert's timer. Falsify by comparing against a history-matched diffusion transformer without mode supervision, against an absolute-time-conditioned baseline, and with history removed. Expected benefit is fewer premature lifts/role switches under changed timing, not proven slip recovery. No contact-force thresholds or calibrated stopping probabilities can be learned from these successes alone.

### Handoff interface

**Entry conditions**

Initialize a causal belief from available state history and role tags. At original start use a masked empty history and open-finger state; at blue entry permit red-held/downward motion with qpos[7:9] about 0.0183 m. Resetting the belief to 'empty hand approaching blue' would contradict the inspected entry states. A short warm history is preferable but must not include demonstration futures.

**Exit conditions**

Posterior mass favors selected-object attached/lifting or carrying, supported by persistent TCP/object relative pose during upward motion, not merely gripper command -1. The recorded exit bands have selected z about 0.28-0.32 m and fingers about 0.01825 m; arm motion may continue laterally. Continue -1 through takeover and provide the causal belief/history if the successor can consume it.

**Failure signatures**

TCP rises without matching object motion; abrupt TCP-object relative transform change; a finger-close command with still-open qpos; or posterior oscillation between previous-held and selected-held. Finger qvel can be about -0.02/+0.026 m/s even while successfully held, so a strict zero-finger-velocity contact test is unsupported. Belief confidence thresholds require later calibration on failures.

**Overlap role**

Shared acquisition/carry windows train the whole open->closing->coupled lift transition, including cases such as demo10204 100 with fingers still closing versus demo10200 100 already near 0.01825 m. Shared red-placement/blue-acquisition windows train previous-held->released->retreat->next-approach; no single instantaneous contact reset is assumed.

**Successor readiness**

A successor needs selected/previous role tags, latest normalized physical state and enough causal history to distinguish release from grasping at similar TCP heights. If no latent-state interface is implemented, pass observation history and let the successor re-encode it; never pass labels computed from future block lift. Supports on-path temporal variation, not arbitrary unobserved contact states.


## Heuristic 3

Clear support before lateral loading: once grasped near its source, a block should gain vertical separation before substantial sideways transport. A soft outcome-space clearance prior may reduce dragging the lower block and make pickup trajectories easier to generalize without replaying a fixed lift waypoint.

**Evidence inspected by the API**

- `demo10200` observations: 0, 100, 150, 170, 235, 265, 280, 300, 450, 465, 500, 520
- `demo10204` observations: 100, 130, 160, 270, 290, 310, 350, 460, 480, 505
- `demo10207` observations: 0, 84, 242, 290, 476

**Interpretation**

Red in demo10200 stays nearly fixed in XY from 100 to 150 while rising about 0.248 m and blue remains at z=0.020 m; appreciable transport begins only later. Demo10204 repeats vertical separation for red at 100/130/160 and blue at 460/480/505. The red release/retreat also separates finger opening from upward empty-hand motion. These observations support an anisotropic, contact-dependent motion preference. They do not prove the full recorded lift height is necessary. This prior changes optimization and inference by disfavoring premature horizontal load; unlike A1/A2 it can constrain candidate actions even with an ordinary flat state encoder.

**Applicability**

For upright blocks and a gravity-aligned table with a free vertical escape corridor, under the declared stiff pd_joint_pos dynamics. Use world z explicitly rather than assuming arbitrary 3D rotational symmetry. Clearance and support geometry must be retained; no obstacles beyond the observed other block/table are supplied.

**Implications for a future training/inference pipeline**

Candidate A3: clearance-guided Diffusion Policy with an action-conditioned response model, not a new spatial graph or semi-Markov decoder. Use causal qpos/qvel, TCP/object poses, goals and invocation tags to condition a temporal action diffusion model. Train F_phi(s_t,a_t:t+H) on real demonstration futures to predict TCP displacement, both object poses and finger separation across H; these futures are labels only. Add a soft task-space energy on predicted responses during selected-object attachment/lift: E = sum w_attach*w_near_source*softplus(h_clear-c_pred)*||Delta p_object,xy||^2 + lambda_other*||Delta p_support||^2. c_pred is predicted bottom-of-object clearance above its source support in world metres. Gates are continuous functions of causal history plus learned attachment estimates; close-to-source gating prevents penalizing placement descent elsewhere. During previous-object completion, use only release/retreat consistency, not the selected-object lift penalty. Use conservative small guidance on denoised action estimates plus denoising and rollout-supervision losses; preserve demonstrated upward motion and compare guidance strengths. At inference compute F_phi from the observed state and candidate action chunk, apply capped energy-gradient guidance, decode globally normalized absolute joint/gripper commands within limits and execute a short prefix at 20 Hz. No measured force or future state is available to the guide. The single full-original-data normalizer is shared; all clearance calculations use denormalized metres. Expected test effect: less sideways load on the support object at similar pickup completion rate.

**Assumptions and limitations**

A learned response model trained on successful paths is unreliable for large denoising perturbations. Vertical-first behavior may reflect the expert planner rather than necessity, and the table-only scene cannot establish collision-safe guidance. Soft, capped penalties and an unguided fallback comparison are required; no safety guarantee is claimed. Ablate the clearance guidance while retaining the auxiliary dynamics model, and separately remove the other-block stability term. Predict reduced support disturbance, but watch for frozen approaches or needless high lifts. Recovery from slip or a blocked vertical corridor is unseen.

### Handoff interface

**Entry conditions**

Same physical initial/open and prior-red-held entry classes as the dataset. A selected/previous role indicator is mandatory so the geometric penalty does not pull the resting blue upward while red placement is unfinished. Source support height is blue top about 0.040 m for red or table z=0 for blue; object half-size is 0.020 m. At late approach, TCP must already be near the demonstrated local grasp geometry before upward transport is encouraged.

**Exit conditions**

Selected block clears its support with closed fingers and a stable grasp transform; continue vertical lifting into the demonstrated red/blue z bands near 0.32/0.28 m and allow early lateral transport. Do not declare handoff from a predicted clearance alone: observed red_pose/blue_pose and causal following must corroborate it. Keep the decoded gripper command -1 during carry transfer.

**Failure signatures**

Predicted lift with measured object still on its support, blue dragged during red acquisition, wrist lateral motion before actual separation, or guidance suppressing necessary pregrasp alignment. During old-object release, red following the retreat is a failed opening, not a valid clearance event. Learned rollout disagreement or guidance repeatedly pushing commands to joint limits should invalidate confidence.

**Overlap role**

Pickup/carry overlap covers final alignment, close, upward separation, full lift and the onset of sideways transport, giving both policies examples on each side of the proposed clearance preference. Red-place/blue-acquire overlap also includes opening then vertical retreat before moving back to blue, so constraints must be conditioned on held versus released roles rather than applied to every low-height TCP.

**Successor readiness**

The successor must inherit actual qpos/qvel, object pose and gripper state, not just the outcome model's planned pose. Carry/place can enter anywhere in the observed pickup overlap and finish the lift if clearance is incomplete. The observed ~0.26 m lifts are examples, not a proven minimum safe height; margins beyond source contact clearance are tunable hypotheses.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
