# Acquire a selected block with incoming-release context

Acquire the intended table block in a stable parallel-jaw grasp, lift it clear and begin goalward motion. When entering from another placement, first finish lowering/releasing that incoming payload and retreat toward the intended block. Preserve the earlier placed object. End ready for delivery, without requiring one exact arm pose or zero velocity.

## Segmentation

Group both red and blue top-down acquisitions across all 12 demonstrations: shared subproblem is transition from an empty or soon-to-be-empty gripper to a lifted, co-moving selected block. Red segments start at observation 0; they include high approach, descent, finger closing, vertical lifting and early goalward motion. Blue segments deliberately start much earlier than empty approach, while red is still closed-grasped over its tray region, so acquisition can finish incoming lowering/release, empty retreat and traversal rather than depend on a perfect single release cut. These phases are contextual lead-in, not an instruction to always place an object before a pickup. Blue ends after grasp, vertical lift and initial translation. Individually inspected red stop indices 182,184,188,185,180,190,187,178,181,183,177,184 have red z about 0.280-0.283 m; blue stops 562,564,576,566,558,578,571,552,561,564,551,566 have blue z about 0.287-0.293 m and fingers about 0.0183 m. They are moving states, not arbitrary fixed poses. Incoming starts 265,267,273,268,263,274,271,259,264,266,258,267 retain differing descent stages (z 0.084-0.169 m). All endpoints were read; intermediate release/retreat/grasp/lift states and representative images were inspected in detail for demo10300 and demo10307 and compared to other trajectories. The contextual incoming transition requires a next-object role distinct from the incoming payload. The three priors isolate representation sharing, variable-duration latent state, and action-conditioned attachment dynamics as independently testable DP changes.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo10300`: [0, 182)
- `demo10300`: [265, 562)
- `demo10301`: [0, 184)
- `demo10301`: [267, 564)
- `demo10302`: [0, 188)
- `demo10302`: [273, 576)
- `demo10303`: [0, 185)
- `demo10303`: [268, 566)
- `demo10304`: [0, 180)
- `demo10304`: [263, 558)
- `demo10305`: [0, 190)
- `demo10305`: [274, 578)
- `demo10306`: [0, 187)
- `demo10306`: [271, 571)
- `demo10307`: [0, 178)
- `demo10307`: [259, 552)
- `demo10308`: [0, 181)
- `demo10308`: [264, 561)
- `demo10309`: [0, 183)
- `demo10309`: [266, 564)
- `demo10310`: [0, 177)
- `demo10310`: [258, 551)
- `demo10311`: [0, 184)
- `demo10311`: [267, 566)

## Heuristic 1

Role-factored relative geometry: acquisition depends more on TCP-to-block geometry and manipulation roles than block color or absolute source xy. Share object-slot encoders and expose relative transforms to reduce coordinate memorization while retaining the fixed-base robot state.

**Evidence inspected by the API**

- `demo10300` observations: 0, 75, 120, 160, 265, 290, 320, 400, 445, 480, 500, 520
- `demo10307` observations: 0, 72, 100, 110, 130, 150, 259, 287, 408, 439, 477, 505, 530
- `demo10305` observations: 0, 79, 274, 303, 428, 457, 496

**Interpretation**

Across demo10300 and demo10307, both colors are approached with open fingers, then supported at about 0.0183 m finger position and lifted while TCP and object move together, despite markedly different arm configurations and source y signs. Demo10305 varies the source position but retains the relation. The TCP is not exactly object-centered, so learn offsets rather than impose coincidence. A raw state network can memorize separate color routes from 12 highly similar demonstrations; role-factored relative geometry pools the reusable local grasp problem while retaining world constraints. This is an assumed generalization benefit of representation sharing, distinct from temporal-mode inference and attachment-based future prediction.

**Applicability**

Same Panda/controller and accessible top-down grasps on similar 0.04 m cubes, with causal object poses and goal roles available. Transfer hypothesis concerns modest source-position changes and sharing red/blue acquisition parameters, not rigidly rotating the entire fixed-base robot or arbitrary relocation of the tray. Retain gravity/world height, other object and joint configuration.

**Implications for a future training/inference pipeline**

Inputs: last 4 causal qpos/qvel, tcp_pose, both object poses/goals and intended role. Form e_i=p_i-p_tcp and R_tcp^T(p_i-p_tcp), R_tcp^T R_i for both object slots, goal-minus-object vectors, absolute world z and qpos arm. A shared MLP encodes object slots; role embeddings plus attention pool selected/other/goal interactions. Keep a separate fixed-base robot stream so object-relative sharing does not falsely imply joint-space translation invariance. Condition a temporal diffusion denoiser on these features; diffuse H=16 normalized original eight-dimensional actions and decode with the shared action normalizer to absolute joint/gripper commands. Train standard noise/score loss, role-balanced sampling; optionally test slot-permutation consistency by relabeling pose+goal+role together while keeping the physical scene/actions identical, not by reflecting joint actions. Inference uses only the causal history and role, samples a chunk and executes 2 steps then replans. No future object labels are inputs. Distinguishing change is representation and parameter tying, not a phase classifier or model-based action ranking. Testable prediction: less overfitting to red/blue world coordinates and lower data requirement for shifted source positions.

**Assumptions and limitations**

Both cube geometries and goals are nearly fixed and starts vary only centimetres, so color-independent geometry is a hypothesis, not established broad equivariance. Removing world/base inputs can fail at joint limits or tray walls. Falsify by comparing matched-capacity raw-world DP, shared role encoder without relative features, and this candidate on held-out trajectories and controlled source translations; measure grasp/lift success and joint-limit violations. Successful data cannot establish arbitrary yaw grasps, order changes or recovery.

### Handoff interface

**Entry conditions**

Intended next-object token and both object/goal poses are required. Supported entries are initial open Panda state (qpos fingers 0.04 m, TCP world z 0.442 m, stationary cubes z 0.020 m), or incoming red placement while requesting blue: red center z 0.084-0.169 m above red_goal, qpos fingers about 0.0183 m, continuing descent. Also support internal overlap states with incoming red already released, retreat underway, or approach/closing/lift of the selected block. World/base and role streams must agree; no unseen free-space reinitialization is implied.

**Exit conditions**

Useful acquisition exit is selected block co-moving with TCP above source and beginning goalward translation, command -1, fingers about 0.0183 m. Observed red center z 0.280-0.283 m and blue 0.287-0.293 m, nonzero arm qvel, near-downward TCP wxyz approximately [0,1,0,0]. Earlier transfer during aligned descent is permitted because delivery has the shared approach/close/lift data; do not open during this transfer.

**Failure signatures**

Selected token addresses a settled block instead of the source; role-conditioned TCP-to-object xy error grows during descent; fingers close without object elevation; object drifts away from TCP during lift; incoming red is moved laterally before release. Larger spatial errors than inspected millimetre-scale offsets are unsupported.

**Overlap role**

Learn incoming red lowering/opening/retreat and traversal to blue in the 149-154-action overlap; learn selected-block final approach, close, lift and initial translation in the 106-122-action overlaps. Relative representation changes its reference by explicit role, not at a hard world pose.

**Successor readiness**

Deliver policy receives the SAME selected payload identity, full qpos/qvel and causal TCP/object history plus its goal. It can continue lowering and close if transfer was early, or preserve -1 and translate if late. Observed grasp offsets include TCP x about 0.008-0.010 m behind object x in world, not perfect center coincidence. Any proposed tolerance band around these offsets must be validated.


## Heuristic 2

Contact-dependent phase, not elapsed time: infer variable-duration manipulation modes and make closing, lifting and incoming release conditional on observed progress. A semi-Markov latent-conditioned DP should avoid averaging incompatible actions at visually/geometrically similar poses.

**Evidence inspected by the API**

- `demo10300` observations: 75, 120, 160, 265, 280, 290, 300, 320, 360, 415, 445, 465, 480, 500, 520
- `demo10307` observations: 72, 100, 110, 130, 150, 165, 259, 287, 310, 335, 365, 439, 455, 465, 477, 487, 505, 530, 540
- `demo10301` observations: 76, 267, 293, 418, 447, 485

**Interpretation**

Demo10307 100 and 110 have nearly the same TCP z about 0.020 m, but fingers change from 0.040 to 0.0183 m and the next useful action changes from close to lift. At 465 and 477 the blue case repeats. Demo10300 280-320 shows the inverse contact transition: near-goal closed hold, partial opening, then empty retreat. Demo10301 release is only being commanded at 293, whereas demo10300 290 is already partly open. A single clock or static pose cannot represent these dependencies. Temporal mode inference addresses this ambiguity; it is a different bias from changing geometric coordinates or learning action-conditioned attachment consequences.

**Applicability**

Useful when the same near-object pose occurs during approach, dwell, closing and lifting, and when successor entry can occur at any supported transition phase. Requires sequential observations or a trained missing-history mask, the same gripper actuation delay and accurate qpos/qvel. No clock-based or fixed-duration assumption should transfer.

**Implications for a future training/inference pipeline**

Use last 8 qpos/qvel, TCP and both block/goal poses, intended role and past issued actions; compute causal 20 Hz pose differences. Train a recurrent belief encoder with discrete latent modes: incoming lower/release, empty retreat/approach, aligned dwell, closing, attached vertical lift, early translation. A semi-Markov transition prior penalizes skipped demonstrated dependencies but allows variable dwell and uncertainty. Obtain training-only soft phase labels from source gripper actions, finger aperture, table/goal-relative height and future co-motion over several frames; mask ambiguous labels, never feed future labels at inference. A phase-conditioned diffusion denoiser generates the original normalized absolute-joint/gripper H=16 action chunk; loss is denoising plus phase CE and transition/duration likelihood, with duration conditioned on progress rather than timestep ID. Deployment recursively filters the latent from causal state, samples a chunk and executes 1-2 actions near contact changes, 2-4 in approach, with learned readiness output. Ablate duration prior separately from recurrence. Expected effect is reduced mode averaging and fewer premature lifts/releases on variable-time transfers, not geometric equivariance.

**Assumptions and limitations**

Mode ordering is observed, but contact itself is not instrumented and the dataset contains no successful reversals or interrupted-grasp recoveries. A strict irreversible automaton can trap an error; use learned transitions with uncertainty, not hard replay timing. Compare recurrent DP without auxiliary phase supervision, feedforward relative DP and the proposed semi-Markov model. Falsification signatures are no improvement at randomized within-overlap entry or worse performance under modest timing/actuator-delay changes. Spatial/contact recovery remains untested.

### Handoff interface

**Entry conditions**

Have intended next-object role and an 8-observation causal buffer (0.40 s sampling span nominally; first-to-last is 0.35 s), or an explicitly masked reset trained inside overlaps. Fingers may be open at initial approach, closing at the selected source, or closed on the previous red payload descending over the tray while blue is pending. The latent must initialize from qpos[7:9], previous gripper commands and pose differences, not assume every invocation starts empty.

**Exit conditions**

Posterior favors selected-block attached/lifting or initial transfer, with object elevation and TCP/object co-motion corroborating closed command -1. Demonstrated exits remain in motion at center heights about 0.28-0.293 m; no zero-qvel requirement. An earlier switch is supported only with the inferred approach/closing phase and buffer passed along.

**Failure signatures**

Phase toggles between release and grasp; predicted progress advances while object remains on table; gripper alternates +1/-1 near the same pose; latent labels a stationary closed hold as empty because fingers are not at -0.01 m. Long uncertainty after reset or a jump to late transport indicates unsupported handoff.

**Overlap role**

Both skills learn the ordered progression through approach/dwell/close/lift/translate; incoming placement/release/retreat/blue-approach is also shared. Expanded windows train phase recognition well before and after each plausible transfer, including partially opening fingers at demo10300 290 and early lifting at demo10307 487.

**Successor readiness**

Pass causal history, role and optionally phase posterior with confidence. Delivery must be able to finish close and vertical lift instead of assuming attachment. For incoming blue acquisition, retain red's identity until its release/retreat is completed. Use posterior readiness as a learned signal; numeric confidence or delay thresholds are proposed hyperparameters, not demonstrated robustness.


## Heuristic 3

Attachment as an action-conditioned invariant: a grasped cube should co-move with TCP under a nearly constant relative transform, whereas a released cube should decouple. A structured future model and uncertainty-aware diffusion-sample ranking should make pickup readiness depend on actual object motion rather than the close command alone.

**Evidence inspected by the API**

- `demo10300` observations: 120, 160, 182, 265, 290, 300, 445, 480, 500, 520, 560
- `demo10307` observations: 100, 110, 130, 150, 165, 259, 287, 310, 477, 487, 505, 530, 540
- `demo10302` observations: 78, 273, 302, 454, 495

**Interpretation**

Demo10300 120->160 moves red from z 0.021 to 0.274 m with TCP at essentially the same height and nearly constant 9.5 mm world-x offset; 480->520 does the same for blue with an approximately 8.2 mm offset. Demo10307 110/130/150 and 477/487/505 corroborate attachment formation. After red release, red remains at z 0.036 while TCP retreats and moves elsewhere. These support different dynamical relations for held versus independent objects, though not failure classification. The proposed bias learns consequences of candidate actions rather than just representing geometry or recognizing elapsed manipulation mode.

**Applicability**

Transfer depends on frictional parallel-jaw attachment, comparable cube mass/compliance and reliable pose tracking. Applies to table pickup and finishing the incoming placement before another pickup. Retain object identity and distinguish the currently attached object from the requested next object; no force sensor is assumed.

**Implications for a future training/inference pipeline**

Condition an original-action DP on causal history of qpos/qvel, tcp_pose, selected/other object poses/goals, role and past actions. Add a trainable action-conditioned rollout model F(history, action_chunk) predicting each object's next pose and an attachment probability. Encode a two-regime dynamics prior: attached object uses T_obj(t+k) approximately T_tcp(t+k)*C with slowly varying C; unattached object uses an independent support/stationary branch. Learn TCP rollout as well, using actual future tcp_pose and object poses within each segment as training-only targets. Joint loss combines diffusion denoising, robust multi-step pose loss, relative-transform constancy when co-motion labels support attachment, and a detach label around demonstrated opening; mask impact/contact frames instead of forcing rigidity there. Labels for successful lifting use future object elevation, not future information at deployment. At inference sample a small set of original joint/gripper chunks and rank with the learned model for selected-object elevation with maintained attachment, preservation of already settled red, and lack of premature lateral motion before lifting; fall back to the highest-likelihood diffusion sample when ranking uncertainty is high. Coefficients and fallback threshold require calibration. Decode all candidates through the common action normalizer and controller bounds. Expected effect: better causal discrimination of close-with-object versus close-empty, beyond a phase label alone.

**Assumptions and limitations**

Nearly all demonstrated grasps succeed, so a learned outcome model can be overconfident on misses and has no counterfactual failure labels. Attachment estimates are kinematic proxies, not measured forces. Compare DP with the same auxiliary encoder but no co-motion loss/ranking, and DP with future prediction but no two-regime attachment structure. The hypothesis is falsified if model ranking does not improve early/late overlap transfers or instead favors immobility. Robust slip recovery and safe rejection thresholds cannot be established here.

### Handoff interface

**Entry conditions**

Causal history of TCP and both object poses, measured finger positions and commands is needed to initialize attachment beliefs. Initial empty entry at TCP z 0.442 m or late red descent with fingers about 0.0183 m while requesting blue is supported. At blue grasp entry, red should be treated as an independent settled object around z 0.036 m, not the attached payload merely because it is at goal.

**Exit conditions**

Selected block has risen with TCP and the relative transform stays consistent over the recent lift, fingers remain near the demonstrated grasp width with command -1. Observed endpoints are elevated and already translating; readiness should depend on co-motion confidence, not height alone. If transferred before confirmed attachment, expose that uncertainty rather than report successful pickup.

**Failure signatures**

Closed fingers with selected z stuck at 0.020 m; TCP rises without selected block; relative transform drifts abruptly; incoming red rises during an intended empty retreat; model predicts attachment to both blocks. Do not interpret oscillatory finger qvel alone as a slipping object.

**Overlap role**

Both policies see empty descent, gripper closure, first object motion and lifted translation over 5.3-6.1 s, enabling evaluation before/after attachment. In the reverse 7.45-7.70 s overlap both learn the red attachment breaking, settling to 0.036 m and remaining stationary while TCP travels toward blue.

**Successor readiness**

Provide selected identity, goal, qpos/qvel, current estimated TCP-to-payload transform and uncertainty, or enough causal history to re-estimate it. Delivery continues closed-finger transport only when supported, or completes the remaining close/lift portion from shared training. A proposed confidence gate must be validated; this dataset supports nominal attachment transitions only.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
