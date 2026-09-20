# Deliver and disengage

Complete any ongoing low approach/grasp/lift, carry the nominated block to its own pad, lower it into the supplied geometric goal region and learn physical release and retreat. If another block remains, continue the noninterfering high approach toward it; otherwise retain the terminal settling context while reporting task completion according to the given one-observation contract.

## Segmentation

Each trajectory supplies red and blue occurrences grouped by the same loaded transport, object-goal placement and disengagement problem. Each starts before contact: red entry TCP z is about 0.046-0.098 m with fingers open; blue entry TCP z is about 0.044-0.135 m. Thus delivery learns the remainder of low approach, closure, lift and establishment of transport rather than assuming acquisition returns a perfectly formed grasp. It then learns elevated transport, goal-aligned descent, release and retreat. Red occurrences extend well past geometric goal achievement through the empty high approach toward blue; the next acquire occurrences start while red is still held and descending. Red delivery endpoints are 425,417,423,409,428,425,430,413,432,427,429,435 for demo10000 through demo10011 respectively, chosen from inspected high-approach states rather than equal-duration cuts. Blue delivery occurrences continue to the original final observations 773,765,770,756,770,771,773,760,775,773,774,777, preserving release, settling and terminal context. Representative front images at demo10000/150 and /300, demo10003/409 and demo10011/520 and /777 support the physical interpretation. Every segment start and stop observation was inspected. The sampled final poses satisfy the supplied separate-pad containment and height contract; no drawer exists and no additional sustained-hold/release criterion is imposed.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo10000`: [80, 425)
- `demo10000`: [470, 773)
- `demo10001`: [78, 417)
- `demo10001`: [463, 765)
- `demo10002`: [79, 423)
- `demo10002`: [468, 770)
- `demo10003`: [76, 409)
- `demo10003`: [455, 756)
- `demo10004`: [81, 428)
- `demo10004`: [472, 770)
- `demo10005`: [80, 425)
- `demo10005`: [470, 771)
- `demo10006`: [82, 430)
- `demo10006`: [474, 773)
- `demo10007`: [77, 413)
- `demo10007`: [459, 760)
- `demo10008`: [83, 432)
- `demo10008`: [477, 775)
- `demo10009`: [80, 427)
- `demo10009`: [473, 773)
- `demo10010`: [81, 429)
- `demo10010`: [475, 774)
- `demo10011`: [84, 435)
- `demo10011`: [480, 777)

## Heuristic 1

Grasp-frame transport with explicit detachment: while grasped, the block and TCP approximately share a rigid transform, but that dependency must end at release. Hypothesis: diffusing the desired object trajectory and compensating the measured grasp offset improves goal placement and prevents inconsistent wrist/object motion.

**Evidence inspected by the API**

- `demo10000` observations: 80, 100, 107, 110, 150, 170, 185, 200, 250, 265, 280, 290, 296, 300, 350, 425, 470, 504, 506, 550, 600, 650, 690, 750
- `demo10003` observations: 76, 105, 181, 195, 255, 285, 335, 409, 455, 495, 558, 585, 679, 756
- `demo10011` observations: 84, 116, 192, 206, 272, 309, 435, 480, 520, 560, 584, 610, 710, 777

**Interpretation**

In demo10000, red TCP-object world X offset is about -9.56 mm at /150, /200 and /280 despite large changes in position and height. Blue shows roughly -8.55 mm at /550 and /600, not the identical red offset. Similar coupling appears in demo10003 and demo10011. The object drops from a held centre slightly above z=0.02 m to the table when fingers open, while subsequent TCP retreat does not move it. This supports a conditional rigid-attachment representation and explains why predicting the object goal instead of merely a TCP goal can help. The alternative is that direct joint replay already captures a nearly fixed grasp; the ablation tests whether explicit compensation helps rather than just adding parameters. This prior changes the geometric latent and decoding, unlike D2's outcome-energy bias and D3's phase/duration model.

**Applicability**

Use for approximately rigid transport of a cube in a top-down Panda finger grasp, with reliable causal TCP and block poses. World gravity, robot configuration and the measured grasp offset must be retained. A free/contact transition model is needed because this skill can start before grasp and ends after release. It is not a policy for deformable objects, rolling grasps or arbitrary reorientation under contact.

**Implications for a future training/inference pipeline**

Candidate D1: attachment-conditioned grasp-frame Diffusion Policy. Causal inputs are qpos/qvel, tcp_pose, red_pose, blue_pose, their goals and carried/next roles. Estimate an attachment probability and B=T_tcp^-1*T_object from recent measured world poses and finger state; use a learned filter whose supervision comes from demonstrated co-motion and later successful lift, with future confirmation used only in training labels. While attached, high-level diffusion generates desired object-pose increments relative to its goal and current pose; construct the coupled TCP reference by T_tcp_ref=T_object_ref*B^-1. Retain world Z and robot joint state. Before attachment and after detachment, a free-TCP branch generates references directly so the full expanded segments remain trainable. A learned conditional action diffusion decodes references plus live state into seven absolute joint targets and one gripper command; no unprovided analytic IK is required. Train object/TCP reference diffusion losses, original-action denoising loss, and a masked soft rigidity loss on predicted relative transforms during held phases. Add modest orientation continuity, not a new upright-task predicate. Future object and TCP trajectories are training targets only; deployment predicts them from the causal filter and replans short action prefixes with updated B. Do not constrain the resting object to follow the retreating wrist. Use the one original-data normalizer for all raw fields/actions. Test against a parameter-matched direct action DP and evaluate object-centred versus TCP-centred placement error, grasp-offset stability and release behavior separately.

**Assumptions and limitations**

Approximately constant grasp transform is supported only after successful closure; the 8-10 mm TCP-to-object offset is not a universal grasp calibration. Simulator compression, small slip and pose noise may make hard equality harmful, so use a soft constraint and an explicit release switch. Falsify with raw-state action DP versus grasp-frame DP, and remove the attachment mask or offset compensation separately. Expected improvement is reduced object placement error and slip-like predictions across different grasp offsets, not demonstrated recovery from real slip. No force/friction identification or failed-drop recovery can be learned reliably from these successful episodes.

### Handoff interface

**Entry conditions**

Delivery accepts a specified carried/requested-object ID and own goal, optionally a next-object ID. In the earliest overlap states, fingers are 0.04 m and TCP is still descending near the block: demo10000/80 TCP z=0.066 m and /470 z=0.128 m; demo10011/480 z=0.044 m. No rigid attachment is assumed at such entries. Intermediate closing or partially lifted entries are supported by the same shared interval. At a usual elevated handoff, the block and TCP are near z=0.28 m with fingers near 0.0183 m and command -1; initialize the grasp transform from current and recent observed poses rather than a nominal zero offset.

**Exit conditions**

Deliver the object's centre and rotated corners into its own pad, then learn the demonstrated opening and noninterfering retreat. For red occurrences the useful end state is red at z≈0.02 m on its pad, fingers≈0.04 m and TCP moving high toward blue at z≈0.266-0.270 m. For blue terminal occurrences, both blocks are at their goals and fingers≈0.04 m, with TCP near z=0.299 m after retreat/settling. Placement can meet the task contract before release; these disengaged states define the skill's useful interface, not extra task requirements.

**Failure signatures**

A drift in T_tcp^-1*T_object during claimed attachment, object descending while TCP remains high, or an object following the wrist after the release model has declared detachment indicates invalid coupling. Driving TCP itself to the pad centre without compensating the object offset can bias placement. Continuing to constrain the released object rigidly to TCP during retreat would lift it again or suppress retreat. Thresholds for transform residual and posterior confidence require calibration beyond these successes.

**Overlap role**

Incoming pickup overlap trains free approach, physical finger closure, attachment estimation and lifting before rigid transport dominates. Outgoing red-to-blue overlap trains goal-aligned lowering, opening, detachment, retreat and high empty movement toward blue; the following acquire policy learns this same bridge from the earlier still-loaded state. The model must disable its rigid constraint after observed detachment and maintain the already placed object rather than drag it along with the empty TCP.

**Successor readiness**

The next acquire policy receives red as predecessor and blue as requested object with live world poses, goals, qpos/qvel and causal history. It may take over while red is still descending inside the long shared interval, provided it can finish release and separation. At the later empty high-approach end state it should continue toward blue without moving red. Pass an attachment estimate only as optional inferred context; raw observations remain the interface for candidates with different internal states. With no next object, report the actual task-contract predicate and the separate disengagement readiness, not a required extra hold.


## Heuristic 2

Optimize the actual containment contract, not an exact replay point: separate geometric task completion from physical disengagement and preserve previously placed blocks. Hypothesis: outcome-aware diffusion favors actions that satisfy the rotated-XY and height tolerances while retaining the release/retreat interface taught by the demonstrations.

**Evidence inspected by the API**

- `demo10000` observations: 200, 250, 265, 280, 290, 294, 296, 300, 305, 350, 425, 600, 630, 650, 675, 690, 691, 692, 700, 720, 750, 773
- `demo10003` observations: 195, 255, 285, 335, 409, 585, 679, 716, 756
- `demo10011` observations: 206, 272, 309, 435, 610, 710, 750, 777

**Interpretation**

The task accepts a region, not one TCP pose or an exact cube centre. Demo10000 red becomes geometrically eligible near /290 while still grasped, then settles after /296; blue at /690 similarly satisfies the stated geometry before release. Red remains unchanged while blue is moved. Demo10003 and demo10011 have slightly different final placements and timing but satisfy the same contract. This suggests a bias toward object containment, height and preservation of completed objects, with readiness learned separately. It addresses misalignment between action reconstruction and task outcome rather than D1's grasp geometry or D3's temporal sequencing. Because no failures were supplied, the analytic predicate is more defensible than claiming a learned success boundary from positive-only data; using it inside a learned rollout is still an uncertain research hypothesis.

**Applicability**

Use when the exact supplied object dimensions and separate pad goals are known, world poses can be measured and the robot/controller matches the training data. The analytic contract transfers to goal geometry changes more directly than joint replay, but moving pads, clutter, new heights and unseen dynamics have no empirical support here. Keep a separate task-completion output and manipulation-readiness output.

**Implications for a future training/inference pipeline**

Candidate D2: contract-aware outcome-guided action Diffusion Policy. A conventional causal state encoder takes qpos/qvel, TCP and both object world poses, goals and object roles, and conditions diffusion directly on normalized eight-channel absolute targets. Train an action-conditioned recurrent rollout head F(h_t,a[t:t+H]) to predict future object/TCP poses and finger apertures from the same observed histories and demonstrated action chunks. Supervise only with real synchronized futures inside retained segments, using pose, rotation and aperture losses; also predict placement-event probability and a separate disengagement-readiness signal from causal history. For geometry, form each cube's eight corners p+R*v, v∈{±0.02}^3 m. Per-axis margin is 0.06-max_corner|corner_xy-goal_xy|; height margin is 0.011-|p_z-goal_z|. At training, penalize negative predicted margins at future-labelled placement events, and drift of the already released completed block; do not require exact centre matching or constrain height to the goal during airborne transit. Combine these auxiliaries with action diffusion loss. At inference, evaluate a small set of denoised chunks or take bounded denoising energy steps using the learned rollout: E=sum squared negative placement margins + protected-object loss + a strong deviation penalty from the unguided diffusion sample. Gate placement terms with the predicted causal event probability; if placement lies beyond the horizon, preserve the BC prior instead of rewarding an impossible immediate goal. Apply the analytic contract to LIVE observations for completion, never to rollout predictions alone. Opening/retreat comes from action imitation and the separate readiness head, not extra contract requirements. Geometry operates in physical world metres/wxyz rotations; actions are decoded and clipped to the supplied controller bounds with the shared original-data normalizer. No external simulator or novel action-outcome labels are assumed. Prediction: fewer contract-specific near-misses and less disturbance of the first placed block than pure pose replay, if the rollout remains calibrated.

**Assumptions and limitations**

All trajectories succeed and end near pad centres; they do not supply negative examples that validate a learned success classifier or counterfactual action model. Use analytic geometry plus supervised rollout regression, not invented failure labels. Rollout-guidance extrapolation can be exploited by the sampler and may trade off acquisition for apparent goal progress. Compare identical DP with no guidance, centre-distance guidance and full rotated-corner/height guidance; separately ablate protected-object terms and report prediction calibration on held-out original trajectories. Claimed tolerance to shifted goals or perturbed grasps is a proposed future test, not established performance. No recovery after falling outside the pad is demonstrated.

### Handoff interface

**Entry conditions**

Accept all demonstrated incoming acquisition states, including open low approach and partially established grasp, with carried-object/next-object roles and exact world goals/dimensions. Early in transport, the goal margin is intentionally negative because object z is about 0.28 m; that must not trigger an immediate descent shortcut. Only enable placement-event outcome guidance when the learned causal event predictor expects placement in the sampled horizon. In the blue occurrence red is already on its own pad and should be treated as protected after release, using observed pose/finger separation rather than a future completion label.

**Exit conditions**

Compute each at_goal from all rotated cube XY corners inside ±0.06 m about the appropriate goal and centre-height error≤0.011 m about 0.02 m; report the conjunction at one observation. Demo10000/690 provides supported completion while blue remains in a -1 grip at z=0.02284 m, with red already placed. Also learn the subsequent opening, settling and retreat to the useful skill end state: terminal TCP≈0.299 m, fingers≈0.04 m and both objects≈0.02 m. Do not redefine task success to require release, clearance, low velocity or a sustained hold.

**Failure signatures**

Positive centre-only success with a rotated corner outside the pad, predicted success while object height is above tolerance, loss of red containment while placing blue, or energy-guided actions moving far outside demonstrated target patterns are failures. A satisfied contract but still loaded gripper is task completion, not necessarily readiness for an empty-hand successor. If rollout predictions disagree persistently with live pose changes, reduce reliance on guidance rather than treat the prediction as observed success.

**Overlap role**

The long outgoing overlap teaches actual lowering, finger opening, object settling, retreat and next approach, even though the geometric contract may already be satisfied during closed-grip lowering. The acquire successor shares this entire transition and can complete physical disengagement. The incoming overlap teaches how acquisition establishes the transport state; the outcome energy must leave this grasp/lift behavior intact rather than optimize only immediate goal distance.

**Successor readiness**

Provide current per-object contract margins, role IDs, qpos/qvel and TCP/object world poses. For a next acquire call, distinguish 'red contained but still held/descending' from 'red released and TCP separating'; both are supported in the shared bridge, but require different actions. The successor must keep the completed block inside its pad while opening and moving away, and must not treat a predicted future margin as a measured one. With no successor, one observed conjunction is sufficient under the supplied contract; optional continued disengagement uses the learned demonstration behavior.


## Heuristic 3

Event progress with elastic duration: the order of manipulation events is more stable than their exact timestamps, and stationary-looking poses can belong to different phases. Hypothesis: phase-aligned diffusion with a learned duration model handles supported handoff timing variation without replaying a fixed episode clock.

**Evidence inspected by the API**

- `demo10000` observations: 80, 100, 106, 107, 110, 125, 150, 170, 185, 250, 280, 290, 296, 305, 315, 350, 365, 425, 504, 505, 506, 570, 690, 691
- `demo10000` observations: 692, 693, 700, 720, 750, 773
- `demo10001` observations: 78, 108, 168, 185, 260, 280, 300, 345, 417, 463, 500, 530, 565, 580, 690, 730, 765
- `demo10011` observations: 84, 116, 192, 272, 309, 359, 435, 480, 520, 560, 584, 610, 710, 750, 777

**Interpretation**

Many near-stationary states have different intended futures. At demo10000/106 fingers are open and the arm nearly settled before a close command at /107; /170 is an elevated hold before horizontal movement; /290 is near the goal but still closed, /296 starts opening, and /305 remains near the same arm pose with much wider fingers. Blue /691-/693 shows opening over several 0.05 s observations. Demo10001 and demo10011 reach corresponding release/retreat phases at different times. This motivates representing progress through events and durations separately from raw elapsed time, especially when a policy is invoked inside an overlap. It is not simply contact estimation: the candidate explicitly aligns and time-decodes the whole movement including free transport and retreat. Distinctive tests must show better temporal continuation, not merely better action denoising from a larger network.

**Applicability**

Use when the manipulation follows the demonstrated partial order: approach/close, lift, transport, lower, release, retreat, then optional next approach. Phase duration may vary, but the controller rate and contact response should be comparable. Causal history and a stable carried/next role assignment must be supplied at takeover; unseen phase reversals and retry sequences are not covered.

**Implications for a future training/inference pipeline**

Candidate D3: event-aligned, duration-elastic Diffusion Policy. Offline annotate approach, closing, lifting, transport, lowering, releasing and empty-retreat intervals using demonstrated command changes, finger positions, TCP/object co-motion and world-height/goal relations; ambiguous event boundaries receive soft labels. These annotations may inspect training futures but are not inference inputs. Resample training reference trajectories onto within-phase progress u∈[0,1], retaining physical duration labels in seconds and original action targets in Panda joint radians. A causal recurrent state encoder predicts phase probabilities, progress, remaining duration and their uncertainty from qpos/qvel, TCP/object poses/goals, roles and past executed commands. Condition a temporal diffusion denoiser on this belief to generate absolute joint-target sequences indexed by progress; a learned positive-duration decoder maps them back to 20 Hz actions, with feedback updates each executed prefix. A phase-conditioned persistent gripper head generates -1/+1 command changes separately from continuous arm interpolation, so closure is not smeared across a dwell. Train action denoising/reconstruction at original timestamps, auxiliary phase/progress/duration losses and a soft ordered-transition regularizer. Mask missing futures at segment endpoints and avoid using raw global trajectory index. Learn dwell length from observations and latent uncertainty, not a hard-coded wait. Use the common original-data normalizer, preserve physical time when deriving velocities, and initialize from random causal overlap starts during training. At inference repeatedly infer progress, denoise remaining local motion, time-decode and reobserve; no future alignment path is supplied. Prediction: more consistent continuation across faster/slower demonstrations and arbitrary supported handoff timing than direct-time DP, distinguishable from D1's geometric coupling and D2's outcome guidance.

**Assumptions and limitations**

The same apparent phase order might simply be a deterministic expert plan, and a recurrent model might memorize elapsed time. The learned duration model must therefore be tested against a fixed-clock DP, a history encoder without progress alignment, and an unwarped phase-conditioned DP. Train/test splits must remain by original trajectory. Test multiple observed overlap entry times and benign temporal resampling only with correctly rescaled qvel/duration labels; do not present resampled data as newly observed physics. Successful demonstrations cannot establish recovery transitions, robustness to contact stalls or a calibrated timeout policy. Excessively monotone progress could hide a missed grasp rather than repair it.

### Handoff interface

**Entry conditions**

Initialize a posterior over phase and normalized progress from recent qpos/qvel, world TCP/object poses, actual finger apertures and past executed gripper commands, plus carried/next roles. Do not set phase to 'transport' merely because delivery has been called. Supported entries include open low approach (demo10000/80, /470), imminent closure (/107 or /504), partially lifted grips and already elevated initial transport. If history is absent, use a broad posterior and a learned observation-based initializer; confidence under such truncation must be evaluated.

**Exit conditions**

The inferred useful final phase is empty retreat/next approach for a nonterminal occurrence, or settled empty retreat for the terminal one. Demo10001/417 is still a moving high approach while red rests at z≈0.02 m and fingers≈0.04 m; demo10011/777 is a nearly stationary terminal TCP at z≈0.299 m, both blocks at z≈0.02 m, fingers≈0.04 m. The posterior must agree with measured finger opening and object separation; neither a fixed elapsed duration nor a high phase index alone establishes readiness.

**Failure signatures**

A phase jump to release with an elevated block, regression to pickup after the block reaches its goal, opening before placement, or indefinitely waiting despite observed aperture/separation indicates an invalid temporal model. Forcing near-zero qvel during every handoff would reject supported moving entries. Failure to distinguish slow target tracking from a dwell can cause repeated targets or phase skipping; the dataset contains no deliberate retries to resolve such cases.

**Overlap role**

Incoming overlap covers the full approach-close-lift transition and early horizontal motion, allowing the successor to infer progress from evolving state. Outgoing overlap starts with a still-grasped descending red block, then traverses release lag, dwell, retreat and high blue approach. Both adjacent policies learn the ordered transition while being initialized at different progress values. The model learns event-conditioned continuation, not a shared fixed episode timestamp or a mandatory switch at an event boundary.

**Successor readiness**

Transfer live state history and role IDs; optionally expose phase posterior for diagnostics, but permit another candidate to reconstruct readiness from raw observations. The next acquire may take over before release during the shared interval and should initialize to predecessor completion, not empty approach. If no next object exists, provide the measured one-observation contract result separately from latent phase completion. The model should continue smooth open-gripper retreat if the caller chooses not to stop at first task success.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
