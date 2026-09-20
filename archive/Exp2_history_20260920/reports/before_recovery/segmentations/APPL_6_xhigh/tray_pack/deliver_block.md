# Deliver a selected block into the tray with grasp and exit context

Finish acquisition if needed, carry the selected block above tray obstacles, lower it into its own marked region and preserve the other block. Learn release, settling and empty-hand retreat; if another block remains, continue the open approach toward it. Satisfy the supplied containment contract while exposing manipulation state separately from formal task success.

## Segmentation

Group red and blue deliveries because the shared problem is moving a potentially not-yet-grasped selected cube into its assigned tray region and leaving a usable manipulation state. Each starts during aligned open-finger descent, not at an idealized fully lifted grasp: red starts 75,76,78,74,73,79,77,72,74,75,71,76 and blue starts 445,447,454,450,443,457,453,439,445,448,438,449. Measured TCP heights span 0.088-0.178 m, blocks are still at source z 0.020 m and fingers near 0.04 m. Include complete closing, vertical lift, early and full transfer, lowering, release and settling. Red segments stop only at the next block's high approach (415,418,427,420,414,428,424,408,416,419,407,420): red z about 0.036 m, fingers open, TCP z about 0.267 m above blue after retreat/traversal. They do not claim to complete the next grasp. Blue delivery segments retain all remaining actions through the supplied final observations, where both objects are contained, released and settled and TCP is open/high. Keeping the terminal holds supports idle behaviour instead of cropping at the first predicate satisfaction. Full rotated containment and z tolerance, rather than opening or clearance, determine formal completion. Individually inspected endpoints and intermediate observations support all these phase ranges. The three candidates intentionally differ in action representation/decoder, fixture-aware predictive computation, and temporal release/readiness inference; none is merely a paraphrase of the expert chronology.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo10300`: [75, 415)
- `demo10300`: [445, 742)
- `demo10301`: [76, 418)
- `demo10301`: [447, 742)
- `demo10302`: [78, 427)
- `demo10302`: [454, 752)
- `demo10303`: [74, 420)
- `demo10303`: [450, 743)
- `demo10304`: [73, 414)
- `demo10304`: [443, 737)
- `demo10305`: [79, 428)
- `demo10305`: [457, 752)
- `demo10306`: [77, 424)
- `demo10306`: [453, 749)
- `demo10307`: [72, 408)
- `demo10307`: [439, 730)
- `demo10308`: [74, 416)
- `demo10308`: [445, 737)
- `demo10309`: [75, 419)
- `demo10309`: [448, 741)
- `demo10310`: [71, 407)
- `demo10310`: [438, 730)
- `demo10311`: [76, 420)
- `demo10311`: [449, 742)

## Heuristic 1

Move the payload, not just the TCP: diffuse goal-relative Cartesian motion and explicitly account for the current grasp transform when learning placement, then decode through local inverse dynamics to absolute joint targets. This should decouple placement precision from modest grasp-offset and joint-route variations.

**Evidence inspected by the API**

- `demo10300` observations: 75, 120, 160, 182, 200, 240, 265, 280, 290, 320, 445, 480, 520, 562, 600, 640, 655, 665, 720
- `demo10307` observations: 72, 110, 130, 165, 178, 259, 287, 439, 477, 505, 530, 552, 632, 641, 646, 659, 700
- `demo10305` observations: 79, 190, 274, 303, 496, 578, 685

**Interpretation**

Red and blue follow analogous high carry and vertical descent despite very different joint vectors. In demo10300, red center at 280 is about [-0.1362,-0.0699,0.0457] while TCP is [-0.1457,-0.0702,0.0455]; blue at 655 has about an 8.2 mm world-x offset instead of 9.5 mm. Demo10307 repeats this nonzero offset. A TCP-only goal loss can therefore solve the wrong geometry. Goal-relative TCP path diffusion with explicit carried transform separates task displacement from arm configuration, and a learned decoder addresses the absolute-joint action contract. The rigid relation is observed only during nominal holding; its validity outside those intervals is an assumption to avoid.

**Applicability**

Top-down carrying and placing with known object/goal poses, the same robot and controller, and small grasp-offset variation. Relative-goal conditioning may transfer to small target/source changes, but fixed tray geometry, world height and robot state remain inputs. Learned inverse dynamics is local to the demonstrated Panda manifold; neither arbitrary robot transfer nor perfect IK is assumed.

**Implications for a future training/inference pipeline**

Inputs are causal qpos/qvel, tcp_pose, both object poses/goals, payload/pending role and past commands. Compute sign-invariant R_tcp, R_obj and online C=T_tcp^-1*T_obj when attached; retain an attachment-validity feature and current world/base state. Train a DP over H=16 future realized TCP translations relative to the selected goal, continuous rotations and original gripper commands. These path targets come from future tcp_pose observations inside the segment (training only); they are sampled, not observed, at deployment. A trainable local inverse-dynamics decoder D(current qpos/qvel, history, sampled path, C, validity, roles) predicts the original normalized seven absolute arm targets for each step, trained against source actions with reconstruction loss; retain current arm configuration/history to select the demonstrated redundancy branch. Joint objective: path diffusion loss plus original-action decoding loss and, on attached frames only, payload-goal consistency using T_tcp_future*C against observed future object poses. For approach/release/retreat, the TCP path remains defined but rigid payload loss is masked. Use global original normalization statistics for physical features and original action decoder outputs; do not fit per-skill or derived-path normalizers. At inference sample a path, decode bounded absolute joint targets, execute 1-2 steps near placement and re-estimate C from new observations. No exact FK/IK model is supplied or assumed. Distinguishing mechanism is action-space geometry and a learned inverse-dynamics decoder, not sample ranking or a phase-hazard controller. Predict improved payload placement under changed but still supported grasp offsets.

**Assumptions and limitations**

Payload and TCP orientations barely vary, so full rotational generalization and learned inverse dynamics far from demonstrated joint configurations are unsupported. Diffusing Cartesian paths plus decoding may introduce off-manifold poses; validate action reconstruction and restrict sampling to learned support rather than claiming exact kinematics. Compare original absolute-joint DP, goal-frame TCP diffusion without attachment-offset conditioning, and the full candidate. Benefit is falsified by no decrease in placement error under small grasp offsets or increased joint-target discontinuity. No collision or recovery guarantee follows.

### Handoff interface

**Entry conditions**

Payload identity/goal token must be available even at pre-grasp entry. Supported open-finger descent entries have selected block z 0.020 m and TCP world z 0.088-0.178 m, fingers near 0.04 m, aligned above source. Later supported entry includes closing, initial lift or elevated transport. The payload-relative transform must be marked invalid until causal grasp/co-motion evidence exists; do not apply carried-block decoding to empty descent.

**Exit conditions**

Selected object is contained at its goal; in demonstrated useful exit it has been released to z about 0.036 m and fingers approach 0.04 m. Red occurrence additionally retreats and approaches blue with TCP z about 0.267 m and empty hand; final blue occurrence retreats to z about 0.3165 m and holds. Task completion may occur earlier near z 0.0455 m while still grasping; this is not equivalent to empty-hand readiness.

**Failure signatures**

TCP reaches the goal marker but cube remains offset/outside containment; inverse decoder yields discontinuous arm targets or moves the wrong null-space branch; estimated payload offset changes during nominal rigid transport; object follows retreat after expected release. These require uncertainty signalling rather than unvalidated correction.

**Overlap role**

Shared descent/close/lift identifies the actual TCP-to-payload offset before transport. Shared final lowering/opening/retreat/next approach allows the next acquire policy to take over while that transform is still transitioning from attached to invalid.

**Successor readiness**

Next acquisition needs the pending identity (blue in these demos), actual finger aperture and causal history indicating whether red is still attached or has settled. It can finish late descent/release if transferred early. At late transfer continue +1 while approaching the pending source, not a closed payload-conditioned motion. Retain goal-relative and world/base states so no coordinate reset creates an action jump.


## Heuristic 2

Rim-aware transport topology: lateral payload transfer should occur in free space above tray walls, while low-altitude motion is reserved for aligned pickup or insertion. Fixture-conditioned predictive ranking of diffusion samples should preserve this dependency when goals or obstacle heights change modestly.

**Evidence inspected by the API**

- `demo10300` observations: 120, 160, 182, 200, 240, 265, 280, 290, 320, 360, 400, 480, 520, 562, 600, 640, 655, 665, 680, 720
- `demo10307` observations: 110, 130, 150, 165, 178, 259, 287, 310, 335, 365, 477, 505, 530, 540, 552, 632, 641, 659, 680, 700
- `demo10302` observations: 78, 188, 273, 302, 427, 495, 576, 684

**Interpretation**

The rim tops are 0.09 m from declared fixtures. Demo10300 lifts near source to center z 0.274-0.282, carries across to z about 0.309, then descends over the region to 0.0457. Blue does likewise; demo10307 supports the same lift/carry/lower topology. After release TCP rises before traversing to the next block while the first remains still. These regularities suggest free-space topology and payload clearance, but do not prove the reason for the waypoint. A raw behavioral clone has little evidence for how to change trajectory around a changed obstacle. Explicit geometry energy gives a falsifiable inductive bias on that gap, separate from pose-frame decoding and event-memory inference.

**Applicability**

Open tray with known fixture boxes, upright world gravity and a held cube that should clear the rim during lateral transit. Transfer to altered tray geometry requires that geometry be supplied correctly and appropriate training/evaluation; only one tray is observed. Requires credible action-conditioned pose predictions and preserves the other packed block as an obstacle.

**Implications for a future training/inference pipeline**

Inputs: causal state history qpos/qvel, tcp_pose, object poses/goals, role tokens, plus the five supplied world-frame axis-aligned fixture boxes. Condition an original normalized eight-action joint-space DP on these features. Train an auxiliary action-chunk-conditioned rollout network predicting future TCP and both object poses; future observations supply training targets only. For each predicted attached cube, transform its eight +/-0.02 m corners by predicted orientation, assess intersection/signed distance to fixture boxes and the other cube. Define an energy E=sum_k collision_cost + misplaced_descent_cost + terminal_containment_error, with mode weights learned from causal contact/height features and demonstrated motion, not a hard time schedule. In free carry, lateral rim crossing penalizes cube-bottom clearance below rim top 0.09 m plus a tunable margin; near the goal, turn off the free-carry height term and allow downward insertion. Also penalize motion of a previously settled other block. Train denoising and multi-step pose prediction; test optional soft energy fine-tuning separately. Deployment samples several action chunks, rolls them out and selects low-E high-likelihood candidates, rejecting guidance when rollout uncertainty is high; execute a short prefix and update. Use bounded original absolute joint/gripper decoding, not Cartesian commands. A TCP proxy can encourage empty-hand retreat clearance, but full gripper/link collision checking is unavailable from the supplied fields. Distinguishing bias is fixture-aware planning/ranking rather than changing action coordinates or release memory.

**Assumptions and limitations**

The rim-crossing explanation is plausible but the demonstrations could simply use a generic high waypoint even without obstacles. No collisions or alternative-height successful routes are present. Test matched raw-joint DP, geometry-free future-model ranking, and fixture-conditioned ranking under controlled rim-height/goal changes; benefit should be strongest near changed obstacles, not just at nominal starts. Cube/TCP geometry does not certify arm or gripper clearance and the model can exploit inaccurate predictions. Off-demonstration rerouting and recovery remain unsupported.

### Handoff interface

**Entry conditions**

Open aligned descent, closing or attached lifting states inside acquisition overlap are supported; selected source z is about 0.020 m before lift. Geometry model needs payload and pending identities, both poses, supplied fixture boxes and current robot state. Clearance cost must be mode-conditioned so it does not forbid intentional source approach or final tray lowering.

**Exit conditions**

Predicted and then observed payload is inside its intended region, released in nominal demonstrations to center z 0.036 m; TCP retreats vertically before long lateral movement. Red exit at the blue overhead approach has TCP z about 0.267 m with open fingers; final exit holds at z 0.3165 m. These are demonstrated routes, not certified safe configurations of every link.

**Failure signatures**

Cube bottom or oriented corners approach a rim box during lateral carry; descending before entering the tray opening; already placed red moves while packing blue; learned rollout disagrees with observed object motion; geometrically clear TCP trajectory nonetheless produces joint/link collision. The last failure is outside this simplified geometry prior's guarantee.

**Overlap role**

The 5.3-6.1 s approach/close/lift overlap teaches when a cube transitions from supported-at-source to a payload that needs rim clearance. The 7.45-7.70 s outgoing overlap teaches final descent, detach, vertical retreat and open-hand traversal, not merely containment at one frame.

**Successor readiness**

Pass role, actual measured state/history and whether payload remains attached; predicted clearance alone is insufficient. If successor takes over early, it must retain descent/release mode and avoid lateral retreat until detach; late handoff is to an open gripper over blue. No numeric safety margin beyond observed paths is validated; the selection agent must test any chosen clearance buffer.


## Heuristic 3

Containment is not disengagement: learn release hysteresis and separate task-success and empty-hand readiness beliefs. A recurrent event-conditioned DP should preserve placed objects and avoid premature next-skill control even when the formal goal becomes true before fingers open.

**Evidence inspected by the API**

- `demo10300` observations: 265, 280, 290, 300, 320, 360, 400, 640, 655, 665, 680, 720
- `demo10307` observations: 259, 287, 310, 335, 365, 632, 641, 646, 659, 680, 700
- `demo10301` observations: 267, 293, 418, 668
- `demo10305` observations: 274, 303, 428, 685

**Interpretation**

Demo10300 280 and 655 already have object centers near 0.0455 m, within the contract's 0.011 m z tolerance, but command remains -1 and fingers are about 0.0183 m. At 290 and 665 fingers are still opening while objects have settled near 0.036; TCP initially stays near 0.0453 before retreat. Demo10301 293 is only the first open command at a still-closed observed state. Demo10307 641/646 to 659/680 confirms the command-to-state lag and subsequent decoupling. Thus task truth, gripper command and usable manipulation interface are different quantities. Separating their temporal beliefs addresses handoff failures that neither geometric action decoding nor obstacle-aware ranking inherently resolves.

**Applicability**

Applies when release commands precede measured finger opening and payload settling, and successful containment can precede full disengagement. Requires the same controller or relearned temporal dynamics, causal finger and object observations, and an intended/pending role interface. Does not require zero velocity or sustained hold for task success.

**Implications for a future training/inference pipeline**

Use causal 8-frame qpos/qvel, tcp_pose, both object poses/goals, past actions and role tokens. A recurrent belief encoder predicts attached, opening, released/settling and empty-retreat probabilities, plus two distinct heads: contract containment computed from current rotated cube corners and center z, and manipulation-readiness learned from future opening/co-motion loss and subsequent object persistence in training. Pseudo-label readiness using demonstration future finger expansion, object settling and TCP/object decoupling; mask ambiguous periods and never provide those futures at deployment. Condition an original-action H=16 diffusion denoiser on the belief; add CE/calibration losses, release-event hazard likelihood and a soft consistency penalty against re-closing during the demonstrated detached-retreat branch. Hysteresis is probabilistic persistence of the contact belief, not a scripted fixed delay; allow uncertainty updates from actual aperture and relative motion. Decode to bounded absolute joint/gripper targets with the global normalizer. At inference filter from current/past observations, sample actions, execute 1-2 steps near release and replan, continuing the learned retreat/hold after the success predicate if the caller needs a reusable empty-hand state. No timestamp or demo ID is an input. Expected effect is fewer premature successor switches and less close/open chatter without changing the contract's success definition.

**Assumptions and limitations**

Only successful release/settle trajectories are available; sustained stiction, bouncing out and regrasp failures are not represented. Learned hysteresis can be too conservative or can freeze the wrong belief. Compare a memoryless DP, a recurrent DP without separate containment/disengagement heads and this candidate at randomized outgoing-overlap entry times and perturbed release latency. Falsify benefit if readiness calibration or post-release object preservation does not improve, or if needless dwell increases without fewer transfer errors. A learned classifier is not proof of contact safety.

### Handoff interface

**Entry conditions**

Can start with open fingers approaching source at the measured delivery entry heights or at any supported close/lift/carry/lower state. Initialize a causal recurrent buffer or trained missing-history state. Keep payload identity latched even if red_at_goal/blue_at_goal becomes true while qpos fingers remain about 0.0183 m and gripper command is -1.

**Exit conditions**

Report contract success separately from manipulation readiness. Demonstrated readiness progresses through closed hold near object z 0.0455 m, open command, fingers increasing toward 0.04 m, object settling to 0.036 m, then TCP retreat. Red useful end is empty open approach above blue at z about 0.267 m; final blue useful end is open retreat/hold at 0.3165 m. A caller may stop on contract success without these extra requirements, but must not label that state an empty-handed acquisition handoff.

**Failure signatures**

Goal flag true but cube rises with retreat; open command issued but fingers remain near grasp width; policy re-closes after opening or switches payload identity before settling; object exits containment after release; repeated phase changes from noisy finger qvel. Readiness should decrease under these observable discrepancies, not be forced by elapsed time.

**Overlap role**

Both adjacent policies learn the entire pre-release lower, close-held goal entry, opening transient, settling, vertical retreat and next approach. Early successor transfer may legitimately receive a held red cube at z 0.084-0.169 m while requesting blue, not an already empty hand. The incoming approach/close/lift overlap also prevents the delivery recurrent state from assuming every entry is held.

**Successor readiness**

Pass intended/pending role, history and separate estimates of containment, attachment and disengagement. Next acquisition can continue lower/open when attachment remains likely, then maintain +1 during retreat/approach; it must not jump directly to closing on blue. Supported partial opening example is demo10300 290 with fingers 0.0371 m, red z 0.036 and TCP z 0.0453 m. Any fixed wait or velocity threshold beyond these observations is a hypothesis to test, not part of task success.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
