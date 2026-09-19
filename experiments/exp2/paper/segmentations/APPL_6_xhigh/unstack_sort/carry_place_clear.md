# Carry, place and clear a selected block toward its goal

Complete an in-progress grasp/lift if necessary, transport the selected block to its own marked pad, lower it, release it and clear the gripper. After red placement, continue toward the exposed blue pickup; after blue, retain the final release, retreat and settling. Judge task achievement by full rotated XY containment and center-z tolerance for both objects, while maintaining a separate physical handoff state.

## Segmentation

Group red and blue by the held-object transport/placement problem, preserving goal identity and next-object context instead of identifying skill with a pad side. Each segment starts before pinch closure during final aligned descent, includes full closure/lift, transport, goal descent, opening and clearance. This deliberately trains takeover while acquisition is unfinished. Red segments extend well beyond pad attainment and release through vertical retreat, horizontal return and approach toward blue; they end before blue closure, at measured open approach heights individually varying from 0.055 to 0.222 m. Blue segments include all remaining actions to the original trajectory endpoint, including settling and open retreat to TCP world z about 0.300 m with nearly zero final arm velocity. Red remains on its pad throughout that last part. Endpoints of all ranges and all original terminal observations were inspected. Final object centers are near (-0.241,-0.248,0.020) and (-0.241,+0.248,0.020) m, with near-identity rotations, well inside the full rotated-containment regions. Front images at demo10204 721 and demo10200 280 support the interpretation. Completion requires red_at_goal AND blue_at_goal for one observation; release/settling/clearance are not additional task requirements but are retained to support interfaces and useful final states. Role tags red/next-blue and blue/no-next must be persistent caller context, not reassigned from a single goal predicate. All three priors are separately trainable Diffusion Policy designs with distinct representation, generative timing or objective/inference changes; compare at the same data/normalization budget and test performance only in the later stage.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo10200`: [80, 430)
- `demo10200`: [440, 732)
- `demo10201`: [75, 410)
- `demo10201`: [425, 717)
- `demo10202`: [80, 422)
- `demo10202`: [438, 729)
- `demo10203`: [78, 420)
- `demo10203`: [436, 728)
- `demo10204`: [77, 418)
- `demo10204`: [434, 721)
- `demo10205`: [79, 421)
- `demo10205`: [437, 723)
- `demo10206`: [81, 424)
- `demo10206`: [440, 729)
- `demo10207`: [84, 434)
- `demo10207`: [450, 739)
- `demo10208`: [82, 430)
- `demo10208`: [446, 737)
- `demo10209`: [76, 419)
- `demo10209`: [435, 723)
- `demo10210`: [80, 426)
- `demo10210`: [442, 730)
- `demo10211`: [78, 417)
- `demo10211`: [433, 721)

## Heuristic 1

Transport the object, not the TCP: represent goal error through a causally estimated rigid grasp transform, with attachment-aware masking. This should share goal-directed carrying across red/blue and retain correct placement despite small TCP-to-object offsets.

**Evidence inspected by the API**

- `demo10200` observations: 80, 100, 150, 200, 250, 265, 280, 300, 390, 450, 500, 550, 600, 650, 700
- `demo10204` observations: 77, 100, 130, 160, 260, 270, 290, 310, 350, 385, 460, 480, 505, 570, 605, 625, 632, 642, 670, 690
- `demo10207` observations: 84, 184, 242, 290, 450, 476, 550, 662

**Interpretation**

In demo10200 200/250, red x is about -0.246/-0.241 m while TCP x is about -0.255/-0.250 m; in the blue transport at 550/600 the offset is again about 0.0088 m. Demo10204 repeats that mapping on the opposite goal direction, and after release at 642->670 the object stays while TCP rises. Thus object transport has a stable grasp-frame relation only in the held phase. Learning that relation can reduce a systematic object-goal error that a TCP-centric representation would miss. This is a spatial/kinematic factorization, distinct from P2's event timing and P3's task-region objective.

**Applicability**

Same block shape, parallel grasp, world-frame object/TCP observations and joint-position controller. The hypothesis covers small grasp-offset/source changes and opposite pad directions while retaining Panda configuration/reachability. A call supplies selected-object and next-object tags (red then blue; blue then none); tags persist through the segment even when the selected object reaches its goal. No unseen deployment sensing or arbitrary scene symmetry is assumed.

**Implications for a future training/inference pipeline**

Candidate P1: grasp-transform-conditioned Diffusion Policy. Inputs: qpos/qvel, TCP and both object world poses (m,wxyz), both goals, selected/next role tags, and a short causal pose history. Use SE(3) relative transforms X_g = inverse(T_tcp)*T_selected and selected-object-to-goal translation; keep quaternion sign invariance. A learned attachment confidence masks X_g during open approach and after release. Encode selected/other/next tokens with shared weights plus a global qpos/absolute TCP branch, then condition an action-chunk diffusion denoiser over globally normalized original 8D absolute actions. Add an auxiliary held-phase loss predicting object displacement from TCP displacement using X_g, supervised by demonstration future poses; the target future transform never enters the inference encoder. The object-goal channel must remain active after the TCP begins retreat so completed-object stability is represented. At deployment estimate X_g only from causal states, denoise and replan at 20 Hz, preserving original pd_joint_pos command semantics and limits. Use the mandated one full-demonstration normalizer, geometry in denormalized metres and rotation matrices, no per-color/skill normalizer. Hypothesis: a held-object representation factors grasp placement from destination, improving placement error for varying grasps compared with assuming the TCP equals the object center.

**Assumptions and limitations**

Grasp offsets vary little (roughly 9 mm world x and small z differences), so this dataset cannot establish robustness to a markedly skewed grasp, rotations or heavy objects. Rigid attachment is an assumption, not measured force closure. Compare against a TCP-to-goal-only encoder and a full flat-state encoder with equal parameter count; perturb source/grasp offsets only in later allowed evaluation. Expected failure is placing TCP on the pad while the object violates containment. This prior should reduce object placement bias, not guarantee slip recovery or robot-base equivariance.

### Handoff interface

**Entry conditions**

Supports the observed open aligned pickup states as well as closure/lift/held states in the shared band: red segment starts TCP z about 0.064-0.101 m, blue starts about 0.021-0.089 m, qpos[7:9] about 0.040 m before closing. For already-held entry, fingers are near 0.01825 m and the object moves with TCP. The relative-grasp transform branch must be masked until coupling is observed; proximity before closing is not attachment.

**Exit conditions**

Selected object is at its pad (world z about 0.020 m after release, XY within a few millimetres of target in these demos), g=+1 and qpos fingers returning to about 0.040 m. For red, continue retreat and return toward blue through the observed endpoint approach band TCP z=0.055-0.222 m with nonzero qvel. For final blue, retain settling and open retreat to TCP z about 0.300 m, then near-zero arm motion at the recording end.

**Failure signatures**

Object-goal error stops decreasing while TCP-goal error shrinks, indicating an incorrect grasp offset; object follows retreat after opening; fingers remain pinched as TCP moves toward blue; or red leaves its pad during blue handling. Transform discontinuities after release must not be interpreted as a changed but still valid grasp.

**Overlap role**

Acquisition/carry overlap trains estimating attachment through final approach, closure and full lift before goal transport. Red carry/blue acquisition overlap learns held descent, placement, release, empty retreat and next approach. Both retain target and next-object roles instead of reassigning the target at the first goal hit.

**Successor readiness**

Blue acquisition can receive a red still descending/held in the long overlap or an open gripper approaching blue; it needs selected/previous identity, current state and causal grasp-history evidence. At final blue completion there is no trained successor: both pad predicates are already satisfied, and open/clear/settled states are retained for usability, not imposed by the task contract. Spatial/contact tolerances outside demonstrated states remain unvalidated.


## Heuristic 2

Separate smooth arm motion from discrete, state-dependent gripper events. A phase-local hybrid diffusion model should learn persistent hold/open decisions and PD tracking lag rather than averaging gripper modes across placement and retreat.

**Evidence inspected by the API**

- `demo10200` observations: 90, 100, 150, 200, 250, 265, 280, 300, 340, 390, 440, 450, 465, 500, 600, 650, 700
- `demo10204` observations: 100, 130, 160, 260, 270, 280, 290, 310, 350, 385, 460, 480, 505, 570, 605, 625, 632, 642, 670, 690
- `demo10201` observations: 75, 170, 225, 270, 425, 455, 525, 640

**Interpretation**

The expert holds g=-1 across large joint motion and the low-height dwell (demo10204 625/632) and then g=+1 at 642 while measured fingers approach their target; retreat follows at 670. Red similarly has a held low pose at 270, partial opening at 280 and open retreat at 310. At demo10200 250, commanded joints lead measured joints during descent, so a setpoint is not an instantaneous Cartesian pose. An event-aware, response-aware sequence model is a distinct explanation for smooth handoffs and addresses diffusion's tendency to average or chatter across contact events. This is not evidence that a fixed timestep schedule is sufficient.

**Applicability**

For this 20 Hz pd_joint_pos interface, near-binary expert gripper commands and the observed approach/close/lift/transport/descend/release/retreat ordering. Absolute joint targets must remain distinct from measured qpos; gripper -1 corresponds to an attempted finger target of -0.010 m, not the measured 0.018 m grasp width. Robot/contact dynamics must be rechecked for another controller.

**Implications for a future training/inference pipeline**

Candidate P2: phase-local hybrid Diffusion Policy with separate arm and gripper generative channels. A causal recurrent encoder consumes a short masked history of all nonconstant state fields, role tags and previously issued commands. Infer soft monotone phases approach/close/lift/carry/descend/open/retreat/next-approach, with next-approach absent for the last block. Train a continuous diffusion head for seven absolute arm-joint targets and a categorical diffusion head for g in {-1,+1}, jointly conditioned on phase; add duration/event-consistency regularization to discourage unsupported rapid reversals without prescribing fixed durations. Supervise phase/event labels using action changes and future demonstrated object following/release, as training-only soft targets. Include an auxiliary PD response head predicting qpos_{t+1}-qpos_t and finger opening from the command/state, with loss on measured futures, to distinguish setpoints from motion. At inference infer phases causally, sample coupled heads, map gripper categories through the same original action normalizer and controller, and execute a short receding prefix. Retain full original-data normalization for observations and action scales, and output only the contracted 8 fields at 20 Hz. This candidate changes temporal organization and discrete/continuous decoding; it need not use P1's relational grasp encoder or P3's goal energy. Testable prediction: fewer event reversals and lower transfer jerk without delaying pad attainment.

**Assumptions and limitations**

Successful traces strongly confound phase with elapsed time and provide little event-duration variability. Semi-Markov regularization can overconstrain legitimate faster transitions. Gripper categorical output is motivated by inspected +/-1 actions but cannot establish optimality of binary control for other objects. Compare phase-aware mixed diffusion to ordinary continuous 8D diffusion and to phase-agnostic binary-gripper diffusion; also remove duration regularization. Expected gains should be reduced release chatter and improved action continuity, not just memorized timing. No interrupted placement or failed-release data establish recovery.

### Handoff interface

**Entry conditions**

Use a history-informed phase belief initialized from current/past qpos, qvel, TCP/object poses and the invocation's persistent target tag. Open approach, closing-in-progress and moving held entry all occur in the shared band; phase must not be initialized automatically to 'carry'. Example demo10204 100 is still closing, while demo10200 100 is nearly pinched at similar TCP height.

**Exit conditions**

For red, release has occurred, fingers have opened toward 0.040 m, object stays at its pad as TCP retreats upward and then returns toward blue. Permit transfer earlier during the learned release/retreat overlap with that phase explicitly inferred. For blue, preserve the demonstrated open retreat to about 0.300 m and terminal hold; do not end the policy solely because a place-phase timer expires.

**Failure signatures**

Gripper command flicker; decoded +1 before object descent completes; measured fingers lagging a predicted release; object lifting again with retreat; or phase jumping to the next object while still attached to red. Joint targets and qpos diverging persistently beyond the demonstrated tracking pattern signal a dynamics/interface mismatch, not evidence to advance the phase.

**Overlap role**

Learn acquisition subphases before the nominal carry phase and release/retreat/next-approach after the nominal place phase. This makes phase inference meaningful across handoffs rather than hard-setting a skill-specific clock. Shared windows include substantial moving-arm states, not only stable endpoints.

**Successor readiness**

Provide causal history/phase probabilities to a compatible successor or let it reinitialize from the same observation history. The successor must continue +1 while clearing red, then decide blue closure from alignment; it cannot inherit a fixed 'open forever' gripper mode. Posterior/dwell tolerances and inference-prefix length are hyperparameters to validate, not measured safety bounds.


## Heuristic 3

Optimize rotated object containment and preserve achieved goals, rather than forcing an exact TCP or pad-center trajectory. A contract-aware outcome prior may exploit task tolerance while retaining release/retreat behavior needed for useful handoffs.

**Evidence inspected by the API**

- `demo10200` observations: 200, 250, 265, 280, 300, 340, 390, 550, 600, 650, 700
- `demo10204` observations: 260, 270, 280, 290, 310, 350, 385, 570, 605, 625, 632, 642, 670, 690
- `demo10208` observations: 182, 240, 286, 548, 658
- `demo10210` observations: 180, 238, 284, 542, 652

**Interpretation**

At demo10200 265 the red center is already near (-0.2407,-0.2484,0.0209) m while fingers are still closed; at 280 it is released near z=0.020 m and remains there through retreat. Demo10204 625/632 likewise has blue near its pad before opening, and both objects remain placed after 642/670. These distinguish task attainment from manipulation-state completion and justify retaining post-success actions. The supplied completion contract, rather than the expert's nearly centered placements, permits a region of acceptable final poses. A tolerance-aware objective and preservation of the other goal address a different learning problem than grasp-offset encoding (P1) or event timing (P2); benefits beyond centered demonstrations remain hypothetical.

**Applicability**

Use when the supplied task contract remains full rotated-XY containment in each 0.12 m square pad with cube half-size 0.020 m and center-z tolerance 0.011 m about 0.020 m, requiring both colors in their own pads. The contract needs one satisfying observation, not release, velocity, sustained hold or TCP clearance. Release/retreat are still learned for composition. World poses and goals are causally available; deployment outside the demonstrated pad layout is untested.

**Implications for a future training/inference pipeline**

Candidate P3: contract-aware outcome-guided Diffusion Policy. Condition a baseline joint-action diffusion denoiser on qpos/qvel, both world object poses, TCP, both goals and persistent target/next tags. Train an action-conditioned rollout ensemble F_phi on real future TCP/object/finger states; futures are supervised labels only. Compute geometric goal slack on denormalized predicted states: for all eight cube corners v in {+/-0.020}^3, require |(p+R(q)v-goal)_x|<=0.060 and likewise y; z uses center |p_z-goal_z|<=0.011. Define a smooth nonnegative violation energy E_goal as squared positive-part violations, zero inside the feasible set rather than exact center error. Apply it to plausible placement-horizon states, not every early pickup frame; train a learned progress head on future phase labels to weight that horizon causally at runtime. Add preservation energy for the already-placed other block and ordinary denoising/action-continuity loss so release and retreat after attainment remain modeled. At inference optionally rerank or softly guide denoised action chunks by E_goal plus model-disagreement penalties, with guidance capped outside supported rollout confidence; retain an unguided baseline. Output globally normalized/decoded absolute 7-joint targets and gripper command within the controller bounds at 20 Hz. All policies share the prescribed original-data normalizer, no per-skill scales; drawer zeros are ignored. Completion is independently evaluated on real observations under the exact contract, not by the smooth energy or a learned classifier. Hypothesis: objective shaping will reduce containment violations without requiring exact pad-center replay.

**Assumptions and limitations**

All recorded placements are almost centered and upright, so the broader feasible pad region and rotated near-edge states are mathematical task information, not demonstrated successful behavior. Outcome-model guidance away from the data may hallucinate success; center-goal behavior may be equally good here. Falsify via a matched dynamics-auxiliary model with guidance removed, a center-distance energy baseline, and later evaluation of pad-edge/orientation variation if authorized. Expected gain is tolerance-aware placement and preservation of completed goals, not proven robustness on unseen rotations, collision avoidance or failed releases.

### Handoff interface

**Entry conditions**

Can enter in the observed open pickup band or with the selected block coupled to the TCP during lifting/transport. Goal-region energy must be phase-gated: a resting not-yet-grasped blue is not an invitation to slide it to its goal. The caller supplies target/next tags; keep red's achieved-pad status as context during blue handling. Pinch state is inferred from qpos[7:9], relative motion and command history, not goal satisfaction.

**Exit conditions**

Task achievement is computed from actual observed poses: all rotated cube corners' XY projections within the selected pad and |z-goal_z|<=0.011 m; both red and blue must satisfy for final task completion. For reusable physical transfer also learn opening and retreat: qpos fingers near 0.040 m, released object not following TCP, red-to-blue approach or final TCP near 0.300 m. At demo10204 625/632, both goals can already be satisfied while blue is still pinched; that is contract success, not yet an empty-hand handoff.

**Failure signatures**

Positive center-distance score but a rotated corner outside the pad; center z outside tolerance; red drifts out while blue is placed; object dragged back out during retreat; or the learned outcome predictor reports success contradicted by observed object poses. Early stop at first goal satisfaction with closed fingers would satisfy this task but violate the intended successor interface.

**Overlap role**

Both adjacent policies see red at goal before opening, partial opening, released settling and a long empty-hand transition toward blue. Both pickup policies see acquisition before goal-directed carrying. The energy therefore must preserve these action phases and cannot replace their demonstrations with a goal predicate or center-reaching script.

**Successor readiness**

Successor acquisition needs an actual manipulation-state interface as well as red_at_goal: it must know whether red is still pinched, opening, or left behind and continue the matching release/retreat actions if taking over early. Final termination checks use observed red_pose/blue_pose quaternions and positions, never predicted success alone; physical release/clearance are optional task conditions but retained dataset behaviors. No empirically validated robustness margin beyond the contract is asserted.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
