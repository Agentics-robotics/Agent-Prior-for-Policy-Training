# Acquire and establish transport

Acquire the requested block in a top-down finger grasp, lift it into the demonstrated high transport region, and establish grasp-maintaining motion toward its own goal. When entered before the predecessor has disengaged, first finish that predecessor's goal-aligned lowering/release and retreat so the requested acquisition is physically possible.

## Segmentation

Each trajectory contributes two occurrences of the same acquisition problem. The initial occurrence starts at observation/action 0 with open fingers and a high TCP. Its exit is selected after the requested red block is lifted near 0.28 m and horizontal motion has begun. The recurrent occurrence starts earlier than blue approach: red is still descending over its pad in a closed grip. It therefore includes a deliberate predecessor-completion bridge through lowering, opening, settling, retreat and high approach, followed by blue descent, closing, lift and onset of transport. Final acquire endpoints were inspected separately: demo10000 185/585, demo10001 185/580, demo10002 185/584, demo10003 181/571, demo10004 184/585, demo10005 190/575, demo10006 190/586, demo10007 182/572, demo10008 190/587, demo10009 187/586, demo10010 189/583, demo10011 192/584. Object z at these endpoints is approximately 0.279-0.282 m, with fingers about 0.0183 m and ongoing motion; they are not stationary cut poses. In demo10005/575 blue has only just begun horizontal motion, whereas demo10002/584 has moved about 27 mm: the learner must condition on state, not equate segment end with an exact displacement. Broad pickup overlap with delivery begins while the fingers are open and TCP is still descending, and extends through closure and the full useful lift transition. Every start and stop observation was read. No failed acquisition or off-demonstration contact recovery is represented.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo10000`: [0, 185)
- `demo10000`: [265, 585)
- `demo10001`: [0, 185)
- `demo10001`: [260, 580)
- `demo10002`: [0, 185)
- `demo10002`: [263, 584)
- `demo10003`: [0, 181)
- `demo10003`: [255, 571)
- `demo10004`: [0, 184)
- `demo10004`: [267, 585)
- `demo10005`: [0, 190)
- `demo10005`: [265, 575)
- `demo10006`: [0, 190)
- `demo10006`: [268, 586)
- `demo10007`: [0, 182)
- `demo10007`: [258, 572)
- `demo10008`: [0, 190)
- `demo10008`: [270, 587)
- `demo10009`: [0, 187)
- `demo10009`: [266, 586)
- `demo10010`: [0, 189)
- `demo10010`: [267, 583)
- `demo10011`: [0, 192)
- `demo10011`: [272, 584)

## Heuristic 1

Role-relative object-goal binding: acquisition should depend primarily on relationships between the requested block, its predecessor, TCP and their respective goals, with shared object encoders rather than separate red/blue motion programs. Hypothesis: explicit role binding reduces wrong-object actions and improves sample efficiency across the two occurrences.

**Evidence inspected by the API**

- `demo10000` observations: 0, 80, 100, 110, 150, 170, 185, 265, 300, 350, 425, 470, 504, 506, 550, 570, 585
- `demo10003` observations: 0, 76, 105, 166, 181, 255, 335, 409, 455, 495, 558, 571
- `demo10011` observations: 0, 84, 116, 192, 272, 309, 435, 480, 520, 560, 584

**Interpretation**

The same top-down relationship appears at red and blue despite different arm configurations: demo10000/100 has TCP about 9.5 mm to negative world X of red, while /504 is near blue with a similar small offset and very different joint 1/7 values. Demo10003 and demo10011 repeat this across different source positions. The second occurrence is harder than an isolated reach because its earliest states still hold the previous object. A role graph addresses the representational problem of identifying which object controls the current manipulation without memorizing color-specific joint trajectories. Fixed world goals and limited jitter could also explain success by replay; keeping absolute robot features and testing weight sharing explicitly distinguishes that alternative. Unlike A2 this prior does not posit a persistent contact phase; unlike A3 it does not constrain a waypoint hierarchy.

**Applicability**

Use for top-down acquisition of a nominated 0.04 m cube on the same tabletop with Panda kinematics, gravity direction and named goal association retained. The caller supplies requested-object and predecessor-object roles; both object poses and goals must be causally observable. Sharing across red/blue object slots is supported; arbitrary base relocation, global rotation, new obstacles, new grasp orientations or reversed task order are not established.

**Implications for a future training/inference pipeline**

Candidate A1: object-role relational encoder plus action-chunk diffusion. Causal inputs are qpos, qvel, tcp_pose, red_pose, blue_pose, red_goal, blue_goal and supplied role IDs; use two recent observations for finite-difference motion when available. Build shared-weight object nodes containing pose, goal, requested/predecessor flags and task-containment margins, with TCP-object and object-goal edges. Encode relative translations in world axes, relative rotations with sign-invariant quaternion/rotation-matrix features, and retain absolute TCP/object height, gravity axis, absolute qpos and base-frame reach information. Object-array order may be permuted with roles and goals permuted consistently; do not reflect joint labels or pretend arbitrary SE(3) invariance. A graph attention encoder conditions a temporal diffusion denoiser over normalized eight-channel absolute joint/gripper target chunks. Train with standard noise-prediction loss and role-consistent slot-permutation augmentation; an optional role-attention diagnostic is not a privileged training input. At inference, bind roles once per call, encode the live scene, denoise and execute a short prefix, then reobserve. Clip decoded targets only to the declared action bounds. No demonstration future or event index is an input. Test held-out-trajectory and red/blue occurrence efficiency against identical action diffusion with a flat encoder; expected advantage is object-goal binding and translation-relative generalization, not a clock or contact-memory effect.

**Assumptions and limitations**

Only small source-position variation is demonstrated: red starts around x=-0.429 to -0.412 m and blue around x=-0.425 to -0.408 m, with goals fixed. Approximate object-role symmetry is a hypothesis, not joint-space reflection symmetry. Absolute base reachability and table height cannot be discarded. Compare shared object-goal graph weights against a parameter-matched concatenated-state DP, and separately remove absolute robot/world features. Predicted benefit is lower red/blue data demand and fewer wrong-object moves, not proven goal relocation transfer. This successful-only set cannot establish reacquisition after a miss or safe handling of an incorrectly nominated predecessor.

### Handoff interface

**Entry conditions**

Inputs include a stable requested-object ID and optional predecessor ID, qpos/qvel and world poses/goals. Initial entry is supported at qpos arm=[0,-0.3,0,-2.1,0,1.8,0.7854] rad, both fingers 0.04 m, TCP z=0.442 m and both blocks at z=0.02 m. Recurrent entry is supported with requested blue still on the table and predecessor red already over its own pad but descending at z=0.126-0.253 m, fingers about 0.0183 m each and command -1. The role graph must represent both objects, not immediately drive to blue while ignoring the held red block. Open/closing/partially lifted requested-object states are also present inside the expanded pickup overlap.

**Exit conditions**

Requested block moves with TCP at about z=0.279-0.282 m; measured fingers are about 0.01824-0.01830 m each, with continuing -1 commands and the first horizontal transport motion. Demo10000/185 and 585 and demo10011/192 and 584 support these states, not a universal numerical tolerance. A proposed broader readiness band z=0.25-0.31 m and fingers=0.015-0.025 m must be calibrated, with relative-motion evidence required rather than width alone. The policy continues coherent closed-gripper targets until the successor takes control.

**Failure signatures**

Graph role confusion sends TCP toward blue before releasing red, moves an already placed block, or closes while TCP-object XY error is large. A rising TCP with requested-object z remaining near 0.02 m indicates failed acquisition; changing relative pose while fingers appear closed suggests slip. A stale requested/predecessor role at transfer is an interface failure even when joint actions are smooth.

**Overlap role**

With prior delivery, this candidate learns to finish red descent, hold the arm while opening, leave red at its pad, rise and move toward blue. With following delivery, both learn low approach with fingers open, closure, lifting and initial transport. The role assignment changes between policy calls but each object-goal pairing stays fixed. This is geometry-conditioned shared behavior, not a requirement to switch at a particular timestep.

**Successor readiness**

Deliver receives the same requested-object ID, the object's own goal, both world object poses, TCP pose and measured arm/finger state. It can start earlier in the overlap before contact is established, but must then finish acquisition. At the usual elevated handoff it should preserve the observed TCP-object offset and -1 command, accepting ongoing horizontal velocity instead of resetting to a canonical joint pose. This encoder has no privileged contact memory to pass; short causal pose differences and current geometry must suffice.


## Heuristic 2

Contact belief rather than command state: successful acquisition depends on the history linking gripper commands, measured finger motion and object-TCP co-motion. Hypothesis: a persistent learned contact belief makes closure, lifting and predecessor release more robust to handoff timing and servo lag.

**Evidence inspected by the API**

- `demo10000` observations: 100, 106, 107, 110, 125, 150, 170, 185, 290, 296, 300, 305, 315, 350, 503, 504, 505, 506, 510, 525, 550, 570
- `demo10003` observations: 105, 166, 181, 285, 335, 495, 558, 571
- `demo10011` observations: 116, 192, 272, 309, 359, 520, 560, 584

**Interpretation**

A command is not a contact state. Demo10000/107 and /504 change to -1 while fingers are still open; /505 and /506 reveal rapid closure, and only later /525-/570 verify the blue block lifting. Release at /296 similarly begins with fingers near 0.0183 m, whereas /305 is nearly open and /350 has a stationary red block far below a retreating TCP. Demo10011/116 even has a small negative red roll where other grasps show positive roll, yet succeeds: a single sign or finger-velocity threshold is not a reliable contact label. This motivates causal latent state and action-observation hysteresis instead of a static width rule. The learning gap is ambiguity during broad handoffs and delayed physical response, distinct from A1's object-slot representation and A3's geometric path factorization.

**Applicability**

Use when fingers, TCP and block poses are observed at 20 Hz and closed-gripper contact produces co-motion during lift. Retain the Panda mimic-gripper mapping, controller lag and cube dimensions. A causal history buffer must be available or reconstructed from the opening part of an overlap; a single isolated snapshot with no prior action history is a weaker supported interface.

**Implications for a future training/inference pipeline**

Candidate A2: contact-belief-conditioned hybrid Diffusion Policy. Feed a causal recurrent encoder the supplied qpos/qvel, world TCP and both object poses/goals, role IDs, finite-difference object/TCP motion and previously EXECUTED gripper commands. Do not feed action[t] when predicting it. Maintain a learned latent belief over predecessor-loaded/releasing, empty-approach, requested-closing and requested-loaded/lifting; phases are explanatory latent states, not a scripted controller. Approximate training labels come from demonstrated command changes, measured finger apertures and subsequent object lift/co-motion. Future-confirmed attachment or release labels are auxiliary supervision only. Train a causal GRU/variational filter with auxiliary next-finger-aperture and relative-transform-change regression plus weak mode classification and diffusion noise loss. Condition the arm-target diffusion on the filtered belief; use a persistent binary gripper head producing -1 or +1, supervised from the original gripper commands, with a learned transition/hysteresis model to discourage framewise toggling. Arm outputs remain seven absolute Panda targets in radians; gripper conversion uses the declared dimensionless command. During deployment update belief at every observation, denoise an arm chunk and sample/predict its aligned gripper sequence, execute a short prefix, then update from actual motion. Use the shared full-demonstration normalizers and no simulated negative rollouts. Train variable causal-history truncation and overlap-start sampling so initialization is explicit. Prediction: better asynchronous handoff and less false-loaded behavior than memoryless A1, especially across command-to-aperture lag.

**Assumptions and limitations**

Contact is inferred, not measured. Compliant compression, simulator finger dynamics and small block shifts could explain apparent hysteresis without a discrete physical state. All inspected grasps succeed, so the latent's missed-grasp/slip classes cannot be validated as recovery modes. Falsify the claim with matched causal-history DP versus recurrent latent DP, removing auxiliary contact/co-motion targets separately; measure gripper oscillations, object lift reliability and performance when switching at different demonstrated overlap states. A model that improves only timestep prediction, not contact-dependent actions, would not support the proposed benefit.

### Handoff interface

**Entry conditions**

Accept open fingers near 0.04 m, closing fingers near 0.0205 m, stable loaded fingers near 0.0183 m, or an opening predecessor grip, provided the caller's requested/predecessor roles and recent qpos/qvel, TCP/object world poses and previously executed commands are known. Demo10000/504 has command -1 but observed fingers still 0.04 m; /505 has about 0.0205 m and finger velocities about -0.34/-0.37 m/s; /506 has about 0.0183 m. The recurrent segment also begins with red still held above its pad. Initialize a distribution over contact modes, not an assumption that entry means empty and open.

**Exit conditions**

Posterior mass favors loaded-lift/transport after several causal samples of TCP-object co-motion at object z near the demonstrated 0.28 m and continued -1 commands. Measured stable finger widths alone do not establish grasp; require the block to leave its z=0.02 m support region with the wrist. Confidence and relative-transform residual thresholds are tunable hypotheses; no contact-force ground truth is supplied. Maintain the grasp while handing over, without an automatic open pulse or hidden-state reset motion.

**Failure signatures**

Predicted loaded state with object still on the table, repeated open/close oscillation, a contact posterior that cannot settle, or posterior disagreement with measured relative motion indicates premature transfer. Finger qvel oscillations alone are not failure evidence: successful closed grips have nonzero opposing finger velocities. Wrong predecessor identity or insufficient history can make release look like failed grasp.

**Overlap role**

The full shared pickup transition teaches open-to-closing-to-loaded-to-lifting mode changes to both acquisition and delivery. The long red-to-blue overlap supplies the reverse transition: still-loaded descent, command opening, fingers physically separating, stationary red while TCP retreats, and empty approach. The candidate learns not to confuse command changes with immediate contact changes; it does not rely on one boundary contact label.

**Successor readiness**

Give the successor raw causal history and role IDs; optionally expose contact-mode probabilities and an estimated grasp transform as diagnostic context, never privileged future labels. A successor that does not consume the latent can recompute readiness from qpos fingers, world TCP/object co-motion and recent commands. Delivery may take over with uncertain closure in the shared interval if it continues verification and lift; it must not assume a stable grasp solely from the -1 command.


## Heuristic 3

Height-separated waypoint hierarchy: contact approach/lift and free-space retargeting occupy different geometric regimes, even when their durations differ. Hypothesis: diffusing a small role-relative waypoint plan before actuator targets reduces low-height lateral shortcuts and makes the long release-to-next-pick bridge easier to learn.

**Evidence inspected by the API**

- `demo10000` observations: 0, 50, 80, 100, 125, 150, 170, 185, 265, 280, 300, 315, 350, 400, 425, 470, 490, 525, 550, 570, 585
- `demo10001` observations: 0, 78, 108, 168, 185, 260, 300, 345, 417, 463, 530, 565, 580
- `demo10011` observations: 0, 84, 116, 192, 272, 309, 359, 435, 480, 520, 560, 584

**Interpretation**

The TCP descends nearly vertically near red in demo10000/50-/100, then lifts red through /125-/170 before horizontal movement at /185. After placement, /315-/350 rise while red remains at its pad; /400-/425 move high toward blue, which is then approached and lifted. Demo10001 and demo10011 exhibit the same organization with distinct timing and joint solutions. Factoring this regularity into geometric plan plus actuator tracking could prevent a learner from averaging high travel and low contact trajectories when learning from few examples. This is a hypothesis about low-dimensional path structure, not proof of a universal collision-free rule. It differs from role sharing and contact-state memory by changing what is generated before the action chunk and which geometric deviations are penalized.

**Applicability**

Use for unobstructed tabletop transfers with downward-facing TCP, near-upright cubes and enough vertical clearance for the demonstrated roughly 0.28-0.30 m high path. World Z and robot-specific actuator tracking must be retained. Caller roles must distinguish the still-held predecessor from the requested block, including the initial occurrence with no predecessor.

**Implications for a future training/inference pipeline**

Candidate A3: hierarchical waypoint Diffusion Policy. High-level diffusion predicts a short ordered sequence of role-relative TCP/object pose knots and positive durations, conditioned on causal qpos/qvel, tcp_pose, object poses/goals and roles. Keep Z in world metres and express XY relative to the relevant predecessor pad or requested block; preserve orientation and absolute robot configuration as context. Offline targets are future demonstration poses at release-clearance, source-hover, source-low and lifted-object events when those events lie in the segment; missing/already completed knots are masked. Future event times train durations and progress, never enter inference. A learned progress selector chooses remaining knots from current observations. A bounded smooth interpolator expands predicted knots into a reference pose trajectory. A second conditional action diffusion maps this reference plus live robot/object state to the original eight absolute targets; train from original actions, not IK commands. Use high-level pose/duration likelihood or diffusion loss, low-level action noise loss, and soft phase-conditioned penalties: low-approach lateral error to the requested block, vertical separation before empty travel, and requested-object lift before sustained loaded horizontal travel. Do not penalize the intentionally held predecessor's final descent. Train the tracker with both future-labelled and predicted/noisy-reference conditioning to expose planning error without inventing robot outcome labels. At inference generate remaining waypoints, interpolate, denoise a short actuator chunk, execute/reobserve and replan; no robot URDF is assumed. Prediction: better path organization and long-bridge sample efficiency, at potential cost of hierarchy error, compared with A1/A2 which diffuse actions without an explicit geometric plan.

**Assumptions and limitations**

The vertical-then-horizontal organization may reflect this expert's planner rather than a universally optimal path. A hard clearance constraint would fail under low ceilings or tasks requiring sliding; therefore constraints are soft and mode-conditioned. Futures supervise geometric plans only within retained segments; no imagined obstacle examples are added. Compare a flat joint-action DP, a hierarchy without geometric clearance penalties, and the full candidate; test path height, action smoothness and handoff completion, not just reconstruction. Successful demonstrations cannot establish obstacle avoidance, robustness to large waypoint errors or corrective regrasp behavior.

### Handoff interface

**Entry conditions**

Accept the initial high empty TCP state, or a goal-aligned predecessor descending while held, or any demonstrated intermediate empty approach/closing/lifting state. The causal inputs must include world height and a current role-conditioned path estimate; do not initialize every call at its first waypoint. Supported predecessor entry z ranges about 0.126-0.253 m, finger positions about 0.0183 m; later bridge states have fingers 0.04 m and TCP rising toward 0.30 m. Obstructed paths and nonvertical entanglement are outside the evidence.

**Exit conditions**

The requested block has reached the lift waypoint near world z=0.28 m and begun a small horizontal displacement toward its goal; fingers remain loaded near 0.0183 m, command -1. Demo10000/185 has about 9 mm initial red translation, demo10001/185 about 32 mm and demo10011/584 about 31 mm blue translation. These are sampled endpoint support, not required displacement thresholds. Abandon or replan a stale waypoint at transfer; continue the current smooth motion until delivery produces its first action.

**Failure signatures**

Low-height lateral motion before release or before a confirmed lift, spline overshoot toward the table, excessive joint-target jumps while geometric waypoints look smooth, or waypoint progress advancing without object lift indicate failure. A posterior that selects a blue approach knot while still holding red is a bridge-role error. A newly predicted path should not pull the robot back to the old start knot.

**Overlap role**

Shared coverage includes release and vertical separation from the previous pad, high retargeting, requested-object descent, closure and the entire lift into early transport. This lets the geometric planner condition on partially completed transitions and lets the low-level diffusion learn physical tracking rather than an exact start pose. The preceding deliver and following deliver policies can complete these same phases; no rigid knot-to-knot switch is imposed.

**Successor readiness**

Delivery needs live qpos/qvel, TCP/object poses, role/goal association and a grasp-maintaining command, not the acquire planner's original future waypoint list. Its first plan should be anchored to current observed pose and velocity. A predicted reached-lift waypoint alone is insufficient: observed object height/co-motion must support it. Carrying at about 0.28 m with ongoing horizontal velocity is the usual handoff; earlier pickup-overlap entry remains supported if delivery finishes the lift.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
