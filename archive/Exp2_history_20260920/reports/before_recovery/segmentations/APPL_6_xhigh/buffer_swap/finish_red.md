# Retrieve buffered red, complete the swap and retain terminal release

Finish blue setdown/release if needed, leave blue at its achieved goal, retrieve red from the buffer and place red fully inside red_goal. Retain the demonstrated finger opening, object settling and TCP retreat through the supplied terminal state without imposing them as additional task-success predicates.

## Segmentation

Each segment starts while blue remains held and descending over its goal, before release and retreat, then covers buffered red approach, finger closure, lift, transport, placement, opening, settling and final retreat. Starts c are 600,605,602,606,598,599,601,613,600,604,607,603 for demos10100-10111; all have closed-grip blue at world z approximately 0.137-0.209 m and red supported at the buffer. Stops are each original final observation, all inspected, with both blocks fully contained near their target centers, z approximately 0.020 m, fingers about 0.040 m and TCP z about 0.300 m. The earliest contract satisfaction can precede release, as shown by demo10111:966, but the remaining release/retreat/settling actions are not excluded. This final transfer is kept distinct from blue transfer because its predecessor is a completed goal and its successor is task termination, requiring different preservation and exit learning questions.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo10100`: [600, 1064)
- `demo10101`: [605, 1064)
- `demo10102`: [602, 1060)
- `demo10103`: [606, 1065)
- `demo10104`: [598, 1057)
- `demo10105`: [599, 1056)
- `demo10106`: [601, 1059)
- `demo10107`: [613, 1071)
- `demo10108`: [600, 1058)
- `demo10109`: [604, 1063)
- `demo10110`: [607, 1064)
- `demo10111`: [603, 1061)

## Heuristic 1

Optimize the actual rotated-footprint completion margins, not just object-center proximity or gripper release. Hypothesis: contract-aware future-outcome guidance improves placement reliability and avoids false completion while retaining demonstrated terminal release and retreat.

**Evidence inspected by the API**

- `demo10100` observations: 600, 630, 640, 650, 850, 900, 960, 980, 990, 1020, 1064
- `demo10111` observations: 603, 643, 844, 941, 966, 978, 997, 1061
- `demo10110` observations: 607, 849, 1064

**Interpretation**

At demo10100:960 red is aligned in XY but z=0.0512 m, which is not successful; by 980 it is supported at z=0.020 and fingers are opening. demo10111:966 already has red z=0.0243 within tolerance while gripper is still closed, showing that release and TCP retreat are not necessary for task success. Endpoints confirm both goals while subsequent release/retreat gives a useful terminal interface. The margin prior directly addresses mismatch between coordinate regression and the declared containment contract; it is an objective/inference design, distinct from recurrent mode memory and grasp-frame representation.

**Applicability**

Known cube half-size 0.02 m and axis-aligned world goal squares with half-width 0.06 m, supplied object wxyz orientation, goal center z=0.02 m and tolerance 0.011 m. Applies to modest position/orientation errors under the same contact dynamics; no alternative goal semantics should be silently substituted.

**Implications for a future training/inference pipeline**

Candidate M: goal-margin-aware diffusion with future-pose auxiliary prediction and soft sampling guidance. Causal inputs are qpos/qvel, tcp_pose, both object poses and both goals, plus finger state/history. Convert each wxyz to R and compute projected half-extents e_x=0.02*sum_j|R_xj|, e_y analog; margins m_x=0.06-|p_x-g_x|-e_x, m_y analog, m_z=0.011-|p_z-g_z|. These exact current-state features are available at deployment. A diffusion denoiser predicts normalized absolute [q_target,g] chunks; a trainable outcome head predicts future object poses/support mode from observation plus denoised chunk, supervised on actual future states. Train standard denoising and pose/support losses plus a soft hinge on negative predicted margins only during labelled place/support windows, not during necessary airborne transit. Add a both-goals preservation term when blue is supported. At inference obtain a placement-mode probability from causal state, use it to softly rank or guide a small fixed set of denoised chunks by predicted margins; never feed future success labels. Decode with full-demo normalization and original joint/gripper limits. Exact observed contract evaluation remains separate from policy guidance. Prediction: this objective aligns errors with real task success better than Euclidean center-distance alone.

**Assumptions and limitations**

All final placements are near region centers with near-identity cube orientation; edge containment and large rotations are not tested. A learned future-pose model could be overconfident, and aggressive guidance could degrade grip. Compare center-distance guidance, unguided DP and full-footprint guidance with equal sampling budget; measure actual containment, not only center error. Successful data cannot establish corner-recovery or near-boundary contact strategies.

### Handoff interface

**Entry conditions**

Earliest state has blue held descending over blue_goal at z about 0.137-0.209 m with g=-1 and red supported at the buffer. The policy must finish that placement/open/retreat before retrieving red. A positive XY margin of suspended blue is not a completed-goal predicate unless z is also within tolerance.

**Exit conditions**

Both full rotated XY footprints lie in their own regions and each center z is within 0.011 m of 0.02 for one observation. Inspected final blue positions are around [-0.351 to -0.3524,-0.201 to -0.199,0.020] m and red around [-0.351,0.195 to 0.196,0.020] m. Demonstrated terminal interface additionally opens fingers to 0.04 m, retreats TCP to z about 0.300 m and nearly stops qvel; these are useful observed actions, not added success requirements.

**Failure signatures**

Center-only prediction reports success while a rotated corner is outside a square; red is aligned in XY but too high; red follows the hand after supposed release; blue is displaced during retrieval. A high learned success probability does not supersede exact geometry computed from observations.

**Overlap role**

The shared interval learns blue descent with closed fingers, support and opening, vertical retreat, red approach/closure and red lift. Thus containment guidance applies first to blue and later to red without assuming a completed blue placement at entry.

**Successor readiness**

There is no next manipulation skill in the supplied task. Expose exact observed red_at_goal AND blue_at_goal to a task-level evaluator using the declared one-observation contract; if execution continues, learn demonstrated release/retreat rather than regrasping. Any deployment stop/hold rule beyond the contract is unspecified.


## Heuristic 2

Remember completed manipulation while inferring the current contact mode: use a soft, observation-correctable progress memory to disambiguate approach, release and terminal retreat. Hypothesis: recurrent diffusion initialized from expanded overlaps avoids repeating completed actions at handoff.

**Evidence inspected by the API**

- `demo10100` observations: 600, 630, 640, 650, 690, 755, 800, 810, 820, 850, 960, 980, 1020, 1064
- `demo10111` observations: 603, 643, 694, 755, 812, 844, 941, 966, 978, 997, 1061
- `demo10107` observations: 613, 654, 821, 855, 1071

**Interpretation**

A low downward-facing TCP over a cube can mean closing for pickup or opening after placement. In demo10100:630 and 640 the scene is nearly unchanged except finger opening/support; later red closure at 810-820 should not reset the task to blue acquisition. Final retreat similarly follows a state already satisfying the goals. demo10111 and demo10107 show the same ordering at different indices. A persistent belief about what has already been completed is therefore a plausible way to resolve aliased local poses. This hypothesis concerns long-range memory and transition initialization, rather than placement margin optimization or geometric grasp compensation.

**Applicability**

The demonstrated red-final order, with causal observations sufficient to infer whether blue is still attached or already left at goal. Same robot and table. Recovery that revisits blue or intentionally rearranges a completed object is not supported by this monotone progress hypothesis.

**Implications for a future training/inference pipeline**

Candidate L: history-dependent progress-memory DP. Encode causal qpos/qvel, finger aperture, TCP and both block world poses/goals plus backward velocities in a GRU/causal transformer. Train latent states for finishing-blue-placement, empty-retreat, red-approach/closing, red-attached-carry, red-place/open, final-retreat/terminal. Training mode labels use gripper transitions and future object-TCP separation/lift evidence; train an auxiliary observed-goal head against exact margins. A soft ordered-transition/persistence regularizer discourages returning to blue manipulation after the model has observed supported blue and TCP departure; it is a learned state representation, not a prescribed motor state machine. Sample random causal entry windows throughout 600:850-like overlaps and train a state-based hidden-state initializer so deployment does not require earlier skill-specific hidden states. The diffusion head conditions on mode belief and state to output absolute seven-joint targets and g, trained by denoising plus mode/progress losses with extra weight near mode changes and terminal tail. Inference updates memory from observations and replans short chunks under the common full-demo normalizer/action bounds. Label futures and trajectory clocks are unavailable at inference. Prediction: fewer repeated grasps and handoff-induced mode resets than memoryless policies.

**Assumptions and limitations**

The observed sequence is acyclic, but that does not establish a universally correct irreversible mode graph. A hard latch could prevent recovery if blue is bumped; prefer a soft persistence prior with an observation-disagreement signal. Compare current-state DP, recurrent DP without ordering loss and the proposed candidate, especially with randomized handoff times. No failed-placement retries or external disturbances are demonstrated.

### Handoff interface

**Entry conditions**

Takeover may precede blue support, during opening, during empty retreat, during red approach, during red closure or after red lift. Initialize a learned recurrent mode from recent qpos/qvel/object/TCP history; do not preset 'blue done' just because this is the final skill. Finger aperture ranges from about 0.018 m held to 0.040 m open inside this entry envelope.

**Exit conditions**

Posterior assigns terminal-arranged/released-or-retreating state while measured geometry satisfies both goals. Finish the observed g=+1 and upward retreat if commands continue; last observations have TCP z about 0.300 m, open fingers and near-zero qvel. The internal terminal state must not turn a near-goal airborne block into success.

**Failure signatures**

Policy re-closes around blue after leaving it, mistakes open low-TCP release for an approach, cycles between retrieve-red and finish-blue, or keeps moving after its internal terminal label despite an unexpected object change. Incorrect irreversible memory can hide disturbance and must be observable as state/label disagreement.

**Overlap role**

Both neighbors learn blue placement before the blue-done belief becomes persistent, then empty retreat and red acquisition. Shared trajectories support initialization at many stages, not only initialization from the final mode of transfer_blue. A terminal flag is not exchanged at one exact cut.

**Successor readiness**

A task evaluator needs observed object geometry and optionally calibrated progress uncertainty, not the recurrent hidden vector. With no successor manipulation, preserve the demonstrated open, clear, nearly stationary end behavior. Additional stopping persistence or reset behavior is a deployment design not supplied here.


## Heuristic 3

Compensate object goals in the current grasp frame: represent the target TCP pose through the measured object-to-tool transform instead of treating the TCP as the object center. Hypothesis: an object-relative diffusion representation improves precise final placement under small grasp-offset changes.

**Evidence inspected by the API**

- `demo10100` observations: 600, 630, 640, 755, 800, 810, 820, 840, 850, 900, 960, 980, 1064
- `demo10111` observations: 603, 643, 755, 812, 844, 941, 966, 978, 1061
- `demo10103` observations: 606, 848, 1065

**Interpretation**

The red-minus-TCP world displacement after the second grasp is about +0.00845 m in x in both demo10100:850/960 and demo10111:844/966, with much smaller y/z differences. That relation is also visible for blue before release, but the transform must be re-estimated across empty travel. A block-centered goal and a TCP-centered motor target are not identical. Encoding the desired TCP pose through the observed grasp transform can reduce the mapping complexity from object goals to joint commands. This is a geometric representation/decoding prior, distinct from candidate M's success objective and candidate L's temporal memory.

**Applicability**

Rigid similarly sized blocks, top-down grasps and the same Panda joint controller. Relative geometry may help with modest observed initial/grasp offsets. Keep world gravity, goal-square axes, joint limits and robot absolute configuration; arbitrary global SE(3) equivariance is invalid for this fixed-base robot.

**Implications for a future training/inference pipeline**

Candidate F: grasp-frame-conditioned diffusion with a kinematics-aware learned decoder, without recurrent progress-memory or contract-margin guidance. Inputs are causal source qpos/qvel, tcp_pose, both object poses/goals and finger history. A short-history attachment/role encoder selects probabilities for blue-attached, empty and red-attached. For an attached object estimate D=T_tcp^-1 T_object, choose desired object orientation by retaining its current approximately upright yaw (goals impose footprint containment, not a specified quaternion), and derive T_tcp_star=T_object_goal D^-1. Encode errors log(T_tcp^-1 T_tcp_star), object-to-goal displacement and world z/goal axes alongside absolute qpos; in empty mode use TCP-to-red approach geometry instead. All transforms are constructed from causal world metre/wxyz fields. A diffusion head predicts task-space residual/action latents; a trainable decoder conditioned on current qpos/qvel outputs absolute seven-joint targets plus g. Train with the usual action-denoising/reconstruction objective and auxiliary subsequent TCP/object-pose prediction using demonstration futures only, so the decoder learns PD lag rather than pretending joint targets equal realized qpos. Do not require external IK, and compare with direct absolute-joint diffusion under identical data. At inference recompute the geometric features and attachment confidence, denoise/decode, apply full-demo normalization inverse and contract bounds, execute short chunks. Prediction: errors induced by changed grasp offsets are smaller than with raw world-state encoding.

**Assumptions and limitations**

Nearly constant grasp offsets and near-identity object orientations may make explicit compensation unnecessary; benefits beyond this narrow distribution are unproven. Current-pose-only offset estimates can confuse pre-contact proximity with attachment, and a learned joint decoder can fail for new arm configurations. Ablate relative transforms/compensated target against equal-size world-coordinate DP, and compare with a constant-offset baseline. Large yaw, slipping and unseen buffer positions are not established.

### Handoff interface

**Entry conditions**

At early entry blue is still held over its goal; red is supported near [-0.181,0,0.020] m. The encoder must estimate which object is attached before using a grasp-relative target. During empty approach use TCP-to-red geometry with fingers open at 0.04 m; after closure near 0.01825 m, update the red offset from actual co-motion.

**Exit conditions**

Red and blue satisfy the declared goals; release and retreat are retained. At demo10111:966 red center is [-0.35095,0.19525,0.02431] and TCP [-0.35941,0.19542,0.02433] m, illustrating why TCP should not be placed at the block-goal center. After opening, stop using attachment compensation and learn empty retreat to roughly z=0.30 m.

**Failure signatures**

TCP is driven to the object-goal center despite nonzero grasp offset, compensation uses the old blue grasp for red, rotated axes are decoded incorrectly, or object motion disagrees with assumed rigid offset. A transform estimated before fingers settle may be wrong even though coordinates are precise.

**Overlap role**

The shared blue placement-to-red lift interval contains both an old attachment, its release, empty travel and a new attachment. Both policies learn to discard the old transform on opening and re-estimate at the new grasp, not carry a single calibration offset across the skill switch.

**Successor readiness**

For terminal evaluation pass measured poses and goals; any grasp-frame latent is internal. If execution continues after success, maintain demonstrated opening/retreat using current qpos/qvel, without demanding exact zero object error or correcting a valid placement. Any allowed new grasp/yaw range needs future validation.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
