# red_to_buffer_with_blue_handoff

Move the red block from the initial lower/blue-goal region to the central temporary buffer and continue the demonstrated transition until blue acquisition is underway.

## Segmentation

Each segment starts at the initial home/open state and includes the full first red manipulation: approach the red block in the lower region, close/lift it, move it to the central free buffer, lower/release it, retreat, and continue through the expanded handoff into blue acquisition. The stop at action index 460 is chosen because inspected observations show red settled at the buffer and the TCP low over the blue block with fingers closing; thus the transition to the next skill is covered substantially beyond red release. All trajectories share this sequence with modest initial object pose variations, so they are grouped as one learning problem.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo10100`: [0, 460)
- `demo10101`: [0, 460)
- `demo10102`: [0, 460)
- `demo10103`: [0, 460)
- `demo10104`: [0, 460)
- `demo10105`: [0, 460)
- `demo10106`: [0, 460)
- `demo10107`: [0, 460)
- `demo10108`: [0, 460)
- `demo10109`: [0, 460)
- `demo10110`: [0, 460)
- `demo10111`: [0, 460)

## Heuristic 1

Object-relative contact-funnel prior: represent the first transfer in the red block frame with an explicit latent contact phase, so the policy learns invariant approach, pinch, lift, carry, lower, and release relationships instead of memorizing absolute joint-time traces.

**Evidence inspected by the API**

- `demo10100` observations: 0, 100, 150, 200, 250, 300, 320, 460
- `demo10105` observations: 0, 320, 460
- `demo10107` observations: 0, 320, 460

**Interpretation**

Across demos the same geometric relation recurs despite initial red variation: at demo10100 index 100 TCP is low near red with closing fingers; by 150-200 red z rises with TCP to 0.26-0.29 m; by 250-300 red is lowered and left at z 0.02 near x -0.181, y near 0. The same start/stop pattern is visible in demo10105 and demo10107 endpoints. The difficult learning problem is associating gripper closure and lift with object contact rather than absolute time. This prior assumes the manipulation can be learned as a sequence of object-relative contact phases; it differs from the buffer-topology prior by emphasizing grasp/carry dynamics rather than why the buffer is chosen.

**Applicability**

Transfers when the first manipulation is a single top-down pinch of the red block from a marked lower region, with a free central buffer available and similar Panda/table/block geometry. It depends on the top-down gripper orientation, block half-size about 0.02 m, gripper aperture mapping in qpos[7:9]/gripper_command, and a collision-free lift height around 0.26-0.30 m.

**Implications for a future training/inference pipeline**

Candidate policy input at inference: causal observation fields qpos, qvel, tcp_pose, red_pose, blue_pose, red_goal, blue_goal. Encode an active-object frame o=red_pose: features [tcp_xyz-red_xyz, blue_xyz-red_xyz, red_goal_xyz-red_xyz, blue_goal_xyz-red_xyz, qpos[7:9], qvel]. Train a latent phase model z_t in {approach, close_lift, carry_to_buffer, lower_release, blue_handoff}; z labels may be computed from demonstration futures/events for training only (red z lift, gripper_command sign, red at buffer, blue contact). The DP denoises action chunks a_{t:t+H}=[joint targets, gripper_command] conditioned on z_t and object-relative features, with auxiliary losses predicting next phase and red_pose_{t+k}-red_pose_t. Deployment computes z_t from the learned causal recurrent encoder, then decodes absolute joint targets. The inductive change is object-relative/contact-phase factorization rather than a single time-indexed joint replay.

**Assumptions and limitations**

The dataset only shows successful top-down pinch trajectories and one central buffer choice; it cannot establish robustness to failed closure, moving blocks, heavy contact, or clutter in the buffer. A falsifying ablation is to train the same policy without object-relative phase features and compare pickup/drop rates under initial red pose perturbations; if no difference appears, the contact-funnel prior is unnecessary. Failure should appear as TCP reaching the nominal script while red_pose does not follow or falls before release.

### Handoff interface

**Entry conditions**

Measured support starts at observation index 0 with qpos arm at home [0,-0.3,0,-2.1,0,1.8,0.785] rad, qpos finger positions about 0.04 m, gripper_command 1.0, TCP at about [-0.384,0,0.442] m in world, red_pose near the lower initial area (examples x -0.360 to -0.342, y -0.211 to -0.189, z 0.02 m), and blue_pose on the upper marked region. Hypothesized tolerance is limited to the initial pose spread inspected across demos, not arbitrary block placements.

**Exit conditions**

Useful end state is red_pose on the central buffer near [-0.181,0,0.02] m, fingers reopened, and the robot already in the blue handoff corridor: demonstrated stop obs index 460 has tcp_pose over the blue object at z about 0.020-0.028 m and qpos[7:9] about 0.018 m as gripper closure on blue begins. This policy may exit earlier in the overlap if a successor trained from index 320 can continue from high open approach.

**Failure signatures**

Red z fails to rise with TCP during the early grasp/lift, red_pose does not settle near [-0.181,0,0.02] by the release portion, qpos[7:9] remains wide while carrying, blue_pose changes before intended handoff, or tcp_pose approaches blue with large lateral error relative to blue_pose. These indicate missed grasp, dropped red, or premature transfer.

**Overlap role**

The overlap with skill 2 is the full red-release/retreat/blue-approach/blue-grasp transition [320,460). Both policies see fingers open at high retreat near index 320 and fingers closing on blue by index 460, so selection can switch while arm motion and contact state are still evolving.

**Successor readiness**

Skill 2 needs blue_pose still on the upper region, red already buffered, gripper open or closing around blue, and tcp_pose either high in the approach corridor (about z 0.28-0.30 m at index 320) or low over blue with qpos[7:9] near 0.018 m by index 460. Measured support spans both; recovery from lateral misses outside this corridor is unseen.


## Heuristic 2

Swap-topology buffer prior: explicitly represent the first red placement as freeing the blue goal by moving red to a central temporary buffer, which should help the learner generalize the subgoal choice under initial pose variation.

**Evidence inspected by the API**

- `demo10100` observations: 0, 250, 300, 320
- `demo10102` observations: 0, 320
- `demo10108` observations: 0, 320

**Interpretation**

At all inspected starts, red begins near the blue goal/lower region while blue occupies the red goal/upper region. In demo10100 red is at [-0.355,-0.209,0.02] at index 0, but at indices 300-320 it has settled near [-0.181,-0.001,0.02] while blue remains unchanged. Demo10102 and demo10108 show the same central-buffer state at index 320 despite different initial red poses. The learning gap is that pure imitation may not infer why red is placed at the center; encoding the buffer as a topological subgoal should help transfer within the same swap structure. This differs from the contact-funnel prior by biasing the high-level placement choice, not the grasp mechanics.

**Applicability**

Transfers to buffer-swap variants where the initially occupied red-goal/blue-start region must be freed before placing red at its final goal, and where a central area around x -0.18, y 0 is free. It requires red_goal and blue_goal fields to remain meaningful world targets and the tabletop to permit temporary placement without fixtures.

**Implications for a future training/inference pipeline**

Inputs are causal red_pose, blue_pose, red_goal, blue_goal, tcp_pose, qpos/qvel. Add a learned or fixed symbolic feature b_t=[red_in_initial_blue_goal, blue_in_red_goal, red_buffered] computed from current poses and goals. Train a subgoal head g_hat_t to predict the demonstrated temporary placement center from future red_pose at release; labels are from future observations only during training. The action decoder is a DP conditioned on [continuous state, b_t, g_hat_t]. Objective: diffusion score loss on absolute joint/gripper actions plus L2(g_hat, red_release_xyz) and cross-entropy for red_buffered. Deployment computes b_t from current poses and rolls out actions toward g_hat; no future labels are used.

**Assumptions and limitations**

This prior assumes the central buffer is the intended temporary storage location, but the data cannot prove optimality versus other free cells. It also cannot prove robustness if the buffer is occupied. Compare against a goal-only policy that omits an explicit buffer variable; if both generalize equally to start pose variation, the topology prior is not adding value. Failure signature is a policy trying to place red directly at red_goal while blue still occupies it or choosing an unobserved buffer.

### Handoff interface

**Entry conditions**

Same measured start support as the segment start: home arm, open fingers about 0.04 m, red in the lower initial region that is also blue_goal, blue occupying the upper region that is red_goal. The prior assumes the current blue_pose makes direct red-to-red_goal placement blocked or undesirable.

**Exit conditions**

The expected end state is topological rather than just kinematic: red is no longer occupying the lower/blue-goal region and is instead at the free central buffer, while blue remains available for skill 2. Demonstrated index 320 states show red_pose near [-0.181, -0.001, 0.02] and blue_pose unchanged; the robot is open and high, able to approach blue.

**Failure signatures**

Red remains in or near the blue_goal region, red is placed in the future blue path/goal, or central buffer placement is outside the demonstrated band (roughly x -0.181, y -0.001, z 0.02). Such errors would block the swap even if a grasp was successful.

**Overlap role**

The [320,460) overlap teaches that once the topological predicate red_buffered is achieved, control may continue toward blue acquisition. Both adjacent policies see red_buffered while TCP transitions from high buffer retreat to low blue contact.

**Successor readiness**

Skill 2 can take over when red_buffered is true, blue_pose is still near the upper/red-goal region, qpos fingers are open (index 320) or are closing on blue (index 460), and no object is being carried except during the very end of the handoff. The demonstrated tolerance is central-buffer placement only; alternative buffer cells are not observed.


## Heuristic 3

Learned overlap-readiness prior: model the red-to-blue transition as a continuum of successor-ready states and train an explicit readiness signal so switching can occur over the whole demonstrated overlap instead of at one fragile timestep.

**Evidence inspected by the API**

- `demo10100` observations: 300, 320, 360, 400, 460
- `demo10103` observations: 320, 400, 460
- `demo10110` observations: 320, 400, 460

**Interpretation**

In demo10100, index 300 shows red released and TCP at z 0.134; index 320 is high over the buffer with open fingers; index 400 is high over/near blue; index 460 has the TCP at table height over blue with fingers closing. Demo10103 and demo10110 endpoints show the same continuum. The hard part for a learned modular system is timing jitter at the boundary. This prior assumes transition readiness is observable from object poses, TCP, and finger state. It differs from the other two by focusing on selection and overlap robustness rather than red transfer or buffer choice.

**Applicability**

Useful when the skill selector may switch anywhere during the demonstrated post-red-release transition and when the next skill needs either an open high approach to blue or the beginning of blue contact. It assumes the same temporal ordering: red buffer first, then blue acquisition.

**Implications for a future training/inference pipeline**

Use a recurrent DP with two heads: action denoising and handoff readiness r_t in [0,1]. Inputs are causal state fields plus derived distances d_red_buffer and d_tcp_blue. Training labels for r_t are generated from segment membership/futures: r=0 before red release, r increases over overlap [320,460), with positive examples when red z is table height and blue is in approach/contact range. Loss = diffusion action loss + BCE(r_t,label) + smoothness penalty on r_t. Deployment computes actions and r_t; the selector may switch when r_t high and successor confidence is high, otherwise this policy continues into blue approach. The prior changes inference by making handoff a learned state estimate, not an externally scripted cut.

**Assumptions and limitations**

Readiness labels are inferred from demonstrations, not from interventions, so the policy may over-trust states that look temporally similar but have wrong contact. A falsifying comparison is a fixed boundary switch at index 320 or 460 versus the learned readiness-gated overlap; if fixed switching is equally robust to timing jitter, the gating prior may be unnecessary. It cannot establish recovery after a missed blue grasp, only smooth transition under demonstrated states.

### Handoff interface

**Entry conditions**

The policy can take over from the task start or continue through red release. For its handoff-specific part, demonstrated entry at index 300-320 has red already released on the buffer, gripper open qpos[7:9] about 0.04 m, and TCP rising from z 0.13 to z 0.30 m.

**Exit conditions**

It exits to skill 2 with a broad set of observed states: high/open near the buffer-to-blue translation (index 320-400) or low/closing on blue (index 460). The useful end is not a single pose but a successor-ready corridor from tcp z about 0.30 down to 0.02 m over blue.

**Failure signatures**

A readiness head triggers before red is released, while fingers are still closed on red, or after the TCP has drifted away from the demonstrated blue approach corridor. Observable failures include qpos[7:9] not open during high retreat, red_pose z >0.03 after supposed release, or blue_pose displaced without a stable grasp.

**Overlap role**

The prior is built specifically around the [320,460) expanded overlap. It treats overlap actions as legitimate outputs of both predecessor and successor and learns a stochastic continuation/termination distribution rather than a hard boundary.

**Successor readiness**

The next policy requires the observation to match one of its supported entry states: red_buffered, blue not yet moved or just contacted, gripper open-to-closing, and tcp_pose within the demonstrated approach tube above blue. It remains unseen to switch with a partially carried red block or with gripper closed away from blue.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
