# Acquire active block

Select the active block, move the open gripper to it from either home or the previous placement retreat, close around it, verify/establish attachment, and lift it into an early carry state so delivery can take over.

## Segmentation

This skill groups the approach, descent, gripper closing, attachment and lift of whichever block is active. The red instances start at the beginning of every demonstration from the home/open state and end at observation 220 after the red block is already lifted and moving toward its goal. The blue instances start at observation 340 while the robot is retreating open-handed from the completed red placement; they include the whole transition toward the blue block, blue descent/closure, lift, and early carry until observation 600. Grouping red and blue acquisition tests whether the common learning problem is active-object pickup despite different colors, y-side, and entry pose. The segment intentionally overlaps with delivery during the close/lift/carry transition and, for blue, overlaps with red delivery during the retreat-to-next-pick transition.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo10000`: [0, 220)
- `demo10000`: [340, 600)
- `demo10001`: [0, 220)
- `demo10001`: [340, 600)
- `demo10002`: [0, 220)
- `demo10002`: [340, 600)
- `demo10003`: [0, 220)
- `demo10003`: [340, 600)
- `demo10004`: [0, 220)
- `demo10004`: [340, 600)
- `demo10005`: [0, 220)
- `demo10005`: [340, 600)
- `demo10006`: [0, 220)
- `demo10006`: [340, 600)
- `demo10007`: [0, 220)
- `demo10007`: [340, 600)
- `demo10008`: [0, 220)
- `demo10008`: [340, 600)
- `demo10009`: [0, 220)
- `demo10009`: [340, 600)
- `demo10010`: [0, 220)
- `demo10010`: [340, 600)
- `demo10011`: [0, 220)
- `demo10011`: [340, 600)

## Heuristic 1

Active-object relative servoing. Represent acquisition actions primarily as functions of tcp_pose relative to the selected block pose, not absolute table coordinates or color-specific trajectories. This should improve transfer to modest object-start variation and to switching between red and blue because approach, descend, close, and lift all preserve a consistent tcp-to-object geometry.

**Evidence inspected by the API**

- `demo10000` observations: 0, 80, 120, 160, 220
- `demo10005` observations: 0, 120, 220
- `demo10007` observations: 340, 440, 500, 600
- `demo10011` observations: 340, 440, 500, 600

**Interpretation**

Across red and blue acquisitions, the robot first positions the tcp above the active block, descends until tcp z is close to the block, closes the gripper, then lifts while maintaining a small tcp-object lateral offset. The same relation appears despite different absolute y signs: red acquisition in demo10000 moves from home to red at y about -0.20, while blue acquisition in demo10007 enters from the red-goal area and moves to blue at y about +0.227. The learning difficulty is generalizing from two colors and variable initial xy positions without memorizing global joint trajectories. This prior differs from the phase prior by changing the spatial representation rather than imposing a temporal model, and differs from the attachment prior by not explicitly predicting contact success.

**Applicability**

Transfers to table-top rectangular/cubic blocks of comparable size, a downward-facing parallel-jaw grasp, and similar Panda joint-position control where the active block pose is observable in the world frame. It requires a scheduler or goal-level policy to provide the active object identity (red for the first occurrence, blue for the second in these demonstrations) and assumes the block starts on the table or the preceding delivery left the robot open and collision-free. It should retain the gripper semantics qpos fingers about 0.04 m open and about 0.018 m around the grasped block.

**Implications for a future training/inference pipeline**

Candidate implementation: add segment metadata active_object in {red,blue}. At inference, the scheduler supplies active_object; causal observations supply qpos, qvel, tcp_pose, red_pose, blue_pose, goals. Construct features in world/table frame: p_o = pose(active_object).xyz, p_tcp = tcp_pose.xyz, d = p_tcp - p_o, object yaw/quaternion if useful, gripper width = qpos[7:9], and inactive-object relative vector for avoidance. Train a diffusion policy over absolute joint/gripper actions but condition the denoiser on f=[d, p_o-goal_o, p_tcp.z, gripper_width, qpos_arm, qvel]. The key prior is translation/color sharing: normalize active-object-relative coordinates and use the same network for red and blue acquisition, with the active color supplied only through selected pose tensors, not a separate red/blue policy. Pseudocode: o = active_object; f_t = concat(tcp_xyz-red_or_blue_xyz[o], qpos, qvel, gripper_width, inactive_xyz-active_xyz); a_{t:t+H}=DiffusionDenoise(f_{t-O:t}). Training labels are demonstrated actions; no future labels needed beyond segment active_object.

**Assumptions and limitations**

The demonstrations do not show failed grasps, occlusions, or recovery from large lateral errors; object-centric invariance is a hypothesis, not established. A failure signature would be a policy that centers above the correct object but mistimes gripper closing or clips the block because joint-space dynamics and wrist orientation were under-modeled. Falsify by comparing against a raw-state diffusion policy with no object-relative features on held-out block start positions; if relative features do not improve success or reduce sample complexity, this prior is not useful.

### Handoff interface

**Entry conditions**

Measured support: red-acquisition entries start at observation 0 with tcp_pose about [-0.384,0,0.442] m world, qpos finger channels 0.04 m, both blocks at z=0.02 m. Blue-acquisition entries start at observation 340 after red placement: red_pose is already near red_goal and z=0.02 m, tcp_pose is above the red goal at z roughly 0.21-0.30 m, gripper fingers are open near 0.04 m, and blue_pose remains on the table near x=-0.41 to -0.425, y=0.21-0.23. Hypothesized tolerance: several cm variation in active-object xy if pose observation remains accurate; no demonstrated tolerance to collisions or fallen blocks.

**Exit conditions**

Useful exit is an active block attached between the fingers and lifted into carry: in the red acquisition stops at observation 220, red_pose z is about 0.289-0.293 m and tcp follows it with fingers near 0.0183 m; in blue acquisition stops at observation 600, blue_pose z is about 0.285-0.291 m and has moved toward its goal with fingers near 0.0183 m. The successor delivery can also take over earlier within the overlap after the fingers are closing and the object begins to rise.

**Failure signatures**

Observable failures are active block xy not under tcp during descent, object z staying near 0.02 m after gripper close, qpos finger channels staying open near 0.04 m when the model believes grasp is complete, tcp-object lateral offset exceeding the demonstrated small offset during lift, nonactive block moved, or sudden qvel/object pose discontinuities suggesting collision. These indicate that transfer to delivery is premature.

**Overlap role**

The overlap with delivery [120,220) and [500,600) is where both policies learn close-to-attachment and early lift/carry. Under this object-centric prior, acquisition continues to servo on active-object-relative tcp_pose until the object is attached; delivery learns to accept the same relative geometry and continue toward the goal.

**Successor readiness**

Delivery needs active object id, object pose, goal pose, and a carry state: fingers commanded closed or closing, qpos fingers approximately 0.018-0.02 m, and active object either beginning to lift from z>0.02 m or already at carry height. The measured overlap includes both early lift and high carry states, so the selector need not switch at a single index.


## Heuristic 2

Monotone acquisition phase prior. Model acquisition as an ordered latent sequence from open approach to descent, closure, attachment lift, and early carry. Explicit phase structure should reduce multimodality in the action distribution and make it easier to switch to delivery at several supported overlap states.

**Evidence inspected by the API**

- `demo10000` observations: 0, 80, 120, 160, 220
- `demo10001` observations: 0, 120, 160, 220
- `demo10008` observations: 340, 440, 500, 600
- `demo10010` observations: 340, 440, 500, 600

**Interpretation**

The same segment contains qualitatively different controls: free-space motion with gripper open at observations 0/340, low contact/closing around 120/500, lift at 140-160/540, and early carry by 220/600. A single memoryless policy may average incompatible gripper and vertical actions near the object. The phase prior addresses temporal organization and speed variation; unlike the object-centric prior it does not mainly claim spatial invariance, and unlike the attachment prior it structures the entire approach-to-lift sequence rather than only the contact check.

**Applicability**

Applies when acquisition is a stereotyped sequence with observable progress variables: tcp-object distance, tcp height, finger width, gripper command, and object height. It depends on a parallel-jaw top/side pinch and on the object being rigid enough that closing then lifting gives monotonic changes in object z. It should transfer to demonstrations with variable speeds because phase is inferred from state rather than fixed timestep.

**Implications for a future training/inference pipeline**

Candidate implementation: train a latent-mode diffusion policy with K=5 ordered modes: approach_above, descend_align, close_contact, lift_attach, early_carry. Causal inputs are qpos/qvel, tcp_pose, active_object pose, finger qpos, and previous observations. Training-only labels can be generated from demonstrations: mode_t = rules using future-free current state where possible, e.g. if gripper_command/action <0 and finger qpos decreasing -> close_contact; if object z_t>0.05 and dz/dt>0 -> lift_attach; if object z>0.15 -> early_carry. A small mode encoder h_t = GRU(obs_{t-O:t}) predicts P(z_t|obs history); diffusion denoiser conditions on h_t and z_t embedding. Objective = diffusion action loss + cross-entropy to heuristic phase labels + monotonic regularizer sum max(0,z_t-z_{t+1}) for ordered phases. Deployment computes z_t causally from the encoder, not from future labels.

**Assumptions and limitations**

Phase labels would be heuristically inferred from successful futures and may be noisy around contact; this dataset cannot prove that a phase-conditioned model recovers if it chooses a wrong phase. An ablation should compare learned latent phase/mode conditioning against a single denoiser with identical relative inputs. If the phase model causes brittle hard switches or no improvement in long-horizon action likelihood, the claimed benefit is falsified.

### Handoff interface

**Entry conditions**

Entry includes either phase 0 free-space approach from home/open gripper (obs 0: tcp z 0.442, fingers 0.04) or phase 0/1 transition from previous red placement (obs 340: tcp above red goal, gripper open). The policy assumes active block is still on table z≈0.02 and not already grasped unless selector intentionally enters in the acquisition-delivery overlap.

**Exit conditions**

Exit when latent phase is lift/carry: active object z has risen well above table, fingers are closed/loaded near 0.0183 m, and tcp follows the object. In demonstrated endpoints at 220 and 600, the object is high (red or blue z≈0.286-0.293) and moving toward its goal, but delivery may take over from the overlap as soon as phase posterior indicates attachment/lift.

**Failure signatures**

Phase-inference failure appears as descending while still far laterally from the block, closing too early in free space, lifting before qpos fingers converge, or staying in close phase after object z rises. Observable state-action mismatch, e.g. command gripper open while object is not yet lifted, should prevent handoff.

**Overlap role**

Both acquisition and delivery train on phase labels for the close/lift/carry overlap. Acquisition learns the transition up to attachment; delivery learns to start at several phase values rather than a single lifted state. The red delivery-to-blue acquisition overlap [340,440) provides approach-phase examples beginning from a retreating tcp rather than home.

**Successor readiness**

A delivery successor using this interface receives the inferred phase or can recompute it from causal observations. It is ready when posterior P(lift_or_carry)>threshold, or when in the demonstrated overlap with fingers closing and tcp-object distance small enough that delivery can finish lift. Measured support: obs 120/500 close-to-object low states and obs 220/600 high carry states.


## Heuristic 3

Attachment-confirmation prior. Treat successful acquisition as achieving a verifiable gripper-object coupling, not just executing a close command. Predict whether the active block is attached from recent tcp/object/finger motion and bias actions to continue closure/lift until attachment is confirmed.

**Evidence inspected by the API**

- `demo10000` observations: 120, 140, 160, 220
- `demo10004` observations: 120, 140, 220
- `demo10000` observations: 500, 540, 600
- `demo10002` observations: 500, 540, 600

**Interpretation**

At red observation 120 the object is still near table/contact, by 140-160 it rises with the tcp, and by 220 it has a stable high offset. Blue shows the same at 500-540-600. The hard part is deciding when a grasp is good enough for transport; pure behavior cloning may proceed because time has elapsed rather than because the object is actually attached. This prior is distinct because it makes the learner predict a latent physical coupling condition and use it for actions/handoff.

**Applicability**

Useful when the gripper may close around the block before transport and when object pose is accurately observed, allowing the policy to infer whether the block is attached by comparing object motion to tcp motion. It depends on contact mechanics similar to the demonstrations: closed finger width around 0.018 m for a 4 cm block, sufficient friction, and no requirement to use suction or pushing.

**Implications for a future training/inference pipeline**

Candidate implementation: augment acquisition policy with an auxiliary attachment predictor c_t = sigmoid(g(obs_{t-L:t}, active_object_pose_{t-L:t}, tcp_pose_{t-L:t}, finger_qpos_{t-L:t})). Training labels are computed from demonstration futures: c_t=1 if over the next 0.5-1.0 s the active object z increases with tcp z and the relative tcp-object offset variance stays below a small threshold; c_t=0 before close/contact. The action denoiser conditions on c_t or its hidden representation. Add losses: L = L_diffusion + lambda BCE(c_t,c*_t) + mu ||Delta p_object - Delta p_tcp||^2 for attached-labeled windows. Deployment uses only history up to t for c_t. This prior changes the objective by requiring predictive contact/attachment state rather than only action imitation.

**Assumptions and limitations**

Successful demonstrations do not include failed contact, so negative labels for attachment are synthetic from pre-contact portions only. The policy might overestimate attachment under visual pose noise. Falsify with an ablation removing the attachment auxiliary prediction or by testing deliberately early switches: if attachment-aware switching does not reduce dropped/ungrafted transports, the prior is unsupported.

### Handoff interface

**Entry conditions**

Policy can take over when tcp is approaching/over active block, gripper is open or beginning to close, and active object is still table-supported. It assumes object pose is visible and finger qpos/vel are available. The demonstrated early contact entries are obs 120 for red and obs 500 for blue, where tcp z is near block top and qpos fingers are changing from open to close.

**Exit conditions**

Exit is not merely a lifted height but an attachment belief: active object follows tcp with a stable relative offset while fingers are closed. Demonstrated support: obs 160 red and obs 540 blue show object raised with qpos near 0.018; obs 220/600 show sustained carry. Delivery is ready when attachment probability is high or when delivery is asked to verify attachment itself within the overlap.

**Failure signatures**

Low attachment probability after a close command, object z remaining at 0.02 m while tcp rises, object slipping relative to tcp, or finger qpos saturating at a value unlike the demonstrated grasp are failure signatures. The selector should not hand off to pure transport in those states.

**Overlap role**

In the [120,220) and [500,600) overlap, both acquisition and delivery can learn to monitor attachment. Acquisition uses attachment as its success condition; delivery uses the same signal to decide whether it can start goal-directed transport or must continue lift/settle.

**Successor readiness**

Successor delivery needs the active block to be mechanically coupled to the gripper or at least in a state it can finish coupling from. Provide or recompute an attachment confidence c_t from current/past tcp_pose and object_pose. Measured states range from low closing to high carry; unseen states include object pinched off-center enough to rotate or slip.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
