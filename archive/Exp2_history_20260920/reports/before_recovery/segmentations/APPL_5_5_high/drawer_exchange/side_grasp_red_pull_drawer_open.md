# side_grasp_red_pull_drawer_open

Open the drawer by side-grasping/pulling the red block, then release and retreat so red can be regrasped from above.

## Segmentation

Segments start at the initial home/open-gripper state and end after drawer opening, side-grasp release and upward reorientation. At index 0 drawer_position is 0 and red is at x about 0.09-0.11; by index 145 fingers are closed around red and drawer begins moving; by index 240 drawer_position is about 0.30 and gripper has reopened; by index 330 tcp is retreating above the open drawer. Stop 330 preserves substantial transition context for the following regrasp skill.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo1000`: [0, 330)
- `demo1001`: [0, 330)
- `demo1002`: [0, 330)
- `demo1003`: [0, 330)
- `demo1004`: [0, 330)
- `demo1005`: [0, 330)
- `demo1006`: [0, 330)
- `demo1007`: [0, 330)
- `demo1008`: [0, 330)
- `demo1009`: [0, 330)
- `demo1010`: [0, 330)
- `demo1011`: [0, 330)

## Heuristic 1

Red-relative side-grasp servo: represent the pull as reaching a fixed pose relative to red, closing, then pulling the red/drawer system, reducing sensitivity to initial red lateral placement.

**Evidence inspected by the API**

- `demo1000` observations: 0, 125, 145, 240, 330
- `demo1001` observations: 0, 125, 145, 240, 330
- `demo1009` observations: 0, 240, 330

**Interpretation**

Across demonstrations, tcp moves from home to a pose just behind red, fingers close around index 145, drawer_position rises to 0.30 by index 240, then fingers reopen and tcp retreats by index 330. The learning difficulty is that absolute joint replay must generalize over red y/x variation; using red/tcp relative geometry should make the initial approach and side grasp reusable rather than memorizing one joint path.

**Applicability**

Transfers when red begins on/in the drawer front region at z about 0.063 m, the drawer slides along world x, and the gripper can side-pinch the red block strongly enough to pull the drawer. Requires the same Panda joint/gripper actuation and comparable drawer friction/contact geometry.

**Implications for a future training/inference pipeline**

Policy candidate uses causal observations qpos,qvel,tcp_pose,red_pose,drawer_position,drawer_velocity and outputs normalized arm_joint_positions plus gripper_command. Encode features e=[T_tcp^{-1}red, drawer_position, drawer_velocity, qpos_fingers]. The prior is a red-relative spatial transformer before a diffusion action decoder: a_t ~ D_theta(a_t | e_t, history). Demonstration futures provide labels for phase/progress only during training, not inference.

**Assumptions and limitations**

Successful demos do not establish behavior if the first side pinch misses, if red rotates out of the drawer slot, or if the drawer sticks. Falsify by comparing to an unconstrained DP on held-out red y positions and drawer friction; this prior should reduce drawer-axis errors and premature release but could fail if the object-relative frame is misestimated.

### Handoff interface

**Entry conditions**

Measured support: at segment starts index 0, qpos arm is the home posture [0,-0.3,0,-2.1,0,1.8,0.785] rad, gripper qpos fingers are open near 0.04 m, drawer_position=0 m, red_pose is near x 0.093-0.108, y -0.077 to -0.055, z 0.063 m, blue is untouched on the table. Hypothesized tolerance is only small object-pose variation similar to the 12 demos.

**Exit conditions**

Useful end state is an open drawer (drawer_position approximately 0.300 m), red translated to x about -0.195 to -0.210 at z 0.063 m, gripper open near 0.04 m, and tcp_pose retreating upward/away by index 330 so the red top can be approached.

**Failure signatures**

drawer_position remains below about 0.26 m by the retreat portion, red_pose is not co-moving with tcp during the pull, gripper qpos does not close to about 0.007 m during contact or reopen near 0.04 m before retreat, or tcp collides/jams at the drawer face.

**Overlap role**

The [240,330) overlap teaches both this skill and red top-regrasp to handle the first-grasp release, drawer-open confirmation, and upward reorientation; successor need not assume a single exact release pose.

**Successor readiness**

red_top_regrasp_lift needs drawer_position >0.26 m, red visible at front-left of the open drawer, gripper open, tcp above/front of red with enough clearance to descend, and red still on the drawer surface rather than already carried.


## Heuristic 2

Contact-phase automaton: encode approach-close-pull-release-retreat as a latent temporal organization to improve gripper timing and avoid blending incompatible actions.

**Evidence inspected by the API**

- `demo1002` observations: 125, 145, 240, 330
- `demo1003` observations: 125, 145, 240, 330
- `demo1011` observations: 240, 330

**Interpretation**

The same indices show abrupt control logic: open gripper during approach, close at contact, maintain closed while drawer opens, then reopen and retreat. A single stationary policy must learn discontinuous gripper/arm couplings; an explicit mode prior separates contact decisions from continuous motion.

**Applicability**

Best for the same mechanical sequence with a drawer whose opening is observable as drawer_position and whose motion is monotonic along the demonstrated axis. It assumes contact transitions can be inferred from qpos finger gap, red/tcp co-motion and drawer velocity.

**Implications for a future training/inference pipeline**

Train a hybrid diffusion/HSMM candidate. Causal inputs are qpos,qvel,tcp_pose,red_pose,drawer_position,drawer_velocity. Training labels z_t in {approach,close,pull,release,retreat} are computed from demonstration futures/thresholds: gripper qpos, drawer_position derivative, and red motion. Inference maintains p(z_t|o_<=t); action decoder D_theta(a|o,z) produces joint targets and gripper command. Loss = diffusion denoising + CE(z_t) + drawer progress regression.

**Assumptions and limitations**

Phase labels inferred from demonstrations are not measured contact forces. The prior cannot prove recovery from a bad pinch or drawer overshoot. Ablate the latent phase model against a single diffusion model; benefit is falsified if phase-conditioned prediction does not improve close/release timing on held-out demos.

### Handoff interface

**Entry conditions**

Can start with open fingers at home or already approaching red as in indices 0-125, provided red_pose is on the drawer front and drawer_position <0.02 m. The gripper_command should be open before the close phase.

**Exit conditions**

Exit after the release/retreat latent state: gripper open near 0.04 m, drawer_position stable near 0.30 m, red on the open drawer surface, tcp rising toward z about 0.37 m by index 330.

**Failure signatures**

Latent state sequence skips close before pulling, drawer_velocity does not become positive during pull, red_pose lags tcp by more than a block width, or the policy remains in pull after drawer_position saturates.

**Overlap role**

The release-and-retreat part is intentionally shared with the next skill so either policy can learn whether to finish opening or begin the top-grasp reorientation from a partially changing arm pose.

**Successor readiness**

Successor receives a discrete/continuous state with phase=retreat/open, drawer_position >0.26 m and gripper open, not phase=pull with fingers closed.


## Heuristic 3

Drawer-progress coordinate: organize the pull around monotonic scalar drawer opening, biasing actions toward world-x progress until the drawer-open threshold is safely exceeded.

**Evidence inspected by the API**

- `demo1004` observations: 145, 240, 330
- `demo1005` observations: 145, 240, 330
- `demo1006` observations: 145, 240, 330

**Interpretation**

The task-relevant result of this skill is the drawer crossing 0.26 m, observed consistently by index 240. A progress coordinate can give the learner a low-dimensional termination and control variable, reducing ambiguity between pulling farther and retreating.

**Applicability**

Applies to drawers whose useful progress is the scalar drawer_position and where pulling red/drawer along world -x is adequate. Requires reliable state observation of drawer_position and drawer_velocity.

**Implications for a future training/inference pipeline**

Candidate augments DP with scalar progress c=drawer_position/0.30 and target c*=1. Inputs include drawer_position and drawer_velocity in native metres and world tcp/red features. Add auxiliary heads predicting c_{t+H} and sign(drawer_velocity). Decoder action is penalized during closed-gripper phases for tcp velocity components orthogonal to world x in Cartesian reconstruction from FK: L=L_diff+lambda||v_tcp,yz||^2+MSE(c_{future}). Labels c_future are from future observations for training only.

**Assumptions and limitations**

A pure 1-D progress bias may ignore lateral alignment or red grasp quality; if red slips sideways it may still believe progress is adequate. Compare against the red-relative prior; falsification is good drawer progress prediction but worse grasp/release accuracy.

### Handoff interface

**Entry conditions**

Start is supported from closed drawer_position 0 and red at z 0.063 with tcp above/approaching. It can also take over in overlap after the drawer is already open if gripper has released.

**Exit conditions**

Drawer_position strictly exceeds the task threshold 0.26 m and is near 0.30 m; drawer_velocity near zero; red and tcp no longer need to apply pulling force; fingers open for successor descent.

**Failure signatures**

Drawer progress is nonmonotonic, tcp moves mostly off the world-x pull line while fingers are closed, or release occurs before drawer_position >0.26 m.

**Overlap role**

In [240,330), both policies see drawer progress already achieved and learn the transition from progress-controlled pulling to pose-controlled top-grasp setup.

**Successor readiness**

The next policy needs stable drawer_position >0.26 m and a red_pose that remains reachable; this prior should hand off only after its progress head predicts completion with high confidence.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
