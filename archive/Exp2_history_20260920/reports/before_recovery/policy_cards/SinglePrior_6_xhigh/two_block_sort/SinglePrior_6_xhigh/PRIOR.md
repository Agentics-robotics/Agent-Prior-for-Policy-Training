# SinglePrior_6_xhigh: goal-relative predictive object factorization

## Scientific choice

This is one full-task learned action diffusion policy, not a collection of skills. The selected prior is that repeated manipulation is more economically represented through **object–TCP–paired-goal relations and their short-horizon evolution** than by unrelated coordinates for every color and every point in a long demonstration. A shared object encoder and shared relational forecasting head expose that structure to one diffusion U-Net. Ordered color identity, the absolute robot state and the original observation history remain available because absolute Panda joint commands are not equivariant under swapping or translating objects.

The expected benefit is sample efficiency and state-based transition discrimination. An unplaced object on the table, a grasped object moving with the TCP, a lowered object on its pad, and a released object left behind have different relational evolution even when some robot configurations are similar. Forecast supervision encourages the learned condition to retain these distinctions. It is a hypothesis grounded in the training demonstrations, not a measured rollout improvement.

## Training evidence inspected

I read the assignment, numerical public interface and backbone, whole-trajectory overviews for all twelve complete demonstrations, and dense source observations/actions around transitions in demo10000 and demo10011. The complete trajectories, including transitions and final retreats, are used for training; none is cut into skills.

| Trajectory | Inspected source evidence in addition to reset and final state |
| --- | --- |
| demo10000 (0–773) | 24-frame overview; detailed indices 104–108, 110–112, 116, 122, 285, 291–292, 294–298, 312, 350, 380, 502, 505, 510, 683, 690–693, 704 |
| demo10001 (0–765) | 153 red lifted; 306 red released; 459 blue approach/descent; 612 blue transport |
| demo10002 (0–770) | 154 red lifted; 308 red released; 462 blue descent; 616 blue transport |
| demo10003 (0–756) | 151 red lifted; 302 red released; 453 blue descent; 604 blue transport |
| demo10004 (0–770) | 154 red lifted; 308 red released; 462 blue descent; 616 blue transport |
| demo10005 (0–771) | 257 red high near its pad; 514 blue newly lifted with red already placed |
| demo10006 (0–773) | 257 red high near its pad; 515 blue closed grasp at table height |
| demo10007 (0–760) | 253 red lowering near its pad; 506 blue closed grasp at table height |
| demo10008 (0–775) | 258 red lowering near its pad; 516 blue closed grasp at table height |
| demo10009 (0–773) | 257 red lowering near its pad; 515 blue closed grasp at table height |
| demo10010 (0–774) | 258 red high near its pad; 516 blue just starting to lift |
| demo10011 (0–777) | Overview 259 and 518; detailed 103, 108, 124, 288, 302, 348, 378, 503, 507, 529, 682, 696 |

Specific observations motivating the representation:

* In demo10000 reset, red is approximately (-0.4196, -0.2018, 0.0200), blue (-0.4182, 0.2132, 0.0200), and the open TCP is at z=0.442. At index 100 the TCP is down at red while the command is still open. Index 107 changes the command from +1 to -1 with almost unchanged joint targets; index 108 has finger positions about 0.0204 m and by 110 about 0.0183 m. At 122 lift is only beginning, at 134 red is z=0.0557, and at 168 red is z=0.2783 with TCP z=0.2784. The small approximately 9.6 mm x-offset between TCP and red persists during transport. Absolute proximity alone does not specify the grasp stage; finger state and relational motion matter.
* At demo10000 201 the red block and TCP travel together toward the red pad. At 235 red is around (-0.1742, -0.2432, 0.2930), high over the goal rather than completed. At 285 red is z=0.0287; at 296 the gripper opens with red z=0.0227; at 297 red settles to z=0.0200. Height and goal error must both be represented. Release is a demonstrated transition, not an extra evaluation success requirement.
* In demo10000 312–350 the open TCP retreats vertically while red remains on its pad. At 380 it travels toward blue, whose original position is unchanged. At 502 blue approach is still open; the overview shows command -1 at 504; at 505 the fingers have begun closing; by 510 they are near 0.0183 m. The same contact/transport geometry recurs with a different object and different absolute joint targets.
* In demo10000 537–638 blue is lifted and transported while red remains placed. At 683 and 690 blue is lowered near its pad but still held; at 691 the command opens and at 692 the blue block settles. At 739 and 773 both objects remain placed while the TCP is high. The model is trained on these tail states even though the evaluator can terminate earlier.
* Demo10011 confirms the pattern with timing and pose variation: 103 open approach, 108 closing, 124 red lift onset, 288 lowered red, 302 open after placement, 348 retreat, 378 inter-object travel, 503/507 closed blue grasp, 529 blue lifted, 682 blue lowering, 696 open and blue settled. In particular, blue grasp occurs earlier than in demo10000, so an episode-index schedule would be an inappropriate deployment representation.
* Across the inspected resets, red x spans approximately -0.4290 to -0.4119 m and blue x -0.4247 to -0.4080 m. Red y is approximately -0.2110 to -0.1912 m; blue y 0.2082 to 0.2319 m. Goals are fixed at (-0.18, -0.25, 0.02) and (-0.18, 0.25, 0.02). All observed final states have both objects near those goals. These are small layout changes, not evidence of broad spatial generalization or alternative task orders.

## Implemented model

`policy.py` contains the entire model and loss. There is one `DiffusionBackbone` with the supplied 128/256/512 U-Net widths, kernel 5, groups 8, and diffusion timestep embedding 128. It predicts epsilon for the existing [B,16,8] normalized absolute joint/gripper action tensor. The native seven-joint-plus-gripper action parameterization, action normalizer and sampler are not changed.

### Causal object features

Each of the two input observations contributes 39 features for each object:

1. Object position and its paired goal position, divided by 0.5 m (six values).
2. TCP minus object, goal minus object, and goal minus TCP, divided by 0.25 m (nine values).
3. Object world rotation and TCP-relative object rotation, each flattened to nine values. Wxyz quaternions are unit-normalized before conversion to rotation matrices.
4. Both finger positions divided by 0.04 m (two values).
5. Two signed XY containment margins divided by pad half-width 0.06 m. Projected cube extent is `0.02 * sum(abs(R), columns)` for each world axis, and the margin subtracts that extent and absolute center-to-goal displacement.
6. TCP–object and object–goal Euclidean distances divided by 0.25 m (two values).

The two frame vectors are concatenated, not averaged. Additional features are object displacement, TCP displacement and their difference over the one-observation interval, each divided by 0.025 m, plus the two finger changes divided by 0.02 m. These are displacements, not velocities in metres per second. A two-component red/blue identity finishes the 91-dimensional input. The feature scales are fixed physical scales, not a per-phase empirical refit. No observation clipping is applied. Quaternion-derived features are sign invariant; the retained original raw-normalized quaternion channels mean the entire network is not guaranteed sign invariant.

The same 91→192→128 MLP is applied to both objects (SiLU, with LayerNorm before the final SiLU). Both ordered tokens are preserved. A 222→192→128 scene MLP receives the full 94-dimensional shared-normalized observation history and the mean of the two object tokens. Thus both object configurations, arm qpos/qvel, TCP pose and gripper observations can influence forecasts. The zero drawer compatibility channels are simply retained in the original observation vector; no drawer behavior is introduced.

The geometric margins are only inputs. They do not choose an object, prohibit an action, add a reward, trigger a phase transition, or stop deployment. Z differences remain explicit in the three-dimensional relations and distances.

### Learned relational forecasts

For each object, a shared 256→192→24 head takes its 128-dimensional token and the 128-dimensional scene embedding. It predicts six-dimensional **changes** in relations at four future states: TCP-minus-object (three components) and object-minus-own-goal (three components). Relation changes are expressed in units of 0.1 m. A separate 128→64→4 head predicts change in mean finger position in units of 0.04 m.

The four future labels use action slots [1,5,9,15]. Under the supplied alignment, slot j is action t-1+j and its label is the state after that action, namely t+j. These labels are therefore states t+1, t+5, t+9 and t+15. Each label subtracts the most recent observed relation or finger mean. No slot-zero duplicate of the current state is used as a forecast target.

The U-Net condition is exactly 402 values:

* 94 original history values normalized with the shared limits normalizer;
* 256 values from the two ordered shared object tokens;
* 48 predicted relational changes;
* four predicted finger changes.

The forecasts are computed only from the two causal observations, at both training and inference. Ground-truth futures are never substituted into conditioning. There is no action-conditioned rollout or claimed physics model inside this auxiliary head: it learns a short-horizon demonstrator forecast, not a guarantee about the consequences of each sampled action. Keeping the original history and tokens alongside forecasts avoids making a possibly averaged forecast a hard control bottleneck.

## Objectives and actual gradient paths

The primary term is exactly the supplied masked epsilon MSE:

`L_diffusion = epsilon_loss(predicted_epsilon, noise, mask)`.

Forecast loss uses elementwise Smooth L1 with beta 0.5. Future-label masks are intersected with action masks at the four selected slots. Relational errors are averaged over the two objects, six coordinates and valid slots. Finger errors are separately averaged over valid slots. The returned prior term already includes its weight:

`L_prior = 0.1 * (L_relation + 0.5 * L_finger)`

`L_total = L_diffusion + L_prior`.

These are losses on trainable predictions, not penalties computed only from observed states. In detail:

* Epsilon loss updates the entire U-Net. Through FiLM conditioning it also updates the shared object encoder, scene encoder and both forecasting heads. Forecast outputs are not detached before conditioning.
* Relational forecast loss directly updates the shared relation head and, through its inputs, the scene encoder and object encoder. Finger forecast loss updates the finger head and the same shared encoders.
* The auxiliary term does not directly supervise U-Net output or enforce consistency between sampled actions and forecasts. Its direct role is representation learning; epsilon-loss gradients provide the action-learning connection.
* Physical feature calculations and labels have no trainable parameters. In particular, observed containment margins are not presented as a standalone differentiable prior loss.

There is no auxiliary reconstruction of denoised x0 and no division by diffusion alpha-bar. Auxiliary gradients therefore have no high-noise inverse-SNR amplification. The forward mapping remains epsilon prediction for the ordinary DDPM process; there is no residual-action decoding or alternative sampler.

## Training, deployment and boundaries

All source windows come from the twelve supplied complete trajectories, using the repository's fixed batch masks and shared normalizer. The recipe is unchanged: seed 0; 60,000 updates; batch 128; history 2; horizon 16; execution prefix 8; DDPM100 training and sampling; epsilon prediction; clip_sample true; AdamW learning rate 1e-4 and weight decay 1e-6; cosine LR with 500 warmup steps; gradient clipping at 1; EMA 0.999. Final EMA at the predeclared budget is the only selected checkpoint. No rollout-based selection, trajectory replay, augmentation, extra data or per-phase fitting is used.

`forward` is stateless. Its only deployment inputs are noisy action, diffusion timestep and two raw observations. The diffusion timestep is not an episode clock. There is no persistent memory, policy selector, external skill schedule, hand-coded waypoint, IK solver or controller. Both forecasting heads are parameters of the same model/checkpoint and supply conditioning, never executable actions. Future observations and masks occur only in `compute_loss` as labels. The single model also learns the observed released-object retreats and inter-object transitions from the complete action loss.

The unchanged evaluator decides simultaneous geometric red/blue goal success, or reaches its physical step limit. This package adds no gripper-release, TCP-clearance, velocity or hold-duration condition. Metadata is documentation only.

## Limitations and falsifiability

The demonstrations provide little recovery coverage and narrow positional/orientational variation. They consistently show red before blue in the inspected task progression. Identity and ordered absolute state allow learning that preference, but there is no claim that arbitrary order changes, missing objects, unusual grasps or displaced pads will be solved. Sharing object semantics is not a proof of symmetry in absolute joint actions.

Two observations cannot distinguish all near-identical stationary dwell states. The precise open/close decision at the end of a pause may remain ambiguous. State-only forecasts can average divergent futures; the modest robust auxiliary loss and uncompressed history path reduce, but cannot eliminate, that risk. The model could also underuse the forecast condition. No learned forecast is guaranteed physically accurate, and no collision or grasp constraint is enforced.

The interpretation can be tested later by comparing against an otherwise matched policy without relational features/forecast training, but no such candidate is trained or selected here. Interface checks test shape, finite gradients, sampling and checkpoint reload only, not task success. No test layouts or rollout performance have been used for this design.
