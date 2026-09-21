# Occupancy-aware relational diffusion with causal interaction forecasts

## Scope and scientific choice

Policy: `SinglePrior_6_xhigh`. Interface skill: `full_task`. Candidate identity: 1.

This is one learned action diffusion policy for the entire buffer swap, not a collection of skills. Its inductive bias is that manipulation decisions depend on **relations among the gripper, both objects and their assigned regions**, and that a useful representation of those relations should predict near-term physical interaction. In particular, the other object's position relative to a goal should be explicit rather than recovered indirectly from an unstructured state vector. The observed arrangement can provide task-progress information even without an episode clock.

I implement this as shared object relational encoders, one learned cross-object message round, and an auxiliary causal motion-forecast head whose predictions also condition the denoiser. There is no discrete phase variable, hard attention to a chosen block, target-switch rule, prescribed buffer coordinate, inverse kinematics, action controller, or external memory. Color role information is retained; exchangeability is a weight-sharing bias, not an enforced red/blue symmetry of behavior. The network must learn where to put the temporarily displaced object and how to command the joints from the demonstrations.

## Evidence actually inspected

I read the assignment, interface, numerical helpers and U-Net source before implementing. I inspected a 24-frame overview of `demo10100`, a four-frame whole-trajectory overview of each other complete demonstration, and denser source steps in `demo10100` and `demo10107`.

The sparse cross-demonstration evidence is below. In every listed first interior row, red is resting near (-0.181, 0, 0.020) while blue is still near its initial positive-y region. In every second interior row, blue is near its negative-y goal while red remains in that central area. At the final row both colors are near their assigned opposite regions. These are sparse observations of the complete trajectories, not phase boundaries used for training.

| Trajectory | Overview source indices inspected |
| --- | --- |
| demo10100 | 0, 46, 92, 138, 185, 231, 277, 323, 370, 416, 462, 508, 555, 601, 647, 693, 740, 786, 832, 878, 925, 971, 1017, 1064 |
| demo10101 | 0, 354, 709, 1064 |
| demo10102 | 0, 353, 706, 1060 |
| demo10103 | 0, 355, 710, 1065 |
| demo10104 | 0, 352, 704, 1057 |
| demo10105 | 0, 352, 704, 1056 |
| demo10106 | 0, 353, 706, 1059 |
| demo10107 | 0, 357, 714, 1071 |
| demo10108 | 0, 352, 705, 1058 |
| demo10109 | 0, 354, 708, 1063 |
| demo10110 | 0, 354, 709, 1064 |
| demo10111 | 0, 353, 707, 1061 |

The statement about the two interior rows applies to the four-frame overviews; the denser demo10100 overview exposes the full progression:

- At `demo10100:0`, red is (-0.3550, -0.2094, 0.0200), blue is (-0.3601, 0.1945, 0.0200), and the open TCP is at height 0.4420. Each object's destination is initially occupied by the other object.
- `demo10100:96` has the open TCP beside red at table height. At 104, the command is -1 and fingers have closed to approximately 0.0183 each. At 116 lifting begins; at 138 red has risen to z=0.1706. Thus aperture, TCP/object relative motion and object height distinguish approach from transport even when xy is nearly unchanged.
- At 185 red is moving through (-0.2791, -0.1160, 0.2867), not directly toward its occupied goal. At 260 it is held near (-0.1814, -0.0004, 0.0232); at 268 the command is +1 and red has settled on the table. At 277 it is released in the temporary area, not at its own goal. A monotonically decreasing goal-distance penalty would incorrectly oppose this useful transition.
- At 450 the gripper is closing around blue while red remains buffered. At 470 blue starts rising, and at 508 it is carried at z=0.2791. At 555 blue has crossed toward negative y. At 630 it is down near (-0.3523, -0.2007, 0.0219); at 640 the fingers are opening. This second transport frees red's destination.
- At 740 and 786 the TCP returns to the buffered red. At 810 the close command begins although the observed fingers are still open; at 824 they are closed. At 832 red is lifting, at 925 it is carried near its goal, and at 971 it is at z=0.0236. At 980 it is released; 1017 and 1064 show retreat. The later return to the same buffered object must be distinguished from the earlier buffer deposit using blue's arrangement and robot state.

The denser additional source readings were `demo10100:[96,104,116,260,268,450,470,630,640,810,824,980]` and `demo10107:[96,112,264,280,450,474,634,650,806,830,968,984]`. In demo10107 the analogous open/closed first grasp is visible at 96/112, the buffered red deposit at 264/280, the blue grasp at 450/474, blue placement/release at 634/650, the buffered-red regrasp at 806/830, and final placement/open command at 968/984. Transition timing varies: for example, blue's gripper command is still +1 at demo10107:450 but already -1 at demo10100:450. Source time is therefore neither an input nor a control schedule.

Initial blue x ranges from about -0.3614 (demo10107:0) to -0.3386 (demo10103:0), with only small perturbations in the other reset coordinates. Goals remain fixed and observed orientations are almost constant. This is evidence for a common interaction structure, but weak evidence for generalization to widely different scenes.

## Exact parameterization and forward computation

The inputs remain noisy normalized actions [B,16,8], a diffusion timestep, and two raw observations [B,2,47]. The output is epsilon [B,16,8]. There is no changed action coordinate system: seven absolute Panda joint targets and the native gripper command are normalized and decoded by the repository.

1. **Unchanged observation path.** The supplied shared mean and half-range buffers normalize all 47 channels without clipping. Both ordered frames are flattened to 94 features. An MLP 94 -> 192 -> 96 with Mish and final LayerNorm/Mish makes a robot/context token. All 94 normalized features also have a direct shortcut to the diffusion condition. This preserves world positions, joint configuration, velocities and quaternions: relative Cartesian inputs alone would not determine absolute joint commands.
2. **Object relations.** For each color in each frame, 31 features comprise six 3-vectors, its four quaternion components, one relative height, four continuous proximity features, two finger apertures and a two-component color identity. The vectors are object-minus-TCP, assigned-goal-minus-object, other-object-minus-assigned-goal, other-object-minus-object, assigned-goal-minus-TCP and opposite-goal-minus-object. Vectors and relative heights are scaled by 0.25 m. Finger positions use 0.04 m. Proximity is exp(-0.5 times squared scaled distance): TCP proximity uses 0.04 m; own-goal, opposite-goal and other-object-at-own-goal features use [0.06,0.06,0.03] m. These are smooth descriptors, NOT contact or success predicates, constraints, probabilities, or runtime switches.
3. **Causal temporal relations.** Two ordered per-object frames contribute 62 features. Object displacement and change of object-minus-TCP displacement add six features, scaled by 0.025 m (a stable step-displacement scale at 20 Hz). They are computed only from the two supplied observations. The shared MLP 68 -> 192 -> 96, again with final LayerNorm/Mish, produces two object tokens. No narrow empirical position subrange is used to divide the derived geometry.
4. **One message round.** For each object, concatenate its token, the other object's token and the robot token (288 values). A shared MLP 288 -> 192 -> 96 predicts a residual message. LayerNorm of token plus message gives the updated token. Concatenating robot, red and blue tokens gives a 288-dimensional scene representation. This explicitly gives each object's processing access to the state of the other object and robot.
5. **Causal interaction forecast.** An MLP 288 -> 192 -> 33 predicts three eleven-dimensional vectors. Each contains TCP, red and blue displacement (nine coordinates in 0.10 m units), and absolute two-finger aperture in 0.04 m units. The three labels correspond to future slots 4, 8 and 15. Under the supplied action/future alignment, future slot zero is the state after action t-1, so these are approximately 0.20, 0.40 and 0.75 seconds ahead of the current observation. The selected horizons span part and all of the eight-action execution window and most of the sixteen-slot chunk.
6. **One diffusion U-Net.** Concatenate the 94 normalized observations, 288 scene features and 33 predicted forecast values into a 415-dimensional global condition. The public DiffusionBackbone uses exactly the fixed widths 128/256/512, timestep embedding 128, kernel 5, eight groups and its learned FiLM conditioning. It alone outputs the action noise prediction. The two helper prediction methods execute the same encode/backbone path; the training method additionally returns the forecast for supervision.

All encoders, the forecast head and the U-Net are jointly trained and stored in the same model/checkpoint. The intermediate hidden MLP activation is Mish; no hard routing, phase-specific modules or mutable hidden state are used. Forward can be called any number of times during DDPM without advancing task state.

## Objective and real gradient paths

The diffusion term is the repository's mask-normalized epsilon MSE over all valid action slots and all eight channels. The auxiliary term is Smooth L1 (beta 0.25) between the three predicted interaction vectors and the corresponding future observation labels. Cartesian targets are displacement from the latest causal observation; aperture targets are absolute future finger positions. Future labels are selected with slots [4,8,15] and their future masks. The auxiliary term is summed over valid coordinates and divided by valid future slots times eleven, with a denominator floor of one.

`loss = diffusion_loss + 0.10 * prior_loss`.

The weight, scales, widths and horizons were predeclared without rollout feedback. No future labels, native expert actions, phase annotations or demonstration IDs enter forward. The denoiser receives its own forecast head's predictions during BOTH training and deployment: there is no teacher forcing or privileged feature substitution.

Gradient paths are explicit:

- Epsilon MSE trains the complete U-Net and backpropagates through its learned condition projections into the scene encoders and forecast head.
- The forecast is not detached from the condition. Consequently the action loss can tune which predictive information helps denoising.
- The future-label loss trains the forecast head, robot encoder, shared object encoder and message encoder. It does not directly train U-Net weights, since the forecast head precedes that U-Net; its effects on action predictions pass through their shared conditioning representation and the learned forecasts.
- Fixed geometric features have no parameters and are not presented as an observed-state-only learning penalty. They structure differentiable trainable processing. The auxiliary loss compares trainable predictions with labels and therefore has a real learning effect.

No x0 is reconstructed for the prior. Consequently no division by sqrt(alpha_bar) amplifies the auxiliary gradients at high diffusion noise. The future masks handle complete-trajectory padding. We do not impose goal-distance monotonicity, terminal-goal attraction at each timestep, zero object motion, or a grasp/release schedule; such penalties would conflict with observed buffering and transit.

## Training and autonomous execution contract

Train on all twelve original full trajectories without cuts or per-phase normalizers. Use the unchanged repository recipe: seed 0, 60,000 updates, batch 128, two observations, horizon 16, eight executed actions, epsilon DDPM with 100 training and sampling steps and clipped samples, AdamW learning rate 1e-4 and weight decay 1e-6, 500-step warmup plus cosine schedule, gradient clipping at 1, EMA 0.999, and last EMA selection. No checkpoint is selected by evaluation. The auxiliary heads are included in EMA with the rest of the network.

The metadata is documentation, not a deployment controller. The evaluator's simultaneous geometric success (red_at_goal AND blue_at_goal) or physical step limit is unchanged. No extra release, speed, TCP-clearance or hold requirement is introduced, although complete demonstrated release and retreat samples remain in the learning dataset.

## Expected benefit, limitations and uncertainty

The expected benefit is sample-efficient differentiation of approach, coupled carrying, placement, and revisiting a buffer, grounded in explicit occupancy and near-term motion rather than demonstration time. Shared object processing can reuse information from all three pick/place cycles. The forecast objective encourages a condition that preserves which bodies are about to move and what the fingers are doing, rather than merely reconstructing static pose. The full observation shortcut permits the denoiser to use joint posture and detail that the structured encoder may not retain.

This is an inductive bias, not a guarantee of vacancy planning or correct mechanics. The learned forecast is behavior-conditioned prediction from a short history, not an action-conditioned simulator, collision checker, force model or online optimizer. It can average ambiguous futures at dwell points, and it only predicts subsecond motion, not whole-task consequences. Its utility may be limited if the U-Net ignores it; no result demonstrating a benefit is available.

The two causal observations cannot resolve every alias, slip, missed grasp or earlier event. Proximity does not prove that an object is held. Gaussian goal descriptors do not implement the evaluator's rotated containment test; the original quaternions remain available, but the data provide little orientation variation. Explicit red/blue roles preserve the demonstrated asymmetry and may reduce transfer to reversed-order tasks. The full normalized shortcut still contains narrowly ranged raw coordinates, so the stable derived scales do not eliminate all extrapolation risk. Relative features do not confer equivariance on absolute joint actions.

All demonstrations use a similar choreography and fixed goals with small reset changes. Different buffer locations, obstructed paths, changed robot calibration, large layout changes and recovery from unusual states are not established by this training evidence. Interface checks are only numerical/package diagnostics with two fresh updates and sampler/EMA reload, not task-performance measurements; no test layouts or rollout feedback were used in this design.
