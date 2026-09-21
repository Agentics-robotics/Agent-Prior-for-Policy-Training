# Prior: acquire_block__h01

## Assigned hypothesis and scope

The assigned heuristic (index 1) is **Role-factored relative geometry**. Its original statement is: "Role-factored relative geometry: acquisition depends more on TCP-to-block geometry and manipulation roles than block color or absolute source xy. Share object-slot encoders and expose relative transforms to reduce coordinate memorization while retaining the fixed-base robot state."

This package implements that representation-sharing hypothesis as a learned action diffusion policy. It does not edit the source heuristic, replace it with temporal-mode classification, or use attachment prediction/action ranking. Relative geometry is an inductive bias, not a guarantee of spatial invariance. In particular, the Panda actions are absolute joint targets, not equivariant Cartesian displacements.

The assigned skill is the FULL expanded acquisition slice: initially approach, close around, lift and begin translating red; on its second invocation finish incoming red lowering/release, retreat and traverse to blue, then approach, close, lift and begin translating blue. All 24 assigned half-open ranges are preserved in pipeline.json. No placement-overlap frames are dropped. The model is trained on every available masked action slot, including shared transitions. The measured overlaps with deliver_block are 106-122 actions for selected-block approach/close/lift and 149-154 actions for incoming red placement/release/retreat/traversal. In demo10300 these are [75,182), [265,415), and [445,562). These are training-support intervals, not runtime phase schedules.

## Necessary adaptations to the executable interface

1. **History and execution:** the research suggestion of four observations and two executed actions is not available. This experiment fixes two causal observations, horizon 16, and eight executed actions per replan. The encoder uses both observations and their differences, plus measured qvel. It has no persistent hidden state and no time-index input. Action slots are the framework's t-1 through t+14; neither loss nor forward shifts these slots.
2. **Intended-object token:** forward accepts only noisy actions, timestep and a [B,2,47] state history. There is no dynamic next-object token or extra channel. We therefore cannot faithfully supply the original explicit intent input. Instead, both pose/goal bundles enter shared geometry encoders and three learned scene-conditioned latent-role queries pool them. The queries can represent simultaneous incoming-payload and next-source information but receive no semantic role labels; their interpretations are not guaranteed. Goal location, object-goal displacement, object/TCP relation, world height and robot motion supply context for the demonstrated red-first ordering. There is no hard pose-based switch. A request to acquire blue from the otherwise identical initial state cannot be distinguished from a request to acquire red. The API should select this policy only when its intended object is consistent with demonstrated scene/order context. This is a material limitation, not evidence of arbitrary intent following.
3. **Sampler and consistency:** batch construction is framework-owned. We do not claim role-balanced resampling. All examples use ordinary masked diffusion loss. Complete pose-and-goal bundle exchange is handled structurally through shared encoders, common scales and symmetric attention pooling, so no additional consistency loss is used. Exchanging just the poses but not the associated goals is NOT the same physical problem.
4. **No unprovided modalities or solvers:** this is a state-only network. No image encoder, IK/FK provider, grasp oracle, contact solver, collision filter or model-based ranking is used. No sampled action is replaced with a scripted action.

## Causal representation and learned architecture

All coordinates below are world metres; quaternion input is wxyz. Fields are qpos 0:9, qvel 9:18, TCP pose 18:25, red pose 25:32, blue pose 32:39, red goal 41:44 and blue goal 44:47. The constant drawer compatibility fields 39:41 are ignored. Poses and their corresponding goals are bundled into two interchangeable slots. There are no color-specific encoder parameters or slot-position embeddings.

Quaternions are unit-normalized and converted to rotation matrices. This removes quaternion sign ambiguity without asserting whole-scene rotational invariance. For each slot i and each causal state, the 36 features are:

- world offset p_i - p_tcp (3), TCP-frame offset R_tcp^T(p_i-p_tcp) (3);
- relative rotation R_tcp^T R_i flattened (9);
- world goal error g_i-p_i and its TCP-frame version (3 + 3);
- absolute object and goal positions using a common center/scale (3 + 3);
- world heights of object, TCP and goal (3);
- other-object-minus-object and other-goal-minus-goal vectors (3 + 3).

Ordered old and current features and current-minus-old features form 108 numbers per slot. A single 108-192-128 SiLU MLP with final LayerNorm is applied to both. Four-head, width-128 self-attention followed by a shared residual feedforward layer allows interaction between the two slots. There is no object-specific processing bypass.

The separate fixed-base stream uses each state's normalized qpos/qvel (18), normalized world TCP xyz (3), TCP rotation matrix (9) and absolute TCP height (1). Concatenating the two states and their difference gives 93 features, encoded by a 93-192-128 MLP. Thus elbow configuration, finger positions, joint velocities, gravity and the robot's base frame remain visible even when relative block geometry is similar.

The mean of the two object embeddings and the robot embedding generate three width-128 queries. Independent learned role embeddings are added to those queries. Four-head cross-attention pools the two object tokens for each query; these are continuous learned mixtures rather than a selected-object argmax or a discrete phase classifier. The three results, mean object embedding and robot embedding pass through a 640-384-256 fusion MLP. Concatenation with the explicit 128-dimensional robot stream gives a 384-dimensional global condition.

The public Conditional U-Net has widths 128/256/512, kernel 5, groups 8, timestep embedding 128, and FiLM conditioning. It receives the original noisy normalized 16-by-8 action sequence and predicts its epsilon. The eight actions retain their native meaning after framework decoding: seven absolute Panda joint targets and one gripper command, with increasing command toward open. Command -1 in these demonstrations does not imply measured negative finger positions: object contact leaves fingers around 0.0183 m.

### Normalization

The only fitted statistics are the supplied shared limits normalizer, fitted once to all 8,897 observations in all 12 COMPLETE original training demonstrations, not these skill slices. Its normalizer SHA256 is `874039c9e091b43eacce196d1283f9944073808a6049a059b2e07938248a7ce8`. The supplied `std` entries are half-ranges, not standard deviations. Robot qpos/qvel and TCP position use the exact supplied affine normalizer, without observation clipping.

For geometric feature engineering, use one isotropic length L = max of the nine supplied TCP/red/blue xyz half-ranges = 0.26858651638031006 m. Both slots, all displacements, positions and heights use this L. The common position center is the supplied TCP xyz mean. Heights retain the world zero and are divided by L without centering. Rotation matrices are dimensionless. This derives a common geometric unit from the frozen global fit; it is not a fitted per-skill or per-color scale. In particular, relative vectors are formed in raw physical coordinates BEFORE scaling and rotation. Both history samples and their differences use these same units. Differences are per observation, not claimed velocities in m/s.

This construction is algebraically invariant to swapping the complete two pose-and-goal bundles in both observations, up to floating-point roundoff, while leaving actions and robot state unchanged. It is NOT invariant to moving the physical goals, swapping an object's goal alone, translating the robot, reflecting joint actions or rotating gravity. Absolute world features remain intentionally available and may still enable coordinate memorization.

## Objective, gradients and inference

The sole objective is masked epsilon mean squared error:

    diffusion_loss = sum(mask * (epsilon_theta - sampled_noise)^2)
                     / max(8 * sum(mask), 1)
    prior_loss = 0
    loss = diffusion_loss

The mask excludes padded action targets, including those near slice boundaries. Noise is sampled and applied by the fixed DDPM pipeline. The prediction depends on the U-Net, fusion MLP, role-query generator and embeddings, both attention modules, shared slot encoder and robot encoder. All receive gradients from the denoising objective. The prior is architectural, so reporting a zero prior_loss is intentional. There is no penalty on observed poses advertised as a trainable constraint. Future observations, future masks and native future states are unused even in training; no auxiliary dynamics or contact supervision is claimed.

Sampling is fixed 100-step DDPM with clipping and shared action decoding; execute eight actions and replan with the next causal history. No guidance, rejection sampler, IK conversion or post-sampling controller is supplied. Final training is seed 0, 20,000 AdamW updates, batch 128, learning rate 1e-4, weight decay 1e-6, gradient norm cap 1, 500 warmup updates and cosine decay, EMA 0.999. Selection is the last EMA at that budget, not performance-based tuning. Only torch and appl.public are required by policy.py.

## Training evidence and full-slice handoff

The measured boundary statistics and original observations were inspected; these are successful demonstrator states, not learned-policy evaluations or hard entry filters. HANDOFF.json provides operational guidance to the inference API, including the missing-token limitation.

- **Initial red acquisition:** demo10300 index 0 has TCP z 0.44196 m, fingers 0.04 m each and both cubes z 0.020 m. Index 75 has open fingers while TCP z is 0.12135 m over red. At 120 command is -1 and measured fingers are about 0.01828/0.01824 m, but red z is only 0.02080 m. Finger closure alone therefore does not establish a useful lifted exit. By 160 red z is 0.27369 m. At 179-181 red and TCP translate toward the goal while remaining together: red x moves -0.39513 to -0.38688 m and TCP x -0.40464 to -0.39638 m; red z rises 0.28028 to 0.28123 m, with command -1 and nonzero qvel. The learned offset is roughly 0.0095 m along world x, not exact TCP centering.
- **Incoming red placement while blue is next:** demo10300 at 265-266 has red close to red_goal in xy, still held, descending from z 0.11212 to 0.10223 m; blue remains at its source. At 289-290 red is near z 0.036 m, command is +1 and fingers widen from about 0.03518 to 0.03707 m. At 319-320 red stays settled while the open TCP retreats upward from z 0.16554 to 0.17754 m. Index 400 is traversal over toward blue, not a new free-space reset. The representation retains both incoming and prospective object slots so it need not move incoming red laterally merely because blue is the next acquisition. The data, not a coded constraint, teaches this behavior.
- **Blue acquisition and late exit:** demo10300 index 445 has open-finger descent near blue at TCP z 0.11583 m. At 480 fingers are closed but blue remains at z 0.02005 m; at 500 it has started lifting; at 520 blue z is 0.21320 m. At 559-561 blue z is 0.28703-0.28898 m, moving goalward with TCP, measured fingers around 0.01829/0.01825 m, and command -1. TCP x is about 0.0082 m behind blue x. Red remains in its goal. No zero-velocity exit is demanded.
- **Support across starts:** demo10307 has initial red x -0.39110 m versus demo10305 -0.41087 m. Demo10307 72/100/110/130/177 supports approach, open-at-contact, closure, lift and early translation; 259/287/408 supports incoming red lowering, release and traversal; 439/477/505/530/551 supports blue acquisition. Demo10305 274/303 supports incoming red descent/release, and 457/496/577 supports blue approach/closure/goalward translation. These centimetre-scale source changes are not broad relocation evidence.

Two-state changes in TCP/object poses, measured qvel, finger width, goal error and absolute height make partial-transition entries distinguishable: held red descending over its goal differs from released red with an open retreating TCP; blue-on-table with an open approaching TCP differs from a closed TCP rising together with blue. No elapsed demonstration index is input. Motion and object separation remain observable when a single geometric alignment condition is already true. Padding at a slice start may eliminate the pose difference; measured qvel and geometry still provide information. No unobservable dwell clock is reconstructed, so nearly identical wait states may remain ambiguous.

For a useful late acquisition exit, preserve the selected payload and provide deliver_block with the SAME object identity/goal at the API level, full qpos/qvel and uninterrupted causal TCP/object history. Observed red endpoint centers span about z 0.280-0.283 m and blue 0.287-0.293 m; these are support, not required heights. Earlier transfer during aligned approach, closure or lift is also supported by shared training with delivery. Do not introduce a gripper opening simply because policy identity changes. Incoming red goal alignment is not itself the exit of blue acquisition: continue through lowering/release/retreat and blue acquisition unless another policy explicitly takes over that supported transition.

## Applicability, limitations and falsification

Use on the same fixed-base Panda/controller, accessible similar cubes, matching object/goal observations and the demonstrated ordering. Intended role must agree with scene context because the executable model lacks an independent role token. Both red and blue world/goal roles remain inferable but broad color-independent transfer has not been established. Do not use finger width alone as proof of grasp: co-motion and elevation are stronger observations, still not force/contact measurements. Larger offset errors, cube yaw changes, different fixtures, order reversal, missing poses, drops and recovery have no demonstrated support. There are no hard joint-limit, collision, containment or grasp-stability constraints.

This architectural prior can still overfit 12 similar trajectories or have attention queries collapse. A proposed evaluation, NOT performed here, is a matched-capacity raw-world diffusion baseline and a shared role encoder without relative features, using held-out whole trajectories and controlled source translations. Measure grasp/lift success, closed-without-lift failures, acquired-object drift, incoming red disturbance and joint-limit violations. Slot relabeling can separately check the claimed representation symmetry. Neither two-update interface validation nor checkpoint reload establishes task success.

The task completion contract remains exactly red_at_goal AND blue_at_goal with the provided full rotated containment and z tolerance. It does not require release, additional TCP clearance, zero velocity or a sustained hold. Our handoff cues guide policy invocation and transfer only; they do not add conditions to task success.
