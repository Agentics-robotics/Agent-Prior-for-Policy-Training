# Payload-aware geometric latent diffusion policy

Policy: `deliver_block__h01`  
Skill: `deliver_block`  
Assigned heuristic: 1, unchanged  
Experiment: M1_v2

## Original hypothesis and implementation status

The assigned statement is: "Move the payload, not just the TCP: diffuse goal-relative Cartesian motion and explicitly account for the current grasp transform when learning placement, then decode through local inverse dynamics to absolute joint targets. This should decouple placement precision from modest grasp-offset and joint-route variations."

The original heuristic is not edited by this package. The following are implementation choices and limitations, not replacements for its research claims.

**Necessary interface adaptation.** The frozen runtime noises, samples, clips and decodes eight normalized original joint/gripper actions. `forward` receives a noisy joint sequence, a diffusion timestep and exactly two causal observations; it cannot replace the sampler with a ten-dimensional Cartesian-path sampler. This package therefore implements a **joint-action DDPM with a learned Cartesian-path latent and a learned inverse-dynamics conditioning route**. At every diffusion step a temporal predictor maps the noisy action sequence and causal state to a clean TCP path. A learned local decoder maps that path, grasp geometry and current robot configuration to a joint proposal. The epsilon U-Net receives the entire path and proposal as conditioning. The path changes with the stochastic action sample during denoising; it is not an independently noised Cartesian DDPM variable. Final actions come from the fixed DDPM, not by bypassing it with the proposal. A differentiable action-to-path cycle also trains the actual denoised action estimate. This retains the payload-offset and local inverse-dynamics mechanism but is not the original proposal's exact path-only factorization.

Two other interface limitations matter. There is no external payload/pending identity channel and no past-command channel. Internal causal proximity roles, both objects' completion cues, qpos/qvel and two-state differences substitute for them. The policy cannot honor an arbitrary color instruction when state does not disambiguate it. The fixed executor uses eight actions per invocation, not the suggested one or two near placement. Neither adaptation is hidden by metadata overrides.

## Data, full scope and inspected evidence

The package preserves all 24 assigned expanded half-open slices and the supplied masks. It has no phase filtering, color-specific training subset, index-based controller, skill-local normalizer or replay. Early descent and grasp formation, high carry, lowering, opening, settling, empty retreat, the red-to-blue approach, and final blue hold all receive diffusion and path/action supervision. Only the rigid-payload term is conditional on attachment evidence. The current anchor remains fixed over each predicted horizon; future goal/object states never select the deployment role.

Inspected original observations/actions include:

- `demo10300`: 75, 120, 160, 181, 200, 265, 280, 290, 320, 414, 445, 480, 520, 561, 640, 655, 665, 720, 741.
- `demo10307`: 72, 109, 110, 129, 130, 177, 178, 258, 259, 286, 287, 407, 439, 476, 477, 504, 505, 631, 632, 645, 646, 658, 659, 700.

In demo10300 at 75 the red block is at table height 0.020 m, TCP z is 0.12135 m, both fingers are about 0.04 m, and the command is +1. At 120 the fingers are near 0.01826 m, the command is -1, and TCP and block are near z=0.0208 m: closure alone must not be mistaken for proven attachment. By 160 they have risen together to about 0.2736 m. At 280 the red center is [-0.136192, -0.069899, 0.045736] and TCP is [-0.145695, -0.070178, 0.045524] m, a nonzero approximately 9.5 mm world-x offset. At 655 blue has an approximately 8.2 mm world-x offset. The difference is not removed by commanding the TCP to the goal marker.

At 290 red has settled to z=0.0360 m, fingers have opened to about 0.0371 m and the command is +1. At 320 TCP rises without red; at 414 TCP is near blue with z=0.26767 m and open fingers. The blue slice starts again during open descent at 445, not at a clean carry boundary. At 665 blue has settled to 0.0360 m and fingers are about 0.03822 m. At 720/741 TCP is near 0.3165 m, both blocks are in the tray and the hand is open.

Consecutive observations in demo10307 distinguish actual co-motion from proximity: at 129/130 red and TCP both rise by about 7.12 mm with nearly unchanged offset, and at 504/505 blue and TCP both rise by about 10.34 mm. Conversely at 109/110 and 476/477 the closed hand and table object are nearly stationary. The transform is conservatively invalid there until lift-compatible evidence is available. At 645/646 blue is still grasped near 0.0455 m; at 658/659 it is at 0.0360 m and fingers approach 0.04 m.

The supplied measured boundary evidence spans entry TCP heights 0.0877--0.1779 m with open fingers and the selected source block at 0.020 m. End measurements mix red-to-blue empty approach (TCP near 0.267 m) and final empty retreat (near 0.3165 m). Shared overlaps include demo10300 [75,182), [265,415), [445,562) and demo10307 [72,178), [259,408), [439,552), with analogous expanded overlaps across all training demonstrations. These are observed support, not applicability guarantees or prescribed switch times.

## Causal representation and attachment estimate

Observations are world metres and wxyz unit quaternions. The model consumes qpos 0:9, qvel 9:18, TCP pose 18:25, red pose 25:32, blue pose 32:39, compatibility channels 39:41 and both goals 41:47. There are no images. Drawer features stay zero for this task. The two observations are used as given; the implementation has no hidden state or clock.

Quaternions become rotation matrices, then the first two matrix columns. This makes the encoded physical orientation invariant to quaternion sign, not invariant to arbitrary world rotations or translations. The world/base configuration, world height, both absolute object positions, goals and robot state remain inputs. Path rotations are also represented by two columns and reconstructed with Gram-Schmidt. No Euler discontinuity or learned quaternion sign convention is needed.

For each object the model computes the full observed relative transform C = T_tcp^-1 T_object in both causal observations. It also retains the change in translation and rotation. The translation is **not assumed attached** merely because it can be computed. The attachment confidence is zero unless both observations have maximum per-finger position below 0.027 m, object/TCP distance below 0.030 m, relative translation change below 0.003 m, rotation-matrix Frobenius change below 0.12, and some nonduplicate history. It additionally requires both object heights above 0.026 m, or object and TCP displacements above 0.0004 m with displacement mismatch below 0.002 m. The nonzero confidence is exp(-||delta C_translation/0.002||^2 - (delta C_rotation/0.12)^2). These hand-chosen physical evidence tolerances are not fitted feature normalizers or contact labels. An elevated, closed, close and stable relation is inferred grasp evidence, not a grasp guarantee; stationary closed-at-table observations remain invalid. Open descent and released retreat get zero attachment confidence.

Internal red/blue role weights are a softmax of negative TCP-object distance: -16 times shared-scale XY squared distance plus one-quarter shared-scale Z squared distance. The resulting weighted goal is a coordinate anchor, not an action command. Both role weights, both unmasked observed relative transforms and their explicit validity values enter the encoder. Invalid relative geometry is still useful for approach, but carried-object decoder features and rigid losses are masked. Causal rotated XY extent and height-margin sigmoid cues describe whether each object appears contained; pending weights come from the objects not yet apparently complete. They neither terminate inference nor prescribe a color schedule. They are not a calibrated success probability. As the empty hand approaches blue, role weights move continuously toward blue without resetting world/proprioceptive inputs.

The 159-dimensional context comprises two 53-dimensional sign-invariant observation encodings, both TCP-goal displacements, both object-goal errors, both C translations and rotations, both C translation changes, validity and proximity roles, TCP and object displacements, and completion/pending cues. A 256-to-128 MLP with LayerNorm produces learned context.

## Architecture and inverse dynamics

A width-128 temporal network with four residual FiLM blocks (dilations 1,2,4,1), sinusoidal diffusion time features and learned horizon-position embeddings predicts H=16 path entries. Each contains:

1. TCP XYZ relative to the causal goal anchor, divided by the shared TCP XYZ half-ranges;
2. six continuous world rotation components, orthonormalized to a rotation matrix;
3. the original gripper command, bounded by tanh.

The path predictor uses the current TCP path as a residual reference, not as an action controller. Current rotation plus a learned residual is orthonormalized. It is trained on realized post-action future TCP observations and source gripper commands. At deployment no future observations enter this computation.

A width-128 three-block temporal decoder receives the path, its reconstructed world XYZ and current-TCP displacement, and both validity-masked payload-goal errors computed from T_tcp_predicted C_current. FiLM provides the complete learned causal context, including current qpos/qvel and history. Its seven outputs are learned residuals around atanh(shared-normalized current qpos clipped to +/-0.98), then tanh-bounded to original normalized arm-target ranges. The gripper component is the path gripper prediction. This is a supervised, local learned inverse-dynamics approximation on the demonstrated redundancy branch. It does not know Panda link geometry, solve IK, enforce joint-rate constraints or prove realizability of the predicted path.

The 128-dimensional context, flattened 160-dimensional path and 128-dimensional decoded proposal enter a 384-to-256 condition MLP. The public Conditional U-Net then predicts epsilon for the noisy eight-dimensional joint action sequence. Its widths, timestep embedding and other fixed recipe settings are unchanged. The proposal has no direct control authority. The score network can use noisy actions and context as well as the geometric route; perfect dependence on the path bottleneck is not guaranteed.

## Shared normalization and training-only labels

All native observations and original actions use the one supplied normalizer fitted on 8,897 samples from the 12 COMPLETE original demonstrations. No statistics are fitted here. Normalizer identity: `874039c9e091b43eacce196d1283f9944073808a6049a059b2e07938248a7ce8`. In this limits normalizer, `std` is half-range. Derived translations, including goal-relative paths, C translations and displacement features, use the existing TCP XYZ half-ranges [approximately 0.14115, 0.26859, 0.21308] m. Rotations have unit geometric scale. Thresholds used for attachment/containment cues are physical task hypotheses, not tiny per-skill normalizers.

Action slots are t-1 through t+14, with future state j observed after action j. The targets use exactly that alignment, including the past-aligned first slot. Future masks and action masks are multiplied for future-derived losses. Invalid path slots are replaced with the current path before the teacher-path decoder; losses stay masked. Adjacent-increment losses require both adjacent slots valid. No target crosses a skill slice. Future positions, rotations and finger states are labels or label masks only, never deployment conditioning.

## Actual objective and gradient paths

Let a denote the original normalized action, y its noised version, P the path predictor, D the local decoder, and E the epsilon model. The main loss is the fixed masked epsilon MSE. All auxiliary MSEs average over valid elements, with denominator at least one; reliability factors multiply the numerator, rather than being renormalized away.

Define path error as mean XYZ squared error in shared TCP scales + 0.1 mean rotation-six squared error + 0.25 gripper squared error. Define payload error as mean payload XYZ squared error in shared TCP scales + 0.1 mean rotation-matrix squared error. Payload XYZ comparisons subtract each object's own current goal from both prediction and future observation. Thus they train the demonstrated approach to the goal, not an indiscriminate zero-goal attraction during lift/carry.

The reported `prior_loss` is:

- **1.0 noisy-path loss:** P(y,t,history) versus the realized future TCP/gripper path, weighted by 0.25+0.75 alpha_bar.
- **0.5 clean-path loss:** P(a,0,history) versus that same target. This calibrates a shared local action-to-path map.
- **0.5 teacher inverse reconstruction:** D(target_path,history) versus the original seven normalized joint targets.
- **0.5 predicted-path reconstruction:** D(P(y,t,history),history) versus all eight original action components, with the noisy-path reliability factor.
- **0.1 increment reconstruction:** proposal and teacher-decoded seven-joint increments versus the demonstrated action increments. The proposal part has the reliability factor. This is not a hard rate bound or a zero-motion penalty.
- **2.0 attached-payload consistency:** predicted TCP pose composed with current C versus future object pose, weighted by the reliability factor. Current attachment confidence must be nonzero; future label fingers must remain below 0.027 m, object/TCP distance below 0.030 m, and future relative transform must remain within 0.004 m and 0.15 rotation-matrix Frobenius distance of current C. Release and changing attachment invalidate this supervision.
- **0.2 denoised-action path cycle:** x0 = (y - sqrt(1-alpha_bar) E)/sqrt(alpha_bar), auxiliary-clipped to [-1.25,1.25], then P(x0,0,history) versus the future path, weighted by alpha_bar squared.
- **1.0 denoised-action payload cycle:** the same cycled TCP path composed with current C versus future object poses, using the same attachment label mask and alpha_bar-squared weight.

Total loss is epsilon loss plus prior loss. Path losses train the path network/context encoder. Reconstruction and increment losses train the decoder and, for predicted paths, the path network. Attached losses differentiate through predicted translations, predicted rotations and their composition with C; they are not penalties computed solely on observed poses. Cycle losses additionally backpropagate through x0 into the actual epsilon network, as well as through the shared path model. No detach blocks these paths. High-noise amplification is suppressed by the cycle weight and auxiliary clipping. Confidence, role and target masks are causal features or training labels, not trainable classifiers; the losses they gate still depend on trainable predictions. No all-observation constant penalty is called a training loss.

## Handoff, continuation and success

See HANDOFF.json for the API-facing conditions. At an open descent entry, aligned source geometry, open fingers, TCP height, qvel and two-state motion identify the unfinished acquisition. The rigid transform is invalid, but the unconditional path and action losses train descent and closure. At a later lift/carry entry, stable closed-hand co-motion and current elevated object position allow the transform-aware route to take over without knowing a trajectory index. At lowering/release overlap, height, actual finger aperture and object/TCP separation discriminate attached placement from empty-hand retreat. At red-to-blue approach both objects remain visible and pending/role cues describe blue while preserving red's measured state.

Containment is not manipulation readiness. The completion contract permits success near object z=0.0455 m while still closed if both objects satisfy full rotated XY containment and height tolerance. It requires neither gripper release, TCP clearance, velocity threshold nor sustained hold. This package never adds those requirements to formal task success. If continued manipulation is desired, useful demonstrated handoff is an object settled near 0.036 m, fingers approaching 0.04 m each, and causal separation rather than continued attachment during retreat. A successor can also take over earlier within the lowering/opening overlap if it is given the actual two-state history and can finish release. Do not reset to an invented empty-hand state.

The API may continue this skill through those shared transitions or transfer to acquisition according to observations. It should not use the cited source indices as a schedule. If formal task completion is already true, no additional retreat is necessary to satisfy the supplied task contract. The trained final retreat/hold remains available for a useful exit when continued execution is chosen.

## Dependencies, limitations and falsifiability

Only Torch, math and the provided public numerical module are used. Training remains seed 0, 20,000 updates, batch 128, H=16, two observations, execution 8, 100-step epsilon DDPM, AdamW 1e-4 with 1e-6 decay, gradient clipping 1, cosine/500 warmup, EMA 0.999 and last EMA selection. No pretrained external model, simulator kinematics, image encoder, scripted controller, action replay, sample ranking or phase-hazard controller is present.

Supported evidence is local to these top-down Panda motions, fixed tray geometry and controller. Rotation variation is very small. Role inference can be ambiguous after an unusual partial transition, both objects close to the TCP, an unsupported requested order or a dropped object. Two states cannot prove a grasp; elevated proximity can be misleading. A learned action-to-path map can rely on context or have model error, and the score network can partially bypass the geometric proposal. Tanh bounds and sampled action clipping restrict numeric ranges but do not enforce collision avoidance, physical feasibility, containment, continuity or correct null-space branch. No calibrated uncertainty output is added to the eight-dimensional forward interface. Internal validity/proximity weights are cues, not proof; the API should treat unexpected offsets, wrong-block motion, post-release co-motion or discontinuous joint behavior as uncertainty and seek a supported policy rather than assume correction.

The short interface test establishes only package validity, two real gradient updates, fixed DDPM sampling and EMA reload compatibility. It does not measure task success, transfer or placement improvement. Suggested future ablations are absolute-joint DP, goal-relative latent without C/validity, and this complete candidate under supported small grasp offsets. Improved payload placement and reduced off-branch/discontinuous decoding are testable hypotheses, not reported results. Independent Cartesian sampling, adaptive short execution and arbitrary rotational/robot transfer remain unimplemented.
