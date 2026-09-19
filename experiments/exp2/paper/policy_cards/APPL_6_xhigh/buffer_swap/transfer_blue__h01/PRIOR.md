# transfer_blue__h01: attachment-conditioned learned diffusion

## Identity and original research hypothesis

Assigned skill: `transfer_blue`, heuristic 1, M1_v2. The original heuristic statement is:

> Rigid attachment should make object and TCP motion predictable together: learn an attachment-conditioned invariant transform and penalize inconsistent predicted transport. Hypothesis: object-centric auxiliary dynamics improve carry stability and grasp-offset robustness.

This document describes our implementation choices separately from that unchanged hypothesis. The assigned candidate proposes an attachment-aware diffusion policy, a learned state-transition head accounting for action-target/actual-state lag, and a gated rigid-transform auxiliary loss. We implement these parts. The proposed optional candidate-chunk scoring is **not** implemented. The expected improvement in carry stability is a hypothesis, not a measured result.

## Scope and measured handoff evidence

The policy learns the **entire expanded slice**, without filtering empty motion or assigning a time schedule to phases. The twelve unchanged half-open source ranges are:

| Demonstration | Range |
|---|---|
| demo10100 | [240,850) |
| demo10101 | [242,847) |
| demo10102 | [240,843) |
| demo10103 | [244,848) |
| demo10104 | [238,839) |
| demo10105 | [240,838) |
| demo10106 | [241,842) |
| demo10107 | [246,855) |
| demo10108 | [239,841) |
| demo10109 | [243,846) |
| demo10110 | [245,849) |
| demo10111 | [242,844) |

The incoming overlap with buffer_red is roughly 240--250 actions per demonstration, and includes red buffer descent/release, empty approach, and blue pickup. The outgoing overlap with finish_red is roughly 239--250 actions and includes blue descent/setdown/open/retreat and buffered-red pickup/lift. For demo10100 the exact overlaps are [240,490) and [600,850). We do not restrict this model to the middle blue carry.

Actual observations read during implementation establish:

* demo10100:240: red is still held, at z=0.162787 m, TCP z=0.162973 m, fingers approximately 0.01828 and 0.01824 m, commanded grip -1. This is not an empty-gripper entry. Across measured start boundaries red height ranges about 0.094--0.239 m and summed aperture about 0.03651--0.03652 m.
* demo10100:275: red is supported near [-0.1810,-0.0012,0.0200] m; TCP remains low, each finger is about 0.03960 m and grip is +1. Geometry and finger opening distinguish release from attached descent, without elapsed-time input.
* demo10100:455,480,490,540,600: blue is first closed on near its source and then rises, crosses the workspace, and descends. At 480,490,540,600 its world displacement from TCP remains approximately [0.00864,-0.00018,-0.00018] m. The code does not install that displacement as a calibration constant; it computes a full relative transform from current observations.
* demo10111:242--243 shows red descending together with TCP; 484 shows blue lifted with a slightly different approximately 8.72 mm world-x offset; 603--604 shows blue and TCP descending together. At demo10100:480, measured joint 2 is 0.3384 rad but its action target is 0.2844 rad. This is concrete reason not to identify commanded joint targets with realized poses.
* demo10100:630: blue is near goal with z=0.02192 m and grip still -1. At 640 it is at z=0.02000 m, grip +1, and each finger about 0.03822 m. At 690 TCP is at z=0.29108 m while blue remains supported. Near-goal XY alone would therefore mistake suspended carry for completed setdown, and stable table geometry alone does not identify whether release has occurred.
* demo10100:810 commands closing while fingers are still about 0.040 m; 820 has fingers near 0.01825 m and red still near z=0.0201 m; 849 has red at z=0.20283 m and TCP at z=0.20281 m. demo10111:812--813 is closed on red near table height, and 843 has red at z=0.17010 m. Measured final boundaries span roughly red z=0.170--0.213 m. Red has a new grasp transform, not blue's prior offset.

These are demonstrations and approximate contact cues, not evidence of this learned policy's execution success. Finger qvel can show small asymmetric simulated contact motion even at nearly constant aperture; the model sees qvel but labels use measured aperture and pose differences rather than assuming zero finger velocity proves attachment.

## Causal representation and learned architecture

Only raw observation history `[B,2,47]` reaches deployment forward. It contains qpos 0:9, qvel 9:18, world TCP pose 18:25, red pose 25:32, blue pose 32:39, zero drawer compatibility channels 39:41, and world goals 41:47. Positions are metres, arm joints are radians, and quaternions are wxyz. Both observed times, their normalized difference, and the two objects remain visible throughout all transitions. No trajectory index, phase clock, future state, force, or previous skill's inferred offset is required.

The supplied single full-original-demonstration mean/half-range normalizer is stored as model buffers. It is not refitted. Encoded actions are the original eight absolute commands under that same shared action affine map. Relative translations and goal-minus-object features use the maximum of the shared TCP xyz half-ranges (approximately 0.21313 m), equally for both objects and rotated axes. This is derived from the shared normalizer, not a new per-skill fit. Raw normalized history uses all original scales. Unit quaternion components keep unit bounds. Physical loss tolerances below are not replacements for the shared normalizer.

For each object and each causal time we compute `D_i = T_TCP^-1 T_i`, using quaternion multiplication and rotation. Quaternion features are unit-normalized and sign-canonicalized by their largest-magnitude component. This avoids unstable sign selection near the approximately 180-degree TCP orientation in these data, but is not a globally continuous orientation chart.

A 175-input, two-layer 256-wide SiLU MLP with final LayerNorm consumes 94 history values, 28 relative-pose values, 47 normalized differences, and six goal errors. Its heads predict:

1. A categorical probability for none/red/blue attachment.
2. For each object, a learned temporal blend and a bounded correction of its two measured relative transforms. Translation corrections are bounded componentwise to 5 mm and rotation-vector corrections to 0.05 rad. Quaternion blending aligns signs first. These are estimated offsets, **not** action displacements or grip rules. They initialize from observed geometry rather than assuming blue is already attached.

The global diffusion condition concatenates the learned 256-vector, the 94 normalized history values, the three attachment probabilities, and both seven-coordinate estimated transforms weighted by their own attachment probability: 367 dimensions. Original world and joint features remain available even in the none mode. The standard learned conditional U-Net from appl.public predicts epsilon on `[B,16,8]`, with the prescribed widths 128/256/512, kernel 5, groups 8, and timestep embedding 128. The fixed framework performs DDPM noising, sampling, clipping and action decoding. There is no scripted controller, IK, action replay, or hard attachment projection. A relative-frame conditioning feature does not make absolute joint actions equivariant.

## Learned action-conditioned dynamics and temporal alignment

A 256-state GRUCell is initialized from the same causal condition. At each predicted future slot it receives encoded action, native target-minus-predicted-qpos lag, normalized previous predicted qpos and poses, and a local horizon fraction. This fraction is merely an index within the 16-slot prediction horizon, not a demonstration phase or skill schedule. The lag includes seven joint targets and two known finger targets `0.025*g+0.015`; it is an input to a learned predictor, not an imposed physical update. A learned output MLP predicts qpos residuals, three position residuals, three rotation-vector residuals, and none/red/blue logits. Predictions are residuals relative to the last causal state; rotations compose with its measured orientations. Actual qpos and all three poses are supervised independently. Qvel is a causal input, not a predicted target. No analytical FK connects joints to TCP.

The supplied batch aligns actions t-1 through t+14 with states after each action. Slot 0 is therefore the already-observed current state at t, not a new integration step. The rollout copies this state, then predicts slots 1--15 from actions t--t+14. Dynamics supervision excludes copied slot 0. Both future_mask and action mask are applied. The rollout is recursive in predicted states rather than teacher-forced with future states. The frozen sampling algorithm calls only the causal encoder and diffusion network; the dynamics head is a training auxiliary, not an inference-time action optimizer.

## Training targets, losses, and gradient paths

All auxiliaries compare **trainable predictions** with labels or with other predictions. Geometry-only quantities are labels/weights and do not appear as standalone purported training penalties.

### Weak attachment labels

Training future poses yield relative transforms and adjacent relative-transform changes. Labels multiply soft proximity, measured finger closure, relative co-motion, and native close-command factors. Proximity uses a 35 mm midpoint and 6 mm sigmoid width; summed finger aperture uses a 55 mm midpoint and 4 mm width. Co-motion is a Gaussian of translation change scaled by 6 mm and shortest SO(3) change scaled by 0.12 rad. Opening command +1 makes the attachment score zero. Scores for red and blue are normalized if needed, with remaining probability assigned to none. These label constants encode approximate cube/gripper geometry and robust noise tolerances, not trajectory timing. No object is selected by a hard deployment controller.

The causal attachment logits receive slot-0 soft-label cross entropy; that label uses the current after-action state, preceding causal pose, and training action t-1. Rollout attachment logits receive future soft labels. Future co-motion and commanded opening thus provide training-only event supervision. A stationary closed gripper near a supported cube can have high inferred attachment even before lift. This uncertainty is unavoidable without contact labels and is explicitly a limitation.

### Loss definitions

Diffusion loss is the supplied masked epsilon MSE and has coefficient 1.

`L_state` is masked Smooth-L1 on predicted qpos errors divided by the shared qpos scales, plus predicted xyz errors divided by the common shared TCP position scale, plus shortest quaternion-log errors divided by 0.15 rad. Each group is averaged over its coordinates. Demonstration-action and reconstructed-action rollouts are independently supervised against synchronized future states.

`L_offset` fits each causal learned transform to future observed relative transforms, only over a continuously attached prefix. Prefix gates require attachment score above 0.6 and valid labels at every preceding slot; they stop at release and do not restart on a different grasp. This teaches smoothing and small corrections rather than a universal 8--9 mm offset.

`L_rigid` uses the sum of translation Smooth-L1 with 0.02 m scaling and shortest SO(3)-log Smooth-L1 with 0.15 rad scaling. It has two terms: adjacent predicted relative transforms are compared during attached pairs, and predicted transforms are compared with the learned causal transform during the continuously attached prefix. Pair gates are products of adjacent soft training attachment scores and masks, so newly acquired objects inside the horizon can receive pairwise supervision even when no valid causal anchor exists. Release zeros the event-gated scores. The mask cannot be turned off by lowering the model's predicted attachment probability. Both pose prediction and causal transform estimation receive gradients.

The implemented metric separates translation and rotation; it is **not** the exact coupled SE(3) logarithm from the research sketch. Full quaternion rotations and TCP-inverse translation are nevertheless computed, so this is more than a constant world-offset penalty. A robust product metric and supervised transition gating are the adaptation to compliant grasps and near-constant demonstrated orientations.

The reconstructed clean action is the fixed DDPM epsilon inversion, clamped to [-1.25,1.25] only before the training dynamics head. Reconstructed-path errors are weighted by alpha_bar squared within masked averages to emphasize reliable low-noise samples. This prevents unbounded high-noise action estimates from dominating the auxiliary. It does not change epsilon training or the fixed deployment sampler.

The exact total is:

```
L = L_epsilon
  + 0.080 L_state(demonstrated actions)
  + 0.040 L_state(reconstructed actions)
  + 0.030 L_attachment(causal)
  + 0.015 L_attachment(demonstration rollout)
  + 0.005 L_attachment(reconstructed rollout)
  + 0.015 L_offset
  + 0.010 L_rigid(demonstration rollout)
  + 0.010 L_rigid(reconstructed rollout).
```

Weighted reductions divide by the sum of valid weights with a denominator floor of one; no padded labels train the predictor. The demonstrated-action losses train F and the shared encoder. Reconstructed-action losses also backpropagate through F's action inputs, x0, and epsilon to the denoiser; there is no detach on that path. Offset and causal attachment losses train heads whose outputs condition diffusion, and the primary denoising loss also trains those heads. Trivial all-off attachment gating cannot eliminate the supervised rigidity loss. The state targets prevent a static co-motion solution from satisfying moving demonstrations without error. These are soft biases, not guarantees that F remains accurate on sampled actions.

## Handoff, applicability, and continuation

HANDOFF.json supplies semantic guidance to the selecting inference API. The model itself has no scheduled phase switch or termination controller. Its representation can distinguish red-held descent, open table release, blue pickup, attached blue carry, supported blue with retreating TCP, and red regrasp using object identity/location, goal error, aperture, arm velocity, and causal relative-pose motion. It estimates two offsets anew on each invocation, so it cannot accidentally inherit blue's stored transform as the new red grasp. Ambiguous stationary grasp/release moments remain learned from the entire imitation distribution, not solved by the auxiliary alone.

Supported takeover can occur while red is descending at the buffer, not just once blue is held. Later takeover during either shared transition is also represented in training. Continue if local completion requires finishing red release, blue closure/lift/carry, blue lowering/opening/retreat, or red pickup/lift. A useful late exit is blue supported in its goal and red lifting at the buffer with fingers near 0.01825 m each and stable relative transform. finish_red should receive actual two-step qpos/qvel, TCP/red/blue poses and goals, not predicted F states or inferred attachment as fact. An earlier handoff in the outgoing overlap can ask finish_red to finish blue release itself.

The global task is red_at_goal AND blue_at_goal under the supplied full-rotated-XY containment and z-tolerance contract. Blue goal is [-0.35,-0.20,0.02] m, red goal [-0.35,+0.20,0.02] m. Half-region XY size is 0.06 m, cube half-size 0.02 m, and z tolerance 0.011 m. No additional release, clearance, velocity, or sustained-hold requirement is added. Suggested open/clear/lift states are useful **skill handoffs**, not changes to task success. This skill normally exits before red reaches its final goal.

## Dependencies, budget, and limitations

Only torch and appl.public are imported. All modules and normalization buffers participate in state_dict/EMA checkpoints. Training uses the unchanged seed-0 budget of 20,000 AdamW updates, batch 128, learning rate 1e-4, weight decay 1e-6, cosine decay with 500 warmup, gradient norm limit 1, EMA 0.999, horizon 16, history 2, execution 8, and fixed 100-step DDPM. Selection is the last EMA at that declared budget. There is no data access, per-skill normalization fitting, external pretrained model, image encoder, candidate search, or runtime file I/O in policy.py.

The approximation is designed for rigid cubes, accurate state observations and the same parallel-jaw gripper. Soft fingertips and PD lag permit deviations from rigidity. Two observations provide short smoothing, not a persistent grasp filter. A learned dynamics head is not robot kinematics and can extrapolate incorrectly; the denoiser may exploit its errors. Commanded opening may precede measured release, so conservative loss gating can omit a few still-attached steps. Stationary close contact and a true grasp cannot always be separated. Quaternion feature canonicalization has chart boundaries, and broad rotational generalization is untested. There is no guaranteed recovery for slip, drops, larger offsets, nonrigid objects, sensor failures, or out-of-support handoffs. Suggested failure cues are relative-transform drift, an object failing to rise with TCP, opening during unsupported carry, or blue suspended when assumed supported; they are for external interpretation, not implemented correction rules.

A clean scientific ablation would remove the attachment/offset/rigidity auxiliary terms while retaining identical architecture and primary denoising training, and compare transport error, opening during carry, offset residuals, and task success. No such experiment or final-budget performance feedback is available in this implementation session. The package interface check is only a short training/sampling/checkpoint integrity test, not a task-performance evaluation.
