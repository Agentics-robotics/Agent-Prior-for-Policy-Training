# Carry, place and clear: grasp-transform-conditioned action diffusion

Policy: `carry_place_clear__h01`  
Skill: `carry_place_clear`  
Assigned heuristic: 1, M1_v2

## Original hypothesis (unchanged)

> Transport the object, not the TCP: represent goal error through a causally estimated rigid grasp transform, with attachment-aware masking. This should share goal-directed carrying across red/blue and retain correct placement despite small TCP-to-object offsets.

This package implements that spatial/kinematic representation hypothesis, not a new task-region objective or a scripted event controller. The original heuristic and segmentation remain unchanged. The following are implementation choices and limitations, not modifications to the source heuristic.

## Observed basis and scope

The model trains on **all 24 assigned expanded slices**, with no phase filtering, boundary trimming, fixed time schedule or success-based truncation. In demo10200 these are [80,430) and [440,732); the corresponding ranges for the other eleven demonstrations remain as supplied. Every valid action slot, including acquisition and successor-transition overlap, contributes to the standard diffusion objective.

I inspected actual states/actions in demo10200, demo10204 and demo10207. Examples:

* demo10200 at 80: open fingers approximately 0.040 m each, TCP z 0.0772 m, red z 0.0600 m. By 100 the fingers are near 0.01825 m, with a close command and red still near its starting height; by 150 red and TCP have lifted to 0.3082/0.3096 m. Proximity at 80 does not establish attachment.
* demo10200 at 200/250: red x -0.24624/-0.24099 m versus TCP x -0.25530/-0.25005 m. At 550/600 blue x -0.28927/-0.23899 m versus TCP x -0.29809/-0.24781 m. An approximately 9 mm offset appears for both destinations. We do not substitute a fixed 9 mm offset in the policy: the actual causal relative transform is computed for both objects.
* demo10204 at 260/270 and 625/632: objects can be within pad-height tolerance while fingers remain pinched and the action still commands closing. At 642 the blue object is at z 0.0200 m, fingers near 0.03976 m, and the command is open. By 670 TCP has risen to z 0.1805 m while blue stays at the pad. A transform measured after release is not a new grasp.
* demo10200 at 390/420/429: red stays at its pad, the gripper is open, and the TCP returns toward blue, descending from z 0.2680 through 0.1348 to 0.0612 m. demo10204 at 417 ends this slice earlier in that approach, z 0.2052 m with nonzero velocity; demo10207 at 433 is z 0.1447 m. These are continuation states, not stationary endpoints.
* demo10200 at 650/700/731, demo10204 at 642/690/720 and demo10207 at 662/738 include final blue release, open retreat and settling. Final TCP height is about 0.300 m, with both objects left on their pads.

The measured cross-demonstration entry bands have red TCP z about 0.064-0.101 m and blue z about 0.021-0.089 m, generally open fingers. The measured red exit approach band is z about 0.055-0.222 m. These are observed training support, not applicability guarantees or runtime acceptance thresholds. Long red-carry/blue-acquisition overlaps include held descent, release, retreat and next approach, not just a single boundary frame.

## Necessary interface adaptations

1. **Invocation tags are absent.** The hypothesis assumes selected/next tags that persist through a call, but `forward(noisy_action, timestep, raw_history)` supplies only two 47D causal observations. The batch also supplies no segment role labels. This implementation therefore retains BOTH red and blue tokens at every step, in fixed order with explicit identity features. Red has a persistent red-to-blue next-object displacement and a next-present bit; blue has next-present zero. Both goal channels remain active even after placement. There is no hard selected-object switch at the first goal hit, no inferred elapsed demonstration index, and no mutable latch whose training and sampling semantics would differ. Learned fusion uses configuration, finger state, object positions/goals and causal motion to determine which relationship matters. This supports the demonstrated red-then-blue scene evolution but cannot obey an arbitrary externally assigned target tag or guarantee invocation-level identity under ambiguous/out-of-order scenes.
2. **Only two causal states are available.** Attachment is a conservative estimate of *observed moving coupling*, not force closure or long-memory grasp tracking. Open/near states do not unlock the grasp branch. A stationary held object may lose this branch; the regular approach geometry, goal errors and robot configuration remain available so the learned diffusion policy can still produce holding/release/lift actions. With only two indistinguishable stationary observations, remembering a previous grasp is impossible here.
3. **Joint actions, not Cartesian commands.** The denoiser predicts normalized original eight-dimensional absolute joint/gripper actions. There is no IK/FK provider. An auxiliary learned action-to-future-pose network is the executable substitute for an analytic joint-to-object transport loss. It trains the representation and denoiser but does not impose exact kinematic feasibility or act as an inference controller.
4. **Fixed execution cadence.** The host uses the assigned horizon 16, history 2, DDPM 100 steps and execution chunk 8. No custom 20 Hz replan loop is implemented; observation/control timing and chunk execution remain host-owned. The seven Panda absolute joint targets and increasing-to-open gripper semantics are unchanged.

## Causal representation and learned modules

`policy.py` constructs a learned epsilon diffusion policy using the public Conditional U-Net (widths 128/256/512, kernel 5, groups 8, time embedding 128) with a 256D FiLM condition.

All quaternion inputs (wxyz) are normalized and converted to rotation matrices. Thus the encoder and pose labels are invariant to quaternion **sign**, not to a change of world frame. For each object i and each of the two observations:

* `r_i = R_tcp^T (p_i - p_tcp)`;
* `R_gi = R_tcp^T R_i`;
* the causal grasp transform is `X_gi = (r_i, R_gi)`;
* both world and TCP-frame object-to-goal translation are represented.

Attachment logits use a shared 35-input, 64-hidden MLP: normalized finger positions/velocities from both observations, local offsets, relative-rotation change, TCP/object displacements, offset change and motion magnitudes. Learned sigmoid confidence is multiplied by a conservative causal eligibility mask: total finger width below 0.058 m previously and 0.055 m currently; TCP-object distance below 0.045 m in both frames; both TCP and object displacement greater than 0.00002 m. These are representation-support checks only, not action rules or API rejection thresholds. The gate is recomputed from the history on every call. Its supervised soft target is eligibility times exp(-squared local-offset change / 0.0015^2 - squared relative-rotation matrix change / 0.03^2). These physical bandwidths were chosen for the weak attachment proxy, not fitted as observation normalizers. They are not measured contact tolerances. Quiet contact, nearly stationary held objects, sliding objects and short slips are ambiguous.

Each shared object encoder receives 94 features: 60 from the two per-frame 30D scene tokens; 24 from the confidence-masked two-frame grasp transform; confidence; and nine motion coordinates. Scene tokens include absolute object position, its rotation matrix, absolute goal, world/local goal error, ordinary TCP-relative approach displacement, next-object displacement, persistent identity and next-presence. The ordinary approach displacement is deliberately available before attachment; it is not treated as a rigid carry transform. Sharing object-encoder weights encourages reuse between the two colors, while identity and absolute geometry retain task asymmetry. Both tokens are concatenated, never discarded or reassigned at success.

A separate 72D robot-history/motion input contains globally normalized qpos/qvel, absolute TCP position and TCP rotation matrices and changes. Its MLP produces 128D, as does each shared object encoder. The three embeddings feed 384-to-256 fusion with LayerNorm. The absolute branch is necessary for Panda reachability/configuration dependence; this is not an SE(3)-equivariant joint-action model.

The host-supplied full-demonstration normalizer is used unchanged for qpos/qvel/TCP state and encoded actions. Geometry is first computed in raw world metres and rotation matrices. World vector scaling is the axiswise maximum of the supplied TCP/red/blue positional half-ranges; local-frame vectors use the maximum of that scale vector. The TCP positional mean is the common origin for absolute geometric tokens. No statistics are refitted on slices, colors, phases or batches. Drawer channels are unused fixed-zero compatibility fields.

## Losses and actual gradient paths

Let `a` be alpha_bar. The clean normalized action estimate is

`x0 = (noisy_action - sqrt(1-a) * predicted_epsilon) / sqrt(a)`.

Only the auxiliary network sees this estimate clipped to [-2,2]. The DDPM forward output and sampler are unchanged.

1. **Diffusion:** the public masked epsilon MSE on all eight action coordinates and every valid slot. This is the primary learned action objective over the entire expanded slice.
2. **Attachment:** binary cross entropy between learned attachment logits and the detached soft two-frame evidence label above. The label is geometric, but the loss is NOT a penalty on observations alone: its trainable logits receive gradients. Diffusion and auxiliary pose losses also backpropagate through the confidence-gated encoding wherever eligibility is nonzero. The hard support mask itself is not trainable.
3. **Learned future dynamics:** a 512-hidden MLP receives the 256D condition and the entire encoded action chunk. For each slot it predicts normalized TCP displacement, a TCP relative rotation vector and independent displacements for both objects. The predicted TCP rotation is `R_tcp,current * Exp(rotation_vector)`. Its free-object outputs learn both transported motion and post-release/completed-object stability. This training-only predictor is evaluated once on demonstration actions and once on reconstructed x0.
4. **Rigid held-object supervision:** using the causal current `X_gi`, the auxiliary prediction is `p_i,pred = p_tcp,pred + R_tcp,pred r_i`; its predicted rotation is `R_tcp,pred R_gi`. These predictions are compared to demonstration future object poses. A detached mask requires causal moving-coupling support at the start, valid future/action slots, uninterrupted future finger closure, future proximity and current-to-future offset consistency (Gaussian bandwidth 0.004 m). Prefix closure drops every slot at and after opening. The learned gate cannot turn off this loss to avoid supervision because masks use observed evidence labels, not its predictions. No future transform enters the inference encoder.

Each dynamics loss is TCP positional Huber plus 0.1 TCP rotation-matrix MSE, plus 0.5 free-object positional Huber, plus rigid-object positional Huber and 0.1 rigid-object rotation-matrix MSE on the held mask. Huber beta is 0.05 in the same globally scaled position coordinates; losses average vector components and valid masks. Action slots have the required t-1 through t+14 alignment; each future label is the state after its corresponding action, and all targets use both future_mask and action mask. Label arrays are never shifted into causal history. Masks respect slice padding; there is no cross-slice target construction.

Total loss is:

`L = L_epsilon + 0.02 L_attachment + 0.10 L_dynamics(demo actions) + 0.05 L_dynamics(x0)`.

The reconstructed-action dynamics errors have a per-example `a^2` multiplier, without renormalizing that multiplier away, to attenuate high-noise amplification. Their gradients flow through the learned pose predictor into x0, epsilon and the denoiser, and through the shared condition into the object/robot encoders and eligible attachment confidence. Demonstration-action dynamics also trains the pose predictor and encoders on the real action-state relation. This is a learned consistency regularizer, not a guarantee that the predictor must use every action input or that physical dynamics are exact. No auxiliary is solely an observed-pose constant.

## Deployment and handoff

Only causal qpos, qvel, TCP pose, both object poses and both goal coordinates enter `forward`. No future observations, demonstration action chunks, source indices, training labels or auxiliary dynamics outputs are inference inputs. Forward always predicts epsilon; the host samples and decodes actions using the original normalizer and fixed DDPM. There is no external low-level controller or replay.

The same learned model covers an open aligned pickup, closure and full lift before carrying, already-held mid-transport entry, held pad descent, release, empty retreat and either the red-to-blue approach or final blue settling. Fingers and causal relative motion distinguish open approach from moving attachment; object-to-pad error distinguishes lift/transport/descent from empty retreat; both object goals and persistent red-to-blue relationships retain completed-red stability during the next approach. Absolute TCP height/position, qvel and causal pose change distinguish ascent, descent, lateral return and settling. These are observation cues the model can learn from, not prescribed actions or a proof that two frames resolve every ambiguity.

`HANDOFF.json` gives the full semantic interface and observation evidence. Continue this policy when the selected manipulation is still closing, lifting, lowering, opening or clearing, unless transferring inside a supported overlap. For red, a useful late transfer has red stationary at its pad, fingers opening/open and TCP returning toward exposed blue; nonzero velocity is supported. An earlier transfer is possible in the long overlap if the successor handles red still held/descending. For blue there is no trained successor; the final open/clear/settled states are retained for usability. The host should check both pad predicates independently and not infer task completion from TCP proximity alone.

## Completion contract, limitations and dependencies

The task contract is **red_at_goal AND blue_at_goal**, with full rotated XY containment for half-size 0.020 m blocks in half-XY 0.060 m pads and center-z tolerance 0.011 m. Goals are red (-0.24,-0.25,0.02) and blue (-0.24,+0.25,0.02) metres. One observation suffices. Release, TCP clearance, speed limits and a sustained hold are NOT additional success requirements. Handoff opening/retreat guidance must not redefine success.

Expected warning signs are shrinking TCP-goal error without corresponding object-goal progress; an object following open retreat; closed fingers during return toward blue; and completed red moving off its pad during blue handling. The confidence gate is not a recovery mechanism. Small demonstrated offsets and nearly fixed orientations do not validate skew grasps, substantial rotation, heavy objects, arbitrary source/goal perturbations, slip recovery or arbitrary scene symmetries. There are no images, force sensors, contact ground truth, analytic FK/IK or hard feasibility/containment constraints. Exact selected-role persistence is not implementable with the given stateless forward signature; retaining both tokens is the explicit approximation.

Runtime dependencies are Torch and the provided `appl.public` numerical backbone/loss. The immutable host recipe remains seed 0, 20,000 updates, batch 128, AdamW 1e-4 with weight decay 1e-6, cosine schedule and 500 warmup, gradient norm 1, EMA .999, last EMA selection, and 100-step DDPM train/inference. Interface checks test execution and checkpoint consistency, not task performance. No task-performance result or robustness claim is made. A future allowed ablation against equally sized TCP-to-goal-only and flat-state encoders would be needed to test the scientific hypothesis.
