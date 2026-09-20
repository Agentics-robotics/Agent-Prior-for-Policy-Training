# insert_blue__h01 — Articulated containment margins

## Assigned research hypothesis, unchanged

The assigned statement is: **“Articulated containment margins: encode moving-cavity/pad geometry, rotated extents and strict opening/height thresholds. Hypothesis: goal-margin learning generalizes better than aiming TCP or block center at a single demonstrated point.”**

The supplied geometry and handoff identity are retained. This implementation does not edit the original heuristic. The remaining sections describe executable choices, adaptations and limitations, not additional claims made by that heuristic.

## Full-slice behavior and handoff

This policy is trained on every action in all twelve assigned expanded slices, with no phase deletion, completion-based trimming, resampling by phase or per-skill normalization fit. The exact inclusive-start/exclusive-stop ranges are recorded in pipeline.json. In particular, the roughly 270–278-action overlap with evacuate_red is intentionally retained: red descent onto the pad, red release, empty retreat and movement toward blue, blue approach, closure and initial lift. Later training covers blue transport, lowering into the still-open drawer, release, settling and upward retreat. The last movements are useful optional continuation rather than required task-completion steps.

Measured starts across demonstrations have drawer d approximately 0.300 m, red/TCP heights approximately 0.211/0.212 m, blue on the outside surface at z approximately 0.020 m, and each finger near 0.0182 m (total separation approximately 0.0365 m). Red is not yet on the pad under the strict Z predicate. The current policy must therefore retain unfinished red margins despite its insert_blue name. The observation tokens never delete red or substitute an assumption that it is already placed.

Actual inspected evidence includes:

- demo1000:575 has red z=0.210927 m, TCP z=0.211799 m, closed-command action g=-1 and blue still outside. At 595 red z=0.031365 m, and at 605 z=0.032308 m: proximity to the pad alone is not red success. At 615 red has settled to 0.020 m, fingers are approximately 0.03976 m each, and g=+1.
- demo1000:700 shows the empty open-finger transition, TCP z=0.294 m, while red stays on the pad. At 795 the TCP is near blue with open fingers; at 815 fingers are approximately 0.01826/0.01822 m, blue z=0.02003 m and TCP z=0.01924 m. At 825 blue starts to rise, and 845 has blue/TCP z=0.14011/0.13933 m. These relations support, but do not prove, a grasp. The 845 observation is the overlap boundary state; the preceding overlap actions end at 844.
- demo1001:581,615,825,858 provide the same early takeover, red release, blue closure and rising-blue phases with a different outside blue XY location. At 858, still within the shared action range [581,859), blue/TCP z=0.12990/0.12898 m.
- demo1000:960 has blue at z=0.16520 m over the cavity, still too high for success. At 980 blue z=0.0669845 m and g=-1 already satisfies the simultaneous spatial contract with the placed red and open drawer. At 1000 fingers are approximately 0.03999 m and blue z=0.0629996 m; yaw differs after release. At 1040 and 1064 the empty TCP is near z=0.323 m.
- demo1001:990 similarly has blue z=0.0669725 m with closed fingers. At 1015 blue is settled near z=0.063 m and the open TCP is rising. At 1076 the open TCP has reached z=0.32316 m.
- demo1008:985 has blue z=0.066996 m with g=-1; 1005 and 1070 retain containment after release with a different quaternion/yaw than demo1000. At 1070 blue is approximately [-0.172709,0.072784,0.063000] m and d=0.297386 m.

The supplied measured full-slice endpoint statistics have per-finger opening near 0.04 m, TCP z near 0.323 m, red near [-0.176 to -0.177,-0.298 to -0.304,0.020] m, and blue near [-0.1717 to -0.1728,0.0728 to 0.0742,0.063] m. These are observed support, not admissibility thresholds or learned-policy performance results.

HANDOFF.json describes API-level invocation and continuation. No phase clock, source index, trajectory ID, prescribed invocation schedule or nonlearned action controller enters the model. Two-frame finger openings and qvel, TCP-object offsets, object height and position displacements, red/blue margins and drawer d provide context for takeover partway through the overlap. These cues distinguish red still descending, red settled with an empty retreating TCP, blue approached with open fingers, and blue moving with a nearby closed gripper. Short history cannot resolve every stationary-hold ambiguity or certify contact.

## Exact geometric representation

All geometric positions and lengths are world-frame metres; rotations are supplied as wxyz quaternions. Each quaternion is normalized numerically and converted to a full 3x3 rotation matrix R. For either block, the axis-aligned extent is

`e_i = 0.02 * sum_j abs(R_ij)`.

The cavity center is recomputed from causal drawer position:

`c = [0.19 - d, 0, 0.035]`.

The pad center is `p = [-0.18,-0.30,0.02]`. The ordered nine signed margins are:

1. red XY: `[0.06,0.06] - abs(red_xy-p_xy) - e_red_xy`;
2. red lower and upper Z: `red_z-0.014`, `0.031-red_z`;
3. blue XY: `[0.172,0.182] - abs(blue_xy-c_xy) - e_blue_xy`;
4. blue lower and upper Z: `blue_z-0.053`, `0.074-blue_z`;
5. opening: `d-0.26`.

For the exact contract, the four XY margins must be **nonnegative** and the four Z margins and opening margin **strictly positive**, all at the same observation. The network consumes the signed continuous values, not a discontinuous success flag. Absolute value is used exactly in containment calculations; the auxiliary loss never differentiates through future labels. Orientations also enter learned tokens as all nine matrix elements, including TCP orientation. There is no raw-quaternion bypass. This removes q versus -q ambiguity; it does not make world-frame geometry or absolute joint actions spatially invariant/equivariant.

The fixed supplied contract is authoritative. The demonstration red_goal/blue_goal point fields (41:47) are not used as substitute cavity/pad geometry and no exact demonstrated blue landing point is hardcoded. Blue's demonstrated off-center placement remains learnable from diffusion imitation. Revised geometry would require a revised contract and implementation; this policy does not infer cabinet placement from images.

## Shared normalization

Raw proprioception and absolute position features use the framework's complete-original-demonstration `(raw-mean)/std` values; here std denotes half-range, not empirical standard deviation. Buffers are initialized directly from spec.normalizer and saved with the model. No local fit, narrow per-skill range, clipping, running normalizer or learned rescaling of the native action representation is introduced.

Derived red target offsets and margins use shared red-position coordinate half-ranges. Blue target X uses the maximum of shared blue-X and drawer-position half-ranges, while blue Y/Z use shared blue-position half-ranges. TCP-object offsets use the coordinatewise maximum of the shared TCP and respective object half-ranges. Rotated extents use shared object-position half-ranges; rotations are dimensionless. Lower and upper Z margins use the same object's global Z half-range, not the narrow success interval. Drawer margin uses the shared drawer-position half-range. History displacements are differences of these already-scaled features, not velocities divided by an invented time interval. These transformations derive only from the common normalizer and known contract, not statistics of the selected skill.

## Learned diffusion architecture

policy.py implements a custom transformer, not the optional public U-Net. Model width is 192, attention has six heads, and feedforward width is 768; dropout is zero. For each of the two causal observations there are five tokens:

- **Robot (30 features):** shared-normalized qpos 0:9, qvel 9:18, TCP position 18:21 and nine TCP rotation-matrix elements.
- **Red and blue (25 each):** normalized position, rotation matrix, scaled rotated XYZ extents, TCP-object offset, object-to-pad/cavity offset, and the four corresponding containment/height margins. The two objects share an MLP but have distinct learned type embeddings.
- **Articulated geometry (18):** normalized drawer position/velocity, TCP-to-cavity and TCP-to-pad offsets, normalized cavity/pad positions and scaled pad/cavity half-widths.
- **Joint margins (9):** all nine margins together, maintaining the simultaneous red/blue/opening relation.

Each token feature vector is concatenated with its inter-frame difference; the older frame has zero difference. MLPs embed the doubled feature vectors to width 192. Learned type and two-frame position embeddings identify object and history roles. Two self-attention observation layers produce ten causal memory tokens. Attention can mix both observed frames because neither is in the deployment future.

Sixteen shared-normalized noisy eight-channel action tokens receive learned action-slot positions and a 128-dimensional sinusoidal diffusion-time embedding passed through a learned MLP. Four action blocks apply self-attention over the noisy action chunk, cross-attention to observed memory, feedforward layers and time-dependent scale/shift of layer-normalized activations. Bidirectional action attention is legal: future *noisy actions* are sampler variables, not future observations. The epsilon head maps the final normalized token latents to eight noise channels. The fixed outer DDPM sampler handles 100 noising/denoising steps, clipping and shared affine action decoding. The output is seven absolute Panda joint targets and one continuous gripper command; there is no IK, residual controller, action replay or geometric postprocessing.

Inference uses only `forward(noisy_action,timestep,raw_history)`. The future head is unused by sampling. The framework executes its aligned eight-action prefix and replans using fresh causal history. All assigned default training values are unchanged: 20,000 updates, batch 128, horizon 16, history 2, seed 0, AdamW learning rate 1e-4 and weight decay 1e-6, gradient norm bound 1, cosine schedule with 500 warmup updates, EMA 0.999 and last-EMA selection.

## Losses and gradient paths

The primary objective is the public masked epsilon MSE:

`L_diff = sum(mask * (epsilon_pred - noise)^2) / (8 * sum(mask))`, with a denominator floor for empty masks.

A trainable two-layer future head maps final action-token denoiser latents to nine normalized margin changes. Predicted future margins equal causal current normalized margins plus the predicted change. Labels are exact geometric margins computed under no_grad from demonstration future observations, never inputs to the denoiser. The selected after-action slots are 1,4,8,15. Since slot j is after action t-1+j, these correspond to states at t+1,t+4,t+8,t+15. Slot zero corresponds to the already observed current state and is not supervised.

For each selected valid slot, use smooth L1 with beta=0.1 on normalized predicted versus label margins. Fixed component weights `[1,1,2,2,1,1,2,2,1]` emphasize the narrow red/blue Z predicates without refitting their scales. The regression is divided by valid-slot count times the sum of component weights. Validity is the product of the supplied future and action masks; padding is not trained and labels never cross a slice. If no selected future is valid, the masked prior contribution is zero.

`L_total = L_diff + 0.2 * L_future_margin`.

`prior_loss` reports the unmultiplied future-margin term. Gradients from it update the future head, shared final denoiser latents, all action-attention blocks and the causal observation encoder; the epsilon output head is trained by diffusion MSE. The current-margin residual baseline is observed and nontrainable, but the prediction and error depend on the learned future head. Thus this is not a constant observed-pose penalty. There is no clean-action reconstruction division by sqrt(alpha_bar), avoiding high-noise amplification. At high noise, future regression can predict a conditional mean given history; it is not asserted to be deterministic dynamics.

No term forces blue-inside, red-success or monotonic improvement at every training frame. The future labels include negative margins during unfinished red descent, outside blue acquisition and raised transport. This is essential for coherent full-slice training. No constraint makes a predicted margin a physically guaranteed consequence of the sampled joint action; the intended effect is representation learning through shared denoiser latents, not a separate rollout planner, candidate scorer or dynamics validator.

## Termination, applicability and limitations

A monitor can accept a **single simultaneous** successful observation, including blue still grasped. Do not add release, settling time, object-speed, TCP-clearance or sustained-hold requirements. If the task is already complete, continuing for the demonstrated release/retreat is optional. Red-on-pad alone is not task completion, and fingers closing near blue alone does not establish successful acquisition or justify stopping before insertion. If blue rises with a close TCP and closed fingers, that is supported context for continuing this same policy into transport rather than a requirement to transfer to another option. No downstream manipulation is required; optional idle continuation should leave the objects inside their spatial goals and the drawer open, with an open retreating TCP as useful demonstrated context only.

Applicability requires the known prismatic drawer axis, fixed world pad/cavity geometry, 0.02 m block half-size and reliable causal pose/proprioceptive state. The code uses Torch and the public epsilon-loss helper only. It has no image encoder, geometry estimator, collision checking, joint-space FK, IK, force sensing or long-term contact memory. Two-frame proximity and correlated movement are grasp cues, not force/contact measurements. Unexpected closure, dropped blue, red displaced from the pad or novel obstructions are untested recovery settings and warrant external reassessment rather than confidence in automatic recovery.

Training endpoints are almost upright and comfortably inside XY boundaries, with d around 0.297–0.300 m. Exact margin computation does not establish improved generalization for highly rotated blocks, edge placements or substantially different openings. Strict Z boundaries are represented numerically but are not enforced on generated trajectories. The completion monitor, not the auxiliary head, evaluates actual success. Benefits versus center-distance-only or unstructured matched-capacity encoders remain hypotheses; no such ablation or final task-performance feedback is available here. Interface checks only test executability, real training updates and sampler/EMA reload compatibility, not manipulation success.
