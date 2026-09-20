# SinglePrior_6_xhigh: Object/drawer-relative anticipation

## Scientific choice

This submission is one full-task action Diffusion Policy. Its selected inductive bias is that manipulation progress is better represented by **which entities move relative to the gripper, goals and drawer, and how those relations are about to change**, rather than by an episode clock or a fixed skill sequence. The same learned object encoder processes red and blue, with separate learned role embeddings. A short-horizon relational forecaster provides differentiable anticipation features to one conditional diffusion U-Net. It is trained from the original demonstrations, not from phase labels or a controller.

The expected benefit is statistical efficiency: direct relative coordinates expose grasp alignment, goal error and drawer-relative support; shared processing reuses the grasp/transport geometry of both objects. Predicting relation changes should encourage the condition representation to distinguish approach, object transport and release even where individual positions overlap. This is an expectation based on training evidence, not a measured success claim.

## Training evidence inspected

I read the assignment, numerical interface and backbone implementation, inspected a whole-trajectory overview of every demonstration, and read dense original observations/actions around transitions in demo1000 and demo1011. The following are source observation indices, not labels supplied to the model.

| Trajectory | Whole-trajectory overview indices inspected |
| --- | --- |
| demo1000 | 0, 71, 142, 213, 284, 355, 426, 497, 568, 639, 710, 781, 852, 923, 994, 1065 |
| demo1001 | 0, 215, 430, 646, 861, 1077 |
| demo1002 | 0, 212, 425, 637, 850, 1063 |
| demo1003 | 0, 213, 426, 639, 852, 1066 |
| demo1004 | 0, 213, 426, 639, 852, 1065 |
| demo1005 | 0, 214, 428, 642, 856, 1070 |
| demo1006 | 0, 212, 425, 638, 851, 1064 |
| demo1007 | 0, 213, 427, 641, 855, 1069 |
| demo1008 | 0, 214, 428, 642, 856, 1071 |
| demo1009 | 0, 213, 426, 639, 852, 1065 |
| demo1010 | 0, 213, 427, 641, 855, 1069 |
| demo1011 | 0, 213, 426, 639, 852, 1065 |

The overviews consistently show drawer opening before red removal, red on the pad before blue placement, and a retreat after placement. Initial red XY varies approximately from (0.093, -0.074) to (0.108, -0.077), with other examples reaching y=-0.055; initial blue covers approximately x=-0.414 to -0.387 and y=0.287 to 0.315. This is modest variation, not evidence for arbitrary layouts.

Specific evidence for the representation:

* **Drawer-supported motion:** demo1000 at 0 has red x=0.1055 and drawer opening 0; at 213 red x=-0.1891 and opening 0.2918. At 220 the drawer is at 0.30 and red x=-0.1972. Red's world position changes substantially while its drawer-relative position remains approximately fixed. The forecaster therefore includes both world TCP-relative and moving-drawer-relative relations, rather than treating a world-position change as an object grasp.
* **Handle grip and departure:** demo1000 at 132 and 140 holds the TCP near (-0.148, 0, 0.128) with open fingers; at 150 the fingers are near 0.007 m and the action commands close. At 160 pulling has started. At 220 and 230 the drawer is fully open but fingers remain closed; at 244 fingers are open and the arm is beginning to depart. Drawer opening alone cannot specify the next arm action.
* **Red grasp and lift:** demo1000 at 438 has TCP z=0.0642 near red z=0.0630 with open fingers. At 446 fingers are closing; at 454 the finger positions are about 0.0183/0.0182 and the TCP and red are still near the drawer floor. At 466 both begin rising; at 497 red z=0.3186 and TCP z=0.3195. The object-to-TCP displacement remains small during transport, while object-to-drawer and goal errors change.
* **Red placement and release:** demo1000 at 595 has red z=0.03137, and at 605 red z=0.03231 despite being over the pad. These are above the strict red height interval. At 615 red has settled to z=0.0200 with open fingers; at 625 it stays put while the TCP rises. Simply being over the pad or having closed fingers is not a sufficient progress representation. Dense demo1011 samples 601, 611 and 621 show the same relation change with slightly different coordinates.
* **Blue reuses the manipulation pattern:** demo1000 at 800 reaches blue near table height with open fingers; at 810 fingers close; at 820 the closed grasp is approximately stationary; at 832 blue and TCP lift together. At 955 and 965 both descend toward the drawer. At 975 blue is at approximately (-0.1726, 0.0735, 0.0677), with red on its pad and drawer opening 0.2973, while the gripper still commands close. These coordinates satisfy the supplied simultaneous geometry; release is not an extra task requirement. At 985 and 1005 blue remains on the drawer floor as fingers open and the arm retreats. The final 1065 observation retains all goals.
* **Transition timing is not identical:** demo1011 samples 139, 145, 233, 241, 443, 451, 461, 808, 816, 826, 966, 976 and 986 corroborate the approach/hold/lift/release pattern. Its blue is already in a stationary open-gripper approach at 808, closes by 816 and begins lifting by 826; the analogous demo1000 samples are not identical states at identical indices. A clock-based schedule is neither necessary nor implemented.

All supplied complete trajectory ranges remain in training. No actions are omitted or resegmented. Post-success release/retreat examples are retained as transition context, not promoted to completion requirements.

## Implemented representation and architecture

`policy.py` uses only torch and the public DiffusionBackbone/epsilon_loss helpers. The forward inputs are the fixed noisy action tensor, diffusion timestep and two raw observations. The history order is previous then current. There is no cross-call mutable state, previous-action input, trajectory identity, progress label, episode time, external memory, IK or action override.

The moving drawer center is the supplied geometric expression `(0.19 - drawer_position, 0, 0.035)`. Goals come from observation channels 41:44 and 44:47; in particular, blue's observed goal is not silently replaced by the drawer center.

For each object and each observation, a 16-dimensional local vector contains:

1. object minus TCP, goal minus object, and object minus moving drawer center (nine coordinates), divided by 0.25 m;
2. object height divided by 0.25 m;
3. the first two columns of the object's rotation matrix, computed from its normalized wxyz quaternion. This removes quaternion-sign ambiguity in the object branch.

Each object's encoder input concatenates previous and current local vectors, their ten positional differences multiplied by 10 (equivalent to displacement divided by 0.025 m), and a learned eight-dimensional red/blue role vector: 50 features total. A shared 50->128->128 SiLU MLP with final LayerNorm produces two ordered 128-dimensional tokens. Role-specific behavior remains possible; sharing does not make red and blue interchangeable.

The original 94-dimensional normalized history also passes through a 94->192->128 MLP. Its normalization is exactly the shared limits normalizer, without clipping or per-stage refitting. The ordered object tokens and global token concatenate to 384 dimensions, then pass through a 384->256->128 fusion MLP with LayerNorm. This preserves absolute joint and TCP information needed for absolute Panda joint targets.

A continuous 24-coordinate relation vector R is also formed, in this order:

* red-TCP, red_goal-red, red-drawer_center;
* blue-TCP, blue_goal-blue, blue-drawer_center;
* TCP-drawer_center;
* drawer opening and the two finger positions.

The current R is scaled by 0.25 m for the 21 Cartesian coordinates, 0.30 m for opening and 0.04 m for fingers. Its one-frame change is scaled by 0.025 m for Cartesian/opening channels and 0.01 m for fingers. These fixed physical scales avoid division by narrow demonstration subranges. This difference is a displacement feature, not an asserted velocity or a fitted physics model.

A 176->256->72 forecaster receives fused latent (128), current scaled relations (24), and relation changes (24). It predicts three 24-dimensional changes in R at future slots 4, 8 and 15, relative to the current observation. The output layer has small nonzero random weights and zero biases at initialization. The forecaster is causal and independent of the denoising timestep. It represents expected demonstrated continuation, not an action-conditioned dynamics model.

One public DiffusionBackbone receives a 342-dimensional condition: original normalized history (94), current scaled relations (24), one-frame relation changes (24), fused latent (128), and the **predicted**, not ground-truth, future changes (72). Its widths are 128/256/512, diffusion embedding 128, temporal kernel 5 and groups 8, as required by the fixed recipe. The raw-history and current-relation paths bypass the forecast so the denoiser need not rely exclusively on a potentially averaged future estimate. All trainable components belong to one module and one checkpoint; the forecaster is a representation component, not a second action policy.

The output is always epsilon for the native normalized 16-by-8 action sequence. The public sampler and action decoder remain unchanged. There is no action residual reinterpretation or manually produced gripper command.

## Objective, masks and actual gradient paths

The diffusion objective is the public masked epsilon MSE over all valid action slots/channels. For j in {4,8,15}, the anticipation label is

`Y_j = (R(future_obs[j]) - R(raw_obs[-1])) / forecast_scale`.

The forecast scale is 0.05 m for Cartesian/opening channels and 0.02 m for finger positions. Since action slot j is t-1+j and the future slot is the observation after that action, future slot j corresponds to observation t+j. These supervision horizons are therefore approximately 0.20, 0.40 and 0.75 seconds at 20 Hz, spanning the execution prefix and longer chunk context.

The prior loss is elementwise Smooth L1 (beta=1) between the trainable forecast and Y, multiplied by the corresponding future_mask. It is divided by `24 * sum(valid forecast slots)`, with denominator at least one. No invalid padded future contributes to the label loss. The total loss is

`loss = diffusion_loss + 0.1 * prior_loss`.

The returned prior_loss is the unweighted auxiliary value. Its coefficient is declared in pipeline.json. Targets are computed under no_grad. The forecaster itself is never detached or teacher-forced.

Gradient paths are real and explicit:

* epsilon MSE trains the diffusion U-Net;
* through its conditioning layers it also trains the fusion, global/object encoders, role embeddings, and forecast head;
* masked future regression directly trains the forecast head and the same fusion/global/object encoders and role embeddings used for diffusion conditioning;
* the auxiliary objective does not directly supervise U-Net weights. Its influence on denoising is mediated by the shared condition representation and the forecast that the U-Net actually consumes.

No observed-state-only penalty is claimed to teach the model. No reconstruction of x0 is used in the auxiliary objective, so it does not amplify prediction errors by dividing by sqrt(alpha_bar) at high diffusion noise. Action padding and future padding are handled separately using the supplied masks.

## Training, completion and limitations

The predeclared recipe remains seed 0, 60,000 updates, batch 128, two observations, horizon 16, eight-step execution, DDPM100 epsilon prediction and clip_sample=true, AdamW learning rate 1e-4 and weight decay 1e-6, cosine learning rate with 500 warmup steps, gradient norm clipping at 1 and EMA decay 0.999. The last EMA checkpoint is selected. There is one candidate and no rollout-based selection.

Deployment always evaluates this same learned policy. The evaluator alone decides completion from simultaneous drawer opening strictly above 0.26 m, red pad containment and height, and blue drawer containment and height, including rotated-object XY extents. The policy neither implements its own success detector nor adds a release, speed, clearance or dwell requirement.

Limitations are substantive. Two observations cannot always distinguish identical stationary dwell states before a demonstrated change; small measured velocities are not a reliable substitute for missing long-term memory. A deterministic future regressor can average alternatives and may bias toward the dominant demonstration order; the diffusion model's raw-state bypass helps express residual uncertainty but does not prove recovery ability. Short-horizon labels do not guarantee long-horizon completion. The forecaster is not exact contact physics, a collision model or an inverse kinematics solver. Positions and rotation features do not enforce grasp attachment or containment. Relative geometry does not make absolute joint actions translation- or rotation-equivariant. The drawer center expression assumes the fixed task geometry. Both shared scales and training coverage come from only twelve nearby successful demonstrations; novel layouts, object rotations, missed grasps and substantial perturbations remain unvalidated. The interface check tests execution, differentiability, sampling and checkpoint reload, not autonomous task success.
