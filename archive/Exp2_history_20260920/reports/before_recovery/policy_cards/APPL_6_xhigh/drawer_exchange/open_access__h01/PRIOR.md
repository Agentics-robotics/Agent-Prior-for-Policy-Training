# open_access__h01: Prismatic relational action diffusion

## Assigned research hypothesis (unchanged)

Heuristic 1, **Prismatic relational coordinates**:

> Prismatic relational coordinates: separate articulation along the drawer axis from object offsets while retaining robot reachability. Hypothesis: this representation improves generalization to placement and opening variation rather than memorizing an absolute TCP path.

The assignment's rationale is that red moves with the drawer during opening while blue remains outside, and later red is stationary while the released TCP retreats and rotates before pickup. A moving drawer frame separates articulation from placement. The hypothesis is a representation hypothesis, not a time-indexed phase model or action-candidate lookahead. This implementation does not edit the source heuristic. The following sections describe our executable adaptation and its limitations.

## Scope and evidence: the full M1_v2 slice

All twelve assigned slices are retained without resegmentation: start 0 for all trajectories; exclusive stops 475 for demo1000, 476 for demo1001, 475 for demo1002/1003/1004, 474 for demo1005, 475 for demo1006, 474 for demo1007/1008 and 475 for demo1009/1010/1011. The shared overlap with evacuate_red is [195, stop), comprising 279-281 actions per trajectory. Training covers approach, handle closure, loaded pull, full opening, handle release, upward retreat, red approach and descent, red closure, and initial lift. We do not discard the transition after reaching the drawer threshold.

Actual source observations/actions inspected:

- demo1000 at 0, 140, 150, 160, 195, 215, 230, 240, 300, 390, 440, 450, 465 and 474.
- demo1001 at 0, 195, 240, 300, 390, 440, 450, 474 and 475.
- The assignment's measured start/end distributions and all twelve overlap ranges were also read.

Selected measurements (world metres; fingers are individual qpos values, not total width):

| Observation | Drawer and manipulation evidence |
| --- | --- |
| demo1000:0 | d=0, red x=0.10551 and z=0.063, TCP approximately [-0.38387,0,0.44196], fingers=0.04, action g=+1. |
| demo1000:140-160 | TCP near [-0.148,0,0.128]; g changes from +1 to -1; fingers settle near 0.0071 each and drawer begins moving. |
| demo1000:195 | d=0.18630, drawer speed=0.10710 m/s, TCP x=-0.32631, red x=-0.08354; g=-1. This is an early overlap state, not an already completed opening. |
| demo1000:215/230 | d approximately 0.297/0.300; TCP z approximately 0.128, fingers still near 0.0071, g=-1. Open geometry alone does not identify release. |
| demo1000:240 | d approximately 0.300, TCP x=-0.44042, red x=-0.19724, fingers approximately 0.03939, g=+1. |
| demo1000:300 | d=0.300, TCP z=0.36856, red z=0.063; released upward retreat. |
| demo1000:390 | TCP nearly above red in XY and z=0.31294, with a markedly changed orientation. |
| demo1000:440/450 | TCP about 0.009 m to red's -X side at z approximately 0.064. Fingers change from 0.04 to approximately 0.01824, g from +1 to -1. Red has not yet substantially lifted. |
| demo1000:465/474 | Red z increases from 0.07290 to 0.13331 with TCP maintaining approximately [-0.009,0,0.001] relative to red; g remains -1. |
| demo1001:0/195/240 | Red starts at [0.10770,-0.07662,0.063], and later x=-0.08135/-0.19505, while opening motion is almost identical to demo1000. |
| demo1001:474/475 | Consecutive red heights 0.12424 and 0.13356; TCP heights 0.12510 and 0.13442; each finger approximately 0.0182, g=-1. This directly supports positive co-lift near the exit. |

The supplied post-final-action boundary statistics put red z at 0.14203-0.14344, TCP z at 0.14290-0.14430, d at 0.300, and total finger width at 0.03648-0.03649. These are later than the pre-action last rows above and are consistent with one further lift action. They are observed training support, not validated tolerance bands or proof of grasp. Initial red slip also makes red_x+d only approximately constant.

## Implemented representation

The model accepts raw history [B,2,47]. Named inputs used are qpos/qvel, TCP/red/blue world poses, drawer position and drawer velocity. No images or goal-dependent controller are present. The supplied red_goal and blue_goal entries are not used: the fixed geometry and the local opening-to-red-lift slice do not require learned goal relocation, and transfer to new goal placements is not claimed.

For each causal frame:

1. Compute the moving drawer center c=[0.19-d,0,0.035] metres. World -X is the opening direction; the encoder retains signed X, not its absolute value.
2. Build four **typed 64-dimensional nodes**, with distinct MLP parameters:
   - Robot: the 18 normalized qpos/qvel entries, three normalized TCP world coordinates and nine TCP rotation-matrix entries (30 inputs).
   - Drawer: normalized d and drawer velocity, plus world drawer center (5 inputs).
   - Red and blue: their normalized world position and nine rotation-matrix entries (12 inputs each).
3. Encode TCP-c, red-c and blue-c with **separate axial-X and transverse-YZ MLPs**, 32 outputs each. Their concatenation is a 64-dimensional typed drawer-edge descriptor. Independent parameters are used for each edge type.
4. Encode red-TCP and blue-TCP as additional typed 64-dimensional Cartesian edge descriptors. Together these preserve the red-relative cues needed after release without dropping drawer-relative cues needed for early takeover.
5. Perform two learned message-passing rounds on the five undirected relations. Each edge has two independently parameterized directed MLPs taking receiver node, sender node and the fixed-direction edge descriptor (192 inputs, hidden 128, output 64). Incoming messages are averaged by node degree. A node-specific residual update MLP and LayerNorm produce the next nodes. Axial/transverse channels first mix in these learned messages; there is no enforced independence of action coordinates.
6. Concatenate the four final nodes per frame (256 features). Fuse previous, current and current-minus-previous latent features with a 768-to-384-to-256 MLP. This is causal two-frame conditioning, not a persistent phase state or source-index embedding. Joint and drawer velocities are directly observed; object/TCP motion can be inferred from the ordered history and its latent difference. No unprovided observation timestep is used to invent metric pose velocities.

Each quaternion is normalized and converted from wxyz to the full 3x3 rotation matrix. This encoding is invariant to replacing q by -q. Raw quaternion components are not also passed through another branch. This narrow sign-invariance claim does **not** imply cabinet-frame SE(3) invariance or action equivariance. World heights and arm joints are intentionally retained for gravity, robot-base reachability and the absolute controller.

TCP-c describes handle-relative configuration up to the cabinet's fixed geometry. There is no measured handle pose, fitted contact frame, force signal, or hardcoded handle-contact detector. The model must learn the significance of those offsets from demonstrations.

### Shared normalization, not per-skill scaling

The provided complete-original-demonstration normalizer is used unchanged (normalizer digest e75d5e1c8d9862f1611bf577ea5fa6e6d3fd5a25a69627c1f55602c602a7193f; 12 originals, 12,809 fit samples). In limits mode its `std` entries are half ranges. State normalization buffers come from `spec['normalizer']`; there is no local fit, learned rescaling of the dataset statistics, observation clipping, or use of empirical tiny variances.

For a relative displacement p-c, divide by shared scale s_p+[s_d,0,0]. For red-TCP and blue-TCP, divide by the coordinatewise sums of their shared half ranges. These uncentered relative coordinates have conservative fixed scales derived solely from the original common normalizer. Drawer center world coordinates use the shared TCP position mean and half ranges. Rotation matrices use their natural bounded entries with no fitted statistics. The auxiliary uses shared s_d=approximately 0.15 m, not a small displacement-specific scale.

## Learned action diffusion and fixed recipe

The 256-dimensional causal condition drives the supplied Conditional U-Net via global FiLM conditioning. We use the public DiffusionBackbone with widths [128,256,512], kernel 5, groups 8 and diffusion timestep embedding 128. Model.forward(noisy_action, timestep, raw_history) returns an [B,16,8] epsilon prediction in both training and inference.

The action representation is unchanged: seven original-shared-normalized absolute Panda joint targets and one normalized gripper command. These are not Cartesian displacement commands. The outer DDPM performs 100 training/inference steps with clean-sample clipping and affine inverse normalization. Thus output bounding is the fixed sampler's original training-range bounding, not a new kinematic or collision constraint. Native g increases toward open; per-finger target is 0.025*g+0.015 metres. Physical aperture under contact need not equal that target.

Training remains seed 0, 20,000 updates, batch 128, history 2, horizon 16, execution prefix 8, AdamW learning rate 1e-4 and weight decay 1e-6, gradient norm limit 1, cosine schedule with 500 warmup updates, EMA 0.999 and last EMA selection at the declared budget. No data subset, per-phase weighting, schedule, replay, scripted action sequence or separate controller is introduced. Initialization is learned from scratch. Check runs are interface/numerical validation, not task-performance evidence.

## Losses and trainable gradient paths

Let epsilon_hat be the model's prediction and M be the action padding mask. The diffusion loss is the public masked epsilon MSE:

L_diff = sum(M * (epsilon_hat-epsilon)^2) / max(8*sum(M),1).

A 256-to-128-to-16 MLP predicts **drawer displacements from the same causal relational condition**, not from future input or proposed actions. Its target for slot j is

y_j = (future_obs[j,39] - raw_obs[current,39]) / shared_drawer_half_range.

Action slot 0 is action t-1 and its post-state is the already observed current state t. Therefore slot 0 is excluded from this auxiliary. Slot 1 predicts d_(t+1)-d_t, the assigned next-drawer-displacement task. Slots 2-15 extend that same task to cumulative displacements through t+15. This extension gives longer pull and release context at the supplied horizon without adding a forward-search mechanism. The final head's slot-0 output is unused by design. Labels are detached. Validity is future_mask times action mask, with slot 0 set to zero, so no padded or cross-slice future target contributes.

L_prior = sum(valid * (drawer_prediction-y)^2) / max(sum(valid),1).

L_total = L_diff + 0.25 * L_prior.

`prior_loss` reports the unweighted L_prior. L_diff trains both encoder and denoiser. L_prior trains the drawer head and the very same relational/temporal encoder used to condition the DDPM. It is not a penalty calculated only from observed poses. The loss implementation calls the ordinary model forward, then deterministically recomputes the shared encoder for the auxiliary, without caching or detaching that prediction path. No clean-action reconstruction penalty is used, so there is no high-noise x0 amplification. A fully masked auxiliary is a differentiable zero.

The auxiliary is **not action-conditioned dynamics** and does not certify that a particular sampled action opens the drawer. It encourages a representation predictive of demonstrated articulation progress. At inference only the epsilon path is used; the auxiliary outputs are neither sampled-state inputs nor a selection/scoring controller.

## Causal handoff and useful exit

HANDOFF.json gives the selection-facing interface. The policy can take over partway through a demonstrated transition because it observes configuration and motion rather than invocation duration:

- Positive drawer speed, narrow fingers, low TCP near the handle in drawer coordinates, and translating red indicate a loaded pull; at the overlap start d is still only approximately 0.1863 m.
- Full opening with narrow fingers and low TCP differs from full opening with wide fingers and a rising TCP. The model learns release and retreat, not an automatic terminate-on-d threshold.
- During retreat/approach, red remains at drawer-floor height while TCP height and orientation change. Red-relative offsets locate the object despite modest placement differences.
- Near red, finger closure alone is ambiguous. Approximately 0.0182 m fingers, TCP about 9 mm on red's -X side, and correlated positive red/TCP height changes support the late initial-lift state. At this boundary a successor should keep the grasp closed and continue lifting before transporting red.

Both policies cover the entire shared transition, so the inference agent may transfer earlier only if it expects the successor to finish pulling/releasing/retreating as necessary. For this policy the useful late exit is initial red lift, not placement on the pad. Invoking it far beyond that exit is unsupported. No hard phase classifier, termination gate or preset skill schedule exists in policy.py.

## Applicability, limitations and falsifiability

The same Panda, pd_joint_pos controller, world -X drawer axis, gravity and fixed cabinet geometry are assumed. A changed cabinet pose would require a newly supplied frame and a corresponding implementation/data adaptation; this code does not infer such a frame. The observed manifold includes the canonical closed start and its approach/contact/overlap states, not arbitrary open-drawer or arbitrary grasp starts. Use qpos/qvel, causal poses, aperture and drawer states together rather than treating a single geometric threshold as a complete manipulation state.

Only successful data are provided; blocked-drawer recovery, contact robustness, collision avoidance and grasp failure recovery are not established. Width and co-motion are evidence consistent with contact, not direct sensing of contact. Two observations cannot always distinguish identical stationary states just before and after a demonstrated dwell. There is no clock to resolve that ambiguity. Object placements vary modestly, while opening and robot starts are almost identical; a relational encoder may still memorize this narrow distribution. The predictive auxiliary may learn an easy drawer-velocity correlate. There are no external IK/FK calls, image encoders, force models, constraint projections or rollout-based action ranking.

The testable claim remains better sample efficiency/generalization for loaded pull versus red approach under modest placement/opening variation than an unstructured world-state MLP. A capacity-matched world-state encoder comparison on held-out trajectories and controlled variations would test it; no advantage or systematic drawer-position-dependent errors would falsify the claim. Such experiments are not performed or claimed here.

Dependencies are Torch and appl.public only. All neural modules, normalizer buffers and auxiliary parameters are part of the model state dict for EMA/checkpoint reload. No file, network, simulator, source-trajectory or external model access occurs in policy execution.

## Task completion is separate from local handoff

The task succeeds only when all supplied contract conditions hold at one observation: drawer_position strictly greater than 0.26 m, full rotated-extent XY containment of red on the pad centered [-0.18,-0.30] with half-size [0.06,0.06] and 0.014 < red_z < 0.031, and full rotated-extent XY containment of blue inside the drawer centered [0.19-d,0] with half-size [0.172,0.182] and 0.053 < blue_z < 0.074. Blocks have 0.02 m half-size. There is no extra release, velocity, sustained-time or TCP-clearance requirement. This local policy does not accomplish the red-pad or blue-insertion goals by itself. Its retained post-opening actions provide overlap context and a useful red-lift exit; later red transport and blue insertion remain successor skills, not actions silently omitted because opening alone was called task success.
