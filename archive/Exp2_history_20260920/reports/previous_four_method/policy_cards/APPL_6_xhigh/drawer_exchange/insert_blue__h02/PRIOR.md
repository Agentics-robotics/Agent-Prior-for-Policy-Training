# Contact-belief adaptive diffusion for insert_blue

## Identity and original research hypothesis

Policy: `insert_blue__h02`, heuristic index 2, experiment M1_v2.

The assigned statement is: **"Contact-belief adaptive diffusion: infer attachment/support transitions from causal history and shorten commitment around uncertain changes. Hypothesis: event-aware replanning improves release and handoff timing without an episode clock."** The supplied heuristic and segmentation have not been modified. This document describes the executable interpretation, not a replacement source heuristic.

The important ambiguity is attachment versus support, not simply distance to the insertion goal. The same approximately table-height TCP and blue positions occur with open fingers, closing fingers and a stationary closed grasp. During insertion, the simultaneous task geometry can already be satisfied while blue remains held. This implementation learns contact-sensitive action diffusion throughout the incoming transition, acquisition, insertion and optional release/retreat. There is no source-index input, phase schedule, demonstration lookup, scripted action generator or externally computed IK.

## Interface adaptations and scope of the claim

The research proposal suggests a persistent state-space filter updated at 20 Hz, previous commands where available, and shorter executed prefixes when contact is uncertain. The frozen interface supplies only two causal observations, no command-history channel, no recurrent carry across invocations and no execution-duration return. Training and sampling use horizon 16 and execution prefix 8. Consequently:

1. A GRUCell is unrolled over the two available frames and reset on each call. Its softmax output is a discriminative filtered contact belief, not an exact Bayesian posterior. This is a finite-window recurrent filter, not long-duration memory.
2. Qpos/qvel, actual finger positions and pose increments replace unavailable previous commands as causal evidence. Gripper command alone is never an attachment target.
3. Entropy and predicted contact-change hazards implement **soft temporal commitment inside the denoiser**. They change how strongly a current-contact embedding persists into later action slots. They do **not** shorten the actual executed prefix, alter the DDPM schedule or implement a 20 Hz scheduler. The U-Net still receives global history information; gating is not a hard causal barrier between action slots.
4. Variable input prefixes mean one versus two observation frames. Variable output-prefix masks apply only to auxiliary prediction losses; they do not change the required full-horizon diffusion loss or externally executed prefix.

Thus this package tests contact-belief representation and event-sensitive temporal conditioning under fixed execution. It does not establish the full original adaptive-replanning claim. Testing actual execution-prefix adaptation requires an additional supported inference hook and a separately declared experiment. No such hook is assumed here.

## Full expanded training coverage and measured handoff evidence

All assigned half-open source ranges are retained:

| Demonstration | Start | Stop |
|---|---:|---:|
| demo1000 | 575 | 1065 |
| demo1001 | 581 | 1077 |
| demo1002 | 575 | 1063 |
| demo1003 | 575 | 1066 |
| demo1004 | 575 | 1065 |
| demo1005 | 582 | 1070 |
| demo1006 | 575 | 1064 |
| demo1007 | 575 | 1069 |
| demo1008 | 575 | 1071 |
| demo1009 | 575 | 1065 |
| demo1010 | 575 | 1069 |
| demo1011 | 575 | 1065 |

The assigned overlap with evacuate_red lasts 270-278 actions, from incoming red descent through initial blue lift. It is not discarded as another skill's data. Measured starts have total finger width about 0.03649 m, red/TCP heights about 0.211 m and drawer displacement about 0.300 m. Measured ends have total width about 0.080 m, blue height about 0.063 m, TCP height about 0.323 m and drawer displacement about 0.2973-0.2974 m. These are successful training support, not rejection thresholds or guarantees.

Actual source observations inspected for this implementation include:

- **demo1000:575,595,605,615,625.** Red starts descending while held, is near the pad by 595-605, then fingers open and red settles to z=0.020 m. At 615 each finger is about 0.03976 m while TCP remains at z=0.033 m. At 625 TCP has begun rising while red remains on the pad. Therefore red placement/release and decoupling must be learned, not presumed complete at skill entry.
- **demo1000:680,750,805,815,825,830,845.** Empty travel precedes blue approach. At 805 the TCP is near z=0.0193 m with open fingers. At 815 each finger is about 0.0182 m, but blue is still at source height. At 825 blue is barely lifted; at 830 blue is z=0.031 m and at 845 z=0.140 m with nearly unchanged blue-TCP translation. Closure and confirmed transport cannot be identified by an episode clock.
- **demo1001:581,600,615,625,815,825,835,858.** At 615 the fingers are still opening, about 0.03518 m each with positive velocity. At 815 they are still fully open during blue approach; at 825 and 835 they are closed while blue remains near z=0.020 m. Blue is lifted to z=0.130 m by 858. This timing differs from demo1000.
- **demo1000:960,980,1000,1040,1064; demo1001:990,1015,1076; demo1011:980,1000,1064.** Blue is lowered from held transport to about z=0.067 m, then released near z=0.063 m, followed by a rising TCP and an open-gripper terminal pose. At demo1011:980 the simultaneous geometric task can be complete with blue still held. At 1000 the fingers are nearly fully open and TCP retreat is starting. These are distinct continuation states despite both satisfying the task geometry.
- **demo1011:575,815,845,846.** Successful attachment includes asymmetric finger velocities. At 845-846 blue and TCP rise together; a negative velocity on one finger and positive velocity on the other are not automatically failures.

The useful takeover information is in actual fingers, arm velocity, current object locations relative to TCP and goals, drawer displacement/velocity and the latest relative-motion increment. An incoming caller need not supply private latent semantics. If only one unique observation is available, the model is trained to infer from that observation with motion evidence masked. It cannot recover unobserved earlier contact history.

## Inputs and normalization

Inputs are the supplied 47-dimensional world-frame state: 9 qpos, 9 qvel, TCP/red/blue poses with wxyz quaternion components, drawer displacement and velocity, and red/blue goal positions. Positions are in metres, joints in radians. Qvel includes both finger velocities. No camera, force/torque, tactile channel, previous action or future observation enters deployment forward.

Observation normalization is exactly `(raw - shared_mean) / shared_half_range`; no clipping and no per-skill fitting. Actions remain the framework's affinely normalized seven absolute Panda joint targets plus continuous gripper command. The model returns epsilon in that action representation and the common DDPM/decoder handles sampling. In particular, relative-position features do not make absolute joint actions invariant or equivariant.

For derived translations, define S as the axiswise maximum of the three **shared** TCP, red and blue position half-ranges. The six relative vectors are red-TCP, blue-TCP, red-red_goal, blue-blue_goal, TCP-blue_goal and blue-drawer_center, each divided by S. Drawer center is `[0.19-d, 0, 0.035]`. This center differs from the demonstration blue placement goal; both are represented. No tiny empirical segment range is used. Finger and drawer motion use their supplied shared half-ranges. All original quaternion components remain in the normalized state; this model does not implement a special rotation-invariant encoder or a pose-manifold loss.

## Learned modules and conditioning

Each frame has 85 features: normalized state (47), relative translations (18), one-step motion features (18), and two availability bits. The motion state consists of scaled TCP, red and blue positions, red-TCP and blue-TCP translations, both finger positions and drawer displacement. Differences of that state encode co-motion and separation. Motion is an increment per observation, not a separately estimated high-frequency velocity. Supplied qvel remains available.

An 85-160-128 SiLU MLP feeds a 128-dimensional GRUCell. The first-frame update is masked if both supplied frames are identical within 1e-7 maximum absolute component difference; the second is always used. This convention recognizes duplicate left padding without an explicit mask, but also masks truly identical pairs. During training, the older frame is deliberately replaced by the latest frame in 30% of examples. The same deterministic forward implementation is then called for diffusion training and inference. There is no mutable hidden-state cache whose contents could depend on DDPM iteration order.

The final recurrent state predicts:

- An eight-mode soft belief: red-held; red-release-near-pad; free-travel/open-blue-approach; blue-source-contact/closing; blue-held-transport; blue-lowered-held; blue-release/retreat; blue-released-clear.
- A 32-dimensional continuous latent with an 18-dimensional relative-position readout.
- A 16-slot forecast of contact distributions, and a 16-slot adjacent contact-change hazard forecast.

The common 128/256/512 conditional U-Net receives a 185-dimensional condition comprising recurrent state, continuous latent, current mode distribution, normalized mode entropy and predicted hazards. It is the primary learned epsilon predictor. A small learned temporal convolutional residual receives noisy actions, diffusion-time embedding, action-slot embedding, recurrent context and a slot-specific contact embedding. The latter is

`retention_j * embed(current_mode) + (1-retention_j) * embed(forecast_mode_j)`.

Here `retention_j = product_{k<=j}(1-hazard_k) * exp(-2*entropy*j/15)`, with slot-zero hazard fixed to zero because its resulting state is already observed. High predicted entropy or upcoming change reduces persistence of the current mode. This is learned contact-sensitive conditioning, not an action threshold policy. Local slot embeddings refer only to positions in the predicted chunk, not episode time. The residual output layer starts at zero; diffusion and auxiliary gradients train it thereafter.

A separate GRU-based learned dynamics head receives the encoded action at each slot, slot embedding and filtered context. It predicts the change of the 18-dimensional motion state from the current observation. It is used as a training regularizer and is not called to synthesize deployment actions. It is neither robot forward kinematics nor an exact physical rollout.

## Contact labels and future alignment

At observation t, action slots are t-1 through t+14. Future slot j is the state after that action, so slot zero corresponds to the current state. All future targets use `future_mask * mask`; adjacent change labels require both neighboring states. No target crosses a skill boundary.

Contact supervision is deliberately **weak and soft**. Width uses both measured fingers, with a smooth closed score centered at total width 0.058 m, between the observed grasp width about 0.0365 m and open width about 0.080 m. TCP-object proximity has a 0.05 m radial scale. Red release also uses proximity to its pad and low red height. Blue source/transport membership uses a smooth elevation transition centered at 0.038 m, and lowered-blue uses proximity to its goal and height below roughly 0.09 m. Released-blue scores use goal proximity and z proximity to its 0.063 m placement goal. A smooth 0.13 m TCP-blue clearance transition separates release/retreat from released-clear. These are training-label design choices, not success constraints or execution guards.

Training-only co-motion uses a three-observation lookahead when all endpoints are valid: object displacement is compared with TCP displacement, weighted by evidence of actual TCP movement. It reduces confidence in nominal attachment when motion is inconsistent. With no valid full lookahead it contributes a neutral 0.5 score; current and previous labels also use their available endpoints. Static co-location cannot establish attachment, so held-state scores are not treated as certificates. Source closing is distinct from elevated transport. Soft scores are normalized and mixed with 2% uniform mass. The gripper command is not used in these labels, and finger velocity asymmetry is not penalized.

Adjacent changes of the dominant weak mode supervise the hazard head. This discretization can create uncertain label-boundary events and is not a calibrated physical hazard estimator. Positive events receive weight 4 to counter sparse changes. Future mode and hazard predictions at deployment are generated entirely from causal history, not supplied future fields.

## Losses and gradient paths

The full objective is:

`L = L_epsilon + 0.10 L_current_contact + 0.025 L_previous_contact + 0.05 L_future_contact + 0.05 L_change + 0.05 L_relative + 0.15 L_teacher_motion + 0.05 L_denoised_motion`.

- **L_epsilon:** ordinary masked full-horizon epsilon MSE. Every valid action participates, including red release, blue acquisition and optional terminal continuation. Gradients train the U-Net, temporal residual, recurrent encoder, contact/forecast heads and continuous conditioning latent.
- **Contact terms:** soft cross-entropy for the current filter, first-frame filter when available, and 16-slot forecast. Labels are detached; predictions are trainable. Current/previous cross-entropy trains the shared contact readout and encoder; forecast cross-entropy also trains the future-mode head used by the denoiser.
- **L_change:** masked binary cross-entropy on adjacent weak-mode changes, with positive weight 4. It trains the hazard forecast used by the soft commitment gate and global condition.
- **L_relative:** smooth L1, beta=0.1, between the latent readout and current normalized relative vectors. This is a loss on a learned prediction, not a pose-only constant penalty.
- **L_teacher_motion:** the learned recurrent dynamics prediction from clean encoded demonstration actions is compared with masked future motion-state changes using mean smooth L1, beta=0.1. This trains action/state coupling and the causal context encoder.
- **L_denoised_motion:** the same learned head receives `x0 = (noisy_action - sqrt(1-alpha_bar)*epsilon) / sqrt(alpha_bar)`, clipped to [-2,2] only inside this auxiliary path. Its future-state error is multiplied by alpha_bar before masked averaging, suppressing high-noise amplification. Gradients reach the dynamics head, conditioning network and denoiser through x0. No detach severs this denoiser path. The auxiliary clipping is not an alteration of action decoding or the fixed sampler.

Future contact, change and motion objectives randomly use prediction-prefix lengths 4, 8 or 16 with equal probability, intersected with valid masks. Current-state losses are still applied, and diffusion always covers the entire valid action sequence. Denominators count valid targets, not padded length. The weak labels and future motion arrays are constructed under no-grad; all auxiliary errors involve trainable outputs. The learned motion head might exploit state-context correlations rather than causal action effects; this design does not prove identified dynamics.

## Deployment, continuation and success

Deploy `forward(noisy_action, timestep, raw_history)` only. It returns an epsilon tensor shaped [B,16,8]. Future labels and the auxiliary dynamics model are absent from this path. The recurrent filter is recomputed from the current two-frame window at every sampling invocation, with identical causal inputs throughout its DDPM iterations. There is no assumption that it is refreshed on unobserved intermediate execution steps.

The model learns the full transition from held-red descent, through opening and decoupling, empty travel, blue closure and lift, to held transport, lowering, optional release and retreat. Initializing partway through that overlap is supported by the same causal cues; it does not require restarting at the slice start. HANDOFF.json describes when continuing this motion is useful and what a successor should re-observe.

The terminal monitor must evaluate the supplied **simultaneous** task contract: drawer displacement strictly greater than 0.26 m; red center z strictly between 0.014 and 0.031 m and its rotated XY extents contained within the pad centered at [-0.18,-0.30] with half-widths [0.06,0.06]; blue center z strictly between 0.053 and 0.074 m with its rotated XY extents contained in the drawer cavity centered at [0.19-d,0], half-widths [0.172,0.182]. Each block has half-size 0.02 m; rotated axis-aligned extents use its measured wxyz orientation. This contract is documented for the caller; no substitute completion detector is implemented inside the diffusion forward.

No release, TCP clearance, velocity limit, sustained duration or extra stable-state requirement is added to task success. Blue-held near z=0.067 m can already succeed. Continued release, settling and retreat are retained as useful optional motions rather than omitted after the first successful observation. Transfer after release can use observed finger opening, persistent blue height/position and increasing TCP-blue separation; it should not trust a release-mode probability alone. Actual simultaneous goal fields remain authoritative.

## Dependencies, limitations and test plan

Only Torch, math and the public diffusion backbone/loss helper are imported. The fixed shared normalization, DDPM 100-step training/sampling, action decoding, batch 128, 20,000 AdamW updates, EMA 0.999, cosine schedule and last-EMA selection remain the repository's recipe. No private dataset reading, file access, external weights or simulator interfaces occur in model code.

This is a successful-demonstration prior, not demonstrated recovery. Drops, obstructions, drawer closure, unusual grasps and unseen contacts are outside established support. Two frames may not distinguish static supported and held objects. Quaternions are provided but not used for exact contact/containment labels. Weak mode labels are not human annotations, posterior confidence is not calibrated, and the smooth label thresholds need ablation. The motion auxiliary is not a collision constraint or safety guarantee. Event weighting and entropy-based retention do not establish better timing, and actual adaptive execution is absent under this interface.

Predeclared useful scientific comparisons are a memoryless denoiser, a two-frame encoder without contact supervision, removal of motion coupling, removal of the event-conditioned residual, and, only with a supported execution API, fixed versus genuinely adaptive execution prefixes. The interface check validates execution, differentiable GPU updates and DDPM/EMA compatibility; it is not a closed-loop performance evaluation. Final selection remains the last EMA at the assigned budget, without performance-driven tuning here.
