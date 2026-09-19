# acquire_lift__h01: Role-relative geometry with support context

## Assigned scientific hypothesis

The assigned A1 hypothesis is that sharing local TCP/object/support geometry helps learn the common pickup problem at two heights and across small source translations, while global robot configuration prevents a false assumption of whole-arm translation equivariance. It is a representation prior, not a timing model or a motion-constraint prior. The original assignment is not modified by this implementation document. The following sections describe the executable realization and its deviations from the research proposal.

This is a learned epsilon-prediction action diffusion policy. It has no action lookup, scripted joint controller, preset skill schedule, phase-based action switch, external IK, or trajectory replay. Analytic geometry constructs conditioning features only. Every executed action is produced by the fixed DDPM from the learned denoiser.

## Full expanded training support

All 24 assigned expanded slices of the twelve training demonstrations remain unchanged. In particular, the second invocation is NOT cropped to blue pickup. In demo10200 the slices are [0,180) and [235,540). They cover initial red approach, descent, closing, lift and early lateral transport; then finishing red descent onto its pad, opening, retreating, returning to blue, blue descent and closing, blue lift and early transport. The other trajectories retain their own assigned boundaries. No training examples are filtered by distance, finger opening, inferred contact, or a claimed core phase.

The measured overlaps with carry_place_clear include demo10200 [80,180), [235,430), and [440,540). Across the twelve demonstrations, the first overlap lasts 95-100 actions, the inter-object transition overlap 185-195 actions, and the blue pickup overlap 98-102 actions. Sharing these motions is deliberate: an incoming or outgoing policy can operate partway through a transition. There is no demand to stop at one terminal joint configuration.

Inspected observations establish the required behavior, not learned performance:

- demo10200 80 has open fingers near 0.040 m each, TCP z 0.0772 m and red z 0.0600 m. At 100 the command is -1, fingers are about 0.01825 m, and red is still at pickup height. At 150 red has risen to 0.3082 m while blue remains at 0.0200 m. At 179 red is at 0.3129 m and substantial nonzero qvel accompanies lateral transport, still with command -1.
- demo10200 235 begins the blue slice while red is at z 0.2091 m over its pad, TCP z 0.2104 m and closed fingers. The next action continues red placement, not a direct blue approach. At 280 red is at table height, fingers have opened to about 0.03934 m and command is +1. At 330 the open TCP has retreated to z 0.3016 m; at 390 it is above blue. At 440 the open TCP is at z 0.0220 m, and at 450 fingers are closing near 0.0204 m each. At 500 blue is at z 0.2529 m, at 520 at 0.2799 m, and at 539 at 0.2826 m with early lateral motion and command -1.
- demo10204 starts from source XY (-0.33962, 0.00609) m, rather than demo10200 (-0.35330, -0.00529) m. At 100 fingers are still closing, at 130 red is at z 0.1294 m, and at 160 red is at 0.3192 m. The blue slice starts at 230 with red still held at z 0.2772 m. At 265 red is near table height but fingers remain closed and command -1; table height alone is therefore not evidence that release and clearance are complete. At 280 opening is underway. At 505 blue is only at z 0.2518 m, and at 531 it reaches 0.2807 m. These actual observations are used rather than assuming phase from source index.
- demo10208 starts at XY (-0.36141, 0.00977) m; at 240 red remains held over its pad at z 0.2085 m. At 286 it has been placed and the fingers are nearly fully open. At 446 blue descent is still open; at 472 the fingers are closed but blue is still at z 0.0201 m. At 547 blue is at 0.2832 m and moving laterally.

These examples and the assignment's boundary statistics support the entry/exit discussion in HANDOFF.json. They do not establish contact thresholds, recovery success or universal reachability.

## Necessary adaptation: invocation roles are absent from the executable interface

The proposed heuristic asks for selected/previous invocation tags, supplied from segment identity in training and by the caller at deployment. The actual numerical contract provides only `raw_history [B,2,47]` to forward. Neither the training batch nor the supplied spec contains per-example segment identity, selected-object tags or previous-object tags. This package does not pretend those fields exist, encode them in drawer channels, use future goal completion, or infer them from a hidden clock.

Instead, it implements **three learned latent relational attention slots** over both object tokens plus a null option. These slots are trained through the action diffusion objective, without supervised selected/previous/held labels. They can specialize to manipulation-relevant roles, but no particular slot is claimed to be a calibrated semantic role. A mean-pooling path always retains BOTH updated nodes, including red during the entire blue invocation. One-hot red/blue observation-field identities are available to the shared node MLP; they are not invocation tags. Each object's own measured goal and goal-relative displacement also ground its task identity.

This substitution preserves the central spatial-sharing hypothesis, but restricts intent conditioning. In the demonstrated ordered task, red at the source stack versus red over/on its goal, together with TCP association, finger state and motion, makes the observed invocations distinguishable. A still-held red over its pad remains explicitly represented along with blue. The network can learn to finish red descent before selecting a blue-oriented action. It does not implement a rule that all closed-finger states should lift, nor a rule that a blue invocation should immediately approach blue. If the same physical state is presented with two different unobserved caller intentions, this model cannot distinguish them. Arbitrary selected-object commands, wrong-order execution and recovery calls must not be assumed to work.

The suggested comparable sampling of red and blue invocations also cannot be guaranteed through these hooks: the fixed loader is retained, without a metadata-dependent sampler or inverse-length weights. All valid actions in both expanded invocations contribute to the objective, including the longer second invocation. This is a disclosed implementation limitation, not a claim that samples have been balanced.

## Causal representation and learned architecture

Inputs are the two causal states, each with nine qpos components (seven arm radians and two finger metres), nine qvel components, TCP/red/blue world positions in metres and wxyz quaternions, and two world goal positions. The fixed-zero drawer channels are unused after applying the common observation normalization operation. No images, future states, source indices, action history, counters, or persistent learned hidden state are required.

Quaternion inputs are normalized to unit length and converted to 3x3 rotation matrices. Both absolute matrices and relative products are exactly insensitive to quaternion sign. This is not a claim of global rotation invariance. Position differences remain in world axes, the world table normal remains special, and the absolute robot branch remains essential.

For each object and each history time, a shared 40-to-128-to-96 SiLU MLP receives:

1. Absolute world position centered with a shared object center (3), and world rotation matrix (9).
2. Its goal minus object position (3), TCP minus object position (3), and TCP-relative rotation (9).
3. Object bottom height above the z=0 table, using the supplied half-height 0.02 m (1).
4. One-step object displacement (3) and TCP-minus-object displacement change (3). The older frame has zero displacement because no earlier observation is supplied.
5. Red/blue field identity (2), and normalized finger positions and finger velocities (4).

The 33-to-128-to-96 robot-global MLP sees normalized qpos/qvel (18), normalized absolute TCP position (3), absolute TCP rotation matrix (9), and scaled TCP displacement (3). Thus local geometry never discards Panda configuration or global workspace information.

Each directed object-to-object edge has 17 features: relative world position (3), relative rotation (9), vertical bottom-to-other-top gap (1), squared scaled lateral distance (1), and relative displacement change (3). For a red cube sitting above blue, the red-bottom/blue-top gap is near zero. These are support geometry cues, not measured contact. With tilted cubes, the simple vertical gap is only a coarse descriptor, not exact oriented-cube support computation.

Two learned message-passing rounds use shared weights across object identities, edge directions and observation times, but distinct weights per round. Each message MLP maps [receiver node, other node, edge] from 209 through 128 to 96 dimensions. A 288-to-128-to-96 update MLP combines receiver, message and global robot feature; a residual connection and LayerNorm finish the round. This explicitly shares the relational mapping for both pickup heights.

A 192-to-96-to-3 MLP scores each updated object for three latent attention slots, conditioned on the robot embedding. Global features produce null scores and each slot has a trainable null token. Softmax operates over the two objects and null. The three pooled slots (288 dimensions), mean node embedding (96) and robot embedding (96) feed a 480-to-256-to-192 frame MLP. The old frame, new frame and their difference form a 576-dimensional causal temporal feature, projected through 384 to the 256-dimensional denoising condition. Soft attention affects conditioning only; it does not choose or override an action controller.

The condition drives the provided temporal conditional U-Net, with widths 128/256/512, kernel 5, eight groups and timestep embedding 128. Forward predicts epsilon for a [B,16,8] noisy action chunk. Seven outputs correspond to normalized ABSOLUTE joint targets, not joint increments or Cartesian commands; the eighth is the normalized native gripper command, where +1 is open and -1 is closed. The fixed sampler denormalizes actions and applies its configured sample clipping. The policy adds no custom projection, IK or constraint enforcement.

## Normalization

There is no fitting in this policy. The supplied common normalizer was fitted on 8729 samples of all twelve COMPLETE original demonstrations, not these skill slices. Its half-ranges and centers are registered buffers and checkpointed with the model. Standard state features use precisely those supplied observation parameters.

All geometric differences are first computed in physical coordinates, then divided by the coordinatewise maximum of the shared TCP/red/blue position half-ranges: approximately (0.0686444, 0.257927, 0.210996) m. This uses only fixed dataset-wide scales and is identical for both objects and both edge directions. The shared absolute-object center is the arithmetic mean of the supplied red and blue position centers, with the same spatial scale. Displacements use those same spatial half-ranges, not a separately fitted velocity normalizer. Table bottom height and object-to-object support gaps also use the shared z scale. The constants 0.02 and 0.04 m express supplied object dimensions, not new empirical normalization statistics. Rotations and identity features are dimensionless. No per-skill scaling, observation clipping, random spatial augmentation or action representation refitting is performed.

## Objective, gradient paths and training

The only objective is the provided masked epsilon MSE:

`sum(mask * (epsilon_pred - noise)^2) / max(8 * sum(mask), 1)`.

`loss = diffusion_loss` and `prior_loss = 0 * diffusion_loss`. Architectural priors are explicitly permitted to have zero separate auxiliary loss. There are no penalties computed solely from observed positions and mislabeled as trainable constraints. Gradients from epsilon prediction reach the U-Net, condition MLP, attention/null parameters, message and update MLPs, and shared node/global encoders. Fixed geometry and normalizer buffers are not learned. Future observations and future masks are not used by this implementation at all; action padding is handled by `batch['mask']`. Action slots retain the supplied alignment t-1 through t+14; the policy neither shifts them nor constructs cross-slice future windows.

The outer recipe is unchanged: seed 0, 20,000 updates, batch 128, AdamW at 1e-4 with 1e-6 weight decay, norm clipping 1, cosine decay with 500 warmup updates, EMA 0.999, horizon 16, history 2, execution 8, and 100 training/inference DDPM steps with clip_sample enabled. Selection is the last EMA at this predeclared budget, not validation-based checkpoint selection. Deployment repeatedly reconditions on observed history and executes the repository's receding-horizon samples at 20 Hz. The API decides invocation duration; this package has no termination controller.

## Handoff and observation sufficiency

Initial entry is supported at the demonstrated open-finger home state: arm qpos approximately [0,-0.3,0,-2.1,0,1.8,0.7854], TCP z 0.442 m, red/blue z 0.060/0.020 m. Blue entry also includes a held red at z approximately 0.175-0.281 m over the red pad, fingers approximately 0.0183 m each and downward arm motion. The entire red completion, release and retreat sequence is present in the loss. A close TCP-red relationship must not be erased simply because blue is the next intended object.

For takeover partway through a transition, the relevant causal information is qpos/qvel, finger width and opening/closing motion, both object poses and goals, table/support geometry, TCP-object offsets, and changes in those offsets. Open fingers near a supported object suggest an approach or closing transition; closed fingers plus increasing object height and coherent TCP/object displacement support an inferred grasp and lift. Closed fingers while red is over its pad with downward motion instead support finishing placement. Red at goal height with fingers opening and an increasing TCP-red separation supports release/retreat, but geometry alone cannot prove contact or successful release. Short history and qvel aid direction discrimination without requiring a reset to a boundary pose.

Useful late exits have the selected block following TCP, a maintained pinch (roughly 0.0365 m total finger width), and transport-ready height. The measured late bands are roughly red 0.308-0.318 m and blue 0.280-0.285 m, with nonzero arm velocity and lateral motion allowed. The approximately -9 mm world-x TCP-to-block-center offset seen in these grasps is available to learn; exact TCP/object coincidence is not imposed. Earlier handoff is possible within the shared aligned open-descent, closing and lifting bands, provided the successor continues the actual phase rather than assuming a completed grasp. Continued closed-command motion during late transfer is demonstrated. These bands are support examples, not hard tolerances or a guarantee of inferred attachment.

The successor should receive both causal states and both objects/goals, plus the external caller's semantic selected/previous identity if its own interface supports it. This policy does not output a certified held-object bit or a selected-object tag. The inference agent should convey its task intent separately and distinguish held from merely selected using observational evidence. Failure signatures include TCP rising without selected-object rise, disturbance of blue during red separation, or transporting red away from its pad during the nominal blue transition. No recovery policy for those cases is established by the training data.

Task completion is unchanged: red_at_goal AND blue_at_goal under the supplied rotated-XY containment and z-tolerance contract. No extra finger release, TCP-clearance, zero-velocity or sustained-hold requirement is introduced. Release and clearance discussed here are demonstrated transition behaviors and useful handoff context, not new task predicates.

## Limitations and scientific interpretation

This state-based graph cannot establish exact translation equivariance of joint targets, color interchangeability, collision avoidance, grasp success, support preservation or Panda reachability. The identity one-hot and global features intentionally prevent claims of full color/robot-base invariance. Quaternion sign invariance is the only exact representation invariance claimed. Source XY is observed only around x=-0.361 to -0.339 m, y=-0.011 to +0.012 m; orientations are almost fixed. No failed grasps, toppled stacks or wrong-order attempts support recovery claims. At nearly stationary dwell states, two observations can be insufficient to determine expert timing; the diffusion model learns the supported conditional action distribution rather than a hidden timer.

The comparison suggested by the assigned hypothesis remains an equal-capacity flat-state encoder, then removing support-height and previous-object relational context. Those are proposed tests, not experiments performed by this package. A useful benefit would be reduced grasp error across source translations and the two pickup heights, not just reduced epsilon loss. The current deliverable implements the learned relational encoder and fixed diffusion training only; no final rollout performance is asserted.

Dependencies are Torch and the frozen appl.public DiffusionBackbone/epsilon_loss numerical helpers. No image encoders, external kinematics, solvers, datasets, filesystem access or runtime source-index lookup are needed by policy.py.
