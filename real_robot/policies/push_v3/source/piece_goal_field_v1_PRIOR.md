# piece_goal_field_v1 — prior

Responsibility: move one persistent flat rigid instance to a physical goal silhouette/transform, including approach, reversals, contact resets, alignment and requested exit. Dataset `piece_relocation`; exact heuristic `causal_goal_field_servo`. [Usage](piece_goal_field_v1_USAGE.md), [shared science](PRIOR.md), [calling](CALLING.md), [handoff](HANDOFF.json).

## Evidence and independence

Independently trained on the identical 22 invocation groups as the graph alternative, not its weights or recurrent state. A=`episode_2026091922002701`, B=`episode_2026091922062101`.

A supervised ranges: [600,1740) wedge; [1740,1990) barred clearing; [1990,3220) ring; [3220,3700) R staging; [3700,4300) barred staging; [4300,4640) upper barred; [4640,5200) D staging; [5200,5670) R layout; [5670,6120) D alignment; [6120,8150) repeated L recontacts/transport; [8150,8510) barred alignment; [8510,9015) upper-bar alignment/retract/hold.

B supervised ranges: [380,1240) wedge; [1240,1500) barred clearing; [1500,1950) D staging; [1950,2850) ring; [2850,4700) repeated R relocation with reset; [4700,4950) D alignment; [4950,6150) L; [6150,6910) upper barred; [6910,7370) barred alignment; [7370,7730) upper-bar alignment/retract/hold.

Original context buffers, overlap provenance and source endpoints are unchanged. Final zero commands are real labels, not a success classifier. All finite v/w including nonzero w train even if command dq is null. Different orders across episodes are evidence against hardcoded spelling sequences. There are no supplied character masks/identities, contact labels, desired words or verified success labels.

## Dense representation and servo

Calibrated plane-remapped 256-square geometry feeds a **native-resolution 64-square local crop (0.25 m)**, plus coarse whole-scene occupancy. Eight channels: selected signed distance, desired signed distance, other-instance occupancy, TCP heatmap, x/y rigid current-to-goal correspondence, coarse global occupancy and goal validity. Holes are represented in SDF rather than filled convex hulls. Full metric displacement/scale, robot pose/motion/missingness, proxy height, exit error, symmetry and confidence remain low-dimensional features. New shapes are measured silhouettes, not learned font templates.

Rigid registration is a symmetry-aware grid fit; it does not declare a unique canonical object yaw. Near-equal symmetric fits choose a small-angle correspondence. Directional semantic upright remains a separate live completion requirement, not a learned character embedding. Local crops retain small geometry; far-away goals are still expressed by flow/state. Holes near/below the 3.9-mm sampling and 3x3 foreground-close scale are not reliable.

A compact convolutional encoder and state MLP feed a masked GRU over snapshots [11,7,3,0], then five-mode speed/residual heads and native command uncertainty. A deterministic geometry-to-boundary/exit route supplies a constrained anchor, but **there is no learned contour graph or response-based contact selection** in this policy. The dense network learns corrective servo/mode behavior and a motion auxiliary. It is a substantive alternative, not another name for the graph model. Approx. two million independent parameters, batch 32, AdamW 1.5e-4, weight decay 1e-4, norm clip 1, fixed 20k/EMA schedule.

Conditional sequence diffusion was considered for repeated L/R strokes. With two episodes and uncertain weak geometry, a low-degree-of-freedom causal recurrent inverse servo around measured goal fields is more constrained and easier to diagnose. Reset/exit modes avoid an inappropriate whole-demo monotonic-progress objective. No limitation of the executor's one-step prediction hook forced this choice.

## Training-only goals and losses

Each accepted last-supervised endpoint track becomes a fixed achieved goal mask and separate TCP/terminal-command label, not verified original intent. Future displacement supplies only a masked actual-successor motion label. Inference builds the same maps from causal live tracking and caller goals; no endpoint image, source ID, future motion or timestamp-in-episode is an online input.

Huber native-action cloning plus small heteroscedastic NLL, phase CE, residual regularization and confidence-masked motion auxiliary train real network outputs. No unverified physical action-frame augmentation is used. Native linear components are numerically normalized in a reversible local basis; serialized w remains as recorded. Geometry/goal/motion uncertainty is explicit. Missing endpoint masks do not drop difficult actions: those rows retain endpoint-tool-conditioned command/mode imitation with missing object-goal flags, while confident object-goal/motion losses are ineligible. Full preprocessing publishes this per segment, phase and target handle; it is not claimed already audited.

## Adaptation and limits

Shared shape-independent HSV foreground/dual-view local repair and causal tracking are implemented because no perception weights or annotations were supplied. Initial seeds only identify offline physical handles. Board-plane proxy and recorded TCP do not certify physical top height, rod tip/radius or clearance. Current caller-provided union masks/verified plane can replace the uncertain front end through the same executable map conversion. Semantic hypotheses/axes remain upstream real inputs; absent assets cause uncertainty, not invented recognition.

Dense masks may be useful when canonical orientation or local mechanics is uncertain, but this is an agent selection cue, **not a measured ranking**. Shared mask/calibration errors correlate the alternatives. Incorrect holes, correspondence, friction or identity can yield ineffective pushes even when a planar collision screen passes. No complete occlusion recovery, autonomous jam extraction, semantic success, new-font transfer, force measurement or full arm safety is established. The physical decoder remains disabled; candidates and uncertainty are usable offline while missing verified hardware assets are explicit.
