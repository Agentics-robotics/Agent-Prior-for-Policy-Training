# Shared held-spatula interaction with full causal and outcome context

Given a retained spatula and the current egg/pan state, choose short tool motions that establish support, turn/release the egg into the pan, and expose the outcome for verification.

## Dataset rationale

The learned responsibility is continuous feedback-conditioned interaction, not a sequence of independently trained phase policies. Materialize each entire episode separately for initialization, perception annotation, attachment estimation, history and outcomes. Begin dense control anchors only at the inspected retained-tool handoff s. Normal supervision stops three rows before the source end so i+3 targets are real. For three intervention tails use an earlier clean last anchor e, retain the rest as context only, and never use those humans as recovery targets. Do not cut repeated insertion/withdrawal adjustments into disconnected successful-looking snippets. Generic acquisition motion remains unsupervised context in this group.

## Component responsibility: learned_policy

egg_interaction_v1 supplies task-dependent short tool motion to geometry_execution.track_local. Entry requires a retained known spatula, a validated T_ee_blade, identified egg/pan, a task-session initial-face reference and a usable fresh scene observation. It may start in the demonstrated lifted-tool approach envelope or a supported intermediate interaction state; it is not a dropped-tool recovery policy. At 10 Hz return mean motion u[6], uncertainty[6], validity and evidence/status references. u consists of average blade-origin linear velocity in its current local axes (m/s) and a local rotation-vector rate (rad/s), committed for at most 0.10 s. Decode as described in the heuristic and ask the executor to track that geometry; the policy does not issue joint velocity, motor torque or gripper commands. Hold the validated grasp setting. The executor supplies actual arrival/error/constraint feedback; every bounded step is followed by reobservation. A path/timing rejection, grip change, significant scene shift or interruption invalidates the old output. The Agent can request hold, new observation or safe empty-tool retreat. Release and apparent face change are observations, not model-certified success. Additional contact-safe tracking and perception/calibration dependencies are deployment gates, not already supplied abilities.

### interaction_01

Source: `episode_2026091916095801` [0, 1811). Supervision kind: dense; bounds: 600, 1808; decision indices: [].

**Boundary evidence indices**

[0, 600, 1807, 1810]

**Condition evidence indices**

[600, 900, 1200, 1650]

**Decision indices**

[]

**Deployment condition source**

Current retained tool, observed egg/pan and session initial-face goal.

**Label derivation**

Use conversion L at each anchor i with future original row i+3; [0,600) is initialization/acquisition context and [1808,1811) target/outcome buffer.

**Merge check**

Share with all other held-tool episodes, not acquisition control loss.

**Objective**

Approach, insert, lift, release and clear using local tool motion.

**Rationale**

600 is lifted tool; later views show blade beside/under egg, supported lift and egg back in pan.

**Split check**

Keep adjustments and release continuous; no artificial phase labels.

**Supervision exclusions**

[]

**Supervision kind**

dense

**Training condition**

Held-tool control only; causal initial goal from row 0; masks/geometry derived online-equivalently.

**Uncertainty**

Outcome appears changed but no verified task-success label; calibration/contact validity is separately audited.

### interaction_02

Source: `episode_2026091916151401` [0, 2060). Supervision kind: dense; bounds: 600, 2057; decision indices: [].

**Boundary evidence indices**

[0, 600, 2056, 2059]

**Condition evidence indices**

[600, 2056]

**Decision indices**

[]

**Deployment condition source**

Retained tool, current images/geometry and persistent flip goal.

**Label derivation**

L; prefix [0,600) is context, last three rows are future-target/outcome buffer.

**Merge check**

Same local-output contract across episodes.

**Objective**

Feedback-conditioned held-tool interaction.

**Rationale**

600 shows held tool; successive inspected views show prolonged alignment then support and release.

**Split check**

Keep alignment and manipulation in one causal sequence.

**Supervision exclusions**

[]

**Supervision kind**

dense

**Training condition**

Same tool/pan setup, goal turn_over_in_same_pan.

**Uncertainty**

Final gripper command changes without immediate width change; no learned gripper target.

### interaction_03

Source: `episode_2026091916165401` [0, 1912). Supervision kind: dense; bounds: 600, 1909; decision indices: [].

**Boundary evidence indices**

[0, 600, 1908, 1911]

**Condition evidence indices**

[600, 1908]

**Decision indices**

[]

**Deployment condition source**

Observed retained tool and tracked interaction progress.

**Label derivation**

L; pre-600 rows context, final three rows target/outcome evidence.

**Merge check**

Pool with equivalent blade-relative decisions.

**Objective**

Approach and turn interaction with local feedback.

**Rationale**

Held tool at 600, angle adjustment beside egg around 800-1000, later support/lift.

**Split check**

Do not separate tilt corrections from insertion.

**Supervision exclusions**

[]

**Supervision kind**

dense

**Training condition**

Causal scene/goal conditioning, gripper held.

**Uncertainty**

No precise support/contact labels supplied.

### interaction_04

Source: `episode_2026091916183001` [0, 1688). Supervision kind: dense; bounds: 400, 1685; decision indices: [].

**Boundary evidence indices**

[0, 400, 1684, 1687]

**Condition evidence indices**

[400, 1684]

**Decision indices**

[]

**Deployment condition source**

Actual held-tool state, not fixed episode duration.

**Label derivation**

L; pre-400 context and final-three target buffer.

**Merge check**

Earlier handoff is a variation of the same interface.

**Objective**

Held-tool local approach and manipulation.

**Rationale**

Faster acquisition; retain all subsequent insertion adjustments and release.

**Split check**

One coherent feedback interaction despite shorter duration.

**Supervision exclusions**

[]

**Supervision kind**

dense

**Training condition**

Known tool held; goal persists from initial scene.

**Uncertainty**

Visual outcome is evidence, not annotated success.

### interaction_05

Source: `episode_2026091916200101` [0, 2580). Supervision kind: dense; bounds: 650, 2577; decision indices: [].

**Boundary evidence indices**

[0, 650, 2576, 2579]

**Condition evidence indices**

[650, 1400, 1700]

**Decision indices**

[]

**Deployment condition source**

Current blade/egg relation and recent causal history.

**Label derivation**

L; prefix context, final-three target buffer; retain adjustment attempts without success filtering.

**Merge check**

Same physical decision even during renewed insertion.

**Objective**

Adjust insertion, establish support and execute release motion.

**Rationale**

1400 blade crosses above egg, 1700 returns beside it, later insertion/lift follows.

**Split check**

Splitting at the adjustment would hide recovery context and create a false mandatory phase.

**Supervision exclusions**

[]

**Supervision kind**

dense

**Training condition**

Shared held-tool policy including imperfect approaches.

**Uncertainty**

Exact failed-attempt count and contact state are unannotated.

### interaction_06

Source: `episode_2026091916223301` [0, 2102). Supervision kind: dense; bounds: 600, 2099; decision indices: [].

**Boundary evidence indices**

[0, 600, 2098, 2101]

**Condition evidence indices**

[600, 2098]

**Decision indices**

[]

**Deployment condition source**

Tool retention and current tracked egg/pan.

**Label derivation**

L; pre-600 context, final-three future buffer.

**Merge check**

Same tool-relative output and goal.

**Objective**

Feedback-controlled insertion through release.

**Rationale**

Extended support acquisition and changed pan/egg placement are retained.

**Split check**

Keep interaction history through release.

**Supervision exclusions**

[]

**Supervision kind**

dense

**Training condition**

Known tool held and causal flip goal.

**Uncertainty**

Pan displacement must be tracked online, not assumed zero.

### interaction_07

Source: `episode_2026091916244201` [0, 2200). Supervision kind: dense; bounds: 600, 2197; decision indices: [].

**Boundary evidence indices**

[0, 600, 2196, 2199]

**Condition evidence indices**

[600, 2196]

**Decision indices**

[]

**Deployment condition source**

Observed relative blade/egg/pan state.

**Label derivation**

L with real i+3 target rows; prefix and final-three context as above.

**Merge check**

Share across changed egg position.

**Objective**

Held-tool approach, insertion, lift and release.

**Rationale**

Egg initially toward the pan edge; local geometry matters more than episode timing.

**Split check**

Preserve one continuous manipulation episode.

**Supervision exclusions**

[]

**Supervision kind**

dense

**Training condition**

Same known-tool task; no future phase input.

**Uncertainty**

Edge placement does not establish arbitrary geometry transfer.

### interaction_08

Source: `episode_2026091916283401` [0, 1914). Supervision kind: dense; bounds: 400, 1911; decision indices: [].

**Boundary evidence indices**

[0, 400, 1910, 1913]

**Condition evidence indices**

[400, 1910]

**Decision indices**

[]

**Deployment condition source**

Lifted tool and current object observations.

**Label derivation**

L; prefix [0,400) context, final-three buffer.

**Merge check**

Early handoff is within the shared approach envelope.

**Objective**

Choose local motions from early lifted-tool state through release.

**Rationale**

Larger visible egg extent and different pan placement provide observed variation.

**Split check**

Keep free approach and contact transition observable rather than pretending arrival is contact certainty.

**Supervision exclusions**

[]

**Supervision kind**

dense

**Training condition**

Causal observation and persistent same-pan flip goal.

**Uncertainty**

Object identity/material not verified from changed appearance.

### interaction_09

Source: `episode_2026091916295601` [0, 3346). Supervision kind: dense; bounds: 400, 3343; decision indices: [].

**Boundary evidence indices**

[0, 400, 3342, 3345]

**Condition evidence indices**

[400, 1400, 1800]

**Decision indices**

[]

**Deployment condition source**

Causal support/geometry and history, not progress inferred from elapsed time.

**Label derivation**

L; preserve long adjustment history; final-three buffer and acquisition prefix carry no control loss.

**Merge check**

No distinct output type justifies a separate retry policy.

**Objective**

Repeated local insertion adjustment, lift and release.

**Rationale**

Long duration contains substantial blade-angle/support adjustment, not merely longer transit.

**Split check**

Do not cut out low-progress spans or merge disjoint attempts as separate successful rollouts.

**Supervision exclusions**

[]

**Supervision kind**

dense

**Training condition**

Shared policy with episode-balanced weighting, including slow progress.

**Uncertainty**

No ground-truth contact, intent or successful-retry count.

### interaction_10

Source: `episode_2026091916320101` [0, 2773). Supervision kind: dense; bounds: 1000, 2581; decision indices: [].

**Boundary evidence indices**

[0, 1000, 2580, 2583, 2772]

**Condition evidence indices**

[1000, 2580, 2650]

**Decision indices**

[]

**Deployment condition source**

Online retained-tool state and task observations before human involvement.

**Label derivation**

L through i=2580, last target=2583. [0,1000) is context; [2581,2773) is target/outcome/intervention evidence only. No target is drawn from the human-touch frames at 2650 onward.

**Merge check**

Same interaction model; human-assisted tail must not enter autonomous control loss.

**Objective**

Manipulate and release before intervention.

**Rationale**

At 2550-2600 egg is in pan and tool is clear; at 2650 a person touches the egg/pan.

**Split check**

Keep original tail for audit, but cap supervision conservatively instead of treating assistance as a robot retry.

**Supervision exclusions**

[]

**Supervision kind**

dense

**Training condition**

Autonomous-target interval ends before observed intervention; outcome remains unverified.

**Uncertainty**

Human tail cannot establish the robot-only final outcome; exact intervention onset is not a trained label.

### interaction_11

Source: `episode_2026091916335001` [0, 2899). Supervision kind: dense; bounds: 600, 2896; decision indices: [].

**Boundary evidence indices**

[0, 600, 2895, 2898]

**Condition evidence indices**

[600, 2895]

**Decision indices**

[]

**Deployment condition source**

Current observed geometry and causal adjustment history.

**Label derivation**

L; prefix context and final-three target buffer.

**Merge check**

Extended corrections remain the same local-output problem.

**Objective**

Held-tool adjustments and release.

**Rationale**

Blade revisits different egg-relative positions before support and lift.

**Split check**

Avoid artificially selecting only late successful-looking motion.

**Supervision exclusions**

[]

**Supervision kind**

dense

**Training condition**

Known tool and persistent same-pan goal.

**Uncertainty**

Appearance change is not an annotated success label.

### interaction_12

Source: `episode_2026091916411901` [0, 2141). Supervision kind: dense; bounds: 600, 2051; decision indices: [].

**Boundary evidence indices**

[0, 600, 2050, 2053, 2140]

**Condition evidence indices**

[600, 2050, 2140]

**Decision indices**

[]

**Deployment condition source**

Observed held tool and front-handle pan layout; cleared workspace before intervention.

**Label derivation**

L through 2050, target 2053. Prefix is context; [2051,2141) is target/outcome/intervention evidence only.

**Merge check**

Share policy across pan headings; exclude human pan repositioning from control loss.

**Objective**

Interact and release in the changed layout.

**Rationale**

At 2050/2053 egg rests in pan; at 2100/2140 a human repositions the pan.

**Split check**

Keep tail for audit without training robot recovery from human motion.

**Supervision exclusions**

[]

**Supervision kind**

dense

**Training condition**

Held-tool interaction before the human-intervention tail.

**Uncertainty**

Do not use final pan pose or final appearance as autonomous success ground truth.

### interaction_13

Source: `episode_2026091916445601` [0, 2179). Supervision kind: dense; bounds: 600, 2176; decision indices: [].

**Boundary evidence indices**

[0, 600, 2175, 2178]

**Condition evidence indices**

[600, 2175]

**Decision indices**

[]

**Deployment condition source**

Current tool/egg/pan observations in moved-pan layout.

**Label derivation**

L; pre-600 context, final-three target/outcome buffer.

**Merge check**

Same local coordinates despite changed global pan location.

**Objective**

Approach and manipulate with geometry-conditioned feedback.

**Rationale**

Pan position and handle heading differ; pan also moves during interaction.

**Split check**

Keep pan-motion context with local action supervision.

**Supervision exclusions**

[]

**Supervision kind**

dense

**Training condition**

Tracked rather than fixed pan frame.

**Uncertainty**

No assumption that the pan is rigidly anchored to the table.

### interaction_14

Source: `episode_2026091916463201` [0, 2361). Supervision kind: dense; bounds: 600, 2251; decision indices: [].

**Boundary evidence indices**

[0, 600, 2250, 2253, 2360]

**Condition evidence indices**

[600, 2250, 2360]

**Decision indices**

[]

**Deployment condition source**

Retained tool and observed task state before assistance.

**Label derivation**

L through 2250 with target 2253; [2251,2361) is outcome/intervention context, never control loss.

**Merge check**

Do not merge human touching the tool into learned manipulation.

**Objective**

Held-tool interaction through release and initial clearance.

**Rationale**

2250/2253 show egg resting and empty tool clear; final frame shows a human at the spatula.

**Split check**

Retain the intervention for failure/handoff audit, not as another autonomous phase.

**Supervision exclusions**

[]

**Supervision kind**

dense

**Training condition**

Same-pan goal with fixed grasp setting and monitored attachment.

**Uncertainty**

Do not derive attachment or success labels from the human-affected final frames.

### interaction_15

Source: `episode_2026091916480201` [0, 2004). Supervision kind: dense; bounds: 900, 2001; decision indices: [].

**Boundary evidence indices**

[0, 900, 2000, 2003]

**Condition evidence indices**

[900, 2000]

**Decision indices**

[]

**Deployment condition source**

Actual retained-tool arrival, not a shared start index.

**Label derivation**

L; prefix [0,900) context and final-three target buffer.

**Merge check**

Variable pickup duration does not justify a new interaction model.

**Objective**

Held-tool approach through release.

**Rationale**

900 is the inspected lifted-tool approach; later egg support/lift and release are retained.

**Split check**

Keep local continuous interaction; prefix remains available for goal/geometry initialization.

**Supervision exclusions**

[]

**Supervision kind**

dense

**Training condition**

Same geometry-conditioned task after slow pickup.

**Uncertainty**

Final tool near dish is not a demonstrated verified tool-placement/release skill.

### interaction_16

Source: `episode_2026091916505401` [0, 2399). Supervision kind: dense; bounds: 600, 2396; decision indices: [].

**Boundary evidence indices**

[0, 600, 2395, 2398]

**Condition evidence indices**

[600, 2395]

**Decision indices**

[]

**Deployment condition source**

Current central-pan layout and held-tool state.

**Label derivation**

L; prefix context, final-three target buffer.

**Merge check**

Share across pan translations/headings.

**Objective**

Held-tool feedback interaction.

**Rationale**

Different pan layout with prolonged blade alignment before support.

**Split check**

Keep alignment and insertion corrections causal.

**Supervision exclusions**

[]

**Supervision kind**

dense

**Training condition**

Known-tool, same-pan flip goal; no episode-time conditioning.

**Uncertainty**

Held-out layout performance is unknown.

### interaction_17

Source: `episode_2026091916522501` [0, 2011). Supervision kind: dense; bounds: 600, 2008; decision indices: [].

**Boundary evidence indices**

[0, 600, 2007, 2010]

**Condition evidence indices**

[600, 2007]

**Decision indices**

[]

**Deployment condition source**

Tool retention and causal observed egg support.

**Label derivation**

L; prefix and final-three buffers carry no control loss.

**Merge check**

Same policy for shorter/longer attempts.

**Objective**

Local tool interaction and clearance.

**Rationale**

Egg is visibly back in pan by the inspected 1800 view; subsequent clearance remains useful.

**Split check**

No need to split off an independent clearance policy; planner can take over once the Agent verifies empty tool.

**Supervision exclusions**

[]

**Supervision kind**

dense

**Training condition**

Persistent initial-face goal and tracked pan.

**Uncertainty**

Apparent end-state is not a verified success annotation.

### interaction_18

Source: `episode_2026091916534901` [0, 2903). Supervision kind: dense; bounds: 1000, 2900; decision indices: [].

**Boundary evidence indices**

[0, 1000, 2899, 2902]

**Condition evidence indices**

[1000, 1800, 2200]

**Decision indices**

[]

**Deployment condition source**

Observed blade/egg relation after a variable-duration acquisition.

**Label derivation**

L; long prefix is context only, final-three target buffer. Preserve recorded camera timestamp reset without realignment.

**Merge check**

Readjustment at 1800 and later insertion at 2200 belong to the shared decision problem.

**Objective**

Repeated adjustment and release-producing local motion.

**Rationale**

Contains early idle time plus substantial held-tool readjustment, which must not become an episode clock.

**Split check**

Do not split retries or remove pauses solely to make a monotonic phase sequence.

**Supervision exclusions**

[]

**Supervision kind**

dense

**Training condition**

Causal same-pan task and image freshness validity.

**Uncertainty**

Third-camera device timestamps change scale/reset near the tail; sample pairings remain original.

### interaction_19

Source: `episode_2026091916553801` [0, 1834). Supervision kind: dense; bounds: 400, 1831; decision indices: [].

**Boundary evidence indices**

[0, 400, 1830, 1833]

**Condition evidence indices**

[400, 1550, 1580, 1610]

**Decision indices**

[]

**Deployment condition source**

Retained tool with current observed support/release evidence.

**Label derivation**

L; prefix context and final-three future buffer. Release views are outcome evidence, not privileged phase input.

**Merge check**

Same six-coordinate output spans lift and release; no 180-degree wrist-flip primitive is imposed.

**Objective**

Held-tool approach, support and release with immediate feedback.

**Rationale**

1550 shows egg on blade, 1580 an airborne/tilted egg, 1610 egg resting in pan; this motivates local release timing rather than fixed tool rotation.

**Split check**

Preserve the supported-to-airborne-to-pan transition without discontinuous phase cuts.

**Supervision exclusions**

[]

**Supervision kind**

dense

**Training condition**

Goal is opposite initial face in pan, not merely an empty spatula.

**Uncertainty**

Physical flip inference is visually supported but not a supplied verified success label; dynamics/material unmeasured.

### interaction_20

Source: `episode_2026091916583301` [0, 2613). Supervision kind: dense; bounds: 800, 2610; decision indices: [].

**Boundary evidence indices**

[0, 800, 2609, 2612]

**Condition evidence indices**

[800, 2609]

**Decision indices**

[]

**Deployment condition source**

Current lifted tool, egg/pan tracks and original-face goal.

**Label derivation**

L; [0,800) is context, final-three rows target/outcome buffer.

**Merge check**

Pool repeated insertion/tilt structure with the other episodes.

**Objective**

Held-tool interaction through release and clearance.

**Rationale**

800 clearly held tool; later views show insertion, support and empty-tool/egg-in-pan outcome.

**Split check**

Keep continuous progress and adjustments; do not force other episodes' phase timing.

**Supervision exclusions**

[]

**Supervision kind**

dense

**Training condition**

Known-tool task with causal image/geometry inputs.

**Uncertainty**

No verified episode success or material labels; final clearance is not tool placement.


## Heuristic 1

Learn one causal, blade-relative short-motion policy for all held-spatula interaction progress; delegate robot motion generation, known-tool acquisition and safety enforcement to explicit executor adapters.

### Action decoding

Let T_base_B(i)=[R_i,p_i] be the current blade frame, with origin at the centre of the blade's leading edge, x along the blade from handle toward that edge, z the blade-face normal, and y completing a right-handed frame. For u=[v_B,omega_B] and h=0.10 s, p_goal=p_i+R_i v_B h; R_goal=R_i Exp(omega_B h). Convert the blade goal to EE and flange goals using the current validated rigid T_ee_B and T_flange_ee. The supplied executor generates feasible connecting joint motion. This is not a global spatial-twist exponential about the robot base. In contact, demand the corresponding short straight/rotational interpolation with bounded compliance and timing fidelity; reject rather than silently reroute or materially retime. Apply tool/robot speed, displacement, rotation, rim, workspace and force-limit checks from the verified adapter. Out-of-distribution/uncertain goals produce no motion. No gripper action is learned: retain the established tool grasp until the Agent separately commands release in a safe location. If the achieved attachment changes, do not apply the old blade transform.

### Applicability

Same known spatula and receiving pan geometry, grasp compatible with the calibrated tool model, egg object initially in the pan or visibly supported on the blade during a continuing attempt. Covers final free approach, angle and insertion adjustments, withdrawal/reposition within the pan, lifting a supported object, release-producing motions and immediate clearance. Does not assume an obligatory phase clock or a full 180-degree wrist rotation. Exclude autonomous deployment on a dropped/off-table egg, unknown tool, unverified attachment, obstructed workspace, missing calibration or unreliable observations.

### Augmentation

Use photometric brightness/contrast/white-balance and background perturbations consistently across a history; preserve egg-face distinctions rather than randomizing them away. Camera crops/resizes transform masks, keypoints, intrinsics and pixel mappings together. Small planar translation/yaw augmentation is applied jointly to reconstructed scene geometry, caller pan interior, tool/egg geometry and labels, with projective rendering/warping only for features supported on the relevant plane; do not pretend lifted objects lie on the pan plane. A common base-frame rigid coordinate change must transform all camera extrinsics, robot/tool/scene poses and goals together; local blade labels then remain invariant. Do not rotate q/dq numerically as Cartesian vectors: retain them only for pure coordinate relabeling of the same robot state, or omit physical-scene augmentation requiring unavailable IK. No arbitrary scaling of metric tools, mirroring of tool handedness, phase reversal, goal relabeling to arbitrary landing points or temporal speed changes around release. Add calibrated pose/keypoint noise with validity/covariance, camera dropout and history truncation so the model learns abstention rather than substituting privileged poses. Any augmentation producing infeasible geometry is rejected.

### Evidence

- `episode_2026091916095801`: [600, 603, 900, 1200, 1650, 1810]
- `episode_2026091916200101`: [650, 1400, 1700, 2579]
- `episode_2026091916295601`: [400, 1400, 1800, 3345]
- `episode_2026091916411901`: [600, 2050, 2140]
- `episode_2026091916534901`: [1000, 1800, 2200, 2902]
- `episode_2026091916553801`: [400, 1520, 1550, 1580, 1610, 1833]
### Goal conditioning

Input the session's turn_over_in_same_pan goal, selected egg/pan IDs, causal initial-face reference crop [3,64,64], and current permitted pan interior. The initial visible face is the reference, not an absolute food-side class. Do not input future support/phase/success annotations. The desired outcome persists across observed retries. Unsupported arbitrary landing-point commands are rejected at the calling interface.

### Handoff

**entry_conditions**

Executor reports retained known tool; images show consistent tool co-motion, attachment fit passes uncertainty checks, egg/pan are identified and the workspace is cleared. Initialize from recent causal observations, not a stored demonstration pose.

**exit_conditions**

Return bounded local proposals until the Agent observes release, empty-tool clearance and an inspectable resting egg, or until an invalidity/failure requires arbitration. Verified flip completion requires Agent confirmation of opposite-face-up support in the pan; recording termination is not an exit label.

**failure_signatures**

Tool fit changes relative to EE, width departs from validated hold range, no progress over a configured timeout, egg leaves allowed region, pan shifts beyond tracker confidence, support is unobservable, tracking residual exceeds limits, stale image, force/torque limit, human intervention or infeasible motion. These trigger hold/cancel and reobservation, not an untrained recovery action.

**overlap_role**

The full prefix supplies initial-face reference, perception/grasp calibration and causal history without control loss. The handoff row is shared with executor evidence. Final three rows, or longer intervention tails, are targets/outcomes only. Source reuse never supplies future inputs.

**successor_readiness**

Provide the Agent fresh third/wrist images, tool attachment and uncertainty, pan/egg tracks, observed support state with unknown allowed, attempted-release latch and executor status. Empty-tool clearance enables an explicit safe retreat; a still-supported egg may permit continued local interaction. Unknown orientation requires inspection, not automatic retry.

### Heuristic id

blade_relative_local_motion

### Input preprocessing

Raw prerequisites: paired third/wrist RGB, T_base_flange, T_base_ee, q[7], measured dq[7], tau_ext[7], gripper_width_m and gripper_position, t/recv_time and image age/time fields. Retain action_json for audit and command-vs-motion checks, not as a causal teacher input; retain raw depth/media but do not use depth metrically without a separate scale/registration validation. Load each episode's own metadata. Inspected metadata for 16095801, 16411901 and 16583301 have the same recorded calibration, TCP translation [0,0,0.24] m and yaw -90 degrees; this TCP is not a measured spatula-tip transform. Recorded calibration reports about 3.63 mm pose consistency and 4.56 px reprojection RMS, not guaranteed contact accuracy. The wrist factory and calibrated K/dist differ: use the calibrated K/dist with its calibrated extrinsic, validate on the actual RGB stream, and never mix pairs. T_base_cam_wrist=T_base_flange*T_flange_cam. Preserve raw sample pairings and timestamps; negative recorded image age or device timestamp reset is not repaired by shifting rows. Accept only freshness/geometry validated inputs and report uncertainty; exposure synchronization has not been established.
Required preparation, not existing annotations: manually annotate original training images for pan/egg/tool masks and six known-tool landmarks (leading-edge left/right, blade shoulders left/right, handle end and a distinct handle/shaft landmark) with visibility. Use inspected anchors plus every 60th original image as initial annotation candidates, then add difficult occlusion/insertion/release images after review. This auxiliary annotation set is separate from control-anchor loss, keyed to original source index and episode split. No masks/poses exist yet. Train scene_geometry_aux with class-balanced mask CE+Dice, landmark heatmap loss and visibility BCE; held-out annotations measure its uncertainty. At deployment this model replaces manual labels; the Agent provides initial object selection/ROI, not a future pose. Measure the actual rigid spatula landmark coordinates, collision mesh, jaw geometry, pan radius/interior surface height and table plane before implementation. Fit the known tool by calibrated PnP/multiview reprojection with ambiguity checks. Fit the pan's rim/centre on its measured support plane; do not substitute the old calibration-board plane for the pan surface. Define pan frame P with origin at its fitted centre, z the measured upward normal, x projected base x and y right-handed. Thus pan-handle heading is an obstacle cue, not an artificial required policy orientation. Estimate T_ee_B from visible post-grasp tool observations, lock it while its residuals remain consistent and invalidate on slip. When tool visibility is lost briefly, propagate only with kinematics and increased uncertainty. No future-smoothed pose may be a training input.
The egg mask yields image centroid/extents. Only intersect its rays with the pan plane when observed to be pan-supported, or the current blade plane when visibly supported there; otherwise its metric position is unknown, not a fabricated planar point. Keep 2D mask evidence for airborne/occluded cases. Support is a conservative geometric/temporal observation with an unknown value, not a force-contact label. Use both views without assuming exposure-level simultaneity; high reprojection disagreement during motion requests stationary reobservation.
Policy tensors: five causal observations, nominal offsets 0,-0.1,-0.2,-0.3,-0.4 s chosen from existing earlier paired rows, with actual time offsets and validity masks. Each has two object-centred, aspect-preserving letterboxed RGB crops [2,3,224,224], semantic probability maps [2,4,224,224], tool landmarks/visibility [2,6,3], proprioception [22]=q7+dq7+tau7+width1, tool pose in P [9]=translation3+continuous rotation6, egg centroid [3] plus metric-valid bit, planar extents[2], pan radius[1], and covariance/visibility blocks explicitly typed in the later loader. Supply camera age[2], crop-to-original transforms and calibration IDs. Gripper_position is retained for diagnostics rather than treated as force. Normalize metric quantities by fixed physical training scales, not per-episode extrema. Estimate torque baseline from causal stationary free-space data or calibrated setup; do not call tau_ext a measured Cartesian wrench. Before the first usable history, pad with invalid entries, not future frames. Shared small CNN spatial pooling plus geometry MLP feeds the GRU; no elapsed-episode clock is an input.

### Limitations

All performance and invariance claims are untested. Raw camera/robot pairing is not verified exposure synchronization, and future measured displacement can reflect controller lag or stiction rather than operator intent. The action w field must not be blindly decoded using teleop rotation_frame=ee: inspected w vectors vary with EE orientation and may already be transformed by track_twist. This design deliberately derives geometric targets from measured transforms instead; later controller binding must still verify units and timing. Tool geometry and masks must be created, and attachment errors directly corrupt blade targets. An auxiliary model trained on this small setup may fail under occlusion or novel appearance. Unknown success labels preclude claiming all demonstrations succeeded or training a trustworthy binary stop classifier from their endings. Generic geometry tracking is not a validated compliant/contact controller; safe contact binding and test data are mandatory. Material, heat, food compliance, alternative tools, arbitrary entry directions and robust failure recovery are not established.

### Pipeline implications

Target conversion L: for each declared dense anchor i, use only original rows i and i+3 of the same materialized episode, with dt=t[i+3]-t[i]. With the current causal attachment estimate held fixed over this short interval, obtain T_base_B at both robot poses. Label v_B=R_i^T(p[i+3]-p[i])/dt and omega_B=Log(R_i^T R[i+3])/dt. These six rates describe achieved local blade motion, not raw command velocities or contact force. Require finite SE(3), positive dt in an audited nominal range (proposed 0.05-0.20 s), and valid attachment; log rejected labels and reasons without altering raw cuts or substituting zeros. The target row is training-only future evidence. Terminal three-row buffers make the conversion possible at the last normal anchor; special tails cap targets before inspected human intervention. Do not infer stop labels from tiny terminal displacement. Train the actor by mixture negative log likelihood with translation/rotation scales, endpoint Huber/geodesic consistency and weak within-mode temporal regularization; never average incompatible insertion and withdrawal modes. A soft blade-axis/tilt regularizer encourages the observed low-dimensional insertion structure but does not hard-zero lateral motion, roll or yaw. Use episode-balanced sampling and cap static/near-zero windows at 10% of each minibatch; retain their exact rows. Sample adjustment-heavy and release windows rather than discarding unsuccessful-looking approaches. Repeated commands at 30 Hz are not independent 20 Hz decisions. Train with truncated BPTT over 2 s, causal warm-up up to 1 s inside the materialized episode, and random history truncation. Reset memory at original episode boundaries and actual reacquisition/interruption, not at arbitrary phase times. Auxiliary scene annotations are prepared from the same retained media under the same split; manual labels and future outcome audits are never runtime actor inputs. Publish annotation versions, fitted geometry, target validity counts and calibration uncertainty before training. Do not run code or training in this cut stage.

### Policy contract

**caller_arguments**

session_id; task=turn_over_in_same_pan; egg_track_id; pan_track_id; spatula_template_id; initial_face_reference; allowed_pan_interior; validated T_ee_B; geometry/calibration versions; safety envelope; fresh paired observations; reset/resume flag. No phase chosen from episode time and no supplied future success.

**input_output_contract**

Causal tensors/geometry are specified in input_preprocessing. Return u[6] in current blade axes (m/s and rad/s), predicted spread[6], validity/reason, support/visibility evidence references, proposed horizon<=0.10 s, and current observation ID. The selected Gaussian mode is one consistent motion proposal, not an average of modes. The executor must reject expired or infeasible proposals. No joints, gripper commands, force targets or guaranteed flip flag are returned.

**memory_and_handoff**

Maintain a 128-state GRU and short causal frame buffer within a session; the Agent separately retains the original-face goal and an observed release-attempt latch. Initialize from recent actual observations on entry; missing history is masked. Continuous adjustments reuse state. After interruption, slip, object reassignment or new grasp, clear motion memory and warm up from fresh causal observations, while preserving the original task goal unless the Agent explicitly starts a new task.

**selection_cues**

Use when a known spatula is visibly retained and the egg/pan are observable in the demonstrated workspace; it handles the transition from lifted-tool approach through interaction progress. Use geometry_execution alone for calibrated acquisition and empty-tool free-space moves. Unknown geometry/support, a dropped tool or an off-pan object requests assistance rather than a different untrained phase policy.

**status_and_progress**

At each 10 Hz call provide active or invalid with reason, uncertainty, track visibility and observed support evidence. Executor feedback reports local goal progress. The Agent monitors mask/pose changes, support acquisition, release and settled appearance; no-progress timeout and repeated rejected goals stop the cycle. completed_flip is issued only by the Agent's explicit visual verification against the initial reference, with unknown allowed. The scene model is not a safety-rated human detector.

### Policy id

egg_interaction_v1

### Rationale

Observed regularity is a rigid gripped spatula with nearly constant gripper feedback, repeated insertion-angle adjustments, supported lifting and a short release event, across changing pan locations/headings. The residual learning problem is what local blade motion to request now, not how to solve arm kinematics or how long an episode should run. Six blade-relative motion coordinates, causal object-centred vision, a pan-relative scene and a held-gripper constraint are an implementable bias that removes predictable robot/camera variation while preserving interaction choices. The episode_2026091916553801 frames 1550/1580/1610 show that the egg can leave the tool and land after a relatively small tool rotation; a hard-coded 180-degree wrist flip would be the wrong prior. Dense feedback is chosen because support and insertion change within an attempt and are not guaranteed by the supplied planner. Sharing phases avoids brittle handoffs and uses adjustments as data. Expected generalization is within the observed geometry/appearance envelope, pending evaluation.


All scientific text above is unchanged Runtime API output. Rendering and slicing are developer-owned. No policy code or training exists in this stage.
