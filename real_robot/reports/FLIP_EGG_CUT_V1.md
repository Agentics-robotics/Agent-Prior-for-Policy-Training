# flip_egg: API segmentation and heuristic designs

Review date: 2026-09-21. Completed stage: cut/grouping and heuristic documents only.
No policy code, preprocessing implementation, training, or physical robot execution has occurred.

Model: `gpt-6-astra`; reasoning: `xhigh`; verified API calls: 79.
Reported input/output tokens: 8,689,843 / 27,577. Estimated standard API cost: USD 12.4355 (not an invoice).

## Published datasets

| Dataset | Original segments | Heuristic designs | Documents |
| --- | ---: | ---: | --- |
| geometry_execution | 20 | 0 | [heuristics](../data/flip_egg/cut_v1/datasets/geometry_execution/heuristic.md), [dataset](../data/flip_egg/cut_v1/datasets/geometry_execution/dataset.json) |
| held_spatula_interaction | 20 | 1 | [heuristics](../data/flip_egg/cut_v1/datasets/held_spatula_interaction/heuristic.md), [dataset](../data/flip_egg/cut_v1/datasets/held_spatula_interaction/dataset.json) |

## Coverage

{
  "episode_2026091916095801": {
    "assigned_unique": 1811,
    "assignment_records": 2412,
    "context_only_unique": 603,
    "context_without_executor_or_supervision_unique": 3,
    "excluded": 0,
    "executor_assignment_records": 601,
    "executor_unique": 601,
    "reused_unique": 601,
    "source_records": 1811,
    "supervised_unique": 1208,
    "supervision_assignment_records": 1208
  },
  "episode_2026091916151401": {
    "assigned_unique": 2060,
    "assignment_records": 2661,
    "context_only_unique": 603,
    "context_without_executor_or_supervision_unique": 3,
    "excluded": 0,
    "executor_assignment_records": 601,
    "executor_unique": 601,
    "reused_unique": 601,
    "source_records": 2060,
    "supervised_unique": 1457,
    "supervision_assignment_records": 1457
  },
  "episode_2026091916165401": {
    "assigned_unique": 1912,
    "assignment_records": 2513,
    "context_only_unique": 603,
    "context_without_executor_or_supervision_unique": 3,
    "excluded": 0,
    "executor_assignment_records": 601,
    "executor_unique": 601,
    "reused_unique": 601,
    "source_records": 1912,
    "supervised_unique": 1309,
    "supervision_assignment_records": 1309
  },
  "episode_2026091916183001": {
    "assigned_unique": 1688,
    "assignment_records": 2089,
    "context_only_unique": 403,
    "context_without_executor_or_supervision_unique": 3,
    "excluded": 0,
    "executor_assignment_records": 401,
    "executor_unique": 401,
    "reused_unique": 401,
    "source_records": 1688,
    "supervised_unique": 1285,
    "supervision_assignment_records": 1285
  },
  "episode_2026091916200101": {
    "assigned_unique": 2580,
    "assignment_records": 3231,
    "context_only_unique": 653,
    "context_without_executor_or_supervision_unique": 3,
    "excluded": 0,
    "executor_assignment_records": 651,
    "executor_unique": 651,
    "reused_unique": 651,
    "source_records": 2580,
    "supervised_unique": 1927,
    "supervision_assignment_records": 1927
  },
  "episode_2026091916223301": {
    "assigned_unique": 2102,
    "assignment_records": 2703,
    "context_only_unique": 603,
    "context_without_executor_or_supervision_unique": 3,
    "excluded": 0,
    "executor_assignment_records": 601,
    "executor_unique": 601,
    "reused_unique": 601,
    "source_records": 2102,
    "supervised_unique": 1499,
    "supervision_assignment_records": 1499
  },
  "episode_2026091916244201": {
    "assigned_unique": 2200,
    "assignment_records": 2801,
    "context_only_unique": 603,
    "context_without_executor_or_supervision_unique": 3,
    "excluded": 0,
    "executor_assignment_records": 601,
    "executor_unique": 601,
    "reused_unique": 601,
    "source_records": 2200,
    "supervised_unique": 1597,
    "supervision_assignment_records": 1597
  },
  "episode_2026091916283401": {
    "assigned_unique": 1914,
    "assignment_records": 2315,
    "context_only_unique": 403,
    "context_without_executor_or_supervision_unique": 3,
    "excluded": 0,
    "executor_assignment_records": 401,
    "executor_unique": 401,
    "reused_unique": 401,
    "source_records": 1914,
    "supervised_unique": 1511,
    "supervision_assignment_records": 1511
  },
  "episode_2026091916295601": {
    "assigned_unique": 3346,
    "assignment_records": 3747,
    "context_only_unique": 403,
    "context_without_executor_or_supervision_unique": 3,
    "excluded": 0,
    "executor_assignment_records": 401,
    "executor_unique": 401,
    "reused_unique": 401,
    "source_records": 3346,
    "supervised_unique": 2943,
    "supervision_assignment_records": 2943
  },
  "episode_2026091916320101": {
    "assigned_unique": 2773,
    "assignment_records": 3774,
    "context_only_unique": 1192,
    "context_without_executor_or_supervision_unique": 192,
    "excluded": 0,
    "executor_assignment_records": 1001,
    "executor_unique": 1001,
    "reused_unique": 1001,
    "source_records": 2773,
    "supervised_unique": 1581,
    "supervision_assignment_records": 1581
  },
  "episode_2026091916335001": {
    "assigned_unique": 2899,
    "assignment_records": 3500,
    "context_only_unique": 603,
    "context_without_executor_or_supervision_unique": 3,
    "excluded": 0,
    "executor_assignment_records": 601,
    "executor_unique": 601,
    "reused_unique": 601,
    "source_records": 2899,
    "supervised_unique": 2296,
    "supervision_assignment_records": 2296
  },
  "episode_2026091916411901": {
    "assigned_unique": 2141,
    "assignment_records": 2742,
    "context_only_unique": 690,
    "context_without_executor_or_supervision_unique": 90,
    "excluded": 0,
    "executor_assignment_records": 601,
    "executor_unique": 601,
    "reused_unique": 601,
    "source_records": 2141,
    "supervised_unique": 1451,
    "supervision_assignment_records": 1451
  },
  "episode_2026091916445601": {
    "assigned_unique": 2179,
    "assignment_records": 2780,
    "context_only_unique": 603,
    "context_without_executor_or_supervision_unique": 3,
    "excluded": 0,
    "executor_assignment_records": 601,
    "executor_unique": 601,
    "reused_unique": 601,
    "source_records": 2179,
    "supervised_unique": 1576,
    "supervision_assignment_records": 1576
  },
  "episode_2026091916463201": {
    "assigned_unique": 2361,
    "assignment_records": 2962,
    "context_only_unique": 710,
    "context_without_executor_or_supervision_unique": 110,
    "excluded": 0,
    "executor_assignment_records": 601,
    "executor_unique": 601,
    "reused_unique": 601,
    "source_records": 2361,
    "supervised_unique": 1651,
    "supervision_assignment_records": 1651
  },
  "episode_2026091916480201": {
    "assigned_unique": 2004,
    "assignment_records": 2905,
    "context_only_unique": 903,
    "context_without_executor_or_supervision_unique": 3,
    "excluded": 0,
    "executor_assignment_records": 901,
    "executor_unique": 901,
    "reused_unique": 901,
    "source_records": 2004,
    "supervised_unique": 1101,
    "supervision_assignment_records": 1101
  },
  "episode_2026091916505401": {
    "assigned_unique": 2399,
    "assignment_records": 3000,
    "context_only_unique": 603,
    "context_without_executor_or_supervision_unique": 3,
    "excluded": 0,
    "executor_assignment_records": 601,
    "executor_unique": 601,
    "reused_unique": 601,
    "source_records": 2399,
    "supervised_unique": 1796,
    "supervision_assignment_records": 1796
  },
  "episode_2026091916522501": {
    "assigned_unique": 2011,
    "assignment_records": 2612,
    "context_only_unique": 603,
    "context_without_executor_or_supervision_unique": 3,
    "excluded": 0,
    "executor_assignment_records": 601,
    "executor_unique": 601,
    "reused_unique": 601,
    "source_records": 2011,
    "supervised_unique": 1408,
    "supervision_assignment_records": 1408
  },
  "episode_2026091916534901": {
    "assigned_unique": 2903,
    "assignment_records": 3904,
    "context_only_unique": 1003,
    "context_without_executor_or_supervision_unique": 3,
    "excluded": 0,
    "executor_assignment_records": 1001,
    "executor_unique": 1001,
    "reused_unique": 1001,
    "source_records": 2903,
    "supervised_unique": 1900,
    "supervision_assignment_records": 1900
  },
  "episode_2026091916553801": {
    "assigned_unique": 1834,
    "assignment_records": 2235,
    "context_only_unique": 403,
    "context_without_executor_or_supervision_unique": 3,
    "excluded": 0,
    "executor_assignment_records": 401,
    "executor_unique": 401,
    "reused_unique": 401,
    "source_records": 1834,
    "supervised_unique": 1431,
    "supervision_assignment_records": 1431
  },
  "episode_2026091916583301": {
    "assigned_unique": 2613,
    "assignment_records": 3414,
    "context_only_unique": 803,
    "context_without_executor_or_supervision_unique": 3,
    "excluded": 0,
    "executor_assignment_records": 801,
    "executor_unique": 801,
    "reused_unique": 801,
    "source_records": 2613,
    "supervised_unique": 1810,
    "supervision_assignment_records": 1810
  }
}

All intervals use original paired indices [start, stop). Raw NPZ slices retain missing values and original action JSON. Each segment has an independent sequence boundary and original media references with SHA-256 hashes. No timestamp re-pairing or cross-segment trajectory concatenation occurred.

## Task interpretation (verbatim API output)

Interpret flip_egg as turning over the demonstrated egg object using a gripper-held spatula and returning it to the pan. All 20 authorized episodes show a black slotted spatula initially supported near/on a small dish, a pan containing an egg-shaped object, robot handle acquisition, tool approach and manipulation. Evidence supports insertion/angle adjustment, lifting on the blade and release back into the pan; in episode_2026091916553801 the egg is visibly airborne/tilted at 1580 between supported and resting views. Do not infer material, heat, cooking state or universal success from the task name or image appearance. Recording complete is only recording completion. Some tails include human contact with egg/pan/tool. Tool return to its rest with release is not established as a required task phase. The design stops after task-outcome inspection or assistance and does not claim tested robot execution.

## Goal conditioning (verbatim API output)

The supported caller goal is turn_over_in_same_pan(egg_track, pan_track, initial_face_reference, allowed_pan_interior). The initial reference is a causal crop acquired when the task begins, before tool interaction; training uses original row 0 for that reference after manually confirming the selected egg. It is not the episode's final image. allowed_pan_interior is the observed pan interior eroded by a configured margin, not a fabricated target point inferred from the eventual landing. The Agent retains this goal across retries so that a second call does not redefine the already-turned face as the new target. There are no labels for arbitrary landing locations or arbitrary foods, so do not relabel such goals. The policy is conditioned on the goal type, initial reference and tracked scene, not privileged phase or flip-success labels. Outcome-side annotations, if added in later preparation, are audit evidence only in this design; completion remains Agent-confirmed or unknown.

## Portfolio rationale (verbatim API output)

Choose one learned control policy after allocating motion generation to the executor. There is no useful evidence for separately trained pickup, scoop, lift, flick, withdraw and retry motor policies: they share a single held tool, the same six local motion quantities and directly observable interaction progress, with uncertain boundaries and many adjustments. A fixed known-tool grasp recipe is preferable to fitting a second control policy to nearly identical successful handle pinches. The remaining interaction is not reducible to a contact point alone: angle, insertion, support and release timing change within attempts, while the supplied planner does not establish contact dynamics. Therefore dense, receding-horizon local geometry is justified. One auxiliary perception model is explicitly budgeted and annotated, giving two trained components total, not two callable motion policies. No alternative-policy quota or redundant independently trained portfolio is used.

## Model inventory (verbatim API output)

Two trained components are proposed, neither implemented here: (1) egg_interaction_v1, one independently callable control policy with a compact two-view image encoder, geometric/proprioceptive MLP, 128-unit GRU and a three-component diagonal-Gaussian six-motion-coordinate head; its selected prior is blade_relative_local_motion. (2) scene_geometry_aux, one shared two-camera U-Net-style encoder/decoder with 32/64/128 channels, four semantic classes (background, pan, egg object, spatula), six spatula landmark heatmaps and visibility outputs. It provides perception to both the executor and policy; it is not another callable control policy. Train both from scratch with the annotation requirements below; no external pretrained weights, foundation-model detections, unrecorded object poses, force models or learned success classifiers are assumed. PnP/model fitting, attachment estimation, geometric normalization, gripper recipe, watchdogs, collision checking, tracking, status logic and Agent verification are nonlearned integrations. A safety-rated workspace interlock or equivalent supervised operating procedure is a separate required dependency, not scene_geometry_aux.

## Generalization design (verbatim API output)

Target generalization is changed observed initial placement and interaction progress for this same tool/pan/egg-object setup, not episode replay. Inspected scenes show tool-rest placement variation, egg position/appearance variation, different pan translations and handle headings, partial insertion, repeated angle adjustments, and pan displacement during interaction. Early episodes have the pan handle directed left; later episodes include front-facing handles and different pan locations. Pool all interaction stages and adjustment attempts into one conditional problem. The rigid spatula, constrained grasp, approximately planar receiving surface and fixed camera calibration allow strong geometric normalization; retain all six motion coordinates because insertion, pitching, lateral correction and clearance are not one-dimensional. Use metric tool-relative motion and pan-relative state, camera masks, causal temporal history and a fixed hold-gripper constraint. This should reduce learning of robot kinematics and global translations while leaving contact strategy to learning; benefit is a hypothesis. Primary split by complete contiguous layout blocks: training episodes 16095801 through 16244201 and 16411901 through 16480201 (11 episodes); validation 16283401 through 16335001 (4); held-out test 16505401 through 16583301 (5), using the full trajectory IDs in the catalog. All reused rows, goal references, annotations and derived labels from an episode remain in its partition. Audit cross-episode near-duplicate image clusters before training; if a cluster spans partitions, move its whole related episode cluster to one partition and publish the revision, rather than splitting duplicate frames. Shared physical objects across these blocks are intentional within-setup evaluation, not evidence of novel-object transfer. Also report leave-layout-block-out sensitivity and progress-conditioned tests. No random row split.

## Calling contract (verbatim API output)

There is ONE independently callable learned motion policy, egg_interaction_v1, and ONE supplied-executor responsibility group, geometry_execution. scene_geometry_aux is an auxiliary observation model, not a control policy. All interfaces below are designs, not installed robot APIs. StartSession(task=turn_over_in_same_pan, egg ROI/track ID, pan ROI/track ID, spatula template ID, initial-face reference, safety configuration) initializes causal tracking. geometry_execution.acquire_tool receives an explicit tool-relative grasp recipe, scene geometry, robot state and collision constraints; it plans approach/lift and operates a separately bound gripper adapter. On confirmed tool retention, pass the measured attachment transform and fresh observations to egg_interaction_v1. The policy returns a six-dimensional local blade motion specification for at most 0.10 s, not joints, torques or a complete flip. geometry_execution.track_local converts it into reachable robot geometry and tracks it subject to the contact adapter contract. Reobserve and call again at 10 Hz; stop rather than queue stale decisions. The Agent sees planned/tracking/arrived/interrupted/infeasible/invalid_observation/contact_limit/tool_slip plus raw outcome images. Arrival means a geometry objective was reached, not that the egg flipped. After release and clearance, the Agent verifies the opposite face is up and the egg is supported inside the pan, or requests another observation, a supported-state retry, or assistance. Interruption cancels pending goals; preserve the task's initial-face reference but reset policy recurrent state and revalidate tool attachment before resuming. No episode index or elapsed episode time is a caller argument.

## Capability coverage (verbatim API output)

Responsibility allocation: the supplied premise covers feasible connecting motion and low-level tracking from explicit goals, not grasp choice, contact dynamics or success recognition. A calibrated fixed recipe for the demonstrated spatula supplies the grasp objective; a new gripper/retention adapter supplies close-and-test behavior. Executor prefixes retain open-hand approach, alignment, closure, lift and early free-space transfer without training a pickup motor policy. At their inspected endpoints the tool is visually held and widths are approximately 0.03349 or 0.034039 m, compared with approximately 0.138353 m initially; these numbers alone are not contact labels. The learned policy supplies task-dependent final approach, blade angle, insertion/withdrawal adjustments, support acquisition, lift, release-producing motion and immediate clearance, expressed as short geometric motion. The executor tracks these decisions but does not invent them. Evidence includes episode_2026091916095801 at 450/600/900/1200/1650/1810, extended adjustments in episode_2026091916200101 at 1400/1700 and episode_2026091916295601 at 1400/1800, and the supported-to-airborne-to-pan transition in episode_2026091916553801 at 1550/1580/1610. scene_geometry_aux supplies observed masks/keypoints; deterministic geometry fitting and the Agent supply identity, constraints and outcome interpretation. Human interventions in the tails of 6320101, 6411901 and 6463201 are observation/assistance evidence, never autonomous policy targets. Safe stop, assistance, optional free-space retreat and retry gating are execution logic. No demonstrated behavior is assigned to an assumed autonomous flip planner.

## Sharing and diversity audit (verbatim API output)

Share the interaction policy across all 20 episodes, both initial visible-face appearances, all inspected pan headings, and initial or repeated insertion adjustments. Condition on causal state and goal rather than choosing a model by trajectory ID. The early grasp/transfer behavior is assigned to calibrated geometry and executor logic, not used to increase the number of control models. The same scene_geometry_aux weights serve both cameras, with camera-specific calibration and visibility; wrist occlusion and third-view context are complementary rather than independent examples. Do not count repeated static rows, shared initial-face references or overlapping executor cuts as independent demonstrations. Stratify evaluation by layout block, supported versus unsupported egg observations, adjustment-heavy episodes and release events. The dataset has no established material/temperature diversity, no annotated success rate and no demonstrated tool family; do not claim those kinds of diversity.

## Motion coverage (verbatim API output)

All 45,730 original paired rows are retained in the learned group's 20 full-episode sequences, with control loss restricted to the declared held-tool intervals. Twenty overlapping executor prefixes contain 12,570 records of acquisition/free motion evidence. The dense intervals declare 32,737 control anchors; this is not a success count and usable labels remain subject to explicit validity auditing. Twenty handoff rows are both executor evidence and control anchors. There are 12,993 unique rows without control anchors, including acquisition context and 443 unique tail/context rows outside executor coverage. The first/last included states and supervised image anchors have been inspected, and both camera views were inspected in every episode. Three human-intervention tails stop control supervision conservatively before intervention while retaining the original ending. Other final three rows are future-target/outcome buffers. No source row is excluded, renumbered or timestamp-repaired. Original commands, including null dq and null target_pose, remain untouched; measured dq is never used as command dq.

## Unobserved cases (verbatim API output)

Additional data and integration tests are required for different tools or grasp offsets, different foods/materials/temperatures, unknown pan geometry, arbitrary approach directions, major camera changes, force-sensitive or deforming contact, adhesion, tool slip/drop, off-pan egg recovery, pan tipping, clutter, severe occlusion and reliable human-safe operation. The recordings do not establish calibrated grip force, depth reconstruction, contact ground truth, exposure synchronization, a Flip egg planner binding or a safe contact-tracking adapter. Required next-stage work is manual scene annotation, metric tool/pan/jaw measurement, camera/TCP/attachment validation, gripper semantics verification, compliant/force-limited tracking integration, source-keyed target validity auditing, and held-out evaluation with verified flip outcomes. Stop this stage before any preprocessing/policy code, training or robot execution.

## Overlap rationale (verbatim API output)

Each learned interval materializes the complete episode so that initialization images, tool-grasp geometry, the initial-face reference, causal history and outcome evidence remain available. Its early prefix has no control loss. Each executor interval [0,s+1) duplicates that same prefix as explicitly tagged execution/calibration evidence and shares row s as the measured handoff observation. This reuse is deliberate, not two independent trajectories or doubled policy supervision. Index every derived item by (trajectory_id,sample_index,role); apply one control loss per declared anchor and one annotation weight per source image. No window joins episodes. Outcome buffers may be read for targets or audits, never as causal policy input.

## Evidence

- [API-authored plan](../data/flip_egg/cut_v1/plan.json)
- [Published artifact hashes](../data/flip_egg/cut_v1/manifest.json)
- [Independent validation](../runs/flip_egg/cut_v1/validation.json)
- [API cost ledger](../runs/flip_egg/cut_v1/cost_ledger.json)
- [Source inventory](../runs/flip_egg/cut_v1/source_manifest.json)
- [Proposed policy catalog](../data/flip_egg/cut_v1/POLICY_CATALOG.md)
- [Frozen general prompt](../runs/flip_egg/cut_v1/interface_snapshot/real_robot/flip_egg_cut_v1/general_prompt.md)
- [Frozen task specification](../runs/flip_egg/cut_v1/interface_snapshot/real_robot/task_specifications/flip_egg_v1.md)
