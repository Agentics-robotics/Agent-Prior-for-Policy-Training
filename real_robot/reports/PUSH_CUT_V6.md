# push_letters: API segmentation and heuristic designs

Review date: 2026-09-21. Completed stage: cut/grouping and heuristic documents only.
No policy code, preprocessing implementation, training, or physical robot execution has occurred.

Model: `gpt-6-astra`; reasoning: `xhigh`; verified API calls: 24.
Reported input/output tokens: 1,305,654 / 22,122. Estimated standard API cost: USD 3.7576 (not an invoice).

## Published datasets

| Dataset | Original segments | Heuristic designs | Documents |
| --- | ---: | ---: | --- |
| shape_push_decisions | 12 | 1 | [heuristics](../data/push_letters/cut_v6/datasets/shape_push_decisions/heuristic.md), [dataset](../data/push_letters/cut_v6/datasets/shape_push_decisions/dataset.json) |
| geometric_motion_executor | 2 | 0 | [heuristics](../data/push_letters/cut_v6/datasets/geometric_motion_executor/heuristic.md), [dataset](../data/push_letters/cut_v6/datasets/geometric_motion_executor/dataset.json) |

## Coverage

{
  "episode_2026091922002701": {
    "assigned_unique": 9015,
    "assignment_records": 17935,
    "context_only_unique": 8992,
    "context_without_executor_or_supervision_unique": 0,
    "excluded": 0,
    "executor_assignment_records": 9015,
    "executor_unique": 9015,
    "reused_unique": 8615,
    "source_records": 9015,
    "supervised_unique": 23,
    "supervision_assignment_records": 23
  },
  "episode_2026091922062101": {
    "assigned_unique": 7730,
    "assignment_records": 15525,
    "context_only_unique": 7706,
    "context_without_executor_or_supervision_unique": 0,
    "excluded": 0,
    "executor_assignment_records": 7730,
    "executor_unique": 7730,
    "reused_unique": 7490,
    "source_records": 7730,
    "supervised_unique": 24,
    "supervision_assignment_records": 24
  }
}

All intervals use original paired indices [start, stop). Raw NPZ slices retain missing values and original action JSON. Each segment has an independent sequence boundary and original media references with SHA-256 hashes. No timestamp re-pairing or cross-segment trajectory concatenation occurred.

## Task interpretation (verbatim API output)

Push physical letter instances into a requested word with an explicit spatial arrangement. The recordings show flat brown pieces manipulated by a long rigid white pushing tool, including outer contacts, accessible inner-loop contacts, planar translation/rotation, temporary relocation, neighbor clearance, repeated reapproach and final adjustments. No grasping is supported by inspected media or valid gripper feedback. Two episodes appear to use the same seven-piece inventory in different initial arrangements and end in a word-like layout, but neither intended string nor correctness/success is supplied. Design for new semantic identities AND new physical shapes, new words/layouts and new initial arrangements, with impossible-inventory/ambiguity handling distinct from physical task completion. Scope is a responsibility map, sparse target contracts, priors and exact raw cuts; not implementation, model weights or robot execution.

## Goal conditioning (verbatim API output)

The physical policy goal is {instance_id, G in SE(2) of plane frame P, target reference contour, translation/yaw tolerances, protected-instance set, workspace/staging constraints}. P is a calibrated metric tabletop frame with normal toward free space; its transform to base must be validated, not equated automatically with metadata board coordinates. Each track's reference contour is initialized from its first sufficiently visible causal observation; its gauge may be arbitrary. G transforms that same contour to the desired pose, so policy use does not require a known font or semantic canonical template. The Agent's semantic upright direction is used only to assign G; identity-discriminating orientations must not be collapsed by geometric symmetry (e.g. a rotated shape may denote another character). For symmetric contours retain equivalent geometric pose hypotheses but check semantic orientation separately. At training, per-anchor G is hindsight-relabeled from the same physical instance at the explicit future witness f listed in each segment. This is an achieved local spatial objective, not an inferred intended word or assertion of successful teleoperation. Future masks, tracking and pose estimates are labels only. At deployment G comes from the current request/layout planner, never a future image. For far goals the Agent requests intermediate collision-free spatial goals; those are objectives, not a guaranteed object-motion path. A word string alone is never the geometric policy input.

## Portfolio rationale (verbatim API output)

The remaining interaction problem is a single reusable conditional choice, not separate travel, turning, letter or word policies. A contact with an eccentric moment arm can turn an object; a different contact can translate it; a new observation can request a correction with the same output schema. One small scorer is sufficient to propose contact/direction/distance while the supplied executor handles connecting trajectories. It is deliberately not an end-to-end semantic or motor network. 47 candidate decisions and two episodes do not support independent specialist networks or a portfolio quota. No independently trained alternative is included. Two frozen perception priors are explicit external dependencies rather than claims of learning unseen identity from the push recordings.

## Model inventory (verbatim API output)

Exactly ONE new trainable, independently callable control policy: shape_push_v1, a small shared candidate-scoring geometric network trained at the 47 listed sparse decision anchors. No joint-velocity policy, learned travel policy, word-specific policy, learned forward dynamics model or trained termination classifier is proposed. TWO proposed external frozen learned dependencies, not trained on these cuts and not supplied assets: SAM 2.1 Hiera-small class-agnostic mask proposal/video propagation model, and Qwen2.5-VL-7B-Instruct for open-vocabulary instance identity/uprightness hypotheses from original RGB crops plus scene context and the request. Their exact downloaded revisions, licenses, hashes, preprocessing and runtime costs must be pinned before implementation; neither is assumed installed or validated here. SAM proposals are automatic grid/region proposals filtered by calibrated table geometry, not human per-move prompts. VLM labels never define physical boundaries; cross-check candidates over views and expose unknown. No confidence from either model is treated as a calibrated probability until evaluated. Deterministic modules: undistortion/ray-plane geometry, mask fusion and contour extraction, uncertainty-aware rigid-contour tracking, candidate construction, goal canonicalization, assignment/layout/dependency-graph logic, executor adapter and geometric progress tests. The supplied High-level Agent is upstream and not trained here. Offline mask/contact/instance review is an annotation requirement, not another hidden learned model. These external semantic assets provide broad prior knowledge absent from two demonstrations; arbitrary invented glyph semantics can remain unidentifiable without a user convention. Every trainable or pretrained dependency is named here.

## Generalization design (verbatim API output)

Separate four generalizations. Identity: use an external open-vocabulary vision-language prior over candidate physical instance crops, not a classifier trained on the recorded alphabet; retain unknown and orientation ambiguities, verify inventory counts, and permit a user explanation/exemplar only when genuinely ambiguous. Shape: the manipulation policy consumes measured contours including holes, tool geometry, goal displacement fields and local free space, never character IDs, colors, font IDs or stored letter templates. Share contact scoring across all boundary points and all instances; goal-aligned geometric canonicalization provides planar rigid-transform equivariance. This supports a hypothesis of transferring local contact structure to entirely unseen shapes, while full-contour/global goal context prevents a purely local rule from ignoring torque/concavity. No two-episode evidence establishes that transfer. Words/layout: map any feasible string to explicit per-instance target transforms and solve assignment/ordering/staging from current occupancy; no trained word or action-order vocabulary. Arrangements: online replanning, scene-relative coordinates and obstacle masks separate object choices from robot travel paths. Required tests: (a) semantic recognition/instance counting/uprightness for identities absent from both demonstrations, including ambiguous rotations and repeated characters; (b) manipulation with correct oracle instance/pose inputs on genuinely held-out physical letter geometries, fonts, holes, concavities and aspect ratios; report pose error, contacts, slips and collateral displacement separately from recognition; (c) novel words/layouts with familiar physical pieces; (d) jointly novel identities, physical shapes, words and cluttered initial arrangements end-to-end without oracle inputs; (e) infeasible inventories and ambiguous requests with correct rejection. Hold out whole manufactured shape families, not just letter positions. Test material/friction/thickness shifts independently; conservative failure does not count as successful unseen-shape manipulation. Required new-shape evaluation and further interaction data remain part of the deployment work, not excluded tasks.

## Calling contract (verbatim API output)

Interfaces are proposed bindings, not remotely available robot APIs. (1) observe_scene(rgb_third, rgb_wrist, original timestamps/ages, q, T_base_flange, T_base_ee, calibration_id, prior_tracks) -> Scene{revision,time,plane_frame P,workspace_polygon,instance_tracks,uncertainty,validity}. Each track has a persistent physical instance_id, measured outer/inner contours in P, top height/thickness with uncertainty, reference-contour rigid transform, identity hypotheses and upright-orientation hypotheses. Frozen perception dependencies and geometric conversion are specified in model_inventory and the learned design. Never use future images in this call. (2) High-level Agent receives request{word,reading_direction,layout_region/baseline,spacing,tolerances,optional explicit per-instance target poses}; resolves inventory by injective matching of character occurrences to distinct tracks. Missing inventory -> impossible_request; unresolved identity/uprightness -> ambiguous_request, not success. If layout unspecified, propose and obtain acceptance of an explicit baseline origin, direction, upright direction, spacing and region, not a hidden demonstrated layout. Pack measured shapes without overlap; build a dependency graph of blocked slots and choose a free staging pose when necessary. Select an unblocked instance/target by current geometry and measured progress, not a fixed spelling order. (3) propose_push(instance_id, goal_SE2_in_P, scene_revision, robot/tool geometry, constraints, history_token) -> one ranked contact/stroke specification, uncertainty and status. Call once before approach, then again or revalidate at executor arrival using a fresh scene. (4) execute_geometry(mode=prepare|push|withdraw|view|hold, specification, scene_revision, limits) -> ticket, progress, completion/failure and latest state. A prepare completion means robot arrival, NOT object-goal completion. The Agent executes at most one bounded stroke, observes again, recomputes pose error, and repeats or selects another instance/staging goal. On interruption stop, invalidate the old ticket, reobserve and replan; never resume a stale stroke by sample number. Example: request a feasible novel word with baseline in P; observe and match each occurrence, including separate tracks for repeated letters; assign measured target contours; choose track j with an unblocked slot; propose_push(j,Gj); prepare(contact); observe and call propose_push(j,Gj) again if contact geometry changed; push <=20 mm; observe; if error remains choose another contact, otherwise withdraw and select the next unblocked track. Verify the entire requested arrangement independently at the end. User clarification is permitted for semantic/layout ambiguity, not human selection of each move. No literal recording indices, inferred recorded words or episode endpoints enter this interface.

## Capability coverage (verbatim API output)

Responsibility is joint, not one trained policy per motion. Supplied robot planner/controllers provide collision-aware free-space approach, descent to a specified standoff, lifting/withdrawal, tool reorientation, travel between letters and reconnecting after a failed contact. A required adapter also tracks an explicitly bounded contact stroke with selected-object contact allowance and conservative guards; this is not a supplied object dynamics model. One learned geometric decision policy supplies WHERE and HOW to push: outer or accessible inner boundary contact, planar direction and short distance, conditioned on current full shape, obstacles and requested object pose. Repeated calls generate translations, eccentric rotational adjustments and repeated attempts. High-level Agent plus frozen semantic perception handle recognition, instance assignment, feasibility of spelling, explicit spatial layout, staging/order selection, and whole-task verification. Sources show seven flat pieces resembling W/O/R/L/D and two serif-bar letters that can look H/I-like under rotation; these are visual hypotheses, not annotations. A: 500/800/1100/1600 first piece turning/relocation; 1800 and 2100-3200 neighbor clearance and ring positioning; 3500/4300/4800 temporary relocation, other-piece movement and interior-loop pushing; 5100-6200 placement/correction; 6200-8250 repeated open-corner recontacts/turns; 8250-9014 final bar-piece adjustments. B supports different starting positions/routes: 300-1450 first piece and neighbor, 1450/1700 interior-loop relocation, 1950-2850 ring, 2850-4550 legged piece and correction/reapproach, 4550-6200 loop/open-corner placement, 6200-7729 final bar adjustments. These roles cover every source record through the executor evidence group. No grasping, trustworthy gripper state, certified contact force or annotated word-success behavior is established.

## Sharing and diversity audit (verbatim API output)

Pool decisions across outer contacts on zigzag, ring, legged and open-corner pieces, inner-loop contacts on the rectangular-loop piece, both bar-shaped pieces, different spatial locations, pre-approach observations and in-contact/recontact observations. Evidence includes A800/830 (outer edge), A4800/4810 and B1700 (inner boundary), A6700/6900/7200/7600/7800 and B5100/5500/5850 (repeated open-corner attempts), B3900/3930 and B4200 (correction and lifted reapproach). IDs resembling letters are for visual audit only. Different task order/routes across the episodes are not new independent shapes. Contact priors are shared because outputs, frame, available geometry and feedback cycle are identical. Scene semantics, image segmentation and robot motion remain separate because they solve different problems and have different data sources. Weight each unique anchor once, balance episodes and interaction events, and cap event total weight so the long L-like correction run does not dominate. No random frame train/test split. For a provisional experiment train all A cuts and evaluate all B cuts, then swap; frozen-dependency tuning or annotation choices using B invalidate it as an untouched test. Two folds estimate only same-inventory transfer. Held-out physical-shape/word tests need new independent episodes. Keep all mask annotations, future goals, augmented copies, executor evidence and repeated physical-instance material grouped with their original episode; never use a held-out episode's labels to tune conversion or thresholds.

## Motion coverage (verbatim API output)

Retain all 16,745 original paired rows without exclusion. The executor group has two complete sequences, 9,015 and 7,730 rows, preserving startup waits, free travel, descents, changes of height/orientation, actual pushes, pauses, recontacts, all corrections and terminal observations. It contains zero policy-loss rows. The learned group has 12 irregular interaction-centered cuts with preceding buffers, totaling 16,715 materialized rows, 16,105 unique source rows and 47 unique sparse decision anchors (23 A, 24 B). The 610 repeated rows within learned cuts are intentional boundary buffers, not independent observations. Early A[0,400) and B[0,240) are executor evidence only. All learned rows are also retained in the executor group; the unique cross-group overlap is 16,105. Non-anchor rows are history, contact/goal derivation and execution/outcome evidence, not unrequested dense command targets. Coverage of demonstrated behavior is consequently much larger than 47 decisions. This does not imply all derived targets will pass the later geometry/contact validity audit. Source endpoints have no fabricated successor; completion of recording is not task success.

## Unobserved cases (verbatim API output)

Not established by these recordings: held-out physical shapes/letter identities/words, repeated-character inventories, broad fonts/materials/friction/heights, semantic disambiguation, stacked/overlapping or deformable pieces, very small holes, tipping, tool entrapment, exact goal tolerance success, autonomous instance segmentation/tracking, fully integrated planner/contact-controller guards, synchronized metric RGB-depth reconstruction, calibrated tip contact surface, object force/mass/center-of-pressure, deliberate recovery from a labeled failure or a verified final success. Address these through the specified open-vocabulary perception and geometry-conditioned closed loop, independent tests, validated calibration and targeted data acquisition for novel geometry/interactions. Do not count accepting an arbitrary string as manipulating unseen shapes. Clear perception/geometry/calibration or safety failures must halt/request a view/clarification or report blocked; such rejection is not task success. All new annotations, dependencies and integrations described here are requirements for later authorized work, not assets silently assumed to exist.

## Overlap rationale (verbatim API output)

Each learned event cut includes 60 preceding rows (100 for the first A cut) before its supervised range. Adjacent cuts share 61 rows: preceding causal history plus the common inspected outcome/switch observation. Sparse decision anchors are owned by exactly one segment; a shared boundary may be an outcome in the preceding cut and a decision in the next without duplicate loss. Whole-episode executor copies preserve motion/context continuity for integration audit and perception review, but do not contribute imitation loss. Never concatenate disjoint cuts. Training windows and target searches are confined to their own materialized sequence. All copies of an original trajectory/instance/event must share one split and one source-keyed cache; duplicate source targets are weighted once.

## Evidence

- [API-authored plan](../data/push_letters/cut_v6/plan.json)
- [Published artifact hashes](../data/push_letters/cut_v6/manifest.json)
- [Independent validation](../runs/push_letters/cut_v6_with_example/validation.json)
- [API cost ledger](../runs/push_letters/cut_v6_with_example/cost_ledger.json)
- [Source inventory](../runs/push_letters/cut_v6_with_example/source_manifest.json)
- [Proposed policy catalog](../data/push_letters/cut_v6/POLICY_CATALOG.md)
- [Frozen general prompt](../runs/push_letters/cut_v6_with_example/interface_snapshot/real_robot/prompts/general_cut_and_prior_v5.md)
- [Frozen task specification](../runs/push_letters/cut_v6_with_example/interface_snapshot/real_robot/task_specifications/push_letters_v2.md)
