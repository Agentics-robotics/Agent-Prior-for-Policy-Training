# Shared goal-conditioned geometric push decisions

Supply the minimal task-dependent geometric interaction specification needed to change one selected letter instance toward a caller-defined pose, reusing one model across physical shapes and words.

## Dataset rationale

Six interaction-centered sequences per episode, not cuts from numerical motion bins and not separate policies per letter. Boundaries are inspected changes in selected-object activity or a useful outcome/reapproach state. Multi-object clearance intervals intentionally keep neighboring interventions in one sequence because actor/goal conditioning is assigned at each sparse anchor, not to the entire interval. Each cut retains causal preceding context and future goal/contact evidence. Materialized first/last states and supervised boundary images were inspected; all sparse anchors have inspected paired states and images. Shape names below are provisional visual referents for later track annotation.

## Component responsibility: learned_policy

propose_push is a task-dependent geometric decision service, not a robot command stream. Required inputs: latest validated Scene, selected track and SE(2) target, current tool transform/geometry, protected objects, reachability mask if available, caller limits and causal history. It returns {scene_revision,instance_id,contact_point_P[2] m,boundary_loop_id,inward_push_unit_P[2],stroke_length_m,preferred_contact_height/pose_constraints derived by adapter,ranked_alternatives,validity,status,history_token}. The only learned quantities are the contact candidate, planar direction and distance; height, approach clearance, tool orientation feasibility, speed, force/torque guards and withdrawal are geometric/control responsibilities. Free-space prepare moves to a safe offset from the selected contact, possibly by descending into a sufficiently wide hole; inner-loop insertion requires full tool-envelope clearance, not a generic straight-line move through material. The executor reobserves at arrival. Commit to at most one <=min(20 mm,0.2 object diameter) stroke, at commissioned conservative speed and <=1 s between visual reconsiderations; subdivide further for risk/uncertainty. In-contact calls may continue locally without a full lift, subject to fresh guard checks. Completion of the stroke triggers observation and object-error assessment, not success by trajectory termination. Selection and history semantics are in policy_contract. The adapter must be implemented against the user's planner/controller with verified Cartesian tracking/contact-permission, stopping behavior and tool/plane calibration; none was exercised here.

### a_zigzag

Source: `episode_2026091922002701` [400, 1801). Supervision kind: sparse; bounds: 500, 1801; decision indices: [500, 800, 1100, 1600].

**Boundary evidence indices**

[400, 500, 1800]

**Condition evidence indices**

[500, 800, 1100, 1600]

**Decision indices**

[500, 800, 1100, 1600]

**Deployment condition source**

Agent-selected physical zigzag-shaped instance and explicit target pose from current request/staging plan; never select it because its recorded turn was first.

**Label derivation**

Apply the common target conversion. Actor is the zigzag W/M-like piece initially near the upper-left cluster. Anchor -> future goal witness: 500->1100, 800->1100, 1100->1600, 1600->1800. 500 is pre-contact approach context; 800/1100/1600 cover changed contacts/poses. Retained 700/800/830 and 1600/1800 images/states support contact and relocation inference. Future witnesses are achieved poses only, not desired recorded spelling.

**Merge check**

Could share one longer raw sequence with next event, but ring/neighbor interventions would obscure goal derivation and overweight travel. Share model weights and source split, not word-level labels.

**Objective**

Choose contacts for initial turning, translation and correction of the zigzag piece.

**Rationale**

500 shows approach above the unchanged inventory; 800/1100 show reoriented first piece; 1600 shows it displaced to the lower-right; 1800 is following-object context/outcome.

**Split check**

Do not split each robot pause/contact into an independent policy. Sparse anchors distinguish pre-approach from local corrections while context retains the whole relocation.

**Supervision exclusions**

[]

**Supervision kind**

sparse

**Training condition**

Current actor geometry and per-anchor hindsight SE(2) goal, obstacles/tool geometry and causal history. No character/word input.

**Uncertainty**

W/M orientation and exact contact onset are unannotated; tool pose/metric labels require review. 1800 is an outcome, not annotated success.

### a_ring_and_clearance

Source: `episode_2026091922002701` [1740, 3301). Supervision kind: sparse; bounds: 1800, 3301; decision indices: [1800, 2100, 2400, 2800].

**Boundary evidence indices**

[1740, 1800, 3300]

**Condition evidence indices**

[1800, 2100, 2400, 2800, 3200]

**Decision indices**

[1800, 2100, 2400, 2800]

**Deployment condition source**

Agent may request neighbor clearance or ring placement, with explicit selected instance and target; selection is driven by current blocked geometry.

**Label derivation**

Common conversion with 1800->2100 for the central serif-bar piece initially just above the ring; 2100->2800, 2400->2800 and 2800->3200 for the ring-shaped piece. Track these separately. 2100 is an approach observation, 2400 an outer-edge interaction, 2800 a different-side correction. 3200/3300 retain achieved ring pose and departure context.

**Merge check**

Do not merge losses as one actor or a single push. Neighbor clearance is a separate caller-conditioned anchor; sharing one buffered sequence preserves clutter context without a separate clearing policy.

**Objective**

Select a short clearance push and ring translation/orientation corrections.

**Rationale**

1800-2100 changes the neighbor; 2400/2800/3200 show ring displacement and alignment; 3300 shows departure toward the remaining cluster.

**Split check**

Actor changes are explicit in sparse target mapping. Splitting into dense contact/travel skills would train planner-provided motion unnecessarily.

**Supervision exclusions**

[]

**Supervision kind**

sparse

**Training condition**

Per-anchor selected physical instance, causal contour/obstacles and hindsight goal transform; clearance and placement use identical control schema.

**Uncertainty**

Neighbor movement and exact intent are visual inferences. Incidental motion of another track must not be labeled as the selected actor's controllable stroke.

### a_cluster_reorganization

Source: `episode_2026091922002701` [3240, 5101). Supervision kind: sparse; bounds: 3300, 5101; decision indices: [3500, 4300, 4800].

**Boundary evidence indices**

[3240, 3300, 5100]

**Condition evidence indices**

[3500, 3700, 4300, 4800, 4810]

**Decision indices**

[3500, 4300, 4800]

**Deployment condition source**

Current scene dependency graph selects a staging/placement objective for one track. No memorized R/H/D sequence.

**Label derivation**

Common conversion: legged bowl/R-like piece 3500->3700; upper bar/H-like piece 4300->4800; rectangular-loop/D-like piece 4800->5100. At 4800/4810 the wrist view shows the tool inside the loop; derive inner-boundary contact, not an outer-edge proxy. 3300 is retained transit/uncertain switching context without an action anchor.

**Merge check**

Keep this irregular cluster-reorganization interval together for obstacle and collateral-motion evidence. Do not infer one global word goal or merge actor identities; they are selected separately at anchors.

**Objective**

Share staging, neighbor relocation and accessible internal-boundary pushing decisions.

**Rationale**

3500/3700 show temporary placement of the legged piece; 4300/4800 show bar relocation and interior-loop contact; 5100 shows loop moved to a different part of the table.

**Split check**

Three anchors represent physically different choices with the same output. Splitting by outer/inner contact into trained specialists would waste the common geometric prior; preserve loop type as input.

**Supervision exclusions**

[]

**Supervision kind**

sparse

**Training condition**

Anchor-specific track and hindsight pose, with full clutter contours and protected objects. Accessible hole geometry is measured, not keyed to D identity.

**Uncertainty**

3500 contact partly occluded; masks/tracks need review. Actual inter-object contact must be treated as outcome evidence or invalid primitive target, not silently safe.

### a_legged_and_loop_finish

Source: `episode_2026091922002701` [5040, 6201). Supervision kind: sparse; bounds: 5100, 6201; decision indices: [5100, 5600, 6000].

**Boundary evidence indices**

[5040, 5100, 6200]

**Condition evidence indices**

[5100, 5600, 6000]

**Decision indices**

[5100, 5600, 6000]

**Deployment condition source**

Agent requests measured goal pose for the legged or rectangular-loop track according to current layout; correction is triggered by residual pose error.

**Label derivation**

Common conversion: legged piece 5100->5600 and 5600->6000; rectangular-loop piece 6000->6200. 5100 has an elevated external-torque pattern relative to nearby observations; it is not a force label, and contact/occlusion/primitive fit require review rather than cloning that force response.

**Merge check**

Related placement/correction data share one sequence and model. The 5100 shared boundary is an outcome in the preceding segment and a decision only here.

**Objective**

Select placement and fine recontact corrections near already arranged pieces.

**Rationale**

5100 begins another legged-piece contact, 5600 is near the ring, 6000/6200 show rectangular-loop orientation/position correction before leaving for the open-corner piece.

**Split check**

No new correction policy; local goal errors and observed feedback explain the change. A contact that fails the stroke assumptions gets a validity mask, not a fabricated label.

**Supervision exclusions**

[]

**Supervision kind**

sparse

**Training condition**

Current selected shape, nearby protected arrangements, caller-equivalent hindsight target and short causal history.

**Uncertainty**

No force calibration or success label. Small loop adjustment may be near pose-estimation noise; mask uncertain displacement/goal components.

### a_open_corner_retries

Source: `episode_2026091922002701` [6140, 8251). Supervision kind: sparse; bounds: 6200, 8251; decision indices: [6200, 6700, 6900, 7200, 7600, 7800].

**Boundary evidence indices**

[6140, 6200, 8250]

**Condition evidence indices**

[6200, 6700, 6900, 6930, 7200, 7600, 7800, 8000]

**Decision indices**

[6200, 6700, 6900, 7200, 7600, 7800]

**Deployment condition source**

Agent-selected open-corner instance and explicit spatial pose. Recontact chosen after measured error/slip, not elapsed recording phase.

**Label derivation**

Common conversion for the same open-right-angle/L-like track: 6200->6700, 6700->6900, 6900->7200, 7200->7600, 7600->7800, 7800->8250. Each f is an achieved intermediate pose, including turns that need subsequent corrections, not a label of monotonic progress toward an inferred word. 6900/6930 show local motion with changing tool rotation; restrict the fitted primitive to a valid short prefix or flag it unsupported.

**Merge check**

Keep repeated attempts on the same instance together; otherwise nearly identical before/after shapes would leak across splits and retries could be mistaken for different skills.

**Objective**

Choose successive contacts around an open corner for rotation, translation and reattempts.

**Rationale**

6700,6900,7200 and7600 show substantial orientation changes/reapproaches before 7800/8000/8250 placement near other pieces. This is evidence against a single long open-loop push.

**Split check**

Six sparse choices distinguish repeated interactions; do not split on every sign change or learn a fixed six-stage script. An online retry may occur at any location.

**Supervision exclusions**

[]

**Supervision kind**

sparse

**Training condition**

One persistent physical track, per-anchor geometry/goal and optional causal prior response, including inner-concavity accessibility and neighboring pieces.

**Uncertainty**

Contact surfaces are partly occluded, tilt changes are significant, and some intermediate outcomes are not aligned. They are not all asserted successful demonstrations.

### a_bar_finish

Source: `episode_2026091922002701` [8190, 9015). Supervision kind: sparse; bounds: 8250, 9015; decision indices: [8250, 8500, 8800].

**Boundary evidence indices**

[8190, 8250, 9014]

**Condition evidence indices**

[8250, 8500, 8800, 9014]

**Decision indices**

[8250, 8500, 8800]

**Deployment condition source**

Agent selects remaining unmatched/incorrectly placed bar instance and requests its own explicit slot/orientation; final verification is external.

**Label derivation**

Common conversion: central/upper serif-bar track 8250->8500; lower bar track 8500->8800 and 8800->9014. Distinguish two physical instances despite potentially confusing H/I-like rotations. Last index is a real observation only; do not require 9015 or generate a terminal action target.

**Merge check**

Do not append a fabricated success/stop example. This short finishing interval shares shape policy with earlier coarse bar movements.

**Objective**

Final bar-piece pose adjustments and measured outcome evidence.

**Rationale**

8250/8500 show the upper bar repositioned; 8800/9014 show the lower one adjusted while the other pieces remain arranged.

**Split check**

Keep both instance-conditioned finishing choices together with outcome; not enough independent evidence for a fine-alignment specialist or success classifier.

**Supervision exclusions**

[]

**Supervision kind**

sparse

**Training condition**

Caller-selected bar track with hindsight geometric goal; no use of inferred final phrase.

**Uncertainty**

Episode endpoint is not task success; bar semantic labels/uprightness remain ambiguous until annotation. Final pose may be partially occluded.

### b_zigzag_and_clearance

Source: `episode_2026091922062101` [240, 1451). Supervision kind: sparse; bounds: 300, 1451; decision indices: [300, 650, 1000, 1300].

**Boundary evidence indices**

[240, 300, 1450]

**Condition evidence indices**

[300, 650, 1000, 1300]

**Decision indices**

[300, 650, 1000, 1300]

**Deployment condition source**

Current scene/slot assignment chooses zigzag instance or a blocking bar for clearance. Initial arrangement differs from A; no fixed initial pose assumed.

**Label derivation**

Common conversion: zigzag piece 300->650, 650->1000, 1000->1300; central-right serif-bar piece 1300->1450. Wrist 1000 disambiguates an outer contact despite third-view arm occlusion. 300 is pre-contact; 650/1000 are local contacts and changed orientations.

**Merge check**

Share with A zigzag and clearance decisions at model level; keep this source sequence separate and in a single episode split.

**Objective**

Turn/place the zigzag piece from a different initial arrangement and clear a neighbor.

**Rationale**

300 shows approach from above; 650/1000 show alternate-side turning; 1300/1450 show the neighbor shifted while the zigzag piece remains lower-right.

**Split check**

The actor switch at1300 is explicitly conditioned, not a dense sequence-level actor label. Pre-approach and in-contact cases use the same geometric primitive.

**Supervision exclusions**

[]

**Supervision kind**

sparse

**Training condition**

Per-anchor physical track, causal current geometry, tool/obstacles and hindsight target transform.

**Uncertainty**

No verified semantic identity or intent; third view is strongly occluded at1000. Use wrist/rigid tracking and mask unreliable targets.

### b_loop_and_ring

Source: `episode_2026091922062101` [1390, 2851). Supervision kind: sparse; bounds: 1450, 2851; decision indices: [1450, 1700, 1950, 2300, 2600].

**Boundary evidence indices**

[1390, 1450, 2850]

**Condition evidence indices**

[1450, 1700, 1950, 2300, 2600]

**Decision indices**

[1450, 1700, 1950, 2300, 2600]

**Deployment condition source**

Agent assigns explicit staging/slot poses for the rectangular-loop or ring track, and selects contacts using measured geometry.

**Label derivation**

Common conversion: rectangular-loop track 1450->1950 and1700->1950; ring track1950->2300,2300->2600,2600->2850. 1450 is prior to connecting motion;1700 wrist shows contact inside the rectangular loop.1950 is a pre-ring-contact observation after relocation of the other track.

**Merge check**

Retain both related interactions as actor-conditioned anchors with clutter context. Do not infer one continuous action through the switch or a common object identity.

**Objective**

Interior-boundary relocation and ring orientation/placement from a new route.

**Rationale**

1700 visibly pushes inside a hole;1950 shows loop relocated;2300/2600/2850 show ring rotations and translation near the zigzag piece.

**Split check**

Interior and exterior contact geometry differ but candidate representation covers both; no separate learned insertion/travel policy is required.

**Supervision exclusions**

[]

**Supervision kind**

sparse

**Training condition**

Selected track at each anchor with per-anchor hindsight pose, measured holes, tool envelope and surrounding geometry.

**Uncertainty**

Loop insertion clearance and actual tool contact height are unverified. Reject label/execution if calibration uncertainty prevents a safe fit.

### b_legged_recontacts

Source: `episode_2026091922062101` [2790, 4551). Supervision kind: sparse; bounds: 2850, 4551; decision indices: [2850, 3250, 3600, 3900, 4200].

**Boundary evidence indices**

[2790, 2850, 4550]

**Condition evidence indices**

[2850, 3250, 3600, 3900, 3930, 4200]

**Decision indices**

[2850, 3250, 3600, 3900, 4200]

**Deployment condition source**

Agent-selected legged instance with explicit target, requesting correction when measured orientation/position error remains.

**Label derivation**

Common conversion for the same legged bowl track:2850->3600,3250->3600,3600->3900,3900->4200,4200->4550. 2850 is pre-approach;3250/3600/3900 are local interactions;4200 is a lifted reapproach after an orientation change.3900/3930 provide a short outer-boundary stroke witness. Motion summaries report large joint-rate transients in this area; never imitate them as desired dq or infer object contact solely from them.

**Merge check**

Keep the move and its correction/reapproach together. Splitting the high-rate transition into a motor skill would train a capability the supplied controller already owns and could reward an unsafe transient.

**Objective**

Move and rotate a legged/concave piece, then reobserve and correct.

**Rationale**

3250 at its starting side,3600 translated/turned,3900 beside the ring,4200 tool lifted and4550 after another alignment show multiple contacts rather than one path.

**Split check**

Five explicit decisions share one model; pauses and transient robot posture changes remain context/executor evidence, not densely supervised control.

**Supervision exclusions**

[]

**Supervision kind**

sparse

**Training condition**

Persistent selected track, current contour/goal error and obstacles, causal history with no future pose input.

**Uncertainty**

Unexplained joint-rate spike is retained for integration audit, not labeled failure or successful force control. Contact/primitive compatibility near lift requires review.

### b_loop_finish

Source: `episode_2026091922062101` [4490, 5101). Supervision kind: sparse; bounds: 4550, 5101; decision indices: [4550, 4800].

**Boundary evidence indices**

[4490, 4550, 5100]

**Condition evidence indices**

[4550, 4800, 5100]

**Decision indices**

[4550, 4800]

**Deployment condition source**

Explicit loop-piece target pose from layout or correction request, not a fixed order after R-like manipulation.

**Label derivation**

Common conversion for rectangular-loop track4550->5100 and4800->5100.4550 is before approach;4800 is contact/correction;5100 supplies achieved loop pose while the tool has transitioned toward another piece.

**Merge check**

Keep separate from following open-corner segment so future targets cannot accidentally register the wrong object. Shared boundary is outcome here, next decision there.

**Objective**

Reorient/reposition the previously moved rectangular-loop piece.

**Rationale**

4550 shows angled loop and raised tool;4800 shows lower contact and corrected orientation;5100 retains the loop outcome.

**Split check**

Two anchors are sufficient for pre-arrival/local variants; splitting further would remove their useful outcome context.

**Supervision exclusions**

[]

**Supervision kind**

sparse

**Training condition**

Rectangular-loop track, measured contour and per-anchor hindsight spatial target with nearby arranged pieces protected.

**Uncertainty**

Fine yaw/position target may be limited by mask/calibration uncertainty; the shaped hole does not itself prove contact.

### b_open_corner

Source: `episode_2026091922062101` [5040, 6201). Supervision kind: sparse; bounds: 5100, 6201; decision indices: [5100, 5500, 5850].

**Boundary evidence indices**

[5040, 5100, 6200]

**Condition evidence indices**

[5100, 5500, 5530, 5850, 6200]

**Decision indices**

[5100, 5500, 5850]

**Deployment condition source**

Agent chooses the open-corner instance and a layout/staging pose; repeated contacts depend on observed response.

**Label derivation**

Common conversion for open-right-angle track5100->5500,5500->5850,5850->6200.5500/5530 wrist and state pair support a corner-contact local stroke;5850 is a different-side correction.6200 is the actual achieved arrangement before bar-piece work.

**Merge check**

Pool with A repeated corner attempts but do not concatenate trajectories. This shorter route is useful diversity, not proof the shape is mastered.

**Objective**

Choose corner contacts and successive corrections from a different initial orientation/location.

**Rationale**

5100/5500/5850 show changing sides/orientation and6200 shows the piece between other arranged shapes.

**Split check**

One sequence retains recontact history; not separate orientation and translation policies because both use contact/direction/distance output.

**Supervision exclusions**

[]

**Supervision kind**

sparse

**Training condition**

Selected open-corner track with full contour/hole-free topology, local obstacle context and hindsight target.

**Uncertainty**

Tip occlusion and unusual concavity contact require reviewed contour/tool geometry; no force or friction labels are available.

### b_bar_finish

Source: `episode_2026091922062101` [6140, 7730). Supervision kind: sparse; bounds: 6200, 7730; decision indices: [6200, 6500, 6850, 7200, 7500].

**Boundary evidence indices**

[6140, 6200, 7729]

**Condition evidence indices**

[6200, 6500, 6850, 7200, 7500, 7729]

**Decision indices**

[6200, 6500, 6850, 7200, 7500]

**Deployment condition source**

Agent chooses one of two distinct bar instances, gives an explicit target pose and verifies the complete request separately.

**Label derivation**

Common conversion: initially upper-left bar track6200->6850,6500->6850,6850->7200; initially central-right serif-bar track7200->7500; first bar track again7500->7729.6500 is raised reapproach,6850 a translation,7200 another instance correction,7500 final recontact.7729 is a real final observation only; no7730 successor is invented.

**Merge check**

Keep the return to the first bar in the same finishing sequence to preserve history and demonstrate revisiting an instance, rather than hardcoding one pass through a word.

**Objective**

Translate, revisit and align remaining bar pieces and retain unannotated final outcome.

**Rationale**

6200/6500 show reapproach,6850 moves one bar down,7200 adjusts the other,7500 returns to the first and7729 shows the final observed arrangement.

**Split check**

Do not create a separate terminal/success policy. Two physical tracks remain distinct despite visual rotational ambiguity.

**Supervision exclusions**

[]

**Supervision kind**

sparse

**Training condition**

Per-anchor bar-track selection and hindsight target, same shared shape policy and causal scene history.

**Uncertainty**

H/I-like semantic labels are not verified; final observation does not establish requested-word correctness, exact spacing or success.


## Heuristic 1

Learn one letter-identity-blind geometric contact/stroke selector. Use a small point-feature encoder (shared two-layer 64-unit MLP with permutation-invariant pooling) and a shared two-layer 64-unit candidate scorer conditioned on pooled actor/goal context, local curvature/normals, inner/outer-loop geometry, obstacle distance features, tool dimensions and recent measured motion. Softmax over feasible joint contact/direction/distance candidates; bounded features, weight decay and early stopping at episode-level validation. No large image encoder is trained from two episodes. Geometry canonicalization provides exact coordinate equivariance where valid; physical scale/friction are not assumed invariant. Learn supervised choice, not an uncalibrated object forward model. Stop/validity/progress and word-level decisions remain explicit deterministic or Agent responsibilities.

### Action decoding

Construct candidate contacts on all observed outer and inner contours, with outward normals pointing into adjacent free space. Sample 128 boundary points proportionally by arc length while reserving coverage of each accessible loop/corner. Pair each with directions at {-60,-40,-20,0,20,40,60} degrees about the inward normal and distances {5,10,20} mm clipped to <=0.2 diameter and caller limits. Omit candidates whose starting tool footprint cannot fit, whose push is outward/pulling, whose conservative swept envelope intersects protected objects/table edge, or whose required clearance exceeds uncertainty-adjusted free space. Network chooses a joint contact/direction/distance candidate, not three independent regressions; a ranked top three can be checked by the supplied planner. Contact point means a point on the object's side boundary at the chosen contact height, NOT TCP origin. Adapter uses calibrated full tool geometry to solve a tool pose that supports that contact and stroke, with table clearance; transforms it from P to base and to flange using the verified TCP/tool chain. Use deterministic near-normal-to-table tool alignment and a collision-checked yaw when possible; tilt/yaw are not assumed constant in recordings and their re-realization at deployment requires testing. If a demonstrated tilted/rotating contact cannot be realized by this primitive, flag the label unsupported rather than pretend it is identical. Chosen stroke is a local tool displacement, not a prediction that the object translates by that displacement. Small strokes plus measured feedback handle eccentric rotation and slip; no unverified dynamics planner is concealed here.

### Applicability

One visible tracked rigid, approximately planar letter piece resting on a calibrated table; actor contour and desired pose available; safe tool access to an outer or sufficiently wide inner boundary; surrounding letters/workspace modeled; physical size/thickness/tool envelope in commissioned range. Character identity, word and font are deliberately absent from policy input. Entry may be before robot travel, at a standoff, or at an existing safe contact. Geometry uncertainty, unsupported stacking, unstable tipping, an indistinguishable instance, unmodeled multi-object contact or a goal beyond reach produce a request for another view/staging/replanning, not a guessed push.

### Augmentation

Apply common SE(2) transforms to current/history contours, normals, tool footprints/poses, obstacle/workspace geometry, all goal contours and target contact/direction; translation changes points, rotation changes vectors/yaws. Apply in canonical geometry, not by pretending fixed-camera pixels have physically moved. Resample candidate points and use soft nearest-candidate labels. Photometric/crop perturbations belong only to frozen-perception robustness evaluation unless that dependency is separately authorized for fine-tuning. Add bounded contour/calibration noise consistent with estimated uncertainty to causal input, leaving the clean reviewed target as a denoising objective; reject topological changes/closed holes. Modest uniform scale tests must scale geometry, tool size, distances, heights and workspace together and retain metric size features; arbitrary object-only scaling is not a valid physics augmentation. Do not mirror character identities or relabel goals to unreachable words. Hindsight goal relabeling uses only the same tracked instance at the declared future f, never an unobserved desired pose. No synthetic success/failure labels from word appearance.

### Evidence

- `episode_2026091922002701`: [500, 700, 800, 830, 1100, 1600, 1800, 2100, 2400, 2800, 3500, 3700, 4300, 4800, 4810, 5100, 5600, 6000, 6200, 6700, 6900, 6930, 7200, 7600, 7800, 8250, 8500, 8800, 9014]
- `episode_2026091922062101`: [300, 650, 1000, 1300, 1450, 1700, 1950, 2300, 2600, 2850, 3250, 3600, 3900, 3930, 4200, 4550, 4800, 5100, 5500, 5530, 5850, 6200, 6500, 6850, 7200, 7500, 7729]
### Goal conditioning

For each current contour sample x_i in P, compute the vector from its current position to its position under caller G, plus global translation error and sin/cos yaw error with allowed symmetry hypotheses. Goal frame F is centered at actor centroid; its x-axis is toward target centroid if displacement >5 mm, otherwise the explicit layout baseline. Transform all geometry into F and normalize coordinates by actor diameter while retaining diameter, height, tool dimensions and stroke lengths in meters as scalar inputs. This deterministic canonicalization is the equivariance prior. A geometry-identical goal produces the same control problem regardless of word/character labels. Training uses explicit per-anchor future witness f and rigid registration to derive G; no future contour, observed future image, demonstrated manipulation order or future contact enters causal features. Progress/success tolerances are caller-specified and not silently loosened to fit calibration.

### Handoff

**entry_conditions**

Agent has selected one physical instance and an explicit feasible target/staging pose; Scene revision, plane calibration, tool model and uncertainty are valid; requested reading orientation has been resolved upstream. At the pre-approach call history is initialized from within-cut preceding rows or live causal observations; missing history is masked. At arrival/correction, use latest geometry, current contact state estimate and previous ticket outcome.

**exit_conditions**

Return proposed specification or deterministic already_at_goal, needs_view, no_safe_candidate, unreachable_or_blocked, stale_scene or unsupported_contact. A proposal is a decision completion, not task completion. Already_at_goal requires measured pose/contour error within requested tolerances for a stable observation interval; it is not a learned stop classifier.

**failure_signatures**

Masks merge/split, identity track changes, tool/object occlusion removes contact geometry, excessive calibration residual, lost contact, unexpected neighbor motion, slip/no progress, increasing pose error, torque/velocity guard, table-edge approach, infeasible insertion or planner failure. Confidence scores are not safety certificates. After two nonprogressing strokes on the same objective, request a new contact/view or staging goal; cap retries under caller budget and report blocked rather than oscillate indefinitely.

**overlap_role**

Boundary/buffer overlap supplies causal history and outcome verification without duplicate decision loss. The full executor trace supplies integration evidence only. Pre-motion and in-contact anchors share the same output schema; these are teleoperation states, not falsely labeled recorded planner-arrival events.

**successor_readiness**

Executor receives a fresh, reachable contact/stroke spec with object-contact allowance, full tool/robot scene geometry, margins and limits. Agent receives predicted choice plus uncertainty, current/goal error and a revision-linked history token. Following execution, only a new observation can authorize continuation; retreat/view/another actor uses the supplied executor, not another trained travel policy.

### Heuristic id

contour_contact_prior

### Input preprocessing

Raw assets must remain available: original 1280x720 third/wrist RGB, raw depth PNGs, all NPZ fields, action_json, frame mapping/timestamps/ages, metadata and both intrinsics sets. Tool previews are not training preprocessing. Offline and online: undistort RGB using the matching calibrated K/distortion; use T_base_cam for third and T_base_flange*T_flange_cam for wrist. Prefer metadata calibration wrist K paired with that hand-eye solution over mixing factory K without validation. Metadata has TCP xyz[0,0,0.2], rpy[0,0,-90] deg, pose consistency 3.230258 mm and reprojection RMS 4.420426 px. These are recorded calibration facts, not verified physical tip/table accuracy. Establish P, table normal, tabletop height, per-piece top heights, calibrated tip/contact surface and full tool collision envelope before metric labels or execution. Offline proposal: triangulate manually reviewed static tabletop/top-surface correspondences across calibrated views/poses and fit planes; validate against known-size metrology/board scale. Metadata board pose alone does not establish the actual contact plane. Online use a commissioned table/tool calibration and verify it from visible landmarks; take a new planner-generated view if needed. Depth is optional only after unit scale, intrinsics and RGB registration are independently verified; otherwise leave raw depth unused, never reinterpret PNG values as meters. SAM automatic proposals/propagation provide mask candidates; filter by table support, remove projected robot/tool, associate across views and rigid-contour tracks, and retain holes. VLM produces semantic/upright hypotheses separately, not control features. Offline a reviewer must assign persistent physical instance tracks, correct masks, mark occlusion/contact uncertainty and review the derived contacts; none of these annotations currently exists. Online replace reviewed labels with causal proposals, multi-view checks and rigid tracking; do not feed offline masks or future tracks at deployment. Fuse visible contours on each measured top plane, with uncertainty from calibration, segmentation and image age; use stable frames to initialize the shape and retain multiple pose hypotheses for symmetry. Never shift/re-pair source rows. Negative/positive img_age fields do not justify clock correction; paired-index consistency is not exposure synchronization. Reject geometry whose timing uncertainty at observed speed dominates the contact margin. Policy tensors: 256 arc-length contour points with xy, outward normal, local curvature, loop/visibility flags, per-point goal vector; 128 candidate contacts with sampled signed-distance/free-space features at the tool-start/stroke corridor; actor/context/goal/workspace/tool-footprint rasters 64x64 in F at a declared metric extent, used for deterministic distance-feature sampling; scalar metric size/height, current projected tip position, tolerances and uncertainty. Include at most last three causal geometry snapshots a,a-6,a-12 plus elapsed times/validity and last ticket result. Rows a-60..a provide track initialization, not future input. q and full transforms go to executor; no joint-configuration memorization in the policy. dq is measured state, not command; preserve null action values and expose missingness. gripper=-1/width NaN are invalid and never features. tau_ext is not ground-truth contact force or a calibrated wrench.

### Limitations

47 selected anchors are supervision locations, not 47 already validated metric contact labels. Future implementation must audit masks, plane/tool geometry, contact onset and linear primitive fit; record per-target validity/uncertainty and report actual usable counts rather than silently claiming all targets valid. Sparse candidates cannot establish a predictive dynamics model, reliable failure classifier or arbitrary-shape success. Two recordings use similar inventory/materials and scene. Internal-hole insertion, tip tilt changes and crowded corrections need especially careful geometry validation. If a label fails primitive fit, keep its raw evidence and mask invalid target channels; do not invent an easier stroke. The minimum-data policy may fail on narrow, top-heavy, flexible, disconnected or highly concave new letters. The required unseen-shape task is addressed through representation, feedback and targeted additional tests/data, but is not proven. External segmentation/semantic models can miss unusual shapes or confuse rotated H/I/W/M-like pieces; autonomous operation stops for unresolved ambiguity. Fine tolerances below validated uncertainty are unsupported, not silently relaxed.

### Pipeline implications

Target conversion common to all segments: each listed anchor a has actor track and future witness f in label_derivation. Derive hindsight G by uncertainty-aware rigid registration of that actor's causal reference contour to its observed contour at f; use geometric symmetry hypotheses and no word labels. Within [a,f], review the first intended contact with that actor (or c=a if already in contact) using both RGB views, calibrated tool surface and coincident object motion. Contact is not inferred from TCP speed or tau_ext alone. Express the contact at c on the actor boundary and pull it back through the actor's rigid transform to the shape at a. Extract the first contiguous approximately planar tool-surface stroke after c: terminate at loss of contact, another actor contact, a direction bend >15 degrees, substantial tool-orientation/height change, 1 s elapsed, 20 mm arclength, 0.2 actor diameter, or f, whichever comes first. Fit its net in-plane direction/displacement using measured transforms plus calibrated physical tool geometry; recorded v/w/dq are audit cues only, with units/frame decoding verified before any use. Quantize the contact/direction/length to feasible candidates with a soft tolerance kernel based on propagated uncertainty; do not label all unchosen candidates as physical failures. For pre-motion anchors, future c and stroke are training-only targets; inputs still end at a. For in-contact anchors, online current geometry replaces future contact estimation. Use shortest valid prefix, not the entire long manipulation, for action supervision. If insufficient motion, ambiguous contact, intervening object disturbance, topology mismatch or uncertainty exceeds margin, mark the relevant target invalid and retain review reason; if no target survives the anchor contributes no geometric-action loss and that count must be disclosed. There is no auto-generated task-success target. Train only at the 47 listed anchors, unique-key weighted, with NLL on soft joint-candidate labels and small regularization; optionally report contact/direction/length errors separately. Review annotations and derived metrics are future work, not published numeric replacements. Buffer history is at most 60 rows, masked at starts. No recurrent hidden state survives a changed instance, calibration or interruption. Whole-episode train/validation grouping and overlap accounting are defined above.

### Policy contract

**caller_arguments**

instance_id, target reference-contour transform G in P, metric position/yaw tolerances, scene_revision, plane/tool/robot calibration identifiers, protected tracks and workspace polygon, approach/stroke/force-guard limits, retry budget, baseline direction, optional last execution ticket/history token. The caller names a physical instance and objective, not an inferred demonstration label or command sequence.

**input_output_contract**

Causal Scene geometry/timing plus short history -> ranked joint contact/direction/distance specifications in P with validity flags, scene revision, uncertainty, and deterministic status. Point/length units are meters, angles radians; transforms use explicit source/destination frame tags. Contact points are on the piece boundary, not the flange/TCP. The adapter performs all metric transformations to executable robot objectives. Robot dq command vectors are never network output.

**memory_and_handoff**

Stateless network over up to three causal snapshots; external token records instance/reference-contour ID, scene/calibration revision, goal, last attempted candidate, ticket outcome and retry count. Initialize from within-segment buffers at training, from live observations at deployment; mask unavailable samples. Reset on new instance, unexpected track association, changed calibration, goal replacement or interruption. Retain previous measured response only when it concerns the same physical instance/goal and is causal.

**selection_cues**

Invoke for a selected rigid planar piece not at its requested geometric pose when at least one safe contact is possible. Same API before travel, at standoff, in contact and after correction. It is not selected by letter category or visible word. Request new perception or a planner view instead if geometry is obscured/uncertain; choose another/staging target if blocked.

**status_and_progress**

proposed means a local execution objective exists. already_at_goal requires independently measured stable pose error within caller tolerance. needs_view, ambiguous_geometry, no_safe_candidate, unsupported_contact, stale_scene and executor_replan are actionable non-success statuses. After each stroke report measured translation/yaw/contour error change, contact retention estimate, protected-object motion, elapsed stroke budget and retry count. Do not report probability of task success from softmax score.

### Policy id

shape_push_v1

### Rationale

Observed pushes are predominantly planar and use a small tool contact while letter motion can rotate or translate; planner-synthesizable travel between them is extensive. A800/830, A4800/4810, B3900/3930 and B5500/5530 demonstrate local contact motion; A6200-8250 shows why one long open-loop trajectory is not the transferable unit. Residual learning problem is selection of an effective local interaction for a requested object displacement. Bias: rigid-shape geometry, shared boundary scoring, symmetry-aware goal errors and short receding-horizon strokes. Mechanism: deterministic candidate generation plus one small shared MLP scorer using local contour/free-space features and pooled full-shape/goal context. Expected benefit is fewer learned degrees of freedom and transfer across positions, words and physical contours without motor-trajectory imitation. This benefit is a testable hypothesis, not established unseen-shape competence.


All scientific text above is unchanged Runtime API output. Rendering and slicing are developer-owned. No policy code or training exists in this stage.
