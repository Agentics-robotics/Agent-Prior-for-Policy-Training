# Single-piece relocation, staging and same-piece recontact

Move one caller-selected physical piece to a feasible staging or placement footprint, including same-piece recontact and conditioned tool handoff.

## Dataset rationale

Sixteen irregular same-instance objectives from both trajectories. Staging and final placement share a local geometry-to-goal problem; the agent supplies different goals and protected neighbors. Sequences retain nonmonotone attempts, lifts and side changes until target switch. Different targets are separate sequences; no concatenated continuity. Broad boundaries are within release/approach corridors and include E-conditioned tool departure, not force-certified contact splits.

### a_zigzag_move

Source: `episode_2026091922002701` [585, 1765). Supervised interval: [600, 1750).

**Boundary evidence indices**

[585, 599, 600, 1749, 1750, 1764]

**Condition evidence indices**

[600, 750, 1749]

**Deployment condition source**

G caller-selected physical zigzag or any other tracked geometry with an explicit local slot.

**Label derivation**

G seed600: zigzag cutout below rod near upper-left cluster; goal1749: same piece at front/right. E=T_base_ee1749. Original D labels.

**Merge check**

Same one-instance relocation objective as B zigzag and other shapes; no character embedding.

**Objective**

Relocate and turn zigzag piece out of initial cluster.

**Rationale**

750 wrist shows edge contact;1749 shows placed piece and rod departing. Context585-599 is approach history,1750-1764 next-target transition only.

**Split check**

Keep all turning/translation/recontact in one goal cycle; switch target only after1749.

**Supervision exclusions**

[]

**Training condition**

instance=zigzag_at600; achieved footprint1749; E1749; other pieces protected.

**Uncertainty**

W/M-like interpretation and desired spelling unverified; goal is achieved placement only.

### a_flanged_stage_one

Source: `episode_2026091922002701` [1735, 1945). Supervised interval: [1750, 1930).

**Boundary evidence indices**

[1735, 1749, 1750, 1929, 1930, 1944]

**Condition evidence indices**

[1750, 1929]

**Deployment condition source**

Agent may request a temporary staging footprint to open space.

**Label derivation**

G seed1750 central flanged bar immediately above ring, not upper H-like piece; goal1929 after leftward adjustment. E1929; D labels.

**Merge check**

Small staging displacement is still relocation to a supplied local goal, not word completion.

**Objective**

Shift central flanged piece to clear access around ring.

**Rationale**

1750 rod departs zigzag;1929 central flanged piece is shifted/turned.15-row external context disambiguates target switch.

**Split check**

Separate from zigzag and ring goals; do not label preceding zigzag action for this instance.

**Supervision exclusions**

[]

**Training condition**

instance=central_flanged_at1750; staging footprint1929; E1929.

**Uncertainty**

Clearance purpose is evidence-based interpretation, not annotated intent.

### a_ring_move

Source: `episode_2026091922002701` [1915, 3215). Supervised interval: [1930, 3200).

**Boundary evidence indices**

[1915, 1929, 1930, 3199, 3200, 3214]

**Condition evidence indices**

[1930, 2250, 3000, 3199]

**Deployment condition source**

Caller supplies selected ring-like instance and arbitrary feasible footprint.

**Label derivation**

G seed1930 lower oval ring; goal3199 right-side ring above zigzag. E3199; D labels. Ring symmetries retain multiple angle hypotheses.

**Merge check**

Same geometry-conditioned pushing with different outer/inner boundaries and starting orientation.

**Objective**

Translate/turn ring-like piece into right-side placement with recontact.

**Rationale**

2250 wrist supports outer-boundary push;3000 and3199 show aligned ring and rod release.15-row buffers cover entry/next transition.

**Split check**

Do not split approach, contact-side changes and final correction into separate object IDs.

**Supervision exclusions**

[]

**Training condition**

instance=ring_at1930; footprint3199; E3199; protect zigzag and other pieces.

**Uncertainty**

Oval symmetry means geometric orientation does not uniquely identify semantic upright.

### a_branched_stage

Source: `episode_2026091922002701` [3185, 3795). Supervised interval: [3200, 3780).

**Boundary evidence indices**

[3185, 3199, 3200, 3779, 3780, 3794]

**Condition evidence indices**

[3200, 3500, 3779]

**Deployment condition source**

Agent supplies a staging goal from current free-space reasoning.

**Label derivation**

G seed3200 left R-like branched piece above central flanged bar; goal3779 same piece near rear beside L-like piece. E3779; D labels.

**Merge check**

Staging and later return are distinct invocations of the same reusable local-goal problem.

**Objective**

Move R-like piece rearward, including rotation, to a temporary location.

**Rationale**

3500 shows turned piece under rod;3779 shows rear placement and tool traversing away. Intermediate proximity to flanged piece is not a target switch.

**Split check**

Later R return5050 is noncontiguous and separately indexed; do not concatenate.

**Supervision exclusions**

[]

**Training condition**

instance=branched_at3200; staging footprint3779; E3779.

**Uncertainty**

R label is visual mnemonic; exact contact onset unverified.

### a_flanged_stage_two

Source: `episode_2026091922002701` [3765, 4295). Supervised interval: [3780, 4280).

**Boundary evidence indices**

[3765, 3779, 3780, 4279, 4280, 4294]

**Condition evidence indices**

[3780, 4279]

**Deployment condition source**

Current flanged track and temporary agent layout/clearance goal.

**Label derivation**

G seed3780 central flanged bar left of ring; goal4279 moved up-left. E4279; D labels.

**Merge check**

Revisit of earlier staged piece is a new local-goal call, not new character-specific model.

**Objective**

Restage central flanged piece up-left.

**Rationale**

Start includes approach; end shows moved piece while tip heads toward the upper H-like piece. E explains departure.

**Split check**

Keep same-piece contact corrections; end before upper H-like transport.

**Supervision exclusions**

[]

**Training condition**

instance=central_flanged_at3780; footprint4279; E4279.

**Uncertainty**

Geometric identity must be tracked separately from visually similar H-like piece.

### a_hlike_stage

Source: `episode_2026091922002701` [4265, 4685). Supervised interval: [4280, 4670).

**Boundary evidence indices**

[4265, 4279, 4280, 4669, 4670, 4684]

**Condition evidence indices**

[4280, 4350, 4669]

**Deployment condition source**

Caller chooses upper H-like physical instance, not the other flanged piece.

**Label derivation**

G seed4280 upper H-like cutout under rod; goal4669 same piece at front-left beside zigzag/ring. E4669 above D-like access region; verify clear height. D labels.

**Merge check**

Relocation to temporary/front slot, sharing contact geometry mechanisms with B H-like transport.

**Objective**

Transport H-like piece from rear-central cluster to front-left staging.

**Rationale**

4350 wrist shows concave boundary contact;4669 H-like piece is front-left and tool over next D-like piece without its intended movement yet.

**Split check**

E-conditioned departure retained, but subsequent D-like movement belongs to next sequence.

**Supervision exclusions**

[]

**Training condition**

instance=upper_hlike_at4280; footprint4669; E4669; protect other flanged instance.

**Uncertainty**

End image projects rod into D hole, but EE is elevated; clearance requires calibration, not image overlap alone.

### a_dlike_stage

Source: `episode_2026091922002701` [4655, 5065). Supervised interval: [4670, 5050).

**Boundary evidence indices**

[4655, 4669, 4670, 5049, 5050, 5064]

**Condition evidence indices**

[4670, 5049]

**Deployment condition source**

Agent requests reachable temporary location for chosen enclosed D-like geometry.

**Label derivation**

G seed4670 left D-like piece with one large hole; goal5049 at rear-right of branched piece. E5049; D labels.

**Merge check**

Long staging move, same one-piece achieved-goal supervision as B D staging.

**Objective**

Move D-like piece from left/front to rear/right staging.

**Rationale**

4670 rod descending over D-like region;5049 piece displaced rearward and rod returning toward branched target.

**Split check**

Do not merge later D correction5730; intervening R-like move is another call.

**Supervision exclusions**

[]

**Training condition**

instance=dlike_at4670; footprint5049; E5049.

**Uncertainty**

Hole contact possible but no ground-truth contact labels.

### a_branched_return

Source: `episode_2026091922002701` [5035, 5745). Supervised interval: [5050, 5730).

**Boundary evidence indices**

[5035, 5049, 5050, 5729, 5730, 5744]

**Condition evidence indices**

[5050, 5250, 5729]

**Deployment condition source**

Agent chooses staged branched track and new right-column footprint.

**Label derivation**

G seed5050 rear branched piece left of D-like; goal5729 placed above ring; E5729; D labels.

**Merge check**

Same instance as earlier staging but different goal and surrounding layout; separate temporal sequence.

**Objective**

Return/relocate branched piece into column above ring.

**Rationale**

5250 shows approach/contact after staging;5729 shows relocated piece and rod moving toward D-like correction.

**Split check**

Retain access and recontacts, stop target responsibility before D-like correction.

**Supervision exclusions**

[]

**Training condition**

instance=branched_at5050; footprint5729; E5729; protect ring/zigzag.

**Uncertainty**

Achieved column arrangement is not a verified requested word.

### a_corner_move

Source: `episode_2026091922002701` [6185, 8145). Supervised interval: [6200, 8130).

**Boundary evidence indices**

[6185, 6199, 6200, 8129, 8130, 8144]

**Condition evidence indices**

[6200, 6400, 7500, 7800, 8129]

**Deployment condition source**

Caller specifies L-like or any selected shape and desired slot/orientation; policy chooses contacts.

**Label derivation**

G seed6200 left-rear L-shaped piece; goal8129 between D-like and branched pieces in right column. E8129; D labels; track through repeated turns rather than identity re-detection.

**Merge check**

Same local pose objective but harder nonconvex turning/recontact; retain variability rather than average it into straight push.

**Objective**

Rotate, reposition and translate L-like piece through several attempts into its column slot.

**Rationale**

6400 starts contact;7500 wrist shows inside-corner contact;7800 still correcting angle;8129 shows final local placement/release.

**Split check**

Long duration is warranted by one physical target/goal and repeated attempts. Late7800-8130 is separately reusable for alignment, not a gap.

**Supervision exclusions**

[]

**Training condition**

instance=corner_at6200; footprint8129; E8129; protect right-column pieces.

**Uncertainty**

Goal inference is achieved pose; no claim every intermediate attempt improved error.

### b_zigzag_move

Source: `episode_2026091922062101` [405, 1245). Supervised interval: [420, 1230).

**Boundary evidence indices**

[405, 419, 420, 1229, 1230, 1244]

**Condition evidence indices**

[420, 600, 1229]

**Deployment condition source**

Current zigzag track with explicitly supplied staging/final footprint.

**Label derivation**

G seed420 front zigzag via wrist420; goal1229 farther right and reoriented. E1229; D labels.

**Merge check**

Alternative initial arrangement/rotation of A zigzag objective, not a new skill.

**Objective**

Turn and relocate front zigzag piece.

**Rationale**

Wrist420 resolves arm occlusion;600 shows turned piece;1229 shows local placement and departure toward flanged piece.

**Split check**

Begin after access; keep repeated turning contacts, end before flanged displacement.

**Supervision exclusions**

[]

**Training condition**

instance=zigzag_at420; footprint1229; E1229.

**Uncertainty**

W/M appearance orientation is not semantic ground truth.

### b_flanged_stage

Source: `episode_2026091922062101` [1215, 1515). Supervised interval: [1230, 1500).

**Boundary evidence indices**

[1215, 1229, 1230, 1499, 1500, 1514]

**Condition evidence indices**

[1230, 1499]

**Deployment condition source**

Agent requests clearance/staging goal for the central flanged physical track.

**Label derivation**

G seed1230 central-right flanged bar, not rear-left H-like piece; goal1499 centrally staged to left of original position. E1499; D labels.

**Merge check**

Matches A clearance/revisit behavior under local geometry goals.

**Objective**

Shift central flanged piece out of the planned right column.

**Rationale**

1230 tip approaches its right boundary;1499 piece shifted left and rod departing.

**Split check**

Separate from front zigzag and subsequent D-like staging; retain entry and departure.

**Supervision exclusions**

[]

**Training condition**

instance=central_flanged_at1230; footprint1499; E1499.

**Uncertainty**

Clearance intent inferred; local achieved goal is unambiguous geometrically.

### b_dlike_stage

Source: `episode_2026091922062101` [1485, 1925). Supervised interval: [1500, 1910).

**Boundary evidence indices**

[1485, 1499, 1500, 1909, 1910, 1924]

**Condition evidence indices**

[1500, 1800, 1909]

**Deployment condition source**

Agent specifies reachable rear staging footprint for D-like instance.

**Label derivation**

G seed1500 large front-left D-like piece; goal1909 rear-right beside ring; E1909; D labels.

**Merge check**

Same long D-like staging as A, occurs earlier in this episode's order.

**Objective**

Move D-like piece rearward before ring relocation.

**Rationale**

1800 shows tool within/above displaced piece;1909 shows release and next-target access.

**Split check**

Do not use later D alignment goal as this segment's goal; staged endpoint is distinct.

**Supervision exclusions**

[]

**Training condition**

instance=dlike_at1500; footprint1909; E1909.

**Uncertainty**

D-like identity mnemonic and hole contact inferred, not provided labels.

### b_ring_move

Source: `episode_2026091922062101` [1895, 2835). Supervised interval: [1910, 2820).

**Boundary evidence indices**

[1895, 1909, 1910, 2819, 2820, 2834]

**Condition evidence indices**

[1910, 2400, 2750, 2819]

**Deployment condition source**

Selected ring-like track and explicit slot goal.

**Label derivation**

G seed1910 oval ring near rear-center; goal2819 above zigzag in right column. E2819; D labels; preserve symmetry alternatives.

**Merge check**

Same ring relocation across a different initial arrangement and preceding order.

**Objective**

Move ring downward/right and turn into column.

**Rationale**

2400 shows contact;2750 release;2819 transfer begins toward next target, captured by E.

**Split check**

Contact/side changes remain in one sequence; tool-only2750-2940 reuse has different explicit goal.

**Supervision exclusions**

[]

**Training condition**

instance=ring_at1910; footprint2819; E2819.

**Uncertainty**

Symmetry/upright orientation cannot be inferred uniquely from ring geometry.

### b_branched_move_retry

Source: `episode_2026091922062101` [2805, 4725). Supervised interval: [2820, 4710).

**Boundary evidence indices**

[2805, 2819, 2820, 4709, 4710, 4724]

**Condition evidence indices**

[2820, 3600, 3900, 4050, 4200, 4650, 4709]

**Deployment condition source**

Caller requests branched physical instance and slot above ring.

**Label derivation**

G seed2820 R-like piece on left; goal4709 above ring. E4709; D labels. Later pose/flow targets for response losses restricted to this sequence and confidence audited.

**Merge check**

Same geometry-to-goal responsibility with nonmonotone retries and tool posture change, not a new semantic stage.

**Objective**

Relocate R-like piece and repeatedly correct its pose/contact.

**Rationale**

3600 rearward staging within attempt,3900 column entry,4050/4200 lift/posture adjustment,4650 correction,4709 release. Larger measured joint velocities are retained, not called failure without evidence.

**Split check**

One target with multiple attempts: preserve history rather than cut at every pause.4050-4710 also trains local alignment under same achieved goal.

**Supervision exclusions**

[]

**Training condition**

instance=branched_at2820; footprint4709; E4709; protect ring/zigzag/D-like neighbors.

**Uncertainty**

No contact-force or operator intent labels; large robot posture change is observed, its cause unresolved.

### b_corner_move

Source: `episode_2026091922062101` [4895, 6155). Supervised interval: [4910, 6140).

**Boundary evidence indices**

[4895, 4909, 4910, 6139, 6140, 6154]

**Condition evidence indices**

[4910, 5400, 6000, 6139]

**Deployment condition source**

Agent gives target slot and orientation for selected corner-shaped piece.

**Label derivation**

G seed4910 rear-left L-shaped piece; goal6139 between D-like and branched pieces. E6139; D labels.

**Merge check**

Second nonconvex corner manipulation with a different start/order, shares full relocation objective with A.

**Objective**

Translate and rotate corner-shaped piece into column with recontact.

**Rationale**

5400 still diagonal under tool,6000 near slot,6139 release toward left H-like piece. Buffers retain approach/handoff context.

**Split check**

Keep successive recontacts as one same-piece goal cycle; do not include next H-like displacement.

**Supervision exclusions**

[]

**Training condition**

instance=corner_at4910; footprint6139; E6139; protect neighboring column pieces.

**Uncertainty**

Goal comes from achieved pose, not inferred spelling.

### b_hlike_stage

Source: `episode_2026091922062101` [6125, 6945). Supervised interval: [6140, 6930).

**Boundary evidence indices**

[6125, 6139, 6140, 6929, 6930, 6944]

**Condition evidence indices**

[6140, 6600, 6929]

**Deployment condition source**

Agent identifies rear-left H-like physical instance and supplied temporary/front goal.

**Label derivation**

G seed6140 rear-left H-like cutout; goal6929 moved to front-left below central flanged piece. E6929; D labels.

**Merge check**

Relocation/staging responsibility as A upper H-like move, with a later revisit in both episodes.

**Objective**

Bring rear H-like piece to front-left staging.

**Rationale**

6600 shows contact after access;6929 H-like piece is front-left and rod approaches other flanged instance.

**Split check**

Later7330 H-like adjustment is a separate call after another piece moved; no concatenation.

**Supervision exclusions**

[]

**Training condition**

instance=rear_hlike_at6140; footprint6929; E6929.

**Uncertainty**

Track must not swap the two similar flanged shapes.


## Heuristic 1

Learn geometry-shared contact proposals and a short-horizon object-response ensemble for closed-loop single-piece relocation.

### Action decoding

Use D with a geometric contact-motion proposal and q-conditioned kinematic/residual joint decoder trained on finite action.dq. Auxiliary v/w targets retain their original base/EE frames. The learned forward response predicts object displacement, not measured robot dq as action. Execute only the first safe proposal, reobserve and replan at verified cadence; do not apply open-loop recorded joint trajectories.

### Applicability

One selected rigid planar instance, reliable current contour/track, feasible local footprint goal and commissioned geometry. Supports outer boundaries, holes and concavities where rod fit/access is verified; can choose recontact and intermediate pushes internally but not switch physical target without a new agent call.

### Augmentation

Use passive SE(2) coordinate re-expression of contour points, normals, tool state, fixed goal and task-space labels together; q and physical joint command labels remain unchanged and the decoder receives the frame transform. Metric physical translation/rotation augmentations require validated IK and scene/collision regeneration, not naive dq rotation. Photometric changes only affect perception inputs. Do not distort shapes and retain old action/object-response labels: novel-shape dynamics require new evidence or a separately validated simulator, not fake demonstration relabels.

### Evidence

- `episode_2026091922002701`: [750, 2250, 3500, 4350, 5250, 6400, 7500, 7800]
- `episode_2026091922062101`: [600, 2400, 3600, 3900, 4200, 5400, 6600]
### Goal conditioning

G instance track, fixed achieved or requested footprint/SE(2), semantic upright cue only if supplied by agent, protected masks, optional E waypoint. Mode proposals use current goal-relative contour error, not a learned W/O/R class embedding.

### Handoff

**entry_conditions**

Selected instance geometrically distinguishable, stable support, safe tip approach available or supplied by access policy, reachable goal and protected-neighbor map.

**exit_conditions**

Observed object footprint/orientation within uncertainty-aware tolerance for five valid observations; optionally E reached safely. Otherwise intermediate progress or NEED_REPOSITION is reported rather than terminal success.

**failure_signatures**

Wrong-piece motion, slip/stall, unexpected rotation, rod trapped in hole/concavity, large model disagreement, lost identity, edge risk or protected-piece displacement.

**overlap_role**

Full same-instance cycles retain approach/recontact/withdrawal supervision.15-row external buffers are history/handoff only. Late L/R portions overlap local_alignment with identical goal provenance.

**successor_readiness**

Return current object footprint, uncertainty and tip/contact state. Agent may call alignment, access, or another staging relocation; completed local placement does not authorize unchecked next motion.

### Heuristic id

boundary_response_prior

### Input preprocessing

P/S/G. Sample outer and inner contour points with normals/visibility and instance membership; build a permutation-invariant local graph containing rod boundary, goal contour and nearby obstacles in a common goal-relative table frame when verified, otherwise third-UV with projective uncertainty. An equivariant graph proposal network scores boundary contacts and depart/approach/push modes. A small learned response ensemble predicts observed rigid object delta over short horizons from local geometry, current tool state and ORIGINAL commands. Train response targets only from valid tracked poses at later rows within this segment; they are auxiliary labels, never online inputs. Online update response uncertainty from past executed actions/current visual motion without privileged friction or force labels.

### Limitations

Quasi-static/local response is approximate, not guaranteed for arbitrary fonts, mass, friction, thin branches or multi-object jams. Two recordings provide little dynamics diversity. Calibration/shape visibility can make this policy unavailable; choose visual alternative or stop, not invented contours.

### Pipeline implications

Implement boundary sampling, topology-preserving holes, equivariant network, contact candidate accessibility, ensemble dynamics, collision-aware short-horizon optimizer and command decoder. Acquire mask/track annotations and tool/plane assets; quantify which staging/retry/contact modes survive preprocessing. Do not replace difficult missing-pose examples with claimed geometric supervision.

### Policy contract

**caller_arguments**

policy_id=push_boundary_v1; instance_id, fixed goal footprint and optional verified SE(2), orientation ambiguity set, protected IDs, E, tolerances/time/safety.

**input_output_contract**

Group piece_relocation. Current causal RGB/geometry/proprioception/history -> proposed seven-joint velocities through D plus predicted object response/mode confidence. Own local contact choice, same-piece recontact and departure.

**memory_and_handoff**

Reset goal-specific state on call; carry causal tracker state and past-response uncertainty only. Keep model adaptation local to current tool/material context; reset on shape/material change or track loss.

**selection_cues**

Prefer when contours and table geometry are reliable, especially unseen shape with clear boundary, large translation/turn or deliberate staging. Switch to visual alternative for geometric ambiguity only if semantic identity and safety remain reliable.

**status_and_progress**

Expose contour/SE(2) residual, predicted/observed displacement, ensemble disagreement, active boundary region and stall count. Report LOCAL_GOAL_REACHED, NEED_REPOSITION or BLOCKED with evidence.

### Policy id

push_boundary_v1

### Rationale

The rod pushes different geometries and deliberately changes contact sides; A L-like cycle and B R-like retries show that translation-only centroid control is insufficient. Shared local boundary response and feedback should generalize physical interactions more plausibly than per-character action lookup, but require independent novel-shape testing.


## Heuristic 2

Learn a selected-instance, fixed-goal-mask dual-view temporal command policy with short receding-horizon chunks.

### Action decoding

Predict continuous-time short-horizon seven-joint command chunks, trained on finite ORIGINAL action.dq at original source t queries; auxiliary heads predict recorded v/w and near-term visual displacement. At deployment query the first control-time command and use D safety checks. Never substitute measured dq, fabricate missing commands, or blindly replay an entire chunk.

### Applicability

Same selected-instance/goal responsibility as push_boundary_v1, useful when partial occlusion or appearance makes hard contour registration unstable but dual-view track identity and safety bounds remain trustworthy. Not a fallback for absent target identity or uncalibrated robot execution.

### Augmentation

Synchronized color/lighting changes, sticker-text masking, background randomization and explicit camera/patch dropout across a history; apply target/goal mask coordinate warps consistently with image crops and projected tool features. Valid passive view-coordinate changes leave physical q/dq labels intact. No horizontal reflection that changes letter semantics without updating orientation cue; no artificial shape replacement with unchanged action labels.

### Evidence

- `episode_2026091922002701`: [600, 750, 1749, 2250, 4350, 7500]
- `episode_2026091922062101`: [419, 420, 3900, 4050, 4200, 4709]
### Goal conditioning

Same G/E public interface. Rasterize desired footprint and orientation cue on current third-view input, with selected-instance mask and track token. Future goal RGB is never an input: training uses only derived goal silhouette/pose, and online render uses caller geometry plus current observed entry shape.

### Handoff

**entry_conditions**

Trustworthy selected instance, at least one usable camera view, calibrated safety envelope and feasible goal; history masking handles unavailable second view.

**exit_conditions**

External current-observation goal/clearance monitor, not chunk length or recorded duration, declares completion.

**failure_signatures**

Cross-view identity mismatch, attention on wrong piece, repeated action chunk disagreement/clipping, no visible progress, drift under prolonged occlusion or collision risk.

**overlap_role**

Full relocation cycles teach visual continuation through occlusion/recontact, with external15-row buffers available only causally. Same late overlaps as boundary policy; not extra independent data.

**successor_readiness**

Return goal residual/track confidence and actual tip state; caller decides alignment or access. No hidden spelling progress state is passed.

### Heuristic id

dual_view_mask_goal_prior

### Input preprocessing

P RGB full views plus original-coordinate target/tool crops, per-view validity, q/measured dq/tau/T_base_ee, previous executed commands and time deltas over0.5s. A pretrained visual encoder with a small independently trained temporal goal-cross-attention/action-chunk model uses selected-object and desired-footprint channels. Training-only mask/flow annotations may supervise attention, pose-progress and cross-view consistency; online masks come from causal S. Use camera-dropout training so partial arm occlusion (B419) is not mistaken for target absence. This policy does not require metric object pose as an input, though the common execution safety envelope must still be commissioned.

### Limitations

Visual representations may memorize brown material/background and do not establish new-shape manipulation. Different letters can have unseen dynamics even if the encoder recognizes them. Encoder assets and sufficient geometry-diverse data are missing; no claim of end-to-end open-world competence.

### Pipeline implications

Acquire compatible visual weights and licenses; implement goal raster/crop mapping, temporal masks and continuous-time chunk labels without re-pairing. Audit goal-channel leakage, sticker dependence and view availability per behavior; train and evaluate independently of boundary policy.

### Policy contract

**caller_arguments**

policy_id=push_visual_v1; same instance_id/goal/protected_ids/E/tolerance/time/safety interface as boundary alternative; caller need not choose a contact point.

**input_output_contract**

Group piece_relocation. Causal dual RGB/proprio/history plus current masks and fixed goal raster -> short command proposals through D and diagnostic confidence. Own same-piece pushing/recontact, not word semantics.

**memory_and_handoff**

Initialize from bounded current history; reset action chunk on new goal, track uncertainty or stop; no hidden state copied from boundary policy. Transfer only shared explicit observations/track state.

**selection_cues**

Choose for robust dual-view observations with unreliable hard contour pose, or as an independently validated alternative after a geometric proposal stalls. Do not select solely because a character is unseen.

**status_and_progress**

External geometric/image residual and internal ensemble/chunk uncertainty, camera availability and track confidence. Explicit BLOCKED/LOST_TRACK rather than action completion by elapsed time.

### Policy id

push_visual_v1

### Rationale

Third views are repeatedly obscured by the arm while wrist views show the actual contact boundary. Preserving dual-view temporal evidence offers a substantive alternative to committing to a single accurate contour/pose estimate; generalization to appearance and unseen geometry remains an evaluation hypothesis.


All scientific text above is unchanged Runtime API output. Rendering and slicing are developer-owned. No policy code or training exists in this stage.
