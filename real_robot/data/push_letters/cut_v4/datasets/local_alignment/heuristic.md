# Local alignment, final correction and revisits

Correct a staged or near-goal piece's footprint/orientation with short feedback-driven pushes, recontact and release while preserving neighboring placements.

## Dataset rationale

Six later distinct invocations after staging plus two late subintervals reused from difficult relocations. These share correction of a selected piece relative to a fixed local footprint with nearby pieces protected. They are not defined merely by low speed: free reposition/lift phases remain represented.15-row buffers preserve target entry and handoff context.

### a_dlike_align

Source: `episode_2026091922002701` [5715, 6215). Supervised interval: [5730, 6200).

**Boundary evidence indices**

[5715, 5729, 5730, 6199, 6200, 6214]

**Condition evidence indices**

[5730, 5850, 6199]

**Deployment condition source**

Agent requests D-like staged piece's corrected local pose.

**Label derivation**

G seed5730 D-like piece at rear-right, goal6199 reoriented/shifted there; E6199; original D labels.

**Merge check**

Same near-goal correction responsibility as B D-like revisit.

**Objective**

Correct staged D-like pose after branched-piece placement.

**Rationale**

5850 wrist shows edge/corner contact;6199 piece aligned and tool traversing away. Buffer includes prior R release and next L approach only as context.

**Split check**

Separate later correction from earlier D staging and intervening R relocation.

**Supervision exclusions**

[]

**Training condition**

instance=dlike_at5730; footprint6199; E6199; protect right-column neighbors.

**Uncertainty**

Alignment endpoint is inferred achieved pose, not verified word success.

### a_flanged_align

Source: `episode_2026091922002701` [8115, 8495). Supervised interval: [8130, 8480).

**Boundary evidence indices**

[8115, 8129, 8130, 8479, 8480, 8494]

**Condition evidence indices**

[8130, 8479]

**Deployment condition source**

Agent supplies corrected footprint for the already staged central flanged track.

**Label derivation**

G seed8130 left-middle flanged bar; goal8479 turned/shifted into its later pose. E8479; D labels.

**Merge check**

Revisit after earlier staging is common local alignment behavior, independent of object spelling.

**Objective**

Adjust position/orientation of staged central flanged piece.

**Rationale**

8130 shows staged oblique piece;8479 shows straighter left-row pose and release/lift. Context ties entry to L completion but no future online input.

**Split check**

Keep whole local correction including approach; stop before H-like revisit.

**Supervision exclusions**

[]

**Training condition**

instance=left_middle_flanged_at8130; footprint8479; E8479.

**Uncertainty**

Two flanged instances require stable tracking; semantic I/H not used as the ID.

### a_hlike_align_terminal

Source: `episode_2026091922002701` [8465, 9015). Supervised interval: [8480, 9015).

**Boundary evidence indices**

[8465, 8479, 8480, 9014]

**Condition evidence indices**

[8480, 9014]

**Deployment condition source**

Caller selects front-left H-like instance and explicit local correction goal.

**Label derivation**

G seed8480 front-left H-like cutout; achieved footprint9014 inferred with audited occlusion-aware tracking; E9014; D labels include actual terminal zero command.

**Merge check**

Same correction/revisit objective as B final H-like move, not episode-success classification.

**Objective**

Reposition front-left H-like piece and release/settle.

**Rationale**

8480 rod leaves previous flanged target;9014 H-like piece has shifted back/up and tool is lifted, with finite zero command. No synthetic row9015.

**Split check**

Terminal range ends at real N=9015; no after-buffer or N+1 observation invented.

**Supervision exclusions**

[]

**Training condition**

instance=front_hlike_at8480; footprint9014; E9014; protect all other placements.

**Uncertainty**

Arm occludes some neighbors; terminal scene not annotated successful. Goal mask requires audited tracking rather than perfect visibility assumption.

### b_dlike_align

Source: `episode_2026091922062101` [4695, 4925). Supervised interval: [4710, 4910).

**Boundary evidence indices**

[4695, 4709, 4710, 4909, 4910, 4924]

**Condition evidence indices**

[4710, 4909]

**Deployment condition source**

Agent requests a correction to the already staged D-like piece.

**Label derivation**

G seed4710 rear-right D-like piece; goal4909 reoriented above column; E4909; D labels.

**Merge check**

Same staged-object fine alignment as A, shorter and different preceding actions.

**Objective**

Correct D-like orientation/position after R-like placement.

**Rationale**

4710 tool approaches;4909 D-like piece straightened and tip departs toward L-like piece.

**Split check**

Keep separate from D staging1500-1910; interleaving is explicit.

**Supervision exclusions**

[]

**Training condition**

instance=dlike_at4710; footprint4909; E4909; protect column.

**Uncertainty**

No metric precision/success annotation supplied.

### b_flanged_align

Source: `episode_2026091922062101` [6915, 7345). Supervised interval: [6930, 7330).

**Boundary evidence indices**

[6915, 6929, 6930, 7329, 7330, 7344]

**Condition evidence indices**

[6930, 7329]

**Deployment condition source**

Caller requests local adjustment for middle flanged physical track.

**Label derivation**

G seed6930 center flanged bar above front H-like piece; goal7329 moved up/turned; E7329; D labels.

**Merge check**

Same local revisit after staging as A flanged alignment.

**Objective**

Align central flanged piece while preserving right column.

**Rationale**

6930 begins access from prior H-like relocation;7329 shows moved piece and tool departure.

**Split check**

Do not merge adjacent H-like target, even though geometry is similar.

**Supervision exclusions**

[]

**Training condition**

instance=middle_flanged_at6930; footprint7329; E7329; protect front H-like and right column.

**Uncertainty**

Geometric similarity can cause tracker swaps; reject ambiguous training identities.

### b_hlike_align_terminal

Source: `episode_2026091922062101` [7315, 7730). Supervised interval: [7330, 7730).

**Boundary evidence indices**

[7315, 7329, 7330, 7729]

**Condition evidence indices**

[7330, 7550, 7729]

**Deployment condition source**

Agent selects front-left H-like track for its explicit final local footprint.

**Label derivation**

G seed7330 front-left H-like piece; goal7729 shifted right/up; E7729; D labels, including real terminal zero command.

**Merge check**

Same local correction/release as A final revisit; episode end is not success label.

**Objective**

Finish a local H-like correction and settle tool.

**Rationale**

7550 shows contact correction;7729 shows final observed arrangement and lifted tip. No source row7730 exists.

**Split check**

Terminal segment has no following buffer; retain real final observation/action.

**Supervision exclusions**

[]

**Training condition**

instance=front_hlike_at7330; footprint7729; E7729; preserve others.

**Uncertainty**

No verified task-success label; monitor must independently validate scene.

### a_corner_late_align

Source: `episode_2026091922002701` [7785, 8145). Supervised interval: [7800, 8130).

**Boundary evidence indices**

[7785, 7800, 8129, 8130, 8144]

**Condition evidence indices**

[7800, 8129]

**Deployment condition source**

Agent can invoke alignment after coarse placement of a corner-shaped piece.

**Label derivation**

G seed7800 diagonal L-like piece near intended gap; goal8129 same goal as parent a_corner_move. E8129; D labels; no alternate success claim.

**Merge check**

Near-slot same-piece correction/recontact fits alignment objective; intentional cross-group reuse.

**Objective**

Correct final corner-piece orientation/spacing after gross motion.

**Rationale**

7800 still angled near column;8129 now in gap. Parent supplies full earlier approach, this cut supplies an independently invocable late entry.

**Split check**

Do not include early large L turns in local-only objective; no new target is introduced.

**Supervision exclusions**

[]

**Training condition**

instance=corner_at7800; footprint8129; E8129; protect neighboring column pieces.

**Uncertainty**

Locality is geometric, not guaranteed low speed; keep recontact motion.

### b_branched_late_align

Source: `episode_2026091922062101` [4035, 4725). Supervised interval: [4050, 4710).

**Boundary evidence indices**

[4035, 4050, 4709, 4710, 4724]

**Condition evidence indices**

[4050, 4200, 4650, 4709]

**Deployment condition source**

Agent invokes local correction after R-like piece reaches the column but retains pose error/contact difficulty.

**Label derivation**

G seed4050 branched piece already above ring; goal4709 identical to parent b_branched_move_retry; E4709; D labels. Future response labels only within this sequence.

**Merge check**

Repeated local correction plus lift/recontact shares alignment problem, not a separate robot-posture task.

**Objective**

Retain late R-like correction attempts, tool lift/posture adjustment and release.

**Rationale**

4050/4200 show large tool reposition over near-slot piece,4650 correction,4709 release. Both command/state and wrist images were inspected; high joint speed is not equated to unusable motion.

**Split check**

Late entry gives agent a recovery/retry option without replaying earlier gross relocation; no claim to recover arbitrary jams.

**Supervision exclusions**

[]

**Training condition**

instance=branched_at4050; footprint4709; E4709; protect ring and other column pieces.

**Uncertainty**

Cause of posture change unresolved; do not label it force-contact recovery or proven optimal action.


## Heuristic 1

Learn local goal-error correction with contact-mode memory, online observed-response adaptation and a conservative trust region.

### Action decoding

Use D with separate free-access and contact-stroke command heads. Predict finite original command.dq, with a q-conditioned local response/kinematic prior; fast free reposition examples remain supervised, not globally clipped in training as if all rows were slow contact. During verified contact use a conservative short-horizon trust region and reobserve; runtime clipping is reported. Learned local object response uses executed command history, not measured dq as its action label.

### Applicability

Selected piece already staged or near its desired footprint, neighbors should stay fixed, current track/goal residual measurable. Supports recontact and small rotations/translations, not large multi-piece rearrangement or rod entrapment recovery.

### Augmentation

Photometric changes and passive common-frame transforms as for boundary policy. Relabel goal only to confidently observed achieved poses within the same correction sequence; never generate attractive but unachieved small-error goals and claim demonstrated action validity. Add uncertainty/dropout to observation estimates with explicit masks while retaining true label provenance. No shape deformation with unchanged response targets.

### Evidence

- `episode_2026091922002701`: [5730, 5850, 6199, 7800, 8129, 8130, 8479, 8480, 9014]
- `episode_2026091922062101`: [4050, 4200, 4650, 4709, 4710, 4909, 6930, 7329, 7330, 7550, 7729]
### Goal conditioning

G selected physical instance, fixed target footprint/orientation ambiguity set, tighter caller tolerances bounded by uncertainty, protected instance IDs, E and retry/time budgets. Local error and response estimates update causally; goal never drifts toward current pose.

### Handoff

**entry_conditions**

Reliable near-goal/staged instance and protected-neighbor tracks, safe accessible boundary, verified controller/tool limits. If residual is outside locally trained envelope, recommend relocation.

**exit_conditions**

Current footprint/orientation match and protected pieces unchanged within tolerances for five valid observations; release/clearance or E condition satisfied. A recording endpoint alone is not success.

**failure_signatures**

Oscillation, sign-inconsistent object response, repeated no-progress contact, neighboring piece disturbance, torque anomaly, symmetry ambiguity, lost view or error leaving local envelope.

**overlap_role**

Six independent late revisits plus A late L/B late R overlapping full relocation sequences. Overlap trains agent-invocable local correction instead of forcing entire relocation to repeat; outside buffers are context only.

**successor_readiness**

Return stable local residual/uncertainty, neighbor changes and tool-contact status. Agent may verify layout, move another piece via access, or restage a blocked target.

### Heuristic id

local_response_trust_prior

### Input preprocessing

P/S/G with high-resolution target/wrist crops and local neighbor geometry, recent finite commands and proprioception. Learn a contact-mode recurrent network and a local object-response residual conditioned on boundary geometry; use causal secant/ensemble response updates from observed object displacement after executed commands. Contact probability is a weak inference from tip proximity, motion coupling and tau trends, never force ground truth. Select short translation/rotation strokes to reduce weighted contour and orientation error under a trust region and protected-neighbor penalties; allow distinct lift/reposition mode when the local contact response is poor. Train future object-delta/termination auxiliary labels only from valid rows inside materialized sequence; no future labels are online features.

### Limitations

Local feedback can still fail on novel shape/friction or ambiguous orientation. No calibrated wrench or slip labels. Demos contain moderate corrections and fast approach portions; not evidence of millimetre precision. Tolerances must respect calibration/perception uncertainty.

### Pipeline implications

Implement per-mode residual fitting, online response confidence, local contact candidate accessibility and stall/oscillation monitor. Validate fine-goal labels on all six revisit sequences and both overlaps; record support loss if mask/pose uncertainty prevents geometric training. Acquire new-shape correction and protected-neighbor data rather than asserting success.

### Policy contract

**caller_arguments**

policy_id=align_local_v1; selected instance_id, fixed goal footprint/optional verified SE(2), orientation alternatives, protected IDs, E, tolerances, maximum corrections/time and safety.

**input_output_contract**

Group local_alignment. Causal local geometry/dual RGB/proprio/history -> short joint velocity proposals through D plus local response/uncertainty diagnostics. Own correction/contact reset/release, not object reassignment.

**memory_and_handoff**

Initialize from current0.5s history; reset adaptation on instance/goal change or track loss; preserve only causal explicit response summaries when resuming same call. End status includes whether tip is still near contact.

**selection_cues**

Choose for late slot/orientation error, revisit after neighbors move, or a coarse policy reaches near-goal but not desired tolerance. Choose relocation for a large move, access for wrong-side tool placement, stop for unreliable identity.

**status_and_progress**

Report local contour/angle error, predicted versus observed stroke response, protected displacement, contact mode, retry count and uncertainty. Terminal status based on current verification, not low imitation loss.

### Policy id

align_local_v1

### Rationale

Both episodes revisit D-like and flanged pieces, and the long L/R attempts end with local pose corrections. A short-stroke response prior should avoid overshoot and shape-class memorization while allowing recontact; this needs tests on truly novel geometry and friction.


All scientific text above is unchanged Runtime API output. Rendering and slicing are developer-owned. No policy code or training exists in this stage.
