# Relocate or align one planar physical piece

Move one explicitly selected flat physical instance to a caller-specified local footprint/pose, including temporary clearance, staged placement, repeated contact changes, fine alignment and safe handoff.

## Dataset rationale

Twenty-two one-instance invocations are separated where the active piece changes or a previously staged piece is revisited. Both large relocations and small pose refinements share a geometry-to-goal objective; do not split L-like contact retries into disconnected successes. Twenty-row context buffers cover causal initialization and handoff auditing, while supervised approaches and exit-tool goals account for useful transit. Every inference of piece identity is morphological/track-based, not a supplied character label.

### a_wedge_move

Source: `episode_2026091922002701` [580, 1760). Supervised interval: [600, 1740).

**Boundary evidence indices**

[580, 599, 600, 1739, 1740, 1759]

**Condition evidence indices**

[600, 1300, 1739]

**Deployment condition source**

HLA selects a current wedge-shaped physical instance by tracker ID and supplies its requested slot footprint/exit waypoint.

**Label derivation**

RULE P from selected wedge piece at600 to its footprint at1739; exit TCP/twist at1739. 1300 wrist/third views verify the target is the wedge, not adjacent R-like.

**Merge check**

Same rigid planar instance-to-goal problem as B wedge and all other pieces; geometry, not character identity, expresses variation.

**Objective**

Extract and rotate/translate the wedge-shaped piece from the upper cluster to the foreground-right area.

**Rationale**

650 precedes large motion,1300 shows side contact,1550 and1739 show foreground arrival; keep approach and clearance supervision with20-row context on each end.

**Split check**

End before middle barred-piece manipulation; do not train its later displacement under the wedge ID.

**Supervision exclusions**

[]

**Training condition**

Selected wedge-shaped W/M-looking piece initially above R-like; achieved footprint at1739 is a local hindsight goal, other instances protected subject to observed incidental-motion audit.

**Uncertainty**

W versus M is a visual orientation interpretation, not a verified semantic label; exact contact onset is weakly inferred.

### a_bar_clear

Source: `episode_2026091922002701` [1720, 2010). Supervised interval: [1740, 1990).

**Boundary evidence indices**

[1720, 1739, 1740, 1989, 1990, 2009]

**Condition evidence indices**

[1740, 1950, 1989]

**Deployment condition source**

HLA supplies a temporary clearing pose for the currently obstructing barred-piece instance.

**Label derivation**

RULE P for the barred piece initially between upper barred and O-like, with goal/exit at1989.

**Merge check**

Clearing a blocker and spelling placement differ only in local geometric goal, so share the same policy.

**Objective**

Shift the middle barred piece leftward to open the O-like approach corridor.

**Rationale**

1740 enters from wedge clearance;1950/1989 show the middle barred piece shifted while O-like remains. Pre/post20 rows are context only.

**Split check**

Separate this supporting displacement from both wedge and O-like movement rather than hiding it as an approach accident.

**Supervision exclusions**

[]

**Training condition**

Target middle barred piece visible near R/D/O, not the upper barred piece; achieved pose1989; explicit temporary-clearance goal.

**Uncertainty**

Interpretation as deliberate clearance is inferred from sequence; no intention annotation or canonical I/H label exists.

### a_ring_move

Source: `episode_2026091922002701` [1970, 3240). Supervised interval: [1990, 3220).

**Boundary evidence indices**

[1970, 1989, 1990, 3219, 3220, 3239]

**Condition evidence indices**

[1990, 2600, 3219]

**Deployment condition source**

HLA assigns the ring-like instance a metric footprint goal and chosen safe tool exit.

**Label derivation**

RULE P on O-like ring, endpoint3219; symmetry handled by contour plus semantic orientation uncertainty.

**Merge check**

Combines with B ring transport and other shape-normalized motion without a ring-specific policy.

**Objective**

Translate/rotate O-like toward the placed wedge and refine its alignment.

**Rationale**

1990 shows precontact approach,2600 shows displaced ring,3100/3219 capture final adjustment/clearance. All repeated short strokes retained.

**Split check**

Do not merge the next R staging with ring alignment; retain internal reapproaches because the instance/goal stays fixed.

**Supervision exclusions**

[]

**Training condition**

Selected oval ring originally foreground-left; goal footprint3219; exit TCP/twist3219.

**Uncertainty**

No exact ring yaw is claimed when contour symmetry makes it unobservable; small unintended neighbor motion requires monitoring.

### a_r_stage

Source: `episode_2026091922002701` [3200, 3720). Supervised interval: [3220, 3700).

**Boundary evidence indices**

[3200, 3219, 3220, 3699, 3700, 3719]

**Condition evidence indices**

[3220, 3400, 3699]

**Deployment condition source**

HLA chooses a temporary reachable staging footprint, based on current occupancy rather than final word order.

**Label derivation**

RULE P for R-like piece initially left of upper barred piece, goal/exit3699.

**Merge check**

Same piece appears again in a_r_layout with a different goal; preserve separate calls rather than contradictory labels in one sequence.

**Objective**

Move R-like out of the initial cluster to an upper staging pose.

**Rationale**

3400 shows rod at R-like;3699 shows R-like moved upward with tool clear. Include entry from ring and departure context.

**Split check**

A distinct temporary-goal invocation ends before middle barred-piece work; later R placement is noncontiguous.

**Supervision exclusions**

[]

**Training condition**

Target R-shaped physical piece near left cluster; achieved staging pose3699; not its final episode pose.

**Uncertainty**

R is a descriptive shape alias; temporary-goal interpretation is inferred, not task annotation.

### a_bar_stage

Source: `episode_2026091922002701` [3680, 4320). Supervised interval: [3700, 4300).

**Boundary evidence indices**

[3680, 3699, 3700, 4299, 4300, 4319]

**Condition evidence indices**

[3700, 3900, 4100, 4299]

**Deployment condition source**

HLA picks the current middle barred-piece track and a clearing/staging footprint.

**Label derivation**

RULE P for the previously shifted middle barred instance; goal and exit4299.

**Merge check**

Repeated invocation of the same physical piece with a new local goal; shares contact mechanics with B central barred moves.

**Objective**

Reorient and move the middle barred piece toward the left staging area.

**Rationale**

3900 wrist shows near-boundary tool adjustment,4100 a new contact side,4299 shows this piece staged left while rod approaches the upper barred piece. Keep all posture adjustments.

**Split check**

Separate at upper-bar target change; optional exit-tool goal explains final travel without labeling upper-bar displacement here.

**Supervision exclusions**

[]

**Training condition**

Selected barred piece now left/below upper barred, same track as a_bar_clear; endpoint4299 footprint and tool waypoint.

**Uncertainty**

Visually similar barred pieces require persistent track audit; semantic I/H assignment is not supplied.

### a_upper_bar_move

Source: `episode_2026091922002701` [4280, 4660). Supervised interval: [4300, 4640).

**Boundary evidence indices**

[4280, 4299, 4300, 4639, 4640, 4659]

**Condition evidence indices**

[4300, 4350, 4550, 4639]

**Deployment condition source**

HLA selects the other barred-piece instance and its planned footprint.

**Label derivation**

RULE P for upper barred piece at4300, endpoint4639; tool exit may be above D-like but may not push D-like.

**Merge check**

Same relocation objective with a recess contact, not a separate letter class policy.

**Objective**

Move the upper barred piece down toward foreground-left.

**Rationale**

4350 begins contact on upper barred piece;4550 wrist shows rod in its recess;4639 piece has moved foreground-left and tool has lifted/traveled toward D-like.

**Split check**

Stop before D-like pushing; materialized successor rows only audit its precontact state.

**Supervision exclusions**

[]

**Training condition**

Upper barred instance near staged R at4300, not the already left-staged barred piece; goal4639, exit4639.

**Uncertainty**

Tool contact within the recess is visually supported, not force-labeled; safe recess width must be measured online.

### a_d_stage

Source: `episode_2026091922002701` [4620, 5220). Supervised interval: [4640, 5200).

**Boundary evidence indices**

[4620, 4639, 4640, 5199, 5200, 5219]

**Condition evidence indices**

[4640, 4750, 5199]

**Deployment condition source**

HLA selects D-like instance and a temporary or final slot footprint with an accessible hole/contact side.

**Label derivation**

RULE P for D-like from foreground-left to endpoint5199; track nearby R-like as a protected object and audit incidental motion.

**Merge check**

Pairs with B D relocation and both later D refinements; same goal-conditioned planar transformation.

**Objective**

Move D-like from foreground-left to upper-right using its internal opening.

**Rationale**

4750 wrist/third show the rod inside D-like;4950/5199 show it at upper-right. Retain approach, long displacement, lift and exit.

**Split check**

Separate later R manipulation and later D orientation correction instead of assigning final D pose to every earlier move.

**Supervision exclusions**

[]

**Training condition**

D-shaped ring piece initially left/foreground; achieved pose5199, exit5199.

**Uncertainty**

Inner-wall versus top-edge contact must be audited at implementation; later movement means this is not declared final success.

### a_r_layout

Source: `episode_2026091922002701` [5180, 5690). Supervised interval: [5200, 5670).

**Boundary evidence indices**

[5180, 5199, 5200, 5669, 5670, 5689]

**Condition evidence indices**

[5200, 5450, 5669]

**Deployment condition source**

HLA revisits a staged instance and supplies a new local footprint goal from the current layout.

**Label derivation**

RULE P for upper R-like piece, endpoint5669; D-like is tracked separately even where nearby.

**Merge check**

Same instance as a_r_stage, different invocation/goal; combine in training via current geometry and goal, never by temporal concatenation.

**Objective**

Rotate/translate staged R-like into the column above O-like.

**Rationale**

5200 begins near staged R,5450 shows off-center turning close to D,5669 shows R near its achieved column pose.

**Split check**

End before D orientation work; retain local corrections and close-neighbor evidence.

**Supervision exclusions**

[]

**Training condition**

Target R-like adjacent to D at5200; achieved footprint5669 and tool exit5669; not inferred word conditioning.

**Uncertainty**

Close R/D proximity can cause coupled motion; neighbor displacement is a monitored outcome, not an authorized second target.

### a_d_align

Source: `episode_2026091922002701` [5650, 6140). Supervised interval: [5670, 6120).

**Boundary evidence indices**

[5650, 5669, 5670, 6119, 6120, 6139]

**Condition evidence indices**

[5670, 5900, 6119]

**Deployment condition source**

HLA requests a pose refinement on the current D-like track.

**Label derivation**

RULE P for upper D-like, goal/exit6119.

**Merge check**

Fine yaw/translation correction shares the same learned contact variables as gross D transport; tolerance/goal error expresses scale.

**Objective**

Rotate and refine D-like at the top of the emerging column.

**Rationale**

5670 D remains oblique;5900 shows contact after rotation;6119 shows aligned D and departing tool.

**Split check**

End before L work; final tool transit is retained under explicit exit waypoint, with20-row context.

**Supervision exclusions**

[]

**Training condition**

Target D-like above R-like; local achieved pose6119, not the full episode arrangement.

**Uncertainty**

No externally specified orientation or placement tolerance is recorded; endpoint relabeling is hindsight supervision only.

### a_l_move

Source: `episode_2026091922002701` [6100, 8170). Supervised interval: [6120, 8150).

**Boundary evidence indices**

[6100, 6119, 6120, 8149, 8150, 8169]

**Condition evidence indices**

[6120, 6500, 6800, 7100, 7400, 7800, 8050, 8149]

**Deployment condition source**

HLA selects L-like or a novel concave instance and supplies a footprint goal; policy chooses multiple contacts causally.

**Label derivation**

RULE P for upper-left L-like, achieved footprint8149 and exit8149; preserve all intermediate reversals rather than label them failures or independent completed goals.

**Merge check**

Combines with B L movement as long multi-stroke examples of the same one-piece goal problem; no L-specific policy.

**Objective**

Repeatedly recontact, turn and transport L-like into the gap between R-like and D-like.

**Rationale**

6500,6800,7100,7400 show different L poses/contact sides;7800/8050 show transport/final refinement;8149 shows completed local placement and exit toward barred piece. Early6120-6250 also supports tool_position.

**Split check**

Do not split at every pause or unsuccessful rotation: selected instance and final local purpose remain constant. Split when another barred piece becomes active.

**Supervision exclusions**

[]

**Training condition**

Selected two-arm L-like piece initially upper-left; fixed hindsight footprint8149, explicit exit waypoint8149; mode/history distinguish resets from pushes.

**Uncertainty**

Long nonmonotonic progress is observed; whether individual turns were failed attempts or intentional intermediate contacts is unresolved. All remain useful action evidence.

### a_bar_align

Source: `episode_2026091922002701` [8130, 8530). Supervised interval: [8150, 8510).

**Boundary evidence indices**

[8130, 8149, 8150, 8509, 8510, 8529]

**Condition evidence indices**

[8150, 8500, 8509]

**Deployment condition source**

HLA revisits the left-staged barred instance and specifies a final local layout slot.

**Label derivation**

RULE P for left-staged barred piece, endpoint8509 and tool exit8509.

**Merge check**

Repeated same-instance refinement combines with its earlier clearance/staging under different goals.

**Objective**

Rotate/translate left-staged barred piece beside the column.

**Rationale**

8150 rod approaches the left-staged piece;8300 rotation visible in survey;8500/8509 show its achieved side placement and departure.

**Split check**

Separate from both L and foreground barred-piece work; end while tool is clear of this instance.

**Supervision exclusions**

[]

**Training condition**

Target left-staged barred piece from a_bar_stage, not foreground barred; goal8509 and exit8509.

**Uncertainty**

Similar barred shapes must be tracked, not relabeled by apparent character after rotation.

### a_upper_bar_align

Source: `episode_2026091922002701` [8490, 9015). Supervised interval: [8510, 9015).

**Boundary evidence indices**

[8490, 8509, 8510, 8950, 9014]

**Condition evidence indices**

[8510, 8950, 9014]

**Deployment condition source**

HLA selects the remaining foreground barred instance and requests its slot plus a safe stopped tool pose.

**Label derivation**

RULE P at9014; A8950 and9014 actual zero twist supervise holding. No fabricated terminal successor or word-success label.

**Merge check**

Same target class of problem as B final barred refinement, with a different starting arrangement.

**Objective**

Bring foreground barred piece upward into a side arrangement, retract and hold.

**Rationale**

8510 enters from previous placement;8750 shows foreground contact in survey;8950/9014 show changed piece placement and stable terminal tool. Only20-row prefix because source ends.

**Split check**

Keep terminal withdrawal/hold as supporting behavior; episode endpoint does not define task success.

**Supervision exclusions**

[]

**Training condition**

Target foreground-left barred piece moved by a_upper_bar_move; achieved footprint9014 and stopped tool endpoint.

**Uncertainty**

Final visual arrangement resembles lettering but no requested word or success annotation is supplied; tool occludes part of scene.

### b_wedge_move

Source: `episode_2026091922062101` [360, 1260). Supervised interval: [380, 1240).

**Boundary evidence indices**

[360, 379, 380, 1239, 1240, 1259]

**Condition evidence indices**

[380, 1239]

**Deployment condition source**

HLA chooses the foreground wedge instance and its explicit slot footprint.

**Label derivation**

RULE P for foreground wedge, endpoint1239 and exit1239; no W/M class supplied to policy.

**Merge check**

Adds a new initial arrangement for the same geometry-goal behavior as A wedge; not a new shape-generalization claim.

**Objective**

Reorient and place foreground wedge-shaped piece, with approach and side changes.

**Rationale**

380 wrist resolves rod/wedge geometry;860/1050 survey shows changing approach and orientation;1239 shows placement and tool traveling toward central barred piece.

**Split check**

Stop before central barred displacement; preserve remaining descent after tool startup.

**Supervision exclusions**

[]

**Training condition**

W/M-looking foreground wedge at380; achieved footprint1239 and tool exit1239.

**Uncertainty**

Third-view arm occlusion is significant; dual-view tracking/audit needed, not wholesale exclusion.

### b_bar_clear

Source: `episode_2026091922062101` [1220, 1520). Supervised interval: [1240, 1500).

**Boundary evidence indices**

[1220, 1239, 1240, 1499, 1500, 1519]

**Condition evidence indices**

[1240, 1499]

**Deployment condition source**

HLA can request temporary blocker clearance on a selected instance before placing another.

**Label derivation**

RULE P for central barred piece at1240, endpoint1499/exit1499.

**Merge check**

Same supporting clearance objective as A barred clearing, using the identical physical-goal interface.

**Objective**

Move central barred piece leftward to open the center corridor.

**Rationale**

1240 wrist shows tool near its edge;1450 survey and1499 show it left of its initial position with D still foreground-left.

**Split check**

Do not merge with D transfer; temporary clearing pose is not the piece's later final pose.

**Supervision exclusions**

[]

**Training condition**

Target central/right barred piece, not upper-left barred; achieved pose1499 and exit1499.

**Uncertainty**

Clearance intention is inferred; track identity must distinguish the two barred pieces.

### b_d_stage

Source: `episode_2026091922062101` [1480, 1970). Supervised interval: [1500, 1950).

**Boundary evidence indices**

[1480, 1499, 1500, 1949, 1950, 1969]

**Condition evidence indices**

[1500, 1720, 1949]

**Deployment condition source**

HLA selects D-like instance for a staging or layout pose chosen from current occupancy.

**Label derivation**

RULE P for foreground-left D-like, endpoint1949 and exit1949.

**Merge check**

Same inner-hole transport structure as A D stage despite different execution order.

**Objective**

Push D-like through the cleared region to upper-right.

**Rationale**

1720 wrist clearly shows rod in D opening;1900/1949 show upper-right D and departing tool. This happens before O/R in B.

**Split check**

End before O-like movement; do not impose A's manipulation order or final D yaw.

**Supervision exclusions**

[]

**Training condition**

D-shaped piece initially foreground-left; achieved staging pose1949 and tool exit1949.

**Uncertainty**

Large tool occlusion is partly resolved by wrist; endpoint is an achieved local pose, not verified intent.

### b_ring_move

Source: `episode_2026091922062101` [1930, 2870). Supervised interval: [1950, 2850).

**Boundary evidence indices**

[1930, 1949, 1950, 2849, 2850, 2869]

**Condition evidence indices**

[1950, 2400, 2849]

**Deployment condition source**

HLA selects ring instance and supplies its spatial goal independently of character order.

**Label derivation**

RULE P for ring above central barred at1950, endpoint2849/exit2849.

**Merge check**

Same ring/other-piece relocation objective; symmetry-aware contour goals make it trainable with A ring examples.

**Objective**

Turn/translate ring down into a slot above the foreground wedge and refine it.

**Rationale**

2150/2400 survey show progressive ring movement;2700 and2849 show achieved pose and tool exit. Keep fine adjustments.

**Split check**

Separate from long R manipulation; internal ring reapproaches stay in this sequence.

**Supervision exclusions**

[]

**Training condition**

Selected oval ring near D at1950; achieved footprint2849, no forced canonical angle if symmetric.

**Uncertainty**

Shape symmetry and top-surface parallax limit exact angular labels; no verified O class is supplied.

### b_r_move

Source: `episode_2026091922062101` [2830, 4720). Supervised interval: [2850, 4700).

**Boundary evidence indices**

[2830, 2849, 2850, 4680, 4699, 4700, 4719]

**Condition evidence indices**

[2850, 3000, 3900, 4000, 4120, 4160, 4250, 4300, 4699]

**Deployment condition source**

HLA selects the R-like instance or a novel irregular shape and supplies a local slot goal; internal reset decisions are autonomous.

**Label derivation**

RULE P for left R-like, achieved footprint4699 and exit4699. Recontact/posture modes inferred with uncertainty from tool/object motion, not force labels.

**Merge check**

Combines with both A R invocations via explicit current/goal geometry; no splitting into disconnected retries or false successes.

**Objective**

Relocate and align R-like above O-like with repeated contact acquisition and a lift/reconfiguration reset.

**Rationale**

3000 shows hole-neighborhood approach,3900 transport,4000-4250 reset,4300 new approach;4650-4669 still contain final R adjustments, so boundary was refined to4700 after4680/4699 inspection. Reuse reset4000-4250 for tool_position.

**Split check**

Keep reset and resumed correction under the same target/goal; stop only when tool departs toward D, not at the coarse motion-bin boundary or the earlier4650 candidate.

**Supervision exclusions**

[]

**Training condition**

R-like piece initially on left of central barred; fixed achieved goal4699 with exit4699 and protected neighbors.

**Uncertainty**

Contact and friction are unobserved; high joint reconfiguration is real but not diagnosed as controller failure. All retained subject to decoder/safety audit.

### b_d_align

Source: `episode_2026091922062101` [4680, 4970). Supervised interval: [4700, 4950).

**Boundary evidence indices**

[4680, 4699, 4700, 4949, 4950, 4969]

**Condition evidence indices**

[4700, 4949]

**Deployment condition source**

HLA requests orientation/position refinement on the upper D-like instance.

**Label derivation**

RULE P for D-like above R, endpoint4949 and exit4949.

**Merge check**

Same fine correction problem as A D alignment and gross D moves, expressed by smaller goal error.

**Objective**

Turn D-like to align with the developing column.

**Rationale**

4700 begins approach after final R adjustment;4750 survey shows rod near D;4949 shows changed D orientation and lifted departure toward L.

**Split check**

Short independent revisit, not continuation of earlier B D stage; L work receives a new instance condition.

**Supervision exclusions**

[]

**Training condition**

Target D-like upper-right; achieved footprint4949 and tool exit4949.

**Uncertainty**

No original orientation target supplied; endpoint rigid registration is an inferred training label.

### b_l_move

Source: `episode_2026091922062101` [4930, 6170). Supervised interval: [4950, 6150).

**Boundary evidence indices**

[4930, 4949, 4950, 6149, 6150, 6169]

**Condition evidence indices**

[4950, 5200, 5500, 5750, 6149]

**Deployment condition source**

HLA supplies a slot footprint for the tracked L-like/concave instance; policy handles side selection and retries.

**Label derivation**

RULE P for upper L-like, endpoint6149 and exit6149. Do not treat early descent at5050 as universally noncontact; it remains object-policy supervision.

**Merge check**

Same long contact-switching problem as A L move with fewer reorientations and a different initial location.

**Objective**

Reorient, transport and refine L-like between R-like and D-like.

**Rationale**

5200 shows L partly moved,5500 off-center turn,5750 and6000 correction,6149 final pose with tool departing. All intermediate contacts retained.

**Split check**

No cuts at pauses or reversals; end at change to upper-left barred instance.

**Supervision exclusions**

[]

**Training condition**

Target two-arm L-like initially above ring/R; fixed hindsight goal6149 and exit6149.

**Uncertainty**

L semantic identity and success remain unverified; thin/concave contact feasibility on novel shapes needs separate tests.

### b_upper_bar_move

Source: `episode_2026091922062101` [6130, 6930). Supervised interval: [6150, 6910).

**Boundary evidence indices**

[6130, 6149, 6150, 6909, 6910, 6929]

**Condition evidence indices**

[6150, 6500, 6909]

**Deployment condition source**

HLA selects upper-left barred instance and an intermediate/slot footprint with neighbor protection.

**Label derivation**

RULE P for upper-left barred piece, endpoint6909 and exit6909.

**Merge check**

Adds late-order barred transport to the same group as A upper-bar transport, with different surrounding occupancy.

**Objective**

Rotate and translate upper-left barred piece downward toward foreground-left.

**Rationale**

6150 starts free approach,6500 shows changed orientation and lifted tool,6750 survey shows downward movement,6909 shows foreground placement and departure.

**Split check**

Separate next central barred correction; later revisit to this foreground piece is another local-goal call.

**Supervision exclusions**

[]

**Training condition**

Target originally upper-left barred piece, not central barred; achieved staging footprint6909 and exit6909.

**Uncertainty**

Similar silhouettes require instance persistence across orientation changes; no letter label is assumed.

### b_bar_align

Source: `episode_2026091922062101` [6890, 7390). Supervised interval: [6910, 7370).

**Boundary evidence indices**

[6890, 6909, 6910, 7369, 7370, 7389]

**Condition evidence indices**

[6910, 7369]

**Deployment condition source**

HLA revisits the central barred piece with its intended side-layout footprint.

**Label derivation**

RULE P for central barred piece left of R/O at6910, endpoint7369/exit7369.

**Merge check**

Repeated invocation of B clearance instance with a new explicit goal, compatible with A barred alignment.

**Objective**

Move/rotate central barred piece upward beside the column.

**Rationale**

6950/7200 survey shows central barred contact and motion;7369 shows its achieved placement and rod approaching foreground barred piece.

**Split check**

Exit waypoint explains clearance toward next piece without authorizing its push; new target condition begins7370.

**Supervision exclusions**

[]

**Training condition**

Target same central barred instance as b_bar_clear; endpoint7369 and exit7369.

**Uncertainty**

No semantic success inferred from apparent lettering; pose fitting must not swap barred instance identities.

### b_upper_bar_align

Source: `episode_2026091922062101` [7350, 7730). Supervised interval: [7370, 7730).

**Boundary evidence indices**

[7350, 7369, 7370, 7729]

**Condition evidence indices**

[7370, 7729]

**Deployment condition source**

HLA selects foreground barred instance, requests its last local footprint and a safe stopped exit pose.

**Label derivation**

RULE P at7729; actual zero command at7729 supports hold but not an N+1 state or verified task completion.

**Merge check**

Same final refinement/withdrawal behavior as A foreground barred revisit, not an episode-specific terminal policy.

**Objective**

Refine foreground barred placement, withdraw and hold.

**Rationale**

7370 approach,7450/7650 survey shows movement,7670 wrist and7729 show terminal rod/piece relation and lift. Keep final hold; no suffix beyond source.

**Split check**

Retain terminal actions as part of safe completion of a local call; do not train a word-success classifier on episode end.

**Supervision exclusions**

[]

**Training condition**

Target foreground barred piece from b_upper_bar_move; achieved footprint7729 and stopped tool endpoint7729.

**Uncertainty**

Final scene/word success unannotated and partly occluded; HLA must independently verify requested layout and identities.


## Heuristic 1

Learn boundary-relative contact decisions and local response residuals inside a deterministic, receding-horizon piece-to-pose controller.

### Action decoding

Use COMMON DECODER D specified in tool_position. For active contact, output a local contact-normal/tangent velocity and bounded height/orientation residual. Rotate local linear velocity into W then base; only after w-frame verification convert angular residuals to the serialized convention. In approach/reset/exit modes allow full 3D translational residuals and bounded angular corrections; retain A3900/4319/5219/7400 supervision rather than silently setting w=0. A verified follower resolves robot redundancy, not a learned q-trajectory replay. Compare against original command dq when nonnull, never measured dq.

### Applicability

Flat, trackable, rigid pieces with an accessible boundary, including holes/concavities large enough for the verified rod and uncertainty margin. Works on geometry regardless of character identity; intended for novel shapes but no novel-shape success is demonstrated. Reject stacked/tipped pieces, unreachable targets, ambiguous protected contacts or contact candidates with excessive uncertainty.

### Augmentation

SE(2) transform current/goal contours, normals, obstacles, rod pose, exit waypoint, contact candidate and velocity labels together. Keep physical scale as a feature; do not claim arbitrary shape warps preserve pushing dynamics. Use contact-frame normalization without confusing it with physically moving the raw robot. Reject unreachable/collision-inconsistent synthetic placements. Add correlated mask/calibration noise with confidence masks; photometric changes only affect the perception branch. No character-ID substitution can stand in for shape augmentation. Friction variations are evaluation/robustness hypotheses, not invented supervised transitions.

### Evidence

- `episode_2026091922002701`: [1300, 1950, 2600, 3400, 3900, 4100, 4350, 4550, 4750, 5450, 5900, 6500, 6800, 7100, 7400, 7800, 8050, 8950]
- `episode_2026091922062101`: [380, 1240, 1720, 2400, 3000, 3900, 4000, 4120, 4160, 4250, 4300, 5200, 5500, 5750, 6500, 7729]
### Goal conditioning

RULE P: persistent selected instance, desired rigid footprint transform/goal contour in W, optional semantic orientation constraint, protected instances, tolerances and optional exit-tool pose/twist. Geometric planner may choose multiple contacts but cannot change the target piece or the desired world goal. Episode endpoints and inferred words are not inputs.

### Handoff

**entry_conditions**

Valid selected-instance track and current contour/height uncertainty; tool in reachable free space or at a plausible boundary; collision-free candidate contact or a feasible reset path. If tool starts high, internal approach or an explicit tool call is required.

**exit_conditions**

Current target geometry satisfies caller tolerance for several fresh observations, protected pieces checked, and requested exit waypoint attained. Return local-goal reached separately from word success. If object goal is reached but exit clearance is unsafe, report NEEDS_REPOSITION instead of pushing another object.

**failure_signatures**

Object fails to move under repeated contact, unexpected rotation, contact candidate changes identity, rod enters an inadequately sized hole, neighbor displacement, uncertain pose, rising tau_ext, joint-limit alarm, or persistent lack of goal-error reduction. B4120-4200-like high redundant joint motion triggers a guarded reset/replan, not imitation of unsafe joint magnitudes.

**overlap_role**

Each invocation contains approach, repeated strokes and clearance supervision; neighboring-object buffers only initialize/audit. Explicit tool overlaps permit an external reposition call without losing the object goal.

**successor_readiness**

Return achieved footprint and covariance, changed neighbors, tool pose/velocity and proposed safe next contact/exit status. HLA decides whether to refine this piece, clear another, or advance to the next slot.

### Heuristic id

boundary_mechanics_residual

### Input preprocessing

Use C; construct a multiresolution contour graph containing sampled outer and inner boundary points in meters, local normals, curvature, clearance, target correspondence and neighboring footprints. Do NOT substitute convex hull for actual contact geometry. Use object-relative coordinates with world/metric scale retained, rod radius/tilt, current tool offset and goal SE(2) error; a confidence-weighted set of orientations handles symmetry. Camera RGB crops (proposed256x256 from original mapped pixels) assist tracking, not alphabet classification. Up to12 causal frames supply measured tool/object motion and last commands. Contact/mode labels are inferred offline from rod-boundary proximity and subsequent object motion with uncertainty masks; they are training targets only. Online mode inference uses current/past motion and geometry, not future displacement. Use the same causal track estimator for policy inputs; smoothed tracks only supervise response/goal labels.

### Limitations

A quasi-static local response is only approximate for nonconvex pieces, sliding, corner pivots and uncertain friction. Existing data have few materials/shapes and no calibrated forces; tau_ext cannot identify friction or exact contact alone. Tool tilt, inner-hole contacts and occlusion require uncertainty-aware candidates. This design must not assume all unseen letters have the same mass distribution or font.

### Pipeline implications

Implement deterministic collision-feasible boundary candidate generation and short-horizon SE(2) goal planning, with a small graph model predicting response residual/uncertainty and a phase/contact score. Learn low-dimensional speed, tangent/normal ratio and bounded posture residual, rather than seven independent joint velocities. Use masked action cloning plus auxiliary one-step piece-displacement loss where an actual next frame is available and track confidence permits; audit contact-mode weak labels. Learn approach/contact/reset/exit/hold gating from geometry and command evidence. Replan after each short stroke; no open-loop whole-letter trajectory. Export component-specific coverage counts so holes, clearance moves and L retries are not filtered away.

### Policy contract

**caller_arguments**

policy_id=piece_contact_graph_v1; instance_id; goal_SE2_W or goal_footprint_W; semantic_orientation_constraint if required; protected_instance_ids; tolerances; optional exit_tool_base pose/arrival twist; clearance/speed limits; timeout/retry budget. Dataset responsibility: piece_relocation.

**input_output_contract**

Current causal dual-view/robot scene representation and caller goal -> bounded command via D, progress/error, uncertainty and phase/contact candidate. Policy selects approach side and strokes internally; HLA supplies the physical objective, not human-selected pushes.

**memory_and_handoff**

New instance/goal resets mode and response memory; warm-start with12 valid observations and actual last command. Preserve only validated tracker identity/geometry across calls. A tool reset pauses the object policy and resumes with new observations, recomputed candidates and the same goal.

**selection_cues**

Prefer when metric local contours and rod geometry are confident and contact-model uncertainty is low. For symmetric objects or unreliable canonical pose but good dense masks, consider piece_goal_field_v1. Neither alternative bypasses shared perception or controller safety gates.

**status_and_progress**

Report positional/angular or contour error with symmetry flag, active boundary ID, contact confidence, approach/contact/reset/exit phase, predicted versus observed progress, retry count, protected-piece motion and readiness. Never report a verified character/word solely from manipulation completion.

### Policy id

piece_contact_graph_v1

### Rationale

A4750 and B1720 show inside-hole D-like pushing; A4550 shows a barred-piece recess; A6500/6800/7100/7400/7800 and B5500/5750 show L-like contact switching rather than one straight stroke. The reusable structure is a rigid piece on a plane driven by a rod at a boundary, with tightly constrained height and only contact choice, in-plane velocity and small posture residuals varying. A generic contour-contact prior plus learned response residual concentrates limited data on interaction mechanics and is a plausible unseen-shape transfer mechanism. It does not establish transfer merely by accepting a new letter label.


## Heuristic 2

Learn a causal, geometry-only goal-field servo with mode-aware resets and deterministic collision projection, avoiding alphabet-specific poses.

### Action decoding

Use D. The goal-field servo proposes W-frame vx,vy plus a discrete approach/contact/reset/exit/hold phase, height-rate and bounded physical angular residual. A deterministic geometric safety projection prevents penetration of protected footprints and unsafe rod sweep, then verified frame/origin conversion and the joint-velocity follower generate the actual command. Retain original recorded v,w as behavior targets and action dq as an auxiliary audit only. Do not treat goal-mask pixel displacement as metric controller velocity without the calibrated warp.

### Applicability

Same flat-piece deployment domain as the contact-graph alternative, particularly when a dense target/obstacle mask is reliable but a unique canonical angle or local contact-mechanics fit is uncertain. Goal raster must represent the actual currently observed unseen shape, not a learned glyph template. Cannot operate through complete occlusion or unverified plane/rod geometry.

### Augmentation

Jointly rigid-transform current/goal SDFs, tool heatmap, obstacle channels, exit waypoint and action vectors in W; preserve metric extent and robot reachability. Photometric augmentation and partial-occlusion masks train the perception/confidence path without changing actions. Add small registration perturbations coherently to current tool/object/goal channels and supervision conversion. Do not independently rotate the goal while retaining an incompatible action, apply arbitrary nonrigid glyph morphs, or use mirrored spelling as a valid semantic label.

### Evidence

- `episode_2026091922002701`: [1300, 2600, 3219, 4100, 4550, 5450, 6500, 7100, 7800, 8050, 8509, 9014]
- `episode_2026091922062101`: [380, 1240, 2400, 2849, 3900, 4000, 4250, 4300, 5500, 5750, 6149, 6909, 7369, 7729]
### Goal conditioning

Same RULE P caller interface as the contact-graph policy. Render the current selected shape transformed to the requested location/orientation into a goal SDF, plus a directed semantic orientation marker when necessary. Keep desired goal fixed in W and recompute current-to-goal fields causally. A symmetric mask without a semantic orientation marker is insufficient to assert correct spelling for an ambiguous letter.

### Handoff

**entry_conditions**

Valid persistent target mask, current-goal correspondence, calibrated W warp and rod/obstacle channels. Geometric safety projection must find an admissible action; start high via approach mode or tool_waypoint_v1.

**exit_conditions**

Confidence-weighted contour-distance/overlap and position/orientation constraints meet caller tolerance over fresh observations; protected-instance motion checked and requested tool exit satisfied. Reached mask goal is a local physical result, not a recognition result.

**failure_signatures**

Mask swaps, unstable correspondence, error reduction stalls or reverses repeatedly, target disappears, predicted action crosses unsafe concavity/neighbor, tool moves but object does not, or controller/torque watchdog alarms.

**overlap_role**

All approach, correction and recontact actions remain supervised with mode/history; prefix/suffix observations supply initialization and label audit only. The two tool overlaps can be executed by a separate call while keeping the same goal field.

**successor_readiness**

Return current footprint/uncertainty, contour-error map, last phase/command and safe-to-handoff flag; HLA may retry, choose the mechanics alternative, reposition the tool or move a blocker.

### Heuristic id

causal_goal_field_servo

### Input preprocessing

Use C. Rasterize a proposed256x256 W-aligned metric local map with channels for selected-instance signed distance including holes, desired-instance signed distance, other-piece occupancy, table edge/reachability, tool footprint/height/tilt and validity. Include vector current-to-goal correspondence obtained by rigid registration or confidence-weighted contour matching, not alphabet templates or PCA alone. Keep metric scale and a coarse whole-scene map so crop normalization does not erase obstacle distance. Feed12 causal map/state/command frames to a small recurrent convolutional model; mask absent/stale views. Learn auxiliary selected-piece motion from within-cut actual successor frames, using offline audited motion as a target, never as an input. Deployment uses the same causal mask/tracker and HLA goal transform; no endpoint image required.

### Limitations

Dense image-space/plane-space servo is less explicitly grounded in friction than the mechanics prior and can propose ineffective contacts on unfamiliar concavities. Geometry safety projection does not prove task progress. Incorrect masks or plane thickness can corrupt both alternatives. New letters and combined novel-word/novel-shape tests remain required.

### Pipeline implications

Independent weights from the graph policy. Implement SDF/correspondence conversion, compact recurrent inverse servo, phase head and geometric action projection. Train masked behavior cloning plus goal-error/progress and one-step motion auxiliary targets; phase separates nonmonotonic reset actions from contact-progress actions, so do not impose monotonic error reduction on the entire demonstration. Audit phase-specific retention and compare on exactly the same held-out calls. Unlike the first prior, inference need not fit a friction/contact-response model or choose a unique canonical object frame.

### Policy contract

**caller_arguments**

policy_id=piece_goal_field_v1; same physical instance/goal/protection/exit/tolerance/budget arguments as piece_contact_graph_v1. Dataset responsibility: piece_relocation. A goal mask may be supplied directly only with its calibrated W mapping and instance identity.

**input_output_contract**

Causal robot observations and current/goal metric maps -> one bounded command via D and status/progress maps. Internal responsibilities include mode inference, corrective servo and contact-reset proposal; HLA retains target choice and overall sequencing.

**memory_and_handoff**

Reset recurrent servo on new goal/instance; use up to12 prior live frames with validity masks. After a tool-policy call rebuild maps and initialize history from actual commands, never carry a memorized segment index.

**selection_cues**

Prefer for a well-segmented shape with ambiguous orientation or lower confidence in the mechanics model, and for small contour alignment after gross placement. If masks/correspondence are unreliable, stop rather than choosing it as an ungrounded fallback.

**status_and_progress**

Return contour error, semantic-orientation validity, predicted local progress/uncertainty, phase, contact-proximity confidence, collision-projection magnitude, changed neighbors and termination/failure vocabulary from the global contract.

### Policy id

piece_goal_field_v1

### Rationale

O-like moves in A2600/3219 and B2400/2849 can be expressed without a unique glyph coordinate frame, while L-like changes pose repeatedly at A6500/7100/7800/8050 and B5500/5750. Current and desired geometry, not a letter class, define both problems. A dense metric goal field and causal mode memory let a small model learn local corrective velocities while deterministic registration absorbs translation/rotation variation. This is a substantive alternative to learned local mechanics, with different failure modes and an explicit independent evaluation obligation.


All scientific text above is unchanged Runtime API output. Rendering and slicing are developer-owned. No policy code or training exists in this stage.
