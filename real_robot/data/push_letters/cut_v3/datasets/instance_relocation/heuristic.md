# Instance-conditioned relocation, staging and alignment

Relocate one caller-selected physical piece to one explicit spatial placement, whether temporary staging or alignment, while retaining the demonstrated approach, contact changes, corrections and withdrawal. The HLA retains all multi-instance and word-level control.

## Dataset rationale

Cuts follow changes of selected physical instance and fixed hindsight placement, not equal duration or a predefined task phase order. All reference letters below are visual mnemonics for specific pieces, not verified semantic labels. A and B prefixes denote their original episodes only. First/last paired records of every cut and images of their first/last supervised rows were inspected; neighborhoods and wrist detail were used for uncertain handoffs. Exact material boundaries select a handoff REGION, not a claimed exact contact-change timestamp. Thirty rows on both sides of each inter-instance boundary are explicitly excluded from action loss, retaining context and goal evidence without inventing exposure-level synchronization. Within each interval the declared instance and goal are fixed; approach, perimeter travel, pressing, contact loss/re-contact, orientation corrections and withdrawal can be produced for that goal without another semantic label pass. Repeated visits to R/D/I/H have different local endpoints and are distinct invocations, even if a human might have had one larger word objective. L's prolonged reorientation and transport is kept together because the inspected evidence supports one continuing selected instance and one hindsight endpoint, not an intervening target switch. Differences are observations/goals in a shared learning problem rather than a reason for one policy per segment.

### a_w

Source: `episode_2026091922002701` [0, 1720). Supervised interval: [0, 1720).

**Boundary evidence indices**

[0, 1700, 1719, 1720, 1850]

**Condition evidence indices**

[0, 1700, 1719]

**Deployment condition source**

HLA selects a currently tracked instance and image placement through the common interface; it does not request the demonstration's W or order.

**Label derivation**

G with goal anchor 1719, original goal box [500,490,695,640]. Select the zigzag piece initially near third-preview (190,105), visually W/M-like. Supervise finite recorded command dq only, with the listed edge exclusion and common validity rules.

**Merge check**

Do not merge with the following I-stage block: the contacted/selected physical instance and its achieved placement change.

**Objective**

Approach, reorient and transport the initially upper-middle zigzag piece to the lower-right staging/word-row position.

**Rationale**

Initial inventory at 0, local tool/W evidence at 1700 and completed visible displacement at 1719 support this fixed hindsight goal before motion toward the central crossbar piece.

**Split check**

Initial descent, perimeter travel, repeated W contacts and withdrawal share the selected piece and endpoint. No inspected evidence requires a different internal instance or goal assignment.

**Supervision exclusions**

[{"reason": "Uncertain release versus next-instance approach ownership at the W-to-I handoff; retained goal/context only.", "start": 1690, "stop": 1720}]

**Training condition**

Selected episode-local zigzag/W-like instance; fixed achieved placement G(1719). No word/class token is a policy input.

**Uncertainty**

W versus M depends on orientation; use physical referent. Initial missing dq commands carry no action loss. Goal is hindsight, not verified original intent or word success.

### a_i_stage1

Source: `episode_2026091922002701` [1720, 1920). Supervised interval: [1720, 1920).

**Boundary evidence indices**

[1719, 1720, 1850, 1919, 1920, 2000]

**Condition evidence indices**

[1720, 1850, 1919]

**Deployment condition source**

HLA may choose a temporary free-space placement for a selected blocking instance, using the same spatial-goal call.

**Label derivation**

G at 1919, goal box [275,310,465,432]. Select the lower of the two crossbar-shaped pieces, adjacent to O, as visible at 1850; mnemonic I-like. This is its local attained pose, NOT its final episode pose.

**Merge check**

Both adjacent blocks use different selected instances (zigzag then annular O); a merge would hide a target switch and temporary objective.

**Objective**

Briefly displace/reorient the lower crossbar piece away from its initial position before approaching O.

**Rationale**

At 1720 the tool leaves the zigzag piece; by 1850 it is at the lower crossbar piece, whose pose differs from the initial scene; 1919-2000 shows departure toward O.

**Split check**

One short selected-instance displacement; the goal is explicitly the local 1919 pose, so no unresolved eventual-word objective is used.

**Supervision exclusions**

[{"reason": "W release versus I approach ambiguity.", "start": 1720, "stop": 1750}, {"reason": "I release versus O approach ambiguity.", "start": 1890, "stop": 1920}]

**Training condition**

Lower crossbar/I-like instance, local staging placement G(1919).

**Uncertainty**

The short adjustment's original purpose is unrecorded; intentional staging versus incidental correction is not certified. Label only its observed selected-piece/local outcome, not a successful clearance strategy.

### a_o

Source: `episode_2026091922002701` [1920, 3220). Supervised interval: [1920, 3220).

**Boundary evidence indices**

[1919, 1920, 2000, 3200, 3219, 3220, 3300]

**Condition evidence indices**

[1920, 2000, 3200, 3219]

**Deployment condition source**

HLA-selected instance and current spatial placement; object identity comes from automatic tracking.

**Label derivation**

G at 3219, goal box [505,365,680,490]. Select the annular O-like piece initially near third-preview (190,250).

**Merge check**

I-stage before and R-stage after have different target instances. Do not let those actions share O's goal.

**Objective**

Approach the annular piece from its left/lower side, push rightward and align it above the moved zigzag piece.

**Rationale**

Views 2000-2400 show approach and wrist-confirmed O-edge interaction; 3200-3219 shows the displaced annulus and tool departing.

**Split check**

Re-contact and orientation correction around the same annular piece retain one endpoint; no separate letter or staging goal appears in inspected views.

**Supervision exclusions**

[{"reason": "Previous I release/next O approach uncertain.", "start": 1920, "stop": 1950}, {"reason": "O release/R approach uncertain.", "start": 3190, "stop": 3220}]

**Training condition**

Annular/O-like instance; achieved placement G(3219).

**Uncertainty**

Annular symmetry makes exact yaw ambiguous. Preserve orientation ambiguity rather than fabricating a signed angle; do not infer contact force.

### a_r_stage

Source: `episode_2026091922002701` [3220, 3600). Supervised interval: [3220, 3600).

**Boundary evidence indices**

[3219, 3220, 3300, 3450, 3599, 3600]

**Condition evidence indices**

[3220, 3450, 3599]

**Deployment condition source**

Caller may request temporary placement of any selected piece; no fixed final-word target is implied.

**Label derivation**

G at 3599, goal box [405,135,545,245]. Select the R-like piece initially near third-preview (155,150), not the nearby lower crossbar piece.

**Merge check**

Target changes from O to R and then to I. Later a_r has a DIFFERENT local R goal and is not merged through intervening objects.

**Objective**

Move and rotate the R-like piece upward to a temporary location near the upper crossbar piece.

**Rationale**

3300 approach passes the crossbar, 3450 directly engages R, and 3599-3600 shows R at an upper location, unlike its eventual row placement.

**Split check**

Passage near I is approach context, not a second assigned target; the supervised interior is tied to R and its upper staging pose.

**Supervision exclusions**

[{"reason": "O release/R approach boundary ambiguity.", "start": 3220, "stop": 3250}, {"reason": "R withdrawal/next-I transit boundary ambiguity.", "start": 3570, "stop": 3600}]

**Training condition**

R-like instance, temporary placement G(3599), not the final R row slot.

**Uncertainty**

Original staging intention is inferred, not recorded. Small neighboring shifts may be incidental; no multi-object goal is labeled.

### a_i_stage2

Source: `episode_2026091922002701` [3600, 4230). Supervised interval: [3600, 4230).

**Boundary evidence indices**

[3599, 3600, 4150, 4229, 4230, 4400]

**Condition evidence indices**

[3600, 4150, 4229]

**Deployment condition source**

HLA chooses a new placement for the same scene instance when re-planning; changing its old placement starts a fresh invocation.

**Label derivation**

G at 4229, goal box [250,245,425,385]. Select the lower crossbar/I-like instance from a_i_stage1; at this endpoint it is left/up of its prior staging pose.

**Merge check**

Preceding R and following upper crossbar/H are different pieces. Earlier I-stage1 has a different local endpoint and an intervening O/R manipulation.

**Objective**

Re-approach and move the lower crossbar piece further left/up, opening space around the upper crossbar piece.

**Rationale**

3600 begins the return from R, 3900-4150 shows I interaction and changed orientation/location, and 4229-4230 shows withdrawal before H motion.

**Split check**

The curved approach and I correction share the fixed local endpoint; no additional target switch is supported inside the supervised interior.

**Supervision exclusions**

[{"reason": "R release/I approach uncertain.", "start": 3600, "stop": 3630}, {"reason": "I release/H approach uncertain.", "start": 4200, "stop": 4230}]

**Training condition**

Lower crossbar/I-like instance; second local staging placement G(4229).

**Uncertainty**

Opening space is an interpretation of the observed displacement, not a recorded task instruction. Do not merge it with the final I alignment goal.

### a_h_stage

Source: `episode_2026091922002701` [4230, 4620). Supervised interval: [4230, 4620).

**Boundary evidence indices**

[4229, 4230, 4400, 4619, 4620]

**Condition evidence indices**

[4230, 4400, 4619]

**Deployment condition source**

HLA can request a temporary relocation before assigning a later final word slot.

**Label derivation**

G at 4619, goal box [250,462,448,605]. Select the upper crossbar/H-like piece initially near third-preview (245,142).

**Merge check**

I before and D after change target; a_h later revisits H with a different endpoint, so this is not final alignment supervision.

**Objective**

Push the upper crossbar piece down to the lower-left area and reorient it.

**Rationale**

At 4230 the upper piece remains in its initial neighborhood; 4400 and 4500-4600 show transport and reorientation, with the local result visible at 4619.

**Split check**

Transport and rotation are continuous execution for the same temporary placement; no further conditioning switch is left for annotation.

**Supervision exclusions**

[{"reason": "I-to-H handoff ambiguity.", "start": 4230, "stop": 4260}, {"reason": "H-to-D handoff ambiguity.", "start": 4590, "stop": 4620}]

**Training condition**

Upper crossbar/H-like instance; temporary lower-left placement G(4619).

**Uncertainty**

H/I naming is only a shape mnemonic; use the physical track. Endpoint is not its final requested position.

### a_d_stage

Source: `episode_2026091922002701` [4620, 4950). Supervised interval: [4620, 4950).

**Boundary evidence indices**

[4619, 4620, 4800, 4949, 4950]

**Condition evidence indices**

[4620, 4800, 4949]

**Deployment condition source**

Same common selected-instance/placement interface; internal contact choice may engage a visible hole but is not caller-mandated.

**Label derivation**

G at 4949, goal box [528,150,645,260]. Select the D-like single-hole piece initially near third-preview (105,215); wrist 4800 shows inner-hole engagement.

**Merge check**

H before and R after are different pieces. Later a_d_align changes D's attained position/orientation after the intervening R manipulation.

**Objective**

Engage the D-like piece, including its inner opening, and transport it upward/right to a temporary upper location.

**Rationale**

4620 departure toward D, wrist/third 4800 interaction and 4949 withdrawn tool beside displaced D support this local placement.

**Split check**

Single D transport/withdrawal with one endpoint; inner versus outer contact is a policy-internal choice rather than an unresolved task switch.

**Supervision exclusions**

[{"reason": "H release/D approach uncertainty.", "start": 4620, "stop": 4650}, {"reason": "D release/R approach uncertainty; 5000 already approaches R, so cut is earlier.", "start": 4920, "stop": 4950}]

**Training condition**

D-like instance; staging placement G(4949).

**Uncertainty**

Inner-hole contact is visually supported, not instrumented contact ground truth; unfamiliar holes may jam and are not certified safe.

### a_r

Source: `episode_2026091922002701` [4950, 5770). Supervised interval: [4950, 5770).

**Boundary evidence indices**

[4949, 4950, 5600, 5769, 5770]

**Condition evidence indices**

[4950, 5600, 5769]

**Deployment condition source**

HLA selects the previously staged instance and supplies its new desired image placement.

**Label derivation**

G at 5769, goal box [490,263,642,370]. Select the R-like instance in the upper temporary location at 4950; goal is its new placement above O.

**Merge check**

Do not merge with either D block or with earlier a_r_stage: instance or spatial objective changes.

**Objective**

Re-contact staged R, rotate and transport it into the column/row above O, retaining contact-side corrections.

**Rationale**

4950-5350 approach and displacement from R's staged location, wrist/third 5600 contact and 5769 local result support the new R condition.

**Split check**

Rotation, approach around the concavity and final correction all concern the same R and local endpoint.

**Supervision exclusions**

[{"reason": "D-to-R approach boundary ambiguity.", "start": 4950, "stop": 4980}, {"reason": "R release/D alignment approach boundary ambiguity.", "start": 5740, "stop": 5770}]

**Training condition**

R-like instance; later achieved placement G(5769), distinct from G(3599).

**Uncertainty**

Nearby D may shift during R work; other-object motion is monitored, not silently labeled as successful intentional coordination.

### a_d_align

Source: `episode_2026091922002701` [5770, 6140). Supervised interval: [5770, 6140).

**Boundary evidence indices**

[5769, 5770, 6100, 6139, 6140]

**Condition evidence indices**

[5770, 6100, 6139]

**Deployment condition source**

A new goal invocation for a previously moved instance; goal is spatial, not an instruction to replay a correction.

**Label derivation**

G at 6139, goal box [500,135,635,217]. Select the upper D-like piece and its new more aligned orientation/position.

**Merge check**

R and L neighbors are different selected objects, and a_d_stage had a different local D endpoint.

**Objective**

Correct staged D's orientation and position at the upper end of the developing arrangement.

**Rationale**

5770 begins near D after R; 6100 and 6139 show changed D orientation and tool withdrawal/transit.

**Split check**

One local pose correction plus withdrawal; no internal objective change is required.

**Supervision exclusions**

[{"reason": "R-to-D handoff uncertainty.", "start": 5770, "stop": 5800}, {"reason": "D-to-L withdrawal/transit uncertainty.", "start": 6110, "stop": 6140}]

**Training condition**

D-like instance; alignment placement G(6139).

**Uncertainty**

Tolerance and success are not annotated. Base z and tool proximity do not establish contact.

### a_l

Source: `episode_2026091922002701` [6140, 8050). Supervised interval: [6140, 8050).

**Boundary evidence indices**

[6139, 6140, 8000, 8049, 8050]

**Condition evidence indices**

[6140, 8000, 8049]

**Deployment condition source**

HLA supplies selected-instance and placement, leaving repeated re-contact choices to the policy.

**Label derivation**

G at 8049, goal box [490,190,635,284]. Select the L-shaped piece initially at upper-left third-preview (142,98).

**Merge check**

Preceding D and following I change instance. Do not combine them merely because they contribute to the same apparent word.

**Objective**

Repeatedly reorient/re-contact the L-shaped piece, then move it between the R and D placements.

**Rationale**

Overview 6400,7200 and 8000 shows prolonged L work with corrections and eventual rightward transport; 8049 pins its achieved placement.

**Split check**

The intermediate L rotations and delayed translation retain one selected instance and fixed hindsight endpoint; no intervening other-instance manipulation is supported by inspected evidence. Do not cut solely because it is long.

**Supervision exclusions**

[{"reason": "D release/L approach uncertainty.", "start": 6140, "stop": 6170}, {"reason": "L release/I approach uncertainty.", "start": 8020, "stop": 8050}]

**Training condition**

L-shaped instance; achieved placement G(8049).

**Uncertainty**

Repeated contact attempts are not separate verified failures or intentions. Preserve them as closed-loop corrections; no new-shape benefit is proven.

### a_i_align

Source: `episode_2026091922002701` [8050, 8500). Supervised interval: [8050, 8500).

**Boundary evidence indices**

[8049, 8050, 8400, 8499, 8500]

**Condition evidence indices**

[8050, 8400, 8499]

**Deployment condition source**

Caller selects the earlier staged lower crossbar instance and a new spatial goal.

**Label derivation**

G at 8499, goal box [315,238,477,343]. Select the lower crossbar/I-like track, left of R at 8050.

**Merge check**

L before and H after are different pieces; earlier I staging goals differ and have intervening manipulations.

**Objective**

Move and align the staged I-like piece to the left of the R/L area.

**Rationale**

8150 approach, 8400 aligned I and 8499 withdrawal before H approach distinguish this revisit from earlier I staging.

**Split check**

One fixed I placement; contact-side changes do not require another high-level objective.

**Supervision exclusions**

[{"reason": "L release/I approach uncertainty.", "start": 8050, "stop": 8080}, {"reason": "I release/H approach uncertainty, checked at 8470-8550.", "start": 8470, "stop": 8500}]

**Training condition**

Lower crossbar/I-like instance; later placement G(8499).

**Uncertainty**

Final glyph identity/readability is not a verified label. Apparent alignment is a hindsight outcome.

### a_h_align

Source: `episode_2026091922002701` [8500, 9015). Supervised interval: [8500, 9015).

**Boundary evidence indices**

[8499, 8500, 9014]

**Condition evidence indices**

[8500, 9014]

**Deployment condition source**

HLA requests selected-piece correction and checks its own word/layout afterward.

**Label derivation**

G at 9014, goal box [290,333,470,452]. Select the upper-crossbar/H-like track now in the lower-left staging location.

**Merge check**

Do not merge with preceding I; no successor record or next objective is invented at the episode end.

**Objective**

Re-approach the staged H-like piece from below, push it upward and align near the I-like piece, then stop/withdraw.

**Rationale**

8500-8700 shows approach to the lower-left piece; final third/wrist 9014 shows the moved piece and stationary command, not an annotated task success.

**Split check**

One instance and fixed endpoint through the final correction. No automatic final-word success or terminal action is added.

**Supervision exclusions**

[{"reason": "Previous I withdrawal versus H approach ambiguity.", "start": 8500, "stop": 8530}]

**Training condition**

Upper crossbar/H-like instance; final recorded local placement G(9014), without success label.

**Uncertainty**

Recording ends without a verified success or long stability observation. Last command is retained; no row 9015 is fabricated.

### b_w

Source: `episode_2026091922062101` [0, 1220). Supervised interval: [0, 1220).

**Boundary evidence indices**

[0, 650, 1219, 1220, 1300]

**Condition evidence indices**

[0, 650, 1219]

**Deployment condition source**

Same autonomous scene-instance and explicit spatial-goal call; no recorded alphabet or order is required.

**Label derivation**

G at 1219, goal box [598,492,805,651]. Select the zigzag/W-like piece initially at the bottom of the third view.

**Merge check**

Next block selects the nearby crossbar/I-like instance; merging would hide that switch.

**Objective**

Descend, re-contact, rotate and shift the zigzag piece to the lower-right arrangement position.

**Rationale**

0 inventory, wrist/third 650 W interaction and 1219 W displacement/tool departure establish the block.

**Split check**

Multiple W contacts and initial approach share one achieved placement; no internal task-word label is needed.

**Supervision exclusions**

[{"reason": "W release/I approach handoff uncertainty.", "start": 1190, "stop": 1220}]

**Training condition**

B episode zigzag/W-like instance; achieved placement G(1219).

**Uncertainty**

Null initial command dq is missing supervision, not zero. W/M semantic reading is not an input.

### b_i_stage

Source: `episode_2026091922062101` [1220, 1440). Supervised interval: [1220, 1440).

**Boundary evidence indices**

[1219, 1220, 1300, 1439, 1440]

**Condition evidence indices**

[1220, 1300, 1439]

**Deployment condition source**

HLA chooses an intermediate spatial placement for a currently tracked instance.

**Label derivation**

G at 1439, goal box [408,312,571,430]. Select the central/right crossbar/I-like piece, not the upper-left H-like piece.

**Merge check**

W and D neighbors are different instances. Later b_i_align has a new I endpoint after intervening moves.

**Objective**

Move the central crossbar piece leftward and reorient it to a local staging location.

**Rationale**

Wrist 1300 shows the tool at the crossbar piece; 1439-1440 shows it shifted left and the tool departing toward D.

**Split check**

One short target and local outcome; no unresolved eventual final-slot objective is imposed.

**Supervision exclusions**

[{"reason": "W-to-I handoff uncertainty.", "start": 1220, "stop": 1250}, {"reason": "I-to-D handoff uncertainty.", "start": 1410, "stop": 1440}]

**Training condition**

Central crossbar/I-like instance; local placement G(1439).

**Uncertainty**

Staging intention is inferred from the observed move, not recorded. Do not call it verified obstacle clearance.

### b_d_stage

Source: `episode_2026091922062101` [1440, 1900). Supervised interval: [1440, 1900).

**Boundary evidence indices**

[1439, 1440, 1800, 1899, 1900]

**Condition evidence indices**

[1440, 1800, 1899]

**Deployment condition source**

Common instance placement call; internal tool-contact selection can use a visible inner boundary.

**Label derivation**

G at 1899, goal box [590,148,713,252]. Select the D-like single-hole piece initially at lower-left third-preview (158,274).

**Merge check**

I before and O after change instance; later b_d_align has a distinct D pose goal.

**Objective**

Engage D's opening and transport the piece from lower-left to the upper-right temporary location.

**Rationale**

1800 wrist/third supports D-hole engagement and transport; 1899 shows upper D and withdrawn tool.

**Split check**

One instance transport plus approach/withdrawal; inner-hole contact is not a second task.

**Supervision exclusions**

[{"reason": "I release/D approach uncertainty.", "start": 1440, "stop": 1470}, {"reason": "D release/O approach uncertainty.", "start": 1870, "stop": 1900}]

**Training condition**

D-like instance; staging placement G(1899).

**Uncertainty**

Possible neighboring contact near O is not a multi-object supervision assignment. No force/contact labels are claimed.

### b_o

Source: `episode_2026091922062101` [1900, 2820). Supervised interval: [1900, 2820).

**Boundary evidence indices**

[1899, 1900, 2600, 2819, 2820]

**Condition evidence indices**

[1900, 2600, 2819]

**Deployment condition source**

HLA-selected annular or other instance and explicit current placement goal.

**Label derivation**

G at 2819, goal box [592,363,762,490]. Select the O-like annular piece near third-preview (289,126) at entry.

**Merge check**

Adjacent D and R calls target other pieces, despite common word-like arrangement.

**Objective**

Rotate and move the annular piece downward to a placement above the zigzag piece.

**Rationale**

1900 initial O location, 2600 interaction and 2819 achieved O placement support this objective.

**Split check**

Reorientation and downward transport share one endpoint and instance; no additional semantic cuts are required.

**Supervision exclusions**

[{"reason": "D-to-O handoff uncertainty.", "start": 1900, "stop": 1930}, {"reason": "O-to-R handoff uncertainty.", "start": 2790, "stop": 2820}]

**Training condition**

Annular/O-like instance; achieved placement G(2819).

**Uncertainty**

Symmetry may make signed orientation unobservable; retain the ambiguity set and use full foreground agreement.

### b_r

Source: `episode_2026091922062101` [2820, 4560). Supervised interval: [2820, 4560).

**Boundary evidence indices**

[2819, 2820, 3900, 4050, 4200, 4559, 4560]

**Condition evidence indices**

[2820, 3900, 4050, 4200, 4559]

**Deployment condition source**

Selected current instance and spatial placement; causal progress informs internal re-contact, not a hindsight recovery command.

**Label derivation**

G at 4559, goal box [580,258,740,370]. Select the R-like piece initially at left-middle third-preview (142,187).

**Merge check**

O before and D after are different targets. The unusual motion around 4050-4200 does not change the selected R or its goal.

**Objective**

Transport and rotate R toward the row above O, including tool withdrawal, posture adjustment and re-contact corrections.

**Rationale**

3900 and 4050 show R at changing orientations; 4200 wrist/third still shows R as the local target despite elevated tool/posture motion; 4559 pins aligned local outcome.

**Split check**

Retain the correction rather than silently discarding it or inventing an unobserved failure task. Current/past observations plus fixed R goal suffice for all labels; no internal target switch is unresolved.

**Supervision exclusions**

[{"reason": "O release/R approach uncertainty.", "start": 2820, "stop": 2850}, {"reason": "R release/D approach uncertainty.", "start": 4530, "stop": 4560}]

**Training condition**

R-like instance; achieved placement G(4559), constant through its correction/re-contact.

**Uncertainty**

Relatively large measured/commanded joint motion around 4200 is observed; singularity, slip or operator intention is not diagnosed. Backend safety validation must cover it.

### b_d_align

Source: `episode_2026091922062101` [4560, 4840). Supervised interval: [4560, 4840).

**Boundary evidence indices**

[4559, 4560, 4839, 4840]

**Condition evidence indices**

[4560, 4839]

**Deployment condition source**

A new goal for the previously staged D instance, expressed through the same caller contract.

**Label derivation**

G at 4839, goal box [587,123,716,206]. Select the upper-right D-like piece, now aligned horizontally in the third view.

**Merge check**

R and L neighbors select different pieces; earlier D-stage has a different local placement.

**Objective**

Correct D's orientation and small position at the upper end of the arrangement.

**Rationale**

4560 tool approaches D from R, 4800-4839 shows the corrected D; by 4889 tool is at L, motivating the earlier handoff cut.

**Split check**

One local D alignment/withdrawal objective; no later L actions are assigned its condition.

**Supervision exclusions**

[{"reason": "R-to-D handoff uncertainty.", "start": 4560, "stop": 4590}, {"reason": "D release/L approach uncertainty.", "start": 4810, "stop": 4840}]

**Training condition**

D-like instance; alignment placement G(4839).

**Uncertainty**

Tool partly occludes the goal contour; G must remove it and flag unresolved contour/orientation uncertainty instead of hallucinating a mask.

### b_l

Source: `episode_2026091922062101` [4840, 6080). Supervised interval: [4840, 6080).

**Boundary evidence indices**

[4839, 4840, 5850, 6079, 6080]

**Condition evidence indices**

[4840, 5850, 6079]

**Deployment condition source**

HLA requests a selected piece's explicit placement, with re-contact internally chosen.

**Label derivation**

G at 6079, goal box [580,186,726,278]. Select L-shaped piece near third-preview (262,72) at entry.

**Merge check**

D before and H after change instance; do not infer one task-wide policy condition.

**Objective**

Re-contact and transport L into the gap between R and D, with final position/orientation corrections.

**Rationale**

4840 follows D release, 5200-5850 shows L motion and wrist contact, and 6079 shows local achieved placement and withdrawal.

**Split check**

Same L/goal through repeated approaches; changing contact side is not a semantic goal change.

**Supervision exclusions**

[{"reason": "D release/L approach uncertainty.", "start": 4840, "stop": 4870}, {"reason": "L release/H approach uncertainty.", "start": 6050, "stop": 6080}]

**Training condition**

L-shaped instance; achieved placement G(6079).

**Uncertainty**

Image-plane goal accuracy is limited by calibration, perspective and segmentation. No metric placement precision is claimed.

### b_h_stage

Source: `episode_2026091922062101` [6080, 6900). Supervised interval: [6080, 6900).

**Boundary evidence indices**

[6079, 6080, 6899, 6900]

**Condition evidence indices**

[6080, 6899]

**Deployment condition source**

HLA may choose a staging location for a blocking instance before final placement.

**Label derivation**

G at 6899, goal box [201,387,415,529]. Select the upper-left crossbar/H-like instance initially near third-preview (158,104).

**Merge check**

L and I neighbors differ. Later b_h_align returns to H with a new endpoint; merging through I would conceal a conditioning switch.

**Objective**

Rotate and move the upper-left crossbar piece downward to a lower-left staging location.

**Rationale**

6080 begins after L, 6500-6800 shows H reorientation/downward travel, 6899 shows released H before approach to I at 6949.

**Split check**

One H staging goal; intermediate rotations are continuous execution, not unspecified subtask labels.

**Supervision exclusions**

[{"reason": "L-to-H handoff uncertainty.", "start": 6080, "stop": 6110}, {"reason": "H release/I approach uncertainty; 6949 already engages I.", "start": 6870, "stop": 6900}]

**Training condition**

Upper-left crossbar/H-like instance; temporary placement G(6899).

**Uncertainty**

Purpose of staging is not recorded. Preserve physical instance identity despite rotation-dependent H/I appearance.

### b_i_align

Source: `episode_2026091922062101` [6900, 7300). Supervised interval: [6900, 7300).

**Boundary evidence indices**

[6899, 6900, 7050, 7299, 7300]

**Condition evidence indices**

[6900, 7050, 7299]

**Deployment condition source**

Caller selects the previously staged central crossbar instance and a new layout goal.

**Label derivation**

G at 7299, goal box [368,232,529,333]. Select the central crossbar/I-like track from b_i_stage.

**Merge check**

Both adjacent H blocks target a different physical piece and different H placements; do not merge.

**Objective**

Re-contact I, translate/reorient it upward and align to the left of the R/L area.

**Rationale**

6900 tool transits from H, 7050 shows I interaction and 7299 shows its new local placement before H revisit.

**Split check**

One I goal through small corrections; earlier I stage endpoint is explicitly different.

**Supervision exclusions**

[{"reason": "H release/I approach uncertainty.", "start": 6900, "stop": 6930}, {"reason": "I release/H approach uncertainty.", "start": 7270, "stop": 7300}]

**Training condition**

Central crossbar/I-like instance; later placement G(7299).

**Uncertainty**

Goal is a visually inferred placement, not a certified glyph reading or task success.

### b_h_align

Source: `episode_2026091922062101` [7300, 7730). Supervised interval: [7300, 7730).

**Boundary evidence indices**

[7299, 7300, 7350, 7729]

**Condition evidence indices**

[7300, 7350, 7729]

**Deployment condition source**

HLA supplies H-instance correction goal and subsequently verifies the entire requested word/layout.

**Label derivation**

G at 7729, goal box [352,370,529,493]. Select the lower-left crossbar/H-like track previously staged by b_h_stage.

**Merge check**

Previous I is a different selected instance. Do not invent an unrecorded successor or task-completion phase.

**Objective**

Re-approach staged H and adjust it right/up near I, then stop/withdraw.

**Rationale**

7300 approach, wrist/third 7350 local target and 7729 final placement/stationary command support the last local goal.

**Split check**

Fixed H goal throughout final correction; episode end supplies no additional success semantics.

**Supervision exclusions**

[{"reason": "I withdrawal/H approach uncertainty.", "start": 7300, "stop": 7330}]

**Training condition**

Upper-left-origin crossbar/H-like instance; last recorded local placement G(7729), without success label.

**Uncertainty**

No verified final-word success or sustained stability annotation. Preserve original last record; no synthetic terminal row.


## Heuristic 1

Learn one reusable instance-to-placement policy with local boundary/topology sharing and closed-loop geometry feedback; never choose a spelling sequence or retrieve a known letter motion.

### Action decoding

Predict the original finite 7-component action_json.dq conditional distribution; execute only its selected current command through the common verified joint_velocity backend. q and measured dq are inputs, not labels. No Jacobian inversion, depth reconstruction or assumed v-to-dq conversion is required. Use verified joint velocity/acceleration/workspace limits and watchdog; if a proposed command violates them, return blocked rather than silently claiming it realizes the goal. Rad/s, joint order and 20-Hz hold semantics remain verification gates; until satisfied outputs are non-executable candidates. Do not command a gripper for this installed pushing tool.

### Applicability

Select for a single flat, separately tracked piece with a usable current contour, visible holes/concavities, trusted tool projection and a valid image goal in the recorded workspace. Especially motivates shape-conditioned transport, reorientation, staging and re-contact. Candidate contour quality, occlusion fraction and registration residual are online selection cues, not hidden contact state. Image/chart or instance ambiguity makes the policy unavailable; no unknown letter class fallback is necessary because it consumes geometry rather than an alphabet ID.

### Augmentation

Use illumination/background/texture variation consistently on current and goal RGB, mask-boundary noise bounded by calibration/segmentation uncertainty, and causal frame dropout with validity masks. Permute boundary-node ordering while preserving adjacency and target/obstacle identity. Translation/rotation of a local coordinate representation is a change of chart: transform ALL contours, tool projections, goal and corresponding coordinate features together, keeping absolute state/readout calibration so the unchanged joint command has the same meaning. Do not rotate a scene crop alone or assign a different desired pose to the same action. Do not warp a familiar shape into an unseen shape and call the old contact motion a valid new-shape demonstration. Hindsight goals are exactly the specified attained anchors, not arbitrary word slots.

### Evidence

- `episode_2026091922002701`: [0, 1700, 1850, 3200, 3450, 3600, 4150, 4400, 4800, 5600, 6100, 8000, 8400, 9014]
- `episode_2026091922062101`: [650, 1300, 1800, 2600, 3900, 4050, 4200, 5850, 7050, 7350, 7729]
### Goal conditioning

Use common rule G for training. At deployment use the caller's scene-scoped instance and explicit image-chart spatial goal rendered from that instance's currently observed shape, with fixed reference template and symmetry/ambiguity masks. Geometry and goal discrepancies, not W/O/R class names, condition the network. Do not replace the desired shape/rotation with an endpoint-derived online target. A caller change of goal or instance cancels the existing call; automatic contour updates for the same instance are continuous inputs.

### Handoff

**entry_conditions**

Verified backend/tool calibration, fresh paired sensors, uniquely selected instance, valid goal and workspace, reliable contour and no unresolved stack/overlap. Initial tool can be elevated or in a safe previously observed low-clearance neighborhood; dangerous unsupported travel is rejected.

**exit_conditions**

Return goal_observed only after current target-goal foreground/pose agreement remains within caller tolerances over a proposed 0.5-second causal window and the tool is visibly clear or motion can be stopped safely. Report ambiguous orientation separately. Timeout, loss of track, hazardous approach or caller cancel also terminates; none means success.

**failure_signatures**

Large contour/goal registration residual, fragmented or swapped mask, tool projection disagreement, commanded motion with no selected-piece response, unintended neighbor motion, repeated error oscillation, increasing uncertainty, joint limit/safety rejection, or force-residual anomaly. No tau_ext threshold is a ground-truth contact label.

**overlap_role**

No cross-segment motor memory. Excluded handoff bands preserve observation context without labeling next-target motion as the old goal; reset graph temporal state at a new invocation. Same-group policy reuse adds no evidence.

**successor_readiness**

Return current instance/goal residuals, track validity, tool clearance and a scene version. The HLA must re-check all placements and choose the successor target; this policy never automatically follows the demonstrated order.

### Heuristic id

boundary_graph

### Input preprocessing

Implement the common loader, validity masks, undistortion, automatic instance segmentation/tracking and tool projection in model_inventory. In the fixed third image chart sample variable-size ordered outer/inner boundary graphs at approximately uniform arc length; preserve holes, attach curvature, local tangent/normal with orientation/visibility confidence, target/other-instance flags, tool-relative vectors, and goal-contour distance features. Include workspace boundary and separate tool/global nodes, full q/measured dq/tau_ext/EE transforms, camera identity/calibration confidence and elapsed-time features. Normalize image coordinates by width/height without treating pixels as meters. Use the last up-to-eight valid causal observations and explicit dt/missing masks; no history crosses a segment. Do not infer a complete occluded contour by looking forward; predicted contours have a distinct validity/confidence channel. Current RGB segmentation and track initialization must work without a human point prompt online.

### Limitations

Two episodes, one tool and similar material/scale. Visible contour is not mass distribution, friction or center of pressure. Inner-hole engagement can jam an unfamiliar concavity. Shape transfer, collision avoidance and tolerances are unvalidated. A geometric bottleneck loses texture and can confuse touching pieces or symmetries. Current image geometry is not verified metric geometry; wide layout/perspective changes need the declared plane asset. Cannot recover a fallen/flipped/stacked piece or invent a feasible inventory. Missing perception/controller assets prevent autonomous deployment.

### Pipeline implications

Train a boundary-adjacency message-passing encoder with shared local edge/contact-affinity functions and an object/tool/global readout that emits joint commands conditioned on q and goal. Add a temporal state encoder for recent selected-piece response and re-contact. Soft attention to reachable visible contour neighborhoods biases the representation without hard-coding a particular side or object order. A separate global pathway must cover elevated approach and withdrawal, so all retained labels are definable without a later phase-label pass. Auxiliary prediction of the selected object's next observed displacement can be trained only for valid same-segment successors; it is a training target, never an online feature. Contact-affinity supervision, if used, is explicitly a weak proxy from tool-to-contour proximity plus subsequent piece motion, not contact truth. Implement and ship all converters, uncertainty masks, graph builder, decoder and monitor with this policy in the later stage.

### Policy contract

**caller_arguments**

Common instance_id, observation template reference, explicit image placement/foreground goal, workspace/obstacle references, tolerance and budget. Caller controls WHAT instance/placement, when to stop, and when to switch; it does not choose latent boundary attention or individual contact points. No untrained contact-side argument is offered.

**input_output_contract**

Causal two-view observations and state enter the automatic converter; the graph policy consumes the derived geometry/history plus robot state and fixed goal. Outputs one candidate command dq[7] and shared status/progress fields, including contour completeness, registration ambiguity and contact-affinity uncertainty. Gripper feedback remains invalid. All output units/actuation are subject to the common controller gate.

**memory_and_handoff**

Maintain only within-call temporal graph state and causal object response. Reset at a new goal/instance or alternative-policy handoff; use available current perception tracks, not another policy's hidden state. Training histories are padded/masked at the original cut boundary.

**selection_cues**

Prefer testing this alternative when boundary continuity, selected-instance association and projected tool/edge location are reliable, especially for unfamiliar visible shapes. Compare with visual_push when edges are fragmented but RGB/history is informative. This is a proposed selection rule, not an empirical ranking; if identity/goal is ambiguous neither policy is appropriate.

**status_and_progress**

Return centroid and foreground/rotation residual with ambiguity set, recent trend, geometry confidence, number of re-contact/oscillation events, possible neighbor displacement and blocked/uncertain/goal_observed flags. All use current/past estimates. Numerical safety/success thresholds require validation and cannot be inferred from episode endings.

### Policy id

contour_push

### Rationale

Observed regularity: the same rod engages straight edges, curved edges, corners and the D hole, and repeatedly changes contact while relocating differently shaped pieces (A4800, A8000; B650, B1800, B5850). Learning problem: infer goal-directed joint commands for a selected current shape, including approach/re-contact, without letter-class-specific trajectories. Inductive bias: local boundary adjacency and object/tool/goal-relative geometry share contact features across shapes while a global robot-state readout preserves executable joint semantics. Implementable mechanism: variable-size boundary graph message passing, temporal response state and a supervised 7-command head, with optional within-segment displacement prediction. Falsifiable expected benefit: improved held-out-shape displacement/orientation control versus the RGB alternative or graph-without-boundary ablation when contours are reliable. This expectation has not been tested.


## Heuristic 2

Learn an independent two-view temporal visual relocation policy that retains near-contact appearance and causal response history lost by a contour-only representation.

### Action decoding

Use its own learned 7-component command dq head and the SAME verified direct joint_velocity decoder contract, safety gate and stop/watchdog as contour_push. No blend or ensemble of the two motor outputs. Do not integrate command dq as a pose target; do not substitute state dq or v/w when dq is missing. All command order/units/timing must be verified before execution. No gripper actuation is part of this policy.

### Applicability

Alternative for the same one-instance relocation responsibility when the target is still uniquely identifiable but contours are partially occluded/fragmented, lighting is variable, or the wrist image contains useful near-contact appearance cues. Both camera freshness and causal tracking confidence must be adequate; fully lost identity or an invalid goal is not an excuse to continue blind. Applies to the same observed approach/contact/correction regimes, not an independently demonstrated broader recovery capability.

### Augmentation

Apply consistent photometric changes to target/current/goal appearance; independently model realistic camera exposure differences with calibration-aware color normalization. Use causal view dropout, occlusion masks, frame-repeat/drop simulation and timestamp masks to train declared missing-view behavior; never supply future frames to fill a gap. Dropout must not create labels for sustained blind motion. Preserve selected-instance/goal alignment in crop jitter and propagate coordinates to all masks and goal channels. Do not change word/instance/desired location alone while retaining the action. No arbitrary image rotation with unchanged robot actions or synthetic shape warping is accepted as a physical demonstration.

### Evidence

- `episode_2026091922002701`: [0, 1700, 1850, 3600, 4400, 4800, 5600, 8000, 8500, 9014]
- `episode_2026091922062101`: [0, 650, 1300, 1800, 3900, 4050, 4200, 5850, 7350, 7729]
### Goal conditioning

Use the same fixed instance and G hindsight spatial goal as the graph alternative. Feed goal foreground and masked RGB target crop in a separate goal stream with absolute image goal coordinates, along with the current target-instance marker. Neither raw episode endpoint frame nor visible word is substituted for a target mask. Online goal images are renderer/user-goal outputs supplied at invocation, not future observations. Changing the requested placement or instance is a new call.

### Handoff

**entry_conditions**

Verified backend/workspace; fresh observations; selected identity and goal valid; enough current/past RGB to localize the target or a brief explicitly uncertain occlusion. Contour completeness may be lower than required by contour_push but the policy cannot start from an uninitialized or swapped target track.

**exit_conditions**

Same observable goal agreement/stability and clearance criterion as the graph alternative, computed by the shared monitor rather than an unexplained neural done bit. Stop also for timeout, cancel, growing uncertainty, invalid sensors or hazards. Final word verification remains with the HLA.

**failure_signatures**

Visual target-marker drift or identity swap, incompatible views, increasing uncertainty under occlusion, error plateau/oscillation, repeated tool motion without target progress, neighbor displacement or command safety rejection. A confident recurrent hidden state is not permission to ignore invalid current observations.

**overlap_role**

Use only causal within-segment histories. Explicit edge-band loss exclusions prevent ambiguous release/next-approach labels from driving the visual temporal model. It is an independent same-data alternative, not a continuation trained on the graph policy's hidden state.

**successor_readiness**

Expose the same current scene version, instance/goal residual and clearance information. The HLA can stop and re-invoke either policy after fresh association; there is no automatic mode cascade or assumed episode phase.

### Heuristic id

dual_view_memory

### Input preprocessing

Implement all common pairing/validity, camera and automatic target-tracking components. Undistort each view with its own declared calibration; retain third global context plus a selected-instance/tool neighborhood crop and wrist contact crop, with crop-to-original coordinate matrices and validity masks. Proposed policy input resolution: 640x360 global images and up to 384x384 detail crops; this is independent of the inspection preview and must preserve small tool/edge detail through evaluation. Include a target mask/soft marker, goal foreground/crop, absolute chart coordinates, robot q/measured dq/tau_ext/transforms, dt, image ages, visibility and missing-gripper masks. Encode last up-to-fifteen causal paired observations; explicit masks handle repeated frames and segment starts. When masks are fragmented, retain the soft marker/track prediction with uncertainty and use RGB rather than supplying a fictional full contour. Persistent ambiguity returns unavailable.

### Limitations

Appearance can overfit brown pieces, stickers, white background, this camera and this inventory; recurrence may memorize ordering without strict goal/instance tests. Same unprovided perception/recognition/backend dependencies as the graph policy. Temporal inference cannot recover arbitrarily long occlusion or certify contact. It may transfer worse to unseen physical shapes despite accepting novel goals; no ranking is known. No success beyond the recorded near-planar single-tool regime is established.

### Pipeline implications

Train an independent goal-conditioned two-view spatial encoder plus causal recurrent/attention state and a multimodal 7-command output distribution. Instance markers restrict target attention; global RGB retains other pieces/tool context, while wrist RGB resolves nearby edge/inside-hole appearance. Time/history lets inference distinguish recent progress, withdrawal and re-contact without adding hindsight phase inputs. Teacher-force only past observations/actions already available at that time; if previous action is used online, use the actually issued command, not the demonstrator's unavailable future command. Predict one current command rather than open-loop replay or an unspecified long action chunk. Optional next-observation representation loss stays within each segment and is training-only. Ship its own trained network, input pipeline and decoder/monitor dependencies in the later implementation; no graph motor network is required.

### Policy contract

**caller_arguments**

Identical instance and image-goal interface, tolerance, workspace and budget as contour_push, enabling HLA replacement without an oracle conversion. Caller chooses the responsibility and call lifetime; the model internally chooses approach/contact/re-contact actions, not the next instance or staging destination.

**input_output_contract**

Causal RGB/state/history and target/goal markers to current command dq[7], status and uncertainty/progress. The goal stream is distinct from live observation. No known-alphabet token, demonstration number, verified contact label or depth metric is required. Return unavailable if mandatory scene identity, goal, calibration or backend verification is missing.

**memory_and_handoff**

Call-local recurrent memory, reset at goal/instance/policy change. A fresh call may warm up from permitted current causal observations using masks; no hidden state transfer or cross-cut training windows. A perception track may persist with its own uncertainty but supplies no future information.

**selection_cues**

The HLA may test this alternative when the graph contour is unreliable but one or both RGB views still support stable identity and local tool tracking, or when recent visual response clarifies re-contact. If contour geometry is high-confidence, contour_push offers a more explicit shape bottleneck. Selection and tolerable occlusion duration must be validated; neither is claimed superior.

**status_and_progress**

Return shared geometric progress with uncertainty plus view consistency, target-marker confidence, temporal innovation and predictive action dispersion. During partial occlusion report degraded observability rather than apparent progress from hidden state alone. Use shared current-observation termination checks and reason-coded blocked/uncertain/timeout.

### Policy id

visual_push

### Rationale

Observed regularity: the robot/tool occludes the third view while wrist frames expose the local letter boundary; contact approaches, pauses and re-contact can look similar in one still (A4800 versus A5600; B3900,4050,4200). Learning problem: infer the current command from causal visual interaction history and an explicit fixed target instead of requiring a perfect geometric reconstruction. Inductive bias: target-marked dual-view attention plus recurrent response memory. Implementable mechanism: separately encoded global/wrist/goal images, robot-state fusion, causal temporal model and supervised current-command distribution. Falsifiable benefit: less degradation than a contour bottleneck under short partial occlusion/edge fragmentation with stable identity; test also whether this costs unseen-shape transfer. It is a substantive independent callable alternative, not merely a seed or evaluation baseline.


All scientific text above is unchanged Runtime API output. Rendering and slicing are developer-owned. No policy code or training exists in this stage.
