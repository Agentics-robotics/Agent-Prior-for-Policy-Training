# Deliver active block to its goal

Carry the active block toward its specified goal pad, lower it to the tabletop, open/release it at the goal, and retreat to a clear state that either supports the next block acquisition or final task completion.

## Segmentation

This skill groups active-block transport, lowering, release, and retreat. Red delivery starts at observation 120, before the red block is fully lifted, and ends at 440 after red has been released on its pad and the open gripper has retreated toward the blue side. Blue delivery starts at observation 500, before or during blue contact/closure, and runs to the final action of each demonstration so that release, settling, and final retreat are retained for task completion. This grouping is color-shared because both red and blue deliveries solve the same active-object-to-goal problem with different target pads. The large starts-before-lift and ends-after-release overlaps provide transition support for acquisition/delivery switching and for the red-to-blue handoff.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo10000`: [120, 440)
- `demo10000`: [500, 773)
- `demo10001`: [120, 440)
- `demo10001`: [500, 765)
- `demo10002`: [120, 440)
- `demo10002`: [500, 770)
- `demo10003`: [120, 440)
- `demo10003`: [500, 756)
- `demo10004`: [120, 440)
- `demo10004`: [500, 770)
- `demo10005`: [120, 440)
- `demo10005`: [500, 771)
- `demo10006`: [120, 440)
- `demo10006`: [500, 773)
- `demo10007`: [120, 440)
- `demo10007`: [500, 760)
- `demo10008`: [120, 440)
- `demo10008`: [500, 775)
- `demo10009`: [120, 440)
- `demo10009`: [500, 773)
- `demo10010`: [120, 440)
- `demo10010`: [500, 774)
- `demo10011`: [120, 440)
- `demo10011`: [500, 777)

## Heuristic 1

Goal-relative active-block delivery. Represent delivery in terms of the selected block's pose relative to its own goal pad, sharing one policy across red and blue. This should reduce color/side memorization and improve placement precision under modest start variation.

**Evidence inspected by the API**

- `demo10000` observations: 120, 220, 280, 320, 340, 440
- `demo10001` observations: 120, 220, 280, 320, 340, 440
- `demo10000` observations: 500, 600, 640, 680, 720, 773
- `demo10011` observations: 500, 600, 640, 680, 720, 777

**Interpretation**

In red delivery, the active block moves from initial y around -0.20 to red_goal [-0.18,-0.25,0.02]; in blue delivery, it moves from y around +0.21 to blue_goal [-0.18,+0.25,0.02]. Both have the same structure in goal-relative coordinates: carry high, reduce xy error, lower, then release/retreat. This addresses the learning gap that raw coordinates separate red and blue trajectories by sign and start pose even though the task logic is identical. It differs from the vertical staging prior by focusing on spatial object-goal invariance rather than a temporal height schedule.

**Applicability**

Transfers to the same two-goal sorting structure or analogous tasks where a selected carried block has an explicit world-frame goal pad and can be lowered and released from a top/side grasp. Requires active object identity, goal pose red_goal or blue_goal, accurate object pose, and similar table height/block size. It assumes the block is already in hand or in the close/lift overlap, not that the policy can solve arbitrary grasp failures.

**Implications for a future training/inference pipeline**

Candidate implementation: condition the action diffusion model on active object and goal-relative features: e_o = p_object - p_goal, e_tcp = p_tcp - p_object, z_object, gripper_width, qpos/qvel, and inactive object pose. The same network is shared for red and blue by selecting p_goal = red_goal or blue_goal and p_object = red_pose or blue_pose from causal observations. Action decoder still outputs absolute Panda joint targets and gripper command. Optional auxiliary target computed from demonstration future: terminal signed goal error min_{k<=H} ||p_object_{t+k,xy}-p_goal_xy||. Deployment: o=scheduler.active; f=concat(p_o-p_goal_o, p_tcp-p_o, qpos,qvel,gripper); a=DiffusionDenoise(f history). The implementation change is explicit goal-relative/color-tied representation, not a scripted waypoint controller.

**Assumptions and limitations**

The prior assumes object-goal relative geometry is the main driver; it may underfit joint-space constraints or table contact timing. Demonstrations have fixed goal pads, so generalization to new goal locations is a hypothesis. Falsify by comparing to a raw joint/object-state policy on held-out starts/goals: if relative goal encoding does not improve object-at-goal accuracy or reduce red/blue confusion, the prior is not beneficial.

### Handoff interface

**Entry conditions**

Measured red delivery entry at obs 120: tcp and red block are low/near contact, fingers about 0.018-0.04 and closing; obs 220 shows the same segment already in carry. Measured blue delivery entry at obs 500: tcp is low over blue, often still open or beginning to close, and obs 600 is high carry. The policy assumes active object is the one to deliver and that no other object obstructs the route.

**Exit conditions**

For red, exit at obs 440 leaves red_pose on red_goal at z≈0.02, gripper open about 0.04, tcp high/near the blue side, so blue acquisition can continue. For blue, exit at final observations 756-777 leaves blue_pose on blue_goal, gripper open, tcp clear at z≈0.299. Task predicates red_at_goal and blue_at_goal are satisfied by the final delivery exits.

**Failure signatures**

Object-goal xy error fails to decrease during carry, block z drops early, object is released while outside the goal pad, qpos fingers open before the object is low, or tcp retreats while object remains elevated. Any active object outside the ±0.06 m goal pad or with z not near 0.02 after release means delivery failed.

**Overlap role**

The acquisition->delivery overlap [120,220) and [500,600) lets this goal-relative policy begin while the object is not yet at full carry height. The delivery->acquisition overlap [340,440) after red release teaches both policies the open retreat and traverse toward the next object.

**Successor readiness**

After red delivery, successor acquisition needs red stable at red_goal, fingers open near 0.04, tcp collision-free above the table, and blue still at its start pose. These are measured at obs 340-440. After blue delivery there is no required successor, but final tcp clear and object-at-goal state are retained for task completion evaluation.


## Heuristic 2

Vertical staging and clearance prior. Separate delivery into lift/high-translate, goal descent, release, and retreat stages using object/tcp height and goal error. This should make the policy less likely to drag the block across the table or release at the wrong height.

**Evidence inspected by the API**

- `demo10000` observations: 120, 160, 220, 280, 320, 360, 440
- `demo10005` observations: 120, 220, 280, 320, 340, 440
- `demo10000` observations: 500, 540, 600, 640, 680, 720, 773
- `demo10001` observations: 500, 600, 640, 680, 720, 765

**Interpretation**

The delivery trajectories show clear vertical organization: in demo10000 red is high by 220, descends near the goal by 280, releases by 320, and retreats by 360-440; blue is high at 600/640, low near 680, and retreats by 720-end. This temporal/height regularity is more specific than goal-relative encoding and addresses the risk of averaging high-transport and low-placement actions in one diffusion mode.

**Applicability**

Applies when safe block transport uses a high carry clearance followed by a near-vertical lowering and a post-release retreat, as in these demonstrations. It depends on table height z≈0.02, block half-height 0.02 m, and a gripper/wrist orientation that can lower without toppling. It should transfer if the obstacle environment remains an open tabletop; it is not a general obstacle-avoidance prior.

**Implications for a future training/inference pipeline**

Candidate implementation: hierarchical/staged diffusion with a latent stage s in {lift_to_clearance, translate_high, descend_at_goal, release, retreat}. Causal features include z_object, z_tcp, xy goal error, gripper width, qpos/qvel. Training labels are derived from current/future state: lift_to_clearance if object z rising and below ~0.25; translate_high if object z>0.25 and xy error large; descend_at_goal if xy error small and object z decreasing; release if object z≈0.02 and gripper opening; retreat if gripper open and tcp z increasing/clear. Train p(s_t|history) plus stage-conditioned denoiser pi(a|obs,s). Objective L=L_diffusion + CE(s,s*) + optional violation penalties for predicted actions that lower object far from goal in attached-labeled windows. Deployment predicts s causally from observation history; no future labels are used at inference.

**Assumptions and limitations**

The height thresholds are inferred from successful trajectories and fixed table geometry; they may not suit taller objects or obstacles. An ablation should remove the stage/height latent and train only goal-relative features; if staged height does not reduce dragging/toppling or improve release timing, the prior is falsified. The dataset cannot establish behavior under unexpected obstacles or a block already partly off the goal pad.

### Handoff interface

**Entry conditions**

Can start from close/lift overlap with active object low but grasped/closing (obs 120 or 500) or from high carry (obs 220 or 600). It assumes active object is either attached or can be finished by the inherited close/lift behavior; table and goal are unobstructed.

**Exit conditions**

Exit for red includes a staged retreat: object on pad at z≈0.02, gripper open, tcp z increasing from low release around obs 320 to high safe approach around obs 440. Exit for blue includes release around obs 680-720 and final tcp z≈0.299 at the goal side with fingers open. These states maintain enough clearance for no further manipulation or next acquisition.

**Failure signatures**

Failure signatures are low-altitude lateral dragging before reaching the goal, descending while xy error remains large, opening at carry height, retreating before the object is table-supported, or a block pose tilted/translated after release. The staging policy should delay release/retreat in those observations.

**Overlap role**

This prior treats overlaps as phase-ambiguous but supported: close/lift at the predecessor boundary and retreat/traverse at the successor boundary. Acquisition can take over from red delivery during the open-retreat phase [340,440) because both policies have seen rising/retreating tcp states with fingers open.

**Successor readiness**

The next acquisition needs the retreat phase completed enough to avoid collision with the placed red block, but not a unique pose. Measured supported range spans obs 340 tcp z around 0.21-0.30 over red goal to obs 440 tcp above/near the blue block side. Unseen are large lateral errors while low over the placed block.


## Heuristic 3

Predicate-and-readiness prior. Train delivery to predict and optimize both task success (active block at its goal) and manipulation readiness (released, gripper open, tcp clear) so that red placement hands off cleanly to blue acquisition and blue placement terminates safely.

**Evidence inspected by the API**

- `demo10000` observations: 280, 320, 340, 360, 440
- `demo10001` observations: 280, 320, 340, 360, 440
- `demo10000` observations: 640, 680, 720, 760, 773
- `demo10010` observations: 640, 680, 720, 774

**Interpretation**

The task contract does not require release, but expert demonstrations consistently open the gripper after placement and retreat, preserving a clean interface. Red is at goal and gripper open by around obs 320, yet the demonstration continues through obs 440 to create space for blue pickup; blue similarly releases around 680-720 and ends with tcp clear. The learning challenge is avoiding a policy that stops as soon as xy is correct while still closed/low. This prior differs from vertical staging by optimizing explicit success/readiness predicates rather than only stage sequencing.

**Applicability**

Useful for tasks whose completion is defined by object-at-goal predicates and where release is not required by the formal contract but is consistently demonstrated for stable handoff and final settling. Requires observable red_pose/blue_pose, goals, finger state, and enough post-release data to learn retreat. Transfers to similar placement tasks where object stability after release matters for downstream skill selection.

**Implications for a future training/inference pipeline**

Candidate implementation: add auxiliary heads to the delivery policy predicting red_at_goal/blue_at_goal for the active object and successor_ready over horizons 0.5-2 s. Labels are computed from demonstration observations and task contract: at_goal_t = |p_o.x-goal.x|<0.04? or formal full containment proxy using object half-size and pad half-size 0.06, z within 0.011 of 0.02; successor_ready_t = at_goal_t AND finger_qpos>0.035 AND tcp_z>0.15 for red, or terminal_ready for blue. The denoiser is conditioned on the predicted readiness embedding and trained with L_diffusion + BCE(at_goal,ready) + terminal-weighted action loss on release/retreat windows. At inference, labels are predicted causally from observations; the selector may use ready probability for handoff/termination.

**Assumptions and limitations**

Because all demonstrations succeed, the terminal predicate classifier will mostly see positive final states and synthetic negatives from earlier in the same episodes; it cannot prove robustness to disturbances after release. Falsify by removing terminal/predicate auxiliary losses and checking whether final red/blue containment, release timing, or successor switch success degrades. If no improvement, explicit predicate shaping is unnecessary.

### Handoff interface

**Entry conditions**

This policy can take over from acquisition when a block is carried or being lifted and has access to the task goal predicate. It assumes no need to keep holding the block after it is at goal; gripper release and retreat are acceptable, as demonstrated even though the task contract does not require release.

**Exit conditions**

Exit is an interface state, not only a predicate: active object xy lies within the goal pad and z≈0.02, qpos fingers are open about 0.04, tcp is clear above the table (red stop obs 440 near blue side z≈0.25-0.27; blue final obs 760-777 z≈0.299), and qvel is near zero. For red, successor acquisition is ready because gripper is open and moving/positioned toward blue; for blue, the task can terminate.

**Failure signatures**

Predicate-at-goal classifier low, active object at goal but gripper still closed around it, tcp still low enough to collide with successor approach, block shifts after release, or retreat that recontacts the placed object. These signatures mean the next acquisition or task termination is unsafe.

**Overlap role**

The red post-release overlap [340,440) is deliberately assigned to both delivery and acquisition. This prior makes delivery learn release settling and retreat-to-successor-readiness, while acquisition learns it may enter before the arm reaches a fixed blue pregrasp pose.

**Successor readiness**

The next policy needs a stable open-gripper manipulation state. Measured support includes red settled at goal by obs 320/340, gripper open by obs 320, and tcp retreating to blue side by obs 440. It does not support recovery if red is outside the goal pad or still being held.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
