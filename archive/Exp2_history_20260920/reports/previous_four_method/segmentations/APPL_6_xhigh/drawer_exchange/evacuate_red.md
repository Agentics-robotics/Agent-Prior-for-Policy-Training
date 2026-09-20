# Evacuate red to pad and establish blue pickup

Finish remaining opening/release if necessary, acquire red and place it fully on the outside pad while retaining drawer opening, then release/retreat and establish blue pickup through initial lift for the insertion policy.

## Segmentation

All demonstrations share red evacuation with centimetre-scale source variation and a fixed outside pad. Begin at independently inspected index 195 in every trajectory: d approximately 0.1863 m, vd approximately 0.1071 m/s, handle-gripped fingers and red moving with drawer. This intentionally precedes release and supports taking over while the predecessor still manipulates the drawer. After red acquisition, carry to the pad, lower, release and retreat. Retain the following complete blue approach/pickup through blue z approximately 0.139-0.146 m. Endpoints vary from 845 to 859: demo1001 is at blue z=0.0346 at 845 and 0.1405 at 859; demo1005 reaches 0.1404 at 852; demo1008 reaches 0.1409 at 853; demo1010 reaches 0.1461 at 849. Other individually inspected endpoints are 845/846/847. Images demo1001:615 and demo1008:815 corroborate red release and outside-blue approach. This long option learns prerequisite completion and successor preparation; its unique goal is red evacuation, not complete blue insertion.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo1000`: [195, 845)
- `demo1001`: [195, 859)
- `demo1002`: [195, 847)
- `demo1003`: [195, 845)
- `demo1004`: [195, 846)
- `demo1005`: [195, 852)
- `demo1006`: [195, 846)
- `demo1007`: [195, 847)
- `demo1008`: [195, 853)
- `demo1009`: [195, 846)
- `demo1010`: [195, 849)
- `demo1011`: [195, 846)

## Heuristic 1

Object-role relational attention: represent pickup/placement with object-to-tool and object-to-target relations while retaining drawer and completed-goal context. Hypothesis: explicit roles reduce spurious absolute-coordinate dependence and improve transition reuse.

**Evidence inspected by the API**

- `demo1000` observations: 195, 240, 390, 450, 475, 480, 560, 575, 595, 615, 680, 750, 805, 815, 845
- `demo1001` observations: 195, 240, 390, 450, 476, 581, 600, 615, 760, 825, 859
- `demo1008` observations: 195, 474, 575, 615, 760, 825, 853

**Interpretation**

Red pickup shifts from approximately [-0.197,-0.063] m in demo1000 to [-0.195,-0.077] in demo1001, yet placement uses the same pad. Blue approach shifts across approximately [-0.401,0.291], [-0.409,0.311] and [-0.408,0.315]. TCP is roughly 9 mm to the held object's -X side, so tool position and object goal cannot be equated. Relational encoding shares these dependencies while retaining reachability. It changes observation structure, unlike waypoint hierarchy or an attachment-physics objective.

**Applicability**

Same Panda workspace/gravity with object and pad poses known. Transfer is most plausible for the demonstrated centimetre-scale source variation and modest reachable goal shifts. Global joint translation invariance is not assumed; pad location is supplied, not visually inferred.

**Implications for a future training/inference pipeline**

Build typed robot/drawer/red/blue/pad tokens from causal qpos/qvel, tcp/red/blue poses and d/vd. Include red-TCP and blue-TCP transforms, red-pad displacement, object-drawer relations, physical heights and absolute/base anchors. Use quaternion-sign-invariant rotations and learned role attention conditioned on observed grasp/release evidence, not an elapsed-time selector. Condition action diffusion on these tokens; output eight original-normalized absolute joint/gripper targets with common decoding/bounds. Train denoising plus auxiliary next-object-displacement prediction on attended tokens, using future poses only as targets. At inference recompute relations, infer soft attention and denoise/replan short prefixes. This prior makes object-to-tool/target dependencies explicit while retaining background goals. Prediction: improved sample efficiency for differing red y and blue xy placements relative to a world-coordinate MLP.

**Assumptions and limitations**

Source variation is only centimetres, objects never swap roles and all targets use one workspace. Compare typed relations to absolute-state concatenation and target-only crops; no gain on held-out placements or erroneous role switching falsifies the claim. Color-role exchange, arbitrary object count, broad obstacle avoidance and dropped-block recovery are not established.

### Handoff interface

**Entry conditions**

Earliest entry at 195: d approximately 0.1863 m, vd approximately 0.1071 m/s, TCP approximately [-0.326,0,0.128] m world, fingers approximately 0.0071 m and g=-1, red moving with drawer and blue on table at z=0.020 m. Later supported entries include released handle, empty retreat, red closure or red rising near z=0.143 m. Do not assume red is already held.

**Exit conditions**

Primary goal is full red containment on the world pad centered [-0.18,-0.30,0.02] m with z in (0.014,0.031) and d>0.26 m. Retained suffix releases red, retreats and acquires blue. Late exit has blue z approximately 0.139-0.146 m, fingers approximately 0.0182 m, g=-1 and ongoing lift; preserve that grasp.

**Failure signatures**

Attention stays on red after placement/release, switches to blue before release, uses moving-drawer coordinates for the fixed pad, or blue rises while red is outside the pad. Red/TCP decoupling during transport and d<0.26 m are observable violations.

**Overlap role**

Incoming overlap covers late pull through red pickup; outgoing overlap covers red held-descent through release/retreat and blue acquisition. Retain both objects and drawer in the encoder instead of cropping away the object whose state is becoming a constraint.

**Successor readiness**

Blue-insertion receives full causal state. Its earliest supported takeover still has red z approximately 0.211 m and closed fingers at 575 or 581/582, so it must finish red lowering/release. Later takeover may be free travel or blue lift. The supported set is a temporal manifold, not a rectangular spatial tolerance region.


## Heuristic 2

Clearance waypoint hierarchy: factor relocation into learned spatial intents and local action diffusion. Hypothesis: hierarchical prediction preserves lift-before-translate and release-before-retreat organization across timing variation.

**Evidence inspected by the API**

- `demo1000` observations: 240, 260, 300, 390, 440, 450, 480, 560, 575, 595, 605, 615, 625, 640, 680, 750, 805, 815, 830, 845
- `demo1001` observations: 300, 390, 476, 581, 600, 615, 625, 680, 760, 825, 835, 859
- `demo1005` observations: 474, 582, 615, 815, 845, 852

**Interpretation**

TCP retreats from 0.128 to 0.369 m after handle release; red travels high around z=0.296 at 560 before descent; after red release TCP rises from 0.033 to about 0.300 m before crossing toward blue. Demo1001/1005 reach comparable phases later. Flat long chunks entangle route organization and PD transients. A future spatial-waypoint bottleneck addresses this scale separation; unlike an event classifier it predicts where to go and conditions a separate local policy.

**Applicability**

Observed ordered relocation with room for vertical clearance and similar drawer/pad geometry. Assumes similar route topology without required regrasp. The hierarchy includes the drawer-release prerequisite and blue-acquisition suffix, not just already-grasped transport.

**Implications for a future training/inference pipeline**

Train a two-level Diffusion Policy. A causal full-state encoder feeds a waypoint diffusion model predicting next robot/object waypoint, event type (release, retreat-high, pickup-low, lift-high, target-high, place-low) and estimated duration. Targets come from future tcp_pose, selected object pose, finger width and qpos at observed transitions; they are labels, not causal inputs. Waypoints use object/goal-relative xyz, sign-invariant orientation and robot qpos for posture. A low-level action diffusion decoder conditions on current state and predicted waypoint, outputting seven original-normalized absolute joints plus g. Loss combines waypoint/event/duration supervision and action denoising; mix teacher-forced and predicted waypoints to reduce mismatch. At deployment refresh waypoints at a slower event-aware cadence and actions at 20 Hz, advancing from current state rather than clock alone. No external planner or IK is required. Prediction: a spatial-intent bottleneck improves long-window consistency and tolerance to variable approach duration.

**Assumptions and limitations**

Point poses do not specify complete collision geometry, so a high waypoint is not collision certification. Only one route is demonstrated. Compare to flat action diffusion with the same observations/history; ablate waypoint supervision and vary action horizon. Skipped contacts, high-level lag or no improvement across switch times falsify benefit. Alternative-route and failure recovery behavior are not established.

### Handoff interface

**Entry conditions**

Infer the next feasible waypoint from current manipulation state. Earliest entry is still attached to the moving handle with fingers approximately 0.0071 m; later entries may have open fingers at TCP z approximately 0.369 m or red held near z=0.143 m. Predict unfinished release/retreat prerequisites rather than jumping directly to transport.

**Exit conditions**

Red reaches approximately z=0.020 m on the pad after release; TCP retreats to approximately 0.300 m before blue approach/grasp. The expanded endpoint is during blue lift near z=0.14 m, not at a static completed high waypoint. Replan from ongoing motion instead of forcing a zero-velocity boundary.

**Failure signatures**

Waypoint crosses the drawer rim too low, skips release and moves laterally with red near the table, or is chased after the object fails to follow. Advancing by duration despite unchanged object state is premature transfer.

**Overlap role**

Incoming overlap contains the full release-to-red-lift chain. Outgoing overlap includes red descent/release/retreat and blue pickup/lift. Both policies learn to resume inside these chains rather than only at abstract waypoint boundaries.

**Successor readiness**

Transfer physical state and optionally an advisory predicted waypoint, not a compulsory private code. Blue policy must finish red placement from z approximately 0.211 m on early takeover or continue blue lift with g=-1 on late takeover. Demonstrated heights are references; clearance tolerances remain untested.


## Heuristic 3

Attachment-consistent action learning: approximately conserve a held block's tool-relative transform, then switch that constraint off at release. Hypothesis: a masked rigidity objective improves coordinated arm/gripper chunks without imposing one nominal grasp offset.

**Evidence inspected by the API**

- `demo1000` observations: 195, 450, 465, 475, 480, 560, 575, 595, 605, 615, 815, 830, 845
- `demo1001` observations: 450, 476, 581, 600, 615, 825, 859
- `demo1008` observations: 450, 474, 575, 615, 825, 853
- `demo1010` observations: 815, 845, 849

**Interpretation**

Demo1000 red/TCP pairs at 480 and 560 preserve approximately [-0.0090,0.0002,0.0009] m offset while the arm moves substantially; at 615 red becomes stationary as TCP retreats. Blue has analogous coupled motion at 815-845. Demo1008/1010 have different small rotations/finger dynamics, arguing against one hard-coded grasp transform. The prior constrains consequences of coordinated actions, unlike a relational observation encoder or spatial waypoint plan.

**Applicability**

Rigid blocks, parallel Panda fingers and approximately stable grasps without intended in-hand motion. Causal pose history is needed to estimate attachment; g=-1 is not a contact measurement. Handle attachment requires a prismatic constraint distinct from block rigidity.

**Implications for a future training/inference pipeline**

Compute causal T_tcp^-1*T_object, recent transform changes, qpos[7:9], qvel and available previous commands. Learn an attachment-identity posterior over handle/red/blue/none and a continuous held-object transform. Use co-motion windows for training labels; future windows are labels only. Feed the posterior/transform and full state into an absolute-action diffusion denoiser. Train an action-conditioned short-rollout state head on observed future TCP/object poses. On confidently attached block windows, penalize predicted geodesic/translation changes of T_tcp^-1*T_object; supervise release masks so free retreat is not constrained. For handle windows use only prismatic X coupling. Objective = noise loss + state prediction + masked attachment consistency. At deployment infer attachment causally, denoise and replan short absolute-action prefixes with common normalization/bounds; optionally shorten prefixes when the auxiliary prediction is uncertain. Prediction: this physical dependency reduces independent arm/gripper errors and loss of held objects without prescribing a route hierarchy.

**Assumptions and limitations**

The roughly 9 mm offset is empirical, not universal calibration. Co-motion can indicate support or pushing as well as grasping, so masks are uncertain. Ablate attachment consistency while holding the denoiser fixed and compare lift/drop prediction and handoffs. Suppressing legitimate release falsifies the bias. No dropped-object or regrasp recovery is demonstrated.

### Handoff interface

**Entry conditions**

Earliest entry is handle-attached: d approximately 0.186 m increasing and fingers approximately 0.0071 m. Infer attachment identity from relative pose history instead of initializing a fixed red transform. Supported later entries include free motion, fingers closing from 0.04 toward 0.0182 m around red, and red lift.

**Exit conditions**

Deactivate red rigidity after fingers open and red stays around z=0.020 m while TCP rises. At full suffix endpoint, blue's tool-relative transform is approximately conserved during initial lift: blue near z=0.14 m, TCP roughly 9-10 mm to its -X side, fingers approximately 0.0182 m and g=-1. Preserve blue attachment at transfer.

**Failure signatures**

Object/TCP transform drifts during intended transport, object stays on support as TCP rises, attachment identity changes without release, or red co-motion is enforced during empty retreat. Asymmetric finger qvel alone is not slipping evidence because it occurs in successful grasps.

**Overlap role**

Incoming overlap changes handle coupling to free motion to red attachment; outgoing overlap changes red attachment to released red and then blue attachment. Both policies observe enough before/after closure to estimate a transform instead of assuming a nominal offset.

**Successor readiness**

Give the successor causal pose/finger history to re-estimate attachment. A private transform latent cannot replace physical observations. Early blue-policy takeover needs red descent/release; late takeover needs continuous gripped-blue lift. Greater attachment-error tolerance and slip recovery are unestablished.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
