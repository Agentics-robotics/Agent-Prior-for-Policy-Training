# Place blue inside open drawer with terminal release context

Finish incoming red placement/release if necessary, acquire blue outside and place it fully inside the still-open drawer while red is simultaneously on the pad. Retain release, settling and retreat as useful optional terminal continuation under the one-observation completion rule.

## Segmentation

Start with red still held and descending over pad at z approximately 0.211 m: 575 in ten trajectories, 581 in demo1001 and 582 in demo1005. The prefix finishes red placement, finger opening, settling, retreat, blue approach/descent/closure and initial lift, shared with red evacuation. Then retain blue lift/transport, insertion lowering, release, settling and full terminal retreat/dwell. Stop at each inspected original terminal observation (1063-1077); exclude nothing. All final endpoints have d approximately 0.2973-0.2974 m, red on pad near z=0.020 and blue fully contained near z=0.063. Earlier sampled observations such as demo1000:980 and demo1001:990 already satisfy the one-observation contract while blue is held. The suffix is useful release/idle context, not evidence that release, sustained hold, speed or clearance is required. The final image demo1010:1069 corroborates the arrangement. Group across demonstrations because receiving geometry and manipulation dependencies match despite pickup timing, source placement and release-yaw differences.

Source action ranges use [start, stop); each segment also retains its final observation.

- `demo1000`: [575, 1065)
- `demo1001`: [581, 1077)
- `demo1002`: [575, 1063)
- `demo1003`: [575, 1066)
- `demo1004`: [575, 1065)
- `demo1005`: [582, 1070)
- `demo1006`: [575, 1064)
- `demo1007`: [575, 1069)
- `demo1008`: [575, 1071)
- `demo1009`: [575, 1065)
- `demo1010`: [575, 1069)
- `demo1011`: [575, 1065)

## Heuristic 1

Articulated containment margins: encode moving-cavity/pad geometry, rotated extents and strict opening/height thresholds. Hypothesis: goal-margin learning generalizes better than aiming TCP or block center at a single demonstrated point.

**Evidence inspected by the API**

- `demo1000` observations: 575, 595, 605, 615, 815, 845, 880, 960, 980, 1000, 1040, 1065
- `demo1001` observations: 581, 615, 825, 859, 990, 1015, 1077
- `demo1008` observations: 575, 825, 853, 985, 1005, 1071
- `demo1010` observations: 575, 815, 849, 985, 1005, 1069

**Interpretation**

Drawer d changes from 0.300 to about 0.297 m during red placement, so the receiving cavity is articulated, not a fixed world point. At demo1000:980 and demo1001:990 blue z approximately 0.067 satisfies the goal while g=-1; release changes position/orientation before z settles near 0.063. Demo1008/1010 have different final yaw yet remain contained. The relevant relation is rotated full containment, not exact centering, release or dwell time. This prior changes geometry representation and supervised prediction, not contact memory or candidate scoring.

**Applicability**

Known world-frame pad/drawer geometry, block half-size 0.02 m, wxyz orientation and prismatic drawer axis. Poses and d are causal inputs. Changed target geometry requires a supplied revised contract; cabinet pose is not inferred from an unspecified visual model.

**Implications for a future training/inference pipeline**

From causal state compute c=[0.19-d,0,0.035], pad p=[-0.18,-0.30,0.02] m, and extents e_i=0.02*sum_j abs(R_ij) from wxyz block quaternions. Features: m_blue_xy=[0.172,0.182]-abs(blue_xy-c_xy)-e_blue_xy; m_red_xy=[0.06,0.06]-abs(red_xy-p_xy)-e_red_xy; signed Z-interval margins; d-0.26. Include sign-invariant rotations, TCP-object relations, heights and qpos/qvel. Condition a transformer action diffusion network on goal-margin/object/robot tokens; output original-normalized absolute eight-channel actions with shared decoding and bounds. Add future-margin regression from denoiser latents at several horizons, using masked demonstration futures as targets only. Preserve red and blue tokens throughout; do not require blue-inside at every training frame. At inference compute causal margins and replan short denoised prefixes. Prediction: more robust placement under opening/orientation variation than center-distance conditioning, without changing the completion rule.

**Assumptions and limitations**

Final placements are far from XY boundaries and nearly upright. Exact predicate computation is known, but benefits for highly rotated/edge placements remain hypothetical. Compare margin-aware, center-distance-only and unstructured encoders at matched capacity. No gain or oversensitivity at strict Z thresholds falsifies benefit. Arbitrary rim-clearance and collision-free insertion are not established.

### Handoff interface

**Entry conditions**

Earliest entry has red still held and descending over pad: red/TCP z approximately 0.211/0.212 m, fingers approximately 0.0182 m, g=-1, d approximately 0.300 m and blue z=0.020 m. Later supported entries are red release, empty retreat, blue approach/closure or initial blue lift near z=0.14 m. Compute both objects' margins; red success cannot be assumed at segment start.

**Exit conditions**

Apply exactly the simultaneous contract: d>0.26 m; red z in (0.014,0.031), blue z in (0.053,0.074); full rotated XY containment on pad/in cavity. Inspected finals have d approximately 0.2973-0.2974 m, red near [-0.176 to -0.177,-0.298 to -0.304,0.020] m and blue near [-0.1717 to -0.1728,0.0728 to 0.0742,0.063] m. Retained useful terminal continuation has fingers near 0.04 m and TCP z near 0.323 m; these are not additional success conditions.

**Failure signatures**

Center-inside but negative full-containment margin, blue z>=0.074 m, drawer d<=0.26 m or red leaving pad. Grasped blue near z=0.067 m can already satisfy success; lack of release alone is not failure.

**Overlap role**

Both policies learn red descent/release/retreat and blue approach/closure/lift. Retain unfinished red margins during early takeover and red preservation thereafter rather than deleting red when the option name changes.

**Successor readiness**

No required downstream manipulation. A task monitor can accept one simultaneous successful observation, including grasped blue. If continuing to idle, the retained suffix teaches release, settling and upward retreat. Do not introduce speed, hold-time or clearance requirements; demonstrated terminal values provide context only.


## Heuristic 2

Contact-belief adaptive diffusion: infer attachment/support transitions from causal history and shorten commitment around uncertain changes. Hypothesis: event-aware replanning improves release and handoff timing without an episode clock.

**Evidence inspected by the API**

- `demo1000` observations: 575, 595, 605, 615, 625, 680, 750, 805, 815, 825, 830, 845, 960, 980, 1000, 1040
- `demo1001` observations: 581, 600, 615, 625, 815, 825, 835, 859, 990, 1015
- `demo1008` observations: 575, 615, 630, 815, 825, 853, 985, 1005, 1071
- `demo1011` observations: 575, 815, 845, 846, 980, 1000, 1065

**Interpretation**

At demo1000:805 TCP is already at blue height with open fingers; at 815 width is approximately 0.0182 m, but substantial lift appears only by 830-845. Demo1001 delays this sequence: 815 is still open and 835 is closed on the table. Demo1011:980 is held-blue near z=0.067 whereas 1000 is released-blue near z=0.063 with retreat starting. Similar geometry needs different actions. Causal contact belief plus adaptive replanning addresses ambiguity and stale chunks, unlike a geometric encoder alone.

**Applicability**

Same contact mechanics and reliable causal pose/proprioception history. Especially relevant during red release, blue closure and insertion where supported/held poses can look similar. No force/torque sensor is assumed.

**Implications for a future training/inference pipeline**

Train a causal recurrent state-space encoder on qpos/qvel, tcp/red/blue poses, d/vd and available previous commands. Infer a discrete contact mode and continuous relative-pose latent. Training labels use finger widths, transform changes, elevation and short future co-motion windows; futures are targets only and g alone is not attachment. Condition one action diffusion decoder on the filtered posterior, retaining original-normalized absolute joint/gripper outputs and common decoding. Train denoising plus contact-transition and next-relative-motion prediction losses with variable-prefix masks. At inference update belief at 20 Hz; shorten execution prefixes when posterior entropy or predicted contact-change probability is high and allow longer prefixes during confident free transport. This changes temporal computation rather than scripting a contact threshold controller. Prediction: fewer premature releases and stale long chunks across changing takeover times.

**Assumptions and limitations**

Co-motion labels do not resolve every support/contact configuration. Successful-only training can make posteriors overconfident. Compare with a memoryless margin denoiser and fixed prefix lengths; ablate contact supervision and adaptive horizon separately. No recovery, force regulation or uncertainty calibration under novel contacts is established.

### Handoff interface

**Entry conditions**

Initialize belief from actual fingers and recent red/blue/TCP relative motion, masking missing history. Earliest entry still has red descending with g=-1. At blue approach g can be +1 at TCP z approximately 0.019 m (demo1001:815), whereas demo1000:815 is already closed. Equal timesteps are not equal contact states.

**Exit conditions**

Belief progresses red-held to red-released/free, blue-closing/held, lowered-blue and released-blue/retreating. Task success may occur in blue-held mode near z=0.067 m. Retained terminal mode has blue resting near z=0.063, fingers approximately 0.04 m and TCP near z=0.323 m. Release is useful continuation, not a mandatory success label.

**Failure signatures**

Release while blue height remains outside its interval, retreat before blue decouples from the tool, unsupported phase reversal, or held-blue belief while blue remains at its source as TCP rises. Finger-velocity asymmetry is present in successful data and is not automatic failure.

**Overlap role**

The long overlap trains belief through red grasp persistence, finger opening, empty travel and blue attachment. It supports switching with fingers moving or the arm lifting, but not an unseen drop or obstruction.

**Successor readiness**

A terminal monitor requires measured simultaneous goal fields, not the belief's release label. Continued idle should use observed decoupling to sustain release/retreat. A history buffer may transfer, but another policy must re-infer without shared private latent semantics. Switching confidence thresholds require evaluation.


## Heuristic 3

Concurrent-goal preserving lookahead: rerank diffusion proposals by predicted containment progress and retention of red/drawer goals. Hypothesis: short-horizon checks reduce destructive terminal actions and improve simultaneous success beyond imitation alone.

**Evidence inspected by the API**

- `demo1000` observations: 575, 595, 605, 615, 640, 845, 880, 960, 980, 1000, 1040, 1065
- `demo1001` observations: 581, 600, 615, 859, 990, 1015, 1077
- `demo1008` observations: 575, 853, 985, 1005, 1071
- `demo1010` observations: 575, 849, 985, 1005, 1069

**Interpretation**

Success is conjunctive: correct blue motion is insufficient if red leaves the pad or the drawer closes. Demo1000:595-615 shows d retreating slightly from nearly 0.300 to 0.2973 m while staying open. Blue descends from 0.165 at 960 to approximately 0.067 at 980, then changes pose and settles to 0.063 after release. Demo1008/1010 have different release yaw/translation yet preserve all goals. These observations motivate jointly evaluating action consequences. This is an inference/objective prior, unlike margin encoding or contact filtering; its counterfactual usefulness is a hypothesis.

**Applicability**

Known completion geometry, same robot/controller dynamics and action proposals near the demonstrations. Appropriate for local selection near red release and blue placement, not new recovery plans or collision certification.

**Implications for a future training/inference pipeline**

Train an action-conditioned ensemble transition model on original state/action sequences to predict qpos/qvel, TCP/object poses and d/vd over short horizons using physical increment/geodesic losses. Independently train a full-state/history diffusion proposal on original-normalized absolute actions. At inference sample K chunks, predict rollouts and compute exact margins from world poses with e_i=0.02*sum_j|R_ij|. Rank by state-conditioned progress minus loss of already-achieved red/drawer goals, negative blue containment margins near placement, uncertainty and deviation from demonstrated action likelihood. Learn a remaining-progress head from demonstration futures to avoid forcing blue-inside during the red prerequisite prefix; those labels are never inference inputs. Execute only a short prefix and recompute from actual observations. After observed success score goal preservation, not a compulsory release/dwell objective. Use common action normalization/bounds. There is no force or full collision model. Prediction: local lookahead better preserves concurrent goals than unscored diffusion or an encoder-only geometric prior.

**Assumptions and limitations**

Counterfactual dynamics are weakly identified from successful trajectories; accurate on-path prediction need not rank sampled alternatives correctly. Restrict proposals with support/likelihood penalties and test uncertainty calibration. Compare proposal-only, guided and randomly reranked variants with equal sampling budget. Model exploitation, goal violations or no gain falsify benefit. Rim collision avoidance and failed-placement recovery remain unproven.

### Handoff interface

**Entry conditions**

Allow demonstrated red-lowering entry with d approximately 0.300 m, red z approximately 0.211 m, blue at outside source and closed fingers, or later shared release/approach/lift states. Condition the predictor on the currently coupled object. Off-manifold lost-grasp predictions cannot justify aggressive selection.

**Exit conditions**

Optimize actual simultaneous success, not an exact final joint pose. After success, continue only demonstrated-like release/retreat predicted to preserve d>0.26 and red/blue containment. Retained final state has red z approximately 0.020, blue z approximately 0.063, d approximately 0.2974 m, open fingers and high TCP; stopping earlier on a valid observed conjunction is allowed.

**Failure signatures**

Predicted containment improves but measured blue stays outside, red leaves the pad during blue travel, d declines toward its threshold, or selection favors unnatural pauses/unsupported commands. Prediction error motivates replanning/uncertainty reporting, not a claim of trained recovery.

**Overlap role**

Both policies learn actual red placement/release and blue acquisition. The scorer must preserve those prerequisites rather than seek immediate blue containment while the gripper carries red. Progress conditioning covers the entire overlap.

**Successor readiness**

No mandatory successor. The task monitor uses actual poses/d at one observation; predicted success is insufficient. Continued release/retreat should stay within demonstration support and preserve achieved goals. Never replace the supplied rule with hold-time, speed or release requirements.


## Provenance

Heuristic statements and segment choices are API-authored hypotheses; this stage does not validate their effectiveness.
Provider provenance: `responses_api`. Markdown formatting and data slicing are framework operations.
Segments are derived samples, not additional independent demonstrations.
