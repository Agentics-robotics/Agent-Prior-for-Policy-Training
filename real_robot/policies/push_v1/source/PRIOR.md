# Implemented cut_v3 priors

## Authorization, scope and immutable identities

This package implements exactly `contour_push / boundary_graph` and
`visual_push / dual_view_memory` for the frozen instance-relocation dataset.
The 22 cuts, supervision ranges, 30-row handoff exclusions, physical referents,
and stop-minus-one goal anchors are unchanged. The earlier cut-only boundary
is superseded by the current implementation/training authorization. There is
no third policy, task-order selector, per-letter controller, optimizer, training
CLI, hardware client, preliminary performance study or performance selection.

The executor, not this package, independently trains the two neural models:
20,000 updates each, seed 0, AdamW, 500-step warmup/cosine decay, EMA .999 and
the prescribed checkpoint schedule. All demonstrations are training data.
Parameter initialization, optimizer state and EMA are independent. The package
shares deterministic perception and cached observations, not motor weights.
`check_package` is only the authorized syntax/import/interface check. No robot
success or unseen-shape competence follows from that check or a training loss.

## Actual evidence and asset adaptation

The supplied metadata and original paired-state summaries were read. Third
views at A 0, 1719, 4800, 6139 and 9014 and wrist views at A 0, 4800 and 5600
show separated warm-brown pieces on a neutral light surface and a long neutral
rod, including inner-hole engagement. They do not establish a tool width,
contact force, tabletop plane, controller semantics or success labels. The
metadata identifies the installed TCP as already set as robot EE; projection
therefore uses `T_base_ee` without adding another 0.2 m offset. The calibrated
wrist intrinsics, not the distinct factory wrist intrinsics, are used.

**API-owned perception adaptation:** there are no installed/downloaded generic
segmentation or flow weights. Rather than invent those assets, `perception.py`
implements a class-free material-contrast detector, forward appearance/geometry
association, contour extraction, projection and image conversion. HSV warm
chroma, neutral-surround evidence, connected-component size bounds and small
JPEG-noise closing generate automatic proposals. Foreground holes are retained;
the gray tool is excluded by material contrast rather than painted over using
future images. No alphabet/font bank, trained character recognizer, initial
object-coordinate list, per-step clicks or per-move annotations are used.

This is a substantial change from the source plan's *assumed generic learned
perception*. It makes the present observation interface runnable without those
weights, but its perception generality is limited to contrasting, sufficiently
separated warm-material pieces at approximately the recorded apparent scale.
A new physical outline or character is not looked up and can be represented;
an arbitrary new color/material, touching compound, heavy shadow or occlusion
is NOT guaranteed to be segmented. Confidence/margin thresholds are engineering
heuristics, not empirically calibrated probabilities. Touching pieces can evade
ambiguity gates by forming one component. These failures are shared by both
policies. This is not a claim to meet open-world perception/generalization.

No external tensor assets were requested or loaded. Neural weights start from
scratch. Actual dependencies are the interface-provided torch 2.14.0, numpy
1.26.4, cv2 4.11.0 and scipy 1.17.1, plus standard Python modules and the public
read-only capability. No pretrained-weight provenance/license is being assumed;
no extra model license or overlap claim exists.

## Causal conversion and frozen goal rule G

1. `prepare_data` walks every original segment in forward order, resetting the
   tracker and temporal context at each cut. It decodes each current paired RGB
   once; specified anchors are reread only for label extraction. No timestamps
   alter pairings. There is no N+1 row or prior-segment history.
2. The same `convert_observation` is called by preparation, inventory and `act`.
   It undistorts at original 1280x720 using Brown-Conrady K unchanged, detects
   foreground, updates a forward-only Hungarian tracker, computes robot/timing
   values plus missingness, and caches image representations. Current masks,
   centroids, crops and tool projections never use anchors or action labels.
3. Tracks use observed position, smoothed causal velocity, soft area/shape/color
   cues and overlap. Ambiguous assignments are not accepted. Unobserved tracks
   may persist for three seconds but are not emitted as observed contours.
   Fragmentation may yield only a visible component; hidden shape is not filled.
   An unrecoverable ID break can exclude earlier rows, explicitly counted.
4. The 22 boxes in `goals.py` are verbatim frozen training-label anchors, in
   ORIGINAL distorted pixels. Only foreground inside the designated box is
   extracted and then undistorted. Overlap associates it to a forward-produced
   track. This retrospectively labels **which** causal track is selected, not
   its current pose. Neither boxes nor segment/referent IDs are online selectors.
5. Earliest sufficiently visible causal reference versus goal shape is registered
   on normalized silhouettes at five-degree increments. Near-tied rotation modes
   remain an ambiguity set. Low overlap, extreme area change, no foreground or
   ambiguous anchor association suppress that segment's conditional action
   examples and report a derivation failure. This registration is approximate,
   not a verified planar pose. Anchor RGB contains only the selected foreground,
   never the future scene/word. No unspecified intermediate intention is invented.
6. Required q/EE/flange state, usable paired image ages, finite command labels and
   a currently observed selected component determine eligibility. There is no
   null-command zero filling. Common action examples are exactly the same for
   both priors, with filtering precedence and every per-segment count in
   `metadata.coverage`. All original frames remain numeric context/provenance.
   No actual eligible count or coverage claim is made before executor preparation.
   Excluded motion is not claimed trained. If there are no eligible examples,
   preparation fails rather than training a stub.

`native_action` is float64 parsed directly from finite `action_json.dq`.
Measured state `dq` is a different input. Targets are converted to float32 only
for neural arithmetic. Raw source v/w/spacemouse/target_pose remain untouched in
the read-only recording; none is used to reconstruct missing commands.

Rows get positive elapsed-robot-time weights, clipped to [0.0001, 0.2] s,
normalized within each cut and divided over the cuts for the same physical
referent within that episode. That gives equal referent/episode mass, not extra
evidence for long L trials or revisited instances. The first-row *weight* uses
the segment median dt; it is not timestamp or observation imputation. Repeated
commands/frames retain their elapsed-time mass, not independent-decision status.
Frozen referent grouping is used only for sampling weights and never as a model
feature. No absolute timestamp, source index, segment ID, class or episode token
is fed into a network.

Robot q, measured dq, tau_ext, EE/flange and derived wrist-camera transforms
have explicit finite-value masks. Negative gripper position and nonfinite width
are invalid, not an open/closed measurement. Timestamp differences, repeated
image flags, signed image ages and their validity are features; clocks are not
aligned. State numerical preconditioning is followed by loss-weighted training
mean/std, with an std floor. The native-command mean/std are also saved in
metadata and model buffers for identical reload/prediction behavior. Native
command units are not asserted to be radians/second by normalization.

## `boundary_graph`: mechanism preserved

The graph contains up to 192 padded nodes, with a variable valid-node mask.
Approximately uniform arc-length samples retain separate circular adjacency for
outer boundaries, inner boundaries and each neighbor. Budgets are 64 current,
64 goal and up to 48 nearby-other nodes, plus a separate projected-tool node.
Features include normalized absolute and target-centered position, tangent,
normal, curvature proxy, hole bit, current/goal/neighbor/tool/workspace role,
tool-relative displacement, signed distance to goal and visibility confidence.
The generic builder supports workspace contour nodes, but the trained path uses
no claimed physical workspace contour: caller image workspace/obstacle constraints
are enforced by the executable monitor. This is an explicit adaptation of the
uncommissioned workspace-geometry assumption.

Three learned adjacency message-passing layers (width 192) share local geometry
functions. Learned attention, weakly biased toward current boundaries near the
projected tool, pools a contact-neighborhood representation. There is no hard
coded contact side and no contact label. A separate global pooled/state path
retains absolute robot configuration and supports the demonstrated elevated
approach/withdrawal cases even if projected tool edge support is absent.
An eight-row masked GRU supplies response memory. Adjacency/readout are invariant
to consistent node-index permutation without needing alphabet identity.

The learned action head is a five-component diagonal Gaussian mixture over seven
normalized recorded commands. The candidate is the highest-weight component's
mean, not a replay and not a blended hand controller. The optional learned
auxiliary has a real gradient path: selected-centroid displacement three original
rows ahead, masked unless every intervening row is eligible in the SAME cut.
It is training-only. Contact-affinity supervision is deliberately not fabricated.

## `dual_view_memory`: mechanism preserved

Separate learned encoders consume marked global third RGB, a selected/tool-region
third crop, full-field wrist RGB and a separate full-chart goal foreground plus
masked RGB stream. Third global/detail share encoder weights within this model;
wrist and goal have their own encoders. No weights are shared with the graph
policy. Camera/robot state and crop-to-chart bounds are fused with image codes.
A causal 512-wide GRU over offsets [14,8,4,2,0] supplies temporal interaction
memory. It covers a fifteen-original-row span with five observations, rather
than decoding all fifteen high-resolution images per example. No previous
command input creates teacher-forcing ambiguity.

**Resolution/representation adaptation:** to keep a practical 24-GiB worker,
full-chart third is 256x144, its current detail crop is 128x128 from a cached
640x360 undistorted source, wrist is 224x128, and foreground-isolated full-chart
goal RGB/mask is 256x144. The original 640x360/384-detail proposal was not a
measured minimum. Wrist uses the whole field instead of claiming an accurately
localized tool crop. Goal appearance is kept in its separate full-chart stream,
not a separate high-resolution tight-crop encoder. Small edges/stickers/holes
can be lost; no claimed precision or visual robustness survives merely from
accepting those inputs. Original coordinates, masks and graph contours remain
in the original-size chart; preview inspection resolution did not set these
training choices.

This policy has its own mixture distribution and masked displacement auxiliary.
Unlike the graph, RGB can retain tool/material/occlusion context when the selected
component is fragmented but identifiable. This is a falsifiable inductive bias,
not an observed superiority or a license to act after losing the identity.

## Losses, augmentation and compute

Both optimize mixture negative log likelihood + 0.05 normalized-command Huber
loss + 0.2 masked centroid-displacement Huber loss (displacement scaled by 100
for numerical conditioning). No constant auxiliary is presented as learning.
Predictive dispersion is mixture variance in native command units, not calibrated
safety uncertainty. Prediction and reload checks are deterministic.

Graph extraction receives small coordinate noise consistently across its absolute,
center-relative and tool-relative channels. It is bounded extraction noise, not
physical augmentation. RGB receives consistent third/detail/goal photometry,
separate wrist exposure, and limited **past** view/frame dropout. No sustained
current blind-motion label is invented. There is no arbitrary image rotation,
shape warping, changed goal, changed instance, or changed action frame with the
old command label. Missing history is masked; the current frame is not dropped.

Approximate cache size is 5.5 GB (actual bytes reported), below the 64-GB limit.
It stores each representation once plus index maps, not every repeated history.
Updates only read bounded numeric cache slices. Models have several million
trainable parameters each (within the 1-64 million contract). Float32 batches
are 64 graph examples and 12 five-frame visual examples. No unbounded spatial
transformer or full-resolution per-update image decoding is used. No optimizer
execution occurs in API-authored code.

## Invocation, uncertainty and deployment boundary

`inventory` automatically produces scene-scoped arbitrary instance IDs and
observation templates. `act` associates the caller-selected current template,
then follows a causal ID even after moving away from its initial position.
Goal/instance/policy/call changes reset all motor memory. The explicit renderer
accepts local image-similarity goals, validates optional foreground agreement,
and implements commissioned image-to-plane homography rendering when supplied.
It refuses large uncalibrated perspective moves, infeasible image bounds and
protected-mask overlap. No required localization or renderer is deferred.

A separate HLA/external semantic recognizer still interprets words, upright glyph
meaning, repeated-character inventory allocation, ordering and feasible layouts.
No open-alphabet recognizer is claimed trained on these episodes. A word string
alone is rejected by the motor interface. This division permits new shape/word
requests without pretending that accepting them establishes manipulation ability.

Only currently observed selected masks support actions. Lost identity, invalid
views/state, inconsistent area, restricted-workspace violations, neighbors moving,
calibration changes, cancellation or timeout produce reasoned null-action statuses.
Goal agreement over 0.5 s produces `goal_geometry_observed`, not fabricated final
clearance or word success. Tool projection is a calibrated point with nearby-edge
support, not an automatically known silhouette width; metric clearance and contact
are always unknown here. This weak-point adaptation explicitly reduces the source
plan's trusted tool-geometry and successor-readiness claim. HLA must request a
verified stop and clearance/readability check before another operation.

`decode_action` requires recording/controller units, seven-joint order, timing,
limits, watchdog, stop, tool and workspace commissioning, and current safety
checks bound to the exact candidate. Violations refuse instead of silently
clipping. `enabled` means only that a dry-run command representation passed these
caller-attested gates. There is no actuation code. Offline learned prediction
works without inventing that verification.

## Outstanding scientific claims, not an extra study in this round

Neither the expected held-out-shape benefit of local geometry nor the expected
short-occlusion benefit of RGB memory has been tested. Two episodes cannot establish
friction/contact generality, safe tool-hole interaction, collision recovery or
arbitrary new shapes/materials/layouts. Required future independent tests remain:
physical shape/font/topology-held-out relocation; new words and repeated letters;
new initial arrangements/layouts; their joint combination; impossible/ambiguous
inventories; lighting/occlusion/tool/calibration variation; and graph/history/marker/
goal ablations with a replay baseline. Those are evidence requirements, not extra
policies or authorized preliminary runs. They have not been silently replaced
with row-random splits, training metrics or final-image word guesses.
