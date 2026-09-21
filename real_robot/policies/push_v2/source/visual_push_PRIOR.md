# visual_push prior — dual_view_memory

## Responsibility and independently selected family

Move ONE caller-selected physical piece to ONE fixed explicit image goal, including the demonstrated approach, transport, re-contact, posture/alignment corrections, staging and withdrawal. It is an alternative to contour_push, not its successor stage. It never allocates letters, selects a next piece, chooses a word/layout, or retrieves a recorded phase. Frozen heuristic: **dual_view_memory**.

The chosen learned family is an independent **five-component diagonal-Gaussian current-command mixture** conditioned on dual-view visual history and a separate goal stream. Diffusion Policy was considered independently and not chosen: this contract needs one present command, not long open-loop action sequences, and two episodes supply limited diverse conditional evidence. A small explicit multimodal head with gated visual history is practical and matches that causal contract. This is a design justification, not evidence that diffusion or another policy would be worse. The model is trained from scratch with no contour model weights, shared visual weights, replay, scripted pushing or generic frozen encoder.

## Implemented inductive bias

Four separately parameterized image encoders process:

1. calibrated third RGB plus selected-instance foreground marker at 320x180;
2. a selected-instance/tool-neighborhood RGB/marker crop at 160x160, with its crop-to-original transform cached for audit (not a metric transform);
3. calibrated wrist RGB at 256x144, retaining local rod/edge/hole appearance in its own view;
4. fixed goal foreground plus masked target RGB in the full-chart 160x90 goal image, separate from live observations.

Convolution/GroupNorm/SiLU blocks and spatial grid pooling retain coarse layout. Encodings are fused with 142 measured-state/missingness channels and two masked GRUCell layers of width 384 over eight causal observations at three-row spacing. The target marker identifies the chosen current instance; the wrist stream and temporal response can retain cues discarded by the contour bottleneck. The mixture head learns a seven-command distribution; highest-probability component mean is the candidate, and total mixture spread is a diagnostic, not calibrated certainty. A separate response head is trained with valid within-segment future foreground displacement.

This is substantive dual-view temporal learning, not the graph encoder under another name. No previous demonstrated action is an input, so deployment never needs unavailable teleoperator actions. There are no phase labels, future images, alphabet embeddings or learned done/success bit. Stopping is computed from causal shared geometry monitoring and caller tolerances.

## Shared executable conversion

Original paired RGB is remapped with each camera's own recorded Brown-Conrady calibration; the third original undistorted 1280x720 chart is the goal coordinate system. The automatically segmented selected marker comes from a broad brown-material/light-support detector and forward-only area/color/motion association. Templates and IDs are observation-derived and arbitrary. A fully missing or ambiguous selected identity returns no candidate; recurrence is not permission for blind motion. Online candidates require selected confidence >=.3, allowing more partial observed area than contour_push's .65 gate, but not imputing a fictional full mask. Both cameras and finite required robot states must be present and fresh.

Hindsight goal boxes and anchors select training identity/placement only, after the forward causal tracking pass. They do not supply current pose, current crop center, hidden-state warmup or an online objective. Shared cache preserves original cut boundaries, sample pairing and exclusions. Online goals are rendered from the current invocation template at explicit center/relative rotation, not read from episode endings. Wrist pose derives from flange and calibrated camera transform. Projected TCP is a point with validity, not measured contact/clearance; no depth or commissioned tabletop plane is assumed.

## Losses and practical recipe

Action target = finite original **command** `action_json.dq[7]`; measured `dq` is a feature. Exact native commands are cached in float64 and standardized inside the model using saved buffers. Mixture NLL is supplemented by .15 * smooth-L1 next-observed selected-center displacement (three rows later, normalized image displacement times 100). Both endpoints must pass supervision and perception masks within the same original segment; the auxiliary cannot supply future current features.

Photometric gain [.85,1.15] and small bias are consistent across current/goal RGB streams, with an independent wrist gain to reflect exposure differences. Markers are not geometrically altered. Causal nonlatest-history dropout has probability .08, always retaining the latest observation. No rotation with unchanged robot commands, arbitrary goal relabeling, synthetic contact supervision or fabricated new-shape motion is used. Batch 12 is bounded for a 24-GiB GPU, lr .0002, AdamW weight decay .0001, gradient clip 1.0. The outer fixed 20,000-update seed-0 schedule and EMA apply; no preliminary performance study or checkpoint ranking occurs.

## Explicit adaptations and assumptions

The source's unavailable learned generic segmentation/flow/recognition assets are not claimed installed. Deterministic material-limited foreground/tracking is executable without human per-move labels or alphabet lookup. This reduces generality: novel brown silhouettes can be proposed, but new colors/materials/clutter are not robustly covered. Character/word interpretation and semantic upright direction remain external HLA capabilities; no two-episode open-alphabet learning claim is made.

The source's proposed 640x360 global and up-to-384 detail RGB are reduced to 320x180 global, 256x144 wrist and 160x160 detail; history is eight rather than up to fifteen observations, spaced over about .7 s at original cadence. The goal is a masked foreground/RGB full-chart image rather than an additional high-resolution tight goal crop. This bounds cache and training cost but may lose fine target/tool texture. All coordinates/crop mappings remain tied to original pixels; changing resolution does not make pixels metric. No generic pretrained weights or undisclosed assets are used.

Similarity-only goal rendering refuses clipping, unsupported large translations, unacknowledged approximation and invalid chart/workspace input. Metric/projective layouts need a commissioned renderer, not fabricated `T_base_board` geometry. Goal_observed is explicitly geometric agreement only, with unknown clearance and successor_ready=false. Candidate decoding requires externally verified controller semantics; candidate inference remains executable without that hardware contract.

## Demonstration support and limitations

All 22 frozen segments are assigned, sharing exactly the same eligible dataset as contour_push. W blocks demonstrate initial descent/approach; transport and repeated contacts occur in A/B W,O,L; A4800 and B1800 expose inner-hole appearance. B-R includes 3900/4050/4200 contact/posture changes that motivate temporal interpretation and are retained under one R goal. The A/B staging/revisit blocks have distinct local hindsight goals rather than one eventual word slot. Coverage reports disclose automatic missing-command/perception filtering, not unearned motion coverage.

Proposed benefit under partial boundary fragmentation with still-valid identity is untested. Dual-view input does not prove automatic cross-view correspondence or visibility consistency: that diagnostic is null. Appearance can memorize the brown pieces, camera, stickers, light background, familiar arrangements or visible word-like scene context. Recurrence cannot certify contact, recover arbitrary occlusion, or establish safe travel. Held-out physical-shape/material/layout tests and goal/marker/history ablations are required, not random row splits or two seeds on the same evidence. There is no ranking against contour_push, no success/readability label, no recovery from falls/flips/stacks, no arbitrary homing/free-space planner and no actuation.

Read [PRIOR.md](PRIOR.md), [visual_push_USAGE.md](visual_push_USAGE.md) and `HANDOFF.json.policies.visual_push` for shared assumptions, exact calls and HLA responsibilities.
