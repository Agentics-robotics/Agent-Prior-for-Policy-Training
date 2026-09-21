# piece_contact_graph_v1 — prior

Responsibility: local translation/rotation of one persistent planar physical instance, including approach, contact switching, resets, fine corrections and requested exit. Dataset `piece_relocation`; exact heuristic `boundary_mechanics_residual`. See [usage](piece_contact_graph_v1_USAGE.md), [shared science](PRIOR.md), [calling](CALLING.md), [handoff](HANDOFF.json).

## Frozen training support

A=`episode_2026091922002701`, B=`episode_2026091922062101`. This policy independently receives **all** these supervised intervals, unchanged:

A: wedge [600,1740); barred clearance [1740,1990); ring [1990,3220); R staging [3220,3700); barred staging [3700,4300); upper barred [4300,4640); D staging [4640,5200); R layout [5200,5670); D alignment [5670,6120); L repeated relocation [6120,8150); barred alignment [8150,8510); upper-bar alignment/retract/hold [8510,9015).

B: wedge [380,1240); barred clearance [1240,1500); D staging [1500,1950); ring [1950,2850); R repeated movement/reset [2850,4700); D alignment [4700,4950); L repeated relocation [4950,6150); upper barred [6150,6910); barred alignment [6910,7370); upper-bar alignment/retract/hold [7370,7730).

The original 20-row cut context buffers remain, bounded by recording ends. Morphology names identify inspected instances, not verified characters. D inside-hole/recess pushing, L reversals and repeated recontact, B R lift/reconfiguration, nonzero angular commands and terminal actual zeros are retained. No stroke is relabeled as verified success or removed for nonmonotonic progress. Per-segment preprocessing coverage is emitted by prepare_data; unreliable geometry is masked separately from finite native command loss.

## Model and reusable structure

A 32-node contour graph includes outer **and inner** boundaries, inward signed-distance normals, metric lever arms, current-to-goal rigid flow, tool offset, obstacle clearance, hole flag, scale/confidence/goal validity/symmetry. Four nearest spatial neighbors drive message passing. The nodes are not glyph templates or a convex hull. A compact CNN and robot/error encoder feed a masked 192-dimensional recurrent context over offsets [11,7,3,0].

A geometric score proposes feasible proxy boundaries. Learned node scores and one-step weak response/uncertainty adjust selection. Far candidates propose approach-to-boundary directions; near ones use positive-normal plus bounded tangential components. A soft contact mixture and learned speed/phase gate supply the anchor for bounded native linear/z/angular residuals. Explicit exit/lift anchors override continued contact. This is a constrained one-step receding-horizon proposal, **not a calibrated quasi-static contact simulator, collision-certified candidate set, or full long-horizon optimizer**. The learned response is in proxy motion units; mass, friction and calibrated forces are unavailable.

Conditional diffusion over contacts/strokes was considered. Two episodes and 22 hindsight calls weakly constrain contact multimodality. Explicit boundary alternatives plus low-dimensional response/score/residual learning put less burden on these data and expose useful diagnostics. No horizon limit from the external predict interface dictated this choice.

## Goals and effective losses

An endpoint footprint at the last supervised row is a training-only achieved local goal, subject to confidence/rigid-fit thresholds; the endpoint TCP/twist explains approach/clearance/exit. Online HLA supplies fixed W footprint or SE(2) displacement and optional TCP exit. No future frame or semantic word enters policy input. Symmetry ambiguity is preserved; semantic upright requires an independent causal cue.

Losses: Huber and small heteroscedastic native-action cloning, five-mode CE, small residual norm, confidence-masked nearest-boundary selection CE and actual in-segment successor displacement/angle response loss. The response head also influences candidate scoring, giving a learned planning path rather than a constant analytic penalty. No measured dq is a command label. Native angular channels remain supervised in original numerical serialization, not incorrectly rotated as if their physical frame were verified. Per-policy saved scales and full interaction basis make candidate reconstruction executable.

Independent approximately two-million-parameter model, batch 32, AdamW 2e-4/1e-4 weight decay, gradient norm 1, fixed 20k/EMA schedule. No weight sharing with field/waypoint. No speculative physical augmentation or performance search.

## Dependencies, adaptation and limitations

Uses executable contrast segmentation/causal tracking, full camera/flange transforms and board-plane proxy. No external tensor asset is needed. Without audited masks/plane/rod, confidence is heuristic and local boundary mechanics can be wrong. Fresh external shape-independent masks and measured top-plane frame are supported, not fabricated. Missing endpoint goals train masked-goal command/transition imitation; missing response tracks suppress only auxiliary response loss. Every such degradation is recorded per segment.

Prefer this representation when local contours/holes and contact geometry are credible, not because it has an invented higher success rate. Concavity accessibility is only a proxy clearance screen; robot reachability, tip sweep, height and friction remain unverified. New shape support comes from measured contours rather than an alphabet embedding, but no held-out shape success, open-world recognition or robust jam recovery has been established. Semantic inventory/layout belongs upstream. Physical decoder and readiness remain disabled; see shared documents for audited intent-mapping versus actual hardware distinction.
