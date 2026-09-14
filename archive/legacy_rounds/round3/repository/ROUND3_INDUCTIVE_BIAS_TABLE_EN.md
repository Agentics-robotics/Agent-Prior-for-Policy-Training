**Implemented Inductive Biases and Their Integration into Diffusion Policy — Round 3**

Scope: the six completed Round 3 MetaWorld tasks. Candidate IDs are task-specific. Round 4 has no frozen Agent prior candidates yet.

All variants retain the action noise-prediction loss with weight 1. Each native action has four channels: xyz and gripper. The last column lists additional terms; “None” means the standard action objective remains, despite changes to inputs, architecture, or action coordinates.

| Task | Variant | Inductive bias | Integration into DP training | Additional objective beyond action diffusion |
| --- | --- | --- | --- | --- |
| All tasks | B0 | Standard Diffusion Policy | Raw-state conditioning; predict world-frame xyz actions and gripper commands. | Action diffusion loss only. |
| Pick-and-place over a wall | P1 | Factorized task geometry | Encode world, grasp, goal, and wall-clearance features in four branches; fuse them to condition one denoiser. | None. |
| Pick-and-place over a wall | P2 | Task-stage progression | Predict four stage probabilities from observations; their soft embedding conditions the shared denoiser. | + 0.20 stage cross-entropy. |
| Pick-and-place over a wall | P3 | Future motion and obstacle geometry | Use P1's encoder; jointly diffuse actions + future hand/object displacements (4 + 6 channels); supervise reconstructed clearance and hand–object separation. | + 0.25 future-noise MSE + ramped 0.05 geometry loss. |
| Nut assembly | P1 | Grasp and insertion relations | Add hand–handle, hand–ring, ring–goal and handle–ring relations, alignment errors, and motion features. | None. |
| Nut assembly | P2 | Transport-aligned coordinates | Express positional observations and action xyz in a ring-centered, goal-aligned horizontal frame; retain world z and absolute anchors. | None; action targets change coordinates. |
| Nut assembly | P3 | Task stages and future relations | Use P1 features with a shared encoder; predicted stage probabilities condition DP; a separate head predicts relations at +4 and +16 steps. | Ramped: + 0.10 stage cross-entropy + 0.05 future-relation Smooth L1. |
| Drawer opening | P1 | Cabinet-aligned coordinates | Use current-handle-anchored cabinet coordinates and relative motion; rotate action xyz to cabinet axes; retain world anchors. | None; observation and action representations change. |
| Drawer opening | P2 | Future rail progress | Use world relational inputs/actions; jointly diffuse actions with future remaining rail distance (4 + 1 channels). | + 0.25 future-noise MSE. |
| Drawer opening | P3 | Cabinet frame and future interaction geometry | Use P1 coordinates/actions; jointly diffuse future local hand–handle gap and remaining rail distance (4 + 4 channels). | + 0.25 future-noise MSE. |
| Door opening | P1 | Cabinet-relative contact geometry | Append cabinet-frame relations, motion and door direction to raw state; rotate action xyz into cabinet coordinates. | None; observation and action representations change. |
| Door opening | P2 | Future door interaction geometry | Use world-frame conditioning/actions; jointly diffuse future handle displacement, hand–handle gap and door direction (4 + 8 channels). | + 0.20 future-noise MSE. |
| Door opening | P3 | Cabinet frame and future geometry | Combine P1 conditioning/actions with P2's eight future channels expressed in cabinet coordinates (4 + 8 channels). | + 0.20 future-noise MSE. |
| Side peg insertion | P1 | Explicit grasp and insertion geometry | Append grasp–hand, goal–peg-head and other relations, motion features, and transverse alignment errors; retain world actions. | None. |
| Side peg insertion | P2 | Future peg motion and grasp relations | Use raw-state conditioning; jointly diffuse world actions, future peg-head displacement and grasp–hand gap (4 + 6 channels). | + 0.20 future-noise MSE. |
| Side peg insertion | P3 | Relations and future supervision | Combine P1's relation features with P2's joint action/future-relation diffusion (4 + 6 channels). | + 0.20 future-noise MSE. |
| Side peg insertion | P4 | Hand-relative coordinate revision | Retain P1 relations; replace four retained position blocks with hand-relative values. Absolute hand context remains; world actions are unchanged. | None; trained from scratch only at N = 5 and 20. |
| Stick pushing | P1 | Hand–tool–object–goal relations | Add pairwise hand/stick/container/goal displacements and hand/stick/container motion features; retain world actions. | None. |
| Stick pushing | P2 | Goal-aligned tool-use coordinates | Express positional features and action xyz in a container-centered, goal-aligned horizontal frame; retain world z and frame context. | None; action targets change coordinates. |
| Stick pushing | P3 | Coupled hand–tool–object motion | Use P1 relations; jointly diffuse actions and future hand, stick and container displacements (4 + 9 channels). | + 0.15 future-noise MSE. |

**Loss and implementation notes.** Future-noise MSE is masked epsilon-prediction MSE over auxiliary channels in the same temporal U-Net. These channels are jointly sampled at inference and discarded before action execution. Assembly P3 instead uses a separate future-relation regression head. Stage labels come from demonstrated-state geometry proxies; inference uses predicted soft probabilities. Pick-and-place P3 ramps its geometry weight over 1,000 updates and applies it only when the cumulative diffusion signal coefficient is at least 0.5; its geometry loss matches demonstrated wall clearance and hand–object separation. Assembly P3 ramps both auxiliary weights over 2,000 updates.

**Representation notes.** Cabinet/goal frames are held fixed over the observation history and predicted action chunk; action xyz is transformed back before execution. These designs retain world context and do not imply exact invariance of the entire policy. Candidate preprocessing uses fixed feature scales/identity normalization, while B0 uses training-subset statistics. Drawer P3 adds a future hand–handle gap beyond P2, so it is more than a literal P1 + P2 combination.

**Training scope.** Round 3 used 20,000 updates per model and one training seed. Initial candidates covered N = 2, 5, 10, 20 demonstrations; peg P4 covered N = 5, 20 only. This table describes implemented mechanisms, without attributing observed gains to individual components.

Sources: [shared training loss](src/round3/learning.py), [pick-and-place](src/round3/pickwall_plugin.py), [assembly](src/round3/assembly_plugin.py), [drawer](src/round3/drawer_plugin.py), [door](src/round3/door_plugin.py), [peg](src/round3/peg_plugin.py), [peg revision](src/round3/peg_revision.py), [stick](src/round3/stick_plugin.py).

