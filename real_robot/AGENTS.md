# Real robot workstream

Work on dev and preserve Exp1, Exp2, and the external original recordings.
Use `/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked`
for all Python/project commands. Reuse the locked environment without changing it.

Latest deployment authorization (2026-09-21): package the completed Push policies
for another Linux/NVIDIA GPU computer, with Git-visible exact API source, two
separate inference checkpoints, a portable local Python interface and SCP/rsync
download artifacts. The user will adapt the robot-side interface; do not implement
or operate a robot/network bridge. Preserve the training originals and frozen
scientific code. A separate minimal locked Pixi inference environment under
real_robot/deployment is authorized; do not modify the training environment.
Exporting the exact final EMA weights and required normalization is serialization,
not a new learned design. No new API call or retraining is needed.

Latest authorization (2026-09-20): the user explicitly requested proceeding
directly to API implementation and training of BOTH cut_v3 policies,
contour_push and visual_push, without a separate preliminary validation study.
The API owns policy, preprocessing, goal conversion and decoder code. The
developer owns the public interface, training executor, necessary operational
checks, GPU scheduling and accounting. Preserve cut_v3 and all original data.
Use GPT-6 Astra/xhigh at the already authorized official API endpoint. Train
independent weights and retain executable invocation packages/checkpoints.
Necessary syntax/interface/finite-gradient/checkpoint checks are part of execution;
do not insert a separate semantic/performance validation round. Physical robot
execution and Flip egg are not authorized by this instruction.

Previous authorization (2026-09-20): the user explicitly requested deleting
the previous cuts and calling the API again. Delete cut_v2 artifacts/session
with an exact inventory and retained cost ledger, then run cut_v3 using the
reviewed general prompt and separate Push task specification. This supersedes
the earlier pause. Preserve original data, Exp1 and Exp2. Stop before policy
or preprocessing implementation, training, and robot execution.

Current prompt documents: `prompts/general_cut_and_prior.md` and
`task_specifications/push_letters.md`. The general prompt is task-independent;
the task file supplies the specific objective and generalization requirements.
Explicitly retain multiple heuristics per regrouped dataset, independently
trained callable policies, and a High-level Agent selecting among a sufficiently
diverse, evidence-supported library. Segmentation, grouping, and policy sharing
are separate decisions; segment supervision must have coherent conditions.

Previously executed authorization: complete **Push only**, through API-authored
segmentation, freely regrouped datasets, and heuristic documents. Stop before
policy implementation, training, or physical robot execution. Flip egg is future
work. Do not launch it from this authorization.

Executed user amendment: delete the previous Push cut_v1 results and rerun from
the original data under cut_v2. Retain a deletion inventory and prior cost
accounting; do not feed the removed scientific designs to the new API session.
Deployment explicitly includes completely unseen letter identities/physical
shapes and entirely new words. Require deliberate model/prior/input selection,
an explicit model inventory, transfer rationale and agent calling contract.
Do not silently reduce this requirement to new combinations of known letters.

Developer owns data readers, interfaces, schemas, deterministic measurements,
validation, publication, API transport, and accounting. Runtime API owns the
scientific choices: cuts, grouping, reuse, goal representations, heuristic count
and content, augmentation, preprocessing designs, and handoffs. Later policy and
preprocessing implementation must also be API-authored; the latest instruction
above now authorizes those implementations and training for the two Push policies.
Never hand-edit submitted API designs.

Every new request explicitly uses `gpt-6-astra` and `reasoning.effort=xhigh`;
audit saved wire requests and returned model/effort. No silent fallback or
automatic transport retry. Preserve unknown outcomes and possible charges.

Use existing paired sample indices, not cross-device timestamp matching. Original
recordings have N paired observation/action records (not an invented N+1 record).
Cuts are half-open sample ranges [start, stop). Never concatenate disjoint cuts
into a continuous trajectory. Allow overlap, repeated skills, cross-trajectory
groups, and reuse across groups. Explain all exclusions. One or multiple
heuristics per group are allowed; there is no three-prior quota.

Push generalization goal: arrange completely unseen letter identities/physical
shapes into new requested words. Keep task-specific facts in the task file.
Earlier balanced organization examples in the frozen prompt are historical;
the new general prompt does not prescribe task-specific behavior or structure.
Leave architecture, decomposition and counts to the API. Goal annotations absent
from recordings must remain explicitly API-inferred or proposed future labels.
Cover useful demonstrated motions and report gaps without inventing demonstrations.

Augmentation, line detection, aimbot-like mechanisms, and other inductive biases
are allowed, not mandatory. Every proposed custom input must specify a complete
causal conversion from actually available observations. Its executable conversion
code, dependencies and weights are required in the later implementation stage.
Keep training-only future-derived labels distinct from deployment inputs.

API image evidence defaults to 640x360, aspect ratio preserved; original images
and crops are available on explicit tool requests. Preserve original images for
future training and record coordinate mappings. Reasoning remains xhigh.

All new code, configuration, derived data, reports, and journals live under
real_robot/. Root README/PROGRESS may receive navigation and progress updates.
