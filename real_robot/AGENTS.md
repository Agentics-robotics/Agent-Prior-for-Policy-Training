# Real robot workstream

Work on dev and preserve Exp1, Exp2, and the external original recordings.
Use `/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked`
for all Python/project commands. Reuse the locked environment without changing it.

Latest portability follow-up (2026-09-21): the user asks how to run the current
policies on another computer after git pull. An independent deployment_v3 package
now provides exact API source under policies/push_v3, lossless final EMA exports
under checkpoints/push_v3, and full/weights-only archives under exports/push_v3.
The existing deployment/pixi.toml and lock are reused unchanged for inference.
The actual archive passed a relocated interface check in that environment; all
three candidate outputs match the original worker within zero observed error.
See deployment_v3/README.md and reports/PUSH_V3_DEPLOYMENT.md. Files remain local,
not committed or pushed by this work. Preserve the archives, originals and old
versions. No new API, training or robot IO was part of this portability work;
robot-side communication/controller mapping remains the user's integration.

Latest training authorization (2026-09-21): the user requests API-owned policy
implementation and training for the three completed cut_v5 priors, encouraging
Diffusion Policy and allowing justified alternatives. Run the separate
configs/push_training_v3.json with policy_training_v3 and
prompts/implement_policies_v3.md, Astra/xhigh. Preserve all prior artifacts.
The developer interface retains original v/w labels and samples each policy's
assigned group; all scientific conversions and models remain API-owned.
Train independently for 20,000 updates per policy with seed 0 and final EMA.
Physical GPUs 4,5,7 were inspected free; recheck occupancy before scheduling.
The user's standing official OpenAI data authorization applies. No new cuts,
performance search or robot execution. Retain implementation failures and use
explicit API-authored repair revisions if needed; no automatic transport retry.

training_v3 completion reviewed 2026-09-21: all three models completed 20,000
independent updates and exact final EMA reload checks. The local inventory/act
interfaces return finite candidates and reject stale inputs/nonrigid goals,
with interruption supported. All 26 segments retain action supervision;
one 600-row piece segment lacks a credible inferred object endpoint. Decoder
remains disabled, with no robot execution or generalization evaluation.
Source, caches, independent checkpoints and API journals stay under
runs/push_letters/training_v3; see reports/PUSH_TRAINING_V3.md and its completion
receipt. No training or repair remains pending; preserve this completed run.

Persistent data-processing authorization (2026-09-21): after the explicit cut_v5
payload/destination question, the user replied “允许，为什么最近总问我，之前也不问我啊？之后都允许”.
This authorizes cut_v5 and future sends of relevant demonstration images,
robot states and metadata to the official https://api.openai.com/v1/responses
and its /input_tokens endpoint for the same user-requested real_robot API
cut/prior design workflow. Do not ask for per-run confirmation again within
that scope. Retain the exact authorization and original approval refusal:
authorizations/openai_data_processing_20260921.json and
runs/push_letters/cut_v5/destination_authorization.json. This is data-processing
permission; experiment execution still follows the user's requested task.

Latest cut authorization (2026-09-21): the user approved the reviewed
small-demonstration prompt addition and requested another trial with the next
cut number: cut_v5. Use configs/push_cut_v5.json, the independent
execution/cut_v5.py entry, and prompts/general_cut_and_prior_v3.md. The added
objective guides subtask granularity through reusable object/interaction-relative
structure, constrained quantities and minimal learned action variables; retain
full behavior coverage and compatible buffers. The exact scientific decomposition
and action dimension remain API choices. Keep the existing evidence tools and
task specification, so this run tests the prompt change. Use the same original
Push recordings and official OpenAI Responses destination established in this
conversation, with gpt-6-astra/xhigh. Preserve all earlier versions. Deliver cuts,
priors, preprocessing obligations and calling documents; no implementation,
training or robot execution in this round.

cut_v5 completion reviewed 2026-09-21: 2 dataset groups (tool_position and
piece_relocation), 3 policy/prior designs, 26 segments. All 29 Astra/xhigh calls
and 85 published files passed verification; estimated USD 5.48676. The new
design makes low-dimensional, object/interaction-relative residuals explicit
but retains approach/reset/adjustment inside piece calls, merging the previous
alignment group. It does not publish a standalone planar-push skill. No policy
implementation or training occurred or remains pending under this instruction.
Preserve the completed design and previous versions; do not restart the run.
See reports/PUSH_CUT_V5_REVIEW.md and runs/push_letters/cut_v5/completion_receipt.json.

Completed cut execution instruction (2026-09-21): the user requested starting
the reviewed subtask-first design to see its outputs. The independent run used
`configs/push_cut_v4.json` through `execution/cut_v4.py`, preserving all previous
cuts/training/deployment artifacts. Its scope ended at cut datasets, heuristic
designs, preprocessing obligations and policy calling documents. Local preflight is
complete. The user explicitly authorized sending this run's selected original
Push images, robot states and metadata to https://api.openai.com/v1/responses,
resolving the initial automatic approval block. cut_v4 completed with 3 subtask
datasets, 4 independent policy/prior designs and 28 segments; all 27 requests
and responses used Astra/xhigh, estimated USD 5.3269. All 93 published files
passed verification. Preserve the submitted design, original cuts and journals.
No implementation or training of cut_v4 policies has occurred or is pending
under this instruction. See `reports/PUSH_CUT_V4_REVIEW.md` for the Chinese
review, `reports/PUSH_CUT_V4.md` for exact API text, and
`reports/PUSH_CUT_V4_STATUS.md` for execution provenance. Do not restart this
completed run merely by following these historical execution instructions.

Latest cut-design amendment (2026-09-21):
`prompts/general_cut_and_prior_v2.md` was used for cut_v4. First identify
subtasks whose callable policies collectively cover the demonstrated behaviors
needed for the long-horizon task. Assemble a sub-dataset for each distinct
subtask through freely selected, irregular, overlapping and reused intervals;
then design one or multiple independently trained heuristic-based policies per
group. Retain useful preceding/following buffers with supervision compatible
with the subtask and inductive bias, and explicit context-only roles otherwise.
Preserve full real-robot preprocessing and privileged-input conversion accounts,
including training-only versus deployment information and executable asset
requirements. State positive design objectives without prescribing task-specific
skills or phase sequences. See `reports/CUT_PROMPT_V2.md`. Keep the executed original prompt,
cut_v3 configuration and frozen artifacts unchanged; future execution needs its
own configuration and output paths.

Latest training authorization (2026-09-21): execute an additional independent
Push `training_v2` using `configs/push_training_v2.json`. Preserve cut_v3,
train_v1 and the existing push_v1 deployment. Reuse the unchanged numerical
interface, executor recipe and source cuts. The new English implementation
prompt explicitly allows Diffusion Policy or a different learned family for
each subpolicy, at the API's discretion. Also require API-authored per-policy
prior/usage documents, a catalog and explicit handoff information for deployment
agents. Do not hand-author scientific choices or feed previous performance
diagnoses as design feedback. The same authorized official API data-processing
purpose applies; use gpt-6-astra/xhigh. GPUs 2/3 were inspected free for this run;
preserve unrelated processes and recheck occupancy at scheduling. No robot IO.

Completion reviewed 2026-09-21: training_v2's two models each completed 20,000
updates and exact EMA reload checks. The final push_v2 archive passed a relocated
interface check; all per-policy prior/usage/handoff files are exact API output.
Both families are Gaussian mixtures by API choice. Actual coverage is 6,314
examples from 12/22 cuts. Source is under policies/push_v2, weights under
checkpoints/push_v2, and the separate local host is deployment_v2. Existing
deployment/pixi.toml supplies its locked inference environment. No run remains
pending; this historical authorization does not request a repeat. See
reports/PUSH_TRAINING_V2.md. Preserve both completed versions.

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

Executed cut_v3 prompt documents: `prompts/general_cut_and_prior.md` and
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
