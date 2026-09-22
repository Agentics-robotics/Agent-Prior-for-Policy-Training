# Real robot workstream

Push v6 completion reviewed 2026-09-22: preserve package_00/source hash
6483209318f7f2eaf3b81b39cbf7c3dfc4890d69ed8917e771477841e0a0a80b and checkpoint
80584e08bd99580bc7b438bf3a2db517386f45a86bf5a27d507ff4bf0f940313.
One shape_push_v1 contact/heading/stroke scorer, 18,497 trainable parameters plus
frozen SAM, completed 3,000 updates on GPU4 with final model selected. All130
retained examples (A43/B87;9/12 learning cuts;115 full/15 contact-only) entered
gradients, verified independently against saved sampler RNG and47,986 exposures.
Training pseudo-label contact disagreement16.403mm is not physical endpoint error
or evidence of precise imitation; no generalization/robot evaluation. Reload and
9 API calling cases plus independent worker checks passed.92 Astra/xhigh calls,
estimated$19.6388715, no unknown usage. No further training or performance repair
is pending under this request; retain the executed model and diagnostics.
Independent deployment_pipeline_v2 exports exact source and selected tensors into
policies/push_v6 and checkpoints/push_v6; required frozen SAM asset remains separate.
Actual pipeline_v2_bundle archive passed47-file relocation and real-weight CLI
example checks with zero output difference. Old deployment_pipeline/v5/Flip
artifacts remain unchanged. No Git push, Drive upload or physical execution.
See reports/PUSH_TRAINING_V6.md, reports/PUSH_V6_DEPLOYMENT.md and
reports/PUSH_V6_SYSTEM_WALKTHROUGH.md. The authorization below describes this
completed run; do not launch it again merely by reading these instructions.

Latest user amendment (2026-09-22): cancel and explicitly delete the extra
correction round based on completed Push v5 feedback. Its exclusive code, prompt,
API journals and runtime artifacts have been removed; only deletion/cost receipts
remain in deletions/push_v5_feedback_continuation_20260922 (25 calls, $4.699118).
Run the original Push v5 workflow afresh, calling this new run Push v6. Use
configs/push_training_v6.json and implement_and_train_final_fit.md: original general
prompt plus a task-independent instruction to use as much applicable valid data
as possible and deliver the final fitted model. Do not give this API completed
v5 results/source, the split review, or deleted-round corrections. Preserve original
v5 and Flip v1 and their cuts/prompts/exports. API still owns all scientific choices.
No new cut or Flip retraining. Astra/xhigh, GPU 4 with occupancy checks, the existing
locked environment and standing official OpenAI permission remain applicable.

Portability follow-up completed (2026-09-22): user wants to transfer the two
trained models after Git synchronization, using Drive for downloads. Independent
deployment_pipeline exports exact API source and selected tensors into
policies/{push_v5,flip_egg_v1} and checkpoints/{push_v5,flip_egg_v1}; exports/pipeline_v1
contains two-weight, optional-SAM and complete archives. The actual relocated
archive passed both model calls with zero difference from original worker outputs,
and actual SAM inventory produced unvalidated proposals. No scientific source,
training checkpoint or locked environment changed. No Git push, Drive upload,
new API or physical robot execution. Preserve exported identities and use a new
explicit export version for changes. See deployment_pipeline/README.md and
reports/PIPELINE_DEPLOYMENT.md; integration of live observations and the user's
planner/controller remains at the receiving host.

Both requested training runs completed and verified (2026-09-22). Preserve
Push training_v5 package_01 (800 updates, selected600, 9,641 parameters,
74 A training /94 B development examples; prepared12 groups, gradient6 groups), its
blocked package_00 and all recovery checks. Learned output is now planar EE
continuation; first-contact selection is an API-authored deterministic prior.
Preserve Flip egg training_v1 package_00 (2,000 updates, selected2000,
278,903 parameters, 5,364 train /2,781 validation /2,786 held-out examples).
Its implementation uses joint RGB encoding and measured EE-local motion, replacing
the originally proposed blade/semantic geometry module. Both reload and independent
online-worker checks passed; no physical success/generalization claim. Push's
development endpoint error5.095mm does not beat the goal-translation baseline4.764mm.
Per-task API_USAGE.json and reports/*_API_USAGE.md record all calls/tokens/costs:
Push100 calls/$16.283451 including blocked/recovery work; Flip71/$13.4936575.
All Astra/xhigh, no unresolved usage. New cut prompt remains saved only. No further
training/recovery is pending under this request; future experiments need a new run.
See reports/PUSH_TRAINING_V5.md and reports/FLIP_EGG_TRAINING_V1.md.

Latest user amendment (2026-09-22): preserve the executed Push cut_v6 and Flip
egg cut_v1 prompts/configurations/artifacts. New general_cut_and_prior_v6.md and
new task-specific v3/v2 documents are saved only; do not run a new cut with them.
The user explicitly authorizes formal Runtime API implementation, preprocessing
and training of BOTH existing latest cut designs, via the same task-independent
training_pipeline. Use configs/push_training_v5.json and
configs/flip_egg_training_v1.json, Astra/xhigh, the existing independent locked
training_v4_environment. Use physical GPU 4 with per-job occupancy inspection and
the pipeline's shared device lock; preserve unrelated processes.

The generic workflow fixes execution, feedback, validation and provenance; API
owns scientific choices, including data sampling/derived windows, representations,
policy inventory, optimization and training settings. Previous cut/prior decisions
are the starting design and may receive evidence-based implementation refinements
in separately retained derived plans. In particular, the earlier 47-anchor
training whitelist is historical and must NOT constrain new training. Preserve
all completed v4 source/results exactly. Task-specific documents may supply facts;
do not hardcode a scientific solution into the shared workflow. Bounded API repair
on actual implementation errors is authorized; automatic transport replay is not.
No robot execution. Standing official OpenAI data permission continues to apply.

See training_pipeline/README.md, reports/GENERIC_PROMPT_AND_TRAINING_REVISION.md,
and prompts/archive/20260922_before_representation_revision/manifest.json.
The older instructions below describe their executed rounds, not new constraints.

Work on dev and preserve Exp1, Exp2, and the external original recordings.
Use `/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked`
for all Python/project commands. Reuse the locked environment without changing it.

Latest Push implementation/training authorization (2026-09-21): user explicitly
requests API model writing, necessary data processing and training from cut_v6.
Use configs/push_training_v4.json and independent policy_training_v4. The
training_v4_environment Pixi manifest/lock is a separate authorized dependency
environment for the proposed SAM 2/Qwen libraries; older environment locks stay
unchanged. All new v4 project commands use that locked Pixi environment. One
shape_push_v1 learned geometric decision policy is assigned, not a six-dimensional
EE imitation output. Preserve 47 sparse anchors and executor-only evidence;
API owns scientific preprocessing, inferred annotations, targets, architecture,
loss, fixed update budget (at most 20,000) and calling/adapter logic. Zero-update
data/model interface checks and derived evidence inspection are authorized
implementation checks; retain source revisions and report all invalid targets.
GPU 4 was inspected idle; recheck before jobs. Official Astra/xhigh and standing
data authorization apply. No physical robot execution or new cuts. Preserve
concurrent Flip egg work, earlier Push cuts/models and all original recordings.

training_v4 completion reviewed 2026-09-21: 77 actual Astra/xhigh calls submitted
18 API-authored files; estimated USD 16.095949. The shared contact/direction/stroke
scorer has 15,041 parameters and completed the API-selected fixed 500 updates on
GPU 4. Final EMA reload error is zero and an independent local inventory/act
worker check passed. Actual preprocessing retains only 4 of 47 sparse anchors,
covering 4 of 12 learning intervals; no broad behavior/generalization claim.
Six zero-update checks include two retained API integration failures/repairs.
The API implemented chromatic geometry instead of acquiring SAM/Qwen weights.
Earlier cuts and training_v3 source/checkpoints retain their verified hashes.
See reports/PUSH_TRAINING_V4.md, policy_training_v4/README.md and
runs/push_letters/training_v4/completion_receipt.json. This requested stage is
complete; no further training or repair is pending. Preserve all artifacts.
No robot execution, performance search or independent deployment export occurred.

cut_v6 completion reviewed 2026-09-21: the active with-example attempt completed
24 Astra/xhigh calls, selecting one shape_push_v1 learned contact/direction/short
stroke policy with one prior, plus a supplied geometric executor group. Twelve
learned context cuts contain 47 sparse decision anchors; two full executor traces
retain all 16,745 raw rows. All 51 published files and raw slices passed verification.
These are proposed interfaces and labels, not implemented/trained models or
integrated planner bindings. Two frozen perception dependencies are proposed,
not available verified assets. Preserve the design, old cuts and training_v3.
See reports/PUSH_CUT_V6_REVIEW.md and the with-example completion receipt.
No cut/training remains pending under this authorization; do not restart it.

Independent Flip egg authorization (2026-09-21): the user explicitly requests
API cut/prior design for Flip egg in a separate task while Push cut debugging
continues. This supersedes the historical "Flip egg future work" restriction
only for this design stage. Use all 20 original episodes, Astra/xhigh,
configs/flip_egg_cut_v1.json and the isolated flip_egg_cut_v1 interface. The
general v5 prompt is copied exactly; task_specifications/flip_egg_v1.md supplies
the task and explicitly labels inherited planner capability as a design premise.
Audit Flip egg's actual finite gripper feedback instead of inheriting Push's
missing-feedback description. Runs and cuts live under runs/flip_egg/cut_v1
and data/flip_egg/cut_v1. Standing official OpenAI data-processing authorization
applies. API owns cuts, priors and model IO. No implementation, training or
robot execution. Preserve concurrent Push files and all prior artifacts.

Flip egg cut_v1 completion reviewed 2026-09-21: two data groups, one callable
learned policy egg_interaction_v1, one auxiliary scene_geometry_aux proposal,
40 segments and all 45,730 source rows retained. There are 32,737 declared
dense control anchors and 12,570 overlapping executor-prefix rows. All 79
requests/responses used Astra/xhigh; 129 published files and exact raw slices
passed validation, estimated USD 12.4355135. The API chose six blade-relative
short-motion outputs plus uncertainty, with known-tool acquisition allocated
to a fixed recipe/executor. No implementation, annotation, training or robot
execution occurred or remains pending under this request. Preserve the API
plan and journals. See reports/FLIP_EGG_CUT_V1_REVIEW.md and
runs/flip_egg/cut_v1/completion_receipt.json. Current-conversation payload and
destination approval was obtained after two automatic review rejections;
retain both the approval and original rejection receipts.

Latest cut_v6 prompt steering (2026-09-21): the user explicitly requests a brief,
incidental contact-point example. general_cut_and_prior_v5.md adds only
"Spatial goals (e.g., a contact point)"; policy count/output schema remain API
choices. Use configs/push_cut_v6_with_example.json with execution/cut_v6.py;
the active session is runs/push_letters/cut_v6_with_example and publication
remains data/push_letters/cut_v6. The initial no-example session was explicitly
interrupted, with nine completed calls and one unresolved in-flight call;
retain runs/push_letters/cut_v6/interruption_receipt.json and original journals.
No automatic transport retry or reuse of its scientific design output. The
with-example attempt is complete; these paths document the executed protocol.

Initial cut authorization (2026-09-21): the user requests cut_v6 using a general
prompt plus Push-specific task document, planner-supported learning scope and
no encouragement/quota for multiple heuristics. Runtime API independently
chooses the remaining learning problems, outputs and policy count. Do not feed
the user's desired contact-point example or prescribe one policy. Use
configs/push_cut_v6.json, execution/cut_v6.py and the independent cutting_v6
interface, with general_cut_and_prior_v4.md and push_letters_v2.md. The new
interface supports learned and supplied-executor groups, dense/sparse decision
supervision and executor-only evidence. Planner availability is user-declared;
actual deployment bindings/calibration are not verified by this design stage.
Use gpt-6-astra/xhigh and the standing official OpenAI data authorization.
Preserve all old runs, designs, policies and training; stop after cut/prior and
calling/preprocessing/adapter documents. No training or robot execution.

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
