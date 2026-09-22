# Real robot demonstrations

**Push v6 training and portable package completed (2026-09-22):** one learned
contact/direction/short-stroke scorer with 18,497 trainable parameters and frozen
SAM, 3,000 updates, all 130 retained examples from both recordings used in gradients.
Nine of 12 learning cuts contribute; 15 rows have contact-only supervision.
Training-label contact disagreement is 16.403 mm, not robot positioning error or
evidence of precise imitation. Reload and actual relocated calls match exactly;
no physical/generalization evaluation. See the [training report](reports/PUSH_TRAINING_V6.md),
[complete Chinese system explanation](reports/PUSH_V6_SYSTEM_WALKTHROUGH.md),
[API usage](reports/PUSH_TRAINING_V6_API_USAGE.md), and
[preprocessing cost comparison](reports/PUSH_TRAINING_V6_PREPROCESSING_COST.md).
The [complete package](exports/pipeline_v2/pipeline_v2_bundle.tar.gz) includes weights,
required SAM asset, API calling documents and a runnable real-weight example;
see [installation](deployment_pipeline_v2/README.md) and
[relocation evidence](reports/PUSH_V6_DEPLOYMENT.md). Old v5/Flip artifacts remain
unchanged. No Git push, Drive upload or robot IO was performed.

**Portable Push v5 / Flip egg v1 package (2026-09-22):** after committing/pulling
the new source, download the [two-weight archive](exports/pipeline_v1/pipeline_v1_weights.tar.gz).
Push's optional SAM inventory additionally uses the [SAM archive](exports/pipeline_v1/pipeline_v1_sam_assets.tar.gz).
See [installation and Agent contracts](deployment_pipeline/README.md) and
[relocated verification](reports/PIPELINE_DEPLOYMENT.md). Both model outputs match
their original workers exactly in recorded interface fixtures. No robot IO,
Git push or Drive upload was performed.

**2026-09-22: generic implementation/training workflow.** See
[pipeline and commands](training_pipeline/README.md),
[prompt/ownership correction](reports/GENERIC_PROMPT_AND_TRAINING_REVISION.md),
and [preserved executed prompts](prompts/archive/20260922_before_representation_revision/manifest.json).
The completed training runs use existing Push cut_v6 and Flip egg cut_v1;
the separately saved new cut prompt has not been run.

**Flip egg training_v1 completed (2026-09-22):** 278,903 parameters, 2,000 updates,
10,931 prepared examples with 5,364 assigned to gradient training, checkpoint
reload error zero and isolated online calls verified. See
[training report](reports/FLIP_EGG_TRAINING_V1.md) and
[API calling contract](runs/flip_egg/training_v1/package_00/source/CALLING.md).
**Push training_v5 completed (2026-09-22):** API-owned recovery replaced unreliable
contact-point supervision with goal-conditioned planar EE continuation.
9,641 parameters, 800 updates, selected step 600; 168 prepared examples cover all
12 original groups, with 74 from A used for training and 94 from B for development.
Reload and independent process/planar adapter checks passed. Development endpoint
error is 5.095 mm versus 4.764 mm for the goal-translation baseline; no advantage
over that baseline or physical/generalization success is established. Initial
contact selection is deterministic geometry, not learned by this model. See the
[training report](reports/PUSH_TRAINING_V5.md) and
[API calling contract](runs/push_letters/training_v5/package_01/source/CALLING.md).
The rejected 142-label contact package and intermediate 205-example recovery
remain preserved. Separate API usage: [Push](reports/PUSH_TRAINING_V5_API_USAGE.md),
[Flip](reports/FLIP_EGG_TRAINING_V1_API_USAGE.md), including failed/recovery calls.

**Push training_v4 is complete (reviewed 2026-09-21).** The single cut_v6
`shape_push_v1` model predicts a contact point, planar direction and short stroke.
It has 15,041 parameters and completed the API-selected 500 updates; final EMA
reload error is zero and the local calling check passed. **Only 4/47 decision
anchors survived preprocessing, covering 4/12 learning intervals.** This is a
tiny-data training prototype, with no generalization or robot-performance claim.
All 77 requests/responses used Astra/xhigh; 18 submitted files have exact API
authorship, estimated cost USD 16.095949. Actual perception uses chromatic
geometry, without SAM/Qwen weights. Read the [Chinese training/coverage report](reports/PUSH_TRAINING_V4.md),
[local calling guide](policy_training_v4/README.md),
[API policy catalog](runs/push_letters/training_v4/package_00/source/POLICY_CATALOG.md),
[checkpoint library](runs/push_letters/training_v4/library.json) and
[completion receipt](runs/push_letters/training_v4/completion_receipt.json).
Previous cuts, models and the v3 portable export remain preserved.

**Independent Flip egg cut_v1 is complete (2026-09-21).** API selected two
data groups, one six-dimensional blade-motion policy and one auxiliary scene
model: 40 segments, all 45,730 rows retained, 32,737 control anchors. All 79
Astra/xhigh calls and 129 published files passed verification; estimated cost
USD 12.4355135. See the [Chinese decomposition and IO review](reports/FLIP_EGG_CUT_V1_REVIEW.md)
and [completion receipt](runs/flip_egg/cut_v1/completion_receipt.json).
Its [task specification](task_specifications/flip_egg_v1.md),
[configuration](configs/flip_egg_cut_v1.json), [isolated interface](flip_egg_cut_v1/README.md)
and [execution status](reports/FLIP_EGG_CUT_V1_STATUS.md) are separate from Push.
Scope ends at cuts, priors, preprocessing and calling/executor contracts;
no policy implementation, training or robot execution is included.

**training_v3 portable package is ready (2026-09-21).** The
[download and installation guide](deployment_v3/README.md) covers Linux/NVIDIA
inference on another computer. All three exact final EMA checkpoints and API
prior/usage/handoff documents are in the approximately 22 MiB complete archive.
The actual archive passed a relocated check in the locked inference environment,
with zero candidate-action difference from the original workers. See
[deployment evidence](reports/PUSH_V3_DEPLOYMENT.md). New v3 code is still local,
not committed/pushed; the complete archive works independently of Git publication.

**training_v3 is complete (2026-09-21):** all three cut_v5 priors completed
20,000 independent updates, with zero final EMA reload error. Actual local
calls returned finite six-dimensional candidates and passed stale-input,
invalid-goal and interruption checks. The API chose waypoint residual,
contact-graph and goal-field servo models, with explicit reasons for each
alternative to Diffusion Policy: 15 submitted files, 34 Astra/xhigh calls,
estimated USD 8.6769445. All 26 segments retain action supervision; sample
counts are 1,362 / 15,765 / 15,765. One 600-row relocation segment lacks a
trusted inferred object goal, and all derived perception labels remain unreviewed.
The decoder is disabled; no robot or generalization evaluation occurred.
See the [training report](reports/PUSH_TRAINING_V3.md),
[API calling catalog](runs/push_letters/training_v3/package_00/source/POLICY_CATALOG.md),
[checkpoint library](runs/push_letters/training_v3/library.json),
[completion receipt](runs/push_letters/training_v3/completion_receipt.json),
[configuration](configs/push_training_v3.json) and
[versioned interface](policy_training_v3/INTERFACE.md).
Previous designs, training versions and checkpoints remain preserved.

**cut_v5 is complete (2026-09-21): 2 datasets, 3 policy/prior designs,
26 segments.** The small-demonstration prompt produced more explicit
object/interaction-relative motion residuals and task-space action targets,
while merging alignment into relocation and retaining internal motion modes.
It did not create a standalone planar-push dataset. Read the
[Chinese result and comparison](reports/PUSH_CUT_V5_REVIEW.md),
[API policy catalog](data/push_letters/cut_v5/POLICY_CATALOG.md) and
[execution record](reports/PUSH_CUT_V5_STATUS.md). All 29 Astra/xhigh calls and
85 published files passed verification; estimated USD 5.48676. Its subsequent
policy implementation and training are recorded separately as training_v3 above.
Earlier versions remain preserved. Ongoing
data-processing permission is retained in the
[authorization record](authorizations/openai_data_processing_20260921.json).

**cut_v4 is complete (2026-09-21): 3 subtask datasets, 4 policy/prior designs,
28 segments.** The API separated tool access/clearance, piece relocation and
local alignment, with two alternative relocation priors. All 16,745 source
records are retained; 16,417 have finite native joint commands before input
preprocessing. All 27 Astra/xhigh calls and 93 published files passed verification;
estimated API cost USD 5.3269. No cut_v4 policy implementation or training yet.
Start with the [Chinese review and timeline](reports/PUSH_CUT_V4_REVIEW.md),
[API policy catalog](data/push_letters/cut_v4/POLICY_CATALOG.md), and
[full API report](reports/PUSH_CUT_V4.md). Execution provenance is in the
[status record](reports/PUSH_CUT_V4_STATUS.md).

The [subtask and agent-interface audit](reports/PUSH_CUT_V4_INTERFACE_AUDIT.md)
reviews each policy, measured access support, goal generation and handoff gaps,
and the changes needed for the existing two-policy training interface to support
the new groups. This is a review of the preserved design, without new API calls
or training; all 93 published files retain their hashes.

**Prompt used in cut_v4 (2026-09-21).** The
[new general prompt](prompts/general_cut_and_prior_v2.md) first identifies
subtasks covering the demonstrated behaviors, assembles a training dataset for
each through flexible cuts/reuse and compatible buffers, then designs independent
generalization priors and agent calling contracts. Real-robot preprocessing and
privileged-input requirements remain explicit. See the
[revision record](reports/CUT_PROMPT_V2.md). The executed cut_v3 prompt and
configuration remain unchanged.

**Additional training_v2 is complete (2026-09-21).** Both models completed
20,000 updates and exact EMA reload checks. Diffusion was explicitly permitted;
the API independently chose Gaussian-mixture models again. The new package has
separate policy prior/usage documents, structured handoffs and a v2 host interface.
Actual supervision is 6,314 examples from 12/22 cuts; no robot-performance claim.
See [training and coverage](reports/PUSH_TRAINING_V2.md),
[download/calling guide](deployment_v2/README.md) and
[Agent policy catalog](policies/push_v2/source/POLICY_CATALOG.md).
cut_v3, train_v1 and its push_v1 deployment remain preserved.

**Portable deployment is ready (2026-09-21).** Use the
[download/install guide](deployment/README.md). Exact API source is now also
available in [policies/push_v1/source](policies/push_v1/source/), while the two
inference-only checkpoints go in `checkpoints/push_v1/`. The complete SCP bundle
and weights-only archive are under `exports/push_v1/`; neither includes training
data, caches or API credentials. See the [deployment receipt](reports/PUSH_DEPLOYMENT.md).
The robot-side adapter remains the user's responsibility. Original runs stay intact.

Retained previous round: **train_v1 complete**, reviewed on 2026-09-21 (Asia/Singapore).
The Runtime API has submitted both `contour_push` and `visual_push`,
including their observation/goal conversions and action decoder, in a ten-file
[source package](runs/push_letters/train_v1/package_00/source/). The twenty
GPT-6 Astra/xhigh calls cost an estimated USD 4.1899. The developer
supplies execution interfaces; each policy completed 20,000 independent updates
on GPUs 0 and 1. Both final checkpoints reload with zero candidate-action
difference. No separate preliminary validation study was added.
Current configuration: [push_train_v1.json](configs/push_train_v1.json);
[implementation contract](policy_training/INTERFACE.md). Source, API journals,
prepared caches, independent checkpoints and failures are retained under
`runs/push_letters/train_v1/`. The cut_v3 artifacts below remain frozen.
User authorization now covers official API data processing for this code stage.
Physical robot execution and Flip egg remain outside this stage.

Preparation is complete: **8,606 eligible supervised examples from 18/22 cuts**,
56.56% of the 15,217 candidates with finite commands. Four cuts have no eligible
examples; perception/association and goal derivation account for the additional
exclusions. Both independent training jobs completed, and both checkpoints load
through the local inference-worker interface. See
[PUSH_TRAIN.md](reports/PUSH_TRAIN.md) for exact coverage and limitations.
The [trained library](runs/push_letters/train_v1/library.json) contains checkpoint
paths, hashes, source authorship and training receipts; the
[host guide](policy_training/README.md) describes local loading and invocation.

The API's [calling contract](runs/push_letters/train_v1/package_00/source/CALLING.md)
defines current-instance inventory, spatial goals, memory and command decoding.
The local `policy_training.inference.PolicyProcess` adapter exposes `inventory`,
`act` and `close` through an isolated process with no hardware IO. The package
contains two independently learned Gaussian-mixture action models; no pretrained
perception weights are assumed. Its executable warm-material detector has
material, apparent-scale and occlusion limits, documented by the API in
[PRIOR.md](runs/push_letters/train_v1/package_00/source/PRIOR.md).

Completed work: **cut_v3, explicitly authorized on 2026-09-20**. The previous
cut_v2 outputs and session have been deleted with an inventory and retained cost
accounting. The new run uses these separately supplied English documents:

- [General cutting/prior prompt](prompts/general_cut_and_prior.md): coherent
  semantic segments, separate grouping/policy-sharing decisions, multiple
  independently trained priors per group, and a diverse callable library
  selected by a High-level Agent.
- [Push task specification](task_specifications/push_letters.md): pushing
  completely unseen letter identities/shapes into new requested words, variable
  layouts, task-specific evidence, and deployment requirements.

Both documents and the separate data contract are loaded and frozen by the
[cut_v3 configuration](configs/push_cut_v3.json). The interface represents and
checks segment conditions, context/supervision ranges, exclusions, inspected
boundary evidence, and policy-catalog mappings. Structural validation alone
does not establish that the API's semantic interpretation is correct.
All 12 interface tests passed. The user explicitly authorized sending selected
evidence to `api.openai.com`, resolving the initial prelaunch approval block.
The API submitted **22 semantic segments, one shared dataset group and two
independent heuristic/policy proposals**. All 25 actual requests used the frozen
documents and `gpt-6-astra/xhigh`; 71 published files and exact raw slices passed
verification. Estimated cost for this round: **USD 4.4980**, not an invoice.

The first recording contributes 12 segments and the second contributes 10.
Each pins a physical instance and a local achieved-placement goal with an
original-pixel box and source-frame anchor. Temporary placements and later
revisits remain separate, while both `contour_push` (boundary graph) and
`visual_push` (dual-view temporal RGB) learn from the same proposed group.
All 16,745 original records are retained. API-declared handoff masks exclude
1,200 rows from action loss; 15,545 remain candidates before validity checks.
Of those, 15,217 have finite command dq. These were the counts at the end of
cut_v3, before input conversion; the completed train_v1 count is 8,606 above.

See the [segment list and independent review](reports/PUSH_CUT_REVIEW.md),
[full API report](reports/PUSH_CUT.md),
[heuristic and supervision contracts](data/push_letters/cut_v3/datasets/instance_relocation/heuristic.md),
[proposed policy catalog](data/push_letters/cut_v3/POLICY_CATALOG.md),
[verification](runs/push_letters/cut_v3/validation.json), and
[requirements audit](runs/push_letters/cut_v3/requirements_audit.json).
At completion of cut_v3, perception, masks, goal conversion, policy code and
training remained later obligations. The latest authorization above now advances
the two Push policies into implementation and training; actual controller
verification, robot execution and generalization evaluation remain future work.
Flip egg was outside that Push stage; its separately authorized cut_v1 is
linked above. New Runtime API calls require `gpt-6-astra/xhigh`.

Deployment includes completely unseen letter identities/physical shapes and
entirely new words. Actual generalization requires future validation. The API
must justify its models, priors and inputs, declare dependencies, and define
High-level Agent calls including physical instance selection and spatial goals.
Each supervised segment must have a coherent condition assignment and label
definition; shared policies may learn from many such separate segments.

Historical accounting only: [cut_v1 deletion](deletions/push_cut_v1_20260920/receipt.json)
retains USD 2.5783 estimated cost; [cut_v2 deletion](deletions/push_cut_v2_20260920/receipt.json)
retains USD 2.6044. Those costs total USD 5.1827 and are separate from cut_v3.
No deleted scientific design is supplied to the new API session.

The developer owns interfaces and deterministic execution; Runtime API owns
segmentation, data grouping/reuse, priors and semantic handoffs. Each group may
contain one or several heuristics for independently trained callable policies.

Original data stays at `/home/storage/tianrunhu/real_robot_data`. Paired sample
indices define cuts; timestamps are retained without cross-device re-pairing.
Original recordings contain N paired observation/action records. Published cuts
retain all raw fields (including declared missing values), use [start,stop), and
remain separate sequences. Grouping does not invent continuity between cuts.
See the [data contract](DATA_CONTRACT.md) for exact fields, missingness, source
identity, published slice semantics, and future loader obligations.

| Directory | Ownership and purpose |
| --- | --- |
| `data.py`, `tools.py`, `transport.py`, `run.py` | Developer-owned readers, evidence tools, validator, publisher, transport and audit |
| `configs/`, `prompts/` | Explicit stage configuration and English API instructions |
| `task_specifications/` | Separate task-specific objectives and generalization/deployment requirements |
| `runs/push_letters/cut_v3/` | Source hashes, frozen interface/prompt/specification, complete API/tool journals, costs and validation |
| `data/push_letters/cut_v3/` | API plan, exact NPZ cuts, original-media references, supervision contracts, heuristics and proposed policy catalog |
| `deletions/` | User-authorized deletion inventories and accounting for removed runs |
| `reports/` | Measured outcome summaries and exact API text |
| `tests/` | Synthetic interface fixtures, never experimental candidates |

Default API evidence is 640x360 with aspect ratio preserved. Original resolution
and crops are available on API request. This does not reduce original training
data or change pairing indices. The custom input, augmentation and action
conversions are now implemented in the API-authored train_v1 package; the original
cut documents retain their historical implementation obligations.

Commands below default to cut_v3. Run authorized commands from the repository
root using the existing locked environment:

```bash
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m pytest -q real_robot/tests
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m real_robot.run prepare
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m real_robot.run run
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m real_robot.run status
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m real_robot.run verify
```

`run` sends paid API requests only under current user authorization. The official
transport uses the existing private out-of-repository credential, audits actual
model/effort fields, counts input tokens before generation, and preserves usage
and uncertain charges. There is no automatic retry/fallback. Its USD 100 guard
is a developer execution bound for the Push cut stage, not a user-approved budget
for future policy implementation or training. No credentials enter API tools.
