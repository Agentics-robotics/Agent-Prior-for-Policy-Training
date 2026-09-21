# Push train_v1

Reviewed **2026-09-21** (Asia/Singapore). API implementation, full-data preparation
and **both independent 20,000-update trainings are complete**. The executable
source package, preprocessing cache, final EMA checkpoints and calling contracts
are retained. Completion is recorded in
[execution_status.json](../runs/push_letters/train_v1/execution_status.json) and
the [trained library](../runs/push_letters/train_v1/library.json).

Deployment follow-up (2026-09-21): the same final EMA weights and exact API code
are now available through a [portable download package](../deployment/README.md),
with a [relocation check](PUSH_DEPLOYMENT.md). Training artifacts below remain
unchanged; this adds serialization and a portable host interface only.

## Scope and ownership

The user authorized implementing and training both frozen cut_v3 priors, without
an additional preliminary validation study. The original 22 cuts and all source
recordings remain unchanged. This stage does not run a physical robot or Flip egg.

GPT-6 Astra/xhigh wrote and explicitly submitted ten source/document files in
[package_00/source](../runs/push_letters/train_v1/package_00/source/).
The developer supplies read-only data tools, isolation, optimizer execution,
GPU scheduling, local observation/action transport and accounting. The API owns
model structure, all input/goal transformations, augmentation, losses, invocation
logic and command decoding. No developer-authored scientific replacement is used.

The package hash is
`1498f35a0f7627fea65aeef493fad789c89be1f91df7b82589897a11bd6b6be8`.
The frozen cut plan hash is
`51e620df25f6ced540cde6ae85b5bd8a33a7a4e089018eb56f07c56cc80484a2`.
Exact per-file hashes and submission evidence are in
[submission.json](../runs/push_letters/train_v1/package_00/submission.json).

## Implemented policies

| Policy | API implementation | Fixed training plan |
| --- | --- | --- |
| `contour_push` | Boundary graph, three message-passing layers, eight-row causal GRU, five-component diagonal Gaussian action mixture | GPU 0, batch 64, 20,000 AdamW updates |
| `visual_push` | Third-view global/detail, wrist and foreground-goal CNN inputs; sparse causal history GRU; independent five-component Gaussian action mixture | GPU 1, batch 12, 20,000 AdamW updates |

Both predict seven recorded command-dq values and learn an auxiliary future
centroid displacement from eligible training-only intervals. They have independent
random initializations and weights. Final EMA weights at the fixed update budget
are retained; there is no performance-based checkpoint selection or held-out
success-rate study. Shared normalization and preprocessing do not share trained
motor weights. These are Gaussian-mixture policies, not diffusion policies.

| Completed policy | Trainable parameters | Updates / sample exposures | Training time | Reload max action difference |
| --- | ---: | ---: | ---: | ---: |
| `contour_push` | 3,671,118 | 20,000 / 1,280,000 | 1,800.95 s | 0 |
| `visual_push` | 9,067,629 | 20,000 / 240,000 | 1,471.69 s | 0 |

Training ran concurrently on physical GPUs 0 and 1. The actual neural weights
changed, gradients were finite and nonzero at the recorded checkpoints, and
reloading each final EMA reproduced its candidate actions exactly on the executor's
two-example reload check. This is artifact integrity, not a held-out evaluation.

- Contour: [checkpoint](../runs/push_letters/train_v1/package_00/training/contour_push/last.pt),
  [training receipt](../runs/push_letters/train_v1/package_00/training/contour_push/result.json).
- Visual: [checkpoint](../runs/push_letters/train_v1/package_00/training/visual_push/last.pt),
  [training receipt](../runs/push_letters/train_v1/package_00/training/visual_push/result.json).

Final checkpoint SHA-256:

```text
contour_push da063b987d6c594038a8f7e24f0b4e2778c9c42a8c381f26729af845a5886326
visual_push  56f3044c9444fdaf4495c89a2d3ad13c4dbe8c8d89f8f9c357fcdb3236da8846
```

## Inputs and executable calling path

Sample indices and existing paired images define observations. The converter
does not re-pair images using robot-versus-host absolute timestamps. Original
images are undistorted using their calibration; no unverified depth registration
or additional TCP offset is assumed. Future achieved-placement anchors label
training goals only. Online calls use the current scene and an HLA-selected
physical instance, fixed spatial goal and invocation lifetime.

The API replaced unavailable learned segmentation/flow assets with an executable
warm-material component detector and forward association. No pretrained weights
are assumed or downloaded. This supports class-free shape inputs but has explicit
material, contrast, apparent-scale, separation and occlusion limitations. It is
not arbitrary-material object segmentation. No alphabet recognizer is included;
the HLA or its semantic component must interpret letters, allocate repeated-letter
instances and choose positions/orientations for the requested word.

The exact API interface is documented in
[CALLING.md](../runs/push_letters/train_v1/package_00/source/CALLING.md) and its
adaptations in [PRIOR.md](../runs/push_letters/train_v1/package_00/source/PRIOR.md).
The developer adapter is
[PolicyProcess](../policy_training/inference.py): `inventory(observation,
scene_version)`, `act(observation, call, reset=False, controller_contract=None)`
and `close()`. It isolates generated code and passes current numeric observations
over local IPC. It never writes commands to robot hardware.
The [host usage guide](../policy_training/README.md) shows how to load either
trained policy. Both completed checkpoints also loaded successfully through
that isolated inference-worker interface and exited cleanly, with zero evaluated
observations, optimizer updates or hardware IO; receipts are under
[inference_startup_00](../runs/push_letters/train_v1/inference_startup_00/).

Without a commissioned plane homography, the API permits only explicitly accepted
local image-plane similarity goals within 160 pixels. Larger perspective-changing
moves require that calibration. Candidate dq computation works offline;
hardware command serialization additionally requires verified controller semantics
and limits. Neither training completion nor finite actions establish robot success
or generalization to unseen letters, shapes or words.

## Execution evidence and accounting

Preparation took 1,350.47 seconds and produced a 5,283,216,257-byte numeric cache.
**8,606 supervised examples from 18 of 22 segments** remain, or 56.56% of the
15,217 candidates with finite recorded commands. The first/second original
trajectory contributes 5,428 / 3,178 examples. Both policies use this same eligible
set, with their own batch construction and independent weights.

| Mutually exclusive row outcome | Count |
| --- | ---: |
| Eligible supervision | 8,606 |
| Frozen handoff/supervision exclusion | 1,200 |
| Missing command | 328 |
| Goal foreground/identity derivation failure | 2,410 |
| Invalid perception or selected-instance tracking | 4,201 |
| Invalid required state/timing after preceding filters | 0 |
| Total original paired records | 16,745 |

The exclusion order is part of the API implementation; zero timing exclusions
does not certify all raw timestamps or rows rejected earlier. No cross-device
timestamp re-pairing was performed.

| Segment | Eligible rows | Coverage limitation where material |
| --- | ---: | --- |
| `a_w` | 1,362 | |
| `a_i_stage1` | 93 | |
| `a_o` | 1,210 | |
| `a_r_stage` | 320 | |
| `a_i_stage2` | 561 | |
| `a_h_stage` | 318 | |
| `a_d_stage` | 0 | 270 candidate rows lack valid selected-instance perception |
| `a_r` | 750 | |
| `a_d_align` | 310 | |
| `a_l` | 0 | 1,850 candidate rows: anchor foreground absent |
| `a_i_align` | 19 | 371 candidate rows lack valid selected-instance perception |
| `a_h_align` | 485 | |
| `b_w` | 348 | |
| `b_i_stage` | 160 | |
| `b_d_stage` | 400 | |
| `b_o` | 860 | |
| `b_r` | 216 | 1,464 candidate rows lack valid selected-instance perception |
| `b_d_align` | 0 | 220 candidate rows: ambiguous anchor instance |
| `b_l` | 34 | 1,146 candidate rows lack valid selected-instance perception |
| `b_h_stage` | 760 | |
| `b_i_align` | 0 | 340 candidate rows: ambiguous anchor instance |
| `b_h_align` | 400 | |

Thus this run does **not** demonstrate full motion coverage. In particular the
L-labeled segments and final I alignment have very little or no supervision.
The source recordings and cuts were not removed or altered; exclusion is from
the implemented training inputs, not from the retained dataset. This is an
observed converter limitation, not an action-label repair or success claim.

Full preparation reports per-segment retained/excluded rows in
[prepared/stdout.log](../runs/push_letters/train_v1/package_00/prepared/stdout.log).
Exact input hashes and counts are in
[prepared/result.json](../runs/push_letters/train_v1/package_00/prepared/result.json).
Native action labels and supervision indices were checked against the frozen
source before optimization.

The executor records finite nonzero gradients, actual weight changes, optimizer
updates and checkpoint reload agreement as operational checks. They are not an
additional semantic/performance experiment. Frozen execution source and locked
environment hashes are retained in
[executor_freeze.json](../runs/push_letters/train_v1/package_00/executor_freeze.json).
Each worker retains its device identity and isolation receipt.

API implementation: 20 consumed calls, estimated **USD 4.1899015**, with no
automatic transport retries. This is separate from cut_v3 (**USD 4.4980205**),
making **USD 8.6879220** for the retained cut plus implementation. Earlier deleted
cut rounds cost another **USD 5.1827295**. These are usage-based estimates, not
provider invoices, and exclude local GPU cost. Journals, saved wire requests and
cost ledgers remain in
[design_00](../runs/push_letters/train_v1/design_00/).
All twenty actual requests **and responses** report `gpt-6-astra/xhigh`, as
recorded in [request_identity_audit.json](../runs/push_letters/train_v1/request_identity_audit.json).
The final library maps each exact submitted source-file hash back to its API
`write_file` response and explicit package submission. No repair call, transport
retry or additional training attempt was needed.
