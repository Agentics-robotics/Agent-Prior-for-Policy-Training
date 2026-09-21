# Real robot demonstrations

Current stage: **train_v1 complete**, reviewed on 2026-09-21 (Asia/Singapore).
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
Flip egg remains future work. New Runtime API calls require `gpt-6-astra/xhigh`.

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
