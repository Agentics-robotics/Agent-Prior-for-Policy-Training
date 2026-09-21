# Push training_v2

Reviewed 2026-09-21. **Both independent 20,000-update trainings, exact EMA
exports and the final archive's relocated interface check are complete.**
The original cut_v3, train_v1 and push_v1 artifacts remain preserved.

The user requested an additional training round with explicit permission to use
Diffusion Policy or choose another learned policy family independently for each
subpolicy, and asked for policy-specific prior, usage and handoff information.

The run retains cut_v3's two heuristic assignments, original source recordings,
the existing numerical interface and fixed 20,000-update executor recipe. The
English prompt adds architecture-choice clarification and deployment documentation
requirements. The API will author a new package without train_v1 source or its
performance diagnoses as feedback. New implementation choices may therefore
differ beyond model family; this is not a controlled architecture-only ablation.

- Configuration: [push_training_v2.json](../configs/push_training_v2.json).
- Prompt: [implement_policies_v2.md](../prompts/implement_policies_v2.md).
- Unchanged interface: [INTERFACE.md](../policy_training/INTERFACE.md).
- New run: `real_robot/runs/push_letters/training_v2`.
- Prior run and deployment: preserved under `train_v1` and `push_v1`.

Required API documents are shared `PRIOR.md` / `CALLING.md`, per-policy
`<policy_id>_PRIOR.md` / `<policy_id>_USAGE.md`, `POLICY_CATALOG.md`, and
`HANDOFF.json`. They must describe actual inputs, model mechanisms, observable
selection/entry/continuation/exit conditions, successor readiness, failure cues,
goal conversion and memory reset on switching, with training-evidence limits.

Existing authorization covers sending selected original images, states, cuts,
heuristics and metadata to the official OpenAI Responses API for implementation.
Requests explicitly use gpt-6-astra/xhigh. Local training uses the locked Exp2
environment on GPUs 2/3, inspected free at setup. No physical robot execution,
new cuts, extra policy candidates or Flip egg work is included.

Completion is recorded in [execution_status.json](../runs/push_letters/training_v2/execution_status.json)
and the [trained library](../runs/push_letters/training_v2/library.json).

## Submitted design

The API submitted 15 files in [package_00/source](../runs/push_letters/training_v2/package_00/source/),
hash `7b109fef60fc83744aaa2b73302e87b7fc7995df3a8da327aac275794c605b5a`.
Both policies independently choose a five-component diagonal Gaussian mixture.
The API cites the single-command interface and limited demonstrations as design
reasons; these are not comparative evidence against diffusion. Contour uses
boundary message passing, visual uses distinct image encoders, and both have
independent two-layer gated memory. Batch sizes are 32 / 12, learning rates
0.00025 / 0.0002; the unchanged outer budget is 20,000 updates per policy.

The completed design has 25 consumed gpt-6-astra/xhigh calls, estimated USD
4.5840845 and zero unresolved charge reservations. This is a usage-based estimate,
not an invoice. Source code and documentation are exact API output; no manual
scientific edits were made. [Documentation structure and source hashes](../runs/push_letters/training_v2/documentation_check.json)
passed before data preparation. The API repaired its own unsubmitted top-level
hook declarations in response to the unchanged structural checker.

Agent documentation starts at [POLICY_CATALOG.md](../runs/push_letters/training_v2/package_00/source/POLICY_CATALOG.md).
It links each policy's prior/usage and the shared [HANDOFF.json](../runs/push_letters/training_v2/package_00/source/HANDOFF.json).
Numerical goal preview/reexpression functions are included in the API package.
These are deployment contracts, not evaluated motion/handoff guarantees.

## Prepared data and coverage

Preparation completed in 442.44 seconds and produced a 7,801,253,233-byte cache.
Both policies use the same **6,314 eligible examples from 12/22 cuts**. Original
recordings and cuts were not deleted or relabeled. Unlike v1's separate goal and
tracking counters, this API implementation groups those failures under its
`invalid_perception` counter; the categories should not be compared one-to-one.

| Mutually exclusive outcome | Original rows |
| --- | ---: |
| Eligible examples | 6,314 |
| Frozen handoff exclusion | 1,200 |
| Missing command | 328 |
| Perception/goal association failure | 8,903 |
| Invalid required state after preceding filters | 0 |
| Total original paired records | 16,745 |

The ten empty cuts are `a_i_stage1`, `a_r_stage`, `a_d_align`, `a_l`,
`a_i_align`, `a_h_align`, `b_w`, `b_d_align`, `b_l`, `b_i_align`.
All except `b_w` failed anchor foreground/box association; `b_w` failed the
foreground similarity check. Further per-row failures occur in retained cuts.
Both L-labeled cuts and both final I-alignment cuts have no supervised examples.
This run therefore has narrower observed coverage than v1's 8,606 examples and
18/22 cuts. This is a coverage comparison, not an evaluated policy ranking.
See the [preparation receipt and per-cut counts](../runs/push_letters/training_v2/package_00/prepared/result.json).

The separate [v2 host interface](../deployment_v2/README.md) forwards the exact
API observation inventory, goal preview/reexpression, action and decoder functions.
It adds no scientific rule, controller or motor training update. Its exported
weights and interface-check evidence are recorded below.

## Completed models and checks

| Policy | Parameters | Updates | Sample exposures | Training seconds | EMA reload max action error |
| --- | ---: | ---: | ---: | ---: | ---: |
| contour_push | 3,846,158 | 20,000 | 640,000 | 932.73 | 0 |
| visual_push | 4,331,285 | 20,000 | 240,000 | 1,496.76 | 0 |

Both models have finite nonzero recorded gradients and changed learned weights.
Each uses the final EMA at the declared budget; no performance-based selection,
extra candidate, repair training attempt or robot rollout occurred. Training
receipts are [contour](../runs/push_letters/training_v2/package_00/training/contour_push/result.json)
and [visual](../runs/push_letters/training_v2/package_00/training/visual_push/result.json).
The data-preparation and training stages used physical GPUs 2/3 and have exited.

The [final archive interface check](PUSH_TRAINING_V2_INTERFACE_CHECK.json) ran
from a fresh extraction without original data readers, training runs or caches.
For each model it checked loading, empty-input refusal, inventory on one host-supplied
paired frame, goal preview, goal reexpression, one finite candidate action, and
unverified-controller refusal. It executed zero robot actions and zero candidate
optimizer updates. The unchanged-placement fixture is an interface exercise,
not a task-success, latency or generalization evaluation. An initial archive/check
was retained under `exports/push_v2/initial_build` and the run's
`interface_check_initial.json`; the final archive only completes standalone
download documentation. The published archive's own hash is in the final receipt.

[Preservation checks](../runs/push_letters/training_v2/preservation_after.json)
verified all 27 recorded v1/source/checkpoint files, 15 executor files and 15
submitted API files unchanged. New host/package code is developer-authored;
all policy, perception, goal, decoder and handoff semantics remain API-authored.

## Agent documents and download artifacts

The Git-visible [policy catalog](../policies/push_v2/source/POLICY_CATALOG.md)
links both separate priors and usage guides; [HANDOFF.json](../policies/push_v2/source/HANDOFF.json)
contains their observable selection, entry, continuation, exit, successor-readiness,
switching, failure and reset contracts. Read the developer's
[actual training coverage](../policies/push_v2/TRAINING.md) alongside those designs.
All 15 source/doc files exactly match the API submission.

- [contour_push.pt](../checkpoints/push_v2/contour_push.pt): 15,420,725 bytes,
  58 tensors exactly equal to the original final EMA.
- [visual_push.pt](../checkpoints/push_v2/visual_push.pt): 17,374,623 bytes,
  100 tensors exactly equal to the original final EMA.
- Full standalone bundle: `real_robot/exports/push_v2/push_v2_bundle.tar.gz`,
  30,411,883 bytes; SHA-256 `53db8f837c014b6568c151946ee10634d60daeeb01eaa5940b30271be72eaccd`.
- Weights-only archive: `real_robot/exports/push_v2/push_v2_weights.tar.gz`,
  30,326,245 bytes; SHA-256 `f8f2eacc3a6a3ec3fec938f4c340855ec9069d0b4b037dccebb4476e23e74938`.

Use the [v2 download/calling guide](../deployment_v2/README.md). The v2 helper
and call schema differ from v1; its host entry is
`from real_robot.deployment_v2 import PushPolicy`. Original v1 remains separate.
No remote Git publication, third-party upload or robot-side bridge was performed.
API cost for this additional round remains USD 4.5840845 (25 calls, estimated,
not an invoice; local GPU cost excluded). Training narrower coverage does not
establish that v2 is better or worse than v1; no such comparison was run.

Setup receipts: [exact prompt diff](../runs/push_letters/training_v2/prompt.diff),
[configuration changes](../runs/push_letters/training_v2/setup.json),
[preserved source/checkpoint hashes](../runs/push_letters/training_v2/preservation_before.json),
and [actual request audit](../runs/push_letters/training_v2/request_identity_audit.json).
