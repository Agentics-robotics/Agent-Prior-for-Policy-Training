# Additional APPL study: GPT-6 Astra / xhigh

Authorized 2026-09-17. The user explicitly selected `xhigh` (Extra high), not
`max`, for an additional complete APPL round on the existing five tasks.

## Scope

- Repeat API segmentation, heuristic creation, prior implementation, independent
  policy training and inference-time selection for all five tasks, including the
  original drawer. All API stages request `gpt-6-astra` with `reasoning.effort=xhigh`.
- Reuse the exact original twelve demonstrations per task, task geometry,
  completion contracts, full-demonstration normalization recipe, training seed 0,
  20000 updates per prior, final EMA, DDPM100, history 2, horizon 16 and execution 8.
- Preserve the existing English prompts, tool contracts and API/output limits.
  API alone authors boundaries, heuristics, policy source and semantic handoffs.
  No recovery-data augmentation, role-input changes or test-informed prompt edits.
- Evaluate 30 ID and 30 position-OOD layouts per task with the original declared
  seeds. The executor's 5000-step cap remains hidden from API inputs. These are
  paired follow-up tests on previously analyzed layouts, not new untouched tests.
- Reuse the frozen DP results and retain the original GPT-5.5/high APPL results
  as comparisons. No DP retraining or repeated DP physics is necessary.
- The comparison changes both model and reasoning effort; it does not isolate
  their individual effects. API-selected skill counts and compute may differ.
- No automatic transport retries, rewriting API output, manual candidate repair,
  or performance selection. Incomplete attempts retain unknown outcomes and costs.

## Resources and preservation

Physical GPUs 1, 2, 3 and 5 were initially selected after inspection on 2026-09-17.
A subsequent explicit user amendment raised the simultaneous limit to five GPUs;
device 7 was inspected idle and added. The current pool is 1,2,3,5,7, with multiple
jobs per GPU authorized. Preserve the original four-device scheduling receipts.
Inspect occupancy at launch and preserve unrelated processes.

During training, four preparation workers use the existing `prepare_queue` and
per-policy package locks to prepare subsequent API submissions on those same
devices. Their global round-robin assignments match the training queues exactly.
Bounded interface checks may overlap formal training. Additional training workers
pick already submitted, unstarted packages on the same assigned device, initially
targeting two formal training jobs per GPU with a 10,000 MiB free-memory launch gate.
Manual ready-package admission also inspects current occupancy. Independent queues
can overlap three formal jobs on one allocated GPU when new submissions arrive;
the user's hard limit remains five physical GPUs with multiple jobs per GPU.
Actual overlap is retained in `ready_training_additions/admission_amendment.json`.
The original checkpoint locks prevent duplicate training. Required deployment
checks use the original check implementation; completed exact-checkpoint checks
are reused by the original queues, including checks launched early from the active
session after a queue interruption or while it awaited another model.
No additional candidate, retry, training budget or learned computation is
introduced by resource scheduling. Scheduling receipts
and occupancy are retained under `preparation_overlap/` and `training_overlap/`.

The added device uses two workers from `extra_gpu.py` on unstarted policies from
the end of the same 36-policy portfolio. Resource configuration copies under
`configs/astra_xhigh/resources_5gpu/` differ only in `devices`; the running
coordinator and its original configurations are preserved. Existing package and
checkpoint locks prevent duplicate work when queues meet. Actual device indices,
configuration hashes and the user amendment are recorded in `allocation_5gpu/`.
All policies still pass the original coordinator's checks before global freeze.

A first design request for `tray_pack/deliver_block__h02` received HTTP 429 with
`gateway_concurrency_limit`; no tool or file generation occurred. Its original
receipt is retained, with no automatic retry. Supplemental worker receipts are
also retained. Future evaluation processes use four shared provider
request slots, independently of GPU/episode concurrency. The capacity gate wraps
the unchanged client once, forwarding its exact request and return value without
retry. The earlier outer runner is retained under `resource_versions/`; the new
runner pins the gate source hash and is frozen before evaluation. See
`api_capacity/amendment.json` and `incidents/tray_deliver_block_h02_429/`.

New outputs: `runs/exp2/M1_astra_xhigh/`.
New segmentation: `data/exp2/scaleup_astra_xhigh/<task>/processed/`.
New configurations: `experiments/exp2/configs/astra_xhigh/`.
Orchestration: `experiments/exp2/astra/runner.py`.
Read-only progress: `python -m experiments.exp2.astra.status` in the locked Exp2
Pixi environment. API preparation wrapper: `experiments/exp2/astra/prepare.py`.

The runner reuses the original scientific `src/appl` implementation. It binds the existing
scale-up evaluator to the new configuration/output paths in its own process;
environment, action execution, success predicates, prompts and policy interfaces
are unchanged. Its separate source hash joins the new study freeze. Original
source, data, checkpoints, journals, results, Exp1 and M0 remain preserved.

After all 32 unaffected submitted models and deployment checks finished, one
verified framework correction was applied to `prior_policies/engine.py`: the
model restored for EMA reload verification now also has `requires_grad=False`,
matching the EMA instance. The retained failing checkpoint reproduces identical
weights but a 0.000101447 action difference with mismatched flags; matching flags
reduces that difference to zero. This changes the verification only, preserving
the optimizer, saved checkpoint, sampler, `LoadedPolicy` deployment behavior,
1e-6 tolerance, prompts and candidate source. The framework guard accepts only
this exact one-line difference from the original frozen source and rejects other
changes. Original/corrected source, two zero-update diagnoses and validation are
retained under `incidents/buffer_red_h02_reload/`.

On 2026-09-17 the user explicitly authorized the four proposed recoveries. The
three interrupted sessions with no generated files each receive one new session,
with original API/tool calls deducted from their original budgets. Complete old
attempts remain under `retained_design_attempts/`. The affected buffer design
continues its original API history with at most eight additional calls and two
additional interface checks (44 calls / 7 checks total, unchanged 100-tool limit).
Its original journal, files and conversation prefix are retained; a separately
identified framework message explains the corrected gate. API owns all code and
must explicitly submit an exact successfully checked package. This continuation
used two new API calls and one successful check, without modifying policy files.
Formal training remains 20,000 updates. Recovery authorization and execution
receipts are under `incidents/` and `authorized_recovery/`; `finish_recovery.py`
supervises training, deployment checks and the original 300-test matrix. No further
automatic retry is introduced. The original failed coordinator record is retained
under `authorized_recovery/supervisor/original_coordinator.json`.

The original drawer segmentation used the M1_v2 configuration without `task_id`.
The new drawer configuration preserves that input contract; the outer runner
identifies the task by its filename. Other task configurations retain `task_id`.

## Evidence

`preparation.json` records original inputs, configuration differences, comparator
hashes and the original framework. The later reload-gate revision has its own
receipt and does not rewrite that preparation record. `study_freeze.json` is written only
after every API submission, model and B=1 deployment-worker check passes. Raw API
journals preserve actual request/response model fields and usage. Results, paired
comparisons, original-frame videos and any interrupted prefixes are retained.

After explicitly authorized reconciliation of retained design interruptions,
`python -m experiments.exp2.astra.runner run-evaluation` continues at the original
test matrix without rerunning design queues or overwriting their failed receipts.
Use the locked Exp2 Pixi environment. This entry holds the coordinator lock and
requires all 36 API submissions, completed training processes, checkpoints and
deployment checks before it can freeze or start any test. The early rejection of
an incomplete portfolio was verified with zero API calls and zero simulator steps.

Official model documentation lists `low`, `medium`, `high`, `xhigh`, `max`:
https://developers.openai.com/api/docs/models/gpt-6-astra
The experiment records the exact API value `xhigh`; it does not infer an API
mapping for a UI label such as “Ultra high”.

## Authorized evaluation recovery, 2026-09-17

The original 300 attempts produced 211 complete outcomes and 89 HTTP-interrupted
attempts. The user authorized one reset retest for each of those exact 89 cells,
keeping the other 211 outcomes and all frozen scientific settings. The separate
`recover_evaluation.py` routes outputs to `M1_astra_xhigh/evaluation_recovery`,
verifies actual outgoing gpt-6-astra/xhigh fields without changing the request,
and runs one zero-step interrupted cell alone before admitting four concurrent
cells on physical GPUs 1/2/3/5. Any new execution error stops further admission;
there is no retry of a retest. Original source and attempts remain immutable.

The first retest (buffer_swap / ID / 20122) returned HTTP 503 on its first request
at zero physical steps. Its request and initial prompt exactly match the old
attempt. The coordinator exited with 88 cases unstarted and verified 24,318
original files unchanged. Validation, usage uncertainty, the original proposal,
user authorization and the incomplete outcome are retained in the recovery
report and `incidents/evaluation_outage`. No additional model training occurred.

## Service-restored resumption and explicit follow-up, 2026-09-18

The user reported service restoration and explicitly requested completion.
`evaluation_recovery_20260918` retained a new reset attempt for each authorized
cell, including the preceding zero-step interruption. It ended after 69 attempts:
68 complete outcomes and one HTTP 502 at two_block_sort / OOD / 30027, physical
step 679, request 15. Admission stopped, and all active episodes finished; 20
cases remained unstarted. Its 69 traces/prefixes and 69 videos passed independent
audit. The cumulative matrix reached 279 complete outcomes (181 successes,
98 task failures), with 21 unknown cells. The coordinator exit, all API costs
and 24,351 unchanged earlier files are recorded in that directory.

The user then explicitly selected "授权这 21 项受控续跑（推荐）" for the exact
`incidents/evaluation_outage/followup_proposal_20260918.json`: one new reset for
the interrupted cell and the 20 unstarted cells, one attempt each. Authorization
is stored separately as `followup_authorization_20260918.json`.
`followup_evaluation.py` reuses the frozen evaluator, request gate and policy
library, writing only to `evaluation_followup_20260918`. It starts one episode
alone, then four slots on GPUs 1/2/3/5, with the same fail-stop admission rule.
`followup_report.py` audits new attempts and composes their results with the
retained 279 complete outcomes. Earlier reports, interrupted attempts and
unreported HTTP-error usage remain intact. No prompt, policy, training budget,
geometric goal or API-authored output changes accompany either continuation.

Final review, 2026-09-18: the 21-case follow-up completed with 16 successes,
five task failures and no transport interruption. The combined new-study matrix
contains **300 complete outcomes, 197 successes, 103 task failures and zero
unknowns**: ID 124/150, position OOD 73/150. All 21 new execution audits and video
validations passed, and 32,209 earlier files retained their hashes. The evidence
chain covers all 300 selected outcomes and videos, alongside every interrupted
attempt. All evaluation processes exited; no additional training occurred.
The reused GPT-5.5/high comparator keeps its separate historical unknown cell.
See the [final report](../../runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/REPORT.md),
[analysis](../../runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/ANALYSIS.md),
[completion receipt](../../runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/completion.json)
and [resource release](../../runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/resource_release.json).
