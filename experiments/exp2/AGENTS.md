# Active Exp2 rebuild

Frozen-DP diagnosis (2026-09-19): the user authorized original-training-reset
experiments and detailed comparison against drawer DP to investigate hidden bugs
and narrow data support. The completed developer-owned diagnostic is
analysis/dp_diagnosis_20260919/REPORT.md, with its own plans, 60 learned reset
trials, 60 expert replays, 15 binary-gripper trials, 15 expert-prefix takeovers,
offline sensitivity probes, retained setup/prefix guard incidents, and completion
audit. It adds no training or Runtime API requests. Preserve the original formal
results, frozen scientific source, data, checkpoints and Exp1. Diagnostic caps
are 1,500 steps, and adaptive three-reset results are not formal held-out scores.
Devices 2/3/4/5/7 respected the five-device limit. This work is complete, not an
instruction to replay it. Future baseline changes need a separately declared
comparison; the old DP's gripper feedback instability and narrow intermediate
state support must be acknowledged when writing up the original main matrix.

Historical comparison recovery (2026-09-19): the user explicitly requested rerunning
the one unknown outcome and replacing its reporting cell. Execute exactly one
reset of original APPL buffer_swap / ID / 20116, whose API service interruption
occurred at 650 physical steps. This is reproduction of the historical
GPT-5.5/high configuration, not a new xhigh design experiment. Preserve its actual
model/effort, prompt, nine policy checkpoints, paired initial state and 5000-step
cap. Keep the original interrupted attempt and frozen reports unchanged; new
output is M1_scaleup/evaluation_recovery_20260919 and the paper report selects
the audited replacement. No training, completed-failure retests or automatic
transport retries. Archive the preceding paper ZIP and update the current bundle.
Executor: analysis/recover_scaleup_unknown.py. This recovery completed: 945-step
success, ten consumed API requests, zero retries or training updates. All initial
inputs, API-selected actions and geometric outcomes passed audit. Preserve its
completion.json. Current paper selects 1,200 complete outcomes; historical
canonical reports retain their original 1,199-plus-unknown matrix. New experiment
defaults stay xhigh. This recovery is complete, not pending work to restart.

Single-policy and paper handoff completion (2026-09-18): all five full-task prior
models and all 300 new evaluations are complete and validated: ID 110/150, OOD
16/150, zero deployment API calls. Three coordinators and all 300 policy workers
exited cleanly. Preserve single_policy_astra_xhigh/completion.json and its
incidents/tray_design_http502/recovery_completed.json, including retained failed
request accounting. The full four-method evidence bundle is paper/REPORT.md and
Exp2_paper_bundle.zip; paper_bundle_validation.json checks its hashes, archive,
tables and links. There are 1,199 complete main-matrix outcomes and one historical
APPL 5.5 unknown. Execution/recovery instructions below are history, not pending
work to restart. For read-only report rebuilding use paper.build; preserve the
canonical experiment reports and their retained-incident accounting.

Single-policy resource scheduling follow-up (2026-09-18): within the same five-GPU
authorization, analysis/single_policy_evaluation_tail.py executes the thirty
unstarted original cells 120:150 early on GPUs 0/1/7, using the unchanged original
evaluation command/configurations. Original admission is held before pending
index 90 if these cells are still running, then resumed after their results,
videos and clean exits exist. Existing children continue. The original coordinator
later reuses these outcomes without repeating physics. This only reorders the
same 300 predeclared evaluations; no extra models, updates, API calls or physical
attempts. Preserve evaluation_tail allocation/process/control/completion receipts
and audit its thirty actual executions in addition to the other coordinator logs.

Paper handoff request (2026-09-18): after all single-policy training/evaluation
and audits finish, provide a complete English Exp2 report suitable for a paper
writing assistant. The evidence bundle is experiments/exp2/paper, with methods,
history, four-method tables/figures, exact API source/segmentation/prompt exports,
provenance and an archive. Do not label the bundle complete before all intended
new outcomes are validated; retain the older APPL 5.5 unknown separately.

Single-policy evaluation resource scheduling (2026-09-18): within the existing
five-device authorization, additional GPUs 4/6 evaluate the final 150 predeclared
cells via analysis/single_policy_evaluation_supplement.py. Original GPUs 0/1/7
continue the first 150. Resource-only configuration copies change only devices;
all models, seeds, sampling and scientific source remain frozen. Ten zero-update,
zero-physics B=1 checks compare actions against original deployment receipts before
additional physical evaluation. The original admission coordinator may be paused
before reaching the reserved range, leaving its active children running, and is
resumed after supplemental outcomes/videos and clean exits exist. Its original
completed-result path reuses those outcomes without repeating physics. Preserve
allocation, process, check and pause/resume receipts in evaluation_supplement.

Single-policy recovery authorization (2026-09-18): the user explicitly selected
"授权这一次受控恢复（推荐）" for tray_pack request 21 (HTTP 502).
Execute analysis/single_policy_tray_recovery_proposal.json once: repeat the exact
failed request, then finish the same unsubmitted candidate within the original
remaining 15 API calls, five interface checks and 80 tool calls. Preserve the
entire original attempt and unknown failed-request usage. The other four original
training jobs continue without repetition. After all five models pass checks,
freeze and complete the original 300 evaluations. Recovery is supervised by
analysis/single_policy_recover_tray.py; stop again on a new transport error.

Latest user authorization (2026-09-18): add the single-policy baseline in
SINGLE_POLICY.md. The user explicitly selected one API candidate per task.
Use GPT-6 Astra/xhigh to design one full-task inductive-bias diffusion policy from
the twelve original demonstrations, train 60,000 updates matching naive DP, and
evaluate directly without a runtime agent on the same 30 ID / 30 position-OOD
layouts per task. Preserve all three completed comparators, Exp1 and M0. New
framework and outputs are experiments/exp2/single_policy and
runs/exp2/single_policy_astra_xhigh. API owns prior/source; developer owns framework
and complete-trajectory preparation. No test-feedback design, automatic transport
retry, or manual rewriting of API output. Current prepared allocation uses idle
GPUs 0/1/7 with shared jobs, within the latest five-device maximum.

Follow-up authorization (2026-09-18): the user explicitly selected "授权这 21 项受控续跑（推荐）"
after two_block_sort / OOD / 30027 returned HTTP 502 at step 679. The previous
coordinator stopped with 68 complete outcomes, one interruption and 20 unstarted
cases; all 69 attempts and videos were audited. Execute the exact saved 21-case
followup_proposal_20260918.json: one additional reset for that interrupted cell
and the 20 unstarted cases, one attempt each. Use evaluation_followup_20260918
and followup_evaluation.py, preserving every prior attempt and frozen setting.
Start one episode alone, then at most four on GPUs 1/2/3/5; stop admission on a
new error. This is explicit bounded continuation, not automatic transport retry.
This continuation completed on 2026-09-18: all 300 new-study outcomes and videos
are validated, with 197 successes and 103 task failures. The final receipt is
M1_astra_xhigh/evaluation_followup_20260918/completion.json. Preserve every earlier
attempt and report; these recovery instructions are historical, not work to restart.

Service-restored resumption (2026-09-18): the user said "ok现在好了，继续吧，做完".
Resume the exact remaining 89 unknown Astra evaluation cells, including one new
reset attempt for buffer_swap / ID / 20122 whose authorized 2026-09-17 retest
failed at zero physical steps. Preserve that retest and the original 300 attempts.
Use evaluation_recovery_20260918 for the resumed round, unchanged gpt-6-astra/xhigh
and frozen scientific settings. Start one cell alone, then four concurrent cells
on GPUs 1/2/3/5. Retain fail-stop admission and no automatic transport retries.
Record this explicit resumption separately; completed task failures are not rerun.

Evaluation recovery authorization (2026-09-17): the user replied "ok，把实验做完"
to the exact 89-case proposal under M1_astra_xhigh/incidents/evaluation_outage.
Run one fresh-reset retest for each listed HTTP-interrupted evaluation cell,
retaining the original 300 attempts, frozen settings, model identity and costs.
The 211 completed cells, including 70 task failures, are not retested. Start one
zero-step interrupted cell alone; admit remaining cells only after a clean exit.
On any new execution/transport error, stop admitting further cells and retain
active outcomes. This permits one retest per listed cell, not automatic retries.
Outputs belong to M1_astra_xhigh/evaluation_recovery; no policy/prompt changes,
training updates, or Exp1/M0 changes accompany this recovery.

Explicit recovery authorization (2026-09-17): the user replied "允许恢复" to
the four-case proposal. Recover the three recorded Astra design HTTP interruptions
once each, charging previous calls against their original budgets. Continue the
existing buffer_red__h02 API history with at most eight additional API calls and
two additional interface checks after the recorded framework reload-gate repair.
Preserve original journals and API output. No further automatic retries are
authorized. Receipts are under M1_astra_xhigh/incidents; supervision is provided
by experiments/exp2/astra/finish_recovery.py. Complete the original training and
300-test matrix after all 36 policies pass their required checks.

Latest user amendment (2026-09-17): all future Exp2 Runtime API stages must use
`reasoning.effort=xhigh`. Other new experiments also default to xhigh; Exp1-related
work retains max. Audit actual request fields and preserve historical records.
The new additional five-task APPL study uses gpt-6-astra/xhigh throughout cuts,
heuristic/prior code generation and inference. Follow ASTRA_XHIGH.md; new runtime
outputs are M1_astra_xhigh. Reuse the original demonstrations, frozen DP results,
paired seeds and scientific framework. Newly authored policies are trained anew.
The latest resource amendment (2026-09-17) permits at most five simultaneously
active physical GPUs, with multiple jobs per GPU. The Astra pool is now 1,2,3,5,7
after checking device 7 occupancy. Preserve all unrelated jobs and earlier
four-device receipts. Resource-only configuration copies may add device 7;
scientific settings, source, API output and training budgets remain unchanged.

Latest scale-up authorization (2026-09-16): follow SCALEUP.md. Developer owns
the four additional native tasks and twelve demonstrations per task. Runtime API
alone owns cuts, heuristics, prior implementations, semantic handoffs and inference.
Compare DP and APPL on 30 ID plus 30 position-OOD layouts per task, including the
original drawer, with the same geometric goals and private 5000-step cap. Preserve
Exp1 and frozen M0. Archive the original 1500/3000 initial tests; latest initial
test and shared library are under runs/exp2/M1_initial_test. No test feedback into
design and no provider retries. Subsequent user resource amendment: physical GPUs
0–7 are allowed, multiple jobs per GPU, at most four GPUs active for Exp2 at once.
Inspect occupancy and preserve existing processes. Current scale-up pool: 1,2,3,7.
Earlier GPU 4–7 restrictions and device receipts below describe history.

Latest user amendment (2026-09-16): run a new inference revision using
configs/m1_v2_5000.json and runs/exp2/M1_v2/inference_5000. Reuse all frozen policies
and the same six seeds. The executor privately enforces 5000 physical steps;
API input must omit total/remaining episode budget. API selects policy, duration,
declarative numeric stop conditions and its own notebook. The framework checks
those literal conditions each physical step and measures feedback, without
inventing semantic thresholds or successors. Preserve full raw API history while
projecting complete unchanged exchanges for bounded request context. Private
API request limit: 128; per-invocation limit: 300. No retraining or transport
retry. Preserve both earlier rounds, including the interrupted 6202 record.

Preceding amendment (2026-09-16): the user authorized increasing M1_v2 evaluation
to 3000 physical steps. Reuse the frozen models via configs/m1_v2_3000.json;
new results belong to runs/exp2/M1_v2/budget_3000. Preserve the original 1500-step
configuration/results and all API submissions. Rerun the same six initial seeds,
report incremental evaluation costs and success before/after step 1500. No new
training or prior design is part of this budget change.

The preceding user authorization (2026-09-16) started M1_v2, described in PRIOR_POLICIES.md:
one full-training-demonstration normalizer, new API cuts with substantially larger
overlap, and API-authored handoff information per heuristic and policy. Execute that
workflow through src/appl/prior_policies. APPL_EXP2_REBUILD_AND_EXECUTE.md remains
historical context, not an instruction to restart its retired formal matrix.
This is the single active Exp2 project:
src/appl, experiments/exp2, environments/exp2, data/exp2, runs/exp2.
Old frameworks/ is archived at archive/frameworks/; never import archived code.
Data/assets/calibration inputs live in the real directory data/exp2, outside the archive.
Preserve Exp1 source/data/results/root lock and paper. Work on dev.
Use /home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked.
Every GPU process uses appl.gpu isolation and the latest allocation above; the
historical 4–7 restriction was superseded by the explicit scale-up resource amendment.
Candidate code and MuJoCo scene modelling decisions come from real Runtime API.
Developer implements baseline, training, contracts, controlled tools, diagnostics.
Each retained heuristic gets its own API code, prior document and checkpoint.
M1_v1 source, reports, API journals and traces remain; its checkpoints were deleted
under explicit user authorization. Keep the version/deletion receipts. M1_v2 uses
configs/m1_v2.json and runs/exp2/M1_v2, with no checkpoint or source reuse from v1.
Inference-time API policy selection uses prior/handoff documents and actual observations.
Framework-measured start/end/overlap states are evidence, not semantic gates.
New policy success uses only the agreed three geometric task goals; do not change
the frozen M0 evaluator or retroactively relabel its results.
No hidden-target feedback in design; no automatic provider retries.
Diagnose before formal runs. Failed diagnostic gates are reported, not waived.
Shared implementation bugs are fixed here with incident/version accounting.
