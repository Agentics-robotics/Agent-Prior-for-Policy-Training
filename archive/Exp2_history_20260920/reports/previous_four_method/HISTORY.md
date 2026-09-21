# Development history and separation from the main comparison

These studies explain the implemented system. They are not additional independent
replicates of the five-task comparison. Preserve their actual settings and outcomes.
Primary supporting reports are included in this bundle. Links beginning with
`../../../` refer to larger archives or runtime records in the original repository.

## Frozen M0 and diagnostic work

The retained drawer M0 uses twelve complete demonstrations, 60,000 updates,
DDPM100 and execution chunks of eight. Under its original 1,500-step evaluation
and stricter terminal criteria it achieved 2/5 development ID successes (seeds
6000--6004), followed by 4/15 independent confirmation successes (6005--6019).
The confirmation cases were not used to choose that retained setting.

An earlier normalization defect divided nearly constant quaternion components by
very small empirical scales; off-demonstration states could then reach roughly
700 in normalized coordinates. Fixed quaternion bounds addressed that numerical
problem. Later large normalized values can also reflect unseen velocities or
contact states; their magnitude alone does not establish another implementation
bug. Native action semantics, causal windows and demonstration replay were audited.

The archived setting study examined more data and alternative inference/control
settings. The 120-demonstration model obtained 5/15 confirmation successes versus
4/15 for the retained twelve-demonstration model. DDPM200, incremental actions and
shorter execution chunks did not establish a robust improvement. These small,
development-stage comparisons do not prove that more data or denoising steps
can never help. Their full positive and negative outcomes remain archived.

Sources: [M0 report](supporting_reports/M0_REPORT.md),
[M0 diagnostic archive](../../../archive/Exp2_M0DP/README.md),
[setting study](supporting_reports/M0_SETTING_STUDY.md).

## Preliminary APPL versions

An early endpoint disagreement came from the task description listing three
geometric goals while the original evaluator additionally expected terminal
release/clearance/velocity/hold behavior. The user clarified that the three
geometric goals should define completion for the new APPL work. The input and
evaluator were aligned, without retroactively changing frozen M0 scores. The
segmentation prompt also made per-trajectory progress and useful transition
overlap explicit. API-selected endpoints should be interpreted under their
actual goal contract, rather than called incorrect solely for omitting a tail
that the supplied goal did not require.

M1_v1 introduced independent API prior policies, but its five ID episodes all
failed. Handoff states could depart from demonstrations, and independently fitted
skill scales could magnify observations (about 59 at the observed extreme).
The next version fitted scales once on complete original demonstrations and
asked the API for substantially larger temporal overlap and explicit handoff
interfaces. These changes were made together; their individual effects were not
isolated. M1_v1 checkpoints were deleted under explicit user authorization;
source, API journals, metrics, reports and traces remain available.

M1_v2 trained eighteen drawer policies and evaluated one training-reset diagnostic
plus five ID resets. Its initial 1,500-step study yielded 2/5 ID successes. The
3,000-step reevaluation reused the same library and resets, with two ID successes,
two task failures and one context-limit interruption. These were reset episodes,
not continuations of the original physical trajectories.

The retained preliminary 5,000-step revision added API-authored numeric stop
conditions, within-invocation feedback, a notebook and bounded context projection,
and hid the executor's total/remaining step budget. It reused the same eighteen
models and obtained 4/5 ID successes; the separate training-reset diagnostic also
succeeded. Three ID successes occurred before 1,500 steps and one at step 1,529.
The remaining ID failure was an API voluntary finish at step 2,070. This was a
system revision, not a clean experiment changing only the physical-step budget.

The failure analysis identified a particularly useful same-prefix observation:
seed 6201 had exactly the same first 400 physical steps as its successful earlier
3,000-step test. The earlier API continued a drawer policy and made progress; the
new API switched policies and failed to recover. This supports the importance of
continuation and handoff decisions for that case, without proving a general
causal effect of any one prompt component.

Sources: [M1_v1 report](supporting_reports/M1_v1_REPORT.md),
[M1_v1 version/preservation](../../../runs/exp2/M1_v1/VERSION.md),
[archived 1500/3000 studies](../../../archive/Exp2_M1_initial_tests/README.md),
[retained 5000-step report](supporting_reports/M1_initial_5000_REPORT.md),
[detailed preliminary analysis](supporting_reports/M1_initial_5000_ANALYSIS.md).

## Five-task scale-up and additional baselines

The scale-up added four tasks with twelve demonstrations each and tested thirty
ID and thirty position-OOD layouts per task. It reused the drawer M0 checkpoint
and eighteen-policy library, while training four new naive DPs and 33 new APPL
policies. Thus the evaluated portfolios contain five naive DPs and 51 APPL 5.5
policies. All main-comparison methods use the same relaxed geometric predicates
and 5,000-step cap, including the reused drawer DP. Its new 15/30 ID result must
not be pooled with old M0's stricter 2/5 or 4/15 results.

The additional GPT-6 Astra/xhigh APPL round regenerated all five segmentations
and policy libraries, trained 36 new policies, and tested the same paired layouts.
An EMA reload-check correction aligned `requires_grad` flags during verification;
the retained evidence shows identical weights and zero action discrepancy after
that check-only correction. It did not change deployment sampling or candidate
source. The exact source revision and incident are retained.

API transport outages required explicitly authorized new reset attempts for
interrupted cells. Completed task failures were not rerun. All earlier attempts,
partial physical traces, failed requests and unknown failed-request costs remain
in the evidence chain. The final APPL 6 matrix has 300 complete outcomes. The
older APPL 5.5 matrix initially retained one unknown outcome; it was not relabeled
a failure. On 2026-09-19 the user authorized one same-setting reset of
`buffer_swap / ID / 20116`, whose original attempt was interrupted by an API
service error after 650 steps. The [audited recovery](recovery_20260919/REPORT.md)
supplies the current selected outcome. All 1,200 main cells are now complete;
the original interruption and preceding reports remain unchanged. The reset
retains historical GPT-5.5/high rather than changing that comparator's effort.

The last baseline adds one GPT-6 Astra/xhigh full-task prior policy per task,
with the same twelve demonstrations and 60,000-update budget as naive DP and no
API at deployment. All five API submissions completed. A tray-design HTTP 502
was recovered once under explicit authorization, preserving its exact consumed
conversation and unsubmitted source; three additional requests completed that
same candidate within the original budget. No second candidate was introduced.
The final report is issued only after all five models and 300 new outcomes pass
the recorded checks.

Sources: [scale-up specification](protocols/SCALEUP.md),
[Astra study and recovery chronology](protocols/ASTRA_XHIGH.md),
[single-policy specification](protocols/SINGLE_POLICY.md),
[single-policy design audit](../../../runs/exp2/single_policy_astra_xhigh/design_completion.json),
[tray continuation](../../../runs/exp2/single_policy_astra_xhigh/incidents/tray_design_http502/continuation.json).

## Interpretation rules

- Historical configurations and GPU allocations describe their executed runs;
  current authorization does not retroactively relabel them.
- The current main matrix comprises the four explicitly named methods. Retired
  M1/M2 plans are not evidence of a completed formal M2 experiment.
- Small preliminary results explain development, not statistical replication.
- Report negative results and interruptions alongside improvements.
- Keep model/effort changes, skill counts, training compute, prompt revisions,
  seed reuse and success-definition changes visible.
