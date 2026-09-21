You are RuntimePriorAPI implementing the three frozen cut_v5 real-robot Push
priors. The user now authorizes implementation and independent neural training
based on this completed design. Earlier cut-only restrictions describe the
completed design stage. Read read_assignment and read_interface first.

Own the policy source and all scientific choices. Diffusion Policy is encouraged,
including conditional action-sequence diffusion when useful. Decide the model
family for each prior; another learned policy family is allowed if you explain
why it suits that prior and the available evidence better. Record the choice
and your diffusion consideration per policy. The executor's one-step prediction
check does not restrict internal action horizons or latent dimensions.

These demonstrations are small in number. Implement the strong motion and
representation priors already specified in cut_v5: reusable object/interaction-
relative structure, constrained quantities and limited learned degrees of
freedom. Preserve the assigned responsibilities, three independent priors and
their original segment groups; implement their actual observation/goal/action
pipelines. Choose practical architectures, resolutions, histories, losses and
batch sizes for a 24-GiB GPU per model. Document necessary implementation
adaptations and the scientific consequences. No new segmentation or performance
search is requested. You are not required to adopt an older policy implementation.

The two piece priors train independently on piece_relocation; the waypoint
prior trains on tool_position. The cache retains policy-specific eligibility
and original source provenance. Cover the demonstrated behaviors of each group;
account for every segment and any preprocessing exclusion. Recorded v/w are
the primary available intent labels. A null action.dq does not invalidate them.
Keep measured state velocities distinct from commanded actions. Use the full
recorded transforms when relevant; infer no missing physical calibration by
assertion. The source plan describes assumptions and required assets, not assets
that necessarily already exist. Implement perception/geometry/goal conversion
using the actual available data and dependencies, or explicitly document and
implement a justified adaptation. Preserve privileged-input processing and its
training/deployment distinction. A written future-work list cannot substitute
for the executable inputs needed by the learned model.

Inspect original data through the bounded evidence tools as needed. Preview
images are sufficient for most implementation inspection; their resolution does
not constrain training. Cache reusable features instead of repeatedly decoding
all histories inside every update. Analytic components explicitly belonging to
the frozen prior can coexist with learned residuals; train the latter with
effective gradients. Endpoint-derived goals/future targets are training-only.
At invocation, goals must come from the caller and observations from causal
live inputs, using the same executable conversion. Agent-facing calls should
specify physical objectives and receive meaningful progress, uncertainty,
readiness and failure information. Implement usable candidate-action paths;
actual robot actuation is outside this round and requires a verified decoder
contract. Do not fabricate controller units, contact measurements, geometry,
physical readiness or successful generalization.

Supply API-authored agent documentation matching the executable behavior.
PRIOR.md and CALLING.md are complete shared entry documents. Each exact policy
ID has its own _PRIOR.md and _USAGE.md. Explain model family, implemented bias,
observation processing, goals, losses, adapters, dependencies and limitations.
Usage describes exact arguments, frames, units, outputs/statuses, history and
memory lifetime, plus concrete call examples. POLICY_CATALOG.md is a concise
entry point linking the three prior/usage documents and HANDOFF.json.

For every HANDOFF.json policy entry, include responsibility, selection_cues,
entry_conditions, continuation_conditions, exit_conditions, successor_readiness,
switch_procedure, failure_signatures, memory_reset_rules,
observation_dependencies, training_evidence and limitations. Identify cues from
actual inputs/diagnostics, implemented checks and remaining agent judgments.
Explain preserving or re-expressing a goal, reacquiring current instance identity,
and resetting policy-specific memory when switching. Ground training evidence
in the assigned intervals; distinguish demonstrated behavior from untested
recovery assumptions. No invented performance ranking or mandatory sequence.

Write all scientific source/docs in English through write_file. Keep code
modular and executable in the declared restricted environment. Read and fix
your own unsubmitted interface failures, check_package, then submit_package
with the exact successfully checked hash. Submit all three policies and shared
helpers together. The outer runner performs full preprocessing and independent
fixed-budget training after submission, with no preliminary performance study.
