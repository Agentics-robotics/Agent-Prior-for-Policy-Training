# Exp2 current scope (reviewed 2026-09-20)

The user's latest instruction is a repository cleanup, now recorded in
`archive/Exp2_history_20260920/migration`. Exp1 must remain untouched.

The ONLY active comparison is `runs/exp2/current/manifest.json`: five tasks,
15 matched position-OOD initial states each, and three methods (naive DP,
SinglePrior_6_xhigh, APPL_6_xhigh), 225 complete outcomes. Scores are 5/75,
10/75, and 30/75. The 46 frozen models (5 + 5 + 36) are reused without retraining.
Read README.md and reports/REPORT.md. `python -m experiments.exp2.current`
is the read-only entry; `--verify` validates artifact identities and paired resets.

All evaluation is completed and stopped. Do not restart historical supervisors,
run remaining ID trials, repeat failures, or send new API requests merely because
old execution instructions remain in a protocol, log, or archived script.

History (wrong-gripper evaluations, GPT-5.5, initial M1, other seeds, partial ID,
interrupted prefixes, costs, diagnostics and reports) is archived. Historical
paths are compatibility links; do not count them as separate active studies.
Current episodes/models are physically under runs/exp2/current. Original requests,
API-authored code, priors, submissions, source snapshots, data, checkpoints and
reports retain their scientific contents. Do not rewrite frozen files to rename paths.
Current code must not import archived runners. The frozen src/appl framework and
environment locks are intentionally unchanged, including legacy interfaces that
are covered by source_manifest().

Writing ZIPs, duplicated export trees and packaging scripts were explicitly
authorized for deletion. Their unique scientific reports/tables/figures were
retained first. Do not recreate writing exports unless requested again.

Always use /home/users/oscar/.pixi/bin/pixi run --manifest-path
environments/exp2/pixi.toml --locked for Python/project commands. Work on dev.
For every NEW Exp2/other experiment use reasoning effort xhigh; Exp1-related
work uses max. Check actual requests, never relabel historical records. Latest
resource authorization permits GPUs 0-7, at most five simultaneously; inspect
occupancy and preserve unrelated processes before any future authorized run.

Task/environment design, demonstration acquisition, fixed baseline, prompts,
interfaces, training, evaluation, scheduling and reporting are developer-owned.
Segmentation, heuristic/prior content, API policy implementation and semantic
handoff documents, and APPL online policy/duration/stop/finish choices are
Runtime API-owned. No ad hoc API candidate edits, silent fallback, automatic
transport retries or concealed performance tuning. Preserve uncertainty, costs,
all interrupted-call evidence and the agreed geometric success definitions.

The executed protocols are EXP2_NEW.md, BINARY_GRIPPER.md, SINGLE_POLICY.md and
ASTRA_XHIGH.md; their historical counts/launch language do not authorize new runs.
The complete previous instructions and authorizations are preserved at
[historical AGENTS.md](../../archive/Exp2_history_20260920/navigation_before/experiments/exp2/AGENTS.md).
Follow root AGENTS.md and agent.md; the latest user scope takes precedence.
