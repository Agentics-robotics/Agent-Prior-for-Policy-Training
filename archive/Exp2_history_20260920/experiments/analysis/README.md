# Scale-up execution analysis

The scientific executor is frozen under `src/appl/scaleup`; the study and all
policy/checkpoint hashes are in `runs/exp2/M1_scaleup/study_freeze.json`.

| Entry | Purpose |
| --- | --- |
| `audit_scaleup.py` | Read completed traces and API journals; verify paired resets, geometric success, frozen policy selection, unchanged API arguments and literal per-step stop conditions. |
| `scaleup_execution_audit.json` | Latest audit snapshot. `all_600_audited` requires completed outcomes; `all_600_attempts_audited` also permits explicitly retained HTTP-interrupted prefixes whose final outcomes remain unknown. |
| `live_status.json` | A dated queue snapshot; authoritative live queues remain under `runs/exp2/M1_scaleup/batch`. |
| `api_model_audit.json` | Read-only counts of requested model, reasoning effort and returned model across all current segmentation, prior-design and deployment journals, plus retained historical rounds and the Exp1 report comparison. Exp2 records use gpt-5.5/high; Exp1 records use gpt-6-astra/max. Backend weights behind the local proxy are not independently attested. |
| `first_pair_diagnostics.json` | Read-only measurements for the first declared sorting ID seed, 20000. This illustrative pair is not an aggregate estimate. |
| `first_ood_diagnostics.json` | Original training support and recorded gripper commands for the first sorting position-OOD seed, 30000. |
| `training_gripper_support.json` | Finger-joint-width support in all twelve original demonstrations for each of the five tasks; no <=5 mm states occur. This is a state-support measurement, not a contact classifier. |
| `audit_naive_equivalence.py` / `naive_equivalence_audit.json` | Compare all four new baseline datasets/configurations and final-EMA DDPM100 samples with the original M0 implementations, using only original training observations and CPU computation. |
| `analyze_outcomes.py` / `outcome_diagnostics.json` | Exploratory measurements of completed trials: first/final geometric subgoals, object heights, finger widths and actual normalized action inputs. The output explicitly marks partial coverage. |
| `audit_frozen_inputs.py` / `frozen_inputs_audit.json` | Hash all original and sliced training inputs; verify all 300 predeclared paired layouts remain in their ID/OOD ranges and do not intersect static CAD volumes. |
| `protected_preservation.json` | Compare 85,576 protected Exp1/M0/source/environment files with the pre-existing preservation manifest; no modifications found. |
| `late_drawer_success.json` | Invocation timeline for drawer OOD seed 6406, successful at step 3492. This example was selected after observing its late success; aggregate budget comparisons use all planned trials. |
| `export_first_pairs.py` | Export unchanged original camera frames for the first predeclared ID/OOD seed of each task, with a side-by-side browser at `runs/exp2/M1_scaleup/paired_examples.html` and a frame/video provenance record. |
| `finalize_scaleup.py` | Wait for all fixed dispatchers and require all 600 attempts to be accounted for; audit completed outcomes and retained HTTP-interrupted prefixes, frozen models and all available original-frame videos. Its completion receipt explicitly distinguishes unknown outcomes. It launches no scientific attempts. |
| `summarize_scaleup.py` | After the completion audit, derive `ANALYSIS.md` and `analysis_summary.json` from the original matrix: same-trajectory budget prefixes, termination counts, measured state support and separate scientific/API cost accounting. It creates no new trials. |
| `transport_incident_20116.json` / `transport_retest_proposal.json` | The retained temporary HTTP 502 at buffer-swap APPL ID seed 20116 and a proposed single fresh-reset supplemental retest. The proposal is not authorization or an executed retry. |
| `finalization_revision.json` | Exact previous postprocessor/auditor source and the reporting revision prompted by the observed HTTP interruption. Scientific executor source remains frozen. |
| `supplement_scaleup_dp.py` | Resource-only scheduling of a contiguous subset of already planned DP trials. It calls the unchanged frozen evaluator and retains existing attempts without retry. |
| `resource_versions/` | Exact earlier resource-helper source, retained by hash. |

All commands use the locked Exp2 Pixi environment. A read-only audit can avoid
environment installation locks and write only inside the repository:

```bash
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked --no-install python experiments/exp2/analysis/audit_scaleup.py --output experiments/exp2/analysis/scaleup_execution_audit.json
```

The auditor reads journals only after their episode processes exit and verifies
that no uncheckpointed SQLite WAL remains before opening them in immutable mode.
It does not change original journals, API outputs, models, traces or schedules.

The supplemental dispatcher shares exclusive episode/job directories with the
main matrix. Its assignments are recorded in the `supplement_DP*` plan files
under `runs/exp2/M1_scaleup/batch`. The second queue waits for the first to finish.
The third queue covers the original seed indices 14–11 and takes a slot only
after the second queue has no pending work and has released that specific slot.
The two overlapping dispatchers therefore share at most two extra DP slots per
GPU; their pool remains physical GPUs 1, 2, 3 and 7. Exact preceding dispatcher
source is retained under `resource_versions`; the runtime allocation receipt is
`runs/exp2/M1_scaleup/allocation/dp_slot_handoff.json`.
