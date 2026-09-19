# Instructions for a paper-writing assistant

This is an evidence bundle, not a license to infer missing experiments.

## Read in this order

1. `REPORT.md`: final audited four-method comparison and interpretation.
2. `METHODS.md` and `FORMALISM.md`: executed task, learning, API and evaluation contracts.
3. `HISTORY.md`: development studies that must not be pooled into the main table.
4. `tables/`: machine-readable results, task/data descriptions, paired counts,
   model inventory and compute/accounting scope.
   `CLAIMS.csv` maps measured findings and interpretation limits to their evidence.
   `evaluation_attempts.json` distinguishes the 1,200 planned cells from 1,292
   retained reset attempts, including service interruptions and authorized resets.
   The last historical APPL 5.5 unknown is resolved by the user-authorized
   2026-09-19 reset, documented in `recovery_20260919/REPORT.md`; all 1,200 selected
   outcomes are complete. Supporting historical reports retain their old counts.
5. `ARTIFACTS.json` and `POLICY_INDEX.csv`: exact source/provenance locations and
   API-authored prior documents, code, handoff information and checkpoints.
6. `figures/` and the linked replay indexes: use recorded examples with their
   stated selection rule and outcome. Never present a selected clip as a rate.
7. `execution_examples/index.html`: forty portable original videos of the first
   predeclared seed per task/condition/method. APPL examples include exact
   invocation records, notebooks, initial API requests and explicitly labeled
   public tool-call extracts. No outcomes were used to select these examples.
8. `supporting_reports/README.md`: unchanged historical and diagnostic reports
   included for offline checking. Their original paths are in the adjacent index.
9. `framework/README.md` and `protocols/index.json`: exact framework/environment,
   specification and configuration copies for checking implementation details.
   This is a source-reading bundle, not a standalone simulator installation.

The bundle is ready for writing only when `completion.json` says `complete` and
its file hashes verify. Before that, methods/history are preparation drafts and
the new baseline results are pending. The build script fails rather than filling
missing outcomes with zeros or invented measurements.

## Suggested manuscript structure

- Motivation: long-horizon imitation from a small fixed demonstration set.
- System: API-owned segmentation, prior code and semantic handoffs; bounded
  developer-owned tools, training, observation feedback and independent success.
- Setup: five tasks, twelve demonstrations, four methods, one training seed,
  thirty paired ID and thirty paired position-OOD layouts per task.
- Main results: all task-level rows, aggregates, unknowns and paired comparisons.
- Additional baseline: what offline prior design achieves without a runtime API.
- Failure analysis: measured grasp/goal/termination evidence and selected traces.
- Costs and limitations: parameter/update differences, API calls, token scopes,
  transport resets, small sample size, seed reuse and geometric success semantics.
- Appendix: exact prompts/interfaces, policy index, reproducibility and development
  chronology. Link raw artifacts rather than silently rewriting API prior claims.

## Claims that require restraint

- Do not claim equal compute: full-task methods have equal optimizer updates,
  while architectures differ and APPL has more models and aggregate updates.
- Do not label SinglePrior versus APPL a clean agent-only ablation. Decomposition,
  policy count, data windows and training compute also differ.
- Describe the naive full-task DP actually tested here. The results do not show
  that diffusion policies generally cannot perform long-horizon tasks; APPL and
  SinglePrior also deploy learned diffusion policies.
- Do not attribute APPL 6 versus 5.5 differences solely to xhigh or model family.
- Do not call later paired follow-ups untouched held-out tests. Earlier layouts
  had been analyzed, although test evidence was not supplied to the design API.
- Do not treat API interruption as task failure, or repeated reset recovery as
  continuation from a saved simulator state. Keep every attempt's cost accounting.
- Do not treat successes by step 1500/3000/5000 on one trajectory as three separate
  controlled budget experiments.
- Do not claim release, settling, stability or real-time robot execution. Success
  requires simultaneous geometric predicates; simulator time pauses during API calls.
- Do not claim a 30-layout confidence interval captures training-seed or repeated
  API-generation variance. There is only one trained replicate per model.
- Do not turn a described prior into a proven mechanism, or a failure correlation
  into a causal explanation. Read the API source and retain the stated limitations.
- Do not report hypothetical direct-API cost as an actual invoice. Cached tokens
  are included in input tokens; failed requests can have unreported usage.
- Do not imply Exp1 was rerun, or that an obsolete formal M2 plan was completed.

## Ready-to-use handoff prompt

> Read this bundle's completion receipt, REPORT.md, METHODS.md, HISTORY.md,
> tables and artifact index. Draft the Exp2 methods, results, analysis and
> limitations sections in English. Use only verified numeric results; preserve
> task-specific regressions, unknown outcomes and protocol distinctions. Attribute
> scientific priors to the Runtime API and framework/preparation to the developer.
> Distinguish descriptive evidence from causal claims. Cite exact repository
> artifacts for each quantitative claim and list any additional experiments that
> would be needed to support stronger conclusions. Do not invent missing runs,
> benchmarks, baselines, statistical replicates, references or model capabilities.
