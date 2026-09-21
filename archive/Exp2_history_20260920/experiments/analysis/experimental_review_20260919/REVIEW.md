# Exp2 experimental review

Reviewed 2026-09-19. Developer-authored interpretation of existing artifacts;
not API-authored policy content. No new Runtime API calls, training updates,
simulator steps, policy edits, or changes to frozen results.

## Assessment

The experiment is a useful comparison of four executed systems on five related
state-observed manipulation tasks. Provenance, paired layouts, shared goal
predicates, frozen submissions and explicit interruption accounting are strong.
It does not yet isolate the effects of learned priors, decomposition, overlap,
runtime selection, model family or reasoning effort. A concrete handoff interface
issue and the weak full-task baseline deserve priority over further scale-up.

## 1. Confirmed handoff history loss

The frozen executor sends `reset=True` when a policy differs from the preceding
invocation (`src/appl/prior_policies/deploy.py:196`). The policy worker applies
`LoadedPolicy.reset()`, which clears both the action queue and observation
history (`engine.py:165`). The next call duplicates the current state to fill
the two-frame history (`engine.py:168`). Thus the incoming denoiser sees
`[s_t, s_t]`, even when the environment has an available real predecessor.

Clearing a stale action queue is appropriate. Discarding available physical
observation history removes the actual TCP/object displacement and co-motion
at takeover. Joint-velocity channels remain present, so this does not erase all
motion information. The first generated action prefix can last eight control
steps (0.4 simulated seconds), unless an invocation stop or success interrupts it.

The [read-only evidence](evidence.json) counts 1,867 noninitial policy switches
in the selected 300 APPL 5.5 episodes and 1,385 in the selected 300 APPL 6
episodes. Every episode includes at least one switch. These are exposure counts,
not counts of failures caused by history loss. The code matches the bundled
frozen implementation; the behavior affects both APPL methods and cannot by
itself explain the newer drawer regression.

For drawer ID seed 6302, the actual step 472-to-473 red and TCP z displacements
are both approximately +8.49 mm, immediately before the new system switches
from opening to red evacuation. The incoming duplicate frames discard that
backward displacement. No counterfactual actions or rollouts were generated.
Some API packages explicitly train for duplicated observations, which can
mitigate the effect but does not restore the missing measurement.

Recommended isolated follow-up: preserve the last two global physical
observations on a switch while clearing pending actions, with frozen policies
and a separately versioned executor. Do not overwrite the existing results.

## 2. Baseline competence is insufficiently explained

Naive DP succeeds on only 2/120 ID layouts across the four new tasks, despite
low-dimensional state input, narrow reset offsets and relaxed geometric goals.
That is a substantial warning sign for a paper-level baseline. Existing checks
establish alignment, normalization, action transforms and sampling equivalence
to M0 on inspected inputs; they do not establish closed-loop competence.

Each full-task model receives 60,000 batches of 128 windows, or 7.68 million
window draws. With roughly 9,000--13,000 available windows this is hundreds of
draws per window on average. Too few updates is therefore not the only obvious
explanation, although convergence and overfitting have not been isolated.

Useful follow-up diagnostics: complete-task execution on original training
resets; predictions on expert-state histories; failure location by manipulation
phase; a simple relative-feature/learned-condition baseline under matched data
and training budgets. These tests are not completed by this review.

### Clarification: this is independent of APPL switching

The naive evaluator repeatedly calls its single policy without resetting its
history inside an episode (`src/appl/scaleup/evaluate.py`). Duplicating the first
observation at episode initialization is ordinary boundary handling. The APPL
handoff issue above cannot explain naive-DP failures.

The additional [data and loss audit](naive_data_evidence.json) finds that all four
new-task baselines reached 60,000 updates. Their final logged mean epsilon losses
range from 0.000355 to 0.000496. These are training noise-prediction losses, not
sampled-action or rollout accuracy. Average window draws range from 603 to 880;
closed-command proportions are 49.2--50.2%, excluding a simple overwhelmingly
closed training-action-class explanation. Transition events remain sparse.

Of their 118 ID failures, 70 never raise red 3 cm above reset, 111 never raise
blue by that amount, and 112 have at least 100 consecutive narrow-width states.
These overlapping descriptive categories are not contact labels. They show
that many failures precede completion of basic object manipulation, so long
task duration alone is an inadequate diagnosis. Training-reset rollout
competence and expert-state conditional action accuracy remain key missing
measurements; implementation equivalence to M0 is not a substitute for either.

## 3. Demonstrator timing and the policy observation contract

The new-task collector holds close/open commands for twelve control steps and
adds path-refinement holds. A two-frame history spans one 0.05-second interval;
the fixed gripper dwell lasts 0.6 seconds. Very similar stationary observations
can therefore precede different expert actions, depending on hidden expert
program progress. This is a concrete reason to investigate action-target
ambiguity, not proof that the physical task itself requires an episode clock.

An API-generated drawer contact-memory heuristic proposed 0.5--1 second of
history. Its implemented prior explicitly acknowledges that the fixed interface
supplies only two frames, no previous-command channel and no persistent state.
A GRU over those frames does not implement that proposed long-duration memory.
The API dispatcher, meanwhile, retains prior invocations and a notebook. The
methods consequently differ in available temporal information as well as in
policy architecture. Compare longer-history/learned-memory policies before
attributing a benefit uniquely to language-model reasoning.

## 4. Compute and component effects are confounded

Across five tasks, naive DP and SinglePrior each use five models and 300,000
formal updates. APPL 5.5 uses 51 models/1,020,000 updates; APPL 6 uses 36
models/720,000 updates. Segmentation concentrates training on smaller datasets,
and overlapping actions receive multiple training exposures. Equal original
demonstration counts do not imply equal compute or effective sampling weights.

SinglePrior versus naive DP is informative about the complete offline design
package at equal updates, with architecture/FLOP differences. APPL versus
SinglePrior also changes decomposition, library size, memory and decisions.
GPT-6 versus GPT-5.5 additionally changes effort, cuts and learned policies.

Prioritized controls: vanilla skill DPs on the same frozen cuts with the same
dispatcher; fixed library with an API selector versus a declared fixed scheduler;
and a matched-budget comparison. Each addresses a different causal question.
Three heuristics per skill are not three independent training replicates.

## 5. Handoff documentation is not measured learned competence

The framework stores source-demonstration boundary/support statistics and
API-authored semantic handoff documents. Those are useful design evidence.
They are not success-rate measurements of the trained policy from each entry
state. Shape, gradient and zero-physics deployment checks also do not establish
grasping or recovery competence. A dispatcher can select a plausible documented
prior whose trained policy cannot execute the intended recovery.

Larger temporal overlap supplies transitions on the successful demonstration
paths. It does not supply arbitrary failed-grasp or disturbed-contact states.
Very broad overlap can also broaden skill objectives and increase repeated
training exposure. Its benefit and appropriate size remain unisolated.

## 6. Surprising outcomes and plausible interpretations

- SinglePrior sorting ID is 0/30, while buffer/unstack/tray ID are 29/30, 30/30,
  29/30. Task length or task-name complexity alone does not explain this.
  Examine grasp timing, target ambiguity, coordinate features and the particular
  generated candidate; avoid claiming a generic inability of single policies.
- Drawer APPL 5.5 ID is 29/30 versus APPL 6 at 8/30. The library changed from
  six skills/eighteen models to three/nine, halving its formal training updates
  from 360,000 to 180,000 while broadening skills. The recorded h03 interruption
  and subsequent blue-acquisition failures localize a problematic interaction;
  they do not isolate the effect of the API model or identify one sufficient fix.
- All 103 APPL 6 failures are voluntary API finishes before step 2,409. Hiding
  a 5,000-step cap is consistent with the chosen protocol, but is not a direct
  test of how well the same agent allocates a known budget. Unused steps are not
  proof that additional attempts would succeed; policies may lack recoveries.

## 7. Success semantics, OOD and statistical scope

The shared single-frame geometric predicate is legitimate as the declared
primary metric. It can count a last block while still grasped, and does not
establish stable release/settling. The buffer task does not require visiting its
temporary region, nor does final success enforce the demonstrated object order.
Descriptions of successful stable placement or mandatory task ordering would
overstate the measurement. A release/settling metric would be a new secondary
evaluation; existing traces often end at primary success and cannot reveal what
would happen afterward.

Action normalization uses per-coordinate training extrema and clipped DDPM
clean samples. This bounds sampled absolute joint targets to the learned action
box, which could restrict some OOD corrections. All methods share it; no evidence
here establishes that the selected OOD layouts are unreachable within that box.
Do not remove clipping or change action semantics within frozen comparisons.

Position OOD changes initial XY by a few centimetres around the same two-block
tasks. It does not establish transfer to new objects, visual observations,
controllers or dynamics. Initial-state ID remains a valid distribution label;
closed-loop deviations can nevertheless leave the demonstrated state support.

One training/design realization and one agent trajectory per layout do not
measure seed or API variance. Thirty layouts can reveal large observed gaps,
but near-equal counts should not be ranked confidently. Later studies reused
examined layouts, despite not passing test results to the design API. A fresh
held-out confirmation and additional independent seeds would strengthen claims.

## 8. Reporting consistency

The current result tables correctly include the 2026-09-19 recovery: APPL 5.5
has 181/300 successes. One executive-summary paragraph in `paper/REPORT.md`
still says 180 among 299 complete outcomes; that is the pre-recovery count.
The frozen/source report was not rewritten by this review. Any future report
revision should update its generating template and regenerate hashes/ZIP
together, preserving the preceding bundle.

## Suggested order of further work

1. Isolate the incoming-history issue using frozen models and separate results.
2. Diagnose naive-DP and SinglePrior sorting competence on training resets and
   independent development cases before broadening a hyperparameter search.
3. Add same-cut vanilla-skill and same-library scheduling controls to distinguish
   learned priors, decomposition and runtime selection.
4. Evaluate trained skill entry/exit competence and recovery support explicitly.
5. Add independent seeds/new held-out layouts and a declared stability metric.

These are recommendations, not newly launched experiments. Existing successes,
failures, policy sources, checkpoints and Exp1 remain unchanged.
