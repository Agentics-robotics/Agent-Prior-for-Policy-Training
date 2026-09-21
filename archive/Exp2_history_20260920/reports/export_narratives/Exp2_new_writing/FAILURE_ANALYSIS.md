> **Compact writing companion.** The report below describes the full evidence package. This companion retains reports, results, all 60 numerical demonstrations, 255 videos, submitted policy source/priors, and the two recovery-case traces/responses. Complete per-step traces and full API design/deployment journals are in `Exp2_new_paper_bundle.zip`. Read `COMPANION_SCOPE.md` for exact scope. Scientific API files are unchanged.

# Why APPL struggles on three tasks

Reviewed 2026-09-20. This is a retrospective, repository-authored analysis of the selected 75 OOD traces. It makes no policy, prompt, or evaluator changes and performs no new rollout.

## Evidence standard

**Measured:** action/state traces, first satisfaction of fixed geometric goals, termination status, physical step count, object height and measured finger aperture.

**API interpretation:** the original `reason`, `notebook`, and `finish` arguments. These show what the agent observed/inferred and why it chose a tool. They are not automatically accepted as causal ground truth.

**Hypothesis:** explanations such as phase ambiguity, insufficient support, history reset effects, or conservative stopping. These need controlled tests to determine their contribution.

Sources: [all episode facts](analysis/appl_ood_episode_facts.json), [failure rows](tables/appl_failure_facts.csv), [bottleneck summary](analysis/bottleneck_summary.json), and [all videos](VIDEOS.html). Every complete task failure is included; examples are not substitutions for the denominator.

## 1. Measured bottlenecks

| Task | Success | Main observed failure stages |
| --- | ---: | --- |
| Drawer exchange | 2/15 | 10 never reach red-on-pad; 3 reach red-on-pad but not blue-inside |
| Two-block sorting | 3/15 | 8 do not raise blue by 4 cm; 4 raise blue but never reach its goal |
| Buffer exchange | 4/15 | 4 do not raise initial red by 4 cm; 3 raise red but not blue by 4 cm; 4 reach blue's goal but fail final red placement |
| Unstack and sort | 11/15 | 3 never reach red's goal; 1 reaches red but not blue's goal |
| Tray packing | 10/15 | 1 never reaches red's goal; 4 reach red but not blue's goal |

The 4 cm measurement is **maximum object-center rise relative to that episode's initial object height**. It separates essentially unsupported tabletop motion from larger lifts in these traces, but does not certify grasp/contact. A drop after an earlier lift still belongs in the “rose” category. First-goal counts do not mean a goal remains true: later contacts can invalidate it.

### 1.1 Drawer: opening succeeds, object acquisition and preservation do not

- **15/15** episodes open the drawer at least once.
- **5/15** ever place red on the pad; **2/15** ever place blue inside.
- Among the ten failures before red placement, eight never raise red by 4 cm and two lift it but do not achieve placement.
- In **10 of the 13 failed episodes**, drawer-open is false again at termination, despite being achieved earlier. This is observable regression of a previous goal.
- All thirteen failures have blue-inside false, so their failure cannot be explained solely by the strict drawer-open threshold.

This localizes the dominant problem after initial drawer opening. Contact errors during red approach can shift the drawer and red block; subsequent controllers then encounter a different relationship between tool, handle, cavity, and object than the successful demonstrations. In three episodes red is placed but blue remains outside, while further recovery also loses drawer opening.

Examples include seed 6404 (no meaningful red lift after repeated approaches), 6402 (red rises but returns to the drawer), and 6403 (red goal reached, blue acquisition fails, drawer opening later lost). Their original traces and API explanations are under [drawer OOD episodes](episodes/APPL_6_xhigh/drawer_exchange/OOD).

**Interpretation:** opening/access is not the main isolated weakness. The combined contact geometry, off-nominal grasps, and preservation of an articulated fixture create additional ways to leave successful-demonstration support. This interpretation is consistent with the traces; it is not a separately measured causal contribution of each factor.

### 1.2 Sorting: the same nominal skill behaves very differently on the second object

All **15/15** episodes satisfy red-at-goal at least once. Only **3/15** satisfy blue-at-goal. Among twelve failures, eight never raise blue by 4 cm; four raise it but cannot place it correctly. Eleven failures still preserve red at termination, while seed 30013 later disturbs it.

Representative physical behaviors are empty closed-finger movement after missed blue pickup, approach toward the old source instead of a displaced blue block, transport to the wrong region, and stable grasp followed by high hovering without setdown. Seed 30000 ends with blue held roughly 0.30 m high; seed 30008 never meaningfully lifts blue; seed 30013 moves blue toward the red side and invalidates red's placement.

There is an interface limitation worth distinguishing from an implementation error: the shared acquisition/transport policy does **not** receive an explicit caller-provided “operate on blue” tag. The API can choose the policy, but its notebook cannot inject a target ID into the fixed 47-state policy input. The submitted role-aware priors must infer object role and phase from the causal observations. The [API-authored acquisition prior](policies/APPL_6_xhigh/two_block_sort/acquire_transport__h01/source/PRIOR.md) explicitly describes this adaptation and its learned red-then-blue role resolver.

**Interpretation:** the second pickup starts from a state produced by the first learned controller, rather than a demonstration reset. Modest placement, retreat, finger-state, or object-pose differences can affect which object/phase the learned controller infers. The traces provide concrete examples consistent with this hypothesis; they do not prove that adding a target tag alone would fix the task. SinglePrior's 4/15 versus APPL's 3/15 is also a reminder that extra selection is not automatically beneficial.

### 1.3 Buffer exchange: three acquisitions expose three distinct bottlenecks

The demonstrated sequence requires initial red pickup and buffering, blue pickup and placement, then red reacquisition and placement. Among eleven failures:

- **4** fail initial red acquisition: seeds 30104, 30106, 30110, 30114. Red does not rise by 4 cm. Their API logs describe repeated open approaches without closure.
- **3** raise red but fail blue acquisition: seeds 30101, 30108, 30111. Maximum blue rise remains under 4 cm.
- **4** reach blue's goal but fail final red placement: seeds 30100, 30105, 30109, 30112. The latter three fail to complete red retrieval/transport from the buffer. Seed 30100 brings red near its final target, but its rotated footprint remains just outside the boundary.

The third stage is particularly sensitive to the red pose actually left by the first stage. A buffer position is not a guaranteed canonical reset; it contains the previous policy's errors. The API can read this mismatch and try other controllers, but it cannot invent a new corrective motion outside the learned library.

Seed 30100 is a real precision-related exception: the API reports about 3.1 mm and 0.4 mm of boundary protrusion in two directions. It should remain a failure under the frozen contract. Its existence does not explain the ten other failed buffer episodes, most of which fail before final target placement. Loosening the target would alter the experiment and would not solve the observed missed grasps.

## 2. A shared issue: successful overlap does not supply general recovery coverage

All five tasks have twelve successful demonstrations. API-created overlaps can include predecessor ending, successor approach, closure, lift, and release. This helps some handoffs, as the [recovery cases](RECOVERY_CASES.md) demonstrate.

But an overlap around a successful grasp usually does not contain examples of a grasp that misses, fingers that fully close around nothing, an object pushed off-source or rotated, or a drawer partly reclosed during collision. Such states can have superficially similar tool positions but require different actions. Adding more successful-transition frames does not automatically label the correct corrective behavior there.

All three prior variants of a skill ultimately learn from closely related successful slices. Different network structures and losses provide diversity, but they can share a common failure on an unseen contact state. This explains how the API can correctly detect a failure and still lack an effective available action. It is a capability limitation of the combined trained library and selector, not evidence that the API did not run.

## 3. Why 5,000 steps did not resolve these failures

Every APPL failure is `agent_finished`, not `physical_budget_exhausted`. Their termination steps range from **418 to 2,663**, median **1,049**. The three lower-scoring tasks stop in the ranges 602–2,663 (drawer), 694–1,786 (sorting), and 418–2,568 (buffer).

![Termination steps](figures/termination_steps.png)

The original finish messages repeatedly state that alternate priors failed and the current state lies outside established successful-transition support. This is the API's judgement, not a proved impossibility. It may be sensible when retries worsen the state, and it may also terminate some recoverable episodes too early. **The current data do not determine that tradeoff.**

A larger hidden executor cap alone would not address the API finish decisions observed here. A policy that requires more recovery attempts, restricts voluntary finish, or trains explicit recovery support is a different setting. It should be tested on fresh or declared reused layouts, with retained attempts and costs, rather than retrospectively recoding current failures.

## 4. Why apparently harder tasks perform better

Unstacking preserves relative red/blue geometry under a common XY shift, while independently shifted two-object tasks alter both pickup locations separately. Its learned policy also receives an informative height difference between upper red and lower blue. These are plausible sources of more identifiable roles and a lower-dimensional position shift.

Tray packing and unstacking have their own independently API-designed priors, different nominal paths, handoff slices, and resulting trained functions. Their higher scores do not show that tray walls or stacking are intrinsically easier than two separate pads. The actual difficulty is the interaction of data support, role/phase inference, geometry, and a particular learned policy library. A task description's apparent simplicity does not control those factors.

The successful unstacking recovery explicitly uses a distinct gripper-event mechanism; this supports the possibility that useful learned alternatives are available there. It does not establish that this mechanism explains the task-level score difference. With fifteen layouts per task, rankings remain uncertain.

## 5. What the evidence does and does not say about bugs

The earlier continuous-gripper execution issue has a declared correction in all three compared methods. Saved raw and executed actions passed decoder audits. Binary execution still allows a learned policy to choose the wrong sign: an always-positive output remains open; an empty negative output remains closed. Those current behaviors should not be conflated with the previous decoder mismatch.

Shared full-demonstration normalization is used, so this is not the old per-skill normalization setup. Fixed success predicates and matched initial states passed independent episode audits. Most failures leave entire subgoals unmet; the evaluator alone cannot account for them.

This analysis does **not** prove the absence of every hidden implementation error. History padding on switches, learned phase/role behavior, finite two-frame memory, sampler behavior, and model capacity remain candidates for targeted tests. There is no current matched evidence that more DDPM steps, a larger network, more optimizer updates, or a looser terminal tolerance would eliminate the dominant failed acquisitions.

## 6. Hypotheses to test next, not experiments already performed

| Hypothesis | Evidence motivating it | Discriminating follow-up |
| --- | --- | --- |
| Narrow contact/recovery coverage is the main limit | Missed grasps create displaced/rotated/closed-empty states absent from successful demonstrations | Add a separately controlled set of recovery demonstrations; compare same data-count successful-only augmentation |
| Learned role/phase ambiguity hurts second-object control | Sorting succeeds on red in every episode but often fails on blue; no caller target tag | Compare an explicit target-conditioned interface against the frozen interface, with a declared new design/training round |
| Generated handoff states differ from demonstration handoffs | Buffer retrieval and post-red blue acquisition are bottlenecks | Evaluate frozen subpolicies from expert handoff states and matched learned handoff states |
| Voluntary API finish is sometimes premature | All 45 failures end before the cap | Matched controlled continuation/finish-policy ablation from saved failure states, with complete attempt accounting |
| Switching discards useful short-term history | History is reset and current state is duplicated at a switch | Change only history transfer under fixed frozen controllers and predeclared selection decisions |
| Final precision matters for a minority | Buffer 30100 ends near the boundary | Report a separately labeled threshold-sensitivity analysis without changing the primary success labels |

The strongest current conclusion is that **off-demonstration acquisition and handoff states limit the available learned controllers; online reasoning sometimes recovers, but cannot reliably compensate for missing low-level capability**. The relative contribution of data, conditioning, switching and stopping requires these ablations.
