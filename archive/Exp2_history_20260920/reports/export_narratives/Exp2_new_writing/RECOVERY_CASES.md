> **Compact writing companion.** The report below describes the full evidence package. This companion retains reports, results, all 60 numerical demonstrations, 255 videos, submitted policy source/priors, and the two recovery-case traces/responses. Complete per-step traces and full API design/deployment journals are in `Exp2_new_paper_bundle.zip`. Read `COMPANION_SCOPE.md` for exact scope. Scientific API files are unchanged.

# Two observed API-selected recovery sequences

These cases are selected from the 75 primary OOD episodes. They are robot-task recoveries within one continuous physical episode, not service retries, resource resumption, or resets. Their original API arguments match the execution records. Numerical evidence uses the saved physical trajectory. Both remain successes under the frozen geometric evaluator.

The cases demonstrate recognition, a documented policy choice, changed physical behavior, and eventual task success. They do not establish what would have happened under forced continuation of the original policy, nor a recovery-rate denominator.

## A. Unstack and sort, OOD seed 30204

**Failure:** repeated blue approach without finger closure, after red has already been placed.

| Physical step | Measured event |
| --- | --- |
| 360 | Red-at-goal first becomes true. |
| 396–596 | `acquire_lift__h02` attempts blue acquisition. |
| 596–706 | `carry_place_clear__h01` also attempts it. Across both attempts, aperture remains approximately 0.080 m; blue never rises above 0.02355 m. |
| 706 | API chooses `carry_place_clear__h02`, having read its prior and handoff documentation. |
| 707 | Aperture decreases from approximately 0.080 m to 0.04128 m. Blue is still table-supported. |
| 724 | Continued execution with the same policy raises blue center to 0.12037 m; measured aperture is approximately 0.03653 m. |
| 859 | Both target predicates hold; the evaluator ends the episode successfully. |

Original API choice rationale:

> Two policies remained open despite approaching blue; carry h02 has a genuinely different separate gripper/event mechanism and supports this low open blue pickup overlap.

The referenced [submitted prior](policies/APPL_6_xhigh/unstack_sort/carry_place_clear__h02/source/PRIOR.md) implements learned arm/gripper branches and event-related supervision within the fixed diffusion interface. It is not an executor-scripted closure rule. After the first closure, the API explicitly says that closure alone is not yet a verified grasp and continues to seek actual object rise.

- [Original switch response, invocation 9](episodes/APPL_6_xhigh/unstack_sort/OOD/30204/api/api/0013.response.json).
- [Original continuation response, invocation 10](episodes/APPL_6_xhigh/unstack_sort/OOD/30204/api/api/0014.response.json).
- [Full invocations](episodes/APPL_6_xhigh/unstack_sort/OOD/30204/invocations.json), [trajectory](episodes/APPL_6_xhigh/unstack_sort/OOD/30204/trace.jsonl), [audit](episodes/APPL_6_xhigh/unstack_sort/OOD/30204/audit.json), [result](episodes/APPL_6_xhigh/unstack_sort/OOD/30204/result.json).
- [Replay video](episodes/APPL_6_xhigh/unstack_sort/OOD/30204/replay.mp4).

## B. Buffer exchange, OOD seed 30102

**Failure:** blue is held near its destination but remains high instead of being lowered.

| Physical step | Measured event |
| --- | --- |
| 484–704 | `transfer_blue__h01` transports blue toward the target region but remains high. |
| 704–784 | Another 80 steps with h01 keep blue at 0.2924–0.2978 m. |
| 784–884 | API tries `transfer_blue__h03`; blue remains high, ending at 0.28539 m. |
| 884 | API switches to `finish_red__h01`, whose documented overlap includes preceding blue descent. |
| 893 | Blue descends to 0.2001 m, with close TCP association and maintained aperture. |
| 912 | Blue center reaches 0.02877 m; blue-at-goal becomes true. |
| 1,245 | The same policy subsequently completes red retrieval/placement; both goals hold. |

Original API choice rationale:

> Both transfer controllers plateaued high despite secure observed carry. Finish H01 explicitly learns blue descent and uses observed goal-height margins, so it can plausibly initiate placement from this aligned held state.

The API notes that blue is higher than the measured training start range and calls the takeover an extrapolation. It then observes real descent and continues. The [submitted prior and handoff discussion](policies/APPL_6_xhigh/buffer_swap/finish_red__h01/source/PRIOR.md) explicitly includes blue setdown/opening before red retrieval. This is a use of overlap semantics rather than a rigid interpretation of the skill name.

- [Original takeover response, invocation 7](episodes/APPL_6_xhigh/buffer_swap/OOD/30102/api/api/0011.response.json).
- [Original continuation response, invocation 8](episodes/APPL_6_xhigh/buffer_swap/OOD/30102/api/api/0012.response.json).
- [Full invocations](episodes/APPL_6_xhigh/buffer_swap/OOD/30102/invocations.json), [trajectory](episodes/APPL_6_xhigh/buffer_swap/OOD/30102/trace.jsonl), [audit](episodes/APPL_6_xhigh/buffer_swap/OOD/30102/audit.json), [result](episodes/APPL_6_xhigh/buffer_swap/OOD/30102/result.json).
- [Replay video](episodes/APPL_6_xhigh/buffer_swap/OOD/30102/replay.mp4).

## Publication figure

![Measured recovery traces](figures/recovery_evidence.png)

The dashed lines mark the relevant API-selected switch. Heights/apertures come directly from the recorded state, not from API prose. A height increase by itself is not a ground-truth attachment label. [PDF](figures/recovery_evidence.pdf) and [SVG](figures/recovery_evidence.svg) are provided for publication editing; original video speed is accelerated and should not be used to infer control latency.
