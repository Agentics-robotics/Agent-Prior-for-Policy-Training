# Exp2 binary-gripper inference comparison

Authorized 2026-09-19 after the frozen-model and cross-method diagnostics.

## Completed result — reviewed 2026-09-19

All 600 physical trials, 600 videos and frozen-input checks passed. The supervisor
and all policy workers exited; no trials were retried. Runtime API calls and
learned-model updates are both zero. Peak concurrent physical GPU use was five.

| Method | Original ID → binary ID / 150 | Original OOD → binary OOD / 150 |
| --- | ---: | ---: |
| Naive DP | 17 → 76 (50.7%) | 0 → 7 (4.7%) |
| SinglePrior_6_xhigh | 110 → 139 (92.7%) | 16 → 24 (16.0%) |

SinglePrior achieves 30/30 ID on each of the four new tasks; drawer ID decreases
from 22/30 to 19/30. Naive drawer stays at 15/30, with six paired gains and six
losses. The intervention is not uniformly beneficial. Four successes occur after
1,500 steps; none occur after 3,000. OOD remains difficult.

Read the [writing guide](../../runs/exp2/binary_gripper_v1_20260919/START_HERE.md),
[full report](../../runs/exp2/binary_gripper_v1_20260919/REPORT.md),
[interpretation and failure counts](../../runs/exp2/binary_gripper_v1_20260919/ANALYSIS.md),
[final validation](../../runs/exp2/binary_gripper_v1_20260919/final_validation.json)
and [video browser](../../runs/exp2/binary_gripper_v1_20260919/replays.html).
The scope below records the completed protocol; it is not an instruction to rerun.

## Scope

- Reuse all five naive-DP and all five `SinglePrior_6_xhigh` checkpoints.
- Run each on its original 30 ID and 30 position-OOD seeds: 600 new rollouts.
- Change only final gripper decoding: `+1 if raw_action[7] >= 0 else -1`.
- Preserve seven arm commands with their original bound clipping, native
  controller, normalization, DDPM100, observation history 2, horizon 16,
  execution chunk 8, diffusion seed and geometric completion predicate.
- Stop at success or 5,000 physical steps. No other early stopping.
- No retraining, policy design, Runtime API calls or APPL evaluation.
- Preserve Exp1, frozen Exp2 source, API-generated source, checkpoints and
  original results. Resource-only configuration copies may allow GPUs 0–7.

This is a developer-authored execution correction, not API-generated policy
content. It addresses gradual partial gripper commands; grasp timing and narrow
intermediate-state coverage may still cause failures. The twelve demonstrations
per task are unchanged. Previously examined layouts make this a paired
post-study ablation, not a new untouched test set.

## Execution and evidence

Code: [binary_gripper](binary_gripper/). Output:
[`runs/exp2/binary_gripper_v1_20260919`](../../runs/exp2/binary_gripper_v1_20260919/).

Use the locked Exp2 Pixi environment to invoke
`python -m experiments.exp2.binary_gripper.run prepare`, then `supervise`.
The ten checkpoint/reference-action checks must pass before physical evaluation.
The supervisor inspects live occupancy, starts on idle GPUs 4–7, may add a fifth
idle GPU from 0–7, and never schedules onto a device with unrelated processes.
Multiple evaluation jobs per admitted GPU are permitted; at most five physical
devices are active. There is no automatic physical retry. Unexpected failures
stop new admission while existing workers finish, and retain their traces.

Each rollout saves the initial state, raw and executed actions, full physical
state trace, geometric predicates, completion result, video and validation.
API-authored policies execute in the existing restricted local worker with
network access disabled. Successful completion requires all 600 outcomes,
videos, paired initial states and checkpoint/source hashes to be validated.

The final report compares each corrected baseline with its own original scores,
including paired gains and losses and success by 1,500/3,000/5,000 steps. Both
APPL variants remain historical continuous-gripper results; they must not be
described as matched corrected-executor comparators.
