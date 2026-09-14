# Round 1 environment and data semantics

Run all Python through Pixi. The task adapter is audited against MetaWorld
commit `6e01ad7e2ffb2302e4dca04f796fcd8837df8540`. Runtime source hashes,
field slices, control timings, and native position bounds are recorded in
`artifacts/observation_schema.json`.

The scalar native classes use the official public `Task` / `set_task` interface.
The payload matches MetaWorld's `_encode_task`: the environment class, a frozen
native `rand_vec`, and `partially_observable=False`. Native `reset(seed=...)`
ignores its argument at this commit, so the adapter calls `env.seed(seed)` before
`env.reset()`. All sampled placements are inside `_random_reset_space`.

The mechanism's actual initial base is `data.body("drawer" or "door").xpos`.
The observed interaction point differs: drawer uses `drawer_link` body COM plus
`[0,-0.16,0]` metres; door uses the `handle` geometry center. The door expert
subtracts 0.05 metres from its local handle x coordinate. It receives a copy so
this in-place operation cannot corrupt the stored observation. Neither method
adds this expert-specific adjustment to its input representation.

The verified observation dimension is 39: current 18 values, previous 18
values, and the current goal xyz. Each 18-value block contains hand body COM
xyz, normalized claw separation, handle xyz, its quaternion, and seven zero
placeholders for an absent second object. Drawer quaternion ordering is wxyz;
door quaternion ordering is xyzw. Both are preserved. Native reset repeats the
current block into the previous block. The two-step DP history retains these
native blocks in both representations. No zeros are treated as a real missing
object. There is no history-padding modification.

The relative mapping subtracts each block's own handle position from its hand
position, and the current handle from the current goal. Handle world position
and all remaining fields stay present. Its inverse adds the handle back. This
is performed before the normalizer, with no access to future observations.

Canonical reset restores the freshly constructed model's body/site/equality
arrays and calls `mj_resetData` and `mj_forward` before the public Task/native
reset sequence. Native task reset then derives its target and reward caches
from the requested base. This prevents a reused environment from retaining a
previous episode's model placement. Nothing is moved after native reset.
Records preserve Task parameters and reset seed, complete
`mjSTATE_INTEGRATION`, model body/site/equality arrays, target, native history,
path count, reward initialization quantities, and the original float64 reset
observation. The supported restore operation is this canonical Task+seed reset,
validated against snapshots and action replay; qpos/qvel alone are not a restore
mechanism.

`data/task_sampling_plan.json` freezes independent RNG streams and up to 1,000
training candidates per task before expert collection. Native ranges match the
specification. The pool accepts the first 100 successful demonstrations.
`default_rng(namespace+55)` selects 20 fixed pool indices without replacement.
Each development/IID/OOD split has 20 unique layouts, OOD split equally between
the two extremes. Every eval state is expert-prechecked, with failures retained;
the prescribed generated states are not selected by learned-model performance.
The first collection obtained all 100 demonstrations in 100 attempts for each
task, with every eval precheck succeeding.

Demonstration NPZ arrays are `obs[T+1,39]`, `actions[T,4]`, `rewards[T]`,
`success[T]`, `terminated[T]`, and `truncated[T]`. The actual float32 clipped
action passed to `env.step` is the label. Unclipped expert actions are diagnostic
only. Observations/rewards are stored as float32, flags as booleans, snapshots
as native float64, and embedded metadata as a Unicode JSON scalar. Each episode
has a companion JSON file with versions, layout, seed, outcome, and file hash.
The frozen manifest hashes both files. Evaluation NPZ files only contain reset
snapshots, with expert outcomes in metadata; they contain no demonstration
arrays that can enter training. Training and normalization may read only the
20 listed pool episodes.

`verify_manifest()` refuses missing or changed frozen files.
`audit_dataset()` checks every saved initial snapshot, replays every complete
demonstration using saved action labels, compares observations/rewards/flags,
and checks relative-coordinate round trips. Its independent artifact
`artifacts/dataset_audit.json` names the manifest hash and measured errors.

Source references:

- [MetaWorld task construction](https://github.com/Farama-Foundation/Metaworld/blob/6e01ad7e2ffb2302e4dca04f796fcd8837df8540/metaworld/__init__.py)
- [Native observation, action and reset implementation](https://github.com/Farama-Foundation/Metaworld/blob/6e01ad7e2ffb2302e4dca04f796fcd8837df8540/metaworld/sawyer_xyz_env.py)
- [Drawer task](https://github.com/Farama-Foundation/Metaworld/blob/6e01ad7e2ffb2302e4dca04f796fcd8837df8540/metaworld/envs/sawyer_drawer_open_v3.py)
- [Door task](https://github.com/Farama-Foundation/Metaworld/blob/6e01ad7e2ffb2302e4dca04f796fcd8837df8540/metaworld/envs/sawyer_door_v3.py)
- [Official expert trajectory documentation](https://metaworld.farama.org/benchmark/expert_trajectories/)
