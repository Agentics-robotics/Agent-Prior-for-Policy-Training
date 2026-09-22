# Paired recording and published cut contract

This document describes the developer-owned data interface, not an API heuristic.
Review date: 2026-09-21. Scope: independent Flip egg cut_v1.

## Source identity

The source contains 45,730 paired records in 20 episodes. All original files
remain under `/home/storage/tianrunhu/real_robot_data/flip_egg`.
`real_robot/runs/flip_egg/cut_v1/source_manifest.json` pins SHA-256 hashes for
the numeric arrays, frame mapping, metadata, intrinsics and every media file.
The data audit reports the verified media file count and byte count.

| Trajectory | Paired sample indices | Records |
| --- | --- | ---: |
| `episode_2026091916095801` | 0 through 1810 | 1811 |
| `episode_2026091916151401` | 0 through 2059 | 2060 |
| `episode_2026091916165401` | 0 through 1911 | 1912 |
| `episode_2026091916183001` | 0 through 1687 | 1688 |
| `episode_2026091916200101` | 0 through 2579 | 2580 |
| `episode_2026091916223301` | 0 through 2101 | 2102 |
| `episode_2026091916244201` | 0 through 2199 | 2200 |
| `episode_2026091916283401` | 0 through 1913 | 1914 |
| `episode_2026091916295601` | 0 through 3345 | 3346 |
| `episode_2026091916320101` | 0 through 2772 | 2773 |
| `episode_2026091916335001` | 0 through 2898 | 2899 |
| `episode_2026091916411901` | 0 through 2140 | 2141 |
| `episode_2026091916445601` | 0 through 2178 | 2179 |
| `episode_2026091916463201` | 0 through 2360 | 2361 |
| `episode_2026091916480201` | 0 through 2003 | 2004 |
| `episode_2026091916505401` | 0 through 2398 | 2399 |
| `episode_2026091916522501` | 0 through 2010 | 2011 |
| `episode_2026091916534901` | 0 through 2902 | 2903 |
| `episode_2026091916553801` | 0 through 1833 | 1834 |
| `episode_2026091916583301` | 0 through 2612 | 2613 |

`obs.npz.sample_index`, `frames.json.samples[].index`, and their robot/receive
timestamps were checked for exact agreement. Cuts follow those existing pairings.
Raw robot timestamps and camera-host timestamps are retained separately; there is
no cross-device timestamp matching, offset subtraction, or sample shifting.
Index consistency is not an exposure-level physical synchronization measurement.

## Fields and missingness

| Original field | Interface meaning |
| --- | --- |
| `q`, `dq`, `tau_ext` | Recorded robot state arrays; measured `dq` is not command `dq` |
| `T_base_flange`, `T_base_ee` | Original 4x4 transforms, unchanged |
| `gripper_position`, `gripper_width_m` | All 45,730 rows are finite; numeric ranges are in setup/recording_facts.json. Calibration and grasp/contact semantics still require verification |
| `action_json` | Original JSON strings with `type`, `v`, `w`, `spacemouse`, `gripper`, `dq`, `target_pose` |
| `t`, `recv_time` | Original robot and receiving-host times |
| `img_t_*`, `img_age_*` | Original recorded image timing fields |
| `sample_index` | Original paired sample identifier; never locally renumbered in raw cuts |

Raw NPZ cuts preserve dtype, shape, NaNs, and command JSON exactly. Numerical API
inspection rounds numbers to six decimal places and exposes nonfinite values as
null, with this transformation disclosed. This inspection formatting is not a
training-data transform. Null commands are never silently replaced with zeros.

The metadata records teleoperation at 20 Hz, sampling at 30 Hz, translation frame
`base`, rotation frame `ee`, and command mode `joint_velocity`. It does not by
itself establish the complete deployable controller contract. Before eventual
policy implementation/deployment, units, action timing and decoding must be
verified against the actual controller. Raw depth units and RGB registration are
not silently inferred. No verified object poses, masks, object identity labels, task-goal
annotations, or task-success labels are supplied by these recordings.

## Published dataset layout

Each API responsibility/data group has `dataset.json` and `heuristic.md`, under
the historical `skills` schema key. `component_kind=learned_policy` groups have
selected learned heuristics; `component_kind=supplied_executor` groups have an
empty heuristics list and zero learned policies. `execution_contract` specifies
each group's responsibility and executor binding. Dataset schema is v3.
Each interval gets its own `segments/NNNN/` directory:

- `records.npz`: every original field sliced over [start, stop), with exactly
  stop-start paired records and their original sample indices.
- `samples.json`: original source media paths/hashes and original frame metadata
  for those same samples. It includes `successor_source_index=stop` when a next
  source record exists, otherwise null; it never fabricates a terminal sample.
- `supervision.json`: unchanged API-authored semantic contract for this segment,
  including conditioning/label derivation, supervision and context boundaries,
  supervision exclusions, evidence indices, and merge/split justifications.

The API declares `supervision_kind` for each segment:

- `dense`: a nested [supervised_start,supervised_stop) interval provides decision
  anchors, minus disjoint supervision_exclusions. decision_indices is empty.
- `sparse`: a nested [supervised_start,supervised_stop) interval contains the
  explicit, unique decision_indices. Only those indices are supervised decision
  anchors; their targets may be derived from other retained observations. No
  selected decision may fall in supervision_exclusions. Unlisted rows are
  context/target/outcome evidence, with their role defined by the API.
- `executor_only`: supervised_start and supervised_stop are null;
  decision_indices and supervision_exclusions are empty. The group retains
  execution evidence and requires zero policy-loss rows or learned heuristics.

Learned groups use dense/sparse segments; supplied executor groups use
executor_only segments. Raw slicing is identical for all kinds. Rows are never
dropped or relabeled by publication. The API supplies the target conversion.

Inspect read_steps for every segment's materialized start/stop-1, all declared
condition/boundary evidence and every sparse decision index. Inspect images in
at least one camera at supervised_start and supervised_stop-1 for learned
segments, or materialized start and stop-1 for executor-only segments. Include
these two anchors in boundary_evidence_indices. Boundaries may cite neighboring
source rows; condition evidence must be inside the materialized sequence.
Inspect both camera views at least once per episode. These structural checks
do not establish semantic correctness or that all derived labels are usable.

Coverage reports count supervised anchors and executor records separately.
The retained historical context_only_unique field counts any assigned row
without a supervised anchor, including executor evidence. The narrower
context_without_executor_or_supervision_unique field excludes executor rows.
Overlapping group assignments are counted explicitly.

The top-level policy_catalog.json and POLICY_CATALOG.md map API-authored
heuristic and policy IDs to their data groups and calling contracts. Multiple
independent policy proposals can share one dataset. The execution catalog lists
all group responsibilities, including supplied executors. No policy code or weights
are produced by this publication stage.

Media references resolve against the explicit `source_root` retained in the
index. The original images are not resized or duplicated into each group.
Repeated use across groups is represented explicitly. A future loader must form
training windows within each segment and never treat two disjoint intervals as
a single continuous trajectory. The N paired-record representation is not the
old simulator's implicit observations[N+1]/actions[N] contract.

## API evidence views versus future policy inputs

The evidence tool creates only inspection previews: 640x360 bounding box by
default, preserving aspect ratio; original-resolution and original-pixel crops
are available at the API's request. Every view records its original SHA-256,
source index, crop rectangle, returned size and pixel-coordinate mapping.

Future policy input resolution is a separate API design choice. Any line/keypoint
detector, representation transform, observation history, goal relabeling,
augmentation, or action decoder must later be implemented by the Runtime API
with a complete input/output contract and necessary dependencies. Future-derived
training labels must not become required deployment observations. This cut stage
does not implement or validate those proposed mechanisms.
