# Paired recording and published cut contract

This document describes the developer-owned data interface, not an API heuristic.
Review date: 2026-09-20. Scope: the two authorized Push recordings.

## Source identity

The source audit covers 16,745 paired records and 66,980 RGB/depth files
(8,309,230,726 bytes of media). All original files remain in their external source
directory. `runs/push_letters/cut_v3/source_manifest.json` pins SHA-256 hashes for
the numeric arrays, frame mapping, metadata, intrinsics, and every media file.

| Trajectory | Paired sample indices | Records |
| --- | --- | ---: |
| `episode_2026091922002701` | 0 through 9014 | 9,015 |
| `episode_2026091922062101` | 0 through 7729 | 7,730 |

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
| `gripper_position`, `gripper_width_m` | Push recordings contain -1 and NaN respectively; not valid gripper feedback |
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
not silently inferred. No verified object poses, masks, letter labels, target
words, or task-success labels are supplied by these recordings.

## Published dataset layout

Each API data group has `dataset.json` and `heuristic.md`. It can contain one or
multiple heuristics and any number of separately indexed original intervals.
Each interval gets its own `segments/NNNN/` directory:

- `records.npz`: every original field sliced over [start, stop), with exactly
  stop-start paired records and their original sample indices.
- `samples.json`: original source media paths/hashes and original frame metadata
  for those same samples. It includes `successor_source_index=stop` when a next
  source record exists, otherwise null; it never fabricates a terminal sample.
- `supervision.json`: unchanged API-authored semantic contract for this segment,
  including conditioning/label derivation, supervision and context boundaries,
  supervision exclusions, evidence indices, and merge/split justifications.

The API's [supervised_start,supervised_stop) range is nested inside the
materialized [start,stop) range. Rows outside it are context-only. Declared
supervision exclusions are disjoint subranges of that supervised interval;
retaining a raw row does not authorize computing its policy loss. The API must
inspect state evidence for all declared condition/boundary indices and inspect
images of the first/last supervised row. Boundaries may cite neighboring source
rows; condition-label anchors must remain inside the materialized sequence.
These structural checks do not establish semantic correctness.

The top-level policy_catalog.json and POLICY_CATALOG.md map API-authored
heuristic and policy IDs to their data groups and calling contracts. Multiple
independent policy proposals can share one dataset. No policy code or weights
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
