# Round 3 native task source audit

This is an infrastructure audit by the `r3_factor_audit` Codex subagent. It is not
a policy proposal or a designer evidence bundle. The auditor read native task,
reward, and official expert source, and therefore must not be described as an
isolated D2-only policy designer. No model scores or locked test states were used.

The installed MetaWorld task files were byte-for-byte equal to the vendored
locked source below. Paths are relative to the repository; installed counterparts
are under `.pixi/envs/default/lib/python3.11/site-packages/metaworld/`.

## Registry and task semantics

`vendor_sources/metaworld/metaworld/env_dict.py:27,55,56,68` registers exactly
`assembly-v3`, `peg-insert-side-v3`, `pick-place-wall-v3`, and `stick-push-v3`.
Use these registry classes and goal-observable Task payloads, rather than
inventing environment IDs from informal task labels.

| Task | Source SHA256 | Native object and goal semantics |
| --- | --- | --- |
| assembly-v3 | `df2be328f466f8eb21181c3109ab399a612a11117acd1a9628728726aa9380ea` | Nut with a handle, placed around an upright fixed peg; not arbitrary six-axis precision assembly. |
| peg-insert-side-v3 | `27aae3fc1adb894493f649655d3515b71d2b1819c988513cadabf8d29ebbabf0` | Horizontal peg and independently placed fixed insertion box. |
| pick-place-wall-v3 | `723a32b049557b6daf3bd44c65527f185a4ff4ef2ed216bd06887516efdfe10e` | Object source and destination separated by a fixed wall. |
| stick-push-v3 | `043621a7e9c6cc31630e8d76d76bb7675f5c1f85527d26fbf34a985f54ac4bd2` | Grasp tool and push a sliding thermos toward its target. |

## Actual reset degrees of freedom

### Assembly

Source: `vendor_sources/metaworld/metaworld/envs/sawyer_assembly_peg_v3.py:27-32,57-62,116-129`.
The six-vector is `[nut_body_xyz, peg_goal_xyz]`. Native reset independently calls
the free-object position setter and sets fixed peg body position to
`peg_goal - [0,0,.05]`. The XY separation rejection threshold is .1 m.

The shipped nut bounds are **degenerate** at `[0,.6,.02]`; peg goal has
x in `[-.1,.1]`, y in `[.75,.85]`, and z exactly `.1`. The reset mechanism can
accept varied nut position through Task parameters, but that is a reset-range
extension and must be disclosed and physically calibrated. No mass, friction,
shape, controller, or post-reset relocation change is needed. Nut x and peg x
are independently controllable through this mechanism. Alternatively, peg x
and peg y are independent even within unextended native sampling support.

Initial infrastructure probes at nut x ±.08 and peg x ±.08, including crossed
pairs, succeeded with the official expert. These probes support investigating
that extension; they do not replace the required IID/C/E calibration.

### Peg insertion from the side

Source: `vendor_sources/metaworld/metaworld/envs/sawyer_peg_insertion_side_v3.py:45-50,76-85,136-151`.
The reset vector is `[peg_body_xyz, box_body_xyz]`, with source ranges
x `[0,.2]`, y `[.5,.7]`, z `.02`, and box ranges x `[-.35,-.25]`,
y `[.4,.7]`, z `[-.001,.001]`. Native reset uses both independently,
requiring source/box XY separation at least .1 m. The observation goal is
`box_body_xyz + [.03,0,.13]`, **not** the raw sampled box center.

Source x and box/goal y therefore provide two real independent factors without
altering native bounds. Source and target positions are not coupled by the
reset setter apart from the already satisfied separation condition.

### Stick push

Source: `vendor_sources/metaworld/metaworld/envs/sawyer_stick_push_v3.py:28-31,52-61,147-164`.
The nominal six-vector contains source and target xyz, but only their XY
components survive reset. The actual stick source z is fixed `.02` and target
z is taken from the thermos insertion site (observed `.132`). Sampled source z
`[0,.001]` and target z `[.1319,.1321]` are **not independent effective factors**.

Stick x `[-.08,-.03]`, stick y `[.58,.62]`, target x `[.399,.401]`, and
target y `[.55,.60]` are effective. Stick x and target y are suitable genuine
independent factors. The pushed thermos start is explicitly fixed by native
`obj_init_qpos=[0,0]`; do not claim independent randomized thermos placement.
The thermos moves on two native slide joints with limits ±.2 m; see
`assets/objects/assets/thermos.xml:3-4`.

### Pick-place wall

Source: `vendor_sources/metaworld/metaworld/envs/sawyer_pick_place_wall_v3.py:41-46,72-78,136-150`.
The six-vector is `[object_xyz, goal_xyz]`. Source x `[-.05,.05]`,
y `[.6,.65]`, z `.015`; goal x `[-.05,.05]`, y `[.85,.9]`,
z `[.05,.3]`. Both positions are truly used independently, subject to native
XY distance at least .15 m. Source x and goal x are independent factors; keep
wall geometry fixed and include it in the shared observation specification.

## Shared observation schema pitfalls

Base packing: `vendor_sources/metaworld/metaworld/sawyer_xyz_env.py:475-527`.
The 39 native values are current 18 values, previous 18 values, then goal xyz:
hand position `[0:3]`, normalized gripper separation `[3]`, object1 xyz `[4:7]`,
object1 quaternion `[7:11]`, object2 xyz `[11:14]`, object2 quaternion `[14:18]`.
Absent second objects are padded with zeros; zeros are not valid rotations.
Gripper separation is measured claw distance / .1 m clipped to [0,1].

* Assembly native object1 is the **handle site `RoundNut-8`**, while success
  concerns the ring center `RoundNut`. Its native quaternion is **WXYZ** from
  MuJoCo `body.xquat` (`sawyer_assembly_peg_v3.py:105-112`), unlike the other
  three new tasks' SciPy **XYZW** quaternions. The local handle offset from ring
  center is `[0,-.13,0]` (`assets/objects/assets/assembly_peg.xml:16-17`). Ring
  center can be derived from this static offset and declared quaternion, or
  supplied as an explicit shared geometric channel to B0 and all candidates.
* Peg-side object1 is site `pegGrasp`; success uses `pegHead`.
  The declared local offsets are grasp `[.03,0,.01]` and head `[-.1,0,0]`
  (`assets/sawyer_xyz/sawyer_peg_insertion_side.xml:16-18`). The object quaternion
  is XYZW of the grasp site's rotation. Any added head/axis channel must be
  common to all methods, or derivable from common pose plus these fixed offsets.
* Stick object1 is stick body COM and has XYZW rotation. Object2 xyz is
  `insertion_site + [0,.09,0]`; object2 quaternion is **all zeros**, reflecting
  the native observation convention (`sawyer_stick_push_v3.py:102-124`). Thus
  object2 xyz is the thermos axis point at the insertion height, not its body
  origin or the insertion site itself. Tool box half-extents `[.05,.02,.02]`
  and local end offset `[.05,0,0]` are specified in
  `assets/objects/assets/stick.xml:4-5`.
* Pick-wall object1 is `objGeom` position and XYZW rotation
  (`sawyer_pick_place_wall_v3.py:122-130`). Fixed wall geometry is additional
  static task information, rather than an episode-specific hidden phase.

Raw native assembly quaternion may be kept WXYZ with a precise schema or
converted consistently for every method; do not silently apply XYZW transforms
to it. Native reward evaluation must continue receiving its expected raw form.

## Native success and episode semantics

These criteria are task definitions, not policy inputs or scripted control laws.

| Task | Native success |
| --- | --- |
| assembly | Ring center XY distance to goal < .02 m and ring center z < goal z. Source `sawyer_assembly_peg_v3.py:162-170,219-231`. No orientation or gripper predicate is required by success, although reward has extra shaping. `info.obj_to_target` is always 0 and is unsuitable as a distance diagnostic. |
| peg-insert-side | `norm((pegHead - goal) * [1,2,2]) <= .07`, i.e. a scaled distance, not the distance of observed grasp point. Source `sawyer_peg_insertion_side_v3.py:117,174-181`. |
| pick-place-wall | Native object-to-goal Euclidean distance <= .07 m. Source `sawyer_pick_place_wall_v3.py:105` and its native reward distance computation. |
| stick-push | `norm(object2_xyz-goal) <= .12` **and** native grasp-success predicate: touching main tool object, gripper separation > 0, and stick z > initial stick z + .01 m. Source `sawyer_stick_push_v3.py:81-97`. Reaching target alone is insufficient. |

No new task requires a multi-step hold in native success. All four midpoint
resets and one subsequent zero-action step were verified non-successful.
Native `step` never returns `terminated=True` for ordinary success and returns
`truncated=True` at step 500. Taking another step raises; check success at every
executed action, stop on success/termination/truncation, and never reset the
episode clock when changing policy routing. Native simulator exceptions return
a stable observation and false success, so explicitly check the persistent
`_did_see_sim_exception` flag instead of treating that output as healthy.
Source: `sawyer_xyz_env.py:580-649`.

Observed dt is .0125 s (80 Hz), frame_skip 5. XYZ actions are normalized
increments scaled by .01 m by the native mocap controller; each component is
clipped to [-1,1]. The final action is gripper command, with positive values
closing. Retain actual clipped world commands as training labels.

## Native fixed-body translation and collision probe

Some native resets mutate fixed `model.body_pos`. To investigate the existing
Round 2 static-BVH concern without changing dynamics, the auditor compared
native mutation against a fresh MjSpec compile with identical moved fixed-body
position and free-body qpos. Assembly peg was placed at `[-.07,.83,.05]` and
peg-side box at `[-.25,.68,0]`. Each compared 324 free-object positions around
the fixed object, with 26 and 243 contact-positive positions respectively.
There were **zero contact-pair or penetration-distance mismatches** (distances
rounded to 1e-7 m). Assembly compiled BVH arrays changed; peg-box arrays did not.

This finite geometric probe supports using native **translation** reset and
does not establish a general invariance or justify arbitrary yaw mutation.
Restore canonical body/site/equality arrays and simulator state before each
native reset to avoid retained placement from earlier episodes. Any contrary
physical calibration result must be investigated before freezing layouts.

## Limited official-expert infrastructure probes

`artifacts/round3/source_audit_expert_probes.json` records 18 attempts, six per
assembly, peg-side, and stick-push. All used official expert `obs.copy()`,
actual world actions clipped to [-1,1], native success, and at most 500 steps.
All 18 succeeded, all started non-successful, and none had simulator exceptions.
Assembly included nut-source extension ±.08 m and crossed nut/peg pairs;
peg-side and stick probes included crossed source/target layouts.

These states are infrastructure probes, not demonstrations, not formal
IID/C/E calibration, and not dev/test snapshots. There were no learned model
actions, GPU training, or policy proposals. Official expert source was reviewed
for infrastructure only; its action rules must not enter designer bundles.

## Implemented Round 3 adapter spot audit

After `src/round3/environment.py` and `tasks.py` were written, an additional
48 infrastructure resets were exercised: four new tasks × IID/C/E × four
pairing-cell indices, with seeds 92500 through 92503, each followed by one zero
action. All passed native bounds, reset non-success, history/goal checks, and
finite-step checks. These are source-audit probes, not evaluation snapshots.
The wall has two coincident visual/collision geoms; both have world center
`[.1,.75,.06]` and half-size `[.12,.01,.06]`, matching the added six shared
observation values. Assembly compiled peg body is exactly native goal minus
`[0,0,.05]`; peg-side box is goal minus `[.03,0,.13]`. Assembly WXYZ schema is
explicit and its added center is the ring body origin. Peg-side E box y ranges
`[.425,.45]` and `[.65,.675]` stay within native `[.4,.7]`; source y E also
stays within native `[.5,.7]`. No corrective code edits were needed.
