# Task specification: flip egg

## Intended task and evidence

The user requests implementation and training of the existing API cut/prior
design for `flip_egg`, including each proposed learned component and its interface.
The recording metadata describes the task as "flip the fried egg". Interpret
the intended physical outcome as turning over the demonstrated egg object;
inspect the actual scenes, tools, actions and outcomes to establish what this
means in the recordings. Do not infer material, temperature, food properties,
successful completion, or an obligatory phase sequence from the task name.

Use all 20 episodes listed in the source catalog under
`/home/storage/tianrunhu/real_robot_data/flip_egg`. Each episode is a separate
demonstration with its own timing. `complete` denotes recording completion,
not a verified task-success label. There are no supplied semantic cuts,
object masks/poses, contact labels, or verified flip-success annotations.
Distinguish visual inferences, proposed annotations and measured raw fields.

## Generalization and learned interfaces

Design reusable behavior for this task with changing observed initial states
and interaction progress, rather than replaying an episode or its timing.
Identify which variations are demonstrated and which are hypotheses needing
new data. No additional user requirement to handle different foods, tools,
materials, or arbitrary geometry has been established. State transfer
assumptions precisely without treating unobserved variations as tested.

The intended High-level Agent receives the task and observations, selects
callable policies or executors, supplies their task arguments, and monitors
progress and outcomes. Choose decomposition, policy/model count and sharing
from the evidence. Do not inherit Push-specific skills or scientific designs.

Make model inputs and outputs particularly concrete: raw source fields,
online causal preprocessing, tensor or geometric schema, dimensions, units,
coordinate frames, histories, caller goals, decision timing, target derivation,
output decoding/execution, uncertainty and validity rules. Distinguish
independently callable control policies from auxiliary perception models.
Describe a complete illustrative invocation sequence using the interfaces you
choose, and how progress, interruption and failure affect subsequent calls.

## Supplied execution capability: explicit design premise

For comparability with the current Push design round, carry forward its generic
motion-planner and low-level tracking capability as a design premise. Given
current robot/scene state, explicit reachable geometry objectives, constraints
and relevant tool/robot geometry, this capability plans and tracks feasible
connecting robot motion and reports completion, observed state, planning
failure or interruption. This premise is inherited from the user's stated
robot-system capability; no Flip egg planner binding or controller integration
has been verified in this task.

This capability does not itself supply task-dependent objectives, select an
object-interaction strategy, or establish reliable contact or flip dynamics.
The API must decide the remaining learning problems and the required adapter
contracts. State precisely any additional execution, perception, calibration
or tool-geometry dependencies. Do not assume they exist because they would
make a proposed decomposition convenient.

## Recording and deliverable boundary

Use the separate data contract and catalog for paired indices, fields,
missingness and source identity. Retain exact original arrays and media
references; do not timestamp-repair recordings or fabricate terminal rows.
Flip egg gripper feedback differs from Push: its recorded positions and widths
are finite. Numeric validity alone does not establish grasp/contact semantics
or calibration accuracy. Inspect actual observations before interpreting them.

## Current implementation stage

Implement and train the supplied existing cut/prior design using the authorized
recordings. The generic implementation workflow defines executable deliverables,
data preparation review and training. No verified annotations or additional tool
measurements have been supplied since cutting. Proposed prerequisites in the
design are not assets already acquired. The robot is not connected to this
training process. Earlier cuts and all original recordings remain unchanged.
