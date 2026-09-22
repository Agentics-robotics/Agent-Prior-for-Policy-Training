You are designing a robot learning system from a small set of demonstrations.
Retain the object-relative and interaction-relative representations, strong
motion priors, and complete behavior coverage required by the task specification.

## Supplied execution capabilities

The deployment system provides motion planning and low-level control. Given
appropriate geometric objectives, constraints and current robot/scene state,
these components can generate and track connecting robot motion within the
domains described in the supplied capability contract. They report completion,
execution feedback and planning failures. Inspect that contract to establish
what they solve and what task-dependent information they need from learning.

Design the learned component around the decisions that remain after using
these capabilities. Its output should provide the smallest sufficient,
physically meaningful specification for the executor to make task progress.
The executor supplies intervening motion within its supported domain. Choose
the output variables, representation and decision frequency to suit this
division of responsibility and the available training evidence.

## System-level coverage and learning scope

The complete system must cover the demonstrated behaviors needed for the task.
Assign that coverage jointly to learned decisions, supplied planners/controllers
and observable execution logic. Record each component's responsibility and
the information passed between components.

Concentrate the limited demonstrations on task-dependent choices and interaction
behavior that benefit from learning. Use shared conditional models when the
same decision structure applies across objects, locations and repeated events.
Choose the model count from these remaining learning problems and explain the
distinct responsibility of each learned mapping.

Spatial goals, geometric constraints and local motion specifications are valid
learned outputs. Choose how much future intent each prediction specifies, when
it remains committed during execution, and when fresh observations trigger
another decision. Dense command prediction is available where supported by
the learning problem; sparse decisions can delegate intervening motion to the
supplied execution components.

## Demonstration organization and training targets

Organize the demonstrations around the resulting learned decisions and their
execution context. For each source interval, identify its role in learned
supervision, executor responsibility, observation context or outcome evidence.
Retain original recordings, timestamps and complete provenance for all roles.

Define the observation and goal available when each decision is made, its
training target, and the outcome or future observations used only to derive
supervision. Gather equivalent decisions across different robot paths and
scene configurations when they share the same task-relevant structure.

Account for planner-generated arrival states when specifying online inputs,
label eligibility and reacquisition of current observations. If execution
cannot realize a proposed specification, expose that feedback for a subsequent
decision. Separate the observed effects of an interaction from the commanded
robot motion when defining targets and auxiliary supervision.

## Required design outputs

Publish the system responsibility map, the residual learning problems, each
learned output schema and its semantics, the executor information requirements,
data grouping and target derivation, structural generalization priors, and the
closed-loop invocation protocol. For every learned output field, identify the
decision it communicates and how the supplied executor uses it.

Support the design with inspected demonstrations and the declared execution
capabilities. Document frames, units, timing, physical referents, relevant
calibration, online availability and observable completion conditions. Specify
how training and online inputs retain the same meaning, and how related source
records remain grouped during later evaluation.
