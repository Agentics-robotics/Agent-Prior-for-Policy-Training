# Draft: controller-aware skill and action design

Superseded before execution by `planner_supported_learning_draft.md` after the
user clarified the learning/planning division on 2026-09-21. Retained as an
unexecuted discussion draft; do not combine its concrete skill examples with
the new discovery-oriented API prompt.

Status: proposal for a new design round, 2026-09-21. Not launched, not appended
retroactively to completed API requests. Preserve cut_v5, training_v3 and all
earlier scientific artifacts. This is user-guided design scope, not a submitted
Runtime API policy design or implementation.

## Objective and ownership

Build a reusable skill library for a small demonstration dataset. Jointly
design behavior groups, callable skills, observation/goal representations,
learned action variables and controller adapters. Runtime API owns the concrete
segmentation, scientific representations, priors, models and label conversions.
The developer supplies evidence tools, versioned interfaces, controller
capability descriptions and validation. The deployment owner supplies the actual
controller implementations and calibration. Use gpt-6-astra with reasoning
effort xhigh for any separately authorized new run.

The user's deployment inspection reports some appropriate behavior but also
lifting during intended planar pushing. Treat this as qualitative feedback,
not a diagnosed failure trace or a measured success rate. Use this feedback
to define explicit skill responsibilities and mechanically enforced motion
constraints in the next design.

## Controller capabilities as design inputs

Read a machine-readable inventory of the existing low-level controllers before
selecting output representations. For each controller, inspect its accepted
command type, controlled physical point, coordinate frame, units, timing,
feedback, maintained constraints, completion/failure conditions, robot/tool
dependencies and runnable interface. Distinguish available implementations,
documented but unverified contracts, and missing components.

Preserve demonstrated behavior coverage while deciding which variation should
be learned and which is already handled by a supplied controller. A callable
skill may combine a learned policy and a controller, or use a deterministic
controller directly where its supported objective needs no learned component.

## Task-space policy outputs

A learned policy may output controller-consumable task-space intent instead of
native EE or joint commands. Examples include planar tool-point velocity in a
local interaction frame, displacement over an explicit duration, a contact
choice with a local motion, a constrained speed/progress variable, or a spatial
subgoal. These examples are options; choose a representation justified by the
behavior, demonstrator evidence and available controller capabilities.

For each action space, specify:

- the physical point or object whose motion the output describes;
- the frame, units, dimensions, temporal interpretation and admissible domain;
- the quantities learned, supplied by the caller, or maintained by a controller;
- how corresponding input and output representations transform under a changed
  object position/orientation or robot configuration;
- the executable conversion to the selected low-level controller contract;
- the measurement, target and timing conventions shared by training and online
  inference.

If using contact motion, define whether this means the tool's material contact
point, a selected object material point, or a moving geometric contact location.
Tool-point and object-point velocities have different meanings during sliding.
An intended object motion requires a contact/dynamics realization mechanism;
a coordinate transform alone does not realize it. Specify contact identity and
continuity during an invocation, and explicit transitions when another contact
requires disengagement or repositioning.

## Structural motion priors and skill composition

Identify the task-relevant constraints for each skill from data and controller
capabilities. Encode constrained quantities in the action parameterization and
controller, while learning only the compatible remaining variation. Every
learned residual must remain in the declared admissible subspace.

For supported planar pushing, define the actual tool point and interaction
plane; make planar task motion the learned output while the execution contract
maintains the required relative height and tool orientation. Height correction
for tracking the plane is distinct from an intentional lift/reposition. An
invocation that needs another motion regime returns an explicit handoff request
to the high-level Agent, with the evidence and successor entry requirements.

For an approach-to-plane skill, define an explicit target signed distance from
the relevant surface and a measured arrival condition. Use controller feedback
to converge to that target while maintaining the complementary pose constraints.
Determine whether any speed, path or residual needs learning. The interaction
height comes from caller-supplied or estimated tool/object/surface geometry,
with explicit provenance and online availability.

The library collectively covers approach, interaction, corrections and
repositioning supported by the demonstrations. Runtime API decides the concrete
skill inventory and alternatives; there is no requirement to retain the old
three policy IDs or old two training groups in this new round.

## Data grouping, supervision and buffers

Revisit cuts and action-supervision eligibility to match the newly selected
motion regimes. Gather compatible occurrences across objects and trajectories.
Use overlap for context and stable handoffs with explicit roles. Place motion
outside a skill's output domain into another supported skill's training data;
retain its source/provenance and behavior coverage.

Keep original v/w, measured robot states, timestamps and images unchanged as
reference data. Derive learning targets in each skill's declared action space.
Record whether a target is a transformed command, measured realized motion, a
geometric estimate or a future-derived supervision label. Preserve the
distinction between recorded intentions and measured velocities.

Define the required tool-point transform, physical command interpretation,
surface geometry, temporal alignment and validity for each label conversion.
Missing calibration or a contact label has an explicit effect on eligibility
or on the proposed representation; it is not replaced by asserted ground truth.
Future observations may produce training labels, while deployed conditions must
have the same meaning and be available from current observations, past feedback
or caller-supplied goals.

## Executable action and handoff interfaces

Publish a per-policy action schema and executable adapter. The public interface
should preserve the task-space action, its controller identity, frame and units,
with decoded controller commands separately inspectable. Preserve original
native commands for provenance and relevant diagnostics. Training losses and
reload checks operate on the declared learned output and deterministic adapter;
the framework does not require every skill to regress all six original EE
components.

Make object goals, desired terminal tool state, arrival behavior, historical
sampling times and actual executed-action feedback explicit and consistent
between preparation and deployment. Provide observable readiness, progress,
completion and handoff states that match the executable controller behavior.
Include examples where the Agent acquires the interaction plane, invokes a
constrained interaction policy, and subsequently chooses continuation,
correction or disengagement/repositioning.

For every proposed structural prior, provide executable checks of the claimed
property and evidence of compatibility with its assigned examples. Examples
include admissible decoded motion, transform consistency, plane convergence,
target/feedback agreement between training and online conversion, and behavior
coverage across skill handoffs. These are interface/structural checks; report
generalization and closed-loop task performance only from separate evaluations.

## Submission requirements

For each skill, submit its purpose, dataset groups and exact source intervals,
local goal, learned action schema, maintained constraints, label derivation,
controller dependency and adapter, entry/completion/handoff contract, and
generalization rationale. Show which predictable variation was removed from
the learning problem and what variation remains supported by the data.

The actual controller capability manifest and the next-round executor/schema
must be prepared before using this draft for a new API implementation run.
