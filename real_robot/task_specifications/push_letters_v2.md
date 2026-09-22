# Task specification: push letters into requested words

## Intended task

A robot must push physical letter pieces into an arrangement that spells a
user-requested word. The requested word, available letter instances, their
initial arrangements, and desired spatial layout can vary between requests.
The objective is to respond to the current request and scene, rather than
replay the spelling, object order, or motions in a recorded demonstration.

Inspect the demonstrations to establish what behaviors, objects, and outcomes
are actually supported. Do not treat folder names, inferred visible words, or
episode endpoints as verified per-sample labels or success annotations.

## Required generalization

Deployment will include completely different letters from those in the
demonstrations: both letter identities AND physical letter shapes can be
entirely unseen during policy training. Deployment will also include entirely
new target words absent from the demonstrations, and new initial arrangements.

This requirement is not limited to moving familiar letters to new positions
or recombining only the demonstrated inventory. Address unseen-letter identity,
unseen physical shape, and unseen word/layout demands explicitly. Do not
silently narrow the task to the recorded alphabet, a known font/template set,
or a demonstrated manipulation order.

The actual available inventory must support the requested word; repeated
characters require distinct physical instances. Recognizing an impossible or
ambiguous request is distinct from successfully manipulating a feasible one.

Separate semantic interpretation, current physical geometry, spatial goal
specification, motion generalization, and task sequencing. Decide the necessary
models and division of responsibility from evidence, without a prescribed
recognizer, geometric representation, architecture, or number of policies.

Do not equate accepting a new goal or character label with learning to
manipulate a new shape. Specify independent tests for unseen physical letters
and new words, including tests where these requirements occur together.
State missing assets and limitations without treating the required
generalization merely as an excluded future task. No new-shape success is
established by the existing recordings alone.

## High-level Agent and policy invocation

The intended deployment includes a High-level Agent that receives the user's
request and available scene observations, decides which policy to invoke,
supplies its task arguments, monitors the result, and chooses subsequent calls.
Design a callable learning-and-execution system for this arrangement.

Specify how the High-level Agent can direct the relevant physical instance(s)
and spatial objective through the proposed policy interfaces. The policies
need not correspond one-to-one to letters or to stages. No particular skill
decomposition, policy count, or allocation of low-level internal decisions
is prescribed.

The desired spatial layout must be explicit in the invocation pipeline.
Explain how positions, orientations, spacing, or equivalent goal information
are supplied by the user or proposed by the High-level Agent and represented
for policy use. A word string alone does not define a physical arrangement.
Choose representations and coordinate frames based on the available data and
declared calibration; do not assume unavailable metric ground truth.

The High-level Agent should have complete useful behavior coverage through the
learning-and-execution system. Multiple cut segments may form a shared dataset
group. Choose learned policies and their priors from distinct learning needs;
there is no requested number of policies or heuristic alternatives.

Do not silently replace High-level Agent selection with human selection of
each move, a hardcoded spelling sequence, or a fixed demonstrated trajectory.
If perception, initialization, calibration, or ambiguity resolution requires
human input, disclose that dependency and its effect on the intended autonomy.
Do not assume such inputs or annotations already exist.

## Supplied execution capability for this design round

The user declares that deployment provides a motion planner and low-level
controllers. Given the current robot and scene state, an explicit reachable
geometric objective, motion constraints and relevant tool/robot geometry, these
components can plan and track connecting robot motion, including movement
between task-dependent interaction locations within their feasible domain.
They can report completion, observed state, planning failure and interruption.
The learned output can therefore be an execution specification at a decision
time; a full robot trajectory at every recorded timestep is not required.

Treat collision-aware robot motion generation and tracking as supplied
capabilities. The learning system and High-level Agent must supply the
task-dependent objectives and constraints those capabilities require. These
capabilities do not by themselves choose how to change the letter arrangement
or provide a verified predictive model of object interaction.

This is a user-declared capability for design, not a claim that a remote planner
implementation, controller adapter, tool geometry, scene reconstruction or
metric calibration has been inspected or integrated here. Specify the exact
input/output binding and assumptions needed for the proposed design, including
feedback and failure handling. Preserve the distinction between this supplied
capability and verified facts available in the source recordings.

## Demonstrations and evidence contract

The authorized source is:
`/home/storage/tianrunhu/real_robot_data/push_letters`.

Available source episodes:
- `episode_2026091922002701`
- `episode_2026091922062101`

Use the separately supplied data catalog and data_contract for the actual
observation/action fields, camera metadata, missingness, source hashes, and
recording conventions. Those recording facts are not policy design choices.

For these recordings, existing sample_index / frames.json correspondence is
authoritative. Cuts use the paired indices with [start, stop) ranges. Each
source contains N paired observation/action records; do not invent an N+1
terminal record. Cross-device clock offsets do not authorize timestamp-based
re-pairing or block cutting. Preserve the original images and raw arrays.

There are no supplied verified per-sample object identities, masks, poses,
task goals, or success labels. Distinguish inspected visual inferences from
verified recording fields. Resolve semantic boundaries and condition
assignments using available evidence, documenting uncertainty where needed.

The recordings may be freely cut, selected, regrouped, overlapped, and reused
under the general prompt's provenance and supervision rules. Keep useful
demonstrated motions and context where supported; explain exclusions and
uncovered capabilities.

## Current deliverable boundary

This task specification accompanies the general semantic-cutting and prior
design prompt. The deliverables are cut datasets, explicit conditioning and
supervision definitions, selected priors for learned groups, proposed model
and policy inventories, executor responsibility groups, High-level Agent
calling contracts, and preprocessing/executor-adapter obligations for the later
implementation stage. No heuristic is required for a supplied executor group.

Do not implement policies or preprocessing, train models, execute the robot,
or process the other dataset in this stage. Any later API execution must use
the separately authorized run configuration. Creating this specification
does not itself launch or authorize a new API run.
