# Exp2 methods and experimental scope

This document describes the executed framework. Final outcome tables belong to
`REPORT.md`, which is generated only after the additional single-policy study is
complete and audited. Paths are relative to the repository root unless linked.

## 1. Research question

Exp2 studies long-horizon, state-observed manipulation with twelve successful
demonstrations per task. It compares a conventional full-task diffusion policy,
an API-designed full-task diffusion policy, and APPL systems that combine
API-designed skill policies with API decisions during execution. The design API
can introduce learned representations, architectural structure and auxiliary
objectives. These are hypotheses encoded in trainable models; their descriptions
do not establish that the intended physical property is guaranteed.

The four methods are:

| Method ID | Offline API | Training organization | Deployment |
| --- | --- | --- | --- |
| `naive_DP` | None | One full-task DP per task | Repeated calls to that model |
| `SinglePrior_6_xhigh` | GPT-6 Astra/xhigh | One API-designed full-task DP per task | Repeated calls to that model; zero API requests |
| `APPL_5_5_high` | GPT-5.5/high | API segmentation and independent prior policies | API selects frozen policies, durations and stop conditions |
| `APPL_6_xhigh` | GPT-6 Astra/xhigh | New API segmentation and new independent prior policies | API selects the newly trained frozen policies |

Model names and reasoning efforts refer to the actual service request/response
fields retained in journals. They are not inferred from a UI label. The newer
APPL round changed the model, effort, segmentation and learned library together.
The single-policy baseline uses xhigh; referring to its offline design as
"Exp1-style" does not mean it uses Exp1's max effort or multi-candidate selection.

## 2. Tasks, data and distribution shift

The five tasks are drawer exchange, two-block sorting, exchange via a temporary
buffer region, unstacking followed by sorting, and packing two blocks into an
open tray. They use a Panda robot and two color-identified blocks. The four new
task definitions, acquisition scripts and geometric success contracts are
developer-owned experiment preparation. The API does not author expert rollouts
or gain expert-action access at deployment.

Each task supplies twelve original successful trajectories. A trajectory has T
native actions and T+1 synchronized observations. Full trajectories are used by
both single-policy methods; APPL uses API-selected, potentially overlapping
segments of these same trajectories. All boundaries are per trajectory rather
than a single shared time schedule. API-authored priors and handoff documents
remain unchanged after submission. Failed acquisition attempts and preparation
repairs are retained; demonstration acquisition success is not policy success.

The four new tasks use a fixed native Panda motion-planning expert. It approaches
each object from above, closes the gripper for twelve control steps, lifts and
transports it, opens for twelve steps, retreats, and finishes with twenty settling
steps. Buffer exchange uses its declared temporary placement in this program.
The collector records actual simulator actions and observations, checks physical
pickup and final geometric goals, and does not overwrite object poses after
reset. This produces successful nominal trajectories, not a dataset of failed
grasps and corrective recovery attempts. The original drawer dataset is reused
with its separately retained acquisition provenance.

For the four new tasks, ID reset XY offsets are uniform within +/-0.012 m of
each nominal block location. OOD samples first draw offsets within +/-0.04 m,
then force one randomly chosen axis to have magnitude 0.022--0.04 m. Stacked
blocks share their XY displacement; other blocks have separate offsets. Drawer
ID support uses the original generator (red +/-0.012 m, blue +/-0.015 m), and
drawer OOD applies the same larger-offset procedure around its nominal starts.
Exact reset states, seeds and coordinates are exported with the report.

OOD changes initial positions only. Robot, block geometry, goals, controllers
and success tests are unchanged. This is not evidence for object-category,
visual, dynamics, language or real-world generalization. An ID reset does not
ensure that later missed-grasp or disturbed-object states remain in demonstration
support.

Sources: [task definitions](framework/src/appl/scaleup/tasks.py),
[paired layout protocol](framework/src/appl/scaleup/protocol.py),
[scale-up specification](protocols/SCALEUP.md).

## 3. Observation and action interface

Policies receive the latest two causal state vectors, each with 47 components:

| Indices, Python slice convention | Quantity |
| --- | --- |
| 0:9 | Seven arm and two finger joint positions |
| 9:18 | Corresponding joint velocities |
| 18:25 | TCP position and wxyz quaternion |
| 25:32, 32:39 | Red and blue block poses |
| 39:40, 40:41 | Drawer position and velocity |
| 41:44, 44:47 | Observed red and blue target positions |

Coordinates are in the world frame, positions in metres, and arm angles in
radians. Non-drawer tasks have zero drawer compatibility channels. Policies do
not receive camera images, a task-phase label, elapsed episode time or expert
actions. Images are retained for inspection and replay.

Each action contains seven absolute Panda joint targets and one gripper command
in [-1,1]. Larger gripper commands open the fingers. Native action bounds are
applied before stepping the same environment. These are joint-position actions,
not Cartesian end-effector displacements. Relative state features therefore do
not, by themselves, make the action mapping equivariant.

The deployed simulator is ManiSkill/SAPIEN with CPU PhysX, native joint-position
control and 20 Hz control steps; rendering uses CUDA. It pauses during learned
policy inference and API requests. A 5,000-step cap represents at most 250 seconds of simulated control,
not 250 seconds of wall time. No physical-robot or real-time result is claimed.
The report-time host snapshot records NVIDIA RTX A5000 GPUs with 24 GB memory;
exact package versions, device identifiers and its observation date are in
[runtime_environment.json](runtime_environment.json). That later read-only
snapshot complements the frozen environment lock and per-job device receipts;
it is not used to infer an unrecorded historical driver configuration.

## 4. Shared normalization and diffusion training

One observation/action normalizer per task is fitted to all original training
actions and their causal input observations, before segmentation or overlap.
The terminal T+1 observation is a training label, not an additional action input
used to fit the normalizer. No evaluation observations are used. All current
methods use the same full-demonstration normalization payload for that task.
Quaternion components use fixed unit bounds; constant dimensions and numerical
floors are handled by the frozen implementation. In limits mode, the stored
`std` field denotes a half-range, not an empirical standard deviation. Observed
states outside the training range are not silently clipped back into it.

| Setting | Executed value |
| --- | --- |
| Training seed | 0; one training replicate |
| Observation history | 2 |
| Prediction horizon | 16 actions |
| Executed prefix | 8 current/future actions, then replan |
| Diffusion training steps / deployment steps | 100 / 100, DDPM |
| Noise target and schedule | Epsilon; squared-cosine beta schedule |
| Sample clipping | Enabled |
| Batch size | 128 |
| Optimizer | AdamW, learning rate 1e-4, weight decay 1e-6 |
| LR schedule | Cosine, 500 warmup updates |
| Gradient norm limit | 1 |
| EMA decay / checkpoint selection | 0.999 / last EMA at the declared budget |
| Full-task model updates | 60,000 |
| APPL updates per independent prior policy | 20,000 |

At observation t, the training action window covers t-1 through t+14; execution
starts with the aligned current action rather than executing the historical
slot. Boundary padding is masked. Future observations are allowed as training
labels for API-designed auxiliary objectives, but never as deployment inputs.

Naive DP uses the conventional conditional U-Net with widths 128/256/512,
timestep embedding 128, kernel size 5 and 8 groups: 16,998,408 parameters. The
API can choose a learned architecture within the public 64-million-parameter
limit. An optional public diffusion backbone is available. A standard loss
predicts the Gaussian noise added to normalized action chunks; API policies may
add trainable auxiliary terms or change learned conditioning/structure. Exact
mechanisms, weights and gradient paths are in each API-authored `PRIOR.md` and
`policy.py`, indexed in the final artifact catalog.

Sources: [normalization and windows](framework/src/appl/dp_baseline/data.py),
[shared prior data path](framework/src/appl/prior_policies/data.py),
[prior training engine](framework/src/appl/prior_policies/engine.py),
[single-policy contract](framework/experiments/exp2/single_policy/design.py).
The vendored backbone's retained upstream attribution and license are included
with the [framework source](framework/README.md).

## 5. APPL offline design and runtime execution

The segmentation API reads demonstrations and the goal contract, chooses skill
boundaries and overlap separately for each trajectory, proposes distinct
heuristic priors, and writes explicit handoff information for each heuristic.
The current prompt requests substantially larger transition overlap, informed
by earlier incomplete handoffs. The framework validates indexing, coverage and
submission provenance; it does not invent boundaries or rewrite heuristic text.
Temporal overlap expands demonstrated transitions, without supplying recovery
examples for arbitrary spatial or contact errors.

Each retained heuristic receives an independent API implementation, source
folder, prior/handoff documentation, checkpoint and measured support information.
The API owns the scientific policy code. The framework supplies bounded file
tools, training data, contracts and interface checks. API code executes in
credential-free restricted workers. Submitted candidates are immutable. A
two-update interface check is not a performance evaluation or a warm start for
formal training. A separate actual-deployment check generates nine actions from
original training observations, exercising two denoising chunks with zero
candidate optimizer updates and zero simulator steps. Update budgets count
changes to candidate parameters. Restricted workers separately record one
trusted runtime-initialization optimizer step on synthetic, non-candidate
parameters in `enforcement.json`; it is not candidate training and is excluded
from those budgets. Wall-clock records include initialization overhead.

At runtime the API reads the task/state and policy catalog, reads relevant
`PRIOR.md` and `HANDOFF.json` files, and selects a frozen learned policy. It
chooses an invocation duration of at most 300 control steps and literal numeric
stop conditions. Conditions within each group are conjunctive; matching any
group can return control. The framework checks these exact conditions after
each action and reports observations and within-invocation metric ranges. It
does not invent a successor policy, grasp threshold or recovery controller.

Consecutive invocations of the same policy continue its action stream. Switching
policies resets the incoming observation/action queue, without reseeding its RNG.
Each policy worker is initially seeded from the episode seed. Fixed seeds do
not imply identical random-number consumption across different policy schedules.
The API maintains its own concise notebook; context projection retains the
initial task, latest complete policy reads, recent complete exchanges and the
latest invocation/notebook. The unmodified full history remains in the journal.

The total/remaining 5,000-step cap is hidden from API inputs. The API may call
`finish` before it, but cannot declare success by fiat. The framework independently
measures task success. API generation has no fixed provider sampling seed, so
same-reset repeats are not deterministic agent trajectories. The cap on runtime
API requests is 128 per attempt. Transport errors retain unknown outcomes and
reported/unknown costs; explicitly authorized resets are documented separately.

Sources: [segmentation prompt/tools](framework/src/appl/demonstrations/decouple.py),
[deployment executor](framework/src/appl/prior_policies/deploy.py),
[feedback/context contract](framework/src/appl/prior_policies/feedback.py),
[5000-step revision](protocols/INFERENCE_5000.md).

## 6. Evaluation and success

The main matrix contains five tasks, four methods, two conditions and thirty
paired initial layouts per condition: 1,200 intended method-layout cells. Older
methods are reused from their validated records, not retrained or replayed for
the new baseline. Models and required interface checks are frozen before test
rollouts. Later studies use previously examined paired layouts; they are not
fresh untouched tests. No test outcomes are supplied to the design API, and no
test-driven policy revision or best-checkpoint selection is performed.

Drawer success is the simultaneous conjunction of:

1. Drawer displacement strictly greater than 0.26 m.
2. Full rotated red-block XY containment on its pad and 0.014 < red z < 0.031 m.
3. Full rotated blue-block XY containment in the moving drawer cavity and
   0.053 < blue z < 0.074 m.

The other tasks require both blocks simultaneously inside their assigned goal
regions: abs(center XY - goal XY) + rotated block XY extent <= (0.06,0.06) m,
and absolute center-height error < 0.011 m. Cubes have 0.02 m half-size.

The buffer region and the demonstrated object order describe the expert program;
the final success predicate does not independently require a visit to the buffer
or enforce red-first execution. Other learned physical routes count if they meet
the same simultaneous geometric goals. Intermediate subgoals and semantic skill
handoffs guide the API, while the final task predicate remains independent.

All main-matrix methods use these same geometric predicates. One satisfying
observation suffices. Release, TCP clearance, object speed and sustained holding
are not additional requirements; a block may still be grasped when success is
recorded. The older frozen M0 diagnostic used its original stricter terminal
conditions and must be reported separately.

Report success counts over the thirty planned cells, unknown outcomes separately,
and descriptive Wilson intervals for complete groups. Paired win/loss counts
use only complete matching cells. Intervals over layouts do not quantify
training-seed or repeated-API uncertainty. Prefix success at 1,500/3,000/5,000
steps comes from the same 5,000-step trajectory; it is not a new intervention on
the episode budget. Videos use original 128x128 camera frames, usually every
twenty control steps and displayed at 6 fps (approximately 6x simulated speed).

Sources: [drawer goal contract](contracts/drawer_exchange.json),
[other task predicates](framework/src/appl/scaleup/tasks.py),
[trace audit](framework/src/appl/scaleup/report.py).

## 7. What the comparisons can establish

- SinglePrior versus naive DP compares offline API-designed inductive bias under
  equal demonstration and optimizer-update budgets. Architectures and FLOPs can
  differ; this is not an equal-FLOP comparison.
- APPL versus SinglePrior changes decomposition, model count, aggregate training
  and runtime decisions together. It does not isolate the runtime agent alone.
- APPL 6 versus APPL 5.5 changes model and effort together and regenerates the
  segmentation and policy library. It does not isolate xhigh or model family.
- APPL 5.5 has 51 evaluated prior policies (1,020,000 formal updates); APPL 6 has
  36 (720,000). Each full-task method has five models (300,000 updates). Historical
  reuse and additional interface-check costs are reported separately.
- There is one training seed, only twelve demonstrations per task, thirty layouts
  per condition, and no fresh independent repeat of the runtime API decisions.
  Negative task-specific results must remain visible alongside aggregate gains.

Exp1 is preserved and is outside this report's empirical evidence. Historical
M0 diagnostics and preliminary M1 versions are development context, not extra
replicates of the main matrix. No completed formal M2 comparison is implied by
obsolete filenames or earlier plans.
