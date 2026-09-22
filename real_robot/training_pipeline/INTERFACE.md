# Generic API implementation and training contract, v1

The framework owns data access, isolation, scheduling, provenance and execution.
The Runtime API owns preprocessing, derived sampling/grouping, policy inventory,
representations, models, losses, optimizer, learning-rate schedule, sampling,
stopping and checkpoint selection. Task facts arrive separately in read_assignment.
The source cut/prior is a revisable initial design. Preserve it and record your
changes in package.json.data_plan; no earlier decision-index list is a training
eligibility whitelist. The original authorized episodes are the data boundary.

## Environment and capabilities

The existing locked environment supplies PyTorch, torchvision, numpy, scipy,
OpenCV, Pillow, skimage, diffusers, transformers, SAM 2, safetensors, hydra and
omegaconf. SAM 2's optional CUDA extension is disabled. Pretrained weights are
not implicitly installed. request_asset/request_model can obtain explicitly
chosen public tensor/config/tokenizer assets with hashes and pinned provenance.
Use local weights, local_files_only=True and trust_remote_code=False where
applicable. Assets and source are read-only after submission. No automatic
network retries. There is no robot connection.

Python files may import those numerical libraries, math, json, collections,
itertools, functools, dataclasses, typing, copy, and flat local helpers. Files
have kernel isolation: no network, credentials, shell or arbitrary filesystem
IO. No dynamic reflection, globals/nonlocals, or private dunder access except
__init__. Use `from real_robot.training_pipeline import public`:

- `segments()`: your current derived intervals, in package data_plan order.
- `records(trajectory_id)`: original NPZ arrays with read-only views.
- `rgb(trajectory_id,index,camera)`: original RGB uint8 HWC, third or wrist.
- `depth(trajectory_id,index,camera)`: original PNG values; units are not inferred.
- `metadata(trajectory_id)`, `camera_intrinsics(trajectory_id)`: original JSON.
- `asset_path(name)`, `model_path(name)`: explicitly downloaded local assets.
- `source_json(name)`: your flat JSON resources, including inferred annotations.
- `progress(payload)`, `write_report(name,payload)`, `write_image(name,uint8_array)`:
  preparation progress and flat JSON/PNG evidence, inspectable by the design API.

read_images/read_steps/read_metadata inspect the source before implementation.
Model/perception annotations derived by you are inferred, with source evidence
and uncertainty. Determine factual fields and missingness from the supplied
recording facts; they are not shared semantic assumptions across tasks.

## Package

Required flat files: policy.py, package.json, PRIOR.md, CALLING.md,
POLICY_CATALOG.md, HANDOFF.json, and `<policy_id>_PRIOR.md`/`<policy_id>_USAGE.md`
for every policy. Additional .py/.json/.md files are allowed. Import helpers by
their flat module names. Every file version is journaled; submissions immutable.

package.json exact keys:

- schema: `real_robot.policy_package.pipeline.v1`
- policies: object mapping API-chosen IDs to the policy settings below.
- preprocessing: arbitrary JSON object of API-chosen preprocessing settings.
- data_plan: `{rationale, changes_from_prior, segments}`. Text fields explain
  the final sampling/grouping design and its relation to the previous design.
  Each segment has exactly `{segment_id, trajectory_id, start, stop, rationale}`.
  IDs are unique; ranges are valid original-episode [start,stop). You may revise
  intervals, overlap/reuse data and choose new anchors. Explain semantic changes.
- dependencies, adaptations, limitations: nonempty strings.

Each policies entry has exactly: description, model_family,
architecture_rationale (strings); config (arbitrary object); updates (positive
maximum update count within resource limit); batch_size (positive maximum batch
size); seed (integer 0..2^32-1); ema_decay (null to disable, or [0,1));
max_grad_norm (null to disable clipping, or positive); prediction_dimension
(positive flattened tensor width used by predict for reload checks);
decision_schema (valid JSON Schema for the actual online decision).

IDs use `[a-z][a-z0-9_]{0,63}`. Resource limits in the assignment bound compute,
cache size, parameters and policy count; these are not requested design sizes.
The actual executor decision may be a structured geometry objective, motion or
other API-defined representation. It need not be an EE action or match predict's
flattened diagnostic representation. Document fields, frames, units and meaning.

HANDOFF.json has `policies`, with the same IDs and these fields per policy:
responsibility, selection_cues, entry_conditions, continuation_conditions,
exit_conditions, successor_readiness, switch_procedure, failure_signatures,
memory_reset_rules, observation_dependencies, training_evidence, limitations.
Document observations/goals, callable examples, memory/reset, completion,
failure handling and executor bindings in CALLING.md and per-policy USAGE files.

## Preparation and accounting

`prepare_data(spec) -> {arrays, metadata}` executes once on full authorized data.
spec is a dictionary containing preprocessing, source_plan, data_plan,
trajectory_ids, recording_metadata, policy_ids, device and resource limits under
training. Use API-chosen preprocessing seeds/settings. It is not a Namespace.

arrays is a mapping of identifier names to numeric numpy arrays. All floating
values must be finite; represent missing/partial supervision with explicit masks
of your design and implement their use in compute_loss. This permits different
parts of an example to contribute to different losses. No full-target eligibility
rule is imposed by the framework. The required provenance arrays are:

- example_episode[K]: integer position in spec.trajectory_ids.
- example_segment[K]: integer position in data_plan.segments.
- example_source_index[K]: original paired decision/state index.
- example_variant[K]: nonnegative identifier distinguishing derived examples at
  the same source anchor, e.g. different supported goals or target definitions.
- observation_start/observation_stop[K]: original causal [start,stop) evidence
  inside the declared segment, with stop <= example_source_index+1.
- target_start/target_stop[K]: source evidence [start,stop), in that same segment;
  targets may use future observations. No cross-episode concatenation.
- policy_weight[K,P]: finite nonnegative eligibility/use weights in policy_ids
  order, with positive row sums. Your sampler decides how these weights are used.

Other arrays, tensor shapes, target dimensions and loss masks are your choices.
Exact duplicate (episode,index,segment,variant) identities are rejected. Reuse,
new anchors, dense windows and goal variants are supported. State independent
event diversity separately from derived-example counts.

metadata requires num_examples, coverage, label_provenance, data_audit,
variant_definition and source_accounting. Apart from num_examples and
source_accounting these are API-defined JSON evidence/descriptions. Explain
online vs training-only inputs, uncertainty, grouping and any splits; keep
related source events together for evaluation. Source_accounting is a list of
`{trajectory_id,start,stop,use,reason}`, where use is supervision, context or
excluded. Account for every authorized row; ranges can overlap for multiple
roles. Every retained anchor must fall in a declared supervision range. Report
why evidence was unusable for your selected target and what recovery you tried
or judged appropriate in data_audit. The framework also computes unique source
anchor counts and per-policy/per-segment sample counts mechanically.

Zero examples can be reported to explain a data blocker. A package with missing
policy examples cannot be submitted as ready to train. No numeric sample minimum
claims that the data is scientifically sufficient: you assess that explicitly.

## Model and training hooks

All hooks are top-level synchronous functions in policy.py. Each model spec adds
policy_id, policy_config (that policies entry), and data_metadata to the above.

- `build_model(policy_id,spec) -> torch.nn.Module` with trainable parameters.
- `make_batch(policy_id,arrays,metadata,indices,spec) -> any batch structure`.
- `compute_loss(model,batch,spec) -> dict`: `loss` and any metrics are finite
  scalar torch tensors; loss is differentiable. Choose losses and masks yourself.
- `predict(model,batch,spec) -> finite torch tensor[B,prediction_dimension]`:
  diagnostic prediction for interface/reload checks. Own the actual output form.
- `configure_optimizer(model,spec) -> torch.optim.Optimizer`: instantiate your
  chosen optimizer with model parameters and settings. State is saved.
- `sample_indices(policy_id,arrays,metadata,rng,step,spec) -> int numpy vector`:
  choose eligible rows with positive policy_weight for this policy; size 1 through
  batch_size. rng is a seeded numpy Generator. You choose sampling and splits.
- `before_update(optimizer,step,spec)`: set learning rates or other optimizer
  state according to your schedule. Steps start at one.
- `after_update(model,ema,arrays,metadata,step,metrics,spec) ->
  {stop: bool, select: null|"model"|"ema", metrics: JSON object}`: models are in
  eval mode and gradients disabled. Inspect API-defined validation/criteria as
  needed. select saves that state as the deployment checkpoint; null retains the
  earlier selection. ema is None when disabled. Select at least once before
  stopping or reaching the budget. You choose stopping and checkpoint selection.

The fixed loop calls your sampler/batch/loss, zero_grad, backward, optional
clipping, optimizer.step, optional EMA and after_update. It records actual steps,
checks finite gradients, verifies changed weights and reloads the selected state.
The adapter supports this single-optimizer PyTorch contract. If a necessary
algorithm cannot fit it, state the missing execution capability; do not silently
claim to have implemented a different training loop.

## Online interface and executable checks

- `act(model,observation,call,memory,spec) -> dict` containing memory, decision
  (nullable structured output) and status. You define their semantics and any
  additional progress fields. This must perform the actual online preprocessing.
- `executor_request(decision,executor_contract,spec) -> JSON`: formulate the
  executor objective; no hardware IO occurs in this process.
- `calling_cases(arrays,metadata,spec) -> list[case]`: API-defined test observations
  and calls. Each case has name, observation, call, reset, executor_contract,
  expected_status, expect_decision. Cases execute act and executor_request after
  loading trained weights. Include at least one non-null decision plus relevant
  missing-input, reset, interruption or invalidity behaviors. Document synthetic
  fixtures vs real source data and what these checks establish.
- Optional `inventory(observation,spec)` exposes an Agent-facing scene converter.

Online spec excludes source_plan, trajectory_ids, recording_metadata, data_plan
and data_metadata. Package configuration and declared source JSON/assets remain
available for fitted calibration/normalizers. Persist any fitted information
needed online explicitly; do not depend on training cache/recording access.
PolicyProcess in inference.py wraps the isolated worker, observation serialization
and memory reset. The API catalog/usage documents tell the Agent what to pass.

## Fixed workflow

Read assignment/interface, write files, check_package, check_preprocessing,
inspect preparation evidence and revise as needed within the tool budget.
Preparation also runs loss/backward/predict checks with zero optimizer updates.
submit_package requires the exact checked source hash, preparation manifest hash
and a readiness object with decision=train|blocked, coverage_assessment,
data_recovery_assessment, training_rationale and limitations. Assess the actual
prepared data; the framework checks the assessment's artifact identity.

A train submission automatically runs each chosen policy, checkpoint reload and
calling checks, then publishes the library with authorship/cost receipts. A
blocked submission saves its reason and stops. Execution implementation failures
return their exact diagnostics to a new API revision, within the configured
repair allowance; prior attempts remain intact. Transport errors/interruption
stop without replay. No developer rewrites your scientific files or prescribes a
replacement design. Interface success is not a measured robot success rate.
