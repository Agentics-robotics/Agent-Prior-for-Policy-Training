# Executable cut_v6 geometric decision policy interface

Developer-owned execution/provenance contract. Runtime API owns all scientific
preprocessing, annotation inference, representations, targets, architecture,
priors, losses and observation/goal/executor conversions. Implement the frozen
shape_push_v1 / contour_contact_prior on shape_push_decisions. The supplied
geometric_motion_executor group retains evidence and has no learned policy.

## Available execution environment and evidence

An independent locked Pixi environment provides torch 2.14.0+cu130,
torchvision 0.29.0+cu130, transformers 4.57.6, official SAM 2 code pinned at
2b90b9f5ceec907a1c18123530e92e794ad901a4, numpy 1.26.4, scipy 1.17.1,
cv2 4.11.0, skimage 0.26.0, PIL 11.3.0, diffusers 0.35.1 and safetensors 0.8.0.
SAM 2's optional CUDA extension was explicitly disabled at installation. Use
its supported inference operations without extension-dependent postprocessing;
declare that setting. Python imports also allow math, json, collections,
itertools, functools, dataclasses, typing, copy, hydra, omegaconf and your helpers.

Scientific code has kernel-enforced read/write/device isolation, no network,
credentials, shell, subprocess, environment access, dynamic reflection or
arbitrary filesystem IO. No globals/nonlocals or dunder reflection (except
__init__). Use available library functions and bounded public capabilities.
Library internals can read the granted installed packages and explicitly
downloaded assets. Every third-party model must load locally, with
local_files_only=True and trust_remote_code=False where applicable. Use
weights_only=True/safetensors for tensor files. HF offline mode is enforced.

Use `from real_robot.policy_training_v4 import public`:

- segments(): the 14 frozen intervals in exact plan order, with dataset_group.
  Twelve learned context intervals contain exactly 47 sparse decision indices;
  two executor_only intervals have null supervision bounds.
- records(trajectory_id): complete original NPZ arrays, unchanged and read-only
  in intent. Preserve paired indices; measured dq is not command dq.
- rgb(trajectory_id,index,camera): original RGB uint8 HWC, third or wrist.
- depth(trajectory_id,index,camera): original raw PNG values, no unit conversion
  or registration implied.
- metadata(trajectory_id), camera_intrinsics(trajectory_id): original JSON.
- progress(payload): save/print preparation progress.
- asset_path(name): flat explicitly downloaded tensor/config file.
- model_path(name): explicitly downloaded model snapshot directory.
- source_json(name): read your submitted flat JSON resource, e.g. API-authored
  visual annotations/calibration inference with source evidence.
- write_report(name,payload): write a flat .json under output/evidence.
- write_image(name,uint8_array): write a flat .png under output/evidence.

You can inspect original RGB/state/metadata with evidence tools and put inferred
annotations in your source files. Label such evidence as API/algorithm inferred,
with source indices, uncertainty and method, never as completed human review.
No verified object masks, contact labels, table/tool contact-surface metrology
or human annotation service is supplied. Use available recordings to estimate
what can be supported; disclose unresolved physical quantities and their effect
on labels and deployment. Do not fabricate a verified calibration or contact.

request_asset downloads a specific public tensor/config file; request_model
downloads selected tensor/config/tokenizer files from a public HF repo, pinning
resolved commit and hashes. No executable remote code or automatic retry.
SAM 2.1 and Qwen in the source plan are proposed dependencies, not preloaded
weights. Decide required assets and document any justified implementation
adaptation, including upstream Agent responsibilities outside this policy.

## Package and API-owned documents

Required flat files: policy.py, package.json, PRIOR.md, CALLING.md,
POLICY_CATALOG.md, HANDOFF.json, shape_push_v1_PRIOR.md, shape_push_v1_USAGE.md.
Additional .py/.json/.md helpers are allowed. Files are individually journaled
and frozen together. Import helpers by their flat module names.

package.json exact top-level keys: schema="real_robot.policy_package.v4",
policies, preprocessing, dependencies, adaptations, limitations.
policies has exactly one entry, shape_push_v1, with exact keys:
heuristic_id="contour_contact_prior", batch_size, learning_rate, weight_decay,
max_grad_norm, config, description, model_family, diffusion_consideration,
updates, warmup_updates, prediction_dimension, decision_schema.

- batch_size integer 1..128; learning_rate in (0,.1]; weight_decay in [0,1];
  max_grad_norm in (0,1000].
- updates integer 1..20000: choose and justify a fixed budget before training
  for this small dataset. warmup_updates integer in [0,updates).
- prediction_dimension integer 1..8192, the second dimension of predict's
  checkpoint-check tensor, with all fields/units documented. It is not forced
  to be a six-dimensional EE command.
- decision_schema is a valid JSON Schema object defining a non-null structured
  online geometric decision. Frame, units and physical referents must be explicit.
- config/preprocessing are JSON objects; description/model_family/
  diffusion_consideration/dependencies/adaptations/limitations are nonempty strings.
  The frozen candidate scorer is the starting design; record the actual family
  and whether diffusion adds value here. No alternative-policy quota.

HANDOFF.json has policies.shape_push_v1 with responsibility, selection_cues,
entry_conditions, continuation_conditions, exit_conditions, successor_readiness,
switch_procedure, failure_signatures, memory_reset_rules,
observation_dependencies, training_evidence and limitations.

## policy.py hooks

1. prepare_data(spec) -> {"arrays": dict[str,np.ndarray], "metadata": dict}.
   Called on assigned data with a 24-GiB GPU. Cache reusable perception/features;
   fitting must not re-run image segmentation every optimizer update. Numeric,
   finite arrays only, with explicit missingness masks; at most 64 GB. The
   executor saves read-only .npy arrays for training.

   For K retained decision anchors, mandatory arrays are:
   example_episode integer[K] (ordinal in spec.trajectory_ids),
   example_source_index integer[K] (original paired index),
   example_segment integer[K] (ordinal in public.segments()),
   observation_start/observation_stop integer[K] (half-open causal interval
   inside materialized segment, stop <= anchor+1),
   target_start/target_stop integer[K] (half-open target-evidence interval inside
   materialized segment, includes the anchor),
   policy_weight finite nonnegative float[K,1] with positive rows/column.

   Other feature/target/candidate/mask arrays are your design. There is no
   mandatory native_action array or recorded v/w reconstruction. Every positive
   supervision example must be one of the exact 47 frozen decision indices
   in its learned group. Context/executor rows may supply evidence, history or
   future-derived targets, not new decision anchors. Keep source identities
   unique; weighting/online augmentation expresses repeated use. No new cuts.

   metadata includes num_examples=K, coverage, label_provenance, anchor_audit
   and your normalization/conversion data. anchor_audit is a list containing
   every one of the 47 frozen decisions exactly once, with trajectory_id,
   segment_id, source_index, status=retained|excluded, and nonempty reason.
   Add label quality/uncertainty, physical referents, evidence and conversion
   status as useful. Retained entries exactly match actual cache rows. coverage
   accounts for all 14 segments, distinguishing intentionally untrained executor
   evidence from failed learned targets. Report invalid/rejected labels and
   their reasons. All 47 are candidate anchors, not guaranteed usable labels.

   Write compact per-anchor diagnostic reports and representative original-image
   overlays showing actor, contact, goal/stroke and uncertainty. Keep inferred
   physical geometry distinguishable from recorded measurements. These support
   API review through check_preprocessing/read_preparation_evidence. If no target
   is defensible, still write the diagnostic reports before returning so missing
   evidence is reviewable; never fabricate labels just to obtain nonzero K.

2. make_batch(policy_id,arrays,metadata,indices,spec): your tensor batch on
   spec.device (cuda:0), selecting given rows. Separate causal features from
   future-derived targets/goals; goal input at deployment is caller-supplied.
   Preserve augmentation consistency between geometry, goals and target motion.

3. build_model(policy_id,spec): torch.nn.Module with 1..64 million trainable
   parameters. Contains every trainable component. Frozen perception may live
   in cached preprocessing; document all external assets. spec contains training,
   policy_config (complete policy entry), preprocessing, data_metadata,
   source_plan, trajectory_ids, recording_metadata, policy_ids, policy_datasets,
   segment_dataset_groups and device. Online spec omits source_plan,
   trajectory_ids and recording_metadata. Save needed normalizers/calibration
   provenance in data_metadata/model state and validate actual live calibration.

4. compute_loss(model,batch,spec): dict with differentiable scalar loss and
   optional finite scalar tensor metrics. Real trainable parameters need a
   valid gradient path; fixed geometry/controller components remain explicit.

5. predict(model,batch,spec): finite float tensor[B,prediction_dimension] with
   your documented geometric prediction or numeric decision representation.
   Used for exact final EMA reload consistency, not robot actuation.

6. act(model,observation,call,memory,spec): dict with decision (JSON-native object
   conforming to decision_schema, or null), memory, status, diagnostics. The
   observation carries causal third_rgb/wrist_rgb, q/dq/tau_ext, T_base_ee,
   T_base_flange, t/recv_time, image times/ages, sample_index, gripper fields and
   metadata. No command action_json, future frame, recorded episode/segment ID
   or future-derived target is available online. call supplies your documented
   physical instance, spatial objective, calibration and limits. Share actual
   preprocessing with prepare_data. A missing controller binding must not
   prevent an offline geometric proposal when its own input contract is met.
   Provide meaningful unavailable statuses for genuinely missing information.
   Implement optional inventory(observation,spec) for current geometry/handles
   if required by your usage contract. Document causal history/timing and resets.

7. executor_request(decision,executor_contract,spec): returns enabled, request,
   reason. Produces a structured planner/controller request only, no robot IO.
   Require actual verified binding/calibration for enabled execution; keep the
   geometric learned proposal separate from this integration status. Preserve
   contact point versus TCP/tool-surface semantics and short planar stroke
   constraints. Implement the conversion supported by supplied information;
   unresolved deployment quantities must be explicit.

## Preparation checks, submission and training

check_package validates syntax/imports/hooks/schema/documents only.
check_preprocessing snapshots the current package and executes its actual
prepare_data plus make_batch/build_model/loss/backward/predict interface checks.
There are zero optimizer updates and no performance ranking. Up to six distinct
source revisions can be checked per design session. The returned reports and
images can be inspected with read_preparation_evidence. A failed check retains
its traceback/evidence; fix your unsubmitted implementation explicitly.

Review representative derived targets, then check_package and submit_package
using the exact checked hash. If the last successful preparation source is
unchanged, its cache is preserved into the submitted package to avoid repeating
expensive preparation. Other source revisions require new preparation.

Outer executor uses the submitted fixed updates budget, seed 0, AdamW, chosen
warmup then cosine decay, EMA .999, logs every 100, checkpoints every 2000 and at
the last step, retaining final EMA. All supplied demonstrations are training
data in this run. No held-out performance selection or robot rollout. Explain
any difference from proposed early stopping in the cut-stage design. Validate
finite losses/gradients, effective weight changes and exact checkpoint reload.
Implementation failures are retained for a separate API repair; no automatic
transport retry or architecture search. Do not implement optimizer/CLI/network
clients. The developer handles execution and accounting.
