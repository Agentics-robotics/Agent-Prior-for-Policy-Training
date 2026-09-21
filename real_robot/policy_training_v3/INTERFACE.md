# Executable cut_v5 policy interface

This developer-owned contract supplies execution and provenance. You own all
scientific preprocessing, representations, architectures, priors and conversions.
Implement all three assigned priors in one immutable source package. They train
in independent processes with independent model, optimizer and EMA weights.

## Files and environment

Required files: policy.py, package.json, PRIOR.md, CALLING.md, POLICY_CATALOG.md,
HANDOFF.json, and <policy_id>_PRIOR.md / <policy_id>_USAGE.md for each policy.
Optional flat helper .py/.json/.md files are allowed. Every byte is journaled.
Available modules: torch 2.14.0, numpy 1.26.4, cv2 4.11.0, scipy 1.17.1,
skimage 0.26.0, PIL 11.3.0, diffusers 0.35.1, safetensors 0.8.0, math, json,
collections, itertools, functools, dataclasses, typing, copy, and your helpers.
Torchvision, transformers and segment_anything are not installed. No shell,
network, environment, reflection, arbitrary file access or subprocess exists
in generated code; kernel Landlock/seccomp restrictions apply before import.
request_asset downloads public weights/configuration, not a Python dependency.
Implement the loading/model code with available modules and declare adaptations.

Use `from real_robot.policy_training_v3 import public`:

- public.segments(): 26 original segment dictionaries in fixed plan order,
  each augmented with dataset_group (the original skill_id).
- public.records(trajectory_id): complete original NPZ arrays with original
  indices. action_json is the command; state dq is measured state.
- public.rgb(trajectory_id, sample_index, camera): original RGB uint8 HWC,
  camera third or wrist, respecting paired indices.
- public.metadata(trajectory_id): original recording metadata.
- public.progress(payload): save/print preparation progress; report per segment.
- public.asset_path(name): read-only path for an explicitly downloaded asset;
  load tensors using weights_only=True or safetensors, never arbitrary code.

The full frozen source plan and recording metadata are in spec. No semantic
labels, geometry conversion, controller, tracker or neural encoder is supplied.

## Assignment and package.json

Exact policy IDs, frozen heuristic IDs, dataset groups:

| policy_id | heuristic_id | dataset_group |
| --- | --- | --- |
| tool_waypoint_v1 | waypoint_clearance_residual | tool_position |
| piece_contact_graph_v1 | boundary_mechanics_residual | piece_relocation |
| piece_goal_field_v1 | causal_goal_field_servo | piece_relocation |

package.json exact top-level keys: schema (real_robot.policy_package.v3),
policies, preprocessing, dependencies, adaptations, limitations.
policies has exactly the above three IDs. Each value has exact keys:
heuristic_id, batch_size, learning_rate, weight_decay, max_grad_norm, config,
description, model_family, diffusion_consideration. Last three are nonempty
strings; explain the chosen family and why diffusion is appropriate or why
another family better serves this prior/data. config and preprocessing are
your JSON objects. batch_size is an integer 1..128; learning_rate in (0,.1],
weight_decay in [0,1], max_grad_norm in (0,1000]. dependencies, adaptations,
limitations are nonempty strings describing the actual executable package.

HANDOFF.json has a policies object with exactly these three IDs. Each entry
includes responsibility, selection_cues, entry_conditions,
continuation_conditions, exit_conditions, successor_readiness,
switch_procedure, failure_signatures, memory_reset_rules,
observation_dependencies, training_evidence and limitations.

## Top-level policy.py hooks

1. prepare_data(spec) returns {"arrays": dict[str,np.ndarray], "metadata": dict}.
   Called once on the entire assigned data before training. The framework
   stores numeric arrays as .npy and supplies read-only mmap arrays to workers.
   No object/string arrays; at most 64 GB total. Choose causal history,
   normalization, goals, privileged targets, masks, representations and
   preprocessing. All API-required scientific converters must be implemented.

   Required arrays for K eligible anchor rows and P=3 policies:

   - example_episode: integer [K], ordinal in spec["trajectory_ids"].
   - example_source_index: integer [K], original paired sample index.
   - example_segment: integer [K], ordinal in public.segments().
   - native_action: finite float [K,6], exactly concatenated recorded action.v
     and action.w, in that order and original numerical encoding. This is a
     provenance/reference target; your model may learn a transformed/reduced
     representation and reconstruct this candidate encoding in predict.
   - policy_weight: finite nonnegative float [K,3], column order
     spec["policy_ids"]. Positive means eligible for this policy, with your
     sampling weight; zero excludes it. Every row and column has positive sum.
     A positive entry must match that policy's assigned dataset_group.

   Other numeric arrays can have any shape: original-frame features, indices,
   cached goals, temporal windows, valid masks, optional raw dq/follower labels.
   A missing recorded dq does not invalidate finite v/w labels. Do not impute
   missing commands or substitute measured state velocities for commands.
   Identical (episode,index,segment) anchor triples may occur only once; use
   weights. Reused intervals in different frozen segments remain distinct
   examples with their own goals and masks. Both piece policies receive the
   same source group, but may declare different preprocessing eligibility.

   metadata contains num_examples=K, coverage and your JSON normalization/
   conversion data. coverage must account for every source segment and every
   policy, with retained/excluded counts and reasons. Preserve original indices
   and supervised ranges/exclusions; context-only rows cannot become targets.
   Historical observations stay in that segment's context bounds and remain
   causal. Future action windows/endpoint goals can be training targets only;
   mask windows beyond that segment's supervision and exclusions. No new cuts.
   Report any entire unsupported segment instead of hiding it in aggregate counts.

2. make_batch(policy_id, arrays, metadata, indices, spec) returns your tensor
   batch on spec["device"] (cuda:0). indices selects K anchor rows. The executor
   samples with replacement using only this policy's policy_weight column,
   seed 0. No mutation of mmap arrays. Causal inputs and training-only labels
   must be explicitly distinguished. Your action horizon is unrestricted by
   the one-step external action interface: diffusion may predict a sequence.

3. build_model(policy_id, spec) returns a torch.nn.Module with 1..64 million
   trainable parameters, containing all trainable components. spec contains
   policy_id, training, policy_config (complete policies[policy_id] entry),
   preprocessing, data_metadata, source_plan, trajectory_ids,
   recording_metadata, policy_ids, policy_datasets, segment_dataset_groups,
   and device. Deployment does not provide source_plan, trajectory_ids or
   recording_metadata: save required calibration/normalizers in data_metadata
   or model state and consume actual observation metadata when needed.

4. compute_loss(model,batch,spec) returns dict with differentiable scalar loss
   and optional finite scalar tensor metrics. Each learned auxiliary needs
   a real gradient path; constant penalties do not train a prior. Deterministic
   components from the prior are allowed alongside trained residuals/modules.

5. predict(model,batch,spec) returns finite tensor [B,6] of candidate commands
   in the recorded [v,w] encoding. For sequence models return the first decoded
   step after sampling/reconstruction. Keep normalization in saved spec/model.
   This is a checkpoint reload consistency check with fixed RNG, not a model
   family or action-horizon restriction. No hardware IO.

6. act(model,observation,call,memory,spec) returns a dict with action (finite
   recorded-encoding [v,w][6] when valid, otherwise null), memory, status and
   diagnostics. observation is one live record containing third_rgb, wrist_rgb
   (RGB uint8), q, measured dq, tau_ext, T_base_ee, T_base_flange, gripper
   fields, t, recv_time, sample_index, img_t_wrist, img_t_third, img_age_wrist,
   img_age_third and metadata. No action_json, future frame, source episode/
   segment/object ID is available online. call is your exact documented
   argument contract for current scene identity, explicit spatial goal,
   required calibration and invocation controls. memory=None starts a call.
   Implement causal tracking/history and share preprocessing with prepare_data.
   Preserve identity as pieces move. Return an informative unavailable status
   when required information is absent, not an always-unavailable placeholder.
   An optional inventory(observation,spec) hook may expose current scene handles
   and available geometry to a high-level agent; document its outputs.

7. decode_action(action,controller_contract,spec) returns enabled, command,
   reason. No hardware IO. v/w are recorded intention labels; their actual
   physical frames, scales, rate, tool calibration and controller/follower
   interpretation require explicit verification before a hardware command is
   enabled. A missing hardware contract must not prevent offline learned
   candidates or valid act calls. Do not fabricate calibration or verification.

## Training and checks

20,000 optimizer updates per policy, seed 0, AdamW, EMA .999; choose optimizer
settings above. Executor uses 500-step linear warmup then cosine decay, logs
every 100 steps, checkpoints every 2,000, retains final EMA. One 24-GiB RTX
A5000 per independent policy; host memory is ample. All demonstrations are
training data; no held-out performance evaluation or robot rollout this round.

check_package performs syntax/import allowlist/metadata/hook/document checks
only, zero optimizer updates. submit_package freezes the checked source hash.
Execution checks provenance, group eligibility, finite losses/gradients,
changed weights and exact EMA reload prediction. Implementation errors may
be returned for a separately recorded API repair revision. Losses are not
architecture-search feedback. Do not write an optimizer, training CLI,
evaluation runner or robot/network client.
