# Executable prior-policy package interface

This is a developer-owned execution contract. You own the scientific source.
Implement both assigned policies in one immutable package; each is subsequently
trained in its own process with independent model/optimizer/EMA weights.

## Files and environment

Required files: `policy.py`, `package.json`, `PRIOR.md`, `CALLING.md`. Optional
flat helper `.py` files are allowed. All package bytes are journaled and hashed.
Available modules: torch 2.14.0, numpy 1.26.4, cv2 4.11.0, scipy 1.17.1,
skimage 0.26.0, PIL 11.3.0, diffusers 0.35.1, safetensors 0.8.0, math,
json, collections, itertools, functools, dataclasses, typing, copy, and your
own modules. Torchvision, transformers and segment_anything are NOT installed.
Do not import them or assume their weights. No shell, network, environment,
reflection, arbitrary filesystem access or subprocess capability exists in
generated code. Kernel Landlock/seccomp restrictions apply before import.
Public model assets may be requested with request_asset; that is a download,
not an installation of a model package. Implement dependencies explicitly.

Use `from real_robot.policy_training import public` for read-only capabilities:

- `public.segments()` -> the 22 original segment dictionaries, in fixed order.
- `public.records(trajectory_id)` -> dict of full original NPZ arrays; preserve
  indices. `action_json` contains the recorded command; state dq is different.
- `public.rgb(trajectory_id, sample_index, camera)` -> original RGB uint8 HWC,
  where camera is `third` or `wrist`, following existing pairing indices.
- `public.metadata(trajectory_id)` -> recorded episode metadata.
- `public.progress(payload)` -> save/print preparation progress; use per segment.
- `public.asset_path(name)` -> an explicitly downloaded read-only asset path;
  load tensor weights with weights_only=True or safetensors, never arbitrary code.

The source plan and metadata are in spec. No semantic segmentation or label
converter is supplied by the framework. You must write those conversions.

## package.json

Exact top-level keys: `schema` ("real_robot.policy_package.v1"), `policies`,
`preprocessing`, `dependencies`, `adaptations`, `limitations`.
`policies` is an object with exactly `contour_push` and `visual_push`. Each value
has exact keys: `heuristic_id`, `batch_size`, `learning_rate`, `weight_decay`,
`max_grad_norm`, `config`, `description`. IDs are boundary_graph and
dual_view_memory respectively. batch_size is an integer 1..128; other optimizer
values are finite numbers in sensible positive ranges. config is your JSON object.
`preprocessing` is your JSON configuration object. Remaining fields are nonempty
English strings declaring actual dependencies, adaptations and limitations.

## policy.py numerical hooks

1. `prepare_data(spec)` -> `{"arrays": dict[str,np.ndarray], "metadata": dict}`.
   The framework calls this once for the full data, saves each numeric array as
   `.npy`, and supplies read-only mmap arrays to each training worker. No pickled
   objects or strings as array dtype; at most 64 GB total cache. You choose
   representations/resolutions, causal tracking and cache organization. Do not
   decode full images on each training update. Goal extraction can read the
   specified future anchor; live feature extraction must proceed causally.

   Mandatory arrays, all first dimension K (the actual eligible example count):
   - `example_episode`: integer [K], source episode ordinal in spec.trajectory_ids.
   - `example_source_index`: integer [K], original paired sample index.
   - `example_segment`: integer [K], ordinal in public.segments().
   - `native_action`: float [K,7], exactly the finite recorded command dq.
   - `loss_weight`: positive finite float [K], your declared time/segment weighting.

   Other arrays can have arbitrary numeric shapes, e.g. one cached feature per
   original frame plus mappings, goals, windows, masks or training-only targets.
   Do not duplicate all histories when a causal index map suffices. metadata
   must contain `num_examples`=K, `coverage` and your normalization/conversion
   metadata as JSON. Explain per-segment filtering counts and reasons. All
   examples must stay within their own declared supervision range/exclusions.
   No null command imputation, new cuts or use of another segment as history.
   Invalid perception may exclude a row with a recorded reason; do not silently
   skip all the difficult segments or claim excluded motion is trained.

2. `make_batch(policy_id, arrays, metadata, indices, spec)` -> your batch dict.
   indices is a numpy int64 vector selecting K examples. Construct tensors on
   `spec["device"]` ("cuda:0"). No inplace mutation of mmap arrays. The framework
   samples examples with replacement, weighted by loss_weight, using seed 0.
   Use the SAME common prepared dataset for both priors. policy-specific feature
   encoders/augmentations may differ. All histories/auxiliaries remain causal or
   explicitly training-only and bounded by segment/mask semantics.

3. `build_model(policy_id, spec)` -> a torch.nn.Module (1..64 million trainable
   parameters), containing all trainable components for this policy. Choose
   an actual learned conditional action distribution appropriate to its prior;
   no fixed replay/controller. spec contains `policy_id`, `training` (fixed
   recipe), `policy_config` (the full policies[policy_id] metadata), `preprocessing`,
   `data_metadata` (your prepared metadata), `source_plan`, `trajectory_ids`,
   `recording_metadata`, and `device`.

4. `compute_loss(model, batch, spec)` -> dict with differentiable scalar `loss`
   and any additional finite scalar tensor metrics. Each learned auxiliary
   must have a real gradient path. A constant penalty does not train a prior.

5. `predict(model, batch, spec)` -> finite tensor [B,7] of candidate native
   command dq, with no hardware IO. This is also used for checkpoint reload
   consistency on the same batch/RNG seed. Keep all normalizers in saved spec
   or model state. Shape of the returned tensor must match actual batch size.

6. `act(model, observation, call, memory, spec)` -> dict with `action` (finite
   native command dq[7] when valid, else null), `memory`, `status`, `diagnostics`.
   observation is ONE current record with `third_rgb`, `wrist_rgb` uint8 RGB,
   `q`, measured `dq`, `tau_ext`, `T_base_ee`, `T_base_flange`, gripper fields,
   `t`, `recv_time`, `sample_index`, `img_t_wrist`, `img_t_third`, `img_age_wrist`,
   `img_age_third`, and `metadata`. No future frames, action_json or training
   segment/object ID is provided online. call is your documented JSON/numeric
   argument contract for current scene instance, template, explicit spatial
   goal and invocation controls. memory is None on a new invocation. Its contents
   are your causal tracking/history state. Return reasoned unavailable status
   when required data are actually absent; do not use an always-unavailable stub.
   The same observation conversion must be shared with prepare_data. Preserve
   online identity even when an instance moves away from its original position.

7. `decode_action(action, controller_contract, spec)` -> dict with `enabled`,
   `command`, `reason`. No robot IO. Require explicit verified controller units,
   joint order, timing and limits before enabling an execution command; absence
   of that contract must not prevent offline training/prediction. This decoder
   is mandatory and must not fabricate hardware verification.

## Executor recipe and scope

20,000 optimizer updates PER policy, seed 0, AdamW, EMA decay .999. You choose
batch size, initial learning rate, weight decay, gradient clipping and model
config in package.json. The executor uses 500-step linear warmup then cosine
decay, logs every 100 steps, checkpoints every 2,000 steps, retains final EMA.
All original demonstrations are training data; there is no held-out performance
study or robot rollout in this round. No performance-based checkpoint selection.
Each worker has one 24-GiB RTX A5000 GPU; host memory is ample for the cache.

check_package performs syntax/import/metadata/hook checks ONLY, no optimizer
updates or preliminary experiment. submit_package freezes the checked source.
The actual preparation/training execution checks provenance, sample masks,
finite losses/gradients, changed weights and checkpoint reload. Implementation
errors can be returned for an explicitly recorded API repair revision; losses
are not offered as architecture-search feedback. Do not write your own optimizer,
training CLI, evaluation runner or hardware network client.

PRIOR.md must map each implemented mechanism and adaptation back to its frozen
heuristic. CALLING.md must completely specify raw observations, arguments,
coordinate frames, units, validity, memory, normalization, status, dependencies,
examples for the High-level Agent and decoder prerequisites.
