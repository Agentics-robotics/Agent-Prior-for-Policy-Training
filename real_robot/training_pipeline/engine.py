"""Trusted preparation, optimization and checkpoints for API-authored policies."""

from pathlib import Path
import copy
import hashlib
import importlib.util
import math
import random
import time

import numpy as np
import torch

from appl.io import ROOT, atomic, digest, object_hash, read
from real_robot.data import Sources
from . import public
from .security import lockdown


def module_at(source):
    definition = importlib.util.spec_from_file_location(
        "api_robot_policy", Path(source) / "policy.py"
    )
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module


def source_identity(source):
    return {p.name: digest(p) for p in Path(source).iterdir() if p.is_file()}


def specification(cfg, source):
    package = read(Path(source) / "package.json")
    cut_cfg = read(ROOT / cfg["cut_config"])
    sources = Sources(cut_cfg)
    plan = read(ROOT / cfg["cut_output"] / "plan.json")
    spec = dict(
        training=cfg["training"],
        preprocessing=package["preprocessing"],
        source_plan=plan,
        trajectory_ids=cut_cfg["trajectory_ids"],
        recording_metadata={tid: ep["meta"] for tid, ep in sources.episodes.items()},
        policy_ids=list(package["policies"]),
        device="cpu" if cfg.get("test_cpu", False) else "cuda:0",
    )
    segments = package["data_plan"]["segments"]
    for segment in segments:
        sources.interval(segment)
    spec["data_plan"] = package["data_plan"]
    return package, sources, spec, segments


def initialize(cfg, source, output):
    import cv2

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    package, sources, spec, segments = specification(cfg, source)
    assets = ROOT / cfg["run"] / "assets"
    public.configure(sources, segments, output, assets, source)
    identity = lockdown(
        source, output, [sources.base, ROOT / cfg["cut_output"], assets], gpu=not cfg.get("test_cpu", False)
    )
    return package, sources, spec, segments, identity


def validate_prepared(prepared, cfg, spec, segments, sources):
    """Validate source identity, causal windows and accounting, not a sampling recipe."""
    if set(prepared) != {"arrays", "metadata"}:
        raise ValueError("prepare_data must return arrays and metadata")
    arrays, metadata = prepared["arrays"], prepared["metadata"]
    vectors = {"example_episode", "example_source_index", "example_segment", "example_variant",
               "observation_start", "observation_stop", "target_start", "target_stop"}
    required = vectors | {"policy_weight"}
    if not required <= arrays.keys():
        raise ValueError("Missing provenance arrays: " + str(required - arrays.keys()))
    count = metadata["num_examples"]
    if type(count) is not int or count < 0:
        raise ValueError("Invalid example count")
    if not {"coverage", "source_accounting", "label_provenance", "data_audit", "variant_definition"} <= metadata.keys():
        raise ValueError("Document coverage, source accounting, targets, variants and data-quality findings")
    total = 0
    for name, array in arrays.items():
        if not name.isidentifier() or not isinstance(array, np.ndarray) or array.dtype.kind not in "buifc":
            raise ValueError("Cache requires named numeric numpy arrays: " + name)
        total += array.nbytes
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError("Use finite model values with explicit missingness masks: " + name)
    if total > cfg["training"]["max_cache_bytes"]:
        raise ValueError("Prepared cache exceeds configured resource bound")
    for name in vectors:
        if arrays[name].shape != (count,) or arrays[name].dtype.kind not in "iu":
            raise ValueError("Expected integer provenance vector: " + name)
    weight = arrays["policy_weight"]
    policies = spec["policy_ids"]
    if weight.shape != (count, len(policies)) or np.any(weight < 0) or (count and np.any(weight.sum(axis=1) <= 0)):
        raise ValueError("policy_weight[K,P] declares eligibility; each example needs a positive use")
    accounting = {tid: np.zeros(ep["n"], dtype=np.uint8) for tid, ep in sources.episodes.items()}
    bits = {"supervision": 1, "context": 2, "excluded": 4}
    for entry in metadata["source_accounting"]:
        ep = sources.interval(entry)
        if entry["use"] not in bits or not isinstance(entry["reason"], str) or not entry["reason"].strip():
            raise ValueError("Account for source use/exclusion with a reason")
        accounting[entry["trajectory_id"]][entry["start"]:entry["stop"]] |= bits[entry["use"]]
    if any(np.any(v == 0) for v in accounting.values()):
        raise ValueError("Every authorized source row requires an accounted use or exclusion")
    counts = {seg["segment_id"]: 0 for seg in segments}
    policy_counts = {policy: dict.fromkeys(counts, 0) for policy in policies}
    seen, anchors = set(), {tid: set() for tid in sources.episodes}
    for k in range(count):
        ep_index, index, ordinal, variant = (int(arrays[key][k]) for key in
            ("example_episode", "example_source_index", "example_segment", "example_variant"))
        if not 0 <= ep_index < len(spec["trajectory_ids"]) or not 0 <= ordinal < len(segments) or variant < 0:
            raise ValueError("Invalid source identity or example variant")
        segment = segments[ordinal]
        tid = spec["trajectory_ids"][ep_index]
        key = (ep_index, index, ordinal, variant)
        if key in seen:
            raise ValueError("Duplicate example identity; distinguish derived variants explicitly")
        seen.add(key)
        if segment["trajectory_id"] != tid or not segment["start"] <= index < segment["stop"]:
            raise ValueError("Example outside API-declared source interval")
        if not accounting[tid][index] & bits["supervision"]:
            raise ValueError("Example contradicts declared source use")
        if not segment["start"] <= arrays["observation_start"][k] < arrays["observation_stop"][k] <= index + 1:
            raise ValueError("Observation window must be causal and inside declared interval")
        if not segment["start"] <= arrays["target_start"][k] < arrays["target_stop"][k] <= segment["stop"]:
            raise ValueError("Target evidence crosses declared interval")
        anchors[tid].add(index)
        counts[segment["segment_id"]] += 1
        for j, policy in enumerate(policies):
            if weight[k, j] > 0:
                policy_counts[policy][segment["segment_id"]] += 1
    source_coverage = {tid: dict(
        original_rows=len(mask), unique_supervised_anchors=len(anchors[tid]),
        declared_use_rows={use: int(np.count_nonzero(mask & bit)) for use, bit in bits.items()},
    ) for tid, mask in accounting.items()}
    return dict(num_examples=count, segment_examples=counts,
                empty_segments=[k for k, v in counts.items() if not v],
                policy_segment_examples=policy_counts,
                policy_examples={p: sum(v.values()) for p, v in policy_counts.items()},
                cache_bytes=total, source_coverage=source_coverage)


def prepare(cfg, source, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_identity(source)
    atomic(output / "request.json", dict(
        operation="full_data_preparation", source_hashes=hashes,
        training_updates=0, started=time.time()))
    package, sources, spec, segments, device = initialize(cfg, source, output)
    module = module_at(source)
    started = time.monotonic()
    prepared = module.prepare_data(spec)
    validated = validate_prepared(prepared, cfg, spec, segments, sources)
    arrays, metadata = prepared["arrays"], prepared["metadata"]
    interface_checks = {}
    for policy_id in spec["policy_ids"]:
        params = package["policies"][policy_id]
        model_spec = dict(spec, policy_id=policy_id, policy_config=params, data_metadata=metadata)
        col = spec["policy_ids"].index(policy_id)
        indices = np.flatnonzero(arrays["policy_weight"][:, col] > 0)[:2]
        if not len(indices):
            interface_checks[policy_id] = dict(passed=False, reason="No prepared examples", optimizer_steps=0)
            continue
        torch.manual_seed(params["seed"])
        np.random.seed(params["seed"])
        random.seed(params["seed"])
        model = module.build_model(policy_id, model_spec).to(spec["device"])
        parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if not 0 < parameter_count <= cfg["training"]["parameter_limit"]:
            raise ValueError("Invalid model parameter count")
        batch = module.make_batch(policy_id, arrays, metadata, indices, model_spec)
        values = module.compute_loss(model, batch, model_spec)
        loss = values["loss"]
        if loss.ndim != 0 or not torch.isfinite(loss) or not loss.requires_grad:
            raise ValueError("Model interface loss must be a finite differentiable scalar")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients) or not any(g.abs().max() > 0 for g in gradients):
            raise ValueError("Model interface needs finite nonzero gradients")
        model.eval()
        with torch.no_grad():
            predicted = module.predict(model, batch, model_spec)
        if predicted.shape != (len(indices), params["prediction_dimension"]) or not torch.isfinite(predicted).all():
            raise ValueError("Prediction interface disagrees with declared shape")
        interface_checks[policy_id] = dict(passed=True, trainable_parameters=parameter_count, prediction_shape=list(predicted.shape), finite_nonzero_gradients=True, optimizer_steps=0)
        del model, batch, values, loss, gradients, predicted
    files = {}
    for name, array in arrays.items():
        file = output / (name + ".npy")
        np.save(file, array, allow_pickle=False)
        files[name] = dict(
            file=file.name,
            shape=list(array.shape),
            dtype=str(array.dtype),
            sha256=digest(file),
            bytes=array.nbytes,
        )
    atomic(output / "metadata.json", metadata)
    manifest = dict(
        schema="real_robot.prepared_cache.pipeline.v1",
        source_hashes=hashes,
        source_plan_hash=object_hash(spec["source_plan"]),
        arrays=files,
        metadata_sha256=digest(output / "metadata.json"),
        **validated,
        device=device,
        elapsed_seconds=time.monotonic() - started,
        preparation_is_training=False,
        API_declared_coverage=metadata["coverage"],
        API_data_audit=metadata["data_audit"],
        model_interface_checks=interface_checks,
    )
    atomic(output / "result.json", manifest)
    print(__import__("json").dumps(manifest), flush=True)
    return manifest


def tensor_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        value = tensor.detach().cpu().contiguous()
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def calling_check(module, model, arrays, metadata, spec):
    """Exercise the actual act -> structured output -> executor request path."""
    import json
    import jsonschema
    from .inference import wire, unwire
    online_spec = {k: v for k, v in spec.items()
                   if k not in {"source_plan", "trajectory_ids", "recording_metadata", "data_plan", "data_metadata"}}
    cases = module.calling_cases(arrays, metadata, spec)
    if not isinstance(cases, list) or not cases:
        raise ValueError("Supply executable calling cases with expected decisions/statuses")
    checked, memory, produced = [], None, False
    for case in cases:
        if case["reset"]:
            memory = None
        observation = unwire(wire(case["observation"]))
        result = module.act(model, observation, case["call"], memory, online_spec)
        if not isinstance(result, dict) or not {"memory", "decision", "status"} <= result.keys():
            raise ValueError("act requires memory, decision and status")
        memory, decision = result["memory"], result["decision"]
        if result["status"] != case["expected_status"] or (decision is not None) != case["expect_decision"]:
            raise ValueError("Calling behavior differs from API-declared expectation: " + case["name"])
        if decision is not None:
            produced = True
            jsonschema.validate(decision, spec["policy_config"]["decision_schema"])
            json.dumps(decision, allow_nan=False)
            request = module.executor_request(decision, case["executor_contract"], online_spec)
            json.dumps(request, allow_nan=False)
        json.dumps(wire({k: v for k, v in result.items() if k != "memory"}), allow_nan=False)
        checked.append(dict(name=case["name"], status=result["status"], decision=decision is not None))
    if not produced:
        raise ValueError("At least one calling case must exercise a non-null policy decision")
    return dict(passed=True, cases=checked, hardware_io=False, performance_evaluation=False)


def fit(cfg, source, cache, output, policy_id):
    import json
    output, cache = Path(output), Path(cache)
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_identity(source)
    manifest = read(cache / "result.json")
    if hashes != manifest["source_hashes"] or digest(cache / "metadata.json") != manifest["metadata_sha256"]:
        raise ValueError("Prepared source/metadata identity changed")
    for entry in manifest["arrays"].values():
        if digest(cache / entry["file"]) != entry["sha256"]:
            raise ValueError("Prepared numerical cache changed")
    arrays = {name: np.load(cache / entry["file"], mmap_mode="r", allow_pickle=False)
              for name, entry in manifest["arrays"].items()}
    metadata = read(cache / "metadata.json")
    package, sources, spec, segments = specification(cfg, source)
    spec.update(policy_id=policy_id, policy_config=package["policies"][policy_id], data_metadata=metadata)
    training, params = cfg["training"], spec["policy_config"]
    updates = params["updates"]
    if updates > training["max_updates"]:
        raise ValueError("API budget exceeds configured execution limit")
    request = dict(policy_id=policy_id, source_hashes=hashes,
                   cache_manifest_sha256=digest(cache / "result.json"), spec=spec,
                   seed=params["seed"], maximum_optimizer_steps=updates,
                   checkpoint_selection="API after_update selection, from raw model or optional EMA")
    if (output / "request.json").exists():
        raise ValueError("Training attempt already exists; retain a new implementation revision")
    atomic(output / "request.json", request)
    torch.set_num_threads(2)
    public.configure(sources, segments, output, ROOT / cfg["run"] / "assets", source)
    device = lockdown(source, output,
                      [cache, sources.base, ROOT / cfg["cut_output"], ROOT / cfg["run"] / "assets"],
                      gpu=not cfg.get("test_cpu", False))
    torch.manual_seed(params["seed"])
    np.random.seed(params["seed"])
    random.seed(params["seed"])
    module = module_at(source)
    model = module.build_model(policy_id, spec).to(spec["device"])
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not 0 < count <= training["parameter_limit"]:
        raise ValueError("Model exceeds parameter limit")
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    initial_hash = tensor_hash(initial)
    ema = copy.deepcopy(model).eval().requires_grad_(False) if params["ema_decay"] is not None else None
    optimizer = module.configure_optimizer(model, spec)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError("configure_optimizer must return a torch optimizer")
    model_parameters = {id(p) for p in model.parameters()}
    if any(id(p) not in model_parameters for g in optimizer.param_groups for p in g["params"]):
        raise ValueError("Optimizer owns parameters outside the submitted model")
    sampler = np.random.default_rng(params["seed"])
    eligible = np.flatnonzero(arrays["policy_weight"][:, spec["policy_ids"].index(policy_id)] > 0)
    if not len(eligible):
        raise ValueError("No training examples for this policy")
    began, selected, selected_step, actual_exposures = time.monotonic(), None, None, 0
    losses, gradient_checks = [], []
    for step in range(1, updates + 1):
        model.train()
        module.before_update(optimizer, step, spec)
        indices = module.sample_indices(policy_id, arrays, metadata, sampler, step, spec)
        if not isinstance(indices, np.ndarray) or indices.ndim != 1 or indices.dtype.kind not in "iu" or not 1 <= len(indices) <= params["batch_size"] or not np.isin(indices, eligible).all():
            raise ValueError("API sampler must return eligible integer indices within declared batch bound")
        actual_exposures += len(indices)
        batch = module.make_batch(policy_id, arrays, metadata, indices, spec)
        values = module.compute_loss(model, batch, spec)
        if not isinstance(values, dict) or "loss" not in values:
            raise ValueError("compute_loss must return loss and optional scalar metrics")
        metrics = {}
        for key, value in values.items():
            if not isinstance(value, torch.Tensor) or value.ndim != 0 or not torch.isfinite(value):
                raise ValueError("Loss metrics must be finite scalar tensors: " + key)
            metrics[key] = float(value.detach())
        if not values["loss"].requires_grad:
            raise ValueError("Detached training loss")
        optimizer.zero_grad(set_to_none=True)
        values["loss"].backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
            params["max_grad_norm"] if params["max_grad_norm"] is not None else float("inf"),
            error_if_nonfinite=True)
        if step <= 3:
            gradient_checks.append(dict(step=step, l2_norm=float(norm)))
        optimizer.step()
        if ema is not None:
            with torch.no_grad():
                for a, b in zip(ema.parameters(), model.parameters(), strict=True):
                    a.lerp_(b, 1 - params["ema_decay"])
                for a, b in zip(ema.buffers(), model.buffers(), strict=True):
                    a.copy_(b)
        model.eval()
        with torch.no_grad():
            decision = module.after_update(model, ema, arrays, metadata, step, metrics, spec)
        if set(decision) != {"stop", "select", "metrics"} or type(decision["stop"]) is not bool or decision["select"] not in {None, "model", "ema"}:
            raise ValueError("after_update requires stop, select and metrics")
        json.dumps(decision, allow_nan=False)
        if decision["select"] is not None:
            chosen = ema if decision["select"] == "ema" else model
            if chosen is None:
                raise ValueError("Cannot select disabled EMA")
            selected = {k: v.detach().cpu().clone() for k, v in chosen.state_dict().items()}
            selected_step = step
        losses.append(metrics["loss"])
        ending = step == updates or decision["stop"]
        if step == 1 or step % training["log_every"] == 0 or ending:
            progress = dict(policy_id=policy_id, step=step, maximum_updates=updates,
                            loss_mean=float(np.mean(losses)), metrics=metrics,
                            API_training_decision=decision, gradient_norm=float(norm),
                            elapsed_seconds=time.monotonic() - began)
            atomic(output / "progress.json", progress)
            with (output / "learning.jsonl").open("a") as file:
                file.write(json.dumps(progress) + "\n")
            print(json.dumps(progress), flush=True)
            losses.clear()
        if step % training["checkpoint_every"] == 0 or ending:
            save_checkpoint(output / "last.pt", dict(
                model=model.state_dict(), ema=ema.state_dict() if ema is not None else None,
                selected=selected, selected_step=selected_step, optimizer=optimizer.state_dict(),
                step=step, spec=spec, source_hashes=hashes, request_hash=object_hash(request),
                initial_weights_sha256=initial_hash, numpy_sampler_state=sampler.bit_generator.state,
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                elapsed_seconds=time.monotonic() - began))
        if ending:
            break
    delta = math.sqrt(sum((v.detach().cpu().double() - initial[k].double()).square().sum().item()
                         for k, v in model.state_dict().items()))
    if delta <= 0 or selected is None:
        raise ValueError("Training must change weights and explicitly select a checkpoint")
    selected_model = module.build_model(policy_id, spec).to(spec["device"])
    selected_model.load_state_dict(selected)
    selected_model.eval()
    indices = eligible[:2].astype(np.int64)
    batch = module.make_batch(policy_id, arrays, metadata, indices, spec)
    with torch.no_grad():
        torch.manual_seed(913); np.random.seed(913); random.seed(913)
        predicted = module.predict(selected_model, batch, spec)
        if predicted.shape != (len(indices), params["prediction_dimension"]) or not torch.isfinite(predicted).all():
            raise ValueError("Prediction differs from declared finite [B,D] interface")
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        reloaded = module.build_model(policy_id, spec).to(spec["device"])
        reloaded.load_state_dict(saved["selected"])
        reloaded.eval()
        torch.manual_seed(913); np.random.seed(913); random.seed(913)
        replay = module.predict(reloaded, batch, spec)
        error = float((predicted - replay).abs().max())
        if error > 1e-6:
            raise ValueError("Checkpoint reload changed predictions")
        calls = calling_check(module, reloaded, arrays, metadata, spec)
    atomic(output / "calling_check.json", calls)
    result = dict(policy_id=policy_id, neural_training_verified=True, optimizer_steps=step,
                  selected_step=selected_step, optimizer_class=type(optimizer).__name__,
                  trainable_parameters=count, initial_weights_sha256=initial_hash,
                  final_weights_sha256=tensor_hash(model.state_dict()), weight_l2_change=delta,
                  finite_gradient_checks=gradient_checks, checkpoint_sha256=digest(output / "last.pt"),
                  checkpoint=str(output / "last.pt"), reload_max_prediction_error=error,
                  elapsed_seconds=time.monotonic()-began, device=device, source_hashes=hashes,
                  training_examples=len(eligible), sample_exposures=actual_exposures,
                  calling_check=calls, performance_evaluation=False, robot_execution=False)
    atomic(output / "result.json", result)
    print(json.dumps(result), flush=True)
    return result
