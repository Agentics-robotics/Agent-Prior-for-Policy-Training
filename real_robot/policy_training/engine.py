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
        device="cuda:0",
    )
    segments = [s for g in plan["skills"] for s in g["segments"]]
    return package, sources, spec, segments


def initialize(cfg, source, output):
    import cv2

    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])
    random.seed(cfg["training"]["seed"])
    package, sources, spec, segments = specification(cfg, source)
    assets = ROOT / cfg["run"] / "assets"
    public.configure(sources, segments, output, assets)
    identity = lockdown(
        source, output, [sources.base, ROOT / cfg["cut_output"], assets], gpu=True
    )
    return package, sources, spec, segments, identity


def prepare(cfg, source, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_identity(source)
    atomic(
        output / "request.json",
        dict(
            operation="full_data_preparation",
            source_hashes=hashes,
            training_updates=0,
            started=time.time(),
        ),
    )
    package, sources, spec, segments, device = initialize(cfg, source, output)
    module = module_at(source)
    started = time.monotonic()
    prepared = module.prepare_data(spec)
    if set(prepared) != {"arrays", "metadata"}:
        raise ValueError("prepare_data must return arrays and metadata")
    arrays, metadata = prepared["arrays"], prepared["metadata"]
    required = {
        "example_episode",
        "example_source_index",
        "example_segment",
        "native_action",
        "loss_weight",
    }
    if not required <= arrays.keys():
        raise ValueError(
            "Missing provenance/action arrays: " + str(required - arrays.keys())
        )
    count = metadata["num_examples"]
    if type(count) is not int or count < 1:
        raise ValueError("No eligible examples returned")
    if "coverage" not in metadata:
        raise ValueError("API preprocessing must account for retained/excluded records")
    total = 0
    for name, array in arrays.items():
        if (
            not name.isidentifier()
            or not isinstance(array, np.ndarray)
            or array.dtype.kind not in "buifc"
        ):
            raise ValueError("Cache requires named numeric numpy arrays: " + name)
        total += array.nbytes
    if total > cfg["training"]["max_cache_bytes"]:
        raise ValueError("Prepared cache exceeds declared 64 GB bound")
    for name in required:
        if len(arrays[name]) != count:
            raise ValueError("Example count mismatch: " + name)
    for name in ["example_episode", "example_source_index", "example_segment"]:
        if arrays[name].shape != (count,) or arrays[name].dtype.kind not in "iu":
            raise ValueError("Expected integer provenance vector: " + name)
    if (
        arrays["native_action"].shape != (count, 7)
        or not np.isfinite(arrays["native_action"]).all()
    ):
        raise ValueError("Expected finite native command dq[K,7]")
    weight = arrays["loss_weight"]
    if weight.shape != (count,) or not np.isfinite(weight).all() or np.any(weight <= 0):
        raise ValueError("loss_weight must be finite positive K-vector")
    counts = {s["segment_id"]: 0 for s in segments}
    import json

    for k in range(count):
        ep_index = int(arrays["example_episode"][k])
        index = int(arrays["example_source_index"][k])
        ordinal = int(arrays["example_segment"][k])
        if not 0 <= ep_index < len(spec["trajectory_ids"]) or not 0 <= ordinal < len(
            segments
        ):
            raise ValueError("Unknown source identity in cache")
        segment = segments[ordinal]
        tid = spec["trajectory_ids"][ep_index]
        if (
            segment["trajectory_id"] != tid
            or not segment["supervised_start"] <= index < segment["supervised_stop"]
            or any(
                e["start"] <= index < e["stop"]
                for e in segment["supervision_exclusions"]
            )
        ):
            raise ValueError("Supervision example violates frozen segment/masks")
        action = json.loads(
            str(sources.episode(tid)["arrays"]["action_json"][index])
        ).get("dq")
        if action is None or not np.allclose(
            np.asarray(action), arrays["native_action"][k], rtol=1e-6, atol=1e-7
        ):
            raise ValueError("Native action label differs from recorded command dq")
        counts[segment["segment_id"]] += 1
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
        schema="real_robot.prepared_cache.v1",
        source_hashes=hashes,
        source_plan_hash=object_hash(spec["source_plan"]),
        arrays=files,
        metadata_sha256=digest(output / "metadata.json"),
        num_examples=count,
        segment_examples=counts,
        empty_segments=[k for k, v in counts.items() if not v],
        cache_bytes=total,
        device=device,
        elapsed_seconds=time.monotonic() - started,
        preparation_is_training=False,
        API_declared_coverage=metadata["coverage"],
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


def fit(cfg, source, cache, output, policy_id):
    output, cache = Path(output), Path(cache)
    output.mkdir(parents=True, exist_ok=True)
    hashes = source_identity(source)
    manifest = read(cache / "result.json")
    if hashes != manifest["source_hashes"]:
        raise ValueError("Prepared data and policy source identity differ")
    arrays = {
        name: np.load(cache / entry["file"], mmap_mode="r", allow_pickle=False)
        for name, entry in manifest["arrays"].items()
    }
    metadata = read(cache / "metadata.json")
    package, sources, spec, segments = specification(cfg, source)
    spec.update(
        policy_id=policy_id,
        policy_config=package["policies"][policy_id],
        data_metadata=metadata,
    )
    training = cfg["training"]
    params = spec["policy_config"]
    updates = training["updates"]
    request = dict(
        policy_id=policy_id,
        source_hashes=hashes,
        cache_manifest_sha256=digest(cache / "result.json"),
        metadata_sha256=digest(cache / "metadata.json"),
        spec=spec,
        seed=training["seed"],
        optimizer_steps=updates,
        checkpoint_selection="Final EMA at the fixed budget; no performance selection",
    )
    if (output / "request.json").exists():
        raise ValueError("Training attempt already exists; no silent restart")
    atomic(output / "request.json", request)
    torch.set_num_threads(2)
    public.configure(sources, segments, output, ROOT / cfg["run"] / "assets")
    device = lockdown(
        source,
        output,
        [cache, sources.base, ROOT / cfg["cut_output"], ROOT / cfg["run"] / "assets"],
        gpu=True,
    )
    torch.manual_seed(training["seed"])
    torch.cuda.manual_seed_all(training["seed"])
    np.random.seed(training["seed"])
    random.seed(training["seed"])
    module = module_at(source)
    model = module.build_model(policy_id, spec).to("cuda:0")
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not 0 < count <= training["parameter_limit"]:
        raise ValueError("Trainable parameter limit is 1..64 million")
    initial = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    initial_hash = tensor_hash(initial)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    sampler = np.random.default_rng(training["seed"])
    probabilities = np.asarray(arrays["loss_weight"], dtype=np.float64).copy()
    probabilities /= probabilities.sum()
    model.train()
    began = time.monotonic()
    losses = []
    gradient_checks = []
    parameter_steps = 0
    for step in range(1, updates + 1):
        factor = (
            step / 500
            if step <= 500
            else 0.5 * (1 + math.cos(math.pi * (step - 500) / (updates - 500)))
        )
        for group in optimizer.param_groups:
            group["lr"] = params["learning_rate"] * factor
        indices = sampler.choice(
            metadata["num_examples"],
            size=params["batch_size"],
            replace=True,
            p=probabilities,
        )
        batch = module.make_batch(policy_id, arrays, metadata, indices, spec)
        values = module.compute_loss(model, batch, spec)
        if not isinstance(values, dict) or "loss" not in values:
            raise ValueError("compute_loss must return a dict containing loss")
        metrics = {}
        for key, value in values.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 0
                or not torch.isfinite(value)
            ):
                raise ValueError("Loss metrics must be finite scalar tensors: " + key)
            metrics[key] = float(value.detach())
        loss = values["loss"]
        if not loss.requires_grad:
            raise ValueError("Detached training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), params["max_grad_norm"], error_if_nonfinite=True
        )
        if step <= 3 or step == updates:
            if not float(norm) > 0:
                raise ValueError("No effective neural gradient")
            gradient_checks.append(dict(step=step, l2_norm=float(norm)))
        optimizer.step()
        parameter_steps += 1
        with torch.no_grad():
            for a, b in zip(ema.parameters(), model.parameters(), strict=True):
                a.lerp_(b, 1 - training["ema_decay"])
            for a, b in zip(ema.buffers(), model.buffers(), strict=True):
                a.copy_(b)
        losses.append(metrics["loss"])
        if step == 1 or step % training["log_every"] == 0 or step == updates:
            progress = dict(
                policy_id=policy_id,
                step=step,
                updates=updates,
                loss_mean=float(np.mean(losses)),
                metrics=metrics,
                gradient_norm=float(norm),
                elapsed_seconds=time.monotonic() - began,
            )
            atomic(output / "progress.json", progress)
            with (output / "learning.jsonl").open("a") as file:
                file.write(__import__("json").dumps(progress) + "\n")
            print(__import__("json").dumps(progress), flush=True)
            losses.clear()
        if step % training["checkpoint_every"] == 0 or step == updates:
            save_checkpoint(
                output / "last.pt",
                dict(
                    model=model.state_dict(),
                    ema=ema.state_dict(),
                    optimizer=optimizer.state_dict(),
                    step=step,
                    spec=spec,
                    source_hashes=hashes,
                    request_hash=object_hash(request),
                    initial_weights_sha256=initial_hash,
                    numpy_sampler_state=sampler.bit_generator.state,
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),
                    elapsed_seconds=time.monotonic() - began,
                ),
            )
    delta = math.sqrt(
        sum(
            (value.detach().cpu() - initial[name]).double().square().sum().item()
            for name, value in model.state_dict().items()
        )
    )
    if delta <= 0:
        raise ValueError("Weights did not change")
    indices = np.arange(min(2, metadata["num_examples"]), dtype=np.int64)
    batch = module.make_batch(policy_id, arrays, metadata, indices, spec)
    ema.eval()
    with torch.no_grad():
        torch.manual_seed(913)
        np.random.seed(913)
        random.seed(913)
        predicted = module.predict(ema, batch, spec)
        if predicted.shape != (len(indices), 7) or not torch.isfinite(predicted).all():
            raise ValueError("Prediction must be finite candidate dq[B,7]")
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        reloaded = module.build_model(policy_id, spec).to("cuda:0")
        reloaded.load_state_dict(saved["ema"])
        reloaded.eval()
        torch.manual_seed(913)
        np.random.seed(913)
        random.seed(913)
        replay = module.predict(reloaded, batch, spec)
        error = float((predicted - replay).abs().max())
        if error > 1e-6:
            raise ValueError("Checkpoint reload changed candidate actions")
    result = dict(
        policy_id=policy_id,
        neural_training_verified=True,
        optimizer_steps=parameter_steps,
        trainable_parameters=count,
        initial_weights_sha256=initial_hash,
        final_weights_sha256=tensor_hash(model.state_dict()),
        weight_l2_change=delta,
        finite_nonzero_gradient_checks=gradient_checks,
        checkpoint_sha256=digest(output / "last.pt"),
        checkpoint=str(output / "last.pt"),
        reload_max_action_error=error,
        elapsed_seconds=time.monotonic() - began,
        device=device,
        source_hashes=hashes,
        training_examples=metadata["num_examples"],
        sample_exposures=updates * params["batch_size"],
        performance_evaluation=False,
        robot_execution=False,
    )
    atomic(output / "result.json", result)
    print(__import__("json").dumps(result), flush=True)
    return result
