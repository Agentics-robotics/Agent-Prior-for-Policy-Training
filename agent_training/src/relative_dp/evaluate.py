"""Scalar, paired, resumable EMA evaluation and post-freeze video replay.

Every episode owns its diffusion generator. Development, final tests, and video
replay therefore cannot consume one another's sampling randomness.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .config import CONFIG, RUNS
from .utils import ROOT, atomic_json, object_hash, read_json, sha256


EVALUATION_VERSION = 1
CHECKPOINT_STEPS = (5000, 10000, 20000)
RESULTS = ROOT / "results"
SELECTION_PATH = RESULTS / "checkpoint_selection.json"
FREEZE_PATH = RESULTS / "test_freeze.json"


def sampling_seed(task: str, split: str, episode_id: str) -> int:
    """Stable, representation-independent seeds; never Python's salted hash."""
    token = f"relative-dp-round1/evaluation-v1/{task}/{split}/{episode_id}"
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little") % (2**63 - 1)


def _implementation_hash() -> str:
    names = ("evaluate.py", "model.py", "environment.py", "representation.py", "dataset.py", "train.py", "config.py")
    files = {name: sha256(Path(__file__).with_name(name)) for name in names}
    for path in sorted((Path(__file__).parent / "vendor" / "diffusion_policy").glob("*.py")):
        files[str(path.relative_to(Path(__file__).parent))] = sha256(path)
    files["pixi.lock"] = sha256(ROOT / "pixi.lock")
    return object_hash(files)


def _task_records(manifest: dict, task: str, split: str) -> list[dict]:
    records = manifest["tasks"][task]["splits"][split]
    if len(records) != 20:
        raise ValueError(f"{task}/{split}: expected exactly 20 records, got {len(records)}")
    ids = [record["episode_id"] for record in records]
    if len(set(ids)) != 20:
        raise ValueError(f"Repeated initial-condition IDs in {task}/{split}")
    return records


def checkpoint_path(run_id: str, step: int) -> Path:
    return ROOT / "runs" / run_id / "checkpoints" / f"step_{step:06d}.pt"


def result_path(run_id: str, split: str, step: int | None = None) -> Path:
    suffix = f"dev_step{step:06d}" if split == "dev" else split
    return RESULTS / run_id / f"{suffix}.json"


def summarize_episodes(episodes: list[dict]) -> dict:
    successful = [episode for episode in episodes if episode["success"]]
    return {
        "n": len(episodes),
        "successes": len(successful),
        "success_rate": len(successful) / len(episodes) if episodes else None,
        "mean_success_steps": float(np.mean([e["first_success_step"] for e in successful])) if successful else None,
        "elapsed_seconds": sum(e["elapsed_seconds"] for e in episodes),
        "exceptions": sum(e.get("exception") is not None for e in episodes),
    }


def select_checkpoint(candidates: list[dict]) -> dict:
    """Predeclared success -> successful steps -> earliest update tie-break."""
    if not candidates:
        raise ValueError("No development candidates")
    def key(candidate: dict) -> tuple:
        summary = candidate["summary"]
        steps = summary["mean_success_steps"]
        return (-summary["success_rate"], float("inf") if steps is None else steps, candidate["step"])
    return min(candidates, key=key)


def _json_scalars(info: dict) -> dict:
    return {str(key): value.item() if isinstance(value, np.generic) else value
            for key, value in info.items()
            if isinstance(value, (str, bool, int, float, np.generic))}


def rollout(policy: Any, env: Any, record: dict, seed: int,
            max_steps: int = 500, frame_callback=None) -> dict:
    """History is (o[t-1],o[t]); execute prediction[0:4], stop immediately."""
    from .environment import get_base_position, reset_from_record

    start = time.perf_counter()
    native = getattr(env, "unwrapped", env)
    max_steps = min(max_steps, getattr(native, "max_path_length", max_steps))
    generator = torch.Generator(device=policy.device).manual_seed(seed)
    length, total_return, first_success = 0, 0.0, None
    terminated, truncated, exception = False, False, None
    trace = hashlib.sha256()
    final_info: dict = {}
    diagnostic_maxima: dict[str, float] = {}
    initial_position = record.get("initial_base_position")
    actual_position = None
    try:
        obs, _ = reset_from_record(env, record)
        obs = np.asarray(obs, dtype=np.float32).copy()
        actual_position = np.asarray(get_base_position(env)).tolist()
        if initial_position is not None and not np.allclose(actual_position, initial_position, atol=1e-8, rtol=0):
            raise ValueError(f"Restored base position differs: {actual_position} != {initial_position}")
        history = deque([obs.copy(), obs.copy()], maxlen=2)
        trace.update(obs.tobytes())
        if frame_callback is not None:
            frame_callback(env.render())
        while length < max_steps:
            actions = np.asarray(policy.predict_actions(np.stack(history), generator=generator), dtype=np.float32)
            if actions.shape != (16, 4) or not np.isfinite(actions).all():
                raise ValueError(f"Policy returned invalid action chunk: {actions.shape}")
            for predicted in actions[:4]:
                action = np.clip(predicted, env.action_space.low, env.action_space.high).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                length += 1
                total_return += float(reward)
                obs = np.asarray(obs, dtype=np.float32).copy()
                history.append(obs)
                trace.update(action.tobytes())
                trace.update(obs.tobytes())
                final_info = _json_scalars(info)
                for name in ("near_object", "grasp_success", "in_place_reward"):
                    if name in info and np.isscalar(info[name]):
                        diagnostic_maxima[name] = max(diagnostic_maxima.get(name, -float("inf")), float(info[name]))
                if bool(info.get("success", False)) and first_success is None:
                    first_success = length
                if frame_callback is not None:
                    frame_callback(env.render())
                if first_success is not None or terminated or truncated or length >= max_steps:
                    break
            if first_success is not None or terminated or truncated or length >= max_steps:
                break
    except Exception as error:
        exception = f"{type(error).__name__}: {error}"

    success = first_success is not None
    if exception:
        reason = "runtime_exception"
    elif success:
        reason = None
    elif truncated:
        reason = "environment_truncated"
    elif terminated:
        reason = "environment_terminated_without_success"
    else:
        reason = "episode_step_limit"
    return {
        "episode_id": record["episode_id"],
        "initial_base_position": initial_position,
        "restored_initial_base_position": actual_position,
        "env_seed": record["seed"],
        "task_params": record.get("task_params"),
        "initial_condition_hash": object_hash(record),
        "policy_sampling_seed": seed,
        "success": success,
        "first_success_step": first_success,
        "episode_length": length,
        "return": total_return,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "reached_step_limit": length >= max_steps,
        "failure_reason": reason,
        "exception": exception,
        "elapsed_seconds": time.perf_counter() - start,
        "trajectory_sha256": trace.hexdigest(),
        "final_info": final_info,
        "diagnostic_maxima": diagnostic_maxima,
    }


def _cache_identity(run: dict, step: int, split: str, records: list[dict], manifest_hash: str,
                    device: str | None = None) -> dict:
    from .train import hardware_info
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "evaluation_version": EVALUATION_VERSION,
        "implementation_hash": _implementation_hash(),
        "run": run,
        "step": step,
        "split": split,
        "checkpoint_hash": sha256(checkpoint_path(run["run_id"], step)),
        "data_manifest_hash": manifest_hash,
        "initial_conditions_hash": object_hash(records),
        "policy_sampling_seeds": [sampling_seed(run["task"], split, r["episode_id"]) for r in records],
        "weights": "EMA",
        "max_episode_steps": 500,
        "n_obs_steps": 2,
        "executed_action_slice": [0, 4],
        "inference_steps": CONFIG["inference_steps"],
        "ddim_eta": CONFIG["ddim_eta"],
        "hardware": hardware_info(device),
        "torch_deterministic_algorithms": True,
    }


def _read_cached(path: Path, identity: dict, records: list[dict]) -> dict:
    if not path.exists():
        return {"identity": identity, "complete": False, "episodes": []}
    cache = read_json(path)
    if cache.get("identity") != identity:
        raise ValueError(f"Stale evaluation identity at {path}; preserve and explicitly invalidate old records before rerunning")
    episodes = cache.get("episodes", [])
    if [r["episode_id"] for r in episodes] != [r["episode_id"] for r in records[:len(episodes)]]:
        raise ValueError(f"Evaluation episode order mismatch at {path}")
    if len(episodes) > len(records) or any(e.get("exception") for e in episodes):
        raise ValueError(f"Invalid or exceptional evaluation cache at {path}; inspect before retrying")
    if cache.get("complete") and len(episodes) != len(records):
        raise ValueError(f"Incomplete episode count marked complete at {path}")
    return cache


def evaluate_checkpoint(run: dict, step: int, split: str, manifest: dict,
                        manifest_hash: str, device: str | None = None) -> dict:
    from .environment import make_env
    from .model import load_policy
    from .train import runtime_setup

    runtime_setup()
    records = _task_records(manifest, run["task"], split)
    identity = _cache_identity(run, step, split, records, manifest_hash, device=device)
    path = result_path(run["run_id"], split, step)
    cache = _read_cached(path, identity, records)
    if cache.get("complete"):
        return cache
    policy = load_policy(checkpoint_path(run["run_id"], step), device=device)
    env = make_env(run["task"])
    try:
        for index in range(len(cache["episodes"]), len(records)):
            record = records[index]
            episode = rollout(policy, env, record, identity["policy_sampling_seeds"][index])
            episode.update({
                "run_id": run["run_id"], "task": run["task_name"], "representation": run["representation"],
                "train_n": run["train_n"], "train_seed": run["train_seed"], "checkpoint_step": step,
                "checkpoint_hash": identity["checkpoint_hash"], "data_manifest_hash": manifest_hash, "split": split,
            })
            cache["episodes"].append(episode)
            cache["summary"] = summarize_episodes(cache["episodes"])
            cache["complete"] = len(cache["episodes"]) == 20 and not episode["exception"]
            atomic_json(path, cache)
            print(f"eval {run['run_id']} step={step} {split} {index+1}/20 success={episode['success']} steps={episode['episode_length']}", flush=True)
            if episode["exception"]:
                raise RuntimeError(f"Evaluation aborted with preserved record: {episode['exception']}")
    finally:
        env.close()
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return cache


def _validate_formal_runs(manifest_hash: str) -> None:
    from .train import validate_run_complete
    for run in RUNS:
        complete_path = ROOT / "runs" / run["run_id"] / "complete.json"
        if not complete_path.exists():
            raise RuntimeError(f"Formal training incomplete: {run['run_id']}")
        if not validate_run_complete(run["run_id"], raise_error=True):
            raise RuntimeError(f"Formal training validation failed: {run['run_id']}")
        for step in CHECKPOINT_STEPS:
            path = checkpoint_path(run["run_id"], step)
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("step", checkpoint.get("update")) != step:
                raise ValueError(f"Checkpoint update mismatch: {path}")
            if checkpoint.get("manifest_hash") != manifest_hash or "ema" not in checkpoint:
                raise ValueError(f"Checkpoint data/EMA mismatch: {path}")
            checkpoint_run = checkpoint.get("run_id", checkpoint.get("config", {}).get("run_id"))
            if checkpoint_run != run["run_id"]:
                raise ValueError(f"Checkpoint run mismatch: {path}")


def evaluate_round1(*, videos: bool = True, device: str | None = None) -> dict:
    """All twelve dev evaluations and all four selections precede any test."""
    manifest_path = ROOT / "data" / "dataset_manifest.json"
    from .collection import verify_manifest
    manifest = verify_manifest(manifest_path)
    manifest_hash = sha256(manifest_path)
    _validate_formal_runs(manifest_hash)
    selection_identity = {"data_manifest_hash": manifest_hash, "implementation_hash": _implementation_hash(),
                          "rule": "max success rate; min successful mean steps; earliest update", "run_ids": [r["run_id"] for r in RUNS]}
    selections = {}
    for run in RUNS:
        candidates = []
        for step in CHECKPOINT_STEPS:
            result = evaluate_checkpoint(run, step, "dev", manifest, manifest_hash, device=device)
            candidates.append({"step": step, "summary": result["summary"],
                               "checkpoint_hash": result["identity"]["checkpoint_hash"],
                               "dev_result_hash": sha256(result_path(run["run_id"], "dev", step))})
        selections[run["run_id"]] = select_checkpoint(candidates)
    frozen_selection = {"identity": selection_identity, "complete": True, "selections": selections}
    if SELECTION_PATH.exists():
        if read_json(SELECTION_PATH) != frozen_selection:
            raise ValueError("Frozen checkpoint selections disagree; old test results must not be silently overwritten")
    else:
        atomic_json(SELECTION_PATH, frozen_selection)
    print("All four checkpoint selections frozen; final tests may start.", flush=True)
    test_hashes = {}
    for run in RUNS:
        step = selections[run["run_id"]]["step"]
        for split in ("test_iid", "test_ood"):
            evaluate_checkpoint(run, step, split, manifest, manifest_hash, device=device)
            path = result_path(run["run_id"], split)
            test_hashes[str(path.relative_to(ROOT))] = sha256(path)
    freeze = {"complete": True, "episode_count": 160, "selection_hash": sha256(SELECTION_PATH), "result_hashes": test_hashes}
    if FREEZE_PATH.exists() and read_json(FREEZE_PATH) != freeze:
        raise ValueError("Frozen final test files changed")
    atomic_json(FREEZE_PATH, freeze)
    if videos:
        record_videos(manifest=manifest, device=device)
    return freeze


def record_videos(*, manifest: dict | None = None, device: str | None = None) -> dict:
    """Fixed first-success/first-failure rule, after numerical results freeze."""
    import imageio.v2 as imageio
    from .environment import make_env
    from .model import load_policy
    from .train import runtime_setup

    runtime_setup()
    freeze = read_json(FREEZE_PATH)
    if not freeze.get("complete") or freeze.get("episode_count") != 160:
        raise RuntimeError("Video replay requires all 160 final test records frozen")
    for name, digest in freeze["result_hashes"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Frozen test result changed: {name}")
    if sha256(SELECTION_PATH) != freeze["selection_hash"]:
        raise ValueError("Frozen checkpoint selection changed")
    if manifest is None:
        manifest = read_json(ROOT / "data" / "dataset_manifest.json")
    video_path = RESULTS / "videos" / "manifest.json"
    video_identity = {"test_freeze_hash": sha256(FREEZE_PATH), "rule": "first success and first failure in manifest order",
                      "display_transform": "vertical flip of native corner2 RGB frames; no environment or policy changes"}
    output = {"identity": video_identity, "videos": []}
    if video_path.exists():
        old = read_json(video_path)
        if old.get("identity") == video_identity:
            valid = all(v["status"] == "absent" or (v["status"] == "recorded" and (ROOT/v["path"]).exists() and sha256(ROOT/v["path"]) == v["sha256"])
                        for v in old["videos"])
            if valid and len(old["videos"]) == 16:
                return old
    for run in RUNS:
        for split in ("test_iid", "test_ood"):
            result = read_json(result_path(run["run_id"], split))
            lookup = {r["episode_id"]: r for r in _task_records(manifest, run["task"], split)}
            for success in (True, False):
                kind = "success" if success else "failure"
                entry = {"run_id": run["run_id"], "split": split, "kind": kind}
                episode = next((e for e in result["episodes"] if e["success"] == success), None)
                if episode is None:
                    entry.update(status="absent", reason=f"No {kind} episode in this split")
                    output["videos"].append(entry)
                    atomic_json(video_path, output)
                    continue
                env, writer = None, None
                try:
                    policy = load_policy(checkpoint_path(run["run_id"], episode["checkpoint_step"]), device=device)
                    env = make_env(run["task"], render_mode="rgb_array")
                    native = env.unwrapped
                    fps = 1.0 / (native.model.opt.timestep * native.frame_skip)
                    path = RESULTS / "videos" / f"{run['run_id']}_{split}_{kind}.mp4"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=1)
                    replay = rollout(policy, env, lookup[episode["episode_id"]], episode["policy_sampling_seed"],
                                     frame_callback=lambda frame: writer.append_data(np.flipud(frame).copy()))
                    writer.close()
                    writer = None
                    if replay["exception"]:
                        raise RuntimeError(replay["exception"])
                    if replay["trajectory_sha256"] != episode["trajectory_sha256"]:
                        raise RuntimeError("Video replay trajectory differs from frozen numerical evaluation")
                    entry.update(status="recorded", episode_id=episode["episode_id"], path=str(path.relative_to(ROOT)),
                                 sha256=sha256(path), fps=fps, trajectory_sha256=replay["trajectory_sha256"],
                                 elapsed_seconds=replay["elapsed_seconds"])
                except Exception as error:
                    entry.update(status="unavailable", episode_id=episode["episode_id"], reason=f"{type(error).__name__}: {error}")
                    print(f"Video unavailable: {entry}", flush=True)
                finally:
                    if writer is not None:
                        writer.close()
                    if env is not None:
                        env.close()
                output["videos"].append(entry)
                atomic_json(video_path, output)
    return output


def debug_rollout(checkpoint: str | Path, *, task: str | None = None, device: str | None = None) -> dict:
    """One separate debug rollout; never writes any formal result or selection."""
    from .environment import make_env
    from .model import load_policy
    from .train import runtime_setup
    runtime_setup()
    policy = load_policy(Path(checkpoint), device=device)
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
    task = task or metadata["config"]["task"]
    manifest = read_json(ROOT / "data" / "dataset_manifest.json")
    domain = manifest["ranges"][task]
    position = [float(np.mean(domain[name])) for name in ("train_dev_iid_x", "all_splits_y", "all_splits_z")]
    record = {"episode_id": f"{task}_debug_midpoint", "seed": 900001 if task == "drawer" else 900002,
              "task_name": f"{task}-open-v3", "initial_base_position": position,
              "task_params": {"base_position": position, "partially_observable": False}}
    for split_records in manifest["tasks"][task]["splits"].values():
        if any(np.array_equal(position, r["initial_base_position"]) for r in split_records):
            raise ValueError("Independent debug midpoint unexpectedly overlaps a formal condition")
    env = make_env(task)
    try:
        result = rollout(policy, env, record, sampling_seed(task, "debug", record["episode_id"]))
    finally:
        env.close()
    atomic_json(ROOT / "artifacts" / "debug" / f"{task}_rollout.json", result)
    if result["exception"]:
        raise RuntimeError(result["exception"])
    return result
