"""Frozen task sampling, successful expert pool, and transparent NPZ episodes."""
from __future__ import annotations

import importlib.metadata
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .environment import (METAWORLD_COMMIT, SCHEMA_VERSION, TASK_NAMES, get_base_position,
                          initial_state_hash, make_env, make_expert, observation_schema,
                          reset_from_record, snapshot_initial_state, verify_replay)
from .utils import ROOT, atomic_json, object_hash, read_json, sha256

DEFAULT_MANIFEST = ROOT / "data/dataset_manifest.json"
SPLITS = ("train_pool", "dev", "test_iid", "test_ood")


def position_ranges(schema: dict) -> dict:
    result = {}
    for key, task in schema["tasks"].items():
        low, high = np.asarray(task["native_reset_low"]), np.asarray(task["native_reset_high"])
        expected_low = np.array([-0.1, 0.9, 0]) if key == "drawer" else np.array([0, 0.85, 0.15])
        expected_high = np.array([0.1, 0.9, 0]) if key == "drawer" else np.array([0.1, 0.95, 0.15])
        unchanged = np.allclose(low, expected_low, atol=1e-12, rtol=0) and np.allclose(high, expected_high, atol=1e-12, rtol=0)
        width = high[0] - low[0]
        x_iid = [float(low[0] + .3 * width), float(low[0] + .7 * width)]
        x_ood = [[float(low[0]), float(low[0] + .2 * width)], [float(low[0] + .8 * width), float(high[0])]]
        y_range = [float(low[1]), float(high[1])]
        if key == "door" and unchanged:
            y_range = [0.88, 0.92]
        result[key] = {
            "native_low": low.tolist(), "native_high": high.tolist(),
            "train_dev_iid_x": x_iid, "ood_x": x_ood, "all_splits_y": y_range,
            "all_splits_z": [float(low[2]), float(high[2])],
            "ranges_match_specification": bool(unchanged),
            "adjustment": None if unchanged else "Use native x middle40%, outer20% each with gaps; other dimensions native same distribution",
            "fixed_orientation_and_opening": "unchanged native reset; drawer init_config angle0.3 (native initialization); door doorjoint=0",
        }
    return result


def build_sampling_plan(schema: dict) -> dict:
    ranges = position_ranges(schema)
    plan = {"schema_version": SCHEMA_VERSION, "max_expert_train_attempts_per_task": 1000,
            "train_pool_successes_per_task": 100, "train20_selection_rule": "default_rng(namespace+55).choice(100,20,replace=False), sorted by pool index",
            "evaluation_exclusion_rule": "No expert failures are excluded; all valid generated states retained. Physical invalidity would stop collection for diagnosis.",
            "ranges": ranges, "tasks": {}}
    for key, namespace in (("drawer", 1000), ("door", 2000)):
        task = {"task_name": TASK_NAMES[key], "namespace": namespace, "splits": {}, "sampling_seeds": {}}
        for split_index, split in enumerate(SPLITS):
            stream_seed = namespace + 11 * (split_index + 1)
            rng = np.random.default_rng(stream_seed)
            task["sampling_seeds"][split] = stream_seed
            records = []
            count = 1000 if split == "train_pool" else 20
            for index in range(count):
                domain = ranges[key]
                x_range = domain["ood_x"][int(index >= 10)] if split == "test_ood" else domain["train_dev_iid_x"]
                position = [float(rng.uniform(*x_range)), float(rng.uniform(*domain["all_splits_y"])), float(rng.uniform(*domain["all_splits_z"]))]
                params = {"base_position": position, "partially_observable": False,
                          "native_initial_orientation_and_opening": True}
                records.append({"episode_id": f"{key}_{split}_{index:04d}", "task_name": TASK_NAMES[key],
                                "seed": namespace * 100000 + split_index * 10000 + index,
                                "task_params": params, "task_parameter_hash": object_hash(params),
                                "ood_side": ("left" if index < 10 else "right") if split == "test_ood" else None})
            task["splits"][split] = records
        all_records = sum(task["splits"].values(), [])
        if len({r["task_parameter_hash"] for r in all_records}) != len(all_records):
            raise RuntimeError("Duplicate task parameter layouts in sampling plan")
        plan["tasks"][key] = task
    return plan


def run_expert_episode(env, record: dict) -> tuple[dict[str, np.ndarray], dict]:
    obs, _ = reset_from_record(env, record)
    snapshots = snapshot_initial_state(env, obs)
    expert = make_expert(record["task_name"])
    initial_position = get_base_position(env).tolist()
    observations, actions, unclipped, rewards, successes, terminations, truncations = [obs.copy()], [], [], [], [], [], []
    start = time.monotonic()
    for _ in range(env.max_path_length):
        pristine = obs.copy()
        # Official door expert mutates a slice: give it its own private array.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"Constant\(s\) may be too high.*")
            proposed = np.asarray(expert.get_action(obs.copy()), dtype=np.float32)
        np.testing.assert_array_equal(obs, pristine)
        action = np.clip(proposed, env.action_space.low, env.action_space.high).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action.copy())
        unclipped.append(proposed.copy())
        observations.append(obs.copy())
        rewards.append(float(reward))
        successes.append(bool(info["success"]))
        terminations.append(bool(terminated))
        truncations.append(bool(truncated))
        if successes[-1] or terminated or truncated:
            break
    arrays = dict(snapshots)
    arrays.update(obs=np.asarray(observations, dtype=np.float32), actions=np.asarray(actions, dtype=np.float32),
                  expert_actions_unclipped=np.asarray(unclipped, dtype=np.float32), rewards=np.asarray(rewards, dtype=np.float32),
                  success=np.asarray(successes, dtype=np.bool_), terminated=np.asarray(terminations, dtype=np.bool_),
                  truncated=np.asarray(truncations, dtype=np.bool_))
    summary = {"initial_base_position": initial_position, "initial_state_hash": initial_state_hash(snapshots),
               "expert_success": bool(any(successes)), "expert_episode_length": len(actions),
               "expert_first_success_step": (successes.index(True) + 1) if any(successes) else None,
               "expert_return": float(sum(rewards)), "expert_elapsed_seconds": time.monotonic() - start,
               "expert_action_clipped_entries": int(np.count_nonzero(arrays["actions"] != arrays["expert_actions_unclipped"])),
               "expert_failure_reason": None if any(successes) else ("terminated" if any(terminations) else "time_limit"),
               "initial_condition_id": object_hash({"task_params": record["task_params"], "initial_state_hash": initial_state_hash(snapshots)})}
    return arrays, summary


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temp, path)


def verify_manifest(manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    path = Path(manifest_path)
    manifest = read_json(path)
    if manifest.get("status") != "frozen":
        raise RuntimeError("Dataset manifest not frozen")
    for filename, expected_hash in manifest["input_artifact_hashes"].items():
        if sha256(ROOT / filename) != expected_hash:
            raise RuntimeError(f"Frozen artifact changed: {filename}")
    for task in manifest["tasks"].values():
        if len(task["train20"]) != 20 or len(set(task["train20"])) != 20:
            raise RuntimeError("train20 must contain twenty unique episodes")
        ids, parameters, initial_conditions = set(), set(), set()
        for split in SPLITS:
            records = task["splits"][split]
            if len(records) != (100 if split == "train_pool" else 20):
                raise RuntimeError(f"Incorrect split size: {split}")
            for record in records:
                for seen, item in ((ids, record["episode_id"]), (parameters, record["task_parameter_hash"]), (initial_conditions, record["initial_condition_id"])):
                    if item in seen:
                        raise RuntimeError(f"Duplicate initial condition in dataset: {item}")
                    seen.add(item)
                if sha256(ROOT / record["path"]) != record["sha256"]:
                    raise RuntimeError(f"Changed episode: {record['path']}")
                if sha256(ROOT / record["metadata_path"]) != record["metadata_sha256"]:
                    raise RuntimeError(f"Changed episode metadata: {record['metadata_path']}")
        pool = {r["episode_id"] for r in task["splits"]["train_pool"]}
        if not set(task["train20"]).issubset(pool):
            raise RuntimeError("train20 references outside train_pool")
    return manifest


load_manifest = verify_manifest


def audit_dataset(manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    """Replay frozen raw labels, check every saved reset, and validate coordinates."""
    from .representation import inverse_observation, transform_observation

    manifest = verify_manifest(manifest_path)
    start = time.monotonic()
    result = {"data_manifest_sha256": sha256(manifest_path), "initial_conditions_checked": 0,
              "full_demonstrations_replayed": 0, "action_steps_replayed": 0,
              "max_stored_float32_observation_error": 0.0, "max_roundtrip_error": 0.0,
              "state_snapshot_atol": 1e-9, "stored_float32_observation_atol": 1e-6,
              "action_labels_match_replayed_environment_inputs": True, "tasks": {}}
    for key, task in manifest["tasks"].items():
        env = make_env(key)
        task_result = {"train20_total_steps": 0, "train20_min_length": None, "train20_max_length": None,
                       "train20_mean_length": None, "expert_clipped_entries": 0}
        selected_lengths = []
        try:
            for split, records in task["splits"].items():
                for record in records:
                    obs, _ = reset_from_record(env, record)
                    current_snapshot = snapshot_initial_state(env, obs)
                    with np.load(ROOT / record["path"], allow_pickle=False) as arrays:
                        for field, value in current_snapshot.items():
                            np.testing.assert_allclose(value, arrays[field], atol=1e-9, rtol=0, err_msg=f"{record['episode_id']}:{field}")
                        result["initial_conditions_checked"] += 1
                        if split != "train_pool":
                            continue
                        raw = arrays["obs"]
                        recovered = inverse_observation(transform_observation(raw, "relative"))
                        np.testing.assert_allclose(recovered, raw, atol=1e-6, rtol=0)
                        result["max_roundtrip_error"] = max(result["max_roundtrip_error"], float(np.abs(recovered - raw).max()))
                        actions = arrays["actions"]
                        if raw.shape != (len(actions) + 1, 39) or actions.shape[1:] != (4,):
                            raise RuntimeError(f"Malformed demonstration: {record['episode_id']}")
                        if not (np.all(actions >= -1) and np.all(actions <= 1)):
                            raise RuntimeError("Unclipped action labels")
                        np.testing.assert_allclose(obs, raw[0], atol=1e-6, rtol=0)
                        for index, action in enumerate(actions):
                            obs, reward, terminated, truncated, info = env.step(action.copy())
                            error = float(np.abs(obs - raw[index + 1]).max())
                            result["max_stored_float32_observation_error"] = max(result["max_stored_float32_observation_error"], error)
                            np.testing.assert_allclose(obs, raw[index + 1], atol=1e-6, rtol=0)
                            np.testing.assert_allclose(reward, arrays["rewards"][index], atol=1e-6, rtol=0)
                            assert bool(info["success"]) == bool(arrays["success"][index])
                            assert terminated == bool(arrays["terminated"][index])
                            assert truncated == bool(arrays["truncated"][index])
                            if info["success"] or terminated or truncated:
                                assert index == len(actions) - 1, "Episode contains actions after completion"
                        result["full_demonstrations_replayed"] += 1
                        result["action_steps_replayed"] += len(actions)
                        task_result["expert_clipped_entries"] += record["expert_action_clipped_entries"]
                        if record["episode_id"] in task["train20"]:
                            selected_lengths.append(len(actions))
            task_result.update(train20_total_steps=sum(selected_lengths), train20_min_length=min(selected_lengths),
                               train20_max_length=max(selected_lengths), train20_mean_length=float(np.mean(selected_lengths)))
            result["tasks"][key] = task_result
        finally:
            env.close()
    result["elapsed_seconds"] = time.monotonic() - start
    result["passed"] = True
    atomic_json(ROOT / "artifacts/dataset_audit.json", result)
    return result


def collect_round1() -> dict:
    if DEFAULT_MANIFEST.exists():
        return verify_manifest(DEFAULT_MANIFEST)
    start = time.monotonic()
    schema_path = ROOT / "artifacts/observation_schema.json"
    schema = observation_schema(schema_path)
    plan_path = ROOT / "data/task_sampling_plan.json"
    expected_plan = build_sampling_plan(schema)
    if plan_path.exists():
        if read_json(plan_path) != expected_plan:
            raise RuntimeError("Frozen sampling rules differ; refusing to overwrite")
    else:
        atomic_json(plan_path, expected_plan)
    versions = {name: importlib.metadata.version(name) for name in ("metaworld", "mujoco", "numpy", "gymnasium")}
    versions["metaworld_commit"] = METAWORLD_COMMIT
    manifest = {"schema_version": SCHEMA_VERSION, "status": "collecting", "versions": versions,
                "ranges": expected_plan["ranges"], "tasks": {}, "excluded_attempts": [],
                "input_artifact_hashes": {str(schema_path.relative_to(ROOT)): sha256(schema_path), str(plan_path.relative_to(ROOT)): sha256(plan_path)},
                "raw_data_format": "train_pool: NPZ obs[T+1,39], actions[T,4], rewards/success/terminated/truncated[T], snapshots; eval splits: initial snapshots only (expert precheck summaries, no trajectories)",
                "normalization_and_optimization_eligible": "Only train20; evaluation and remaining pool80 forbidden",
                "reproduction": "reset_from_record with task_params and seed; complete initial snapshots retained; replay checks use independent and reused env instances"}
    for key, task_plan in expected_plan["tasks"].items():
        env = make_env(key)
        task = {"task_name": TASK_NAMES[key], "namespace": task_plan["namespace"],
                "sampling_seeds": task_plan["sampling_seeds"], "splits": {}, "replay_checks": {}}
        manifest["tasks"][key] = task
        try:
            for split in SPLITS:
                collected = []
                attempted = 0
                for planned in task_plan["splits"][split]:
                    if len(collected) >= (100 if split == "train_pool" else 20):
                        break
                    attempted += 1
                    record = dict(planned)
                    path = ROOT / "data" / key / split / (record["episode_id"] + ".npz")
                    meta_path = path.with_suffix(".json")
                    if meta_path.exists():
                        saved = read_json(meta_path)
                        if sha256(path) != saved["sha256"] or saved["task_parameter_hash"] != record["task_parameter_hash"]:
                            raise RuntimeError(f"Partial collection artifact changed: {path}")
                        record = saved
                    else:
                        arrays, summary = run_expert_episode(env, record)
                        record.update(summary)
                        record["versions"] = versions
                        record["schema_version"] = SCHEMA_VERSION
                        record["observation_schema_sha256"] = sha256(schema_path)
                        record["path"] = str(path.relative_to(ROOT))
                        record["metadata_path"] = str(meta_path.relative_to(ROOT))
                        # Eval expert checks are deliberately absent from training data.
                        if split != "train_pool":
                            arrays = {k: v for k, v in arrays.items() if k.startswith("snapshot_")}
                        arrays["metadata_json"] = np.asarray(json.dumps(record, sort_keys=True))
                        _write_npz(path, arrays)
                        record["sha256"] = sha256(path)
                        atomic_json(meta_path, record)
                    record["metadata_sha256"] = sha256(meta_path)
                    if split == "train_pool" and not record["expert_success"]:
                        manifest["excluded_attempts"].append({**record, "exclusion_reason": "unsuccessful expert demonstration, not eligible for successful train_pool"})
                    else:
                        collected.append(record)
                    if attempted % 10 == 0:
                        print(f"collect {key}/{split}: {len(collected)} accepted, {attempted} attempts", flush=True)
                required = 100 if split == "train_pool" else 20
                if len(collected) != required:
                    raise RuntimeError(f"{key}: obtained only {len(collected)} successful demos in {attempted} attempts (cap1000)")
                task["splits"][split] = collected
                task[f"{split}_attempts"] = attempted
                task[f"{split}_expert_successes"] = sum(r["expert_success"] for r in collected)
                coordinates = np.asarray([r["initial_base_position"] for r in collected])
                task[f"{split}_actual_positions"] = {"min": coordinates.min(0).tolist(), "max": coordinates.max(0).tolist(), "mean": coordinates.mean(0).tolist()}
                # Both ends of every frozen split independently replayed; all
                # records retain complete snapshots for arbitrary later replay.
                task["replay_checks"][split] = {r["episode_id"]: verify_replay(r) for r in (collected[0], collected[-1])}
                atomic_json(ROOT / "data/collection_progress.json", manifest)
            selection_seed = task_plan["namespace"] + 55
            selected = np.sort(np.random.default_rng(selection_seed).choice(100, 20, replace=False))
            task["train20_selection_seed"] = selection_seed
            task["train20"] = [task["splits"]["train_pool"][i]["episode_id"] for i in selected]
            task["train20_hashes"] = {r["episode_id"]: r["sha256"] for r in task["splits"]["train_pool"] if r["episode_id"] in task["train20"]}
        finally:
            env.close()
    manifest["status"] = "frozen"
    manifest["collection_invocation_elapsed_seconds"] = time.monotonic() - start
    manifest["expert_rollout_elapsed_seconds_total"] = sum(r["expert_elapsed_seconds"] for task in manifest["tasks"].values() for records in task["splits"].values() for r in records) + sum(r["expert_elapsed_seconds"] for r in manifest["excluded_attempts"])
    atomic_json(DEFAULT_MANIFEST, manifest)
    return verify_manifest(DEFAULT_MANIFEST)
