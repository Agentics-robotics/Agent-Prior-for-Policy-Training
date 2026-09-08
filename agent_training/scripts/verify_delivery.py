"""Independent final acceptance audit; run only after training/report completion.

Usage: pixi run python scripts/verify_delivery.py

This script never imports training/evaluation/collection code, runs an optimizer,
or changes experimental artifacts. It reconstructs acceptance rules from the
specification and frozen files, and loads checkpoints on CPU for inspection.
Only a successful audit writes artifacts/delivery_audit.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
STEPS = (5000, 10000, 20000)
TASKS = ("drawer", "door")
REPRESENTATIONS = ("raw", "relative")
RUN_IDS = [f"{task}_{rep}_n20_s0" for task in TASKS for rep in REPRESENTATIONS]
ARTIFACT_HASHES: dict[str, str] = {}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: str | Path, expected: str | None = None) -> str:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    # Empty Python package markers are legitimate frozen source files. Binary
    # checkpoints/plots and JSON have their contents validated by their readers.
    require(path.is_file(), f"Missing artifact: {path}")
    hash_ = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hash_.update(block)
    value = hash_.hexdigest()
    if expected is not None:
        require(value == expected, f"Artifact hash mismatch: {path}")
    try:
        ARTIFACT_HASHES[str(path.relative_to(ROOT))] = value
    except ValueError:
        ARTIFACT_HASHES[str(path)] = value
    return value


def read(path: str | Path) -> dict:
    digest(path)
    path = Path(path)
    return json.loads((path if path.is_absolute() else ROOT / path).read_text())


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def close(actual, expected, message: str, atol=1e-9) -> None:
    if expected is None:
        require(actual is None or actual == "", message)
    else:
        require(actual is not None and actual != "" and math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=1e-9), message)


def policy_seed(task: str, split: str, episode_id: str) -> int:
    token = f"relative-dp-round1/evaluation-v1/{task}/{split}/{episode_id}"
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little") % (2**63 - 1)


def expected_code_hashes() -> tuple[str, str]:
    base = ROOT / "src/relative_dp"
    training = {name: digest(base / name) for name in ("config.py", "model.py", "dataset.py", "train.py", "representation.py")}
    evaluation = {name: digest(base / name) for name in ("evaluate.py", "model.py", "environment.py", "representation.py", "dataset.py", "train.py", "config.py")}
    for path in sorted((base / "vendor/diffusion_policy").glob("*.py")):
        key = str(path.relative_to(base))
        training[key] = evaluation[key] = digest(path)
    evaluation["pixi.lock"] = digest("pixi.lock")
    return object_hash(training), object_hash(evaluation)


def check_frozen_inputs() -> tuple[dict, dict, str]:
    frozen = read("artifacts/experiment_freeze.json")
    for path, expected in frozen["file_hashes"].items():
        digest(path, expected)
    actual_sources = {str(path.relative_to(ROOT)) for path in (ROOT / "src/relative_dp").rglob("*.py")
                      if path.name not in ("report.py", "cli.py", "doctor.py")}
    recorded_sources = {path for path in frozen["file_hashes"] if path.startswith("src/")}
    require(actual_sources == recorded_sources, "Frozen source file set changed")
    require([run["run_id"] for run in frozen["runs"]] == RUN_IDS, "Formal four-run list changed")
    prescribed = dict(n_obs_steps=2, prediction_horizon=16, n_action_steps=4, action_dim=4,
                      unet_down_dims=[64, 128, 256], diffusion_step_embed_dim=128,
                      kernel_size=5, n_groups=8, batch_size=128, optimizer="AdamW",
                      learning_rate=0.0001, weight_decay=0.000001, betas=[0.9, 0.999],
                      max_grad_norm=1.0, train_updates=20000, lr_schedule="constant",
                      diffusion_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
                      prediction_type="epsilon", inference_scheduler="DDIM", inference_steps=16,
                      ddim_eta=0.0, clip_predicted_clean_actions=True, ema_decay=0.995,
                      train_seed=0, checkpoint_updates=list(STEPS))
    for key, expected in prescribed.items():
        require(frozen["config"].get(key) == expected, f"Prescribed configuration differs: {key}")
    preflight = read("artifacts/preflight.json")
    require(preflight["passed"] and preflight["test_exit_code"] == 0, "Preflight checks did not pass")
    require(preflight["file_hashes"] == frozen["file_hashes"], "Preflight was performed on different frozen inputs")
    for path, expected in preflight["debug_checkpoint_hashes"].items():
        digest(path, expected)
    require(not preflight["debug_rollout"].get("exception"), "Debug rollout has an exception")
    manifest_path = "data/dataset_manifest.json"
    manifest = read(manifest_path)
    manifest_hash = digest(manifest_path)
    require(manifest["status"] == "frozen", "Dataset not frozen")
    for path, expected in manifest["input_artifact_hashes"].items():
        digest(path, expected)
    schema = read("artifacts/observation_schema.json")
    require(schema["observation_dim"] == 39, "Observation schema changed")
    for task, group in manifest["tasks"].items():
        seen_ids, seen_params, seen_conditions = set(), set(), set()
        for split in ("train_pool", "dev", "test_iid", "test_ood"):
            records = group["splits"][split]
            require(len(records) == (100 if split == "train_pool" else 20), f"Wrong data count: {task}/{split}")
            for record in records:
                for seen, value in ((seen_ids, record["episode_id"]), (seen_params, record["task_parameter_hash"]),
                                    (seen_conditions, record["initial_condition_id"])):
                    require(value not in seen, f"Repeated data initial condition: {record['episode_id']}")
                    seen.add(value)
                require(record["task_parameter_hash"] == object_hash(record["task_params"]), "Task parameter hash mismatch")
                require(np.allclose(record["initial_base_position"], record["task_params"]["base_position"], atol=1e-9, rtol=0), "Actual/requested base mismatch")
                domain = manifest["ranges"][task]
                x_range = domain["ood_x"][0 if record["ood_side"] == "left" else 1] if split == "test_ood" else domain["train_dev_iid_x"]
                for value, interval in zip(record["initial_base_position"], (x_range, domain["all_splits_y"], domain["all_splits_z"]), strict=True):
                    require(interval[0] - 1e-12 <= value <= interval[1] + 1e-12, "Actual position outside frozen split")
                if split == "train_pool":
                    require(record["expert_success"] is True, "Failed demonstration in successful pool")
                digest(record["path"], record["sha256"])
                digest(record["metadata_path"], record["metadata_sha256"])
            if split == "test_ood":
                require(sum(r["ood_side"] == "left" for r in records) == 10, "OOD requires ten left and ten right conditions")
        expected_indices = np.sort(np.random.default_rng(group["train20_selection_seed"]).choice(100, 20, replace=False))
        require(group["train20"] == [group["splits"]["train_pool"][i]["episode_id"] for i in expected_indices], "train20 fixed RNG selection changed")
        task_schema = schema["tasks"][task]
        require(task_schema["goal_observable"] and task_schema["max_episode_steps"] == 500, "Native horizon/observability mismatch")
        digest(task_schema["source_file"], task_schema["source_sha256"])
        digest(Path(task_schema["source_file"]).parent.parent / "sawyer_xyz_env.py", schema["base_source_sha256"])
    data_audit = read("artifacts/dataset_audit.json")
    require(data_audit["passed"] and data_audit["data_manifest_sha256"] == manifest_hash, "Dataset audit not valid for frozen manifest")
    require(data_audit["initial_conditions_checked"] == 320 and data_audit["full_demonstrations_replayed"] == 200, "Dataset audit incomplete")
    return frozen, manifest, manifest_hash


def independently_fit_normalizer(task: dict, representation: str) -> dict:
    records = {r["episode_id"]: r for r in task["splits"]["train_pool"]}
    arrays = []
    for episode_id in task["train20"]:
        with np.load(ROOT / records[episode_id]["path"], allow_pickle=False) as episode:
            obs = episode["obs"][:-1].copy()
            if representation == "relative":
                obs[:, :3] -= obs[:, 4:7]
                obs[:, 18:21] -= obs[:, 22:25]
                obs[:, 36:39] -= obs[:, 4:7]
            arrays.append(obs)
    values = np.concatenate(arrays).astype(np.float64)
    empirical_std = values.std(0)
    return {"mean": values.mean(0).astype(np.float32).tolist(),
            "std": np.maximum(empirical_std, 1e-3).astype(np.float32).tolist(),
            "source_ids": task["train20"],
            "source_hashes": {episode_id: records[episode_id]["sha256"] for episode_id in task["train20"]},
            "constant_dimensions": np.flatnonzero(empirical_std < 1e-3).tolist(),
            "representation": representation, "count": len(values),
            "statistics_scope": "train20 obs[0:T], population std, floor=0.001"}


def inspect_training(frozen: dict, manifest: dict, manifest_hash: str, code_hash: str) -> tuple[dict, dict]:
    runs, pairs = {}, {}
    for run in frozen["runs"]:
        run_id = run["run_id"]
        directory = ROOT / "runs" / run_id
        complete = read(directory / "complete.json")
        status = read(directory / "status.json")
        config = read(directory / "config.json")
        identity = read(directory / "identity.json")
        normalizer = read(directory / "normalizer.json")
        expected_config = dict(frozen["config"], **run)
        require(config == expected_config, f"Run config differs: {run_id}")
        require(complete["step"] == status["step"] == 20000 and status["complete"] is True and status["running"] is False, f"Formal run incomplete: {run_id}")
        expected_normalizer = independently_fit_normalizer(manifest["tasks"][run["task"]], run["representation"])
        require(normalizer == expected_normalizer, f"Normalizer differs from independent train20-only statistics: {run_id}")
        require(identity["normalization_source_ids"] == expected_normalizer["source_ids"], "Normalizer identity provenance mismatch")
        for artifact in (complete, identity):
            require(artifact["config_hash"] == object_hash(config) and artifact["manifest_hash"] == manifest_hash and artifact["code_hash"] == code_hash, f"Run identity mismatch: {run_id}")
            require(artifact["debug"] is False, "Debug artifact presented as formal training")
        checkpoint_meta = {}
        for step in STEPS:
            item = complete["checkpoints"][str(step)]
            checkpoint_hash = digest(item["path"], item["sha256"])
            checkpoint = torch.load(ROOT / item["path"], map_location="cpu", weights_only=False)
            require(checkpoint["step"] == checkpoint["update"] == step, "Checkpoint step mismatch")
            require(checkpoint["config"] == config and checkpoint["config_hash"] == object_hash(config), "Checkpoint config mismatch")
            require(checkpoint["run_id"] == run_id and checkpoint["manifest_hash"] == manifest_hash and checkpoint["code_hash"] == code_hash, "Checkpoint identity mismatch")
            require(checkpoint["debug"] is False and checkpoint["normalizer"] == normalizer, "Checkpoint normalization/debug mismatch")
            require(checkpoint["samples_drawn"] == step * 128, "Checkpoint sampled-window count mismatch")
            require(set(checkpoint["rng"]) >= {"loader", "diffusion", "torch", "numpy", "python", "cuda"}, "Checkpoint lacks RNG resume state")
            require(bool(checkpoint["optimizer"]["state"]), "Checkpoint lacks optimizer resume state")
            for group in checkpoint["optimizer"]["param_groups"]:
                require(group["lr"] == 0.0001 and group["weight_decay"] == 0.000001 and list(group["betas"]) == [0.9, 0.999], "Saved optimizer differs from prescribed configuration")
            for state in checkpoint["optimizer"]["state"].values():
                require(int(state["step"]) == step, "Saved optimizer update count differs from checkpoint")
            require(set(checkpoint["model"]) == set(checkpoint["ema"]), "EMA/model state keys differ")
            parameter_count = sum(tensor.numel() for tensor in checkpoint["model"].values())
            require(parameter_count == checkpoint["parameter_count"] == complete["parameter_count"] == identity["parameter_count"], "Network parameter count mismatch")
            for tensor in checkpoint["ema"].values():
                require(bool(torch.isfinite(tensor).all()), "EMA contains nonfinite values")
            for key in ("initial_weights_hash",):
                require(checkpoint[key] == complete[key] == identity[key], f"Checkpoint/marker {key} differs")
            if step == 20000:
                require(checkpoint["pairing_digest"] == complete["pairing_digest"], "Final RNG pairing digest mismatch")
            checkpoint_meta[step] = {"sha256": checkpoint_hash, "pairing_digest": checkpoint["pairing_digest"], "initial_weights_hash": checkpoint["initial_weights_hash"]}
            del checkpoint
        log_path = directory / "train.jsonl"
        digest(log_path)
        log = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        require(log[-1]["step"] == 20000 and len({r["step"] for r in log}) == len(log), "Training log missing final update or duplicates")
        require(all(math.isfinite(r["loss"]) for r in log), "Training loss log contains nonfinite values")
        runs[run_id] = {"run": run, "complete": complete, "config": config, "normalizer": normalizer, "checkpoints": checkpoint_meta}
    for task in TASKS:
        raw, relative = [runs[f"{task}_{rep}_n20_s0"] for rep in REPRESENTATIONS]
        for field in ("initial_weights_hash", "pairing_digest", "parameter_count", "manifest_hash", "step"):
            require(raw["complete"][field] == relative["complete"][field], f"Unfair pair {task}: {field}")
        for step in STEPS:
            require(raw["checkpoints"][step]["pairing_digest"] == relative["checkpoints"][step]["pairing_digest"], f"Noise/window/timestep streams differ at {task}/{step}")
        common = [{k: v for k, v in record["config"].items() if k not in ("run_id", "representation")} for record in (raw, relative)]
        require(common[0] == common[1], f"Paired non-representation config differs: {task}")
        pairs[task] = {"passed": True, "initial_weights_hash": raw["complete"]["initial_weights_hash"],
                       "pairing_digest": raw["complete"]["pairing_digest"], "parameter_count": raw["complete"]["parameter_count"],
                       "train20_count": 20, "normalization_input_states": raw["normalizer"]["count"]}
    return runs, pairs


def summarize(episodes: list[dict]) -> dict:
    success = [e for e in episodes if e["success"]]
    return {"n": len(episodes), "successes": len(success), "success_rate": len(success) / len(episodes),
            "mean_success_steps": sum(e["first_success_step"] for e in success) / len(success) if success else None,
            "elapsed_seconds": sum(e["elapsed_seconds"] for e in episodes),
            "exceptions": sum(e.get("exception") is not None for e in episodes)}


def inspect_evaluation(path: Path, run: dict, step: int, split: str, records: list[dict],
                       checkpoint_hash: str, manifest_hash: str, implementation_hash: str) -> dict:
    result = read(path)
    identity, episodes = result["identity"], result["episodes"]
    require(result["complete"] is True and len(episodes) == 20, f"Evaluation incomplete: {path}")
    require(identity["run"] == run and identity["step"] == step and identity["split"] == split, "Evaluation identity run/step/split mismatch")
    require(identity["checkpoint_hash"] == checkpoint_hash and identity["data_manifest_hash"] == manifest_hash, "Evaluation input hash mismatch")
    require(identity["implementation_hash"] == implementation_hash, "Evaluation implementation hash mismatch")
    require(identity["evaluation_version"] == 1 and identity["torch_deterministic_algorithms"] is True, "Evaluation version/deterministic setting mismatch")
    require(identity["initial_conditions_hash"] == object_hash(records), "Evaluation initial-condition set differs")
    require(identity["weights"] == "EMA" and identity["max_episode_steps"] == 500 and identity["n_obs_steps"] == 2, "Evaluation weights/horizon/history mismatch")
    require(identity["executed_action_slice"] == [0, 4] and identity["inference_steps"] == 16 and identity["ddim_eta"] == 0.0, "Evaluation chunk/scheduler mismatch")
    seeds = [policy_seed(run["task"], split, r["episode_id"]) for r in records]
    require(identity["policy_sampling_seeds"] == seeds, "Evaluation sampling seeds differ from fixed namespace")
    require([e["episode_id"] for e in episodes] == [r["episode_id"] for r in records], "Evaluation episode IDs/order differ")
    for episode, record, seed in zip(episodes, records, seeds, strict=True):
        for field, expected in {"run_id": run["run_id"], "task": run["task_name"], "representation": run["representation"],
                                "train_n": 20, "train_seed": 0, "checkpoint_step": step, "checkpoint_hash": checkpoint_hash,
                                "data_manifest_hash": manifest_hash, "split": split, "env_seed": record["seed"],
                                "policy_sampling_seed": seed, "task_params": record["task_params"],
                                "initial_condition_hash": object_hash(record)}.items():
            require(episode.get(field) == expected, f"Episode metadata differs: {run['run_id']}/{record['episode_id']}/{field}")
        require(episode.get("exception") is None, "Evaluation contains runtime exception")
        require(isinstance(episode["success"], bool) and isinstance(episode["terminated"], bool) and isinstance(episode["truncated"], bool), "Evaluation flags malformed")
        require(1 <= episode["episode_length"] <= 500 and math.isfinite(episode["return"]) and episode["elapsed_seconds"] > 0, "Episode length/return/time invalid")
        for field in ("initial_base_position", "restored_initial_base_position"):
            require(np.allclose(episode[field], record["initial_base_position"], atol=1e-8, rtol=0), "Evaluation initial base differs")
        if episode["success"]:
            require(episode["first_success_step"] == episode["episode_length"] and episode["failure_reason"] is None, "Episode did not stop at first success")
            require(bool(episode["final_info"].get("success")), "Successful episode lacks native success flag")
        else:
            require(episode["first_success_step"] is None and episode["failure_reason"] is not None, "Failed episode metadata inconsistent")
            require(not bool(episode["final_info"].get("success")), "Failed episode has native success flag")
        require(len(episode["trajectory_sha256"]) == 64, "Trajectory digest missing")
    expected_summary = summarize(episodes)
    for field, expected in expected_summary.items():
        close(result["summary"].get(field), expected, f"Evaluation summary differs from episodes: {field}")
    return result


def inspect_results(frozen: dict, manifest: dict, manifest_hash: str, runs: dict, implementation_hash: str) -> tuple[dict, dict]:
    selection_path = ROOT / "results/checkpoint_selection.json"
    selected = read(selection_path)
    require(selected["complete"] is True, "Checkpoint selections not frozen")
    require(selected["identity"]["data_manifest_hash"] == manifest_hash and selected["identity"]["implementation_hash"] == implementation_hash, "Selection identity mismatch")
    require(selected["identity"]["run_ids"] == RUN_IDS, "Selection run list mismatch")
    require(selected["identity"]["rule"] == "max success rate; min successful mean steps; earliest update", "Checkpoint selection rule metadata changed")
    final_freeze = read("results/test_freeze.json")
    require(final_freeze["complete"] and final_freeze["episode_count"] == 160, "Final test freeze incomplete")
    digest(selection_path, final_freeze["selection_hash"])
    tests, computed = {}, {}
    expected_test_paths = set()
    for run in frozen["runs"]:
        run_id = run["run_id"]
        task_records = manifest["tasks"][run["task"]]["splits"]
        candidates = []
        dev_elapsed = 0.0
        for step in STEPS:
            path = ROOT / "results" / run_id / f"dev_step{step:06d}.json"
            checkpoint_hash = runs[run_id]["checkpoints"][step]["sha256"]
            result = inspect_evaluation(path, run, step, "dev", task_records["dev"], checkpoint_hash, manifest_hash, implementation_hash)
            summary = summarize(result["episodes"])
            candidates.append({"step": step, "summary": summary, "checkpoint_hash": checkpoint_hash, "dev_result_hash": digest(path)})
            dev_elapsed += summary["elapsed_seconds"]
        winner = min(candidates, key=lambda c: (-c["summary"]["success_rate"], math.inf if c["summary"]["mean_success_steps"] is None else c["summary"]["mean_success_steps"], c["step"]))
        require(selected["selections"][run_id] == winner, f"Checkpoint selection violates exact predeclared rule: {run_id}")
        computed[run_id] = {"selected_checkpoint": winner["step"], "dev_seconds": dev_elapsed}
        for split, label in (("test_iid", "iid"), ("test_ood", "ood")):
            path = ROOT / "results" / run_id / f"{split}.json"
            relative_path = str(path.relative_to(ROOT))
            expected_test_paths.add(relative_path)
            digest(path, final_freeze["result_hashes"].get(relative_path))
            require(relative_path in final_freeze["result_hashes"], "Test file absent from freeze")
            result = inspect_evaluation(path, run, winner["step"], split, task_records[split], winner["checkpoint_hash"], manifest_hash, implementation_hash)
            tests[(run_id, split)] = result
            computed[run_id][label] = summarize(result["episodes"])
    require(set(final_freeze["result_hashes"]) == expected_test_paths and len(expected_test_paths) == 8, "Final freeze has wrong test file set")
    for task in TASKS:
        for split in ("test_iid", "test_ood"):
            pair_results = [tests[(f"{task}_{rep}_n20_s0", split)] for rep in REPRESENTATIONS]
            require(pair_results[0]["identity"]["hardware"] == pair_results[1]["identity"]["hardware"], "Final paired evaluation hardware differs")
            raw, relative = [tests[(f"{task}_{rep}_n20_s0", split)]["episodes"] for rep in REPRESENTATIONS]
            for left, right in zip(raw, relative, strict=True):
                for field in ("episode_id", "env_seed", "policy_sampling_seed", "initial_base_position", "task_params", "initial_condition_hash"):
                    require(left[field] == right[field], f"Final test pair differs: {task}/{split}/{field}")
    summary_path = ROOT / "results/summary.csv"
    digest(summary_path)
    with summary_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 4 and {row["run_id"] for row in rows} == set(RUN_IDS), "Summary needs four unique formal runs")
    for row in rows:
        run_id = row["run_id"]
        expected = computed[run_id]
        require(row["status"] == row["training_status"] == "complete", "Summary contains incomplete status")
        close(row["last_logged_update"], 20000, "Summary update count wrong")
        close(row["selected_checkpoint"], expected["selected_checkpoint"], "Summary checkpoint wrong")
        close(row["train_seconds"], runs[run_id]["complete"]["elapsed_seconds"], "Summary training time wrong")
        close(row["dev_seconds"], expected["dev_seconds"], "Summary dev time wrong")
        for label in ("iid", "ood"):
            for field, summary_field in (("evaluated", "n"), ("successes", "successes"), ("success_rate", "success_rate"), ("mean_success_steps", "mean_success_steps")):
                close(row[f"{label}_{field}"], expected[label][summary_field], "Summary outcome differs from frozen episodes")
        test_seconds = expected["iid"]["elapsed_seconds"] + expected["ood"]["elapsed_seconds"]
        close(row["test_seconds"], test_seconds, "Summary test time wrong")
        close(row["evaluation_seconds"], test_seconds + expected["dev_seconds"], "Summary evaluation time wrong")
    return tests, computed


def inspect_videos_and_report(tests: dict) -> dict:
    videos = read("results/videos/manifest.json")
    require(videos["identity"]["test_freeze_hash"] == digest("results/test_freeze.json"), "Video manifest belongs to another test freeze")
    entries = videos["videos"]
    require(len(entries) == 16, "Video manifest must record sixteen fixed success/failure slots")
    lookup = {(entry["run_id"], entry["split"], entry["kind"]): entry for entry in entries}
    require(len(lookup) == 16, "Duplicate video selection slot")
    counts = {"recorded": 0, "absent": 0, "unavailable": 0}
    unavailable = []
    for (run_id, split), result in tests.items():
        for success, kind in ((True, "success"), (False, "failure")):
            entry = lookup[(run_id, split, kind)]
            episode = next((e for e in result["episodes"] if e["success"] == success), None)
            if episode is None:
                require(entry["status"] == "absent" and bool(entry.get("reason")), "Nonexistent outcome must be documented as absent video")
            else:
                require(entry["episode_id"] == episode["episode_id"], "Video did not choose first outcome in frozen episode order")
                require(entry["status"] in ("recorded", "unavailable"), "Existing outcome has invalid video status")
                if entry["status"] == "recorded":
                    digest(entry["path"], entry["sha256"])
                    require(entry["trajectory_sha256"] == episode["trajectory_sha256"], "Video replay differs from frozen numerical trajectory")
                    require(entry["fps"] > 0, "Video FPS invalid")
                else:
                    require(bool(entry.get("reason")), "Unavailable video lacks documented reason")
                    unavailable.append({key: entry[key] for key in ("run_id", "split", "kind", "reason")})
            counts[entry["status"]] += 1
    for name in ("iid_success.png", "ood_success.png", "training_and_dev.png"):
        path = ROOT / "results/figures" / name
        digest(path)
        require(path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"Plot is not PNG: {name}")
    for path in ("ROUND1_REPORT.md", "README.md", "PROGRESS.md", "results/compute_cost.json"):
        digest(path)
        require((ROOT / path).stat().st_size > 0, f"Empty delivery document: {path}")
    pair_report = read("results/paired_integrity.json")
    require(pair_report["complete"] and pair_report["passed"] and all(pair_report["pairs"][task]["passed"] for task in TASKS), "Reported pairing integrity incomplete")
    return {"counts": counts, "documented_unavailable": unavailable}


def main() -> int:
    started = time.perf_counter()
    torch.set_num_threads(1)
    try:
        frozen, manifest, manifest_hash = check_frozen_inputs()
        training_hash, evaluation_hash = expected_code_hashes()
        runs, pairs = inspect_training(frozen, manifest, manifest_hash, training_hash)
        tests, computed = inspect_results(frozen, manifest, manifest_hash, runs, evaluation_hash)
        videos = inspect_videos_and_report(tests)
        result = {"passed": True, "audit_version": 1, "independent_of_experiment_implementation": True,
                  "auditor_sha256": digest(Path(__file__)), "data_manifest_sha256": manifest_hash,
                  "formal_runs_checked": 4, "formal_updates_per_run": 20000, "checkpoints_inspected_on_cpu": 12,
                  "development_files_checked": 12, "development_episodes_checked": 240,
                  "final_test_files_checked": 8, "final_test_episodes_checked": 160,
                  "normalizers_independently_refit_from_train20": 4, "pairs": pairs,
                  "results": computed, "videos": videos, "artifact_hashes": dict(sorted(ARTIFACT_HASHES.items())),
                  "elapsed_seconds": time.perf_counter() - started,
                  "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        path = ROOT / "artifacts/delivery_audit.json"
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
        print(f"PASS: 4 formal runs, 12 checkpoints, 240 dev + 160 test episodes; {path}")
        return 0
    except Exception as error:
        print(f"DELIVERY AUDIT FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
