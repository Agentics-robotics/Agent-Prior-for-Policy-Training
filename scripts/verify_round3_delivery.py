"""Read-only, CPU-only acceptance of existing Round 3 numerical artifacts.

Run through Pixi: pixi run python scripts/verify_round3_delivery.py [--stage test].
Dev mode never opens or hashes locked-test files. A passing dev subset audit is
not matrix completion. Test mode requires complete public evidence and verifies
the global freeze before opening sealed manifests or outcome records. No policy
inference, simulator replay, optimization, selection writes, or manifest updates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

# Keep this executable independent of the caller's working directory/install.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for thread_variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_variable] = "1"
import numpy as np
import torch
from relative_dp.config import TRAIN_CONFIG
from relative_dp.model import state_hash
from relative_dp.utils import atomic_json, object_hash
from round3.common import TASKS, NS, CANDIDATES, run_id
from round3.evaluate import COUNTS, SPLITS, metrics, _candidate_key
from round3.learning import Normalizer, initialized
from round3.plugins import load_plugin

R3 = ROOT / "round3"
MILESTONES = (5000, 10000, 20000)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def snapshot_hash(snapshot):
    digest = hashlib.sha256()
    for key, value in sorted(snapshot.items()):
        digest.update(key.encode())
        digest.update(np.asarray(value, dtype=np.float64).tobytes())
    return digest.hexdigest()


def close(actual, expected, message):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)) and
            math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-10), message)


class Audit:
    def __init__(self, stage):
        self.stage = stage
        self.hash_cache, self.artifact_hashes, self.signatures = {}, {}, {}
        self.locked_hash_allowed = self.locked_content_allowed = False
        self.data, self.episodes, self.snapshots, self.training = {}, {}, {}, {}
        self.results = {"dev": {}, "test": {}}
        self.feedback, self.pending = {}, {"training": [], "dev": [], "test": []}
        self.invalid, self.baselines, self.pairing = {}, {}, {}
        self.counts = Counter()
        self.context = "initialization"
        self.gate = None

    def path(self, path, content=False):
        path = (ROOT / path).resolve()
        require(path.is_relative_to(ROOT), f"Artifact escapes repository: {path}")
        if any(x in path.parts for x in ("locked_test_states", "locked_test_results")):
            require(self.locked_content_allowed if content else self.locked_hash_allowed,
                    f"Locked-test access rejected before validated global freeze: {path}")
        return path

    @staticmethod
    def signature(path):
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    def digest(self, path, expected=None):
        path = self.path(path)
        signature = self.signature(path)
        # Cache by physical file, never by claimed hash. Copies are each checked.
        if signature not in self.hash_cache:
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            require(signature == self.signature(path), f"Artifact changed while hashing: {path}")
            self.hash_cache[signature] = h.hexdigest()
            self.counts["physical_files_hashed"] += 1
            self.counts["bytes_hashed"] += signature[2]
        actual = self.hash_cache[signature]
        name = str(path.relative_to(ROOT))
        if name in self.signatures:
            require(self.signatures[name] == signature, f"Previously checked artifact changed: {name}")
        self.signatures[name] = signature
        self.artifact_hashes[name] = actual
        require(expected is None or actual == expected, f"SHA256 mismatch: {name}")
        return actual

    def read(self, path):
        path = self.path(path, content=True)
        self.digest(path)
        return json.loads(path.read_text())

    def arrays(self, path):
        path = self.path(path, content=True)
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive}

    def matrix(self):
        self.context = "run manifest"
        # Mutable coordinator ledger: snapshot once and exclude from final stat check.
        manifest_path = R3 / "run_manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        self.manifest_snapshot_hash = hashlib.sha256(manifest_bytes).hexdigest()
        self.runs = manifest["runs"]
        expected = {run_id(t, c, n) for t in TASKS for n in NS for c in CANDIDATES}
        ids = [r["run_id"] for r in self.runs]
        require(len(ids) == len(set(ids)), "Duplicate formal run ID")
        main = [r for r in self.runs if r["candidate_id"] != "P4"]
        require(len(main) == 96 and {r["run_id"] for r in main} == expected, "Main matrix is not exact 6x4x4x1 = 96")
        require(96 <= len(self.runs) <= 108, "Formal run budget outside [96,108]")
        cfg = manifest["config"]
        for key, value in dict(tasks=TASKS, demonstration_counts=NS, train_seeds=[0], candidates=CANDIDATES,
                               initial_runs=96, maximum_formal_runs=108, updates_per_system=20000,
                               batch_chunks_per_system_update=128, primary_checkpoint="20k EMA",
                               maximum_episode_steps=500, dev_episodes=COUNTS["dev"], test_episodes=COUNTS["test"]).items():
            require(cfg[key] == value, f"Manifest protocol mismatch: {key}")
        self.digest(R3 / "ROUND3_SPEC.txt", cfg["specification_sha256"])
        require(cfg["revisions"] == dict(initial_feasibility=1, dev_feedback=1, feedback_feasibility=1), "Design search budget changed")
        for r in self.runs:
            rid = r["run_id"]
            require(r["task"] in TASKS and r["train_n"] in NS and r["train_seed"] == 0 and r["formal"] is True, f"Nonformal run: {rid}")
            require(rid == run_id(r["task"], r["candidate_id"], r["train_n"]), f"Run ID/config mismatch: {rid}")
            require(r.get("feedback_revision") == (r["candidate_id"] == "P4"), f"Revision flag mismatch: {rid}")
            if r["candidate_id"] == "P4":
                require(r["train_n"] in (5, 20), f"Disallowed P4 demonstration count: {rid}")
            if r["status"] == "invalid":
                reason = r.get("invalid_reason", r.get("reason"))
                require(reason, f"Invalid run lacks reason: {rid}")
                self.invalid[rid] = reason
        # Inspect only public formal directories for unlisted completion records.
        for path in (R3 / "checkpoints").glob("*/complete.json"):
            require(path.parent.name in ids, f"Unlisted formal training completion: {path}")

    def inspect_data(self, task, stage="dev"):
        self.context = f"{task}: {stage} frozen data"
        if stage == "dev":
            path = R3 / "data" / task / "manifest.json"
            data = self.read(path)
            require(data["complete"] is True and data["task"] == task and len(data["train20"]) == 20, f"Incomplete D20: {task}")
            protocol_path = R3 / "protocol_and_task_manifests" / f"{task}.json"
            self.digest(protocol_path, data["task_manifest_hash"])
            self.digest(R3 / "calibration" / task / "results.json", data["calibration_hash"])
            self.digest(R3 / "evidence" / task / "bundle.json", data["evidence_bundle_hash"])
            self.data[task] = data
            self.episodes[task] = []
            records = data["train20"] + [r for s in SPLITS for r in data[stage][s]]
            for n in NS:
                selected = data["train20"][:n]
                expected = dict(episode_ids=[r["episode_id"] for r in selected], transitions=sum(r["transitions"] for r in selected),
                                cell_counts={c: sum(r["cell"] == c for r in selected) for c in ("LL", "HH")})
                require(data["subsets"][str(n)] == expected, f"Nested D{n} IDs/counts changed: {task}")
                require([r["cell"] for r in selected] == ["LL" if i % 2 == 0 else "HH" for i in range(n)], f"Data selection order changed: {task}")
            replay = self.read(path.with_name("replay_audit.json"))
            require(replay["passed"] is True and replay["full_demos"] == 20 and len(replay["records"]) == 20, f"Incomplete replay evidence: {task}")
            require([r["episode_id"] for r in replay["records"]] == [r["episode_id"] for r in data["train20"]] and all(r["passed"] for r in replay["records"]), f"Replay identities mismatch: {task}")
        else:
            path = R3 / "locked_test_states" / task / "manifest.json"
            self.digest(path, self.data[task]["sealed_test_manifest_hash"])
            data = self.read(path)
            records = [r for s in SPLITS for r in data[stage][s]]
        for split, count in COUNTS[stage].items():
            require(len(data[stage][split]) == count, f"Wrong {task}/{stage}/{split} state count")
            require(all(r["split"] == split and r["stage"] == stage for r in data[stage][split]), "State split identity differs")
        for rec in records:
            require(rec["task"] == task, "State task mismatch")
            self.digest(rec["path"], rec["sha256"])
            arrays = self.arrays(rec["path"])
            snapshot = {k: v for k, v in arrays.items() if k.startswith("snapshot_")}
            require(snapshot and snapshot_hash(snapshot) == rec["initial_state_hash"], f"Snapshot initial hash mismatch: {rec['episode_id']}")
            require(np.isfinite(snapshot["snapshot_initial_obs"]).all(), "Nonfinite reset observation")
            for field in ("episode_id", "seed", "inference_seed", "initial_state_hash"):
                key = ("unique", field)
                seen = self.pairing.setdefault(key, set())
                require(rec[field] not in seen, f"Duplicate train/dev/test {field}: {rec[field]}")
                seen.add(rec[field])
            self.snapshots[rec["episode_id"]] = snapshot["snapshot_initial_obs"].astype(np.float32)
            self.counts["reset_snapshots_checked"] += 1
            if rec["stage"] == "train":
                actions, obs = arrays["actions"], arrays["obs"]
                require(actions.shape == (rec["transitions"], 4) and obs.shape == (len(actions) + 1, len(self.snapshots[rec["episode_id"]])), "Malformed demonstration")
                require(np.isfinite(actions).all() and np.isfinite(obs).all() and np.abs(actions).max() <= 1, "Invalid demonstration values")
                require(arrays["success"][-1] and arrays["valid_joint"].all() and rec["expert"]["success"] and not rec["expert"]["exception"], "Invalid expert demonstration")
                np.testing.assert_array_equal(obs[0], self.snapshots[rec["episode_id"]])
                self.episodes[task].append(dict(obs=obs, actions=actions))
                self.counts["demonstrations_checked"] += 1
        return data

    def inspect_design(self, task):
        self.context = f"{task}: initial proposal and implementation identities"
        directory = R3 / "design_records" / task
        proposal = self.read(directory / "proposal.json")
        freeze = self.read(directory / "initial_freeze.json")
        require(freeze["proposal_sha256"] == self.digest(directory / "proposal.json"), f"Initial proposal changed: {task}")
        require([c["candidate_id"] for c in proposal["candidates"]] == ["P1", "P2", "P3"], "Wrong initial candidates")
        require(freeze["evidence_hashes"] == proposal["evidence_hashes"], "Initial evidence freeze mismatch")
        for name, digest in proposal["evidence_hashes"].items():
            require(self.path(name).is_relative_to(R3 / "evidence" / task), "Initial design evidence outside D2 bundle")
            self.digest(name, digest)
        schema = self.read(R3 / "protocol_and_task_manifests" / f"{task}.json")["observation_schema"]
        for candidate in CANDIDATES:
            implementation = self.read(R3 / "configs" / task / f"{candidate}.json")
            require(implementation["candidate_id"] == candidate and implementation["observation_schema_sha256"] == object_hash(schema), "Implementation schema identity differs")
            self.digest(R3 / "ROUND3_SPEC.txt" if candidate == "B0" else directory / "proposal.json", implementation["proposal_sha256"])

    def inspect_training(self, run):
        rid = run["run_id"]
        self.context = f"{rid}: training"
        directory = R3 / "checkpoints" / rid
        path = directory / "complete.json"
        if not path.exists():
            require(run["train_status"] != "completed", f"Ledger claims missing training completion: {rid}")
            if rid not in self.invalid:
                self.pending["training"].append(rid)
            return
        done, identity, cfg = self.read(path), self.read(directory / "identity.json"), self.read(directory / "config.json")
        if run.get("train_complete_sha256"):
            self.digest(path, run["train_complete_sha256"])
        for key in ("run_id", "task", "candidate_id", "train_n"):
            require(done[key] == identity[key] == cfg[key] == run[key], f"Training identity mismatch: {rid}/{key}")
        require(done["step"] == 20000 and done["debug"] is False and not cfg.get("debug", False), "Nonformal or incomplete training")
        shared_keys = ("train_updates", "batch_size", "microbatch_size", "train_seed", "model_init_seed", "loader_seed", "diffusion_seed", "learning_rate", "weight_decay", "betas", "max_grad_norm", "optimizer", "lr_schedule", "checkpoint_updates", "ema_decay", "n_obs_steps", "prediction_horizon", "diffusion_train_timesteps", "prediction_type", "inference_steps", "ddim_eta", "inference_scheduler")
        for key in shared_keys:
            require(cfg[key] == TRAIN_CONFIG[key], f"Shared budget/hyperparameter differs: {rid}/{key}")
        require(cfg["clip_predicted_clean_actions"] is False and cfg["thresholding"] is False, "Wrong DDIM clipping")
        require(done["samples_drawn"] == 20000 * 128, "Whole-system chunk budget mismatch")
        require(object_hash(cfg) == done["config_hash"], "Training configuration hash mismatch")
        implementation = self.read(R3 / "configs" / run["task"] / f"{run['candidate_id']}.json")
        require(cfg["implementation"] == implementation and object_hash(implementation) == done["implementation_hash"] == identity["implementation_hash"], "Implementation hash mismatch")
        data_path = R3 / "data" / run["task"] / "manifest.json"
        require(self.digest(data_path) == done["manifest_hash"] == identity["manifest_hash"], "Training data manifest changed")
        require(object_hash(identity["source_hashes"]) == done["code_hash"] == identity["code_hash"], "Training source-set identity mismatch")
        for name, digest in identity["source_hashes"].items():
            self.digest(name, digest)
            self.digest(directory / "source_snapshot" / name, digest)
        # Derive the source file set without re-hashing common files 96 times.
        import importlib
        module = importlib.import_module(implementation.get("plugin_module", "round3.plugins"))
        expected_sources = {"src/round3/learning.py", "src/round3/plugins.py", "pixi.lock", "src/relative_dp/model.py", "src/relative_dp/train.py", "src/relative_dp/dataset.py", "src/relative_dp/config.py", "src/round2/learning.py", str(Path(module.__file__).resolve().relative_to(ROOT))}
        expected_sources.update(str(p.relative_to(ROOT)) for p in (ROOT / "src/relative_dp/vendor/diffusion_policy").glob("*.py"))
        expected_sources.update(implementation.get("additional_source_files", []))
        require(set(identity["source_hashes"]) == expected_sources, "Training source file omitted or added")
        schema = self.read(R3 / "protocol_and_task_manifests" / f"{run['task']}.json")["observation_schema"]
        require(cfg["observation_schema"] == schema, "Checkpoint observation schema differs")
        plugin = load_plugin(run["task"], schema, implementation)
        records = self.data[run["task"]]["train20"][:run["train_n"]]
        norm = Normalizer.fit(self.episodes[run["task"]][:run["train_n"]], plugin,
                              [r["episode_id"] for r in records], {r["episode_id"]: r["sha256"] for r in records}).as_dict()
        require(norm == self.read(directory / "normalizer.json"), f"Current-DN normalization refit differs: {rid}")
        require(identity["windows"] == norm["count"], "Training transition count differs")
        model = initialized(cfg, plugin, "cpu")
        parameters = sum(p.numel() for p in model.parameters())
        if cfg["raw_dim"] not in self.baselines:
            from round2.learning import Policy
            baseline = Policy(cfg["raw_dim"], dict(TRAIN_CONFIG, clip_predicted_clean_actions=False, thresholding=False))
            self.baselines[cfg["raw_dim"]] = sum(p.numel() for p in baseline.parameters())
            del baseline
        baseline_parameters = self.baselines[cfg["raw_dim"]]
        require(parameters == done["parameter_count"] == identity["parameter_count"] and parameters <= 3 * baseline_parameters, "Parameter budget/count mismatch")
        require(done["baseline_parameter_count"] == identity["baseline_parameter_count"] == baseline_parameters, "B0 parameter reference mismatch")
        require(state_hash(model.state_dict()) == done["initial_weights_hash"] == identity["initial_weights_hash"], "Training was not initialized from prescribed seed/config")
        require(set(done["checkpoints"]) == {str(s) for s in MILESTONES}, "Missing 5k/10k/20k recovery points")
        for step in MILESTONES:
            item = done["checkpoints"][str(step)]
            checkpoint_path = directory / "checkpoints" / f"step_{step:06d}.pt"
            require(self.path(item["path"]) == checkpoint_path, "Checkpoint path mismatch")
            self.digest(checkpoint_path, item["sha256"])
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
            for key in ("run_id", "task", "candidate_id", "train_n", "manifest_hash", "code_hash", "debug", "implementation_hash", "config_hash", "parameter_count", "initial_weights_hash"):
                require(checkpoint[key] == done[key], f"Checkpoint identity mismatch: {rid}/{step}/{key}")
            require(checkpoint["step"] == checkpoint["update"] == step and checkpoint["samples_drawn"] == step * 128, "Checkpoint update/chunk count mismatch")
            require(checkpoint["config"] == cfg and checkpoint["normalizer"] == norm, "Checkpoint config/normalizer mismatch")
            require(set(checkpoint["rng"]) == {"loader", "diffusion", "torch", "numpy", "python", "cuda"}, "Incomplete resume RNG state")
            optimizer = checkpoint["optimizer"]
            expected_parameter_tensors = len(list(model.parameters()))
            require(len(optimizer["state"]) == expected_parameter_tensors and all(int(x["step"]) == step for x in optimizer["state"].values()), "Optimizer whole-system step/state mismatch")
            for group in optimizer["param_groups"]:
                require(group["lr"] == cfg["learning_rate"] and group["weight_decay"] == cfg["weight_decay"] and list(group["betas"]) == cfg["betas"], "Optimizer hyperparameters differ")
            for kind in ("model", "ema"):
                model.load_state_dict(checkpoint[kind], strict=True)
                require(all(torch.isfinite(value).all().item() for value in checkpoint[kind].values()), f"Nonfinite {kind} checkpoint")
            if step == 20000:
                require(checkpoint["pairing_digest"] == done["pairing_digest"], "Final stochastic training digest differs")
                close(checkpoint["elapsed_seconds"], done["elapsed_seconds"], "Training time differs")
            del checkpoint
            self.counts["cpu_checkpoints_checked"] += 1
        rows = [json.loads(line) for line in self.path(directory / "train.jsonl", content=True).read_text().splitlines()]
        self.digest(directory / "train.jsonl")
        steps = [r["step"] for r in rows]
        require(steps == sorted(set(steps)) and steps[0] == 1 and steps[-1] == 20000 and set(range(100, 20001, 100)) <= set(steps), "Training curve update coverage/duplicates differ")
        require(all(math.isfinite(r["loss"]) and math.isfinite(r["grad_norm"]) for r in rows), "Nonfinite training curve")
        require(rows[-1]["pairing_digest"] == done["pairing_digest"], "Training curve final digest differs")
        del model
        self.training[rid] = done
        self.counts["normalizers_refit_from_current_DN"] += 1

    def inspect_trajectory(self, row, reset):
        self.digest(row["trajectory_path"], row["trajectory_hash"])
        arrays = self.arrays(row["trajectory_path"])
        actions, obs = arrays["actions"], arrays["obs"]
        steps = len(actions)
        require(steps <= 500 and actions.shape == (steps, 4) and len(obs) == steps + 1, "Trajectory action/observation budget mismatch")
        require(np.isfinite(actions).all() and np.isfinite(obs).all() and (not steps or np.abs(actions).max() <= 1), "Trajectory contains invalid native action/state")
        np.testing.assert_array_equal(obs[0], self.snapshots[reset["episode_id"]])
        infos = row["infos"]
        require(len(infos) == steps + 1 and not arrays["success"][0], "Trajectory info count or reset success differs")
        for key, default in (("success", False), ("valid_joint", True)):
            np.testing.assert_array_equal(arrays[key], [bool(info.get(key, default)) for info in infos])
        for key in ("terminated", "truncated"):
            require(arrays[key].shape == (steps,), f"Malformed {key} flags")
        reached = np.flatnonzero(arrays["success"])
        first = int(reached[0]) if len(reached) else None
        invalid = bool(not arrays["valid_joint"].all())
        success = first is not None and not invalid and row["exception"] is None
        require(first is None or first == steps, "Actions executed after success")
        require(not arrays["terminated"][:-1].any() and not arrays["truncated"][:-1].any(), "Actions executed after native termination")
        for key, value in dict(steps=steps, success=success, first_success_step=first if success else None,
                               first_threshold_step=first, invalid_joint=invalid,
                               terminated=bool(steps and arrays["terminated"][-1]), truncated=bool(steps and arrays["truncated"][-1])).items():
            require(row[key] == value, f"Trajectory/record {key} mismatch: {row['episode_id']}")
        if success and row["identity"]["task"] in ("drawer", "door"):
            progress = arrays["progress"]
            require(steps >= 3 and np.all(progress[-3:] >= .75) and np.all(progress >= -.01) and np.all(progress <= 1.01), "Cabinet success violates physical hold/joint criterion")
        if row["exception"]:
            reason, category = "exception", "exception"
        elif first is not None:
            reason = "success" if success else "success_threshold_with_joint_violation"
            category = "success" if success else "joint_violation"
        elif row["terminated"] or row["truncated"]:
            reason = "native_terminated" if row["terminated"] else "native_truncated"
            category = "joint_violation" if invalid else reason
        else:
            reason = "max_steps"
            require(steps == 500, "Unexplained short failed trajectory")
            distances = [x["distance"] for x in infos if x.get("distance") is not None]
            contacts = [x["direct_contact"] for x in infos if "direct_contact" in x]
            progress = [x["progress"] for x in infos if x.get("progress") is not None]
            category = ("joint_violation" if invalid else "no_approach" if distances and min(distances) > .08 else
                        "approach_no_contact" if contacts and not any(contacts) else
                        ("contact_no_progress" if any(contacts) else "low_progress") if progress and max(progress) < .1 else "goal_not_completed")
        require(row["termination_reason"] == reason and row["failure_category"] == category, "Outcome category/termination differs from recorded facts")
        clipping = arrays["clipping"]
        require(clipping.shape in ((steps, 4), (steps + 1, 4)) and (len(clipping) == steps or row["exception"]), "Clipping count differs")
        require(row["clipped_action_steps"] == int(np.any(clipping, axis=1).sum()) and row["clipped_coordinates"] == clipping.sum(0).tolist(), "Clipping metrics differ")
        inference = row["inference"]
        require(row["replans"] == len(inference), "Replan count differs")
        np.testing.assert_array_equal(arrays["replan_steps"], [r["at_step"] for r in inference])
        np.testing.assert_array_equal(arrays["inference_seconds"], [r["seconds"] for r in inference])
        require(np.isfinite(arrays["inference_seconds"]).all() and (arrays["inference_seconds"] >= 0).all(), "Invalid inference timings")
        close(row["inference_seconds"], sum(r["seconds"] for r in inference), "Inference total differs")
        for i, call in enumerate(inference):
            require(1 <= call["execute_steps"] <= 16 and 0 <= call["at_step"] <= steps, "Invalid action execution horizon")
            require((i != 0 or call["at_step"] == 0), "First replan does not start at step zero")
            if i:
                require(call["at_step"] == inference[i - 1]["at_step"] + inference[i - 1]["execute_steps"], "Unaccounted action chunk steps")
        require(not steps or inference and 0 <= steps - inference[-1]["at_step"] <= inference[-1]["execute_steps"], "Last replan exceeds execution horizon")
        require(math.isfinite(row["wall_seconds"]) and row["wall_seconds"] >= 0, "Invalid rollout wall time")
        self.counts[f"{row['identity']['stage']}_episodes_checked"] += 1

    def inspect_result(self, run, stage, manifest=None):
        rid = run["run_id"]
        self.context = f"{rid}: {stage} trajectories"
        directory = R3 / ("dev_results" if stage == "dev" else "locked_test_results") / rid
        path = directory / "complete.json"
        if not path.exists():
            require(run[stage + "_status"] != "completed", f"Ledger claims missing {stage} completion: {rid}")
            if rid not in self.invalid:
                self.pending[stage].append(rid)
            return
        require(rid in self.training, "Evaluation exists without validated formal training")
        result = self.read(path)
        if run.get(stage + "_result_hash"):
            self.digest(path, run[stage + "_result_hash"])
        if run.get(stage + "_result_path"):
            require(self.path(run[stage + "_result_path"]) == path, "Ledger evaluation path differs")
        manifest_path = R3 / ("data" if stage == "dev" else "locked_test_states") / run["task"] / "manifest.json"
        expected_identity = dict(run_id=rid, step=20000, stage=stage, task=run["task"], candidate_id=run["candidate_id"],
                                 train_n=run["train_n"], checkpoint_hash=self.training[rid]["checkpoints"]["20000"]["sha256"],
                                 data_manifest_hash=self.digest(manifest_path), evaluation_implementation_hash=self.digest(ROOT / "src/round3/evaluate.py"),
                                 global_freeze_hash=self.digest(R3 / "global_freeze.json") if stage == "test" else None)
        require(result["identity"] == expected_identity, "Evaluation final-20k-EMA identity differs")
        manifest = manifest or self.data[run["task"]]
        expected = [r for split in SPLITS for r in manifest[stage][split]]
        rows = result["records"]
        require([r["episode_id"] for r in rows] == [r["episode_id"] for r in expected] and len({r["episode_id"] for r in rows}) == sum(COUNTS[stage].values()), "Evaluation snapshot set/order/uniqueness differs")
        pairs = []
        for row, reset in zip(rows, expected, strict=True):
            require(row["identity"] == expected_identity and row["split"] == reset["split"], "Per-record identity differs")
            require(row["initial_state_hash"] == reset["initial_state_hash"] and row["reset_record_hash"] == object_hash(reset) and row["inference_seed"] == reset.get("inference_seed", reset.get("policy_noise_seed")), "Per-record reset hash/paired RNG differs")
            require(self.read(directory / (row["episode_id"] + ".json")) == row, "Aggregate row differs from standalone record")
            require(self.path(row["trajectory_path"]) == directory / (row["episode_id"] + ".npz"), "Trajectory path differs")
            self.inspect_trajectory(row, reset)
            pairs.append((row["episode_id"], row["initial_state_hash"], row["reset_record_hash"], row["inference_seed"]))
        pair_key = (run["task"], stage)
        if pair_key in self.pairing:
            require(self.pairing[pair_key] == pairs, "Cross-candidate/N snapshot or seed pairing differs")
        self.pairing[pair_key] = pairs
        for split in SPLITS:
            require(result["metrics"][split] == metrics([r for r in rows if r["split"] == split]), f"Aggregate {split} metrics differ from checked trajectories")
        close(result["ood_success_rate"], (result["metrics"]["C"]["success_rate"] + result["metrics"]["E"]["success_rate"]) / 2, "C/E equal-weight score differs")
        close(result["measured_rollout_wall_seconds"], sum(r["wall_seconds"] for r in rows), "Rollout wall total differs")
        require(result["parameter_count"] == self.training[rid]["parameter_count"], "Reported model size differs")
        # Retain only what later pairing/selection/feedback checks need. Full
        # per-step infos and trajectories are released after each run, keeping
        # peak memory bounded as the matrix grows from a subset to 108 runs.
        compact = {key: value for key, value in result.items() if key != "records"}
        compact["records"] = [{key: row[key] for key in ("episode_id", "success", "steps")}
                              for row in rows]
        self.results[stage][rid] = compact

    def inspect_feedback(self, task, required=False):
        self.context = f"{task}: one feedback cycle"
        directory = R3 / "design_records" / task
        path = directory / "revision.json"
        p4 = [r for r in self.runs if r["task"] == task and r["candidate_id"] == "P4"]
        if not path.exists():
            require(not required and not p4, f"Missing feedback decision: {task}")
            return
        revision = self.read(path)
        choice = revision.get("decision", revision.get("choice", revision.get("candidate_id")))
        if choice is None and revision.get("no_revision"):
            choice = "no_revision"
        require(choice in ("P4", "no_revision"), "Invalid feedback decision")
        require((len(p4) == 2 and {r["train_n"] for r in p4} == {5, 20}) if choice == "P4" else not p4, "P4 plan does not match single feedback decision")
        if "feedback_cycles_used" in revision:
            require(revision["feedback_cycles_used"] == 1, "Feedback cycle budget exceeded")
        if "feedback_cycles_remaining" in revision:
            require(revision["feedback_cycles_remaining"] == 0, "Feedback cycle budget exceeded")
        if "revision_control" in revision and "remaining_development_feedback_cycles" in revision["revision_control"]:
            require(revision["revision_control"]["remaining_development_feedback_cycles"] == 0, "Feedback decision did not exhaust single cycle")
        packet_path = directory / "feedback/bundle.json"
        packet = self.read(packet_path)
        require(packet["task"] == task and packet["stage"] == "dev" and packet["train_n"] == 5 and packet["contains_locked_test"] is False, "Feedback uses disallowed stage/data")
        require(packet["frame_count"] == len(packet["frames"]) <= 8 and packet["additional_policy_calls_for_visuals"] == 0, "Feedback frame/inference budget exceeded")
        self.digest(R3 / "design_records/feedback_frame_rule.json", packet["frame_rule_sha256"])
        for name, digest in packet["bound_files"].items():
            require("locked_test" not in name, "Locked-test evidence in feedback packet")
            self.digest(name, digest)
        evidence_fields = ("evidence_hashes", "feedback_hashes", "consumed_file_hashes")
        for field in evidence_fields:
            for name, digest in revision.get(field, {}).items():
                require("locked_test" not in name, "Locked-test evidence in feedback decision")
                if name == 'AGENTS.md' and self.digest(name) != digest:
                    # User device instructions can change this operational file.
                    # Verify the consumed bytes from their versioned archive;
                    # never waive a training/data/evidence hash mismatch.
                    archive = R3 / 'provenance_snapshots' / digest / 'AGENTS.md'
                    self.digest(archive, digest)
                    provenance = self.read(archive.with_suffix('.provenance.json'))
                    require(provenance['original_path'] == name and provenance['sha256'] == digest,
                            'Operational evidence archive provenance differs')
                    self.counts['archived_operational_evidence_checked'] += 1
                else:
                    self.digest(name, digest)
        packet_name = packet_path.relative_to(ROOT).as_posix()
        explicit_packet = revision.get("feedback_bundle") == packet_name and bool(revision.get("feedback_bundle_sha256"))
        if explicit_packet:
            self.digest(packet_path, revision["feedback_bundle_sha256"])
        require(explicit_packet or any(packet_name in revision.get(field, {}) for field in evidence_fields),
                "Feedback decision does not bind its packet")
        scored = 0
        valid = {r["candidate_id"]: self.results["dev"][r["run_id"]] for r in self.runs if r["task"] == task and r["train_n"] == 5 and r["candidate_id"] in CANDIDATES and r["run_id"] not in self.invalid}
        require(set(packet["results"]) == set(valid), "Feedback packet omitted eligible N5 candidate")
        for candidate, result in valid.items():
            summary = packet["results"][candidate]
            require(summary["metrics"] == result["metrics"] and summary["ood_success_rate"] == result["ood_success_rate"], "Feedback summary differs from checked development")
            scored += len(result["records"])
        require(packet["scored_development_rollouts"] == scored <= 200, "Feedback rollout count differs")
        rule = self.read(R3 / "design_records/feedback_frame_rule.json")
        by_candidate = {c: {r["episode_id"]: r for r in result["records"]} for c, result in valid.items()}
        states = self.data[task]["dev"]["C"]
        selected = next((r for r in states if len({rows[r["episode_id"]]["success"] for rows in by_candidate.values()}) > 1), states[0])
        require(packet["selected_episode_id"] == selected["episode_id"], "Feedback visual selection differs from frozen rule")
        frame_keys = [(f["candidate_id"], f["requested_step"]) for f in packet["frames"]]
        require(len(frame_keys) == len(set(frame_keys)) and set(frame_keys) == {(c, s) for c in valid for s in rule["physical_steps"]}, "Feedback frame slots missing/duplicated")
        for frame in packet["frames"]:
            row = by_candidate[frame["candidate_id"]][selected["episode_id"]]
            require(frame["episode_id"] == selected["episode_id"] and frame["actual_step"] == min(frame["requested_step"], row["steps"]) and frame["success"] == row["success"], "Feedback frame record differs")
            self.digest(frame["path"], frame["sha256"])
        if choice == "P4":
            impl = self.read(R3 / "configs" / task / "P4.json")
            require(impl["proposal_sha256"] == self.digest(path), "P4 implementation revision identity differs")
        self.feedback[task] = dict(decision=choice, cycles=1, frames=len(packet["frames"]), scored_rollouts=scored,
                                   planned_additional_runs=len(p4), completed_additional_runs=sum(r["run_id"] in self.training for r in p4))

    def inspect_selection(self):
        self.context = "frozen per-N development selections"
        selection = self.read(R3 / "selection.json")
        require(selection["frozen"] is True and selection["primary_checkpoint_step"] == 20000 and selection["unavailable"] == [], "Development selection incomplete/unfrozen")
        require(set(selection["groups"]) == {f"{t}:n{n}" for t in TASKS for n in NS}, "Selection group matrix differs")
        for task in TASKS:
            for n in NS:
                candidates, invalid = [], []
                for run in self.runs:
                    if run["task"] != task or run["train_n"] != n:
                        continue
                    rid = run["run_id"]
                    if rid in self.invalid:
                        invalid.append(dict(run_id=rid, candidate_id=run["candidate_id"], reason=self.invalid[rid]))
                        continue
                    result = self.results["dev"][rid]
                    path = R3 / "dev_results" / rid / "complete.json"
                    candidates.append(dict(run_id=rid, candidate_id=run["candidate_id"], parameter_count=result["parameter_count"],
                                           ood_success_rate=result["ood_success_rate"], iid_success_rate=result["metrics"]["IID"]["success_rate"],
                                           dev_result_path=str(path.relative_to(ROOT)), dev_result_hash=self.digest(path)))
                initial = [c for c in candidates if c["candidate_id"] in ("P1", "P2", "P3")]
                winner = min(candidates, key=_candidate_key) if candidates else None
                expected = dict(task=task, train_n=n, candidates=candidates, invalid=invalid,
                                initial_selected=min(initial, key=_candidate_key) if initial else None, final_system=winner,
                                b0_fallback=bool(winner and winner["candidate_id"] == "B0"))
                require(selection["groups"][f"{task}:n{n}"] == expected, f"Frozen selection violates per-N eligibility/tie rule: {task}/N{n}")
        self.counts["frozen_selection_groups_checked"] = 24

    def test_gate(self):
        self.context = "global freeze gate (public inputs first)"
        require(not self.pending["training"] and not self.pending["dev"], "Locked test audit requires all valid formal training and development results")
        for task in TASKS:
            self.inspect_feedback(task, required=True)
        self.inspect_selection()
        gate = self.read(R3 / "global_freeze.json")
        require(gate.get("frozen") is True and gate["primary_checkpoint_step"] == 20000 and gate["expected_main_runs"] == 96 and gate["planned_runs"] == len(self.runs) and gate["selection_groups"] == 24, "Global freeze dimensions/status invalid")
        required = {"round3/selection.json", "src/round3/evaluate.py", "src/round3/environment.py", "src/round3/tasks.py", "round3/design_records/feedback_frame_rule.json"}
        for task in TASKS:
            base = f"round3/design_records/{task}"
            required.update(f"{base}/{name}" for name in ("proposal.json", "revision.json", "initial_freeze.json", "feedback/bundle.json"))
            required.update((f"round3/data/{task}/manifest.json", f"round3/protocol_and_task_manifests/{task}.json", f"round3/locked_test_states/{task}/manifest.json"))
            required.update(self.read(R3 / "design_records" / task / "feedback/bundle.json")["bound_files"])
            for name in ("initial_feasibility_revision.json", "feedback_feasibility_revision.json"):
                if (R3 / "design_records" / task / name).exists():
                    required.add(f"{base}/{name}")
        for run in self.runs:
            rid = run["run_id"]
            if rid in self.invalid:
                continue
            base = f"round3/checkpoints/{rid}"
            required.update(f"{base}/{name}" for name in ("complete.json", "identity.json", "config.json", "checkpoints/step_020000.pt"))
            required.update(self.read(ROOT / base / "identity.json")["source_hashes"])
            required.update((f"round3/dev_results/{rid}/complete.json", f"round3/configs/{run['task']}/{run['candidate_id']}.json"))
        bound = gate["bound_files"]
        require(required <= set(bound), "Global freeze omits required public evidence or sealed manifest")
        for name, digest in bound.items():
            require("locked_test_results" not in name, "Global freeze improperly depends on test outcomes")
            if "locked_test_states" not in name:
                self.digest(name, digest)
        # Only now hash sealed bytes to finish the existing freeze protocol. No
        # sealed JSON/NPZ is parsed until EVERY freeze hash is validated.
        self.locked_hash_allowed = True
        for name, digest in bound.items():
            if "locked_test_states" in name:
                self.digest(name, digest)
        for task in TASKS:
            actual = {str(p.relative_to(ROOT)) for p in (R3 / "locked_test_states" / task).rglob("*") if p.is_file()}
            require(actual <= set(bound), "Global freeze omits a concrete sealed state file")
        self.gate = gate
        self.locked_content_allowed = True

    def finish(self):
        self.context = "final artifact mutation check"
        for name, signature in self.signatures.items():
            require(self.signature(ROOT / name) == signature, f"Artifact changed during audit: {name}")
        main_ids = {r["run_id"] for r in self.runs if r["candidate_id"] != "P4"}
        self.counts.update(planned_main_runs=96, planned_revision_runs=len(self.runs) - 96,
                           training_runs_checked=len(self.training), dev_runs_checked=len(self.results["dev"]),
                           test_runs_checked=len(self.results["test"]), invalid_runs=len(self.invalid), feedback_decisions_checked=len(self.feedback))
        numerical_complete = (self.stage == "test" and not any(self.pending.values()))
        main_fully_trained = main_ids <= self.training.keys()
        return dict(passed=True, stage=self.stage, audit_scope_complete=True, numerical_delivery_complete=numerical_complete,
                    main_matrix_training_complete=main_fully_trained, main_matrix_dev_complete=main_ids <= self.results["dev"].keys(),
                    main_matrix_test_complete=self.stage == "test" and main_ids <= self.results["test"].keys(),
                    counts=dict(self.counts), pending_counts={k: len(v) if self.stage == "test" or k != "test" else None for k, v in self.pending.items()},
                    pending_run_ids={k: v for k, v in self.pending.items() if self.stage == "test" or k != "test"}, invalid_runs=self.invalid,
                    feedback=self.feedback, global_freeze_validated=self.gate is not None,
                    locked_test_contents_accessed=self.locked_content_allowed,
                    run_manifest_snapshot_sha256=self.manifest_snapshot_hash,
                    result_totals={stage: {rid: {"metrics": result["metrics"], "ood_success_rate": result["ood_success_rate"]} for rid, result in results.items()} for stage, results in self.results.items() if stage == "dev" or self.stage == "test"},
                    pairing={f"{task}:{stage}": dict(runs=sum(result["identity"]["task"] == task for result in self.results[stage].values()), snapshots=len(self.pairing.get((task, stage), []))) for task in TASKS for stage in ("dev", "test") if stage == "dev" or self.stage == "test"},
                    artifact_hashes=dict(sorted(self.artifact_hashes.items())))

    def run(self):
        self.matrix()
        for task in TASKS:
            self.inspect_data(task)
            self.inspect_design(task)
        for run in self.runs:
            self.inspect_training(run)
            self.inspect_result(run, "dev")
        for task in TASKS:
            self.inspect_feedback(task)
        if self.stage == "test":
            self.test_gate()
            test_data = {task: self.inspect_data(task, "test") for task in TASKS}
            for run in self.runs:
                self.inspect_result(run, "test", test_data[run["task"]])
            require(not self.pending["test"], "Locked-test numerical delivery is incomplete")
        return self.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("dev", "test"), default="dev")
    parser.add_argument("--output", type=Path, help="Audit JSON only; defaults to a fresh timestamped round3/audits file")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    audit = Audit(args.stage)
    try:
        result = audit.run()
    except Exception as error:
        result = dict(passed=False, stage=args.stage, audit_scope_complete=False, numerical_delivery_complete=False,
                      context=audit.context, error=f"{type(error).__name__}: {error}", counts=dict(audit.counts),
                      pending_counts={k: len(v) if args.stage == "test" or k != "test" else None
                                      for k, v in audit.pending.items()},
                      global_freeze_validated=audit.gate is not None, locked_test_contents_accessed=audit.locked_content_allowed)
    result.update(audit_version=1, auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
                  limits=["CPU artifact verification only; no policy inference, new rollouts, or simulator replay.",
                          "Success and other totals are recomputed from recorded trajectory flags and infos; no unrecorded simulator outcomes are inferred.",
                          "Current-DN normalizers and configured model initialization are recomputed; historical optimizer execution is supported by checkpoint/log evidence, not independently replayed.",
                          "Source, checkpoint, reset, standalone record, and aggregate identities are checked; no claim of adversarially tamper-proof historical provenance.",
                          "The mutable run ledger is snapshotted at audit start; runs completed afterwards may remain pending in this audit.",
                          "A passing dev subset is not complete training, complete evaluation, or final delivery.",
                          "Final report prose, figures, videos, and cross-training-seed stability are outside this numerical audit."])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or R3 / "audits" / f"delivery_verifier_{'development' if args.stage == 'dev' else 'test'}_{stamp}.json"
    output = output.resolve()
    require(output.is_relative_to(R3 / "audits") and output.name.startswith("delivery_verifier_") and output.suffix == ".json", "Audit output must be round3/audits/delivery_verifier_*.json")
    require(not output.exists(), "Refusing to overwrite a prior audit record; choose a new output path")
    atomic_json(output, result)
    concise = {k: result[k] for k in ("passed", "stage", "numerical_delivery_complete", "counts", "pending_counts", "global_freeze_validated", "locked_test_contents_accessed", "elapsed_seconds")}
    concise.update(audit_json=str(output.relative_to(ROOT)))
    if not result["passed"]:
        concise.update(context=result["context"], error=result["error"])
    print(json.dumps(concise, sort_keys=True, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
