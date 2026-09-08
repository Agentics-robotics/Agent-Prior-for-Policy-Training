"""Paired, resumable Round 3 rollouts and development-only selection.

Locked test states are opened only after the global freeze has been verified.
All primary evaluations use the final 20,000-update EMA checkpoint.
"""
from __future__ import annotations

import json
import math
import os
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from relative_dp.utils import ROOT, atomic_json, object_hash, read_json, sha256

R3 = ROOT / "round3"
SPLITS = ("IID", "C", "E")
COUNTS = {"dev": {"IID": 10, "C": 20, "E": 20},
          "test": {"IID": 20, "C": 40, "E": 40}}
STEPS = 20_000
MAX_STEPS = 500


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    return str(value)


def _save_npz(path, arrays):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def wilson_interval(successes, n, z=1.959963984540054):
    """Two-sided 95% binomial Wilson interval, including floor and ceiling."""
    if not n:
        return [None, None]
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def metrics(records):
    n = len(records)
    successes = [row for row in records if row["success"]]
    action_steps = sum(row.get("steps", 0) for row in records)
    replans = sum(row.get("replans", 0) for row in records)
    latency = sum(row.get("inference_seconds", 0.) for row in records)
    return dict(
        n=n, successes=len(successes), success_rate=len(successes) / n if n else None,
        binomial_95ci=wilson_interval(len(successes), n),
        mean_first_success_step=float(np.mean([r["first_success_step"] for r in successes])) if successes else None,
        mean_steps=float(np.mean([r["steps"] for r in records])) if n else None,
        mean_replans=replans / n if n else None,
        inference_seconds=latency, inference_calls=replans,
        mean_inference_latency_ms=1000 * latency / replans if replans else None,
        clipping_step_rate=sum(r.get("clipped_action_steps", 0) for r in records) / action_steps if action_steps else 0.,
        failure_categories=dict(sorted(Counter(r["failure_category"] for r in records).items())),
        exceptions=sum(r.get("exception") is not None for r in records),
        invalid_joint_episodes=sum(r.get("invalid_joint", False) for r in records),
        terminated_episodes=sum(r.get("terminated", False) for r in records),
        truncated_episodes=sum(r.get("truncated", False) for r in records),
        wall_seconds=sum(r.get("wall_seconds", 0.) for r in records),
    )


def _verify_snapshot(env, observation, record):
    path = ROOT / record["path"]
    if record.get("sha256"):
        assert sha256(path) == record["sha256"], f"Reset snapshot changed: {path}"
    with np.load(path, allow_pickle=False) as stored:
        current = env.snapshot(observation)
        for key, value in current.items():
            assert key in stored, f"Missing snapshot field {key}"
            np.testing.assert_allclose(value, stored[key], atol=1e-9, rtol=0,
                                       err_msg=f"Reset snapshot mismatch: {record['episode_id']} {key}")


def run_episode(env, loaded, record, device="cuda", verify_snapshot=True):
    """Evaluate one immutable state; every executed step checks success and stops.

    Policies receive only the two shared observation vectors and their private
    inference RNG. Environment diagnostics never enter the policy interface.
    A route change never resets the physical clock or the episode limit.
    """
    started = time.monotonic()
    observations, actions, infos = [], [], []
    clipping, terminations, truncations, inference = [], [], [], []
    exception = None
    termination_reason = "max_steps"
    first_success = None
    ever_invalid = False
    terminated = truncated = False
    success = False
    # Reset/data-integrity failures are infrastructure errors, not policy failures.
    obs, info = env.reset(record)
    assert not bool(info.get("success", False)), "Frozen reset is already successful"
    if verify_snapshot:
        _verify_snapshot(env, obs, record)
    observations.append(np.array(obs, copy=True))
    infos.append(_jsonable(info))
    ever_invalid = not bool(info.get("valid_joint", True))
    try:
        if hasattr(loaded, "reset"):
            loaded.reset()
        rng_device = torch.device(device).type
        generator = torch.Generator(device=rng_device).manual_seed(
            int(record.get("inference_seed", record.get("policy_noise_seed"))))
        while len(actions) < MAX_STEPS:
            history = np.asarray([observations[max(0, len(observations) - 2)], observations[-1]], np.float32)
            if rng_device == "cuda":
                torch.cuda.synchronize()
            inference_started = time.monotonic()
            output = loaded.actions(history, generator)
            if rng_device == "cuda":
                torch.cuda.synchronize()
            inference_seconds = time.monotonic() - inference_started
            if isinstance(output, tuple):
                chunk, diagnostics = output
            else:
                chunk, diagnostics = output, {}
            if isinstance(chunk, torch.Tensor):
                chunk = chunk.detach().cpu().numpy()
            chunk = np.asarray(chunk)
            assert chunk.shape == (16, 4) and np.isfinite(chunk).all(), "Invalid diffusion action chunk"
            diagnostics = _jsonable(diagnostics or {})
            unclipped = np.asarray(diagnostics.pop("unclipped_world_actions", chunk), np.float32)
            assert unclipped.shape == chunk.shape and np.isfinite(unclipped).all()
            execute = int(diagnostics.get("execution_horizon", diagnostics.get("execute_steps", 4)))
            assert 1 <= execute <= 16, "Execution horizon must be in [1, 16]"
            inference.append(dict(at_step=len(actions), seconds=inference_seconds,
                                  execute_steps=execute, diagnostics=diagnostics))
            for action_index, proposed in enumerate(chunk[:min(execute, MAX_STEPS - len(actions))]):
                # Action frames are decoded by Loaded. This is the sole native cube clip.
                clipping.append(np.abs(unclipped[action_index]) > 1.)
                actual = np.clip(proposed, -1., 1.).astype(np.float32)
                obs, _, terminated, truncated, info = env.step(actual.copy())
                actions.append(actual)
                observations.append(np.array(obs, copy=True))
                infos.append(_jsonable(info))
                terminations.append(bool(terminated))
                truncations.append(bool(truncated))
                ever_invalid |= not bool(info.get("valid_joint", True))
                reached = bool(info.get("success", False))
                if reached and first_success is None:
                    first_success = len(actions)
                if reached:
                    success = not ever_invalid
                    termination_reason = "success" if success else "success_threshold_with_joint_violation"
                    break
                if terminated or truncated:
                    termination_reason = "native_terminated" if terminated else "native_truncated"
                    break
            if first_success is not None or terminated or truncated:
                break
    except Exception:
        exception = traceback.format_exc()
        termination_reason = "exception"
        success = False
    if exception:
        category = "exception"
    elif success:
        category = "success"
    elif ever_invalid:
        category = "joint_violation"
    elif terminated or truncated:
        category = termination_reason
    else:
        distances = [x.get("distance") for x in infos if x.get("distance") is not None]
        contacts = [x.get("direct_contact") for x in infos if "direct_contact" in x]
        progress = [x.get("progress") for x in infos if x.get("progress") is not None]
        if distances and min(distances) > .08:
            category = "no_approach"
        elif contacts and not any(contacts):
            category = "approach_no_contact"
        elif progress and max(progress) < .1:
            category = "contact_no_progress" if any(contacts) else "low_progress"
        else:
            category = "goal_not_completed"
    clipping_array = np.asarray(clipping, bool).reshape(-1, 4)
    result = dict(
        episode_id=record["episode_id"], split=record["split"], cell=record.get("cell"),
        success=bool(success), steps=len(actions), first_success_step=first_success if success else None,
        first_threshold_step=first_success, invalid_joint=bool(ever_invalid),
        terminated=bool(terminated), truncated=bool(truncated), termination_reason=termination_reason,
        failure_category=category, exception=exception, initial_state_hash=record["initial_state_hash"],
        reset_record_hash=object_hash(record), inference_seed=int(record.get("inference_seed", record.get("policy_noise_seed"))),
        replans=len(inference), inference_seconds=sum(x["seconds"] for x in inference),
        inference=inference, infos=infos,
        clipped_action_steps=int(np.any(clipping_array, axis=-1).sum()),
        clipped_coordinates=clipping_array.sum(0).tolist(),
        wall_seconds=time.monotonic() - started,
    )
    obs_dim = observations[0].shape[-1] if observations else getattr(loaded, "obs_dim", 0)
    arrays = dict(obs=np.asarray(observations, np.float32).reshape(-1, obs_dim) if obs_dim else np.empty((0, 0), np.float32),
                  actions=np.asarray(actions, np.float32).reshape(-1, 4), clipping=clipping_array,
                  success=np.asarray([bool(x.get("success", False)) for x in infos]),
                  valid_joint=np.asarray([bool(x.get("valid_joint", True)) for x in infos]),
                  terminated=np.asarray(terminations, bool), truncated=np.asarray(truncations, bool),
                  replan_steps=np.asarray([x["at_step"] for x in inference], np.int32),
                  inference_seconds=np.asarray([x["seconds"] for x in inference]))
    numeric_keys = sorted(set().union(*(x.keys() for x in infos))) if infos else []
    for key in numeric_keys:
        values = [x.get(key) for x in infos]
        if all(v is None or isinstance(v, (int, float, bool)) for v in values):
            if key not in arrays:
                arrays[key] = np.asarray([np.nan if v is None else v for v in values], float)
    return result, arrays


def _checked_result(path):
    result = read_json(path)
    for row in result["records"]:
        assert sha256(ROOT / row["trajectory_path"]) == row["trajectory_hash"]
        assert row["identity"] == result["identity"]
    for split, count in COUNTS[result["identity"]["stage"]].items():
        assert result["metrics"][split]["n"] == count
    return result


def validate_test_gate():
    gate_path = R3 / "global_freeze.json"
    assert gate_path.exists(), "Locked test is blocked until all designs, feedback decisions and dev selections are frozen"
    gate = read_json(gate_path)
    assert gate.get("frozen") is True
    assert gate.get("primary_checkpoint_step") == STEPS
    for path, digest in gate["bound_files"].items():
        assert sha256(ROOT / path) == digest, f"Global freeze input changed: {path}"
    return gate


def worker(run_id, stage, device="cuda"):
    from .common import run_record, update_run
    from .environment import TaskEnv
    from .learning import Loaded

    assert stage in COUNTS
    run = run_record(run_id)
    gate = validate_test_gate() if stage == "test" else None
    checkpoint = R3 / "checkpoints" / run_id / "checkpoints" / "step_020000.pt"
    training_complete = read_json(checkpoint.parents[1] / "complete.json")
    assert training_complete.get("step", training_complete.get("actual_updates", STEPS)) == STEPS
    data_path = (R3 / "data" / run["task"] / "manifest.json" if stage == "dev" else
                 R3 / "locked_test_states" / run["task"] / "manifest.json")
    manifest = read_json(data_path)
    planned = manifest[stage]
    directory = R3 / ("dev_results" if stage == "dev" else "locked_test_results") / run_id
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(run_id=run_id, step=STEPS, stage=stage, task=run["task"],
                    candidate_id=run["candidate_id"], train_n=run["train_n"],
                    checkpoint_hash=sha256(checkpoint), data_manifest_hash=sha256(data_path),
                    evaluation_implementation_hash=sha256(__file__),
                    global_freeze_hash=sha256(R3 / "global_freeze.json") if gate else None)
    complete_path = directory / "complete.json"
    if complete_path.exists():
        old = _checked_result(complete_path)
        assert old["identity"] == identity
        return old
    for split in SPLITS:
        assert len(planned[split]) == COUNTS[stage][split]
    update_run(run_id, **{stage + "_status": "running"})
    loaded = Loaded(checkpoint, device=device)
    env = TaskEnv(run["task"])
    records = []
    started = time.monotonic()
    try:
        for split in SPLITS:
            for rec in planned[split]:
                path = directory / (rec["episode_id"] + ".json")
                if path.exists():
                    row = read_json(path)
                    assert row["identity"] == identity
                    assert row["reset_record_hash"] == object_hash(rec)
                    assert sha256(ROOT / row["trajectory_path"]) == row["trajectory_hash"]
                else:
                    row, arrays = run_episode(env, loaded, rec, device=device)
                    row["split"] = split
                    npz = path.with_suffix(".npz")
                    _save_npz(npz, arrays)
                    row.update(identity=identity, trajectory_path=str(npz.relative_to(ROOT)), trajectory_hash=sha256(npz))
                    atomic_json(path, row)
                records.append(row)
                print(stage, run_id, split, rec["episode_id"], int(row["success"]), row["steps"], row["failure_category"], flush=True)
    finally:
        env.close()
    per_split = {s: metrics([r for r in records if r["split"] == s]) for s in SPLITS}
    result = dict(identity=identity, records=records, metrics=per_split,
                  ood_success_rate=(per_split["C"]["success_rate"] + per_split["E"]["success_rate"]) / 2,
                  last_session_wall_seconds=time.monotonic() - started,
                  measured_rollout_wall_seconds=sum(r["wall_seconds"] for r in records),
                  parameter_count=training_complete.get("parameter_count", training_complete.get("parameters")),
                  uncertainty_scope="Fixed trained model; intervals reflect evaluation-state sampling only, not training seeds or dataset randomness")
    atomic_json(complete_path, result)
    update_run(run_id, **{stage + "_status": "completed", stage + "_result_path": str(complete_path.relative_to(ROOT)),
                         stage + "_result_hash": sha256(complete_path)})
    return result


def _candidate_key(candidate):
    return (-candidate["ood_success_rate"], -candidate["iid_success_rate"],
            candidate["parameter_count"], candidate["candidate_id"])


def _feedback_decision(task):
    path = R3 / "design_records" / task / "revision.json"
    if not path.exists():
        return None
    obj = read_json(path)
    choice = obj.get("decision", obj.get("choice", obj.get("candidate_id")))
    if choice is None and obj.get("no_revision"):
        choice = "no_revision"
    assert choice in ("P4", "no_revision"), f"Invalid feedback decision: {path}"
    return choice


def freeze_selection(require_all=True):
    from .common import TASKS, NS, planned_runs
    runs = list(planned_runs())
    groups, unavailable = {}, []
    for task in TASKS:
        feedback = _feedback_decision(task)
        if require_all:
            assert feedback is not None, f"Pending feedback decision: {task}"
        for n in NS:
            eligible = [r for r in runs if r["task"] == task and r["train_n"] == n and
                        (r["candidate_id"] != "P4" or (n in (5, 20) and feedback == "P4"))]
            candidates, invalid = [], []
            for run in eligible:
                rid = run["run_id"]
                if run.get("status") == "invalid":
                    reason = run.get("invalid_reason", run.get("reason"))
                    assert reason, f"Invalid candidate must retain reason: {rid}"
                    invalid.append(dict(run_id=rid, candidate_id=run["candidate_id"], reason=reason))
                    continue
                path = R3 / "dev_results" / rid / "complete.json"
                if not path.exists():
                    unavailable.append(rid)
                    continue
                result = _checked_result(path)
                parameters = result.get("parameter_count")
                if parameters is None:
                    training = read_json(R3 / "checkpoints" / rid / "complete.json")
                    parameters = training.get("parameter_count", training.get("parameters"))
                assert isinstance(parameters, (int, float)), f"Missing parameter count: {rid}"
                candidates.append(dict(run_id=rid, candidate_id=run["candidate_id"], parameter_count=int(parameters),
                                       ood_success_rate=result["ood_success_rate"], iid_success_rate=result["metrics"]["IID"]["success_rate"],
                                       dev_result_path=str(path.relative_to(ROOT)), dev_result_hash=sha256(path)))
            initial = [r for r in candidates if r["candidate_id"] in ("P1", "P2", "P3")]
            group = dict(task=task, train_n=n, candidates=candidates, invalid=invalid,
                         initial_selected=min(initial, key=_candidate_key) if initial else None,
                         final_system=min(candidates, key=_candidate_key) if candidates else None)
            group["b0_fallback"] = bool(group["final_system"] and group["final_system"]["candidate_id"] == "B0")
            groups[f"{task}:n{n}"] = group
    if require_all:
        assert not unavailable, f"Pending development results: {unavailable}"
        assert len([r for r in runs if r["candidate_id"] != "P4"]) == 96
    result = dict(frozen=require_all, primary_checkpoint_step=STEPS, groups=groups, unavailable=unavailable,
                  rule="Maximum mean C/E success; ties: higher IID, fewer trainable parameters, lexicographic candidate_id",
                  initial_selected_scope="P1/P2/P3 only; selected programmatically using this N's development states",
                  final_system_scope="B0/P1/P2/P3 plus P4 only at N5/N20 where actually trained")
    path = R3 / ("selection.json" if require_all else "selection_preview.json")
    if path.exists() and require_all:
        assert read_json(path) == result, "Frozen development selection changed"
    else:
        atomic_json(path, result)
    return result


def freeze_test_gate():
    from .common import TASKS, planned_runs
    selection = freeze_selection(require_all=True)
    paths = [R3 / "selection.json", Path(__file__)]
    for task in TASKS:
        proposal = R3 / "design_records" / task / "proposal.json"
        revision = R3 / "design_records" / task / "revision.json"
        assert proposal.exists() and revision.exists()
        paths.extend([proposal, revision, R3 / "data" / task / "manifest.json"])
        # Bind concrete test state files by hash without reading their contents.
        locked = R3 / "locked_test_states" / task
        assert (locked / "manifest.json").exists()
        paths.extend(sorted(p for p in locked.rglob("*") if p.is_file()))
    runs = list(planned_runs())
    assert 96 <= len(runs) <= 108
    for run in runs:
        if run.get("status") == "invalid":
            continue
        rid = run["run_id"]
        paths.extend([R3 / "dev_results" / rid / "complete.json",
                      R3 / "checkpoints" / rid / "checkpoints" / "step_020000.pt",
                      R3 / "configs" / run["task"] / (run["candidate_id"] + ".json")])
    gate = dict(frozen=True, primary_checkpoint_step=STEPS, expected_main_runs=96, planned_runs=len(runs),
                bound_files={str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(paths))},
                selection_groups=len(selection["groups"]),
                declaration="All initial designs, all six feedback decisions, candidate implementations, development results and per-N choices frozen before test outcomes")
    path = R3 / "global_freeze.json"
    if path.exists():
        assert read_json(path) == gate
    else:
        atomic_json(path, gate)
    return gate


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("stage", choices=COUNTS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    worker(args.run_id, args.stage, device=args.device)
