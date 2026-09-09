"""Synthetic scheduler and rollout checks; no sealed evaluation data are read."""
from contextlib import contextmanager
from threading import Barrier

import numpy as np
import pytest

from round3 import cli, evaluate
from relative_dp.utils import atomic_json


@contextmanager
def unlocked(*args, **kwargs):
    yield


class SyntheticEnv:
    def __init__(self, success_at=None, terminate_at=None, invalid_at=None):
        self.success_at, self.terminate_at, self.invalid_at = success_at, terminate_at, invalid_at

    def reset(self, record):
        self.steps = 0
        return np.zeros(3), {"success": False, "valid_joint": True}

    def step(self, action):
        assert np.max(np.abs(action)) <= 1
        self.steps += 1
        return np.full(3, self.steps), 0., self.steps == self.terminate_at, False, {
            "success": self.steps == self.success_at,
            "valid_joint": self.steps != self.invalid_at,
        }


class SyntheticPolicy:
    def __init__(self, execute=4):
        self.histories, self.execute = [], execute

    def actions(self, history, generator):
        self.histories.append(history.copy())
        return np.full((16, 4), 2.), {"execution_horizon": self.execute}


def rollout(env, policy=None):
    return evaluate.run_episode(env, policy or SyntheticPolicy(), {
        "episode_id": "synthetic", "split": "C", "initial_state_hash": "synthetic",
        "inference_seed": 17,
    }, device="cpu", verify_snapshot=False)


def test_success_stops_inside_chunk_and_world_clips():
    result, arrays = rollout(SyntheticEnv(success_at=3))
    assert result["success"] and result["steps"] == 3 and result["replans"] == 1
    assert result["clipped_action_steps"] == 3
    np.testing.assert_array_equal(arrays["actions"], np.ones((3, 4)))


def test_invalid_joint_cannot_be_redeemed_by_later_success():
    result, _ = rollout(SyntheticEnv(success_at=7, invalid_at=2))
    assert not result["success"] and result["steps"] == 7
    assert result["first_success_step"] is None and result["first_threshold_step"] == 7
    assert result["failure_category"] == "joint_violation"


def test_termination_history_and_500_step_budget():
    policy = SyntheticPolicy(execute=7)
    result, _ = rollout(SyntheticEnv(), policy)
    assert result["steps"] == 500 and not result["success"]
    np.testing.assert_array_equal(policy.histories[0], np.zeros((2, 3)))
    np.testing.assert_array_equal(policy.histories[1], [[6, 6, 6], [7, 7, 7]])
    result, _ = rollout(SyntheticEnv(terminate_at=2))
    assert result["steps"] == 2 and result["termination_reason"] == "native_terminated"


def test_test_gate_rejects_before_state_access(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "R3", tmp_path)
    with pytest.raises(AssertionError, match="Locked test is blocked"):
        evaluate.validate_test_gate()


def test_partial_data_never_enters_training_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "R3", tmp_path)
    implementation = tmp_path / "implementation.json"
    implementation.write_text("{}")
    monkeypatch.setattr(cli, "implementation_path", lambda *args: implementation)
    monkeypatch.setattr(cli, "implementation_ready", lambda *args: True)
    atomic_json(tmp_path / "design_records" / "task" / "initial_freeze.json", {})
    record = dict(task="task", candidate_id="P1", status="pending", train_status="pending")
    path = tmp_path / "data" / "task" / "manifest.json"
    atomic_json(path, dict(complete=False, train20=list(range(20))))
    assert not cli._eligible(record, "train")
    atomic_json(path, dict(complete=True, train20=list(range(20))))
    assert cli._eligible(record, "train")


def test_stale_recovery_preserves_live_process(tmp_path, monkeypatch):
    records = [dict(run_id="live", train_status="running", pid=123),
               dict(run_id="dead", train_status="completed", dev_status="running", pid=456)]
    updates = []
    monkeypatch.setattr(cli, "planned_runs", lambda: records)
    monkeypatch.setattr(cli, "_pid_live", lambda pid: pid == 123)
    monkeypatch.setattr(cli, "lock", unlocked)
    monkeypatch.setattr(cli, "update_run", lambda rid, **fields: updates.append((rid, fields)))
    monkeypatch.setattr(cli, "event", lambda *a, **k: None)
    assert cli.reconcile_stale_runs() == ["dead"]
    assert updates[0][1]["dev_status"] == "pending"
    assert "train_status" not in updates[0][1]


def test_four_authorized_devices_are_scheduled_concurrently(monkeypatch):
    devices = [0, 1, 2, 3]
    barrier = Barrier(4, timeout=5)
    def queue(gpu, stages):
        barrier.wait()
        return gpu
    monkeypatch.setattr(cli, "GPUS", devices)
    monkeypatch.setattr(cli, "lock", unlocked)
    monkeypatch.setattr(cli, "reconcile_stale_runs", lambda: [])
    monkeypatch.setattr(cli, "gpu_queue", queue)
    assert cli.schedule(["train"]) == devices


def test_individual_worker_overrides_pixi_gpu_activation(monkeypatch):
    monkeypatch.setattr(cli, "GPUS", [0, 1, 2, 3])
    monkeypatch.setattr(cli, "lock", unlocked)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.delenv("ROUND3_GPU_LEASE_PID", raising=False)
    with cli.gpu_context(3):
        assert cli.os.environ["CUDA_VISIBLE_DEVICES"] == "3"
        assert cli.os.environ["MUJOCO_EGL_DEVICE_ID"] == "3"


def test_candidate_requires_successful_mathematical_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    implementation=tmp_path/"implementation.json"
    monkeypatch.setattr(cli,"implementation_path",lambda *args: implementation)
    atomic_json(implementation,{"implementation_audit":"audit.json"})
    rec={"task":"task","candidate_id":"P1"}
    assert not cli.implementation_ready(rec)
    atomic_json(tmp_path/"audit.json",{"passed":False})
    assert not cli.implementation_ready(rec)
    atomic_json(tmp_path/"audit.json",{"status":"passed"})
    assert cli.implementation_ready(rec)
