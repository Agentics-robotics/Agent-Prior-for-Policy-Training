"""Synthetic-only renderer guards: no real locked artifacts, GPU or rollouts."""
import contextlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from relative_dp import utils
from round3 import cli, common, evaluate, report


@pytest.fixture
def driver(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/round3_render_videos.py"
    spec = importlib.util.spec_from_file_location("round3_render_videos", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "R3", tmp_path / "round3")
    monkeypatch.setattr(module, "SCRIPT", tmp_path / "scripts/round3_render_videos.py")
    return module


@pytest.fixture
def accepted(driver, monkeypatch):
    """Tiny completion files plus virtual hashed bytes; all paths are synthetic."""
    runs = [dict(run_id=f"synthetic_{i}", status="completed") for i in range(98)]
    for run in runs:
        path = driver.R3 / "locked_test_results" / run["run_id"] / "complete.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")
    selection = dict(frozen=True, groups={f"{t}:n{n}": {} for t in common.TASKS for n in common.NS})
    gate = dict(planned_runs=98, selection_groups=24, bound_files={"round3/selection.json": "selection"})
    bound = {"round3/global_freeze.json": "gate", "round3/selection.json": "selection"}
    for run in runs:
        prefix = f"round3/locked_test_results/{run['run_id']}/"
        bound[prefix + "complete.json"] = "complete"
        for i in range(100):
            bound[prefix + f"episode_{i}.json"] = "row"
            bound[prefix + f"episode_{i}.npz"] = "trajectory"
    audit = dict(stage="test", pending_counts=dict(training=0, dev=0, test=0),
                 counts=dict(test_runs_checked=98, test_episodes_checked=9800,
                             frozen_selection_groups_checked=24, feedback_decisions_checked=6),
                 result_totals={"test": {r["run_id"]: {} for r in runs}},
                 auditor_sha256="auditor", run_manifest_snapshot_sha256="ledger", artifact_hashes=bound)
    audit.update({field: True for field in ("passed", "audit_scope_complete", "numerical_delivery_complete",
                 "main_matrix_training_complete", "main_matrix_dev_complete", "main_matrix_test_complete",
                 "global_freeze_validated", "locked_test_contents_accessed")})
    audit_path = driver.R3 / "audits/accepted.json"
    objects = {audit_path: audit, driver.R3 / "run_manifest.json": dict(runs=runs),
               driver.R3 / "selection.json": selection}
    hashes = {driver.ROOT / path: digest for path, digest in bound.items()}
    hashes.update({driver.R3 / "run_manifest.json": "ledger", audit_path: "audit",
                   driver.ROOT / "scripts/verify_round3_delivery.py": "auditor",
                   driver.SCRIPT: "driver", driver.ROOT / "src/round3/report.py": "renderer"})
    trace = []

    def digest(path):
        trace.append(("hash", Path(path)))
        return hashes[Path(path)]

    monkeypatch.setattr(utils, "read_json", lambda path: objects[Path(path)])
    monkeypatch.setattr(utils, "sha256", digest)
    monkeypatch.setattr(evaluate, "validate_test_gate", lambda: trace.append(("gate", None)) or gate)
    return SimpleNamespace(audit=audit, path=audit_path, hashes=hashes, trace=trace, runs=runs)


def test_missing_completion_refuses_before_gate_or_outcomes(driver, accepted):
    (driver.R3 / "locked_test_results/synthetic_97/complete.json").unlink()
    with pytest.raises(ValueError, match="waits for all 98"):
        driver.accepted_inputs(accepted.path)
    assert not accepted.trace


@pytest.mark.parametrize("mutation", [
    lambda audit: audit.update(stage="dev"),
    lambda audit: audit.update(passed=False),
    lambda audit: audit.update(numerical_delivery_complete=False),
    lambda audit: audit["counts"].update(test_runs_checked=97),
    lambda audit: audit["pending_counts"].update(test=1),
])
def test_incomplete_acceptance_cannot_reach_gate(driver, accepted, mutation):
    mutation(accepted.audit)
    with pytest.raises(ValueError):
        driver.accepted_inputs(accepted.path)
    assert not accepted.trace


def test_changed_gate_refuses_before_outcome_reads(driver, accepted):
    accepted.hashes[driver.R3 / "global_freeze.json"] = "changed"
    with pytest.raises(ValueError, match="Global freeze differs"):
        driver.accepted_inputs(accepted.path)
    assert not any(kind == "gate" or "locked_test_results" in str(path) for kind, path in accepted.trace)


def test_invalid_gate_refuses_before_outcome_reads(driver, accepted, monkeypatch):
    def invalid():
        raise AssertionError("synthetic full gate invalid")
    monkeypatch.setattr(evaluate, "validate_test_gate", invalid)
    with pytest.raises(AssertionError, match="full gate invalid"):
        driver.accepted_inputs(accepted.path)
    assert not any("locked_test_results" in str(path) for _, path in accepted.trace)


def test_changed_test_trajectory_refuses(driver, accepted):
    accepted.hashes[driver.R3 / "locked_test_results/synthetic_97/episode_99.npz"] = "changed"
    with pytest.raises(ValueError, match="Accepted test artifact changed"):
        driver.accepted_inputs(accepted.path)
    assert ("gate", None) in accepted.trace


def test_missing_audited_evidence_refuses(driver, accepted):
    del accepted.audit["artifact_hashes"]["round3/locked_test_results/synthetic_0/episode_0.npz"]
    with pytest.raises(ValueError, match="omits complete test evidence"):
        driver.accepted_inputs(accepted.path)


def test_accepted_bytes_and_source_fingerprint(driver, accepted):
    selection, inputs = driver.accepted_inputs(accepted.path)
    assert len(selection["groups"]) == 24
    assert inputs["accepted_test_runs"] == 98
    assert inputs["driver_sha256"] == "driver" and inputs["renderer_sha256"] == "renderer"
    gate_index = accepted.trace.index(("gate", None))
    assert all(index > gate_index for index, (_, path) in enumerate(accepted.trace)
               if "locked_test_results" in str(path))


def test_device_override_and_command_are_after_pixi_activation(driver, monkeypatch):
    monkeypatch.setattr(cli, "pixi_binary", lambda: "/synthetic/pixi")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "5")
    monkeypatch.setenv("ROUND3_GPU_LEASE_PID", "123")
    for name in driver.THREADS:
        monkeypatch.setenv(name, "8")
    driver.configure_device(3)
    assert driver.os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    assert driver.os.environ["MUJOCO_EGL_DEVICE_ID"] == "3"
    assert "ROUND3_GPU_LEASE_PID" not in driver.os.environ
    assert all(driver.os.environ[name] == "1" for name in driver.THREADS)
    command = driver.child_command(3, Path("audit"), Path("expected"), Path("output"))
    assert command[:5] == ["/synthetic/pixi", "run", "env", "CUDA_VISIBLE_DEVICES=3", "MUJOCO_EGL_DEVICE_ID=3"]
    assert all(f"{name}=1" in command[3:command.index("python")] for name in driver.THREADS)
    for gpu in (4, 5, -1):
        with pytest.raises(ValueError, match="Unauthorized"):
            driver.configure_device(gpu)
        with pytest.raises(ValueError, match="Unauthorized"):
            driver.child_command(gpu, Path("a"), Path("e"), Path("o"))


def test_gpu_slot_contention_releases_canonical_lease(driver, monkeypatch):
    trace = []

    @contextlib.contextmanager
    def locked(name, blocking):
        assert blocking is False
        trace.append(("acquire", name))
        if name.endswith("slot-1"):
            raise BlockingIOError("synthetic slot busy")
        try:
            yield
        finally:
            trace.append(("release", name))

    monkeypatch.setattr(common, "lock", locked)
    with pytest.raises(BlockingIOError):
        with driver.device_leases(2):
            pytest.fail("Busy card must never reach rendering")
    assert trace == [("acquire", "gpu-2"), ("acquire", "gpu-2-slot-1"), ("release", "gpu-2")]


def test_loads_only_current_tasks_frozen_comparison_runs(driver, monkeypatch):
    task, read_ids = common.TASKS[2], []
    selection = {"groups": {f"{task}:n{n}": {"initial_selected": {
        "run_id": common.run_id(task, "P2", n)}} for n in common.NS}}

    def checked(path):
        rid = path.parent.name
        read_ids.append(rid)
        return {"identity": {"stage": "test", "run_id": rid}}

    monkeypatch.setattr(report, "_checked_result", checked)
    result = driver.needed_results(task, selection)
    expected = [common.run_id(task, candidate, n) for n in (20, 5, 2, 10) for candidate in ("B0", "P2")]
    assert read_ids == expected and set(result) == set(expected)


def test_worker_covers_only_assigned_tasks_under_both_leases(driver, monkeypatch):
    held, calls = set(), []

    @contextlib.contextmanager
    def locked(name, blocking):
        held.add(name)
        try:
            yield
        finally:
            held.remove(name)

    def videos(results, selection, *, tasks, write_manifest):
        assert held == {"gpu-0", "gpu-0-slot-1"}
        assert not write_manifest
        assert results == {tasks[0]: "only needed results"}
        calls.extend(tasks)
        return [{"path": tasks[0] + ".mp4", "identity": {"renderer_sha256": "synthetic"}}]

    expected = {"renderer_sha256": "synthetic"}
    monkeypatch.setattr(common, "lock", locked)
    monkeypatch.setattr(driver, "accepted_inputs", lambda _: ({}, expected))
    monkeypatch.setattr(driver, "input_fingerprint", lambda _: expected)
    monkeypatch.setattr(driver, "needed_results", lambda task, _: {task: "only needed results"})
    monkeypatch.setattr(report, "test_videos", videos)
    output = driver.R3 / "receipt.json"
    driver.worker(0, Path("unused"), expected, output)
    assert calls == driver.assignments()[0] and not held
    assert json.loads(output.read_text())["policy_calls"] == 0
    assert sorted(t for tasks in driver.assignments().values() for t in tasks) == sorted(common.TASKS)


def test_mid_render_source_change_refuses_receipt(driver, monkeypatch):
    @contextlib.contextmanager
    def unlocked(*args, **kwargs):
        yield

    expected = {"renderer_sha256": "before"}
    monkeypatch.setattr(common, "lock", unlocked)
    monkeypatch.setattr(driver, "accepted_inputs", lambda _: ({}, expected))
    monkeypatch.setattr(driver, "input_fingerprint", lambda _: {"renderer_sha256": "after"})
    monkeypatch.setattr(driver, "needed_results", lambda *args: {})
    monkeypatch.setattr(report, "test_videos", lambda *args, **kwargs: [{"path": "synthetic.mp4"}])
    output = driver.R3 / "receipt.json"
    with pytest.raises(ValueError, match="changed during rendering"):
        driver.worker(3, Path("unused"), expected, output)
    assert not output.exists()


def test_child_failure_cleans_only_owned_groups_and_no_success(driver, monkeypatch):
    stopped, children = [], []

    @contextlib.contextmanager
    def unlocked(*args, **kwargs):
        yield

    def start(*args, **kwargs):
        assert kwargs["start_new_session"] is True
        process = SimpleNamespace(pid=1000 + len(children), poll=lambda: None)
        children.append(process)
        return process

    def failed(_):
        raise RuntimeError("synthetic child failed")

    monkeypatch.setattr(common, "lock", unlocked)
    monkeypatch.setattr(driver, "accepted_inputs", lambda _: ({}, {"renderer_sha256": "synthetic"}))
    monkeypatch.setattr(driver, "child_command", lambda *args: ["synthetic-no-execution"])
    monkeypatch.setattr(driver.subprocess, "Popen", start)
    monkeypatch.setattr(driver, "wait_children", failed)
    monkeypatch.setattr(cli, "_process_group_live", lambda _: True)
    monkeypatch.setattr(cli, "_interrupt_and_wait", lambda process: stopped.append(process.pid))
    with pytest.raises(RuntimeError, match="synthetic child failed"):
        driver.launch(Path("unused"))
    assert stopped == [1000, 1001, 1002, 1003]
    assert not list(driver.R3.rglob("complete.json"))


def test_wait_children_propagates_nonzero_exit(driver, monkeypatch):
    monkeypatch.setattr(cli, "_process_group_live", lambda _: False)
    child = SimpleNamespace(pid=123, poll=lambda: 7)
    with pytest.raises(RuntimeError, match="GPU 2 video child exited 7"):
        driver.wait_children([(2, child)])


def test_worker_failure_releases_both_leases_without_receipt(driver, monkeypatch):
    held = set()

    @contextlib.contextmanager
    def locked(name, blocking):
        held.add(name)
        try:
            yield
        finally:
            held.remove(name)

    def changed_inputs(_):
        assert held == {"gpu-3", "gpu-3-slot-1"}
        raise ValueError("synthetic artifact changed")

    monkeypatch.setattr(common, "lock", locked)
    monkeypatch.setattr(driver, "accepted_inputs", changed_inputs)
    output = driver.R3 / "receipt.json"
    with pytest.raises(ValueError, match="artifact changed"):
        driver.worker(3, Path("unused"), {}, output)
    assert not held and not output.exists()


@pytest.mark.parametrize("failure_n", [20, 5, 2, 10, None])
def test_task_filter_preserves_failure_priority(tmp_path, monkeypatch, failure_n):
    task = common.TASKS[0]
    monkeypatch.setattr(report, "R3", tmp_path)
    monkeypatch.setattr(report, "validate_test_gate", lambda: None)
    states = {split: [dict(episode_id=split + "0"), dict(episode_id=split + "1")]
              for split in ("C", "E", "IID")}
    monkeypatch.setattr(report, "read_json", lambda _: {"test": states})
    results, groups, calls = {}, {}, []
    for n in (20, 5, 2, 10):
        selected = common.run_id(task, "P1", n)
        groups[f"{task}:n{n}"] = {"initial_selected": {"run_id": selected}}
        for candidate in ("B0", "P1"):
            results[common.run_id(task, candidate, n)] = {"records": [dict(episode_id=r["episode_id"],
                success=not (failure_n is not None and (20, 5, 2, 10).index(n) >=
                             (20, 5, 2, 10).index(failure_n) and r["episode_id"] == "E0"))
                for split in ("C", "E", "IID") for r in states[split]]}

    def paired(task, reset, left, right, output):
        calls.append((reset["episode_id"], output.name))
        return {"path": str(output)}

    monkeypatch.setattr(report, "_paired_video", paired)
    records = report.test_videos(results, {"groups": groups}, tasks=[task], write_manifest=False)
    assert calls[0] == ("C0", f"{task}_fixed_C_N20.mp4")
    if failure_n is None:
        assert records[-1]["status"] == "no_additional_failure" and len(calls) == 1
    else:
        assert calls[1] == ("E0", f"{task}_first_failure_N{failure_n}.mp4")
    assert not (tmp_path / "videos/manifest.json").exists()
    with pytest.raises(AssertionError, match="Partial task workers"):
        report.test_videos(results, {"groups": groups}, tasks=[task])


def test_synthetic_writer_threads_and_cache_binding(tmp_path, monkeypatch):
    import imageio.v2 as imageio
    from round3 import environment
    monkeypatch.setattr(report, "ROOT", tmp_path)
    trajectory = tmp_path / "trajectory.npz"
    np.savez(trajectory, obs=np.zeros((1, 3)), actions=np.zeros((0, 4)))
    row = dict(trajectory_path="trajectory.npz", trajectory_hash=utils.sha256(trajectory),
               episode_id="synthetic", identity=dict(candidate_id="B0", train_n=20),
               split="C", success=True, termination_reason="success")
    reset = dict(episode_id="synthetic", initial_state_hash="synthetic-state")
    environments, writes = [], []

    class Env:
        native = SimpleNamespace(dt=.01)

        def __init__(self, *args, **kwargs):
            self.closed = False
            environments.append(self)

        def reset(self, reset):
            return np.zeros(3), {}

        def close(self):
            self.closed = True

    def writer(path, **kwargs):
        writes.append(kwargs)
        path.write_bytes(b"synthetic video fixture, no simulator")
        return SimpleNamespace(append_data=lambda _: None, close=lambda: None)

    monkeypatch.setattr(environment, "TaskEnv", Env)
    monkeypatch.setattr(report, "_verify_snapshot", lambda *args: None)
    monkeypatch.setattr(report, "_annotated_frame", lambda *args, **kwargs: np.zeros((2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr(imageio, "get_writer", writer)
    output = tmp_path / "video.mp4"
    first = report._paired_video("synthetic", reset, row, row, output)
    second = report._paired_video("synthetic", reset, row, row, output)
    assert first == second and len(environments) == 2 and all(env.closed for env in environments)
    assert writes[0]["ffmpeg_params"] == ["-threads", "1"]
    assert writes[0]["codec"] == "libx264" and writes[0]["fps"] == 25
    assert first["identity"]["renderer_sha256"] == utils.sha256(report.__file__)
    output.write_bytes(b"changed synthetic cache")
    with pytest.raises(AssertionError):
        report._paired_video("synthetic", reset, row, row, output)


@pytest.mark.parametrize("failure", ["second_environment", "writer_init", "writer_close", "replay_and_close"])
def test_renderer_closes_all_created_resources_on_failure(tmp_path, monkeypatch, failure):
    import imageio.v2 as imageio
    from round3 import environment
    monkeypatch.setattr(report, "ROOT", tmp_path)
    trajectory = tmp_path / "trajectory.npz"
    np.savez(trajectory, obs=np.zeros((1, 3)), actions=np.zeros((0, 4)))
    row = dict(trajectory_path="trajectory.npz", trajectory_hash=utils.sha256(trajectory),
               episode_id="synthetic", identity=dict(candidate_id="B0", train_n=20),
               split="C", success=True, termination_reason="success")
    reset = dict(episode_id="synthetic", initial_state_hash="synthetic-state")
    created, closed = [], []

    class Env:
        native = SimpleNamespace(dt=.01)

        def __init__(self, *args, **kwargs):
            if failure == "second_environment" and created:
                raise RuntimeError("synthetic second environment")
            created.append(self)

        def reset(self, reset):
            return (np.ones(3) if failure == "replay_and_close" else np.zeros(3)), {}

        def close(self):
            closed.append(self)

    def writer(path, **kwargs):
        if failure == "writer_init":
            raise RuntimeError("synthetic writer init")
        path.write_bytes(b"synthetic fixture")

        def close():
            raise RuntimeError("synthetic writer close")

        return SimpleNamespace(append_data=lambda _: None, close=close)

    monkeypatch.setattr(environment, "TaskEnv", Env)
    monkeypatch.setattr(report, "_verify_snapshot", lambda *args: None)
    monkeypatch.setattr(report, "_annotated_frame", lambda *args, **kwargs: np.zeros((2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr(imageio, "get_writer", writer)
    output = tmp_path / "video.mp4"
    with pytest.raises(AssertionError if failure == "replay_and_close" else RuntimeError,
                       match="Video replay differs" if failure == "replay_and_close" else "synthetic"):
        report._paired_video("synthetic", reset, row, row, output)
    assert created and set(created) == set(closed)
    assert not output.exists() and not output.with_suffix(".json").exists()
