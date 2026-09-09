"""Tampering and pre-freeze access tests; never use real locked-test artifacts."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def verifier(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/verify_round3_delivery.py"
    spec = importlib.util.spec_from_file_location("round3_delivery_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "R3", tmp_path / "round3")
    return module


def test_locked_files_rejected_without_opening(verifier, tmp_path):
    audit = verifier.Audit("dev")
    # These deliberately do not exist: permission must fail before file access.
    for name in ("locked_test_states/task/manifest.json", "locked_test_results/run/complete.json"):
        with pytest.raises(ValueError, match="Locked-test access rejected"):
            audit.read(tmp_path / "round3" / name)
        with pytest.raises(ValueError, match="Locked-test access rejected"):
            audit.digest(tmp_path / "round3" / name)
    assert not audit.artifact_hashes


def test_incomplete_gate_cannot_open_sealed_files(verifier, tmp_path, monkeypatch):
    audit = verifier.Audit("test")
    audit.runs = []
    monkeypatch.setattr(audit, "inspect_feedback", lambda *args, **kwargs: None)
    monkeypatch.setattr(audit, "inspect_selection", lambda: None)
    root = tmp_path / "round3"
    root.mkdir()
    (root / "global_freeze.json").write_text(json.dumps(dict(frozen=True, primary_checkpoint_step=20000,
        expected_main_runs=96, planned_runs=0, selection_groups=24, bound_files={})))
    for task in verifier.TASKS:
        folder = root / "design_records" / task / "feedback"
        folder.mkdir(parents=True)
        (folder / "bundle.json").write_text('{"bound_files": {}}')
    with pytest.raises(ValueError, match="omits required"):
        audit.test_gate()
    assert not audit.locked_hash_allowed and not audit.locked_content_allowed


def test_cached_hash_still_rejects_changed_file(verifier, tmp_path):
    audit = verifier.Audit("dev")
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"original")
    digest = audit.digest(path)
    assert audit.digest(path, digest) == digest
    assert audit.counts["physical_files_hashed"] == 1
    path.write_bytes(b"tampered and resized")
    with pytest.raises(ValueError, match="Previously checked artifact changed"):
        audit.digest(path)


def trajectory(verifier, tmp_path):
    audit = verifier.Audit("dev")
    obs = np.zeros((3, 4), np.float32)
    path = tmp_path / "trajectory.npz"
    np.savez(path, obs=obs, actions=np.zeros((2, 4), np.float32),
             success=np.array([False, False, True]), valid_joint=np.ones(3, bool),
             terminated=np.zeros(2, bool), truncated=np.zeros(2, bool), clipping=np.zeros((2, 4), bool),
             replan_steps=np.array([0]), inference_seconds=np.array([0.1]))
    audit.snapshots["episode"] = obs[0]
    row = dict(trajectory_path=str(path), trajectory_hash=audit.digest(path), episode_id="episode",
        identity={"task": "pick-place-wall", "stage": "dev"}, steps=2, success=True,
        first_success_step=2, first_threshold_step=2, invalid_joint=False, terminated=False,
        truncated=False, exception=None, termination_reason="success", failure_category="success",
        infos=[{"success": False}, {"success": False}, {"success": True}],
        clipped_action_steps=0, clipped_coordinates=[0, 0, 0, 0], replans=1,
        inference=[dict(at_step=0, seconds=0.1, execute_steps=4)], inference_seconds=0.1, wall_seconds=0.2)
    return audit, row, {"episode_id": "episode"}


def test_success_flag_cannot_disagree_with_trajectory(verifier, tmp_path):
    audit, row, reset = trajectory(verifier, tmp_path)
    audit.inspect_trajectory(row, reset)
    row["success"] = False
    with pytest.raises(ValueError, match="Trajectory/record success mismatch"):
        audit.inspect_trajectory(row, reset)


def test_inference_total_cannot_be_fabricated(verifier, tmp_path):
    audit, row, reset = trajectory(verifier, tmp_path)
    row["inference_seconds"] = 9.0
    with pytest.raises(ValueError, match="Inference total differs"):
        audit.inspect_trajectory(row, reset)


def test_passing_subset_never_claims_main_matrix_or_delivery(verifier):
    audit = verifier.Audit("dev")
    audit.runs = [dict(run_id=verifier.run_id(t, c, n), task=t, candidate_id=c)
                  for t in verifier.TASKS for n in verifier.NS for c in verifier.CANDIDATES]
    audit.training = {audit.runs[0]["run_id"]: {}}
    audit.manifest_snapshot_hash = "snapshot"
    result = audit.finish()
    assert result["passed"]
    assert not result["numerical_delivery_complete"]
    assert not result["main_matrix_training_complete"]
    assert not result["main_matrix_dev_complete"]
    assert result["counts"]["training_runs_checked"] == 1


@pytest.mark.parametrize("stage, expected_pending_test", [("dev", None), ("test", 0)])
def test_failed_audit_preserves_test_scope(verifier, tmp_path, monkeypatch, capsys,
                                         stage, expected_pending_test):
    """Failure before any artifact access must preserve the requested scope."""
    def fail_before_access(self):
        raise ValueError("incomplete public evidence")

    output = tmp_path / "round3/audits/delivery_verifier_scope.json"
    monkeypatch.setattr(verifier.Audit, "run", fail_before_access)
    monkeypatch.setattr(verifier.torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(verifier.torch, "set_num_interop_threads", lambda count: None)
    monkeypatch.setattr(verifier.sys, "argv", ["verify_round3_delivery.py", "--stage", stage,
                                            "--output", str(output)])
    assert verifier.main() == 1
    result = json.loads(output.read_text())
    concise = json.loads(capsys.readouterr().out)
    for report in (result, concise):
        assert report["pending_counts"]["test"] == expected_pending_test
        assert report["locked_test_contents_accessed"] is False
        assert report["numerical_delivery_complete"] is False


@pytest.mark.parametrize("fields, accepted", [
    ({"feedback_cycles_used": 1}, True),
    ({"feedback_cycles_used": 1, "feedback_cycles_remaining": 0}, True),
    ({"feedback_cycles_used": 2}, False),
    ({"feedback_cycles_used": 1, "feedback_cycles_remaining": 1}, False),
    ({"feedback_cycles_remaining": 1}, False),
])
def test_feedback_budget_optional_remaining_field(verifier, tmp_path, monkeypatch, fields, accepted):
    audit = verifier.Audit("dev")
    audit.runs = []
    path = tmp_path / "round3/design_records/synthetic/revision.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(decision="no_revision", **fields)))

    def read_until_packet(candidate):
        if candidate == path:
            return json.loads(path.read_text())
        raise RuntimeError("reached packet validation")

    monkeypatch.setattr(audit, "read", read_until_packet)
    with pytest.raises(RuntimeError if accepted else ValueError,
                       match="reached packet validation" if accepted else "Feedback cycle budget exceeded"):
        audit.inspect_feedback("synthetic")
