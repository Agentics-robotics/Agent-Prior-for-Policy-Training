"""Developer fixtures for the generic continuation controller; no API calls."""
from pathlib import Path

import pytest

from appl.io import atomic, read
from real_robot.execution import continue_training_pipeline as recovery
from real_robot.execution import train_policies as driver


@pytest.mark.parametrize("outcome", ["train", "blocked", "transport_error"])
def test_recovery_preserves_assessment_and_delegates_design(tmp_path, monkeypatch, outcome):
    cfg = dict(run=str(tmp_path), max_implementation_revisions=3, execution_guard_usd=20)
    initial = dict(status="API_declared_blocked", revision=0, readiness=dict(decision="blocked", reason="fixture labels need investigation"))
    atomic(tmp_path / "workflow.json", initial)
    seen = []

    def design(c, revision, feedback, previous):
        seen.append((revision, feedback, previous))
        assert c["execution_guard_usd"] == 20
        if outcome == "transport_error": raise TimeoutError("unknown request")
        return dict(readiness=dict(decision=outcome))

    monkeypatch.setattr(recovery, "design", design)
    monkeypatch.setattr(recovery, "execute", lambda c,r: dict(fixture=True))
    if outcome == "transport_error":
        with pytest.raises(TimeoutError): recovery.run(cfg)
        assert read(tmp_path/"workflow.json")["status"] == "stopped"
    else:
        result = recovery.run(cfg)
        assert result["status"] == ("complete" if outcome == "train" else "API_declared_blocked")
    assert len(seen) == 1 and seen[0][0] == 1
    assert seen[0][1]["previous_assessment"] == initial["readiness"]
    assert seen[0][2] == tmp_path/"package_00/source"
    assert read(tmp_path/"assessment_before_continuation_00.json") == initial


def test_generic_driver_bounds_recovery_and_never_overwrites(tmp_path, monkeypatch):
    cfg = dict(run=str(tmp_path), max_implementation_revisions=3)
    monkeypatch.setattr(driver, "initial_run", lambda c: dict(status="API_declared_blocked", revision=0))
    calls = []

    def next_round(c):
        calls.append(len(calls)+1)
        return dict(status="API_declared_blocked", revision=calls[-1])

    monkeypatch.setattr(driver, "recovery_run", next_round)
    assert driver.run(cfg)["revision"] == 2
    assert calls == [1,2]
    with pytest.raises(ValueError, match="Existing"):
        driver.run(cfg)
