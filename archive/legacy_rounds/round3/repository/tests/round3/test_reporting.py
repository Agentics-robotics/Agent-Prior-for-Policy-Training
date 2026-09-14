"""Reporting checks on synthetic artifacts only, with no locked-state access."""
import numpy as np
import pytest

from relative_dp.utils import atomic_json, read_json, sha256
from round3 import report


def _result(success):
    return dict(records=[dict(episode_id=f"{split}-{i}", split=split,
                             initial_state_hash=f"state-{split}-{i}", success=value)
                         for split in ("C", "E") for i, value in enumerate(success)])


def test_pairing_bootstrap_includes_negative_effects_and_exact_ties():
    baseline = _result([True, True, False, False])
    same = report.paired_ood_delta(baseline, baseline)
    assert same["delta_pp"] == 0 and same["paired_bootstrap_95ci_pp"] == [0, 0]
    worse = report.paired_ood_delta(_result([False] * 4), baseline)
    assert worse["delta_pp"] == -50
    assert worse["paired_bootstrap_95ci_pp"][1] <= 0
    mismatched = _result([False] * 4)
    mismatched["records"][0]["initial_state_hash"] = "other-state"
    with pytest.raises(AssertionError):
        report.paired_ood_delta(mismatched, baseline)


def test_empty_report_cannot_claim_formal_completion_or_open_test(tmp_path, monkeypatch):
    root = tmp_path / "round3"
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "R3", root)
    monkeypatch.setattr(report, "TASKS", ["synthetic"])
    monkeypatch.setattr(report, "planned_runs", lambda: [dict(run_id="synthetic", task="synthetic", train_n=5,
                        candidate_id="B0", status="pending", feedback_revision=False)])
    monkeypatch.setattr(report, "freeze_selection", lambda **kw: dict(groups={}))
    monkeypatch.setattr(report, "validate_test_gate", lambda: pytest.fail("Dev report accessed test gate"))
    atomic_json(root / "locked_test_states" / "synthetic" / "manifest.json", {"must_not_read": True})
    result = report.generate(make_videos=False)
    assert result["trained"] == 0 and result["evaluated"] == 0 and result["final"] is False
    assert "incomplete" in (root / "ROUND3_REPORT.md").read_text()
    assert read_json(root / "reports" / "integrity.json")["final"] is False


def test_feedback_selects_first_disagreement_and_caps_images(tmp_path, monkeypatch):
    from round3 import diagnostics
    root, task = tmp_path / "round3", "synthetic"
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "R3", root)
    monkeypatch.setattr(diagnostics,"R3",root)
    monkeypatch.setattr(diagnostics,"ROOT",tmp_path)
    monkeypatch.setattr(diagnostics,"summarize_result",lambda *args:dict(splits={},records=[]))
    candidates = ["B0", "P1", "P2", "P3"]
    runs = [dict(run_id=c, task=task, candidate_id=c, train_n=5, status="pending") for c in candidates]
    monkeypatch.setattr(report, "planned_runs", lambda: runs)
    atomic_json(root / "design_records" / task / "proposal.json", {"synthetic": True})
    states = [dict(episode_id=f"state{i}") for i in range(3)]
    atomic_json(root / "data" / task / "manifest.json", dict(complete=True, dev={"C": states}))
    fixtures = {}
    for candidate in candidates:
        records = []
        for i in range(3):
            trajectory = root / "synthetic_trajectories" / f"{candidate}_{i}.npz"
            trajectory.parent.mkdir(parents=True, exist_ok=True)
            # Renderer below is explicitly a stub; this is not simulator evidence.
            trajectory.write_bytes(b"synthetic-test-placeholder")
            records.append(dict(episode_id=f"state{i}", split="C", success=candidate == "P1" and i > 0,
                                steps=3, failure_category="synthetic", termination_reason="synthetic", replans=1,
                                trajectory_path=str(trajectory.relative_to(tmp_path)), trajectory_hash=sha256(trajectory)))
        result = dict(identity={"run_id": candidate}, records=records, metrics={}, ood_success_rate=0., parameter_count=1)
        path = root / "dev_results" / candidate / "complete.json"
        atomic_json(path, result)
        fixtures[str(path)] = result
    monkeypatch.setattr(report, "_checked_result", lambda path: fixtures[str(path)])
    monkeypatch.setattr(report, "_replay_frames", lambda task, reset, row, steps, label:
                        {step: np.zeros((8, 8, 3), np.uint8) for step in steps})
    packet = report.feedback_bundle(task)
    assert packet["selected_episode_id"] == "state1"
    assert packet["frame_count"] == 8 and packet["contains_locked_test"] is False
    assert all(frame["actual_step"] == 3 for frame in packet["frames"])
    assert report.feedback_bundle(task) == packet
    first_image = tmp_path / packet["frames"][0]["path"]
    first_image.write_bytes(b"changed")
    with pytest.raises(AssertionError, match="Frozen feedback changed"):
        report.feedback_bundle(task)


def test_p4_costs_and_effects_keep_base_distinct_from_development_selection(tmp_path, monkeypatch):
    root, task = tmp_path / "round3", "synthetic"
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "R3", root)
    monkeypatch.setattr(report, "TASKS", [task])
    atomic_json(root / "design_records" / task / "revision.json",
                dict(decision="P4", candidate=dict(base_candidate_id="P1", principal_revision_intent="Synthetic representation change")))
    atomic_json(root / "design_records" / task / "feedback" / "bundle.json",
                dict(scored_development_rollouts=200, frame_count=8))
    runs = [dict(run_id=report.run_id(task, "P4", n), task=task, candidate_id="P4", train_n=n) for n in (5, 20)]
    debug_path = root / "debug" / runs[0]["run_id"] / "sessions.jsonl"
    debug_path.parent.mkdir(parents=True)
    debug_path.write_text('{"start_step": 0, "end_step": 20, "wall_seconds": 5.4}\n')
    # One completed formal model and one pending model; debugging is extra work.
    trained = {runs[0]["run_id"]: dict(total_session_wall_seconds=100, gpu_active_work_seconds=80)}
    results, groups = {}, {}
    for n in (5, 20):
        for candidate, successes in (("P1", [True, False, False, False]), ("P3", [True, True, True, False])):
            result = _result(successes)
            result.update(identity=dict(candidate_id=candidate), ood_success_rate=sum(successes)/4)
            results[report.run_id(task, candidate, n)] = result
        groups[f"{task}:n{n}"] = dict(initial_selected=dict(run_id=report.run_id(task, "P3", n)))
    revised = _result([True, True, False, False])
    revised.update(identity=dict(candidate_id="P4"), ood_success_rate=.5)
    results[runs[0]["run_id"]] = revised
    accounting = report.feedback_accounting(runs, trained, results)
    item = accounting[0]
    assert item["p4_base_candidate_id"] == "P1"
    assert item["p4_planned_trainings"] == 2 and item["p4_completed_trainings"] == 1
    assert item["p4_train_wall_seconds"] == 100 and item["p4_gpu_active_work_seconds"] == 80
    assert item["p4_debug_session_updates"] == 20 and item["p4_debug_session_wall_seconds"] == 5.4
    assert item["feedback_development_episodes"] == 200 and item["feedback_packet_frames"] == 8
    effects = report.feedback_effects(accounting, results, dict(groups=groups))
    by_key = {(e["train_n"], e["reference_kind"]): e for e in effects}
    assert by_key[5, "base-candidate"]["reference_candidate_id"] == "P1"
    assert by_key[5, "base-candidate"]["paired_ood_delta"]["delta_pp"] == 25
    assert by_key[5, "initial-selected"]["reference_candidate_id"] == "P3"
    assert by_key[5, "initial-selected"]["paired_ood_delta"]["delta_pp"] == -25
    assert {e["train_n"] for e in effects} == {5, 20}
    assert all(e["p4_ood_success_rate"] is None and e["paired_ood_delta"] is None for e in effects if e["train_n"] == 20)
    assert report.feedback_effects([dict(task=task, decision="no_revision")], results, dict(groups=groups)) == []
