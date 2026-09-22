"""Synthetic responsibility/sparse supervision checks, never API design input."""

from copy import deepcopy
import json

import pytest

from appl.io import object_hash
from real_robot.cutting_v6 import tools
from real_robot.tests.test_cut_contract import fixture, make_plan


@pytest.fixture
def processor(fixture, tmp_path, monkeypatch):
    cfg, _, old = fixture
    old.j.db.close()
    monkeypatch.setattr(tools, "ROOT", tmp_path)
    p = tools.CutTools(cfg)
    yield p
    p.j.db.close()


def plan_v6():
    plan = make_plan()
    for g in plan["skills"]:
        g.update(component_kind="learned_policy", execution_contract="TEST FIXTURE")
        for s in g["segments"]:
            s.update(supervision_kind="dense", decision_indices=[])
    return plan


def executor_group(plan):
    g = plan["skills"][1]
    g.update(component_kind="supplied_executor", heuristics=[])
    for s in g["segments"]:
        s.update(supervision_kind="executor_only", supervised_start=None,
                 supervised_stop=None, decision_indices=[])
    return g


def test_complete_coverage_with_one_policy_and_executor(processor):
    plan = plan_v6()
    executor_group(plan)
    checks = processor.validate(plan)
    assert checks["proposed_policies"] == 1
    assert checks["learned_groups"] == checks["supplied_executor_groups"] == 1
    c = checks["coverage"]["episode_test"]
    assert c["assigned_unique"] == 8
    assert c["supervised_unique"] == 6
    assert c["executor_unique"] == 5


def test_sparse_anchors_and_raw_publication_are_distinct(processor):
    plan = plan_v6()
    executor_group(plan)
    for s in plan["skills"][0]["segments"]:
        s.update(supervision_kind="sparse", decision_indices=[s["start"]])
    checks = processor.validate(plan)
    assert checks["coverage"]["episode_test"]["supervised_unique"] == 2
    manifest = processor.publish(object_hash(plan), plan, checks)
    g = json.loads((processor.output / manifest["datasets"][0]["dataset"]).read_text())
    assert g["component_kind"] == "learned_policy"
    assert g["segments"][0]["records"] == 3
    assert g["segments"][0]["decision_indices"] == [0]
    policies = json.loads((processor.output / "policy_catalog.json").read_text())
    execution = json.loads((processor.output / "execution_catalog.json").read_text())
    assert len(policies) == 1 and len(execution) == 2
    assert execution[1]["policy_ids"] == []


@pytest.mark.parametrize("indices", [[], [0, 0], [3]])
def test_bad_sparse_indices_rejected(processor, indices):
    plan = plan_v6()
    plan["skills"][0]["segments"][0].update(supervision_kind="sparse", decision_indices=indices)
    with pytest.raises(ValueError, match="Sparse decision indices"):
        processor.validate(plan)


def test_sparse_exclusion_cannot_silently_drop_decision(processor):
    plan = plan_v6()
    s = plan["skills"][0]["segments"][0]
    s.update(supervision_kind="sparse", decision_indices=[1],
             supervision_exclusions=[dict(start=1, stop=2, reason="TEST FIXTURE")])
    with pytest.raises(ValueError, match="cannot be excluded"):
        processor.validate(plan)


def test_executor_must_have_no_supervised_interval(processor):
    plan = plan_v6()
    g = executor_group(plan)
    g["segments"][0]["supervised_start"] = 1
    with pytest.raises(ValueError, match="null supervision bounds"):
        processor.validate(plan)


def test_group_role_requires_compatible_policy_and_segment(processor):
    plan = plan_v6()
    plan["skills"][0]["heuristics"] = []
    with pytest.raises(ValueError, match="Learned groups need"):
        processor.validate(plan)
    plan = plan_v6()
    executor_group(plan)["segments"][0]["supervision_kind"] = "dense"
    with pytest.raises(ValueError, match="match group"):
        processor.validate(plan)


def test_sparse_state_and_executor_visual_evidence_required(processor):
    plan = plan_v6()
    executor_group(plan)
    plan["skills"][0]["segments"][0].update(supervision_kind="sparse", decision_indices=[1])
    processor.j.set("inspected_steps", {"episode_test": [0, 2, 3, 4, 5, 6, 7]})
    with pytest.raises(ValueError, match="inspect sparse decision"):
        processor.validate(plan)
    processor.j.set("inspected_steps", {"episode_test": list(range(8))})
    processor.j.set("inspected_images", {"episode_test": {"third": [0, 2, 3, 4, 5, 6, 7], "wrist": [0]}})
    with pytest.raises(ValueError, match="Read_images"):
        processor.validate(plan)
