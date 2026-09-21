"""Synthetic interface fixtures only: no experimental designs or API calls."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from appl.io import atomic
from real_robot import data, tools
from real_robot.data import Sources
from real_robot.tools import CutTools, HEURISTIC
from real_robot.transport import Client, usage_cost


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    # All mock outputs are labelled TEST FIXTURE and never sent to the provider.
    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(tools, "ROOT", tmp_path)
    root = tmp_path / "original"
    ep = root / "episode_test"
    ep.mkdir(parents=True)
    n = 8
    t = np.arange(n) / 30.0 + 100.0
    recv = t - 3.0
    frames = [
        dict(
            index=i,
            robot_time=float(t[i]),
            recv_time=float(recv[i]),
            frames={
                c: dict(frame_number=i, host_ts=float(recv[i]), age_s=0.0)
                for c in ("third", "wrist")
            },
        )
        for i in range(n)
    ]
    pose = np.tile(np.eye(4), (n, 1, 1))
    pose[:, 0, 3] = np.arange(n) * 0.001
    arrays = dict(
        sample_index=np.arange(n),
        t=t,
        recv_time=recv,
        q=np.zeros((n, 7)),
        dq=np.zeros((n, 7)),
        tau_ext=np.zeros((n, 7)),
        T_base_flange=pose,
        T_base_ee=pose,
        gripper_position=np.full(n, -1.0),
        gripper_width_m=np.full(n, np.nan),
        action_json=np.array(
            [
                json.dumps(dict(type="TEST FIXTURE", v=[float(i), 0, 0], dq=None))
                for i in range(n)
            ]
        ),
    )
    np.savez_compressed(ep / "obs.npz", **arrays)
    atomic(ep / "frames.json", dict(samples=frames))
    atomic(ep / "meta.json", dict(duration_s=7 / 30.0, complete=True))
    atomic(ep / "camera_intrinsics.json", {})
    for camera in ("third", "wrist"):
        for depth in (False, True):
            folder = ep / (camera + ("_depth" if depth else ""))
            folder.mkdir()
            for i in range(n):
                Image.new("RGB", (1280, 720), (i, 0, 0)).save(
                    folder / (f"{i:06d}" + (".png" if depth else ".jpg"))
                )
    cfg = dict(
        source_root=str(root),
        trajectory_ids=["episode_test"],
        run="run",
        output="cut",
        preview_width=640,
        preview_height=360,
        max_api_calls=8,
        max_tool_calls=30,
        max_plan_versions=3,
        max_output_tokens=1000,
        model="gpt-6-astra",
        reasoning_effort="xhigh",
    )
    src = Sources(cfg)
    src.inventory(tmp_path / "run")
    processor = CutTools(cfg)
    processor.j.set("inspected_steps", {"episode_test": list(range(n))})
    processor.j.set(
        "inspected_images", {"episode_test": {"third": list(range(n)), "wrist": [0]}}
    )
    return cfg, src, processor


def test_sample_pairing_rejects_different_robot_frame_mapping(fixture):
    cfg, src, _ = fixture
    path = Path(cfg["source_root"]) / "episode_test/frames.json"
    value = json.loads(path.read_text())
    value["samples"][3]["robot_time"] += 1
    atomic(path, value)
    with pytest.raises(ValueError, match="mapping mismatch"):
        Sources(cfg)


def test_preview_coordinates_and_original_evidence(fixture):
    _, src, _ = fixture
    _, m = src.image("episode_test", "third", 2, "preview", None)
    assert m["returned_size"] == [640, 360]
    assert m["coordinate_mapping"]["original_x_per_returned_pixel"] == 2
    _, m = src.image(
        "episode_test",
        "third",
        2,
        "original",
        dict(left=100, top=50, right=500, bottom=250),
    )
    assert m["returned_size"] == [400, 200]
    assert m["coordinate_mapping"]["original_x_offset"] == 100
    with pytest.raises(ValueError, match="Crop"):
        src.image(
            "episode_test",
            "third",
            2,
            "preview",
            dict(left=0, top=0, right=2000, bottom=10),
        )


def test_published_slice_preserves_nan_commands_and_pairing(fixture, tmp_path):
    _, src, processor = fixture
    s = dict(
        trajectory_id="episode_test",
        start=2,
        stop=5,
        rationale="TEST FIXTURE",
        training_condition="TEST FIXTURE",
    )
    result = src.publish_segment(s, tmp_path / "slice", processor.source_manifest)
    with np.load(tmp_path / "slice/records.npz", allow_pickle=False) as z:
        assert z["sample_index"].tolist() == [2, 3, 4]
        assert np.isnan(z["gripper_width_m"]).all()
        assert json.loads(str(z["action_json"][0]))["dq"] is None
    index = json.loads((tmp_path / "slice/samples.json").read_text())
    assert index["successor_source_index"] == 5
    assert [s["source_sample_index"] for s in index["samples"]] == [2, 3, 4]
    assert index["samples"][0]["media"]["wrist"]["source_relative_path"].endswith(
        "/wrist/000002.jpg"
    )
    assert result["records"] == 3


def make_plan():
    h = {k: "TEST FIXTURE" for k in HEURISTIC["properties"]}
    h["heuristic_id"] = "prior_a"
    h["policy_id"] = "policy_a"
    h["policy_contract"] = {
        k: "TEST FIXTURE"
        for k in HEURISTIC["properties"]["policy_contract"]["properties"]
    }
    h["evidence"] = [dict(trajectory_id="episode_test", indices=[1])]
    h["handoff"] = {
        k: "TEST FIXTURE" for k in HEURISTIC["properties"]["handoff"]["properties"]
    }

    def segment(lo, hi):
        return dict(
            trajectory_id="episode_test",
            segment_id=f"segment_{lo}_{hi}",
            start=lo,
            stop=hi,
            supervised_start=lo,
            supervised_stop=hi,
            rationale="TEST FIXTURE",
            objective="TEST FIXTURE",
            training_condition="TEST FIXTURE",
            label_derivation="TEST FIXTURE",
            deployment_condition_source="TEST FIXTURE",
            condition_evidence_indices=[lo],
            boundary_evidence_indices=[lo, hi - 1],
            merge_check="TEST FIXTURE",
            split_check="TEST FIXTURE",
            uncertainty="TEST FIXTURE",
            supervision_exclusions=[],
        )

    h2 = deepcopy(h)
    h2.update(heuristic_id="prior_b", policy_id="policy_b")
    h3 = deepcopy(h)
    h3.update(heuristic_id="prior_c", policy_id="policy_c")
    return dict(
        task_interpretation="TEST FIXTURE",
        goal_conditioning="TEST FIXTURE",
        portfolio_rationale="TEST FIXTURE",
        model_inventory="TEST FIXTURE",
        generalization_design="TEST FIXTURE",
        calling_contract="TEST FIXTURE",
        capability_coverage="TEST FIXTURE",
        sharing_and_diversity_audit="TEST FIXTURE",
        skills=[
            dict(
                skill_id="fixture_a",
                name="TEST FIXTURE",
                subgoal="TEST FIXTURE",
                segmentation_rationale="TEST FIXTURE",
                segments=[segment(0, 3), segment(5, 8)],
                heuristics=[h],
            ),
            dict(
                skill_id="fixture_b",
                name="TEST FIXTURE",
                subgoal="TEST FIXTURE",
                segmentation_rationale="TEST FIXTURE",
                segments=[segment(1, 6)],
                heuristics=[h2, h3],
            ),
        ],
        exclusions=[],
        overlap_rationale="TEST FIXTURE",
        motion_coverage="TEST FIXTURE",
        unobserved_cases="TEST FIXTURE",
    )


def test_free_grouping_and_variable_heuristic_count(fixture):
    _, _, p = fixture
    plan = make_plan()
    v = p.validate(plan)
    assert v["datasets"] == 2 and v["heuristics"] == 3
    assert v["coverage"]["episode_test"] == dict(
        source_records=8,
        assigned_unique=8,
        excluded=0,
        reused_unique=3,
        assignment_records=11,
        supervised_unique=8,
        context_only_unique=0,
        supervision_assignment_records=11,
    )
    p.j.set("inspected_steps", {"episode_test": [0, 1, 2, 5, 6]})
    with pytest.raises(ValueError, match="first/last"):
        p.validate(plan)


def test_uncovered_samples_and_uninspected_evidence_fail(fixture):
    _, _, p = fixture
    plan = make_plan()
    plan["skills"] = plan["skills"][:1]
    with pytest.raises(ValueError, match="Unassigned"):
        p.validate(plan)
    plan["exclusions"] = [
        dict(trajectory_id="episode_test", start=3, stop=5, reason="TEST FIXTURE")
    ]
    assert p.validate(plan)["coverage"]["episode_test"]["excluded"] == 2
    plan["skills"][0]["heuristics"][0]["evidence"][0]["indices"] = [4]
    with pytest.raises(ValueError, match="must be in this dataset"):
        p.validate(plan)


def test_source_mutation_detected(fixture):
    cfg, src, p = fixture
    path = Path(cfg["source_root"]) / "episode_test/meta.json"
    atomic(path, dict(changed=True))
    with pytest.raises(ValueError, match="changed"):
        src.verify_metadata(p.source_manifest)


def test_cost_includes_reasoning_once_and_handles_cache():
    u = dict(
        input_tokens=100000,
        output_tokens=20000,
        input_tokens_details=dict(cached_tokens=20000, cache_write_tokens=0),
        output_tokens_details=dict(reasoning_tokens=15000),
    )
    assert usage_cost(u) == (1.82, False)
    u["input_tokens_details"].pop("cache_write_tokens")
    assert usage_cost(u) == (2.02, True)


def test_transport_rejects_wrong_effort_without_network(fixture):
    _, _, processor = fixture
    client = Client.__new__(Client)
    client.j = processor.j
    with pytest.raises(ValueError, match="model/effort"):
        client.respond(dict(model="gpt-6-astra", reasoning=dict(effort="high")))


def test_complete_publication_keeps_disjoint_segments_independent(fixture):
    _, _, p = fixture
    plan = make_plan()
    version = p.dispatch("write_plan", dict(plan=plan))["plan_hash"]
    p.dispatch("check_plan", {})
    result = p.dispatch("submit_datasets", dict(expected_hash=version))
    assert result["submitted"] and not result["policy_code_written"]
    manifest = json.loads((p.output / "manifest.json").read_text())
    assert manifest["checks"]["heuristics"] == 3
    dataset = json.loads((p.output / "datasets/fixture_a/dataset.json").read_text())
    assert len(dataset["segments"]) == 2
    assert [(s["start"], s["stop"]) for s in dataset["segments"]] == [(0, 3), (5, 8)]
    assert dataset["heuristics"] == plan["skills"][0]["heuristics"]
    assert (
        json.loads(
            (p.output / "datasets/fixture_a/segments/0000/supervision.json").read_text()
        )
        == plan["skills"][0]["segments"][0]
    )
    catalog = json.loads((p.output / "policy_catalog.json").read_text())
    assert [(i["policy_id"], i["dataset_group_id"]) for i in catalog] == [
        ("policy_a", "fixture_a"),
        ("policy_b", "fixture_b"),
        ("policy_c", "fixture_b"),
    ]
    assert not list(p.output.rglob("*.py"))
    with pytest.raises(ValueError, match="immutable"):
        p.dispatch("write_plan", dict(plan=plan))


def test_context_and_supervision_exclusions_have_distinct_coverage(fixture):
    _, _, p = fixture
    plan = make_plan()
    segment = plan["skills"][0]["segments"][0]
    segment.update(
        supervised_start=1,
        boundary_evidence_indices=[1, 2],
        supervision_exclusions=[dict(start=1, stop=2, reason="TEST FIXTURE")],
    )
    coverage = p.validate(plan)["coverage"]["episode_test"]
    assert coverage["assigned_unique"] == 8
    assert coverage["supervised_unique"] == 7
    assert coverage["context_only_unique"] == 1
    assert coverage["supervision_assignment_records"] == 9
    segment["supervised_stop"] = 4
    with pytest.raises(ValueError, match="inside its materialized"):
        p.validate(plan)


def test_semantic_evidence_must_be_inspected_and_in_scope(fixture):
    _, _, p = fixture
    plan = make_plan()
    segment = plan["skills"][0]["segments"][0]
    segment["condition_evidence_indices"] = [7]
    with pytest.raises(ValueError, match="Condition evidence"):
        p.validate(plan)
    segment["condition_evidence_indices"] = [1]
    p.j.set("inspected_images", {"episode_test": {"third": [0], "wrist": [0]}})
    with pytest.raises(ValueError, match="Read_images"):
        p.validate(plan)


def test_independent_alternative_policies_need_distinct_ids(fixture):
    _, _, p = fixture
    plan = make_plan()
    assert p.validate(plan)["proposed_policies"] == 3
    plan["skills"][1]["heuristics"][1]["policy_id"] = "policy_b"
    with pytest.raises(ValueError, match="unique ID"):
        p.validate(plan)
