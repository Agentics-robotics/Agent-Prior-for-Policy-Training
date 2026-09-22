"""Regression checks for API ownership, pipeline state and isolated execution."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import shutil
import os

import numpy as np
import pytest
from PIL import Image

from appl.io import ROOT, atomic, digest, object_hash, read
from real_robot.data import Sources
from real_robot.training_pipeline import contract, engine, public, runner, workflow


@pytest.fixture
def fixture(tmp_path):
    cfg = contract.configuration(ROOT / "real_robot/configs/push_training_v5.json")
    cfg.update(run=str(tmp_path / "run"), config_path=str(tmp_path / "config.json"), test_cpu=True,
               cut_config=str(tmp_path / "cut_config.json"), cut_run=str(tmp_path / "cut"),
               cut_output=str(tmp_path / "cut_output"), max_job_seconds=60)
    if os.environ.get("ROBOT_PIPELINE_TEST_GPU") == "4":
        cfg["test_cpu"] = False
        cfg["max_job_seconds"] = 180
    episode = tmp_path / "data/fixture_episode"
    episode.mkdir(parents=True)
    records = dict(sample_index=np.arange(6), t=np.arange(6, dtype=float), recv_time=np.arange(6, dtype=float))
    np.savez(episode / "obs.npz", **records)
    atomic(episode / "frames.json", dict(samples=[dict(index=i, robot_time=float(i), recv_time=float(i)) for i in range(6)]))
    atomic(episode / "meta.json", dict(duration_s=5, complete=True))
    atomic(episode / "camera_intrinsics.json", {})
    for camera in ("third", "wrist", "third_depth", "wrist_depth"):
        (episode / camera).mkdir()
        for i in range(6):
            suffix = ".png" if "depth" in camera else ".jpg"
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(episode / camera / (f"{i:06d}" + suffix))
    cut_cfg = dict(source_root=str(episode.parent), trajectory_ids=[episode.name], run=cfg["cut_run"], output=cfg["cut_output"])
    atomic(cfg["cut_config"], cut_cfg)
    source_plan = dict(skills=[dict(skill_id="old", component_kind="learned_policy", segments=[
        dict(segment_id="old_sparse", trajectory_id=episode.name, start=0, stop=6,
             supervision_kind="sparse", decision_indices=[1])])])
    atomic(Path(cfg["cut_output"]) / "plan.json", source_plan)
    sources = Sources(cut_cfg)
    sources.inventory(Path(cfg["cut_run"]))
    source = Path(cfg["run"]) / "package_00/source"
    source.mkdir(parents=True)
    shutil.copyfile(Path(__file__).with_name("fixture_policy.py"), source / "policy.py")
    params = dict(description="Synthetic fixture", model_family="Linear test", architecture_rationale="Infrastructure only",
        config=dict(lr=0.05), updates=5, batch_size=3, seed=7, ema_decay=None, max_grad_norm=None,
        prediction_dimension=2, decision_schema=dict(type="object", required=["delta"],
        properties=dict(delta=dict(type="array", minItems=2, maxItems=2, items=dict(type="number"))), additionalProperties=False))
    package = dict(schema="real_robot.policy_package.pipeline.v1", policies={"fixture_a":params, "fixture_b":deepcopy(params)},
        preprocessing={}, data_plan=dict(rationale="Synthetic complete interval", changes_from_prior="Add source anchors and goal variants",
        segments=[dict(segment_id="derived", trajectory_id=episode.name, start=0, stop=6, rationale="Fixture interval")]),
        dependencies="Torch fixture", adaptations="No scientific candidate", limitations="Synthetic infrastructure only")
    atomic(source / "package.json", package)
    atomic(source / "HANDOFF.json", dict(policies={pid:{key:"Fixture" for key in contract.HANDOFF_FIELDS} for pid in package["policies"]}))
    for name in ["PRIOR", "CALLING", "POLICY_CATALOG", "fixture_a_PRIOR", "fixture_a_USAGE", "fixture_b_PRIOR", "fixture_b_USAGE"]:
        (source / (name + ".md")).write_text("Developer-owned synthetic infrastructure fixture. Never a Runtime API-authored scientific candidate. " * 3)
    (Path(cfg["run"]) / "assets").mkdir()
    atomic(cfg["config_path"], cfg)
    checked = contract.validate_package(source, cfg)
    return cfg, sources, source, package, checked


def prepared(fixture):
    cfg, sources, source, package, checked = fixture
    _, _, spec, segments = engine.specification(cfg, source)
    public.configure(sources, segments, source.parent / "prepared", Path(cfg["run"]) / "assets", source)
    value = engine.module_at(source).prepare_data(spec)
    return value, cfg, spec, segments, sources


def test_new_anchors_variants_partial_targets_and_two_policies(fixture):
    result = engine.validate_prepared(*prepared(fixture))
    assert result["num_examples"] == 3
    assert result["source_coverage"]["fixture_episode"]["unique_supervised_anchors"] == 2
    assert result["policy_examples"] == dict(fixture_a=3, fixture_b=3)
    assert read(Path(fixture[0]["cut_output"]) / "plan.json")["skills"][0]["segments"][0]["decision_indices"] == [1]


@pytest.mark.parametrize("kind", ["future", "cross_boundary", "duplicate", "unaccounted", "nonfinite"])
def test_real_provenance_constraints_remain(fixture, kind):
    value, cfg, spec, segments, sources = prepared(fixture)
    if kind == "future": value["arrays"]["observation_stop"][0] = 4
    if kind == "cross_boundary": value["arrays"]["target_stop"][0] = 7
    if kind == "duplicate": value["arrays"]["example_variant"][1] = 0
    if kind == "unaccounted": value["metadata"]["source_accounting"][0]["stop"] = 5
    if kind == "nonfinite": value["arrays"]["target"][0, 0] = np.nan
    with pytest.raises(ValueError):
        engine.validate_prepared(value, cfg, spec, segments, sources)


def test_config_rejects_developer_owned_policy_inventory(fixture):
    cfg = fixture[0]
    cfg["policies"] = ["forced_policy"]
    atomic(cfg["config_path"], cfg)
    with pytest.raises(ValueError, match="API-owned"):
        contract.configuration(cfg["config_path"])


def test_actual_isolated_prepare_train_and_online_call(fixture, monkeypatch):
    cfg, sources, source, package, checked = fixture
    folder = source.parent
    prepared_result = runner.finish_job(runner.start_job(cfg, folder, "prepare", folder / "prepared", 4))
    assert prepared_result["returncode"] == 0, prepared_result
    manifest = read(folder / "prepared/result.json")
    assert manifest["source_coverage"]["fixture_episode"]["unique_supervised_anchors"] == 2
    submission = dict(**checked, assets={}, preparation_hash=digest(folder / "prepared/result.json"),
        source_plan_hash=object_hash(read(Path(cfg["cut_output"])/"plan.json")), readiness=dict(decision="train"))
    atomic(folder / "submission.json", submission)
    # This is developer fixture provenance, so it deliberately does not publish
    # a scientific library or pretend that a provider authored these files.
    monkeypatch.setattr(runner, "freeze_executor", lambda *a: None)
    monkeypatch.setattr(runner, "publish", lambda *a: dict(fixture_complete=True))
    assert runner.execute(cfg, 0)["fixture_complete"]
    for policy_id in package["policies"]:
        result = read(folder / "training" / policy_id / "result.json")
        assert result["optimizer_class"] == "SGD"
        assert result["optimizer_steps"] == 2 < package["policies"][policy_id]["updates"]
        assert result["sample_exposures"] == 6
        assert result["calling_check"]["passed"] and result["reload_max_prediction_error"] == 0
        enforcement = read(folder / "training" / policy_id / "enforcement.json")
        assert enforcement["seccomp"] and enforcement["landlock_abi"] > 0
    from real_robot.training_pipeline.inference import PolicyProcess
    worker = PolicyProcess(cfg["config_path"], "fixture_a", folder / "online", gpu=4)
    try:
        result = worker.act(dict(x=np.array([1.], dtype=np.float32)), {}, reset=True, executor_contract={})
        assert result["status"] == "ready" and len(result["decision"]["delta"]) == 2
        assert result["executor_request"]["hardware_io"] is False
    finally:
        worker.close()


@pytest.mark.parametrize("case", ["repair", "transport_stop", "API_blocked"])
def test_fixed_workflow_delegates_repair_but_never_retries_transport(tmp_path, monkeypatch, case):
    cfg = dict(run=str(tmp_path / "run"), cut_config=str(tmp_path / "cut.json"),
               cut_run=str(tmp_path / "cut"), max_implementation_revisions=3, execution_guard_usd=10)
    atomic(cfg["cut_config"], {})
    atomic(Path(cfg["cut_run"]) / "source_manifest.json", {})
    monkeypatch.setattr(workflow, "preflight", lambda c: dict(passed=True))
    monkeypatch.setattr(workflow, "Sources", lambda c: SimpleNamespace(verify_media=lambda m: None))
    calls, executions = [], []

    def design(c, revision, feedback, previous):
        calls.append((revision, feedback, previous))
        if case == "transport_stop": raise TimeoutError("Uncertain provider request")
        (Path(c["run"]) / f"package_{revision:02d}/source").mkdir(parents=True)
        return dict(readiness=dict(decision="blocked" if case == "API_blocked" else "train"))

    def execute(c, revision):
        executions.append(revision)
        if revision == 0: raise runner.ImplementationFailure(dict(error="Synthetic interface mismatch"))
        return dict(policies=["chosen_by_API"])

    monkeypatch.setattr(workflow, "design", design)
    monkeypatch.setattr(runner, "execute", execute)
    if case == "transport_stop":
        with pytest.raises(TimeoutError): workflow.run(cfg)
        assert len(calls) == 1 and not executions
    else:
        result = workflow.run(cfg)
        if case == "API_blocked":
            assert result["status"] == "API_declared_blocked" and not executions
        else:
            assert result["status"] == "complete" and executions == [0, 1]
            assert calls[1][1]["diagnostics"] == dict(error="Synthetic interface mismatch")
            assert calls[1][2].name == "source"
    with pytest.raises((ValueError, FileExistsError)):
        workflow.run(cfg)
