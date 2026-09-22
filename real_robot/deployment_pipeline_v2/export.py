"""Lossless selected-checkpoint export. Frozen scientific source stays unchanged."""
from pathlib import Path
import shutil

import torch

from appl.io import ROOT, atomic, read, digest
from .common import locations

EXPORTS = {"push_v6": "push_training_v6"}


def export(bundle, config_name):
    cfg = read(ROOT / f"real_robot/configs/{config_name}.json")
    root = ROOT / cfg["run"]
    state = read(root / "workflow.json")
    assert state["status"] == "complete"
    original = root / f"package_{state['revision']:02d}"
    submission = read(original / "submission.json")
    package, weights, assets = locations(bundle)
    if package.exists() or weights.exists():
        raise ValueError("Export must use new package/weight directories")
    (package / "source").mkdir(parents=True)
    weights.mkdir(parents=True)
    assets.mkdir()
    for name, sha in submission["files"].items():
        assert digest(original / "source" / name) == sha
        shutil.copyfile(original / "source" / name, package / "source" / name)
    dependencies = [ROOT / "real_robot/training_pipeline" / name for name in ("__init__.py", "public.py", "security.py")]
    dependencies += [ROOT / "src/appl" / name for name in ("__init__.py", "io.py", "gpu.py", "kernel.py")]
    manifest = dict(schema="real_robot.pipeline_deployment.v2", bundle=bundle,
        source_package_hash=submission["package_hash"], cut_plan_hash=submission["source_plan_hash"],
        source_hashes=submission["files"], scientific_source="Exact API submission bytes",
        runtime_hashes={str(p.relative_to(ROOT)):digest(p) for p in dependencies},
        policies={}, assets={}, robot_execution=False)
    for pid in read(original / "source/package.json")["policies"]:
        folder = original / "training" / pid
        result = read(folder / "result.json")
        assert digest(folder / "last.pt") == result["checkpoint_sha256"]
        saved = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
        # Same online spec filtering as the original frozen inference worker.
        spec = {k:v for k,v in saved["spec"].items() if k not in
                {"source_plan", "trajectory_ids", "recording_metadata", "data_plan", "data_metadata"}}
        payload = dict(schema="real_robot.pipeline_inference_checkpoint.v2", policy_id=pid,
            state_dict=saved["selected"], spec=spec, source_hashes=submission["files"])
        file = weights / f"{pid}.pt"
        torch.save(payload, file)
        reloaded = torch.load(file, map_location="cpu", weights_only=True)
        assert reloaded["state_dict"].keys() == saved["selected"].keys()
        assert all(torch.equal(v,reloaded["state_dict"][k]) for k,v in saved["selected"].items())
        manifest["policies"][pid] = dict(file=file.name, sha256=digest(file), bytes=file.stat().st_size,
            original_checkpoint_sha256=result["checkpoint_sha256"], selected_step=result["selected_step"],
            optimizer_steps=result["optimizer_steps"], parameters=result["trainable_parameters"],
            exact_selected_tensor_equality=True)
    for name, sha in submission["assets"].items():
        original_asset = root / "assets" / name
        assert digest(original_asset) == sha
        if name.endswith(".receipt.json"):
            (package / "asset_receipts").mkdir(exist_ok=True)
            shutil.copyfile(original_asset,package / "asset_receipts" / name)
            continue
        shutil.copyfile(original_asset, assets / name)
        manifest["assets"][name] = dict(sha256=sha, bytes=original_asset.stat().st_size,
            required_for_load=True, required_for="API build_model initializes frozen SAM; required even for geometry-only act")
    shutil.copyfile(root / "completion_receipt.json",package / "TRAINING_RECEIPT.json")
    shutil.copyfile(root / "API_USAGE.json", package / "API_USAGE.json")
    host_check = read(root / "host_worker_check/result.json")
    assert host_check["passed"] and host_check["checkpoint_sha256"] in {
        p["original_checkpoint_sha256"] for p in manifest["policies"].values()}
    example = package / "examples/interface.json"
    atomic(example, dict(
        scope="Offline invocation example: recorded actor contour in a synthetic single-object scene, with a hindsight goal and synthetic commissioning; never a live robot command",
        fixture=host_check["fixture"], expected_status=host_check["cases"][1]["status"],
        robot_execution=False, original_worker_result=host_check["cases"][1]))
    manifest["examples"] = {"interface": dict(file="examples/interface.json", sha256=digest(example))}
    atomic(package / "manifest.json", manifest)
    return dict(bundle=bundle, policies=manifest["policies"], assets=manifest["assets"])


if __name__ == "__main__":
    for name, cfg in EXPORTS.items():
        print(export(name,cfg), flush=True)
