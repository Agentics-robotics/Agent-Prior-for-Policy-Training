"""Lossless export of frozen API source and final EMA tensors, with provenance."""

import argparse
import json
from pathlib import Path
import shutil

import torch

from .common import ROOT, PACKAGE, WEIGHTS, read, digest


def export(training_package, package=PACKAGE, weights=WEIGHTS):
    training_package, package, weights = map(Path, (training_package, package, weights))
    submission = read(training_package / "submission.json")
    source = package / "source"
    source.mkdir(parents=True, exist_ok=True)
    weights.mkdir(parents=True, exist_ok=True)
    if {p.name for p in source.iterdir()} - set(submission["files"]):
        raise ValueError("Unexpected files in deployment source directory")
    for name, sha in submission["files"].items():
        original = training_package / "source" / name
        if digest(original) != sha:
            raise ValueError("Frozen API source changed: " + name)
        target = source / name
        if target.exists():
            if digest(target) != sha:
                raise ValueError("Existing deployment source differs: " + name)
        else:
            shutil.copyfile(original, target)
    manifest = dict(
        schema="real_robot.push_deployment.v1",
        source_package_hash=submission["package_hash"],
        cut_plan_hash=submission["source_plan_hash"],
        source_hashes=submission["files"],
        scientific_source="Exact bytes of API design_00 submission; no developer edits",
        selection="Final EMA at 20000 updates, as in original inference loader",
        policies={},
    )
    metadata_keys = (
        "state_mean",
        "state_std",
        "state_dim",
        "action_mean",
        "action_std",
    )
    for name in ("contour_push", "visual_push"):
        folder = training_package / "training" / name
        result = read(folder / "result.json")
        original = folder / "last.pt"
        if (
            result["optimizer_steps"] != 20000
            or digest(original) != result["checkpoint_sha256"]
        ):
            raise ValueError("Incomplete or modified original checkpoint")
        saved = torch.load(original, map_location="cpu", weights_only=True)
        if saved["step"] != 20000 or saved["source_hashes"] != submission["files"]:
            raise ValueError("Training checkpoint source/step mismatch")
        # These are the exact five constructor fields used by frozen networks.py.
        # Runtime invocation uses policy_id and device only; training trajectories,
        # labels, optimizer state and random-state records are not inference inputs.
        metadata = {key: saved["spec"]["data_metadata"][key] for key in metadata_keys}
        exported = dict(
            schema="real_robot.inference_checkpoint.v1",
            policy_id=name,
            state_dict=saved["ema"],
            source_hashes=submission["files"],
            spec=dict(policy_id=name, device="cuda:0", data_metadata=metadata),
        )
        target = weights / (name + ".pt")
        if not target.exists():
            temporary = target.with_suffix(".tmp")
            torch.save(exported, temporary)
            temporary.replace(target)
        reread = torch.load(target, map_location="cpu", weights_only=True)
        if set(reread) != set(exported):
            raise ValueError("Existing exported checkpoint schema differs")
        for key in exported:
            if key != "state_dict" and reread[key] != exported[key]:
                raise ValueError("Exported metadata differs: " + key)
        if set(reread["state_dict"]) != set(saved["ema"]):
            raise ValueError("Exported tensor keys differ")
        for key, tensor in saved["ema"].items():
            other = reread["state_dict"][key]
            if (
                tensor.dtype != other.dtype
                or tensor.shape != other.shape
                or not torch.equal(tensor, other)
            ):
                raise ValueError("Export changed a final EMA tensor: " + key)
        manifest["policies"][name] = dict(
            file=target.name,
            sha256=digest(target),
            bytes=target.stat().st_size,
            original_checkpoint_sha256=result["checkpoint_sha256"],
            original_bytes=original.stat().st_size,
            optimizer_steps=result["optimizer_steps"],
            trainable_parameters=result["trainable_parameters"],
            training_examples=result["training_examples"],
            exact_final_ema_tensor_equality=True,
            tensor_count=len(saved["ema"]),
        )
    path = package / "manifest.json"
    if path.exists() and read(path) != manifest:
        raise ValueError(
            "Existing deployment manifest differs; retain a new explicit version"
        )
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-package",
        default=str(ROOT / "real_robot/runs/push_letters/train_v1/package_00"),
    )
    parser.add_argument("--package", default=str(PACKAGE))
    parser.add_argument("--weights", default=str(WEIGHTS))
    args = parser.parse_args()
    export(args.training_package, args.package, args.weights)
