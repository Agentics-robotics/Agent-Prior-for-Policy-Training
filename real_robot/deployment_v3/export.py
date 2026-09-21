"""Lossless export of frozen API source and final EMA tensors, with provenance."""

import argparse
import json
from pathlib import Path
import shutil

import torch

from real_robot.deployment.common import ROOT, read, digest
from .policy import PACKAGE, WEIGHTS


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
        api_contract="push_training_v3",
        source_package_hash=submission["package_hash"],
        cut_plan_hash=submission["source_plan_hash"],
        source_hashes=submission["files"],
        scientific_source="Exact bytes of API design_00 submission; no developer edits",
        selection="Final EMA at 20000 updates, as in original inference loader",
        policies={},
    )
    for name in ("tool_waypoint_v1", "piece_contact_graph_v1", "piece_goal_field_v1"):
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
        # Preserve the complete inference spec, as the original trained-policy
        # loader does; only its three explicitly training-only fields are omitted.
        spec = {
            k: v
            for k, v in saved["spec"].items()
            if k not in {"source_plan", "trajectory_ids", "recording_metadata"}
        }
        spec["device"] = "cuda:0"
        exported = dict(
            schema="real_robot.inference_checkpoint.v1",
            policy_id=name,
            state_dict=saved["ema"],
            source_hashes=submission["files"],
            spec=spec,
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
    prepared = read(training_package / "prepared/result.json")
    lines = [
        "# Actual training coverage — training_v3",
        "",
        "Developer-authored summary of execution receipts. Policy designs, priors and calling contracts remain exact API documents in source/.",
        "",
        "Each model completed 20,000 independent optimizer updates (seed 0). The exported tensors are exactly the final EMA, without quantization or performance-based selection.",
        "",
        "| Policy | Parameters | Eligible examples | Updates |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, item in manifest["policies"].items():
        lines.append(
            f"| {name} | {item['trainable_parameters']} | {item['training_examples']} | {item['optimizer_steps']} |"
        )
    counts = prepared["policy_segment_examples"]
    lines += [
        "",
        f"The prepared cache contains {prepared['num_examples']} anchors from 26 frozen segments. Tool sampling uses its 4 assigned segments; graph and field independently use the same 22 piece segments. These are correlated rows from two demonstrations, with 382 reused source rows.",
        "",
        "| Policy | Original cut | Eligible examples |",
        "| --- | --- | ---: |",
    ]
    lines += [f"| {pid} | {name} | {count} |" for pid, segments in counts.items()
              for name, count in segments.items() if count]
    lines += [
        "",
        "Both piece models have 15,165 automatically accepted object-goal rows; a_bar_stage's 600 rows retain native action supervision with a missing-goal mask and endpoint TCP condition. All perception/endpoint/contact labels are unreviewed weak labels. Tool goals use TCP poses, so object-goal masks do not apply to that model.",
        "",
        "All three produce six-dimensional native v/w candidates. They are not hard-constrained 2D push policies. The exact API decoder remains disabled; integrating the recording controller is separate from model inference.",
        "",
        "Neither completed training nor interface checks establish robot success, real-time control, recovery/handoff competence or generalization to unseen letters and words. Read each policy's prior/usage and HANDOFF.json together with these coverage limits.",
        "",
        f"Frozen source package: `{submission['package_hash']}`.",
        "",
    ]
    summary = package / "TRAINING.md"
    content = "\n".join(lines)
    if summary.exists() and summary.read_text() != content:
        raise ValueError("Existing training coverage summary differs")
    summary.write_text(content)
    manifest["training_summary"] = dict(file=summary.name, sha256=digest(summary))
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
        default=str(ROOT / "real_robot/runs/push_letters/training_v3/package_00"),
    )
    parser.add_argument("--package", default=str(PACKAGE))
    parser.add_argument("--weights", default=str(WEIGHTS))
    args = parser.parse_args()
    export(args.training_package, args.package, args.weights)
