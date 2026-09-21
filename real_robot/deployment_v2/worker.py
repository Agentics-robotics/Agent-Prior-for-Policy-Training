"""Isolated inference worker. No training sources, API credentials or hardware IO."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from real_robot.deployment.common import ROOT, verify, wire, unwire


def launch(gpu, args):
    # Reuse recorded UUID/minor verification. This CUDA-only launcher does not
    # require graphics/render nodes used by the frozen simulator launcher.
    from appl.gpu import identity

    device = identity(gpu)
    if device["free_mib"] < 2048:
        raise RuntimeError(
            "Selected GPU has less than 2 GiB free; no alternate selected"
        )
    nodes = [
        Path("/dev/nvidiactl"),
        Path("/dev/nvidia-uvm"),
        Path("/dev") / f"nvidia{device['minor']}",
    ]
    nodes += [p for p in [Path("/dev/nvidia-uvm-tools")] if p.exists()]
    for p in nodes:
        if not p.is_char_device():
            raise RuntimeError(f"Missing NVIDIA character device: {p}")
    command = [
        "bwrap",
        "--die-with-parent",
        "--bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--unsetenv",
        "DISPLAY",
        "--unsetenv",
        "WAYLAND_DISPLAY",
    ]
    env = dict(
        APPL_PHYSICAL_GPU=gpu,
        APPL_GPU_UUID=device["uuid"],
        APPL_GPU_MINOR=device["minor"],
        CUDA_VISIBLE_DEVICES=device["uuid"],
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        LD_LIBRARY_PATH=str(Path(sys.prefix) / "lib"),
    )
    for name, value in env.items():
        command += ["--setenv", name, str(value)]
    for p in nodes:
        command += ["--dev-bind", str(p), str(p)]
    os.execvp(
        command[0],
        command
        + [
            "--",
            sys.executable,
            "-u",
            "-m",
            "real_robot.deployment_v2.worker",
            *args,
            "--isolated",
        ],
    )


def serve(args):
    import numpy as np
    import torch
    import cv2
    from real_robot.policy_training import public
    from real_robot.policy_training.security import lockdown

    package, weights, output = Path(args.package), Path(args.weights), Path(args.output)
    manifest = verify(package, weights, args.policy)
    if manifest.get("api_contract") != "push_training_v2":
        raise ValueError("This worker only accepts the training_v2 API contract")
    item = manifest["policies"][args.policy]
    saved = torch.load(weights / item["file"], map_location="cpu", weights_only=True)
    if (
        saved["schema"] != "real_robot.inference_checkpoint.v1"
        or saved["policy_id"] != args.policy
        or saved["source_hashes"] != manifest["source_hashes"]
    ):
        raise ValueError("Inference checkpoint contract mismatch")
    spec = saved["spec"]
    if spec["policy_id"] != args.policy:
        raise ValueError("Checkpoint policy identity mismatch")
    spec["device"] = "cuda:0"
    public.configure(None, [], output, output / "no_external_assets")
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    device = lockdown(
        package / "source",
        output,
        [
            weights / item["file"],
            ROOT / "real_robot/deployment",
            ROOT / "real_robot/deployment_v2",
        ],
        gpu=True,
    )
    module_spec = importlib.util.spec_from_file_location(
        "api_robot_policy", package / "source/policy.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    model = module.build_model(args.policy, spec).to("cuda:0")
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    memory = None
    inventory_memory = None
    print(
        json.dumps(
            dict(
                status="ready",
                policy_id=args.policy,
                device=device,
                checkpoint_sha256=item["sha256"],
                hardware_io=False,
            )
        ),
        flush=True,
    )
    for line in sys.stdin:
        request = unwire(json.loads(line))
        operation = request["operation"]
        if operation == "observe_scene":
            result = module.observe_scene(
                request["observation"], request["scene_version"], inventory_memory
            )
            inventory_memory = result["memory"]
        elif operation == "preview_goal":
            result = module.preview_goal(
                request["template_ref"],
                request["spatial_goal"],
                request["workspace_polygon_uv"],
            )
        elif operation == "reexpress_goal":
            result = module.reexpress_goal(
                request["new_template"], request["old_goal_mask"]
            )
        elif operation == "act":
            if request.get("reset"):
                memory = None
            with torch.no_grad():
                result = module.act(
                    model, request["observation"], request["call"], memory, spec
                )
            memory = result["memory"]
            action = result["action"]
            if action is not None and (
                np.asarray(action).shape != (7,) or not np.isfinite(action).all()
            ):
                raise ValueError(
                    "API policy returned a nonfinite or malformed candidate"
                )
            if action is not None and request.get("controller_contract") is not None:
                result["decoded"] = module.decode_action(
                    action, request["controller_contract"], spec
                )
        elif operation == "decode":
            result = module.decode_action(
                request["action"], request["controller_contract"], spec
            )
        else:
            raise ValueError("Unknown operation: " + str(operation))
        answer = {k: v for k, v in result.items() if k != "memory"}
        answer["hardware_io"] = False
        print(json.dumps(wire(answer), allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--isolated", action="store_true")
    args = parser.parse_args()
    if not args.isolated:
        return launch(args.gpu, sys.argv[1:])
    nodes = {
        p.name
        for p in Path("/dev").glob("nvidia*")
        if p.name.removeprefix("nvidia").isdigit()
    }
    if nodes != {"nvidia" + os.environ["APPL_GPU_MINOR"]}:
        raise RuntimeError("GPU device isolation mismatch")
    return serve(args)


if __name__ == "__main__":
    main()
