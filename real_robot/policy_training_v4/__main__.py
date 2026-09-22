"""Launch developer-owned GPU workers around immutable API packages."""

import argparse
import json
import os
from pathlib import Path
import sys

from appl.io import ROOT
from .design import configuration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["execute", "worker", "status", "publish"])
    parser.add_argument("--config", default="real_robot/configs/push_training_v4.json")
    parser.add_argument("--revision", type=int, default=0)
    parser.add_argument("--package")
    parser.add_argument("--operation", choices=["prepare", "train", "serve"])
    parser.add_argument("--output")
    parser.add_argument("--policy-id")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--device-isolated", action="store_true")
    args = parser.parse_args()
    cfg = configuration(ROOT / args.config)
    if args.command == "worker":
        if args.gpu not in cfg["devices"]:
            raise ValueError("Unallocated GPU")
        if not args.device_isolated:
            from appl.gpu import launch

            return launch(args.gpu, sys.argv[1:], module="real_robot.policy_training_v4")
        nodes = {
            p.name
            for p in Path("/dev").glob("nvidia*")
            if p.name.removeprefix("nvidia").isdigit()
        }
        if nodes != {"nvidia" + os.environ["APPL_GPU_MINOR"]}:
            raise RuntimeError("GPU isolation mismatch")
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["APPL_GPU_UUID"]
        from .engine import prepare, fit

        package = Path(args.package)
        if args.operation == "prepare":
            return prepare(cfg, package / "source", args.output)
        if args.operation == "train":
            return fit(
                cfg,
                package / "source",
                package / "prepared",
                args.output,
                args.policy_id,
            )
        if args.operation == "serve":
            from .inference import serve

            return serve(cfg, package, args.output, args.policy_id)
        raise ValueError("Missing worker operation")
    from . import runner

    value = (
        runner.status(cfg)
        if args.command == "status"
        else getattr(runner, args.command)(cfg, args.revision)
    )
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
