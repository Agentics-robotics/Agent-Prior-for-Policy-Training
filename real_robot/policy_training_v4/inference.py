"""Local observation-in/geometric-decision-out worker; no robot IO."""

from pathlib import Path
import base64
import json
import subprocess
import sys

import numpy as np

from appl.io import ROOT, digest, read
from .runner import command, environment


def wire(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "buif" or value.nbytes > 32_000_000:
            raise ValueError("Only bounded numeric observation arrays are accepted")
        value = np.ascontiguousarray(value)
        return dict(
            array_base64=base64.b64encode(value.tobytes()).decode(),
            dtype=str(value.dtype),
            shape=list(value.shape),
        )
    if isinstance(value, np.generic):
        return wire(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: wire(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [wire(v) for v in value]
    return value


def unwire(value):
    if isinstance(value, dict) and set(value) == {"array_base64", "dtype", "shape"}:
        dtype = np.dtype(value["dtype"])
        if dtype.kind not in "buif":
            raise ValueError("Object arrays are not observation inputs")
        raw = base64.b64decode(value["array_base64"], validate=True)
        if len(raw) > 32_000_000:
            raise ValueError("Observation array exceeds bound")
        return np.frombuffer(raw, dtype=dtype).reshape(value["shape"]).copy()
    if isinstance(value, dict):
        return {k: unwire(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unwire(v) for v in value]
    return value


class PolicyProcess:
    """High-level Agent host adapter around an isolated trained policy worker.

    act computes geometric decisions only; executor_request describes an objective
    only. Hardware integration is outside this object and this training stage.
    """

    def __init__(self, config, policy_id, output, gpu=0, revision=0):
        cfg = read(config)
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.errors = (self.output / "stderr.log").open("w")
        args = [
            "worker",
            "--config",
            str(Path(config).resolve()),
            "--package",
            str(ROOT / cfg["run"] / f"package_{revision:02d}"),
            "--operation",
            "serve",
            "--output",
            str(self.output),
            "--gpu",
            str(gpu),
            "--policy-id",
            policy_id,
        ]
        self.process = subprocess.Popen(
            command(args),
            cwd=ROOT,
            env=environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.errors,
            text=True,
            bufsize=1,
        )
        line = self.process.stdout.readline()
        if not line or json.loads(line).get("status") != "ready":
            raise RuntimeError(
                "Policy worker failed: "
                + (self.output / "stderr.log").read_text()[-6000:]
            )

    def act(self, observation, call, reset=False, executor_contract=None):
        request = wire(
            dict(
                observation=observation,
                call=call,
                reset=reset,
                executor_contract=executor_contract,
            )
        )
        self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(
                "Policy worker exited: "
                + (self.output / "stderr.log").read_text()[-6000:]
            )
        return unwire(json.loads(line))

    def inventory(self, observation, scene_version):
        request = wire(
            dict(
                operation="inventory",
                observation=observation,
                scene_version=scene_version,
            )
        )
        self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Policy inventory worker exited")
        return unwire(json.loads(line))

    def close(self):
        self.process.stdin.close()
        code = self.process.wait(timeout=30)
        self.errors.close()
        if code:
            raise RuntimeError("Policy inference worker failed")


def serve(cfg, package, output, policy_id):
    import torch
    from . import public
    from .engine import module_at, source_identity
    from .security import lockdown

    package, output = Path(package), Path(output)
    checkpoint = package / "training" / policy_id / "last.pt"
    result = read(checkpoint.parent / "result.json")
    if (
        digest(checkpoint) != result["checkpoint_sha256"]
        or source_identity(package / "source") != result["source_hashes"]
    ):
        raise ValueError("Trained source/checkpoint identity mismatch")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    spec = saved["spec"]
    # Training-only source identities/anchors are not online observation assets.
    spec = {
        k: v
        for k, v in spec.items()
        if k not in {"source_plan", "trajectory_ids", "recording_metadata"}
    }
    spec["device"] = "cuda:0"
    assets = ROOT / cfg["run"] / "assets"
    public.configure(None, [], output, assets, package / "source")
    torch.set_num_threads(2)
    lockdown(package / "source", output, [checkpoint, assets], gpu=True)
    module = module_at(package / "source")
    model = module.build_model(policy_id, spec).to("cuda:0")
    model.load_state_dict(saved["ema"])
    model.eval().requires_grad_(False)
    memory = None
    print(
        json.dumps(dict(status="ready", policy_id=policy_id, hardware_io=False)),
        flush=True,
    )
    for line in sys.stdin:
        request = unwire(json.loads(line))
        if request.get("operation") == "inventory":
            if not callable(getattr(module, "inventory", None)):
                raise ValueError(
                    "This API package does not expose an inventory converter"
                )
            result = module.inventory(request["observation"], spec)
            answer = {k: v for k, v in result.items() if k != "memory"}
            answer["scene_version"] = request["scene_version"]
            answer["hardware_io"] = False
            print(json.dumps(wire(answer), allow_nan=False), flush=True)
            continue
        if request.get("reset"):
            memory = None
        with torch.no_grad():
            result = module.act(
                model, request["observation"], request["call"], memory, spec
            )
        memory = result["memory"]
        decision = result["decision"]
        if decision is not None:
            import jsonschema
            jsonschema.validate(decision, spec["policy_config"]["decision_schema"])
            json.dumps(decision, allow_nan=False)
        answer = {k: v for k, v in result.items() if k != "memory"}
        if request.get("executor_contract") is not None and decision is not None:
            answer["executor_request"] = module.executor_request(
                decision, request["executor_contract"], spec
            )
        answer["hardware_io"] = False
        print(json.dumps(wire(answer), allow_nan=False), flush=True)
