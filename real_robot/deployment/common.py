"""Portable file identity and numeric IPC. No policy or robot semantics."""

import base64
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "real_robot/policies/push_v1"
WEIGHTS = ROOT / "real_robot/checkpoints/push_v1"
MAX_ARRAY_BYTES = 32_000_000


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def verify(package=PACKAGE, weights=WEIGHTS, policy_id=None):
    package, weights = Path(package).resolve(), Path(weights).resolve()
    manifest = read(package / "manifest.json")
    if manifest["schema"] != "real_robot.push_deployment.v1":
        raise ValueError("Unsupported deployment manifest")
    actual = {p.name: digest(p) for p in (package / "source").iterdir() if p.is_file()}
    if actual != manifest["source_hashes"]:
        raise ValueError("API source identity differs from the frozen submission")
    policies = [policy_id] if policy_id else list(manifest["policies"])
    for name in policies:
        item = manifest["policies"][name]
        filename = item["file"]
        if Path(filename).name != filename:
            raise ValueError("Checkpoint filename must be flat")
        file = weights / filename
        if not file.is_file():
            raise FileNotFoundError(f"Download {filename} into {weights}")
        if digest(file) != item["sha256"]:
            raise ValueError("Checkpoint identity mismatch: " + name)
    return manifest


def wire(value):
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "buif" or value.nbytes > MAX_ARRAY_BYTES:
            raise ValueError("Only bounded numeric arrays are accepted")
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
            raise ValueError("Only numeric arrays are accepted")
        raw = base64.b64decode(value["array_base64"], validate=True)
        if len(raw) > MAX_ARRAY_BYTES:
            raise ValueError("Array exceeds observation bound")
        return np.frombuffer(raw, dtype=dtype).reshape(value["shape"]).copy()
    if isinstance(value, dict):
        return {k: unwire(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unwire(v) for v in value]
    return value
