"""File identities and paths only; no task or policy semantics."""
from pathlib import Path

from appl.io import ROOT, read, digest
from real_robot.deployment.common import wire, unwire

BUNDLES = ("push_v5", "flip_egg_v1")


def locations(bundle):
    if bundle not in BUNDLES:
        raise ValueError("Unknown exported bundle: " + bundle)
    package = ROOT / "real_robot/policies" / bundle
    weights = ROOT / "real_robot/checkpoints" / bundle
    return package, weights, weights / "assets"


def verify(package, weights, assets, policy_id=None, require_assets=False):
    package, weights, assets = map(Path, (package, weights, assets))
    manifest = read(package / "manifest.json")
    if manifest["schema"] != "real_robot.pipeline_deployment.v1":
        raise ValueError("Unsupported deployment contract")
    actual = {p.name: digest(p) for p in (package / "source").iterdir() if p.is_file()}
    if actual != manifest["source_hashes"]:
        raise ValueError("Frozen API source differs")
    for filename, sha in manifest["runtime_hashes"].items():
        if digest(ROOT / filename) != sha:
            raise ValueError("Frozen inference dependency differs: " + filename)
    for pid in ([policy_id] if policy_id else manifest["policies"]):
        item = manifest["policies"][pid]
        if Path(item["file"]).name != item["file"]:
            raise ValueError("Expected a flat checkpoint filename")
        path = weights / item["file"]
        if not path.is_file():
            raise FileNotFoundError("Download checkpoint into " + str(path))
        if digest(path) != item["sha256"]:
            raise ValueError("Checkpoint hash mismatch: " + pid)
    for name, item in manifest["assets"].items():
        if Path(name).name != name:
            raise ValueError("Expected a flat asset filename")
        path = assets / name
        if path.exists():
            if digest(path) != item["sha256"]:
                raise ValueError("Asset hash mismatch: " + name)
        elif require_assets:
            raise FileNotFoundError("Download optional inventory asset into " + str(path))
    return manifest
