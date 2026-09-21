"""Explicit v2 download allowlist, with no training data or credentials."""

import json

from real_robot.deployment.bundle import archive
from real_robot.deployment.common import ROOT, digest, verify
from .policy import PACKAGE, WEIGHTS


def build():
    manifest = verify(PACKAGE, WEIGHTS)
    if manifest.get("api_contract") != "push_training_v2":
        raise ValueError("Expected v2 inference package")
    output = ROOT / "real_robot/exports/push_v2"
    output.mkdir(parents=True, exist_ok=True)
    weights = [
        (WEIGHTS / p["file"], "real_robot/checkpoints/push_v2/" + p["file"])
        for p in manifest["policies"].values()
    ]
    files = [ROOT / "real_robot/__init__.py"]
    files += [
        ROOT / "real_robot/policy_training" / n
        for n in ("__init__.py", "public.py", "security.py")
    ]
    files += [
        ROOT / "src/appl" / n for n in ("__init__.py", "io.py", "gpu.py", "kernel.py")
    ]
    files += [
        ROOT / "real_robot/deployment" / n
        for n in (
            "__init__.py",
            "common.py",
            "policy.py",
            "bundle.py",
            "pixi.toml",
            "pixi.lock",
        )
    ]
    files += list((ROOT / "real_robot/deployment_v2").glob("*.py"))
    files += [
        ROOT / "real_robot/deployment_v2/README.md",
        PACKAGE / "manifest.json",
        PACKAGE / "TRAINING.md",
    ]
    files += list((PACKAGE / "source").iterdir())
    full = [(p, "push_v2_bundle/" + str(p.relative_to(ROOT))) for p in sorted(files)]
    full += [(p, "push_v2_bundle/" + n) for p, n in weights]
    full += [
        (ROOT / "real_robot/deployment_v2/BUNDLE_README.md", "push_v2_bundle/README.md")
    ]
    archives = {}
    for name, contents in (
        ("push_v2_bundle.tar.gz", full),
        ("push_v2_weights.tar.gz", weights),
    ):
        path = output / name
        archive(path, contents)
        archives[name] = dict(bytes=path.stat().st_size, sha256=digest(path))
    record = dict(
        schema="real_robot.download_bundle.v1",
        api_contract="push_training_v2",
        archives=archives,
        files={n: dict(bytes=p.stat().st_size, sha256=digest(p)) for p, n in full},
        original_training_artifacts_modified=False,
        API_calls=0,
        hardware_io=False,
    )
    (output / "bundle_manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    (output / "SHA256SUMS").write_text(
        "".join(f"{v['sha256']}  {n}\n" for n, v in archives.items())
    )
    print(json.dumps(archives, indent=2))


if __name__ == "__main__":
    build()
