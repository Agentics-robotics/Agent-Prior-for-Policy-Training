"""Build two SCP-friendly archives without training data, caches or credentials."""

import argparse
import gzip
import json
from pathlib import Path
import tarfile

from .common import ROOT, WEIGHTS, digest, verify


def archive(target, files):
    if target.exists():
        raise FileExistsError("Keep existing exports immutable: " + str(target))
    temporary = target.with_suffix(".partial")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as tar:
                for path, name in files:
                    if path.is_symlink() or not path.is_file():
                        raise ValueError("Only explicit regular files can be bundled")
                    info = tar.gettarinfo(str(path), arcname=name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as source:
                        tar.addfile(info, source)
    temporary.replace(target)


def build(output):
    manifest = verify()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    weights = [
        (WEIGHTS / value["file"], "real_robot/checkpoints/push_v1/" + value["file"])
        for value in manifest["policies"].values()
    ]
    files = [ROOT / "real_robot/__init__.py"]
    files += [
        ROOT / "real_robot/policy_training" / name
        for name in ("__init__.py", "public.py", "security.py")
    ]
    files += [
        ROOT / "src/appl" / name
        for name in ("__init__.py", "io.py", "gpu.py", "kernel.py")
    ]
    files += list((ROOT / "real_robot/deployment").glob("*.py"))
    files += [
        ROOT / "real_robot/deployment" / name
        for name in ("pixi.toml", "pixi.lock", "README.md")
    ]
    files += [ROOT / "real_robot/policies/push_v1/manifest.json"]
    files += list((ROOT / "real_robot/policies/push_v1/source").iterdir())
    selected = [(p, str(p.relative_to(ROOT))) for p in sorted(files)] + weights
    full = [(p, "push_v1_bundle/" + name) for p, name in selected]
    full.append(
        (ROOT / "real_robot/deployment/BUNDLE_README.md", "push_v1_bundle/README.md")
    )
    archives = {}
    for name, contents in (
        ("push_v1_bundle.tar.gz", full),
        ("push_v1_weights.tar.gz", weights),
    ):
        path = output / name
        archive(path, contents)
        archives[name] = dict(bytes=path.stat().st_size, sha256=digest(path))
    inventory = {
        name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p, name in full
    }
    record = dict(
        schema="real_robot.download_bundle.v1",
        archives=archives,
        files=inventory,
        original_training_artifacts_modified=False,
        API_calls=0,
        hardware_io=False,
    )
    (output / "bundle_manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    (output / "SHA256SUMS").write_text(
        "".join(f"{value['sha256']}  {name}\n" for name, value in archives.items())
    )
    print(json.dumps(archives, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "real_robot/exports/push_v1"))
    args = parser.parse_args()
    build(args.output)
