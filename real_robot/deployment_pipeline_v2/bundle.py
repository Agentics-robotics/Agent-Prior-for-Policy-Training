"""Explicit portable download allowlist; excludes recordings and training caches."""
from appl.io import ROOT, atomic, digest
from real_robot.deployment.bundle import archive
from .common import BUNDLES,locations,verify


def build():
    output = ROOT / "real_robot/exports/pipeline_v2"
    output.mkdir(parents=True,exist_ok=True)
    files = {ROOT / "real_robot/__init__.py"}
    for directory,names in {
        "real_robot/training_pipeline":("__init__.py","public.py","security.py"),
        "src/appl":("__init__.py","io.py","gpu.py","kernel.py"),
        "real_robot/deployment":("__init__.py","common.py","policy.py","bundle.py"),
        "real_robot/training_v4_environment":("pixi.toml","pixi.lock"),
    }.items():
        files.update(ROOT / directory / name for name in names)
    files.update((ROOT / "real_robot/deployment_pipeline_v2").glob("*.py"))
    files.add(ROOT / "real_robot/deployment_pipeline_v2/README.md")
    files.add(ROOT / "real_robot/deployment_pipeline_v2/PROVENANCE.json")
    weights,assets = [],[]
    for bundle in BUNDLES:
        package,wd,ad = locations(bundle)
        manifest = verify(package,wd,ad,require_assets=True)
        files.update(p for p in package.rglob("*") if p.is_file())
        weights.extend((wd / p["file"],str((wd / p["file"]).relative_to(ROOT))) for p in manifest["policies"].values())
        assets.extend((ad / name,str((ad / name).relative_to(ROOT))) for name in manifest["assets"])
    full = [(p,"pipeline_v2_bundle/"+str(p.relative_to(ROOT))) for p in sorted(files)]
    full += [(p,"pipeline_v2_bundle/"+n) for p,n in weights+assets]
    archives = {}
    for name,contents in (("pipeline_v2_weights.tar.gz",weights),
                          ("pipeline_v2_sam_assets.tar.gz",assets),
                          ("pipeline_v2_bundle.tar.gz",full)):
        path = output / name
        archive(path,contents)
        archives[name] = dict(bytes=path.stat().st_size,sha256=digest(path))
    record = dict(schema="real_robot.pipeline_download.v2",archives=archives,
        files={name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p,name in full},
        original_training_modified=False,API_calls=0,robot_execution=False)
    atomic(output / "bundle_manifest.json",record)
    (output / "SHA256SUMS").write_text("".join(f"{v['sha256']}  {n}\n" for n,v in archives.items()))
    print(archives,flush=True)


if __name__ == "__main__":
    build()
