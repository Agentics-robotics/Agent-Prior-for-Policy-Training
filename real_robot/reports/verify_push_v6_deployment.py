"""Verify the actual relocated Push v6 archive using the original worker fixture."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

from appl.io import ROOT, PIXI, atomic, read, digest


def main():
    archive = ROOT / "real_robot/exports/pipeline_v2/pipeline_v2_bundle.tar.gz"
    manifest = read(archive.parent / "bundle_manifest.json")
    assert digest(archive) == manifest["archives"][archive.name]["sha256"]
    directory = Path(tempfile.mkdtemp(prefix="push-v6-relocated-"))
    with tarfile.open(archive, "r:gz") as stream:
        names = stream.getnames()
        assert len(names) == len(set(names))
        stream.extractall(directory, filter="data")
    for name, entry in manifest["files"].items():
        assert digest(directory / name) == entry["sha256"], name
    root = directory / "pipeline_v2_bundle"
    assert not (root / "real_robot/runs").exists()
    assert not (root / "real_robot/data.py").exists()
    reference = read(ROOT / "real_robot/runs/push_letters/training_v6/host_worker_check/result.json")
    assert reference["passed"]
    fixture = directory / "host_fixture.json"
    atomic(fixture, reference)
    code = r'''
import sys,json
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);sys.path[:0]=[str(root),str(root/'src')]
from real_robot.deployment_pipeline_v2 import Policy
from real_robot.deployment_pipeline_v2.common import ROOT,locations,verify
assert ROOT==root
reference=json.loads(Path(sys.argv[2]).read_text());f=reference['fixture']
manifest=verify(*locations('push_v6'),require_assets=True)
with Policy('push_v6',gpu=4,timeout_s=180) as p:
    first=p.act(f['observation'],f['call'],reset=True,executor_contract={})
    second=p.act(f['observation'],f['call'],executor_contract=f['executor_contract'])
    errors={}
    for field in ('contact_point_m','direction_unit','stroke_length_m'):
        errors[field]=float(np.max(np.abs(np.asarray(first['decision'][field])-np.asarray(reference['cases'][0]['decision'][field]))))
    assert max(errors.values())<=1e-6,errors
    for answer,original in zip((first,second),reference['cases'][:2]):
        assert answer['status']==original['status']=='proposed'
        assert answer['executor_request']==original['executor_request']
        assert not answer['hardware_io']
    stale=p.act(f['observation'],dict(f['call'],scene_revision=10))
    stopped=p.act(f['observation'],dict(f['call'],interrupted=True))
    incomplete=p.act(dict(f['observation'],geometry_complete_validated=False),f['call'])
    assert stale['status']=='stale_scene' and stale['decision'] is None
    assert stopped['status']=='interrupted' and stopped['decision'] is None
    assert incomplete['status']=='needs_view' and incomplete['decision'] is None
    result=dict(passed=True,root=str(ROOT),original_worker_output_errors=errors,
        statuses=[x['status'] for x in (first,second,stale,stopped,incomplete)],
        executor_objectives_identical=True,required_assets_loaded=list(manifest['assets']),
        ready=p.ready,logs=str(p.output),hardware_io=False)
from real_robot.deployment_pipeline_v2.__main__ import main
import contextlib,io
sys.argv=['deployment_pipeline_v2','example','--bundle','push_v6','--gpu','4']
capture=io.StringIO()
with contextlib.redirect_stdout(capture):
    main()
example=json.loads(capture.getvalue())
assert example['uses_trained_checkpoint'] and not example['hardware_io']
assert example['result']['decision']==second['decision']
assert example['result']['executor_request']==second['executor_request']
result['CLI_example_matches_worker']=True
result['CLI_example_timing']={k:v for k,v in example.items() if k.startswith('observed_') or k=='timing_scope'}
print(json.dumps(result))
'''
    env = dict(PATH=os.defpath, LANG="C.UTF-8", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    lock_path = ROOT / "real_robot/runs/.training_pipeline_locks/gpu_4.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = subprocess.run([PIXI, "run", "--manifest-path", str(ROOT / "real_robot/training_v4_environment/pixi.toml"),
            "--locked", "--no-install", "python", "-I", "-c", code, str(root), str(fixture)],
            cwd=root, env=env, text=True, capture_output=True, timeout=480)
    receipt = dict(date="2026-09-22", passed=result.returncode == 0, archive_sha256=digest(archive),
        relocated_root=str(root), returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
        files_verified=len(manifest["files"]), training_data_bundled=False, synthetic_executor_flags=True,
        API_calls=0, candidate_optimizer_updates=0, robot_actions_executed=0,
        environment_lock_sha256=digest(ROOT / "real_robot/training_v4_environment/pixi.lock"))
    if result.returncode == 0:
        receipt["result"] = json.loads(result.stdout)
    atomic(ROOT / "real_robot/reports/PUSH_V6_DEPLOYMENT_CHECK.json", receipt)
    print(json.dumps(receipt), flush=True)
    if result.returncode:
        raise RuntimeError("Relocated deployment check failed; see saved receipt")


if __name__ == "__main__":
    main()
