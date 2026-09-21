"""Check the actual portable archive using the locked inference environment."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

from appl.io import PIXI
from real_robot.deployment.common import ROOT, digest, read, wire
from real_robot.reports.verify_push_v3 import observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    archive = ROOT / "real_robot/exports/push_v3/push_v3_bundle.tar.gz"
    inventory = read(archive.parent / "bundle_manifest.json")
    assert digest(archive) == inventory["archives"][archive.name]["sha256"]
    directory = Path(tempfile.mkdtemp(prefix="push-v3-relocated-"))
    with tarfile.open(archive, "r:gz") as stream:
        names = stream.getnames()
        assert len(names) == len(set(names)), "Duplicate archive member"
        stream.extractall(directory, filter="data")
    for name, entry in inventory["files"].items():
        assert digest(directory / name) == entry["sha256"], name
    root = directory / "push_v3_bundle"
    assert not (root / "real_robot/runs").exists()
    assert not (root / "real_robot/data.py").exists()
    cfg = read(ROOT / "real_robot/configs/push_training_v3.json")
    ob, tid = observation(cfg, 300)
    reference = {}
    for pid in cfg["policies"]:
        reference[pid] = read(ROOT / cfg["run"] / "interface_check_00" / pid / "responses.json")
    fixture = directory / "host_input.json"
    fixture.write_text(json.dumps(wire(dict(observation=ob, reference=reference)), allow_nan=False))
    code = r'''
import sys, json, contextlib, io
from pathlib import Path
root = Path(sys.argv[1])
sys.path[:0] = [str(root), str(root / "src")]
import numpy as np
from real_robot.deployment.common import ROOT, unwire
from real_robot.deployment_v3 import PushPolicy
from real_robot.deployment_v3.__main__ import main as check
assert ROOT == root
gpu=int(sys.argv[2]); fixture=unwire(json.load(open(sys.argv[3])))
ob=fixture['observation']; references=fixture['reference']
sys.argv=['deployment_v3','check','--files-only']
cli=io.StringIO()
with contextlib.redirect_stdout(cli): check()
cases=[]
for pid, ref in references.items():
    with PushPolicy(pid,gpu=gpu) as p:
        invalid=p.inventory({}, 'invalid')
        assert invalid['status']=='PERCEPTION_UNCERTAIN' and not invalid['instances']
        scene=p.inventory(ob,'bounded-interface-fixture')
        assert scene['instances'] and scene['hardware_io'] is False
        call=ref['call']
        valid=p.act(ob,call,reset=True,controller_contract={})
        a=np.asarray(valid['action'])
        assert a.shape==(6,) and np.isfinite(a).all(), valid
        assert valid['status']==ref['valid']['status']
        err=float(np.max(np.abs(a-np.asarray(ref['valid']['action']))))
        assert err<=1e-6, (pid,err)
        assert valid['decoded']['enabled'] is False and valid['decoded']['command'] is None
        stale=p.act(ob,dict(call,now_s=float(ob['t'])+2.))
        assert stale['status']=='PERCEPTION_UNCERTAIN' and stale['action'] is None
        abort=p.act(ob,dict(call,abort=True))
        assert abort['status']=='ABORTED' and abort['action'] is None
        key='goal_tcp_base' if pid=='tool_waypoint_v1' else 'goal_SE2_W'
        bad=np.asarray(call[key]).copy(); bad[0,0]=2.
        bad_call=dict(call); bad_call[key]=bad.tolist()
        rejected=p.act(ob,bad_call,reset=True)
        assert rejected['status']=='PERCEPTION_UNCERTAIN' and rejected['action'] is None
        gate=p.decode_action(a,{})
        assert gate['enabled'] is False and gate['command'] is None
        assert all(r['hardware_io'] is False for r in (invalid,scene,valid,stale,abort,rejected,gate))
        cases.append(dict(policy_id=pid,ready=p.ready,finite_candidate=True,
                          original_worker_max_action_error=err,
                          stale_input_rejected=True,invalid_goal_rejected=True,
                          interruption_supported=True,decoder_disabled=True,logs=str(p.output)))
print(json.dumps(dict(root=str(ROOT),models=cases,files_only_cli=cli.getvalue(),robot_actions_executed=0)))
'''
    env = dict(PATH=os.defpath, LANG="C.UTF-8", PYTHONNOUSERSITE="1",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    manifest = ROOT / "real_robot/deployment/pixi.toml"
    result = subprocess.run(
        [PIXI, "run", "--manifest-path", str(manifest), "--locked", "--no-install",
         "python", "-I", "-c", code, str(root), str(args.gpu), str(fixture)],
        cwd=root, env=env, text=True, capture_output=True, timeout=300)
    receipt = dict(date="2026-09-21", status="passed" if result.returncode == 0 else "failed",
                   archive_sha256=digest(archive), relocated_root=str(root),
                   returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                   files_verified=len(inventory["files"]), training_runs_present=False,
                   original_data_reader_present=False,
                   inference_environment_manifest_sha256=digest(manifest),
                   inference_environment_lock_sha256=digest(manifest.with_name("pixi.lock")),
                   paired_input=dict(trajectory=tid,sample_index=300),
                   fixture_goal="Current TCP/identity footprint; interface exercise only",
                   API_calls=0,candidate_optimizer_updates=0,
                   performance_evaluation=False,robot_actions_executed=0)
    if result.returncode == 0:
        receipt["result"] = json.loads(result.stdout)
        assert receipt["result"]["root"] == str(root)
    path = ROOT / "real_robot/reports/PUSH_V3_DEPLOYMENT_CHECK.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if result.returncode:
        raise RuntimeError("Portable v3 check failed; retained receipt: " + str(path))


if __name__ == "__main__":
    main()
