"""Integration check of the downloadable archive from a fresh, data-free directory."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    directory = Path(tempfile.mkdtemp(prefix="push-deployment-relocated-"))
    with tarfile.open(args.archive, "r:gz") as stream:
        stream.extractall(directory, filter="data")
    root = directory / "push_v1_bundle"
    assert not (root / "real_robot/runs").exists()
    assert not (root / "real_robot/data.py").exists()
    env = dict(
        PATH=os.defpath,
        LANG="C.UTF-8",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(root) + ":" + str(root / "src"),
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="2",
    )
    code = """
import json,sys
import numpy as np
from real_robot.deployment import PushPolicy
from real_robot.deployment.common import ROOT, verify
manifest=verify()
out=[]
for name in manifest['policies']:
    with PushPolicy(name,gpu=int(sys.argv[1])) as p:
        inventory=p.inventory({},'interface-check')
        assert inventory['status']=='unavailable' and not inventory['instances']
        result=p.act({}, {}, reset=True)
        assert result['status']=='unavailable' and result['action'] is None
        gate=p.decode_action(np.zeros(7,dtype=np.float64), None)
        assert gate['enabled'] is False and gate['command'] is None
        assert all(x['hardware_io'] is False for x in (inventory,result,gate))
        out.append(dict(policy_id=name,ready=p.ready,empty_observation_refused=True,
                        unverified_controller_refused=True,logs=str(p.output)))
print(json.dumps(dict(root=str(ROOT),models=out,robot_actions_executed=0)))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(args.gpu)],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=240,
    )
    receipt = dict(
        archive_sha256=sha(args.archive),
        relocated_root=str(root),
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        checkpoint_bytes=sum(
            p.stat().st_size
            for p in (root / "real_robot/checkpoints/push_v1").glob("*.pt")
        ),
        training_runs_present=False,
        original_data_reader_present=False,
        API_calls=0,
        optimizer_updates=0,
        performance_evaluation=False,
    )
    if result.returncode == 0:
        receipt["result"] = json.loads(result.stdout)
        assert receipt["result"]["root"] == str(root)
        assert len(receipt["result"]["models"]) == 2
        receipt["status"] = "passed"
    else:
        receipt["status"] = "failed"
    Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if result.returncode:
        raise RuntimeError("Relocated deployment check failed; receipt retained")


if __name__ == "__main__":
    main()
