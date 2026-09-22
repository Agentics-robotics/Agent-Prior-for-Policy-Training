"""Exercise the actual relocated archive against original worker responses."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

import numpy as np
from PIL import Image

from appl.io import ROOT, PIXI, atomic, read, digest
from real_robot.deployment.common import wire


def fixtures():
    push_root = ROOT / "real_robot/runs/push_letters/training_v5"
    f = read(push_root / "package_01/prepared/metadata.json")["calling_fixture"]
    chart = np.eye(4); chart[2,3] = .068
    push_ob = dict(scene_revision=1,time_s=10.,calibration_id="offline_host_fixture",
        T_base_ee=f["T_base_ee"],T_base_P=chart.tolist(),uncertainty_m=.008,chart_validated=True,
        instances=[dict(instance_id="recorded_support",support_points_P=f["support"],geometry_complete=False)],
        previous_delta_P_m=f["previous_delta"],history_dt_s=.2)
    push_call = dict(instance_id="recorded_support",goal_R=f["R"],goal_t=f["t"],
        scene_revision=1,now_s=10.01,contact_confirmed=True)
    cut = read(ROOT / "real_robot/configs/push_cut_v6_with_example.json")
    original = Path(cut["source_root"]) / cut["trajectory_ids"][f["source"]["episode"]]
    # Actual source image/calibration, used only as an explicit host test input.
    with Image.open(original / "third" / f"{f['source']['a']:06d}.jpg") as im:
        rgb = np.array(im.convert("RGB"))
    meta = read(original / "meta.json")
    inventory = dict(rgb_third=rgb,camera=meta["calibration"]["third"],
        top_plane=[0.,0.,.068],scene_revision=1)

    flip_root = ROOT / "real_robot/runs/flip_egg/training_v1"
    reference = read(flip_root / "host_worker_check/result.json")
    cut = read(ROOT / "real_robot/configs/flip_egg_cut_v1.json")
    original = Path(cut["source_root"]) / reference["original_episode"]
    index = reference["original_index"]
    with np.load(original / "obs.npz",allow_pickle=False) as z:
        rows = {k:z[k] for k in z.files}
    def rgb_at(i,camera):
        with Image.open(original / camera / f"{i:06d}.jpg") as im:
            return np.array(im.convert("RGB"))
    def sample(i):
        return dict(t=float(rows["t"][i]),T_base_ee=rows["T_base_ee"][i].tolist(),
            tau_ext=rows["tau_ext"][i].tolist(),gripper_width_m=float(rows["gripper_width_m"][i]),
            image_age_s=[float(rows["img_age_third"][i]),float(rows["img_age_wrist"][i])],
            third_rgb=rgb_at(i,"third"),wrist_rgb=rgb_at(i,"wrist"))
    samples = [sample(i) for i in range(index-12,index+1,3)]
    flip_call = dict(session_id="offline_host_fixture",reference_id="original_initial_scene",
        task="turn_over_in_same_pan",tool_template="demonstrated_spatula",initial_third_rgb=rgb_at(0,"third"),
        reset=True,calibration_id="flip_20260919_recorded_rgb_tcp_v1",tool_retained=True,
        scene_valid=True,supported_context=True,authorized_for_proposal=True,max_predictive_spread=[10.]*6)
    return dict(push=dict(observation=push_ob,call=push_call,inventory=inventory,
                        reference=read(push_root / "host_worker_check/result.json")["cases"][0]),
        flip=dict(observation=dict(samples=samples,now=samples[-1]["t"],observation_id="offline_host_0"),
                  call=flip_call,reference=reference["cases"][0]))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--gpu",type=int,required=True)
    args = p.parse_args()
    archive = ROOT / "real_robot/exports/pipeline_v1/pipeline_v1_bundle.tar.gz"
    manifest = read(archive.parent / "bundle_manifest.json")
    assert digest(archive) == manifest["archives"][archive.name]["sha256"]
    directory = Path(tempfile.mkdtemp(prefix="pipeline-relocated-"))
    with tarfile.open(archive,"r:gz") as stream:
        names = stream.getnames(); assert len(names) == len(set(names))
        stream.extractall(directory,filter="data")
    for name,entry in manifest["files"].items():
        assert digest(directory / name) == entry["sha256"]
    root = directory / "pipeline_v1_bundle"
    assert not (root / "real_robot/runs").exists()
    assert not (root / "real_robot/data.py").exists()
    fixture = directory / "host_fixture.json"
    fixture.write_text(json.dumps(wire(fixtures()),allow_nan=False))
    code = r'''
import sys,json
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);sys.path[:0]=[str(root),str(root/'src')]
from real_robot.deployment_pipeline import Policy
from real_robot.deployment_pipeline.common import ROOT,unwire,locations,verify
assert ROOT==root
f=unwire(json.loads(Path(sys.argv[3]).read_text()));cases=[]
for bundle,key,field in [('push_v5','push','delta_P_m'),('flip_egg_v1','flip','velocity')]:
    verify(*locations(bundle),require_assets=True)
    with Policy(bundle,gpu=int(sys.argv[2]),timeout_s=180) as p:
        row=f[key]
        result=p.act(row['observation'],row['call'],reset=True,executor_contract={})
        ref=row['reference'];assert result['status']==ref['status']
        error=float(np.max(np.abs(np.asarray(result['decision'][field])-np.asarray(ref['decision'][field]))))
        assert error<=1e-6,(bundle,error)
        assert result['executor_request']==ref['executor_request']
        interrupted=dict(row['call']);interrupted['interrupted' if key=='push' else 'interrupt']=True
        stopped=p.act(row['observation'],interrupted)
        assert stopped['status']=='interrupted' and stopped['decision'] is None
        extra={}
        if key=='push':
            scene=p.inventory(row['inventory'],'relocated-fixture')
            assert scene['status']=='proposals_require_association_and_geometry_validation'
            assert scene['instances'] and all(not x['geometry_complete'] for x in scene['instances'])
            extra=dict(SAM_inventory_proposals=len(scene['instances']),SAM_geometry_verified=False)
        cases.append(dict(bundle=bundle,original_worker_max_output_error=error,
            interruption_passed=True,unbound_executor_result=result['executor_request'],
            ready=p.ready,logs=str(p.output),**extra))
print(json.dumps(dict(passed=True,root=str(ROOT),models=cases,hardware_io=False)))
'''
    env = dict(PATH=os.defpath,LANG="C.UTF-8",PYTHONNOUSERSITE="1",PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
    result = subprocess.run([PIXI,"run","--manifest-path",str(ROOT / "real_robot/training_v4_environment/pixi.toml"),
        "--locked","--no-install","python","-I","-c",code,str(root),str(args.gpu),str(fixture)],
        cwd=root,env=env,text=True,capture_output=True,timeout=480)
    receipt = dict(date="2026-09-22",passed=result.returncode==0,archive_sha256=digest(archive),
        relocated_root=str(root),returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
        files_verified=len(manifest["files"]),training_data_bundled=False,synthetic_authorization_flags=True,
        API_calls=0,candidate_optimizer_updates=0,robot_actions_executed=0,
        environment_lock_sha256=digest(ROOT / "real_robot/training_v4_environment/pixi.lock"))
    if result.returncode==0:
        receipt["result"]=json.loads(result.stdout)
    atomic(ROOT / "real_robot/reports/PIPELINE_DEPLOYMENT_CHECK.json",receipt)
    print(json.dumps(receipt),flush=True)
    if result.returncode:
        raise RuntimeError("Relocated deployment check failed; see saved receipt")


if __name__ == "__main__":
    main()
