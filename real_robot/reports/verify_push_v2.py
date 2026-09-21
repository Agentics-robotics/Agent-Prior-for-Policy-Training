"""One relocated package interface check; no rollout, scoring or robot IO."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from real_robot.deployment.common import ROOT, digest, wire
from real_robot.reports.profile_push_latency import inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    archive = ROOT / "real_robot/exports/push_v2/push_v2_bundle.tar.gz"
    directory = Path(tempfile.mkdtemp(prefix="push-v2-interface-"))
    with tarfile.open(archive, "r:gz") as stream:
        stream.extractall(directory, filter="data")
    root = directory / "push_v2_bundle"
    assert not (root / "real_robot/runs").exists()
    assert not (root / "real_robot/data.py").exists()
    # A current real paired observation is host-side interface input only.
    # The artificial unchanged-placement goal below is not a manipulation test.
    observation = inputs(300, 1)[0]
    fixture = directory / "host_input.json"
    fixture.write_text(json.dumps(wire(observation), allow_nan=False))
    code = r"""
import json,sys
import numpy as np
from real_robot.deployment.common import ROOT, unwire
from real_robot.deployment_v2 import PushPolicy
ob=unwire(json.load(open(sys.argv[2])))
out=[]
for name in ('contour_push','visual_push'):
    with PushPolicy(name,gpu=int(sys.argv[1])) as p:
        empty=p.observe_scene({},'invalid-input-check')
        assert empty['status']=='unavailable' and not empty['instances']
        scene=p.observe_scene(ob,'interface-check')
        assert scene['instances'], scene
        ref=max(scene['instances'],key=lambda x:x['confidence'])
        goal={'frame':'third_undistorted_original_pixels','center_uv':ref['center_uv'],
              'theta_deg_clockwise':0.,'accept_similarity_approximation':True}
        workspace=[[0,0],[1279,0],[1279,719],[0,719]]
        preview=p.preview_goal(ref,goal,workspace)
        assert preview['foreground_half'].shape==(360,640)
        converted=p.reexpress_goal(ref,preview['foreground_half'])
        assert converted['status']=='goal_reexpressed', converted
        call={'call_id':'interface-check','scene_version':'interface-check',
              'instance_id':ref['instance_id'],'template':ref,'spatial_goal':goal,
              'workspace_polygon_uv':workspace,'tolerances':{'position_px':8.,
              'angle_deg':10.,'foreground_iou':.7,'stability_s':.5,'neighbor_motion_px':20.},
              'max_duration_s':30.}
        result=p.act(ob,call,reset=True)
        assert result['status'] and result['diagnostics']
        action=result['action']
        assert action is None or (np.asarray(action).shape==(7,) and np.isfinite(action).all())
        gate=p.decode_action(np.zeros(7),None)
        assert gate['enabled'] is False and gate['command'] is None
        assert all(x['hardware_io'] is False for x in (empty,scene,preview,converted,result,gate))
        out.append(dict(policy_id=name,ready=p.ready,instances=len(scene['instances']),
                        goal_preview_shape=list(preview['foreground_half'].shape),
                        goal_reexpression_status=converted['status'],act_status=result['status'],
                        finite_candidate=action is not None,unverified_controller_refused=True,
                        logs=str(p.output)))
print(json.dumps(dict(root=str(ROOT),models=out,robot_actions_executed=0)))
"""
    env = dict(
        PATH=os.defpath,
        LANG="C.UTF-8",
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(root) + ":" + str(root / "src"),
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(args.gpu), str(fixture)],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=300,
    )
    receipt = dict(
        archive_sha256=digest(archive),
        relocated_root=str(root),
        status="passed" if result.returncode == 0 else "failed",
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        paired_input=dict(trajectory="episode_2026091922002701", sample_index=300),
        fixture_goal="Unchanged current foreground, interface exercise only",
        training_runs_present=False,
        original_data_reader_present=False,
        API_calls=0,
        candidate_optimizer_updates=0,
        performance_evaluation=False,
        robot_actions_executed=0,
    )
    if result.returncode == 0:
        receipt["result"] = json.loads(result.stdout)
        assert receipt["result"]["root"] == str(root)
    path = ROOT / "real_robot/reports/PUSH_TRAINING_V2_INTERFACE_CHECK.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if result.returncode:
        raise RuntimeError("V2 interface check failed; inspect retained receipt")


if __name__ == "__main__":
    main()
