"""Paired-noise offline sensitivity of tray grasp commands to finger-state drift."""
import argparse
import json
import sys
import numpy as np
from appl.io import atomic,read,digest
from .run import BASE,settings


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--device-isolated',action='store_true');a=p.parse_args()
    if not a.device_isolated:
        from appl.gpu import launch
        return launch(a.gpu,sys.argv[1:],module='experiments.exp2.analysis.dp_diagnosis_20260919.conditioning')
    import torch
    from appl.gpu import verify_cuda
    from appl.prior_policies.data import vector
    from appl.prior_policies.engine import LoadedPolicy,sample
    torch.set_num_threads(2);device=verify_cuda();cfg,source,checkpoint,directory,ids=settings('tray_pack')
    if digest(checkpoint)!=read(BASE/'plan.json')['tasks']['tray_pack']['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    policy=LoadedPolicy(source,checkpoint,seed=0);demo=read(directory/(ids[0]+'.json'))
    path=BASE/'runs/tray_pack'/ids[0]/'frozen_DP'
    states=[read(path/'initial_state.json')]+[json.loads(l)['state'] for l in (path/'trace.jsonl').read_text().splitlines()]
    output=BASE/'conditioning';output.mkdir(exist_ok=False)
    variants=['observed','expert_same_time','replace_finger_qpos','replace_finger_qpos_qvel','replace_arm_qpos_qvel','replace_TCP_pose']
    atomic(output/'plan.json',dict(task='tray_pack',demonstration=ids[0],control_steps=list(range(8,65,8)),
        variants=variants,samples_per_cell=16,random_seed=913,
        interpretation='Offline counterfactual conditioning with identical diffusion noise for each variant. Hybrid observations may be physically inconsistent and are not a deployable controller.',
        source_sha256=digest(__file__),checkpoint_sha256=digest(checkpoint),device=device,optimizer_updates=0,simulator_steps=0,API_requests=0))
    results=[];arrays={}
    for t in range(8,65,8):
        actual=np.stack([vector(states[t-1]),vector(states[t])]);expert=np.stack([vector(demo['observations'][j]['state']) for j in [t-1,t]])
        for variant in variants:
            raw=actual.copy()
            if variant=='expert_same_time':raw=expert.copy()
            elif variant=='replace_finger_qpos':raw[:,7:9]=expert[:,7:9]
            elif variant=='replace_finger_qpos_qvel':raw[:,[7,8,16,17]]=expert[:,[7,8,16,17]]
            elif variant=='replace_arm_qpos_qvel':raw[:,:7]=expert[:,:7];raw[:,9:16]=expert[:,9:16]
            elif variant=='replace_TCP_pose':raw[:,18:25]=expert[:,18:25]
            batch=torch.as_tensor(np.repeat(raw[None],16,axis=0),device='cuda:0')
            prediction=sample(policy.model,batch,policy.spec,torch.Generator(device='cuda:0').manual_seed(913)).cpu().numpy()
            arrays[f'{t}_{variant}']=prediction
            results.append(dict(step=t,variant=variant,first_grip_mean=float(prediction[:,1,7].mean()),
                first_grip_negative_fraction=float((prediction[:,1,7]<0).mean()),
                executed_chunk_grip_mean=float(prediction[:,1:9,7].mean()),executed_chunk_negative_fraction=float((prediction[:,1:9,7]<0).mean())))
    np.savez_compressed(output/'predictions.npz',**arrays);atomic(output/'result.json',results)
    for r in results:
        if r['step'] in [32,40,48,56]:print(r,flush=True)


if __name__=='__main__':main()
