"""Conditional reconstruction on fixed training-state histories, frozen weights."""
import argparse
from pathlib import Path
import sys
import time
from .run import BASE, settings
from appl.io import atomic,read,digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--device-isolated',action='store_true')
    a=p.parse_args()
    if not a.device_isolated:
        from appl.gpu import launch
        return launch(a.gpu,sys.argv[1:],module='experiments.exp2.analysis.dp_diagnosis_20260919.teacher')
    import numpy as np
    import torch
    from appl.gpu import verify_cuda
    from appl.prior_policies.data import vector
    from appl.public import normalize_action
    torch.set_num_threads(2);device=verify_cuda();cfg,source,checkpoint,directory,ids=settings(a.task)
    expected=read(BASE/'plan.json')['tasks'][a.task]['checkpoint_sha256']
    if digest(checkpoint)!=expected:raise ValueError('Frozen checkpoint changed')
    stored=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if a.task=='drawer_exchange':
        from appl.dp_baseline.train import LoadedPolicy,sample,scheduler
        policy=LoadedPolicy(checkpoint,seed=913)
        sample_fn=lambda raw,g:sample(policy.model,policy.module,raw,policy.spec,g)
    else:
        from appl.prior_policies.engine import LoadedPolicy,sample,scheduler
        policy=LoadedPolicy(source,checkpoint,seed=913)
        sample_fn=lambda raw,g:sample(policy.model,raw,policy.spec,g)
    histories=[];targets=[];masks=[];indices=[];events=[]
    for di in [0,5,11]:
        identifier=ids[di];demo=read(directory/(identifier+'.json'));actions=np.asarray(demo['actions'],np.float32)
        obs=np.asarray([vector(o['state']) for o in demo['observations']]);length=len(actions)
        switches=np.flatnonzero(np.sign(actions[1:,-1])!=np.sign(actions[:-1,-1]))+1
        selection=sorted(set(np.linspace(0,length-1,32).astype(int).tolist()) |
            {int(t+d) for t in switches for d in range(-4,5) if 0<=t+d<length})
        for t in selection:
            ai=np.arange(t-1,t+15);histories.append(obs[np.maximum([t-1,t],0)])
            targets.append(actions[np.clip(ai,0,length-1)]);masks.append((ai>=0)&(ai<length))
            indices.append(dict(demonstration=identifier,t=int(t)));events.append(bool(np.any(abs(switches-t)<=4)))
    histories=np.asarray(histories);targets=np.asarray(targets);masks=np.asarray(masks);events=np.asarray(events)
    output=BASE/'teacher'/a.task;output.mkdir(parents=True,exist_ok=False)
    atomic(output/'plan.json',dict(demonstrations=[ids[i] for i in [0,5,11]],selection='32 uniformly spaced indices plus +/-4 indices around every gripper sign change',
        checkpoint_sha256=expected,samples=len(indices),batch_size=16,device=device,source_sha256=digest(__file__),
        caveat='Training-state conditional generation, not closed-loop success. Batch sampling has a fixed RNG but is not the B=1 rollout stream.'))
    results={};predictions={};start=time.monotonic()
    for weights in ['ema','model']:
        policy.model.load_state_dict(stored[weights]);policy.model.eval()
        chunks=[];generator=torch.Generator(device='cuda:0').manual_seed(913)
        for i in range(0,len(histories),16):
            raw=torch.as_tensor(histories[i:i+16],device='cuda:0')
            chunks.append(sample_fn(raw,generator).cpu().numpy())
        predicted=np.concatenate(chunks);predictions[weights]=predicted
        groups={}
        for label,select in [('all',np.ones(len(indices),bool)),('grip_transition_neighborhood',events),('other',~events)]:
            error=predicted[select,1:9]-targets[select,1:9];valid=masks[select,1:9]
            joints=error[:,:,:7][valid];grip=error[:,:,7][valid]
            sign=np.sign(predicted[select,1:9,7])[valid]==np.sign(targets[select,1:9,7])[valid]
            groups[label]=dict(windows=int(select.sum()),joint_RMSE_radians=float(np.sqrt(np.mean(joints**2))),
                joint_abs_error_p95_radians=float(np.quantile(np.abs(joints),.95)),gripper_RMSE=float(np.sqrt(np.mean(grip**2))),
                gripper_sign_accuracy=float(sign.mean()),current_joint_RMSE_radians=float(np.sqrt(np.mean(error[:,0,:7]**2))))
        results[weights]=groups
    np.savez_compressed(output/'predictions.npz',history=histories,target=targets,mask=masks,event=events,**predictions)
    atomic(output/'indices.json',indices)
    atomic(output/'result.json',dict(task=a.task,results=results,samples=len(indices),elapsed_seconds=time.monotonic()-start,
        new_optimizer_updates=0,new_simulator_steps=0,new_runtime_API_requests=0))
    print(a.task,results,flush=True)


if __name__=='__main__':main()
