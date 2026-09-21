"""Compare frozen models' response to small finger-position errors before grasp."""
import argparse
import sys
import numpy as np
from appl.io import read,atomic,digest
from .run import BASE,settings


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--device-isolated',action='store_true');a=p.parse_args()
    if not a.device_isolated:
        from appl.gpu import launch
        return launch(a.gpu,sys.argv[1:],module='experiments.exp2.analysis.dp_diagnosis_20260919.grip_sensitivity')
    import torch
    from appl.gpu import verify_cuda
    from appl.prior_policies.data import vector
    torch.set_num_threads(2);device=verify_cuda();cfg,source,checkpoint,directory,ids=settings(a.task)
    assert digest(checkpoint)==read(BASE/'plan.json')['tasks'][a.task]['checkpoint_sha256']
    if a.task=='drawer_exchange':
        from appl.dp_baseline.train import LoadedPolicy,sample
        policy=LoadedPolicy(checkpoint,seed=0)
        sample_fn=lambda raw,g:sample(policy.model,policy.module,raw,policy.spec,g)
    else:
        from appl.prior_policies.engine import LoadedPolicy,sample
        policy=LoadedPolicy(source,checkpoint,seed=0)
        sample_fn=lambda raw,g:sample(policy.model,raw,policy.spec,g)
    output=BASE/'grip_sensitivity'/a.task;output.mkdir(parents=True,exist_ok=False)
    atomic(output/'plan.json',dict(task=a.task,demonstrations=[ids[i] for i in [0,5,11]],
        selection='32 actions before red/blue pick closing; drawer red/blue are closing events 2/3, other tasks 1/2.',
        total_finger_opening_reduction_mm=[0,1,2,4,8],samples_per_cell=8,seed=913,
        intervention='Subtract half the stated opening reduction from each observed finger position in both history frames. All other inputs unchanged; offline sensitivity only.',
        source_sha256=digest(__file__),checkpoint_sha256=digest(checkpoint),device=device,optimizer_updates=0,simulator_steps=0,API_requests=0))
    results=[];arrays={}
    for di in [0,5,11]:
        demo=read(directory/(ids[di]+'.json'));grips=np.asarray(demo['actions'])[:,-1]
        closes=np.flatnonzero((grips[1:]<0)&(grips[:-1]>=0))+1
        for phase,ci in [('red',1 if a.task=='drawer_exchange' else 0),('blue',2 if a.task=='drawer_exchange' else 1)]:
            t=int(closes[ci])-32;original=np.stack([vector(demo['observations'][j]['state']) for j in [t-1,t]])
            assert min(original[:,7:9].sum(1))>.075
            for mm in [0,1,2,4,8]:
                raw=original.copy();raw[:,7:9]-=mm/2000
                prediction=sample_fn(torch.as_tensor(np.repeat(raw[None],8,axis=0),device='cuda:0'),torch.Generator(device='cuda:0').manual_seed(913)).cpu().numpy()
                arrays[f'{ids[di]}_{phase}_{mm}']=prediction
                results.append(dict(demonstration=ids[di],phase=phase,step=t,reduction_mm=mm,
                    gripper_mean=float(prediction[:,1:9,7].mean()),negative_fraction=float((prediction[:,1:9,7]<0).mean())))
    np.savez_compressed(output/'predictions.npz',**arrays);atomic(output/'result.json',results)
    print(a.task,{mm:float(np.mean([r['gripper_mean'] for r in results if r['reduction_mm']==mm])) for mm in [0,1,2,4,8]},flush=True)


if __name__=='__main__':main()
