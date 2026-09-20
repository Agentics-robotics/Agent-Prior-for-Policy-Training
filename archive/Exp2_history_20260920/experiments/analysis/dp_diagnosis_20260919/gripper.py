"""One-factor test: execute the sign of the frozen DP's gripper command."""
import argparse
import json
import sys
import time
import numpy as np
from appl.io import atomic,read,digest,source_manifest
from .run import BASE,settings,reset_environment,difference,snapshot,video


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--device-isolated',action='store_true');a=p.parse_args()
    if not a.device_isolated:
        from appl.gpu import launch
        return launch(a.gpu,sys.argv[1:],module='experiments.exp2.analysis.dp_diagnosis_20260919.gripper')
    import torch
    from appl.gpu import verify_cuda
    from appl.envs.adapter import step
    from appl.scaleup.tasks import measure
    torch.set_num_threads(2);device=verify_cuda();cfg,source,checkpoint,directory,ids=settings(a.task)
    plan=read(BASE/'plan.json')
    if source_manifest()!=plan['framework_source'] or digest(checkpoint)!=plan['tasks'][a.task]['checkpoint_sha256']:raise ValueError('Frozen inputs changed')
    if a.task=='drawer_exchange':
        from appl.dp_baseline.train import LoadedPolicy
        policy=LoadedPolicy(checkpoint,seed=0)
    else:
        from appl.prior_policies.engine import LoadedPolicy
        policy=LoadedPolicy(source,checkpoint,seed=0)
    goal=read(cfg['completion_contract']);output=BASE/'binary_gripper'/a.task;output.mkdir(parents=True,exist_ok=False)
    atomic(output/'plan.json',dict(task=a.task,demonstrations=[ids[i] for i in [0,5,11]],
        intervention='Only action[7] changes: +1 if frozen model action[7]>=0, else -1. Seven arm targets, model weights, observations, DDPM100, eight-step queue and reset/diffusion seed are unchanged.',
        max_steps=1500,device=device,source_sha256=digest(__file__),checkpoint_sha256=digest(checkpoint),
        training_updates=0,runtime_API_requests=0,note='Adaptive diagnostic motivated by progressive open-finger drift observed in the original reset traces; not a new held-out score.'))
    for di in [0,5,11]:
        identifier=ids[di];demo=read(directory/(identifier+'.json'));seed=int(identifier.removeprefix('demo'))
        root=output/identifier;root.mkdir();env=None
        try:
            env,obs,state=reset_environment(a.task,seed);errors=difference(state,demo['observations'][0]['state'])
            if max(errors.values())>1e-6:raise ValueError('Reset mismatch')
            frames=[snapshot(obs)];policy.history.clear();policy.queue.clear();policy.generator.manual_seed(seed);policy.timings.clear()
            start=time.monotonic();initial={k:state[k+'_pose'][2] for k in ['red','blue']};rise={k:0. for k in initial}
            changes=0;first_goals={}
            with (root/'trace.jsonl').open('w') as stream:
                for t in range(1,1501):
                    raw=np.asarray(policy.action(state),np.float32);space=env.unwrapped.single_action_space
                    action=np.clip(raw,space.low,space.high);action[7]=1. if raw[7]>=0 else -1.;changes+=int(action[7]!=raw[7])
                    obs,state=step(env,action);metrics=measure(state,goal)
                    for k in rise:rise[k]=max(rise[k],state[k+'_pose'][2]-initial[k])
                    for k,v in metrics.items():
                        if v and k not in first_goals:first_goals[k]=t
                    stream.write(json.dumps(dict(step=t,state=state,raw_action=raw.tolist(),action=action.tolist(),metrics=metrics))+'\n')
                    if t%20==0:frames.append(snapshot(obs))
                    if metrics['success']:break
                    if t%100==0:atomic(root/'progress.json',dict(steps=t,elapsed=time.monotonic()-start))
            frames.append(snapshot(obs));video(frames,root)
            result=dict(task=a.task,demonstration=identifier,steps=t,success=metrics['success'],final=metrics,
                first_goals=first_goals,maximum_object_rise=rise,changed_gripper_commands=changes,
                elapsed_seconds=time.monotonic()-start,initial_max_error=max(errors.values()))
            atomic(root/'result.json',result);print(result,flush=True)
        except Exception as error:
            atomic(root/'failure.json',dict(error=str(error),automatic_retry=False));raise
        finally:
            if env is not None:env.close()


if __name__=='__main__':main()
