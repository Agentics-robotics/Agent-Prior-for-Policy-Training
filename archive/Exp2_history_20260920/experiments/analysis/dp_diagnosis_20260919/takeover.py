"""Perfect expert prefix followed by unchanged frozen DP with real causal history."""
import argparse
import json
import sys
import time
import numpy as np
from appl.io import atomic,read,digest,source_manifest
from .run import BASE,settings,reset_environment,difference,snapshot,video


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',required=True)
    p.add_argument('--allow-prefix-contact-differences',action='store_true')
    p.add_argument('--indices',nargs='+',type=int,default=[0,5,11])
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--device-isolated',action='store_true');a=p.parse_args()
    if not a.device_isolated:
        from appl.gpu import launch
        return launch(a.gpu,sys.argv[1:],module='experiments.exp2.analysis.dp_diagnosis_20260919.takeover')
    import torch
    from appl.gpu import verify_cuda
    from appl.envs.adapter import step
    from appl.prior_policies.data import vector
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
    label='takeover_physical_replay' if a.allow_prefix_contact_differences else 'takeover'
    goal=read(cfg['completion_contract']);output=BASE/label/a.task;output.mkdir(parents=True,exist_ok=False)
    atomic(output/'plan.json',dict(task=a.task,demonstrations=[ids[i] for i in a.indices],
        intervention='Replay the exact expert prefix until 32 actions before the blue-pick closing transition (third closing for drawer, second for other tasks); DP receives the actual preceding and current states. No expert actions after takeover.',
        max_policy_steps=1500,device=device,diagnostic_source_sha256=digest(__file__),checkpoint_sha256=digest(checkpoint),
        policy_training_changes=False,runtime_API_requests=0,require_bitwise_prefix=not a.allow_prefix_contact_differences,
        note='Oracle-prefix diagnosis; not a deployable method or held-out performance estimate. Sampling is freshly seeded at takeover. The follow-up records contact replay differences rather than requiring exact equality; initial reset must still match. Previous guard-stopped prefixes are retained and did not execute policy actions.'))
    for di in a.indices:
        identifier=ids[di];demo=read(directory/(identifier+'.json'));seed=int(identifier.removeprefix('demo'))
        grips=np.asarray(demo['actions'])[:,-1]
        closes=np.flatnonzero((grips[1:]<0)&(grips[:-1]>=0))+1
        prefix=int(closes[2 if a.task=='drawer_exchange' else 1])-32
        if prefix<=0:raise ValueError('Invalid takeover prefix')
        root=output/identifier;root.mkdir();env=None
        try:
            env,obs,state=reset_environment(a.task,seed);errors=difference(state,demo['observations'][0]['state'])
            if max(errors.values())>1e-6:raise ValueError('Reset mismatch')
            frames=[snapshot(obs)];previous=state
            for t,action in enumerate(demo['actions'][:prefix],1):
                previous=state;obs,state=step(env,np.asarray(action,np.float32))
                if t%20==0:frames.append(snapshot(obs))
            errors=difference(state,demo['observations'][prefix]['state'])
            atomic(root/'takeover_state.json',dict(step=prefix,previous=previous,current=state,errors=errors))
            if max(errors.values())>1e-6 and not a.allow_prefix_contact_differences:raise ValueError('Prefix replay mismatch')
            policy.history.clear();policy.history.append(vector(previous));policy.queue.clear();policy.generator.manual_seed(seed);policy.timings.clear()
            start=time.monotonic();initial={k:state[k+'_pose'][2] for k in ['red','blue']};rise={k:0. for k in initial}
            with (root/'trace.jsonl').open('w') as stream:
                for t in range(1,1501):
                    raw=np.asarray(policy.action(state),np.float32);space=env.unwrapped.single_action_space
                    action=np.clip(raw,space.low,space.high);obs,state=step(env,action);metrics=measure(state,goal)
                    for k in rise:rise[k]=max(rise[k],state[k+'_pose'][2]-initial[k])
                    stream.write(json.dumps(dict(step=t,total_step=prefix+t,state=state,raw_action=raw.tolist(),action=action.tolist(),metrics=metrics))+'\n')
                    if t%20==0:frames.append(snapshot(obs))
                    if metrics['success']:break
                    if t%100==0:atomic(root/'progress.json',dict(policy_steps=t,elapsed=time.monotonic()-start))
            frames.append(snapshot(obs));video(frames,root)
            result=dict(task=a.task,demonstration=identifier,expert_prefix_steps=prefix,policy_steps=t,
                success=metrics['success'],final=metrics,maximum_object_rise_after_takeover=rise,
                elapsed_seconds=time.monotonic()-start,expert_prefix_max_state_error=max(errors.values()))
            atomic(root/'result.json',result);print(result,flush=True)
        except Exception as error:
            atomic(root/'failure.json',dict(error=str(error),automatic_retry=False));raise
        finally:
            if env is not None:env.close()


if __name__=='__main__':main()
