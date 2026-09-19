"""One native rollout implementation for diagnostics and frozen evaluations."""
from pathlib import Path
import time
import numpy as np
from .io import atomic,event,read,digest
from .envs.adapter import reset,step
from .envs.evaluator import measure,SuccessTracker


def episode(cfg,env,policy,directory,seed,*,variant='standard',offsets=None,skill=None,prefix=None):
    from PIL import Image
    directory=Path(directory)
    if (directory/'result.json').exists():return read(directory/'result.json')
    if directory.exists():
        # A partial physical execution has no exact snapshot. Keep a small
        # interruption receipt; a caller must select a new attempt directory.
        raise RuntimeError('Interrupted episode exists: reconcile under a new attempt ID')
    directory.mkdir(parents=True)
    obs,state=reset(env,seed,variant,offsets)
    atomic(directory/'initial_state.json',state)
    tracker=SuccessTracker(cfg['evaluation']);started=time.monotonic()
    if prefix is not None:
        for action in prefix:
            obs,state=step(env,action);tracker.update(measure(env,state,cfg['evaluation']))
        atomic(directory/'prefix.json',dict(steps=len(prefix),scope='diagnostic original-action prefix; no learned-policy credit',state=state))
    metrics=tracker.last;clipped=0;first_stages={};maximum_drawer=0.;status='timeout'
    space=env.unwrapped.single_action_space
    steps_limit=cfg['evaluation']['max_steps'] if skill is None else 600
    for index in range(steps_limit):
        proposed=np.asarray(policy.action(state),np.float32)
        if proposed.shape!=(8,) or not np.isfinite(proposed).all():raise ValueError('Invalid policy output')
        # The same declared physical actuator saturation applies to every method.
        command=np.clip(proposed,space.low,space.high); clipped+=int(np.any(command!=proposed))
        obs,state=step(env,command)
        metrics=tracker.update(measure(env,state,cfg['evaluation']))
        maximum_drawer=max(maximum_drawer,state['drawer_position'][0])
        for name in ('drawer_open','red_on_pad','blue_inside','gripper_released'):
            if metrics[name] and name not in first_stages:first_stages[name]=index+1
        event(directory/'trace.jsonl','step',step=index+1,state=state,action=command.tolist(),metrics=metrics)
        if index%20==0:
            Image.fromarray(obs['sensor_data']['front']['rgb'][0].cpu().numpy()).save(directory/f'frame_{index+1:04d}.png')
        if (metrics['success'] if skill is None else tracker.skill_succeeded(skill)):
            status='succeeded';break
    Image.fromarray(obs['sensor_data']['front']['rgb'][0].cpu().numpy()).save(directory/'final.png')
    result=dict(seed=seed,variant=variant,offsets=offsets,skill=skill,steps=index+1,status=status,
        success=status=='succeeded',final=metrics,elapsed_seconds=time.monotonic()-started,
        first_geometry_steps=first_stages,maximum_drawer=maximum_drawer,actuator_saturated_steps=clipped,
        inference_chunks=len(policy.timings),local_inference_seconds=sum(policy.timings),
        diagnostic_prefix_steps=0 if prefix is None else len(prefix),completed_evaluation=True)
    atomic(directory/'result.json',result);return result


def diagnostic_rollouts(cfg,checkpoint,output,seeds,*,candidate=None,skill=None,prefix_id=None,boundary=0):
    from .gpu import verify_cuda
    from .envs.native import make_environment
    from .train import LoadedPolicy
    verify_cuda();env=make_environment();results=[]
    try:
        prefix=None if prefix_id is None else read(cfg['data']['directory']/(prefix_id+'.json'))['actions'][:boundary]
        for seed in seeds:
            policy=LoadedPolicy(checkpoint,candidate,seed=seed)
            result=episode(cfg,env,policy,Path(output)/str(seed),seed,skill=skill,prefix=prefix)
            results.append(result);print({k:result[k] for k in ('seed','success','steps','maximum_drawer','status')},flush=True)
            del policy
        atomic(Path(output)/'results.json',results);return results
    finally:env.close()


def capability(cfg):
    from .gpu import verify_cuda
    from .worker import PolicyProcess
    from .envs.native import make_environment
    verify_cuda()
    base=cfg['output']/'api_capability';submission=read(base/'submission.json')
    candidate=base/'versions'/submission['version'];metadata=read(candidate/'candidate.json')
    checkpoint=Path(submission['training']['output'])/'last.pt'
    assert digest(checkpoint)==submission['training']['checkpoint_sha256']
    output=base/'native_capability'
    if (output/'result.json').exists():return read(output/'result.json')
    output.mkdir(parents=True,exist_ok=True)
    env=make_environment();policy=PolicyProcess(cfg,candidate,checkpoint,output/'policy_worker',int(__import__('os').environ['APPL_PHYSICAL_GPU']),1000)
    try:
        boundary=metadata['segments']['demo1000'][0]
        prefix=read(cfg['data']['directory']/'demo1000.json')['actions'][:boundary] if boundary else None
        result=episode(cfg,env,policy,output/'episode',1000,skill=metadata['skill'],prefix=prefix)
        result.update(candidate_version=submission['version'],actually_api_authored=True,formal_result=False,
                      current_skill_handoff_requirement='TCP z > .28 m, release and clearance; no extra controller actions')
        atomic(output/'result.json',result);return result
    finally:policy.close();env.close()
