"""One fixed learned policy per trial, with no inference-time API client."""
import time
import numpy as np
from PIL import Image

from appl.envs.adapter import step
from appl.gpu import verify_cuda
from appl.io import atomic, digest, event, read
from appl.prior_policies.runner import PolicyProcess
from appl.scaleup.evaluate import environment
from appl.scaleup.protocol import initial_layout
from appl.scaleup.report import audit_episode, video
from appl.scaleup.tasks import measure
from .common import BASE, OLD, METHOD, config, folder, framework


def deployment_check(name):
    cfg=config(name);f=folder(name);out=f/'deployment_check'
    if (out/'result.json').exists(): return read(out/'result.json')
    if out.exists(): raise RuntimeError('Retained partial deployment check; no implicit restart')
    out.mkdir();worker=None
    try:
        ds=read(f/'assignment.json')['dataset'];s=read(ds)['segments'][0]
        from pathlib import Path
        original=Path(ds).parent/s['file'];observations=read(original)['observations'][:9]
        atomic(out/'inputs.json',dict(source=str(original),sha256=digest(original),observations=observations,seed=913))
        worker=PolicyProcess(cfg,f,out/'worker',913);actions=[]
        for index,obs in enumerate(observations):
            a=worker.action(obs['state'],reset=index==0)
            if np.shape(a)!=(8,) or not np.isfinite(a).all(): raise ValueError('Invalid B=1 deployment action')
            actions.append(a)
        worker.close();worker=None
        closed=read(out/'worker/closed.json')
        if closed['inference_chunks']!=2 or closed['returncode']!=0: raise ValueError('B=1 worker protocol mismatch')
        record=dict(passed=True,checkpoint_sha256=digest(f/'training/last.pt'),actual_worker=True,
            generated_actions=9,denoising_chunks=2,training_observations_only=True,
            API_calls=0,optimizer_updates=0,simulator_steps=0,device=verify_cuda())
        atomic(out/'actions.json',actions);atomic(out/'result.json',record)
        return record
    except Exception as error:
        atomic(out/'failure.json',dict(error=str(error),automatic_retry=False));raise
    finally:
        if worker is not None: worker.close()


def evaluate(name,condition,seed):
    cfg=config(name);f=folder(name);freeze=read(BASE/'study_freeze.json');study=freeze['study']
    if framework()!=study['framework_files']: raise ValueError('Frozen executor changed')
    if digest(BASE/'preparation.json')!=study['preparation_sha256']: raise ValueError('Frozen preparation changed')
    task=study['tasks'][name]
    if condition not in ('ID','OOD'): raise ValueError('Unknown condition')
    planned=[r for r in task['layouts'][condition] if r['seed']==seed]
    if len(planned)!=1 or planned[0]['layout']!=initial_layout(name,seed,condition):
        raise ValueError('Unplanned test layout')
    prepared=read(BASE/'preparation.json')['tasks'][name]
    for p,key in ((cfg['_path'],'configuration_sha256'),(cfg['completion_contract'],'completion_contract_sha256'),
                  (cfg['output']/'normalization.json','normalization_sha256'),(OLD/name/'task.json','task_spec_sha256')):
        if digest(p)!=prepared[key]: raise ValueError('Frozen evaluation input changed')
    if digest(f/'training/last.pt')!=task['checkpoint_sha256']: raise ValueError('Frozen checkpoint changed')
    for filename,sha in task['source_hashes'].items():
        if digest(f/'source'/filename)!=sha: raise ValueError('Frozen API source changed')
    root=BASE/name/'evaluation'/METHOD/condition/str(seed)
    if (root/'result.json').exists(): return read(root/'result.json')
    if root.exists(): raise RuntimeError('Existing physical attempt retained; no implicit retry')
    root.mkdir(parents=True)
    atomic(root/'plan.json',dict(task_id=name,method=METHOD,condition=condition,seed=seed,
        layout=planned[0]['layout'],device=verify_cuda(),study_sha256=freeze['study_sha256'],max_steps=5000,
        checkpoint_sha256=task['checkpoint_sha256'],deployment_API_calls=0))
    env=None;worker=None;steps=0;started=time.monotonic()
    try:
        env,obs,state=environment(name,seed,condition)
        limits=[];wrapper=env
        while hasattr(wrapper,'env'):
            if '_max_episode_steps' in vars(wrapper): limits.append(wrapper._max_episode_steps)
            wrapper=wrapper.env
        if not limits or any(n!=5000 for n in limits): raise ValueError('Environment cap mismatch')
        atomic(root/'environment.json',dict(time_limits=limits));atomic(root/'initial_state.json',state)
        reference=OLD/name/'evaluation/naive_DP'/condition/str(seed)/'initial_state.json'
        if state!=read(reference): raise ValueError('Original paired reset differs')
        Image.fromarray(obs['sensor_data']['front']['rgb'][0].cpu().numpy()).save(root/'initial.png')
        worker=PolicyProcess(cfg,f,root/'worker',seed)
        contract=read(cfg['completion_contract']);space=env.unwrapped.single_action_space;saturated=0
        for steps in range(1,5001):
            raw=np.asarray(worker.action(state,reset=steps==1),np.float32)
            if raw.shape!=(8,) or not np.isfinite(raw).all(): raise ValueError('Invalid learned action')
            action=np.clip(raw,space.low,space.high);saturated+=int(np.any(raw!=action))
            obs,state=step(env,action);metrics=measure(state,contract)
            event(root/'trace.jsonl','step',step=steps,policy_id=METHOD,state=state,
                raw_action=raw.tolist(),action=action.tolist(),metrics=metrics)
            if steps%20==1:
                Image.fromarray(obs['sensor_data']['front']['rgb'][0].cpu().numpy()).save(root/f'frame_{steps:04d}.png')
            if metrics['success']: break
        Image.fromarray(obs['sensor_data']['front']['rgb'][0].cpu().numpy()).save(root/'final.png')
        worker.close();worker=None
        closed=read(root/'worker/closed.json')
        result=dict(task_id=name,method=METHOD,condition=condition,seed=seed,steps=steps,
            success=metrics['success'],final=metrics,status='succeeded' if metrics['success'] else 'physical_budget_exhausted',
            completed_evaluation=True,elapsed_seconds=time.monotonic()-started,
            actuator_saturated_steps=saturated,inference_chunks=closed['inference_chunks'],
            inference_seconds=closed['inference_seconds'],deployment_API_calls=0)
        atomic(root/'result.json',result)
        audit=audit_episode(root,contract,result)
        atomic(root/'audit.json',dict(passed=True,**audit,paired_initial_state_sha256=digest(reference),
            trace_sha256=digest(root/'trace.jsonl'),checkpoint_sha256=task['checkpoint_sha256'],deployment_API_calls=0))
        print(dict(task=name,condition=condition,seed=seed,success=result['success'],steps=steps),flush=True)
    except Exception as error:
        atomic(root/'failure.json',dict(error=str(error),physical_steps=steps,
            completed_evaluation=False,automatic_retry=False));raise
    finally:
        if worker is not None: worker.close()
        if env is not None: env.close()
    video(root)
    return result
