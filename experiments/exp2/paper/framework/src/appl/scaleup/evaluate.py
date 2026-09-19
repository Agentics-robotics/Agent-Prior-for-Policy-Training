"""One fresh paired-layout trial, with common goals and private physical budget."""
import os
import time
import numpy as np
from PIL import Image
from ..io import ROOT,read,atomic,digest,event,source_manifest,object_hash
from ..envs.adapter import step
from ..prior_policies.deploy import DeploymentTools,library,make_evaluation_environment
from ..prior_policies.feedback import PROMPT as DRAWER_PROMPT
from ..prior_policies.design import client
from ..agent import AgentLoop
from .protocol import BASE,config,initial_layout,naive_paths
from .tasks import measure


def prompt(name):
    if name=='drawer_exchange':return DRAWER_PROMPT
    value=DRAWER_PROMPT.replace('for the drawer task.','for the supplied tabletop manipulation task.')
    value=value.replace('drawer_open, red_on_pad, blue_inside and success are the supplied predicates as',
        'red_at_goal, blue_at_goal and success are the supplied predicates as')
    value=value.replace('TCP-object relation/motion and drawer state','TCP-object relation/motion and fixture geometry')
    value=value.replace('object displacement or disturbed drawer.','object displacement or fixture contact.')
    value=value.replace('Task success requires only simultaneous drawer-open, red-on-pad and blue-inside,',
        'Task success requires only simultaneous red_at_goal and blue_at_goal,')
    return value+'\nThis task has no drawer: drawer_position and drawer_velocity are zero compatibility channels. The full state includes the actual red_goal and blue_goal world coordinates.\n'


def environment(name,seed,condition):
    if name!='drawer_exchange':
        from .environment import make,reset
        env=make(read(BASE/name/'task.json'));obs,state=reset(env,seed,condition)
    else:
        from ..envs.adapter import reset
        from ..envs.scene import initial_state
        env=make_evaluation_environment(5000);original=initial_state(seed);desired=initial_layout(name,seed,condition)
        offsets=np.concatenate([np.asarray(desired[k][:2])-original[k][:2] for k in ('red','blue')]).tolist()
        obs,state=reset(env,seed,offsets=offsets)
    return env,obs,state


def evaluate(name,method,condition,seed):
    from ..gpu import verify_cuda
    cfg=config(name);study=read(BASE/'study_freeze.json');frozen=study['study']['tasks'][name]
    if object_hash(source_manifest())!=study['study']['framework_source']:raise ValueError('Frozen framework source changed')
    if method not in ('naive_DP','APPL') or condition not in ('ID','OOD'):raise ValueError('Invalid matrix cell')
    planned=[v for v in frozen['layouts'][condition] if v['seed']==seed]
    if len(planned)!=1 or planned[0]['layout']!=initial_layout(name,seed,condition):raise ValueError('Unplanned layout')
    for path,key in [(cfg['_path'],'configuration_sha256'),(BASE/name/'task.json','task_spec_sha256'),
        (cfg['completion_contract'],'completion_contract_sha256'),(cfg['output']/'normalization.json','normalization_sha256')]:
        if digest(path)!=frozen[key]:raise ValueError('Frozen input changed: '+str(path))
    root=BASE/name/'evaluation'/method/condition/str(seed)
    if (root/'result.json').exists():return read(root/'result.json')
    if root.exists():raise ValueError('Interrupted physical attempt retained; no implicit retry')
    policies=library(cfg) if method=='APPL' else None
    if policies is not None:
        actual={k:dict(version=p['version'],checkpoint_sha256=p['checkpoint_sha256']) for k,p in policies.items()}
        if actual!=frozen['policies']:raise ValueError('Frozen policy library changed')
    source,checkpoint=naive_paths(name)
    if method=='naive_DP' and digest(checkpoint)!=frozen['naive_checkpoint_sha256']:raise ValueError('Frozen baseline changed')
    root.mkdir(parents=True);atomic(root/'plan.json',dict(task_id=name,method=method,condition=condition,
        seed=seed,layout=planned[0]['layout'],device=verify_cuda(),study_sha256=study['study_sha256'],max_steps=5000))
    env=None;tools=None;steps=0;started=time.monotonic();contract=read(cfg['completion_contract'])
    try:
        env,obs,state=environment(name,seed,condition)
        limits=[];wrapper=env
        while hasattr(wrapper,'env'):
            if '_max_episode_steps' in vars(wrapper):limits.append(wrapper._max_episode_steps)
            wrapper=wrapper.env
        if not limits or any(n!=5000 for n in limits):raise ValueError('Environment physical cap mismatch')
        atomic(root/'environment.json',dict(time_limits=limits));atomic(root/'initial_state.json',state)
        other='APPL' if method=='naive_DP' else 'naive_DP'
        counterpart=BASE/name/'evaluation'/other/condition/str(seed)/'initial_state.json'
        if counterpart.exists() and read(counterpart)!=state:raise ValueError('Paired methods have different reset observations')
        Image.fromarray(obs['sensor_data']['front']['rgb'][0].cpu().numpy()).save(root/'initial.png')
        if method=='APPL':
            tools=DeploymentTools(cfg,env,obs,state,policies,root,seed,goal_measure=measure,
                goal_names=None if name=='drawer_exchange' else ['red_at_goal','blue_at_goal','success'])
            message=dict(completion_contract=contract,observation=tools.visible(),
                policy_catalogue=[dict(**v['metadata'],source_heuristic=v['source_heuristic']) for v in policies.values()])
            atomic(root/'prompt.json',dict(prompt=prompt(name),initial_input=message))
            AgentLoop(tools.j,tools,client(cfg),prompt(name),context_builder=tools.context_input).run(message)
            obs,state,steps=tools.obs,tools.state,tools.steps
            status='physical_budget_exhausted' if steps>=5000 else 'agent_finished'
            details=dict(invocations=len(tools.calls),actuator_saturated_steps=tools.saturated)
        else:
            if name=='drawer_exchange':
                from ..dp_baseline.train import LoadedPolicy
                policy=LoadedPolicy(checkpoint,seed=seed)
            else:
                from ..prior_policies.engine import LoadedPolicy
                policy=LoadedPolicy(source,checkpoint,seed)
            saturated=0;space=env.unwrapped.single_action_space
            for steps in range(1,5001):
                raw=np.asarray(policy.action(state),np.float32)
                if raw.shape!=(8,) or not np.isfinite(raw).all():raise ValueError('Invalid DP action')
                action=np.clip(raw,space.low,space.high);saturated+=int(np.any(raw!=action))
                obs,state=step(env,action);metrics=measure(state,contract)
                event(root/'trace.jsonl','step',step=steps,policy_id='naive_DP',state=state,
                    raw_action=raw.tolist(),action=action.tolist(),metrics=metrics)
                if steps%20==1:Image.fromarray(obs['sensor_data']['front']['rgb'][0].cpu().numpy()).save(root/f'frame_{steps:04d}.png')
                if metrics['success']:break
            status='physical_budget_exhausted';details=dict(actuator_saturated_steps=saturated,
                inference_chunks=len(policy.timings),inference_seconds=sum(policy.timings))
        metrics=measure(state,contract)
        Image.fromarray(obs['sensor_data']['front']['rgb'][0].cpu().numpy()).save(root/'final.png')
        result=dict(task_id=name,method=method,condition=condition,seed=seed,steps=steps,
            success=metrics['success'],final=metrics,status='succeeded' if metrics['success'] else status,
            completed_evaluation=True,elapsed_seconds=time.monotonic()-started,**details)
        atomic(root/'result.json',result);print(result,flush=True);return result
    except Exception as error:
        atomic(root/'failure.json',dict(error=str(error),physical_steps=steps if tools is None else tools.steps,
            automatic_retry=False,completed_evaluation=False,elapsed_seconds=time.monotonic()-started));raise
    finally:
        os.environ.pop('APPL_PRIOR_TOKEN',None)
        if tools is not None:
            try:
                for worker in tools.workers.values():worker.close()
            finally:tools.j.db.close()
        if env is not None:env.close()
