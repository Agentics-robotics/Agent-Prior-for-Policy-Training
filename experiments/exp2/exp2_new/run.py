"""Foreground-supervised inference, with one active physical episode at a time."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from appl.io import ROOT, PIXI, MANIFEST, atomic, digest, read
from .common import BASE, OLD, MODULE, TASKS, METHOD, config, episode_root, prepare, verify, policy_identity
from .budget import BudgetClient, BudgetStop, Ledger


def binary_step(original_step):
    """Correct the executor's already-clipped action in place, before physics.

    DeploymentTools records this same array after step(). Its raw action remains
    independent. With native [-1,1] gripper bounds, clipping preserves the sign.
    All original policy-selection, history, feedback and stopping logic is reused.
    """
    def step(env, action):
        space=env.unwrapped.single_action_space
        if space.low[7]!=-1 or space.high[7]!=1:
            raise ValueError('Binary decoder requires original native gripper bounds')
        action[7]=1.0 if action[7]>=0 else -1.0
        return original_step(env,action)
    return step


def check(task):
    import torch
    from appl.gpu import verify_cuda
    from appl.prior_policies.deploy import library
    from appl.prior_policies.runner import PolicyProcess
    torch.set_num_threads(2);verify(task);cfg=config(task)
    for policy_id,p in library(cfg).items():
        root=BASE/'checks'/task/policy_id;root.mkdir(parents=True,exist_ok=False)
        old=OLD/task/'inference_checks'/policy_id
        inputs=read(old/'inputs.json');expected=read(old/'actions.json');worker=None
        try:
            worker=PolicyProcess(cfg,p['folder'],root/'worker',inputs['seed'])
            actions=[worker.action(o['state'],reset=i==0) for i,o in enumerate(inputs['observations'])]
            error=float(np.max(np.abs(np.asarray(actions)-np.asarray(expected))))
            if len(actions)!=9 or not np.isfinite(actions).all() or error>1e-6:
                raise ValueError('Frozen policy deployment reproduction failed')
            worker.close();worker=None
            atomic(root/'actions.json',actions)
            atomic(root/'result.json',dict(passed=True,max_action_error=error,actions=9,
                checkpoint_sha256=p['checkpoint_sha256'],device=verify_cuda(),API_calls=0,
                physical_steps=0,optimizer_updates=0,reference_inputs_sha256=digest(old/'inputs.json'),
                reference_actions_sha256=digest(old/'actions.json')))
        finally:
            if worker is not None:worker.close()
    print(dict(checked=task,policies=len(library(cfg))),flush=True)


def evaluate(cell):
    import torch
    from PIL import Image
    from appl.gpu import verify_cuda
    from appl.agent import AgentLoop
    from appl.prior_policies import deploy
    from appl.scaleup.evaluate import environment, prompt
    from appl.scaleup.tasks import measure
    from .report import audit, video
    torch.set_num_threads(2);plan=verify(cell['task']);cfg=config(cell['task'])
    policies=deploy.library(cfg)
    if policy_identity(policies)!=plan['tasks'][cell['task']]['policies']:
        raise ValueError('Frozen library changed')
    for identifier in policies:
        check_result=read(BASE/'checks'/cell['task']/identifier/'result.json')
        if not check_result['passed']: raise ValueError('Unchecked policy')
    root=episode_root(cell);root.mkdir(parents=True,exist_ok=False)
    atomic(root/'plan.json',dict(**cell,study_sha256=digest(BASE/'plan.json'),device=verify_cuda(),
                                method=METHOD,max_steps=5000,decoder='sign_zero_opens'))
    env=None;tools=None;error=None;started=time.monotonic();original_step=deploy.step
    deploy.step=binary_step(original_step)
    try:
        env,obs,state=environment(cell['task'],cell['seed'],cell['condition'])
        for reference in cell['baseline_roots'].values():
            initial=Path(reference)/'initial_state.json'
            if digest(initial)!=cell['initial_sha256'] or read(initial)!=state:
                raise ValueError('Paired reset observation changed')
        limits=[];wrapper=env
        while hasattr(wrapper,'env'):
            if '_max_episode_steps' in vars(wrapper): limits.append(wrapper._max_episode_steps)
            wrapper=wrapper.env
        if not limits or any(n!=5000 for n in limits):raise ValueError('Physical cap mismatch')
        space=env.unwrapped.single_action_space
        atomic(root/'environment.json',dict(time_limits=limits,action_low=space.low.tolist(),action_high=space.high.tolist()))
        atomic(root/'initial_state.json',state)
        Image.fromarray(obs['sensor_data']['front']['rgb'][0].cpu().numpy()).save(root/'initial.png')
        tools=deploy.DeploymentTools(cfg,env,obs,state,policies,root,cell['seed'],goal_measure=measure,
              goal_names=None if cell['task']=='drawer_exchange' else ['red_at_goal','blue_at_goal','success'])
        message=dict(completion_contract=tools.contract,observation=tools.visible(),
                     policy_catalogue=[dict(**p['metadata'],source_heuristic=p['source_heuristic']) for p in policies.values()])
        instructions=prompt(cell['task'])
        if instructions!=plan['tasks'][cell['task']]['prompt']:raise ValueError('Original prompt changed')
        atomic(root/'prompt.json',dict(prompt=instructions,initial_input=message,tools=tools.schemas()))
        AgentLoop(tools.j,tools,BudgetClient(tools.j),instructions,context_builder=tools.context_input).run(message)
        metrics=measure(tools.state,tools.contract)
        if not tools.done():raise ValueError('Evaluation returned before a terminal condition')
        result=dict(task=cell['task'],method=METHOD,condition=cell['condition'],seed=cell['seed'],
                    steps=tools.steps,success=metrics['success'],final=metrics,completed_evaluation=True,
                    status='succeeded' if metrics['success'] else ('physical_budget_exhausted' if tools.steps>=5000 else 'agent_finished'),
                    invocations=len(tools.calls),actuator_saturated_steps=tools.saturated,
                    elapsed_seconds=time.monotonic()-started,learned_optimizer_updates=0)
        atomic(root/'result.json',result)
    except Exception as exc:
        error=exc
        reason=str(exc) if isinstance(exc,(RuntimeError,ValueError)) else type(exc).__name__
        atomic(root/'interruption.json',dict(reason=reason,kind='budget' if isinstance(exc,BudgetStop) else 'execution_or_provider',
               steps=0 if tools is None else tools.steps,completed_evaluation=False,success=None,automatic_retry=False))
        Ledger().stop(reason)
    finally:
        deploy.step=original_step
        if tools is not None:
            Image.fromarray(tools.obs['sensor_data']['front']['rgb'][0].cpu().numpy()).save(root/'final.png')
            cleanup=[]
            for worker in tools.workers.values():
                try:worker.close()
                except Exception as exc:cleanup.append(type(exc).__name__)
            tools.j.db.execute('PRAGMA wal_checkpoint(TRUNCATE)');tools.j.db.close()
            if cleanup:
                atomic(root/'cleanup_failure.json',dict(errors=cleanup))
                error=RuntimeError('Policy worker cleanup failed');Ledger().stop(str(error))
        if env is not None:env.close()
    audit(cell)
    if (root/'initial.png').exists() and (root/'final.png').exists():video(root)
    if error is not None:
        print(dict(index=cell['index'],interrupted=True,kind=type(error).__name__),flush=True)
        return 2
    print(dict(index=cell['index'],task=cell['task'],condition=cell['condition'],seed=cell['seed'],
               success=result['success'],steps=result['steps']),flush=True)
    return 0


def command(*args):
    return [PIXI,'run','--manifest-path',str(MANIFEST),'--locked','--no-install','python','-m',MODULE,*map(str,args)]


def supervisor(phase,gpu):
    from appl.gpu import identity
    from appl.journal import lock
    from .report import generate
    plan=verify();root=BASE/'supervisor'/phase;root.mkdir(parents=True,exist_ok=False)
    with lock(BASE/'supervisor.lock'):
        if phase=='evaluate' and not (BASE/'budget/ledger.json').exists():
            raise ValueError('Explicit currency and USD budget authorization required')
        pending=TASKS if phase=='check' else plan['cells'];completed=[];stop=None
        for item in pending:
            if phase=='evaluate' and read(BASE/'budget/ledger.json')['stopped']:
                stop=read(BASE/'budget/ledger.json')['stopped'];break
            ident=identity(gpu)
            occupancy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory',
                                               '--format=csv,noheader,nounits'],text=True)
            if ident['uuid'] in occupancy:
                raise RuntimeError('Selected GPU has another compute process; preserve it and reschedule explicitly')
            key=item if phase=='check' else str(item['index'])
            job=root/key;job.mkdir(parents=True,exist_ok=False)
            args=command(phase,'--task',item,'--gpu',gpu) if phase=='check' else command(phase,'--cell',item['index'],'--gpu',gpu)
            atomic(job/'process.json',dict(command=args,device=ident,occupancy=occupancy,started=time.time()))
            with (job/'stdout.log').open('w') as stdout,(job/'stderr.log').open('w') as stderr:
                process=subprocess.Popen(args,cwd=ROOT,stdout=stdout,stderr=stderr)
                atomic(job/'pid.json',dict(pid=process.pid))
                code=process.wait()
            atomic(job/'process_result.json',dict(returncode=code,finished=time.time()))
            completed.append(dict(item=item,returncode=code))
            if phase=='evaluate':generate()
            print(dict(phase=phase,finished=len(completed),planned=len(pending),last_returncode=code),flush=True)
            if code:
                stop='Episode/check stopped; no further admission or automatic retry';break
        atomic(root/'completion.json',dict(attempted=len(completed),planned=len(pending),processes=completed,
                                          stopped=stop,all_workers_exited=True,maximum_active_devices=1))
        if phase=='evaluate':generate()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','authorize','check','evaluate','supervise-check','supervise-evaluate','report'])
    p.add_argument('--task',choices=TASKS);p.add_argument('--cell',type=int)
    p.add_argument('--gpu',type=int);p.add_argument('--device-isolated',action='store_true')
    p.add_argument('--usd');p.add_argument('--authorization')
    args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='authorize':
        if not args.usd or not args.authorization:p.error('Explicit confirmed budget and authorization are required')
        Ledger().initialize(args.usd,args.authorization)
    elif args.action=='report':
        from .report import generate
        generate()
    elif args.action.startswith('supervise-'):supervisor(args.action.split('-')[1],args.gpu)
    else:
        if args.gpu is None:p.error('An explicitly chosen physical GPU is required')
        if not args.device_isolated:
            from appl.gpu import launch
            import sys
            launch(args.gpu,sys.argv[1:],module=MODULE)
        if args.action=='check':check(args.task)
        else:raise SystemExit(evaluate(verify()['cells'][args.cell]))


if __name__=='__main__':main()
