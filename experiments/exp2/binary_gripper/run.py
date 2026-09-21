"""Supervised offline inference; the only scientific change is gripper decoding."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from appl.io import ROOT,PIXI,MANIFEST,read,atomic,digest
from .common import BASE,MODULE,TASKS,METHODS,prepare,verify,load_policy,close_policy,episode_root,execute_action


def command(*args):
    return [PIXI,'run','--manifest-path',str(MANIFEST),'--locked','--no-install','python','-m',MODULE,*map(str,args)]


def check(cell):
    import torch
    from appl.gpu import verify_cuda
    torch.set_num_threads(2)
    root=BASE/'checks'/cell['task']/cell['method'];root.mkdir(parents=True,exist_ok=False)
    policy=None
    try:
        policy=load_policy(cell,root);old=Path(cell['reference_root']);previous=read(old/'initial_state.json');actual=[];expected=[]
        with (old/'trace.jsonl').open() as f:
            for i,line in enumerate(f):
                row=json.loads(line)
                action=policy.action(previous,reset=i==0) if cell['method']=='SinglePrior_6_xhigh' else policy.action(previous)
                actual.append(action);expected.append(row['raw_action']);previous=row['state']
                if i==16:break
        error=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
        if error>1e-6:raise ValueError(('Original inference not reproduced',error))
        close_policy(policy,cell);policy=None
        atomic(root/'actions.json',dict(actual=actual,reference=expected))
        atomic(root/'result.json',dict(passed=True,max_action_error=error,actions=len(actual),device=verify_cuda(),
            reference_root=str(old),reference_trace_sha256=cell['reference_trace_sha256'],API_calls=0,learned_optimizer_updates=0,physical_steps=0))
    except Exception as error:
        atomic(root/'failure.json',dict(error=str(error),automatic_retry=False));raise
    finally:
        if policy is not None:close_policy(policy,cell)


def evaluate(cell):
    import torch
    from PIL import Image
    from appl.gpu import verify_cuda
    from appl.envs.adapter import step
    from appl.scaleup.evaluate import environment
    from appl.scaleup.tasks import measure
    torch.set_num_threads(2);plan=verify();root=episode_root(cell);root.mkdir(parents=True,exist_ok=False)
    model=plan['models'][cell['task']+'/'+cell['method']];goal=read(model['goal_contract'])
    atomic(root/'plan.json',dict(**cell,study_sha256=digest(BASE/'plan.json'),device=verify_cuda(),pid=os.getpid(),
        maximum_steps=5000,checkpoint_sha256=model['checkpoint_sha256'],API_calls=0,gripper_decoder='sign_zero_opens'))
    env=None;policy=None;started=time.monotonic();steps=0
    try:
        policy=load_policy(cell,root)
        env,obs,state=environment(cell['task'],cell['seed'],cell['condition'])
        reference=Path(cell['reference_root'])/'initial_state.json'
        if digest(reference)!=cell['reference_initial_sha256'] or state!=read(reference):raise ValueError('Paired initial state differs')
        limits=[];wrapper=env
        while hasattr(wrapper,'env'):
            if '_max_episode_steps' in vars(wrapper):limits.append(wrapper._max_episode_steps)
            wrapper=wrapper.env
        if not limits or any(v!=5000 for v in limits):raise ValueError('Wrapper cap mismatch')
        space=env.unwrapped.single_action_space
        atomic(root/'environment.json',dict(time_limits=limits,control_mode=env.unwrapped.control_mode,
            action_low=space.low.tolist(),action_high=space.high.tolist(),
            controller_config=repr(env.unwrapped.agent.controller.configs)))
        atomic(root/'initial_state.json',state)
        def frame(name):Image.fromarray(obs['sensor_data']['front']['rgb'][0].cpu().numpy()).save(root/name)
        frame('initial.png');first={};changed=0;saturated=0
        with (root/'trace.jsonl').open('w',buffering=65536) as stream:
            for steps in range(1,5001):
                raw=policy.action(state,reset=steps==1) if cell['method']=='SinglePrior_6_xhigh' else policy.action(state)
                raw=np.asarray(raw,np.float32);action=execute_action(raw,space.low,space.high)
                changed+=int(action[7]!=raw[7]);saturated+=int(np.any(np.clip(raw,space.low,space.high)!=raw))
                obs,state=step(env,action);metrics=measure(state,goal)
                stream.write(json.dumps(dict(kind='step',step=steps,policy_id=cell['method'],state=state,raw_action=raw.tolist(),action=action.tolist(),metrics=metrics))+'\n')
                for k,v in metrics.items():
                    if v and k not in first:first[k]=steps
                if steps%20==1:frame(f'frame_{steps:04d}.png')
                if metrics['success']:break
                if steps%100==0:
                    stream.flush();atomic(root/'progress.json',dict(steps=steps,first_goals=first,elapsed_seconds=time.monotonic()-started))
        frame('final.png');chunks=len(policy.timings);inference=sum(policy.timings)
        close_policy(policy,cell);policy=None
        result=dict(task=cell['task'],method=cell['method'],variant='binary_gripper_v1',condition=cell['condition'],seed=cell['seed'],
            steps=steps,success=metrics['success'],final=metrics,first_goals=first,completed_evaluation=True,
            status='succeeded' if metrics['success'] else 'physical_budget_exhausted',
            elapsed_seconds=time.monotonic()-started,inference_chunks=chunks,inference_seconds=inference,
            changed_gripper_commands=changed,actuator_saturated_steps=saturated,API_calls=0,learned_optimizer_updates=0)
        atomic(root/'result.json',result)
    except Exception as error:
        atomic(root/'failure.json',dict(error=str(error),physical_steps=steps,automatic_retry=False));raise
    finally:
        if policy is not None:close_policy(policy,cell)
        if env is not None:env.close()
    from .report import audit_episode,video
    audit_episode(cell);video(root)
    print(dict(index=cell['index'],task=cell['task'],method=cell['method'],condition=cell['condition'],seed=cell['seed'],success=result['success'],steps=steps),flush=True)


def occupancy():
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory,process_name','--format=csv,noheader,nounits'],text=True)
    foreign=set()
    for line in text.splitlines():
        if not line.strip():continue
        pid,uuid,*_=line.split(', ')
        path=Path('/proc')/pid/'cmdline'
        argv=path.read_bytes().replace(b'\0',b' ').decode(errors='replace') if path.exists() else ''
        if MODULE not in argv and str(BASE) not in argv:foreign.add(uuid)
    return text,foreign


def supervise():
    from appl.gpu import identity
    import fcntl
    plan=verify();owner=(BASE/'supervisor.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=BASE/'supervisor';out.mkdir(exist_ok=False)
    active={};finished=[];failed=[];devices={g:identity(g) for g in plan['allowed_devices']};last=0
    checks=[next(c for c in plan['cells'] if c['task']==t and c['method']==m and c['condition']=='ID') for t in TASKS for m in METHODS]
    for phase,pending in [('check',checks),('evaluate',list(plan['cells']))]:
        if failed:break
        while pending or active:
            for slot,item in list(active.items()):
                code=item['process'].poll()
                if code is None:continue
                item['stdout'].close();item['stderr'].close()
                result=dict(phase=phase,cell=item['cell'],gpu=slot[0],returncode=code,finished=time.time())
                atomic(item['out']/'process_result.json',result);finished.append(result);del active[slot]
                if code:failed.append(result)
            text,foreign=occupancy();used={g for g,_ in active}
            candidates=[g for g in plan['allowed_devices'] if devices[g]['uuid'] not in foreign]
            admitted=sorted(used)+[g for g in plan['initial_active_devices']+candidates if g not in used and g in candidates]
            admitted=list(dict.fromkeys(admitted))[:plan['maximum_active_devices']]
            if not failed:
                for gpu in admitted:
                    if devices[gpu]['uuid'] in foreign:continue
                    free=identity(gpu)['free_mib']
                    for lane in range(plan['jobs_per_device'] if phase=='evaluate' else 1):
                        slot=(gpu,lane)
                        if not pending or slot in active or free<plan['minimum_free_mib_to_admit']:continue
                        cell=pending.pop(0);job=out/'jobs'/phase/str(cell['index']);job.mkdir(parents=True,exist_ok=False)
                        stdout=(job/'stdout.log').open('w');stderr=(job/'stderr.log').open('w')
                        args=command(phase,'--cell',cell['index'],'--gpu',gpu)
                        proc=subprocess.Popen(args,cwd=ROOT,stdout=stdout,stderr=stderr)
                        atomic(job/'process.json',dict(pid=proc.pid,command=args,gpu=gpu,cell=cell,started=time.time(),occupancy=text))
                        active[slot]=dict(process=proc,stdout=stdout,stderr=stderr,out=job,cell=cell)
                        free-=plan['minimum_free_mib_to_admit']
            status=dict(phase=phase,finished=finished,failures=failed,pending=pending,
                active=[dict(gpu=g,lane=l,pid=v['process'].pid,cell=v['cell']) for (g,l),v in active.items()],time=time.time())
            atomic(out/'status.json',status)
            if time.monotonic()-last>30:
                print(dict(phase=phase,finished=sum(r['phase']==phase for r in finished),active=len(active),pending=len(pending),failures=len(failed)),flush=True);last=time.monotonic()
            if failed and not active:break
            time.sleep(5)
    if failed:
        atomic(out/'failure.json',dict(failures=failed,automatic_retry=False));raise RuntimeError('Study admission stopped after recorded execution error')
    from .report import generate
    generate(require_complete=True)
    atomic(out/'completed.json',dict(finished=time.time(),jobs=finished,active_workers=0))
    print(dict(completed=True,episodes=600,API_calls=0,output=str(BASE)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','supervise','check','evaluate','report'])
    parser.add_argument('--cell',type=int);parser.add_argument('--gpu',type=int);parser.add_argument('--device-isolated',action='store_true')
    args=parser.parse_args()
    if args.mode=='prepare':return prepare()
    if args.mode=='supervise':return supervise()
    if args.mode=='report':
        from .report import generate
        return generate()
    plan=verify();cell=plan['cells'][args.cell]
    if not args.device_isolated:
        from appl.gpu import launch
        return launch(args.gpu,sys.argv[1:],module=MODULE)
    (check if args.mode=='check' else evaluate)(cell)


if __name__=='__main__':main()
