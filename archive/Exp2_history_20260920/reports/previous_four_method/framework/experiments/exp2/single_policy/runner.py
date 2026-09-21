"""Finite supervised execution; no automatic API or physical-attempt retries."""
import argparse
from collections import deque
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from appl.io import ROOT, PIXI, MANIFEST, atomic, digest, read
from .common import BASE, DEVICES, METHOD, MODULE, NAMES, config, folder, prepare, freeze, verify_preparation


def command(*args):
    return [PIXI,'run','--manifest-path',str(MANIFEST),'--locked','--no-install','python','-m',MODULE,*map(str,args)]


def train(name,gpu):
    from experiments.exp2.astra.runner import api_identity
    from appl.prior_policies.report import authorship
    from appl.prior_policies.runner import gpu_job
    from .design import run
    verify_preparation();cfg=config(name);f=folder(name)
    if (f/'failure.json').exists(): raise RuntimeError('Recorded policy failure; no implicit retry')
    try:
        submitted=run(cfg,f,gpu)
        api_identity(f/'design/journal.sqlite');authorship(f,submitted)
        if (f/'training').exists(): raise RuntimeError('Training already attempted; no implicit restart')
        print(dict(stage='formal_training',task=name,gpu=gpu,updates=60000),flush=True)
        gpu_job(cfg,f,f/'source',f/'training',gpu,60000)
        subprocess.run(command('check','--task',name,'--gpu',gpu),cwd=ROOT,check=True)
        print(dict(stage='model_ready',task=name,gpu=gpu),flush=True)
    except Exception as error:
        atomic(f/'failure.json',dict(error=str(error),time=time.time(),automatic_retry=False));raise


def launch_job(phase,job,gpu):
    from appl.gpu import identity
    out=BASE/'jobs'/phase/job['task']
    if phase=='evaluation': out=out/job['condition']/str(job['seed'])
    if out.exists(): raise RuntimeError('Existing job retained; no implicit retry: '+str(out))
    occupancy=identity(gpu);out.mkdir(parents=True)
    args=command('train' if phase=='training' else 'evaluate','--task',job['task'],'--gpu',gpu)
    if phase=='evaluation': args+=['--condition',job['condition'],'--seed',str(job['seed'])]
    stdout=(out/'stdout.log').open('w');stderr=(out/'stderr.log').open('w')
    proc=subprocess.Popen(args,cwd=ROOT,stdout=stdout,stderr=stderr)
    atomic(out/'process.json',dict(pid=proc.pid,command=args,started=time.time(),gpu=gpu,occupancy=occupancy))
    return dict(process=proc,stdout=stdout,stderr=stderr,out=out,job=job,gpu=gpu)


def phase_run(phase,jobs,slots):
    from appl.gpu import identity
    pending=deque(jobs);active={};finished=[];failures=[];last_print=0
    while pending or active:
        for slot,item in list(active.items()):
            code=item['process'].poll()
            if code is None: continue
            item['stdout'].close();item['stderr'].close()
            record=dict(**item['job'],gpu=item['gpu'],returncode=code)
            atomic(item['out']/'process_result.json',record);finished.append(record);del active[slot]
            print(dict(phase=phase,completed=record),flush=True)
            if code: failures.append(record)
        if not failures:
            available={g:identity(g)['free_mib'] for g in DEVICES}
            for slot,gpu in enumerate(slots):
                if not pending or slot in active: continue
                required=10000 if phase=='training' else 6000
                if available[gpu]<required: continue
                job=pending.popleft();active[slot]=launch_job(phase,job,gpu);available[gpu]-=required
        status=dict(phase=phase,time=time.time(),finished=finished,failures=failures,pending=list(pending),
            active=[dict(**v['job'],gpu=v['gpu'],pid=v['process'].pid) for v in active.values()])
        atomic(BASE/'supervisor/status.json',status)
        if time.monotonic()-last_print>30:
            print(dict(phase=phase,finished=len(finished),pending=len(pending),active=status['active'],failures=len(failures)),flush=True)
            last_print=time.monotonic()
        if failures and not active:
            atomic(BASE/f'supervisor/{phase}_failed.json',status)
            raise RuntimeError('Admission stopped after an execution error; original attempts retained')
        if pending or active: time.sleep(5)
    atomic(BASE/f'supervisor/{phase}_completed.json',dict(finished=finished,failures=failures,time=time.time()))


def run():
    prepare()
    out=BASE/'supervisor';out.mkdir(exist_ok=True)
    with (out/'owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'started.json').exists(): raise RuntimeError('Study already started; explicit reconciliation required')
        atomic(out/'started.json',dict(started=time.time(),pid=os.getpid(),user_choice='One candidate per task',
            preparation_sha256=digest(BASE/'preparation.json'),devices=DEVICES,automatic_retries=0))
        try:
            phase_run('training',[dict(task=n) for n in NAMES],[0,1,7,0,1])
            freeze()
            jobs=[]
            for index in range(30):
                for condition in ('ID','OOD'):
                    for name in NAMES:
                        seed=config(name)['evaluation']['seeds' if condition=='ID' else 'ood_seeds'][index]
                        jobs.append(dict(task=name,condition=condition,seed=seed))
            phase_run('evaluation',jobs,[0,1,7,0,1,7])
            from .report import generate
            summary=generate()
            atomic(out/'completed.json',dict(completed=time.time(),summary=summary))
            print(dict(completed=True,summary=summary),flush=True)
        except Exception as error:
            atomic(out/'failure.json',dict(error=str(error),time=time.time(),automatic_retry=False));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run','train','check','freeze','evaluate','report','status'])
    p.add_argument('--task',choices=NAMES);p.add_argument('--gpu',type=int)
    p.add_argument('--condition',choices=['ID','OOD']);p.add_argument('--seed',type=int)
    p.add_argument('--device-isolated',action='store_true')
    a=p.parse_args()
    if a.command in ('train','check','evaluate'):
        if a.task is None or a.gpu not in DEVICES: p.error('Task and allocated GPU required')
    if a.command in ('check','evaluate'):
        if not a.device_isolated:
            from appl.gpu import launch
            return launch(a.gpu,sys.argv[1:],module=MODULE)
        nodes={v.name for v in Path('/dev').glob('nvidia*') if v.name.removeprefix('nvidia').isdigit()}
        if nodes!={'nvidia'+os.environ['APPL_GPU_MINOR']}: raise RuntimeError('GPU isolation failed')
        os.environ['CUDA_VISIBLE_DEVICES']=os.environ['APPL_GPU_UUID'];os.environ['MUJOCO_EGL_DEVICE_ID']='0'
    if a.command=='prepare': print(prepare())
    elif a.command=='run': run()
    elif a.command=='train': train(a.task,a.gpu)
    elif a.command=='freeze': print(freeze())
    elif a.command=='check':
        from .evaluate import deployment_check
        print(deployment_check(a.task),flush=True)
    elif a.command=='evaluate':
        from .evaluate import evaluate
        evaluate(a.task,a.condition,a.seed)
    elif a.command=='report':
        from .report import generate
        print(generate())
    else:
        path=BASE/'supervisor/status.json'
        print(read(path) if path.exists() else dict(status='not_started'))


if __name__=='__main__': main()
