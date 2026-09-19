"""Resource-only parallelism for predeclared, unstarted single-policy trials.

The original coordinator owns the first 150 cells. This supervisor evaluates the
last 150 on two additional authorized GPUs. Before its queue reaches that range,
the original coordinator is paused (its active children continue) if necessary.
It is resumed only after every supplemental result/video/process exit is complete;
its existing completed-result path then verifies/reuses these exact outcomes.
There are no repeated physical trials and no change to frozen scientific code.
"""
import argparse
import copy
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from appl.io import ROOT,PIXI,MANIFEST,read,atomic,digest,event
from experiments.exp2.single_policy.common import BASE,NAMES,config,folder,verify_preparation

MODULE='experiments.exp2.analysis.single_policy_evaluation_supplement'
OUT=BASE/'evaluation_supplement'
DEVICES=[4,6]
ALL_DEVICES=[0,1,4,6,7]


def cells():
    return [dict(task=n,condition=c,seed=config(n)['evaluation']['seeds' if c=='ID' else 'ood_seeds'][i])
        for i in range(30) for c in ('ID','OOD') for n in NAMES]


def command(*args):
    return [PIXI,'run','--manifest-path',str(MANIFEST),'--locked','--no-install','python','-m',MODULE,*map(str,args)]


def verify():
    plan=read(OUT/'allocation.json')
    if digest(__file__)!=plan['source_sha256']:raise ValueError('Resource wrapper changed')
    if digest(BASE/'study_freeze.json')!=plan['study_freeze_sha256']:raise ValueError('Study freeze changed')
    for name,row in plan['configurations'].items():
        original=read(config(name)['_path']);resource=read(row['resource_path'])
        if resource.pop('devices')!=ALL_DEVICES:raise ValueError('Resource allocation changed')
        original.pop('devices')
        if resource!=original or digest(row['resource_path'])!=row['resource_sha256']:
            raise ValueError('Resource copy changed scientific settings')
    return plan


def policy_process(cfg,f,out,seed):
    from appl.prior_policies.runner import PolicyProcess
    plan=verify();name=f.parent.name
    copied=copy.copy(cfg);copied['_path']=plan['configurations'][name]['resource_path']
    # The worker uses this configuration only for the GPU allocation check; its
    # learned model, sampler and normalization load from the same checkpoint.
    return PolicyProcess(copied,f,out,seed)


def check(name,gpu):
    from appl.gpu import verify_cuda
    verify();f=folder(name);reference=read(f/'deployment_check/inputs.json')
    out=OUT/'checks'/str(gpu)/name;out.mkdir(parents=True,exist_ok=False)
    worker=policy_process(config(name),f,out/'worker',reference['seed'])
    try:
        actions=[worker.action(o['state'],reset=i==0) for i,o in enumerate(reference['observations'])]
    finally:worker.close()
    error=float(np.max(np.abs(np.asarray(actions)-np.asarray(read(f/'deployment_check/actions.json')))))
    if error>1e-6:raise ValueError('Resource-only deployment check changed actions')
    atomic(out/'result.json',dict(passed=True,maximum_action_error=error,device=verify_cuda(),
        API_calls=0,optimizer_updates=0,simulator_steps=0,actions=9,
        checkpoint_sha256=digest(f/'training/last.pt')))


def evaluate(name,condition,seed):
    plan=verify();cell=dict(task=name,condition=condition,seed=seed)
    if cell not in plan['supplemental_cells']:raise ValueError('Cell not assigned to this supervisor')
    from experiments.exp2.single_policy import evaluate as executor
    executor.PolicyProcess=policy_process
    return executor.evaluate(name,condition,seed)


def launch(kind,job,gpu):
    if kind=='evaluate':
        episode=BASE/job['task']/'evaluation/SinglePrior_6_xhigh'/job['condition']/str(job['seed'])
        if episode.exists():raise ValueError('Supplemental physical cell already exists')
    out=OUT/'jobs'/kind/str(gpu)/job['task']
    if kind=='evaluate':out=out/job['condition']/str(job['seed'])
    out.mkdir(parents=True,exist_ok=False)
    args=command(kind,'--task',job['task'],'--gpu',gpu)
    if kind=='evaluate':args+=['--condition',job['condition'],'--seed',str(job['seed'])]
    stdout=(out/'stdout.log').open('w');stderr=(out/'stderr.log').open('w')
    p=subprocess.Popen(args,cwd=ROOT,stdout=stdout,stderr=stderr)
    atomic(out/'process.json',dict(pid=p.pid,command=args,gpu=gpu,job=job,started=time.time()))
    return dict(proc=p,out=out,stdout=stdout,stderr=stderr,job=job,gpu=gpu)


def supervise():
    from appl.gpu import identity
    verify_preparation()
    if OUT.exists():raise ValueError('Existing supplemental operation retained; no implicit restart')
    all_cells=cells();selected=all_cells[150:]
    for cell in selected:
        root=BASE/cell['task']/'evaluation/SinglePrior_6_xhigh'/cell['condition']/str(cell['seed'])
        if root.exists():raise ValueError('Supplemental cell already attempted')
    parent=read(BASE/'incidents/tray_design_http502/recovery_started.json')['pid']
    argv=Path(f'/proc/{parent}/cmdline').read_bytes().split(b'\0')
    if b'experiments.exp2.analysis.single_policy_recover_tray' not in argv:
        raise ValueError('Original supervisor identity differs')
    configurations={};OUT.mkdir()
    for name in NAMES:
        original=Path(config(name)['_path']);value=read(original);value['devices']=ALL_DEVICES
        path=OUT/'configurations'/(name+'.json');atomic(path,value)
        configurations[name]=dict(original_path=str(original),original_sha256=digest(original),
            resource_path=str(path),resource_sha256=digest(path))
    plan=dict(created=time.time(),authority='Existing user authorization: any GPUs 0-7, at most five active, multiple jobs per GPU.',
        devices=ALL_DEVICES,additional_devices=DEVICES,original_supervisor_pid=parent,
        configurations=configurations,source_sha256=digest(__file__),
        study_freeze_sha256=digest(BASE/'study_freeze.json'),supplemental_cells=selected,
        original_owned_prefix=150,pause_admission_at_pending_index=120,
        automatic_retries=0,scientific_changes=[],occupancy={g:identity(g) for g in ALL_DEVICES})
    atomic(OUT/'allocation.json',plan)
    paused=False;started_evaluations=False
    positions={(c['task'],c['condition'],c['seed']):i for i,c in enumerate(all_cells)}
    def pause():
        nonlocal paused
        if not paused:
            os.kill(parent,signal.SIGSTOP);paused=True
            event(OUT/'control.jsonl','original_admission_paused',pid=parent,active_children_continue=True)
    def guard():
        status=read(BASE/'supervisor/status.json')
        if status['phase']!='evaluation' or status['failures']:
            raise RuntimeError('Original evaluation supervisor has an unexpected state')
        front=min((positions[(r['task'],r['condition'],r['seed'])] for r in status['pending']),default=300)
        if front>=120:pause()
    def run_phase(kind,jobs,slots):
        pending=list(jobs);active={};done=[];failed=[];last=0
        while pending or active:
            guard()
            for slot,item in list(active.items()):
                code=item['proc'].poll()
                if code is None:continue
                item['stdout'].close();item['stderr'].close()
                record=dict(**item['job'],gpu=item['gpu'],returncode=code)
                atomic(item['out']/'process_result.json',record);done.append(record);del active[slot]
                if code:failed.append(record)
            if failed:pause()
            if not failed:
                free={g:identity(g)['free_mib'] for g in DEVICES}
                for slot,gpu in enumerate(slots):
                    if not pending or slot in active or free[gpu]<6000:continue
                    job=pending.pop(0);active[slot]=launch(kind,job,gpu);free[gpu]-=6000
            status=dict(phase=kind,finished=done,failures=failed,pending=pending,
                active=[dict(**v['job'],gpu=v['gpu'],pid=v['proc'].pid) for v in active.values()],
                original_admission_paused=paused,time=time.time())
            atomic(OUT/'status.json',status)
            if time.monotonic()-last>30:
                print(dict(phase=kind,finished=len(done),pending=len(pending),active=len(active),failures=len(failed),original_paused=paused),flush=True)
                last=time.monotonic()
            if failed and not active:raise RuntimeError('Supplemental execution failed; admission stopped')
            if active or pending:time.sleep(.5)
        return done
    try:
        # Two distinct GPUs each check every frozen model against its original
        # B=1 actions before admitting any additional physical trial.
        checks=[]
        for gpu in DEVICES:
            checks+=run_phase('check',[dict(task=n) for n in NAMES],[gpu])
        if len(checks)!=10:raise ValueError('Incomplete resource equivalence checks')
        atomic(OUT/'checks_completed.json',dict(checks=checks,optimizer_updates=0,simulator_steps=0,API_calls=0))
        started_evaluations=True
        finished=run_phase('evaluate',selected,[4,6,4,6,4,6])
        evidence=[]
        for row in finished:
            root=BASE/row['task']/'evaluation/SinglePrior_6_xhigh'/row['condition']/str(row['seed'])
            r=read(root/'result.json');assert read(root/'audit.json')['passed']
            assert (root/'replay.mp4').exists() and row['returncode']==0
            evidence.append(dict(**row,result_sha256=digest(root/'result.json'),
                video_sha256=digest(root/'replay.mp4'),steps=r['steps'],success=r['success']))
        assert len(evidence)==150
        atomic(OUT/'completed.json',dict(completed=time.time(),physical_trials=150,episodes=evidence,
            no_repeated_physical_trials=True,allocation_sha256=digest(OUT/'allocation.json')))
        if paused:
            os.kill(parent,signal.SIGCONT);paused=False
            event(OUT/'control.jsonl','original_admission_resumed',pid=parent,completed_results_reused=True)
        print(dict(supplement_complete=True,physical_trials=150),flush=True)
    except Exception as error:
        if started_evaluations:pause()
        elif paused:
            os.kill(parent,signal.SIGCONT);paused=False
            event(OUT/'control.jsonl','original_admission_resumed',pid=parent,resource_check_failed_before_physics=True)
        atomic(OUT/'failure.json',dict(error=str(error),time=time.time(),original_paused=paused,
            automatic_retry=False,physical_evaluations_started=started_evaluations));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['supervise','check','evaluate'])
    p.add_argument('--task',choices=NAMES);p.add_argument('--gpu',type=int,choices=DEVICES)
    p.add_argument('--condition',choices=['ID','OOD']);p.add_argument('--seed',type=int)
    p.add_argument('--device-isolated',action='store_true');a=p.parse_args()
    if a.command=='supervise':return supervise()
    if a.task is None or a.gpu is None:p.error('Task and GPU required')
    if not a.device_isolated:
        from appl.gpu import launch as isolate
        return isolate(a.gpu,sys.argv[1:],module=MODULE)
    nodes={v.name for v in Path('/dev').glob('nvidia*') if v.name.removeprefix('nvidia').isdigit()}
    if nodes!={'nvidia'+os.environ['APPL_GPU_MINOR']}:raise RuntimeError('GPU isolation failed')
    os.environ['CUDA_VISIBLE_DEVICES']=os.environ['APPL_GPU_UUID'];os.environ['MUJOCO_EGL_DEVICE_ID']='0'
    if a.command=='check':check(a.task,a.gpu)
    else:evaluate(a.task,a.condition,a.seed)


if __name__=='__main__':main()
