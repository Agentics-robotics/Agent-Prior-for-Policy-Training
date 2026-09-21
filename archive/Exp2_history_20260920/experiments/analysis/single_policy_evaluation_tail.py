"""Resource-only execution of original predeclared cells 120:150.

The original coordinator eventually reuses these results through its frozen
completed-result path. If needed, pause only its admission process before it
can reach these cells. Existing children continue. No model, dataset, seed,
sampler, scientific source, or physical-attempt count changes.
"""
import os
from pathlib import Path
import signal
import subprocess
import time

from appl.io import ROOT, read, atomic, digest, event
from appl.gpu import identity
from experiments.exp2.single_policy.common import BASE, METHOD, verify_preparation
from experiments.exp2.single_policy.runner import command
from experiments.exp2.analysis.single_policy_evaluation_supplement import cells

OUT=BASE/'evaluation_tail'
DEVICES=[0,1,7]
PARENT_MODULE=b'experiments.exp2.analysis.single_policy_recover_tray'


def episode(job):
    return BASE/job['task']/'evaluation'/METHOD/job['condition']/str(job['seed'])


def supervise():
    verify_preparation()
    if OUT.exists():raise ValueError('Existing operation retained; no implicit restart')
    all_cells=cells();selected=all_cells[120:150]
    positions={(r['task'],r['condition'],r['seed']):i for i,r in enumerate(all_cells)}
    parent=read(BASE/'incidents/tray_design_http502/recovery_started.json')['pid']
    def parent_identity():
        argv=Path(f'/proc/{parent}/cmdline').read_bytes().split(b'\0')
        if PARENT_MODULE not in argv:raise ValueError('Original coordinator identity differs')
    parent_identity()
    original=read(BASE/'supervisor/status.json')
    if original['phase']!='evaluation' or original['failures']:raise ValueError('Unexpected original state')
    if any(r not in original['pending'] or episode(r).exists() for r in selected):
        raise ValueError('Every tail cell must be predeclared and unstarted')
    front=min(positions[(r['task'],r['condition'],r['seed'])] for r in original['pending'])
    if front>=90:raise ValueError('Insufficient admission buffer for this operation')
    OUT.mkdir()
    plan=dict(started=time.time(),pid=os.getpid(),source_sha256=digest(__file__),
        authority='Existing authorization for GPUs 0-7, at most five active, multiple jobs per GPU.',
        devices=DEVICES,combined_devices=[0,1,4,6,7],selected_cells=selected,
        original_supervisor_pid=parent,pause_before_pending_index=90,
        original_status_sha256=digest(BASE/'supervisor/status.json'),
        study_freeze_sha256=digest(BASE/'study_freeze.json'),
        scientific_changes=[],API_calls=0,optimizer_updates=0,automatic_retries=0,
        occupancy={gpu:identity(gpu) for gpu in DEVICES})
    atomic(OUT/'allocation.json',plan)
    pending=list(selected);active={};finished=[];failures=[];paused=False;last=0
    def hold():
        nonlocal paused
        if not paused:
            parent_identity();os.kill(parent,signal.SIGSTOP);paused=True
            event(OUT/'control.jsonl','original_admission_paused',pid=parent,active_children_continue=True)
    def guard():
        if digest(__file__)!=plan['source_sha256']:raise ValueError('Resource source changed')
        status=read(BASE/'supervisor/status.json')
        if status['phase']!='evaluation' or status['failures']:
            raise RuntimeError('Original evaluation reported an error')
        front=min((positions[(r['task'],r['condition'],r['seed'])] for r in status['pending']),default=300)
        # The original loop admits at most six cells before its five-second
        # sleep. This thirty-cell buffer precedes our first cell at index 120.
        if front>=90:hold()
    def launch(job,gpu):
        if episode(job).exists():raise ValueError('Tail cell already attempted')
        out=OUT/'jobs'/job['task']/job['condition']/str(job['seed'])
        out.mkdir(parents=True,exist_ok=False)
        args=command('evaluate','--task',job['task'],'--condition',job['condition'],
                     '--seed',job['seed'],'--gpu',gpu)
        stdout=(out/'stdout.log').open('w');stderr=(out/'stderr.log').open('w')
        p=subprocess.Popen(args,cwd=ROOT,stdout=stdout,stderr=stderr)
        atomic(out/'process.json',dict(pid=p.pid,command=args,gpu=gpu,started=time.time(),job=job))
        return dict(proc=p,out=out,stdout=stdout,stderr=stderr,job=job,gpu=gpu)
    try:
        while pending or active:
            guard()
            for slot,item in list(active.items()):
                code=item['proc'].poll()
                if code is None:continue
                item['stdout'].close();item['stderr'].close()
                record=dict(**item['job'],gpu=item['gpu'],returncode=code)
                atomic(item['out']/'process_result.json',record);finished.append(record);del active[slot]
                if code:failures.append(record)
            if failures:hold()
            else:
                available={gpu:identity(gpu)['free_mib'] for gpu in DEVICES}
                for slot,gpu in enumerate([0,1,7,0,1,7]):
                    if not pending or slot in active or available[gpu]<6000:continue
                    guard();job=pending.pop(0);active[slot]=launch(job,gpu);available[gpu]-=6000
            atomic(OUT/'status.json',dict(phase='evaluation',finished=finished,failures=failures,pending=pending,
                active=[dict(**v['job'],gpu=v['gpu'],pid=v['proc'].pid) for v in active.values()],
                original_admission_paused=paused,time=time.time()))
            if time.monotonic()-last>30:
                print(dict(tail_finished=len(finished),pending=len(pending),active=len(active),
                    errors=len(failures),original_paused=paused),flush=True);last=time.monotonic()
            if failures and not active:raise RuntimeError('Tail execution error; original admission held')
            if active or pending:time.sleep(.5)
        evidence=[]
        for row in finished:
            root=episode(row);result=read(root/'result.json')
            assert row['returncode']==0 and read(root/'audit.json')['passed']
            assert (root/'replay.mp4').exists()
            evidence.append(dict(**row,result_sha256=digest(root/'result.json'),
                video_sha256=digest(root/'replay.mp4'),steps=result['steps'],success=result['success']))
        assert len(evidence)==30
        atomic(OUT/'completed.json',dict(completed=time.time(),physical_trials=30,episodes=evidence,
            no_repeated_physical_trials=True,allocation_sha256=digest(OUT/'allocation.json')))
        if paused:
            parent_identity();os.kill(parent,signal.SIGCONT);paused=False
            event(OUT/'control.jsonl','original_admission_resumed',pid=parent,completed_results_reused=True)
        print(dict(tail_complete=True,physical_trials=30),flush=True)
    except Exception as error:
        hold()
        atomic(OUT/'failure.json',dict(error=str(error),time=time.time(),original_paused=paused,
            automatic_retry=False));raise


if __name__=='__main__':supervise()
