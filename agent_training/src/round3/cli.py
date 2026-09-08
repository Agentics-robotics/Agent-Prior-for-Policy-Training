"""Resume frozen artifacts. New designs require the active Codex session."""
import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path
from .common import R3, ROOT, TASKS, GPUS, planned_runs, run_record, update_run, lock, implementation_path, event
from relative_dp.utils import atomic_json, read_json

PIXI=Path('/home/users/oscar/.pixi/bin/pixi')

def _priority(rec):
    first=rec['task'] in TASKS[:2] and rec['train_n'] in (5,20) and rec['candidate_id']!='P4'
    return (0 if first else 1,TASKS.index(rec['task']),[5,20,2,10].index(rec['train_n']),rec['candidate_id'])

def _eligible(rec,stage):
    if rec['status'] in ('invalid','blocked'):
        return False
    if stage=='train':
        return (rec['train_status']=='pending' and implementation_path(rec['task'],rec['candidate_id']).exists()
            and (R3/'data'/rec['task']/'manifest.json').exists()
            and (R3/'design_records'/rec['task']/'initial_freeze.json').exists())
    if stage=='dev':
        return rec['train_status']=='completed' and rec['dev_status']=='pending'
    return rec['train_status']=='completed' and rec['test_status']=='pending' and (R3/'global_freeze.json').exists()

def _child(command,identifier,gpu):
    log=R3/'logs'/f'{identifier}_{command}.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',buffering=1) as output:
        proc=subprocess.Popen([str(PIXI),'run','env',f'CUDA_VISIBLE_DEVICES={gpu}',f'MUJOCO_EGL_DEVICE_ID={gpu}',
            'python','-m','round3.cli',command,'--run-id',identifier],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        event('subprocess_started',run_id=identifier,command=command,pid=proc.pid,gpu=gpu,log=str(log.relative_to(ROOT)))
        result=proc.wait()
    if result:
        raise RuntimeError(f'{command} {identifier} exited {result}; inspect {log}')

def gpu_queue(gpu,stages):
    completed=[]
    with lock('gpu-'+str(gpu),blocking=False):
        while True:
            selected=None
            with lock('scheduler'):
                records=sorted(planned_runs(),key=_priority)
                # Finish each run's dev promptly, permitting early cost/feedback evidence.
                for stage in stages:
                    rec=next((r for r in records if _eligible(r,stage)),None)
                    if rec:
                        field={'train':'train_status','dev':'dev_status','test':'test_status'}[stage]
                        update_run(rec['run_id'],status='running',**{field:'running'},gpu=gpu)
                        selected=rec,stage
                        break
            if selected is None:
                return completed
            rec,stage=selected
            try:
                _child({'train':'train-one','dev':'dev-one','test':'test-one'}[stage],rec['run_id'],gpu)
                completed.append([rec['run_id'],stage])
            except Exception as exc:
                # Preserve the failed process and checkpoint; no silent retry or invalidation.
                field={'train':'train_status','dev':'dev_status','test':'test_status'}[stage]
                update_run(rec['run_id'],status='pending',**{field:'pending'},last_error=str(exc),pid=None)
                raise

def schedule(stages):
    with lock('orchestrator',blocking=False):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            jobs=[executor.submit(gpu_queue,gpu,stages) for gpu in GPUS]
            return [future.result() for future in jobs]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','validate-proposals','train','dev-eval','test','report','resume','all',
        'train-one','dev-one','test-one','debug-one'])
    parser.add_argument('--task',choices=TASKS)
    parser.add_argument('--run-id')
    parser.add_argument('--debug-updates',type=int,default=100)
    parser.add_argument('--stop-after',type=int)
    args=parser.parse_args()
    planned_runs()
    if args.command=='prepare':
        from .prepare import prepare_task
        for task in ([args.task] if args.task else TASKS):
            prepare_task(task)
    elif args.command=='validate-proposals':
        from .proposals import validate_all
        print(validate_all())
    elif args.command in ('train-one','debug-one'):
        from .learning import train
        assert args.run_id
        print(train(args.run_id,debug_updates=args.debug_updates if args.command=='debug-one' else None,stop_after=args.stop_after))
    elif args.command in ('dev-one','test-one'):
        from .evaluate import worker
        assert args.run_id
        print(worker(args.run_id,'dev' if args.command=='dev-one' else 'test'))
    elif args.command=='train':
        if args.run_id:
            from .learning import train
            print(train(args.run_id,stop_after=args.stop_after))
        else:
            print(schedule(['dev','train']))
    elif args.command=='dev-eval':
        print(schedule(['dev']))
    elif args.command=='test':
        from .evaluate import freeze_selection,freeze_test_gate
        freeze_selection(require_all=True)
        freeze_test_gate()
        print(schedule(['test']))
    elif args.command=='report':
        from .report import generate
        print(generate())
    else:
        from .proposals import validate_all,baseline_config
        from .prepare import prepare_task
        # Prepare only as far as the first missing proposal, rather than delaying first training for all tasks.
        for task in TASKS:
            if not (R3/'data'/task/'manifest.json').exists():
                prepare_task(task)
            baseline_config(task)
            if not (R3/'design_records'/task/'initial_freeze.json').exists():
                break
        validation=validate_all()
        print(schedule(['dev','train']))
        missing=[r for r in validation if r['status']=='pending-design']
        revisions=[t for t in TASKS if not (R3/'design_records'/t/'revision.json').exists()]
        pending=[r for r in planned_runs() if r['status'] not in ('completed','invalid') and r['dev_status']!='completed']
        if missing or revisions or pending:
            state=dict(status='pending-design' if missing or revisions else 'pending',
                initial_design_tasks=[r['task'] for r in missing],feedback_decision_tasks=revisions,
                pending_development_runs=[r['run_id'] for r in pending],
                action='Active Codex completes actual proposals/implementations/revisions and resumes; no API key is required.')
            atomic_json(R3/'next_action.json',state)
            print(state)
            return
        from .evaluate import freeze_selection,freeze_test_gate
        freeze_selection(require_all=True)
        freeze_test_gate()
        schedule(['test'])
        from .report import generate
        print(generate())

if __name__=='__main__':
    main()
