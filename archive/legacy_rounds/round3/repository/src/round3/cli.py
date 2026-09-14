"""Resume frozen artifacts. New designs require the active Codex session."""
import argparse
import concurrent.futures
import contextlib
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from .common import R3, ROOT, TASKS, GPUS, WORKERS_PER_GPU, planned_runs, run_record, update_run, lock, implementation_path, event
from relative_dp.utils import atomic_json, read_json, sha256

def pixi_binary():
    candidates=[os.environ.get('PIXI_EXE'), shutil.which('pixi'), str(Path.home()/'.pixi/bin/pixi')]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate,os.X_OK):
            return str(Path(candidate).resolve())
    raise RuntimeError('Pixi executable not found; set PIXI_EXE to the installed Pixi binary')

def data_complete(task):
    path=R3/'data'/task/'manifest.json'
    return path.exists() and read_json(path).get('complete') is True

def _pid_live(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid),0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True

def reconcile_stale_runs():
    with lock('scheduler'):
        return _reconcile_stale_runs()

def _reconcile_stale_runs():
    """Only release abandoned claims after both process and run-lock checks.

    A reused or inaccessible live PID is conservatively left alone. Completed
    artifacts are validated by the stage worker before its manifest is repaired.
    """
    recovered=[]
    for rec in planned_runs():
        fields=[f'{s}_status' for s in ('train','dev','test') if rec.get(f'{s}_status')=='running']
        if not fields:
            continue
        pids=[rec.get(k) for k in ('pid','worker_pid','launcher_pid','scheduler_pid')]
        if any(_pid_live(pid) and int(pid)!=os.getpid() for pid in pids if pid):
            continue
        try:
            with lock('run-'+rec['run_id'],blocking=False):
                update_run(rec['run_id'],status='pending',**{f:'pending' for f in fields},
                    pid=None,worker_pid=None,launcher_pid=None,scheduler_pid=None,
                    recovery_reason='No live recorded process and exclusive run lock acquired')
                recovered.append(rec['run_id'])
        except BlockingIOError:
            continue
    if recovered:
        event('stale_runs_recovered',run_ids=recovered)
    return recovered

@contextlib.contextmanager
def gpu_context(gpu):
    gpu=GPUS[0] if gpu is None else gpu
    if gpu not in GPUS:
        raise ValueError(f'GPU {gpu} not in the authorized devices {GPUS}')
    # This runs INSIDE Pixi, after its historical GPU 1 activation variable.
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    os.environ['MUJOCO_EGL_DEVICE_ID']=str(gpu)
    owner=os.environ.get('ROUND3_GPU_LEASE_PID')
    if owner and _pid_live(owner):
        yield
    else:
        with lock('gpu-'+str(gpu),blocking=False):
            yield

def _priority(rec):
    first=rec['task'] in TASKS[:2] and rec['train_n'] in (5,20) and rec['candidate_id']!='P4'
    return (0 if first else 1,TASKS.index(rec['task']),[5,20,2,10].index(rec['train_n']),rec['candidate_id'])

def implementation_ready(rec):
    if rec['candidate_id']=='B0':
        return True
    path=implementation_path(rec['task'],rec['candidate_id'])
    if not path.exists():
        return False
    audit=read_json(path).get('implementation_audit')
    expected_hash=None
    if isinstance(audit,dict):
        expected_hash=audit.get('sha256')
        audit=audit.get('path')
    if not isinstance(audit,str) or not (ROOT/audit).is_file():
        return False
    if expected_hash and sha256(ROOT/audit)!=expected_hash:
        return False
    result=read_json(ROOT/audit)
    return result.get('passed') is True or result.get('status')=='passed'

def _eligible(rec,stage):
    if rec['status'] in ('invalid','blocked'):
        return False
    if any(_pid_live(rec.get(field)) for field in ('pid','worker_pid','launcher_pid')):
        return False
    if stage=='train':
        return (rec['train_status']=='pending' and implementation_path(rec['task'],rec['candidate_id']).exists()
            and data_complete(rec['task']) and implementation_ready(rec)
            and (R3/'design_records'/rec['task']/'initial_freeze.json').exists())
    if stage=='dev':
        return rec['train_status']=='completed' and rec['dev_status']=='pending'
    return rec['train_status']=='completed' and rec['test_status']=='pending' and (R3/'global_freeze.json').exists()

def _process_group_live(pgid):
    """Check our Linux child group, excluding exited children awaiting reaping."""
    try:
        os.killpg(pgid,0)
    except ProcessLookupError:
        return False
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            # comm may contain spaces and parentheses; fields after its final ')'
            # start with state, parent PID, and process-group ID.
            fields=(path/'stat').read_text().rsplit(')',1)[1].split()
        except (FileNotFoundError,ProcessLookupError):
            continue
        except PermissionError:
            return True
        if int(fields[2])==pgid and fields[0] not in ('Z','X'):
            return True
    return False

def _interrupt_and_wait(proc):
    try:
        os.killpg(proc.pid,signal.SIGINT)
    except ProcessLookupError:
        pass
    # Pixi may exit before its Python worker finishes saving. Keep the parent
    # GPU lease until the entire owned group exits, even after another interrupt.
    while True:
        try:
            proc.wait()
            while _process_group_live(proc.pid):
                time.sleep(.1)
            return
        except (KeyboardInterrupt,InterruptedError):
            continue

def _child(command,identifier,gpu):
    log=R3/'logs'/f'{identifier}_{command}.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',buffering=1) as output:
        proc=subprocess.Popen([pixi_binary(),'run','env',f'CUDA_VISIBLE_DEVICES={gpu}',f'MUJOCO_EGL_DEVICE_ID={gpu}',
            f'ROUND3_GPU_LEASE_PID={os.getpid()}',
            'python','-m','round3.cli',command,'--run-id',identifier,'--gpu',str(gpu)],
            cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        update_run(identifier,launcher_pid=proc.pid,scheduler_pid=os.getpid(),gpu=gpu)
        event('subprocess_started',run_id=identifier,command=command,child_pid=proc.pid,gpu=gpu,log=str(log.relative_to(ROOT)))
        try:
            result=proc.wait()
        except BaseException:
            # Interrupt only this child process group; training saves its state.
            _interrupt_and_wait(proc)
            raise
        finally:
            if proc.poll() is not None:
                with lock('scheduler'):
                    if run_record(identifier).get('launcher_pid')==proc.pid:
                        update_run(identifier,launcher_pid=None,scheduler_pid=None)
    if result:
        raise RuntimeError(f'{command} {identifier} exited {result}; inspect {log}')

def gpu_queue(gpu,stages,slot=0):
    completed=[]
    if gpu not in GPUS or not 0 <= slot < WORKERS_PER_GPU:
        raise ValueError(f'Unauthorized GPU/worker slot: {gpu}/{slot}')
    lease='gpu-'+str(gpu)+(f'-slot-{slot}' if slot else '')
    with lock(lease,blocking=False):
        while True:
            selected=None
            with lock('scheduler'):
                records=sorted(planned_runs(),key=_priority)
                # Finish each run's dev promptly, permitting early cost/feedback evidence.
                for stage in stages:
                    rec=next((r for r in records if _eligible(r,stage)),None)
                    if rec:
                        field={'train':'train_status','dev':'dev_status','test':'test_status'}[stage]
                        update_run(rec['run_id'],status='running',**{field:'running'},gpu=gpu,gpu_slot=slot,scheduler_pid=os.getpid())
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
                update_run(rec['run_id'],status='pending',**{field:'pending'},last_error=str(exc),pid=None,worker_pid=None)
                raise

def schedule(stages):
    with lock('orchestrator',blocking=False):
        reconcile_stale_runs()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(GPUS)*WORKERS_PER_GPU) as executor:
            jobs=[executor.submit(gpu_queue,gpu,stages,slot) for slot in range(WORKERS_PER_GPU) for gpu in GPUS]
            return [future.result() for future in jobs]

def _output_summary(result):
    if isinstance(result,dict) and 'records' in result and 'metrics' in result:
        return dict(identity=result['identity'], metrics={s:{k:m[k] for k in ('n','successes','success_rate','binomial_95ci')} for s,m in result['metrics'].items()}, wall_seconds=result.get('last_session_wall_seconds'))
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','validate-proposals','train','dev-eval','test','report','feedback','resume','all',
        'train-one','dev-one','test-one','debug-one'])
    parser.add_argument('--task',choices=TASKS)
    parser.add_argument('--gpu',type=int,choices=GPUS,help='Physical GPU for an individual worker (default: first authorized GPU)')
    parser.add_argument('--run-id')
    parser.add_argument('--debug-updates',type=int,default=100)
    parser.add_argument('--stop-after',type=int)
    args=parser.parse_args()
    planned_runs()
    if args.command=='prepare':
        from .prepare import prepare_task
        with gpu_context(args.gpu):
            for task in ([args.task] if args.task else TASKS):
                prepare_task(task)
    elif args.command=='validate-proposals':
        from .proposals import validate_all
        print(validate_all())
    elif args.command in ('train-one','debug-one'):
        assert args.run_id
        if args.command=='train-one':
            assert implementation_ready(run_record(args.run_id)), 'Candidate implementation audit must pass before formal training'
        with gpu_context(args.gpu):
            from .learning import train
            print(train(args.run_id,debug_updates=args.debug_updates if args.command=='debug-one' else None,stop_after=args.stop_after))
    elif args.command in ('dev-one','test-one'):
        assert args.run_id
        with gpu_context(args.gpu):
            from .evaluate import worker
            print(_output_summary(worker(args.run_id,'dev' if args.command=='dev-one' else 'test')))
    elif args.command=='train':
        if args.run_id:
            assert implementation_ready(run_record(args.run_id)), 'Candidate implementation audit must pass before formal training'
            with gpu_context(args.gpu):
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
        with gpu_context(args.gpu):
            from .report import generate
            print(generate())
    elif args.command=='feedback':
        if not args.task:
            parser.error('feedback requires --task')
        with gpu_context(args.gpu):
            from .report import feedback_bundle
            print(feedback_bundle(args.task))
    else:
        from .proposals import validate_all,baseline_config
        from .prepare import prepare_task
        # Prepare only as far as the first missing proposal, rather than delaying first training for all tasks.
        for task in TASKS:
            if not data_complete(task):
                with gpu_context(args.gpu):
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
        with gpu_context(args.gpu):
            from .report import generate
            print(generate())

if __name__=='__main__':
    main()
