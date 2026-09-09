"""Locked, resumable experiment entry points. Failed stages return nonzero."""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

from .config import RUNS, TRAIN_CONFIG
from .utils import ROOT, atomic_json, object_hash, read_json, sha256


def fingerprint():
    paths = sorted((ROOT/'src/relative_dp').rglob('*.py'))
    paths = [p for p in paths if p.name not in ('report.py','cli.py','doctor.py')]
    paths += sorted((ROOT/'tests').glob('test_*.py'))
    paths += [ROOT/'pixi.toml', ROOT/'pixi.lock', ROOT/'pyproject.toml', ROOT/'data/dataset_manifest.json', ROOT/'artifacts/observation_schema.json']
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def freeze_experiment():
    from .collection import verify_manifest
    verify_manifest()
    identity = dict(config=TRAIN_CONFIG,runs=RUNS,file_hashes=fingerprint())
    path=ROOT/'artifacts/experiment_freeze.json'
    if path.exists():
        if read_json(path)!=identity:
            raise RuntimeError('Frozen experiment differs. Preserve old records and diagnose before any formal work.')
    else:
        atomic_json(path,identity)
    return identity


def preflight():
    import pytest
    from .collection import verify_manifest
    from .train import debug_train
    from .evaluate import debug_rollout
    verify_manifest()
    identity=fingerprint()
    path=ROOT/'artifacts/preflight.json'
    if path.exists():
        old=read_json(path)
        if old.get('file_hashes')==identity and old.get('passed') and all(sha256(ROOT/p)==h for p,h in old['debug_checkpoint_hashes'].items()):
            return old
        raise RuntimeError('Preflight identity changed; inspect and archive prior debug artifacts before rerun')
    start=time.perf_counter()
    code=int(pytest.main(['-q',str(ROOT/'tests'),'--junitxml='+str(ROOT/'artifacts/preflight_tests.xml')]))
    if code:
        raise RuntimeError(f'Correctness tests failed with exit code {code}')
    debug=debug_train(updates=10)
    ckpt=ROOT/debug['checkpoints']['10']['path']
    rollout=debug_rollout(ckpt)
    if rollout.get('exception'):
        raise RuntimeError(f'Debug rollout failed: {rollout["exception"]}')
    result=dict(passed=True,file_hashes=identity,test_exit_code=code,
                debug_checkpoint_hashes={str(ckpt.relative_to(ROOT)):sha256(ckpt)},
                debug_rollout=rollout,elapsed_seconds=time.perf_counter()-start)
    atomic_json(path,result)
    freeze_experiment()
    return result


def train_isolated(run_ids=None):
    """Sequential fresh processes prevent inference CUDA-graph state leaking into training.

    Formal model/data/training code and saved RNG states are unchanged. The
    parent alone owns the pipeline lock; each child calls train_run directly.
    """
    from .train import validate_run_complete
    pixi=shutil.which('pixi') or str(Path.home()/'.pixi/bin/pixi')
    for run_id in (run_ids or [r['run_id'] for r in RUNS]):
        if validate_run_complete(run_id):
            print(f'{run_id}: validated complete; skipping',flush=True)
            continue
        command=[pixi,'run','python','-c',
                 'import sys; from relative_dp.train import train_run; train_run(sys.argv[1])',run_id]
        child=subprocess.Popen(command,cwd=ROOT,start_new_session=True)
        atomic_json(ROOT/'artifacts/active_training_process.json',dict(parent_pid=os.getpid(),child_process_group=child.pid,run_id=run_id,command=command))
        def forward_signal(signum,frame):
            if child.poll() is None:
                os.killpg(child.pid,signum)
        handlers={sig:signal.signal(sig,forward_signal) for sig in (signal.SIGINT,signal.SIGTERM)}
        try:
            code=child.wait()
        finally:
            for sig,handler in handlers.items(): signal.signal(sig,handler)
        if code:
            raise subprocess.CalledProcessError(code,command)
        validate_run_complete(run_id,raise_error=True)


class Progress:
    def __init__(self):
        self.phase='starting'; self.finished=False; self.failed=None
        self.started=time.time(); self.stop=threading.Event()
        self.stages=[]; self.stage_started=time.perf_counter()
    def write(self):
        records=[]
        for run in RUNS:
            path=ROOT/'runs'/run['run_id']/'status.json'
            status=read_json(path) if path.exists() else {}
            records.append((run['run_id'],status.get('step',0),bool(status.get('complete',False))))
        text=['# Round 1 progress','',f'Phase: {self.phase}',f'PID: {os.getpid()}',
              f'Updated UTC: {time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}',
              f'Invocation elapsed: {time.time()-self.started:.1f} seconds','',
              '| Run | Saved updates | Completion marker |','|---|---:|---|']
        text += [f'| {r} | {s} / 20000 | {"complete" if c else "incomplete"} |' for r,s,c in records]
        text += ['',f'All pipeline stages finished: {self.finished}',f'Error: {self.failed or "none"}','',
                 'Project: `/home/users/oscar/Agent_Training/agent_training`.',
                 'Progress above reads saved checkpoint status. Authoritative completion checks validate configurations and hashes.',
                 'Background output: `logs/round1.log` (when started with the documented background command).','',
                 'Resume all stages: `/home/users/oscar/.pixi/bin/pixi run round1`.',
                 'Resume one run: `/home/users/oscar/.pixi/bin/pixi run train-round1 --run-id drawer_raw_n20_s0`.',
                 'Training checkpoints preserve optimizer, EMA, independent RNGs and sample position. Do not start a second process while the recorded PID is active.','',
                 'Final evaluation/report remain incomplete until `results/test_freeze.json` validates 160 records and `ROUND1_REPORT.md` is generated.']
        tmp=ROOT/f'PROGRESS.md.tmp.{threading.get_ident()}'
        tmp.write_text('\n'.join(text)+'\n'); os.replace(tmp,ROOT/'PROGRESS.md')
    def ticker(self):
        while not self.stop.wait(30):
            self.write()
    def stage(self,name):
        if self.phase!='starting':
            self.stages.append(dict(stage=self.phase,elapsed_seconds=time.perf_counter()-self.stage_started))
        self.stage_started=time.perf_counter()
        self.phase=name; self.write(); print(f'\nRound1 stage: {name}',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['doctor','collect','preflight','train','evaluate','report','round1'])
    parser.add_argument('--run-id',action='append',choices=[r['run_id'] for r in RUNS])
    parser.add_argument('--no-videos',action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT)
    (ROOT/'artifacts').mkdir(exist_ok=True)
    lock=(ROOT/'artifacts/pipeline.lock').open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another project pipeline is active; inspect PROGRESS.md and existing logs.')
    progress=Progress()
    thread=threading.Thread(target=progress.ticker,daemon=True); thread.start()
    try:
        if args.command in ('doctor','round1'):
            progress.stage('doctor')
            from .doctor import doctor
            from .environment import observation_schema
            doctor(); observation_schema(ROOT/'artifacts/observation_schema.json')
        if args.command in ('collect','round1'):
            progress.stage('collect / validate frozen data')
            from .collection import collect_round1
            collect_round1()
        if args.command in ('preflight','train','round1'):
            progress.stage('correctness checks and independent debug training / rollout')
            preflight(); freeze_experiment()
        if args.command in ('train','round1'):
            progress.stage('four formal trainings' if not args.run_id else 'selected formal training')
            train_isolated(run_ids=args.run_id)
        if args.command in ('evaluate','round1'):
            progress.stage('development checkpoint selection, final IID/OOD evaluation, videos')
            freeze_experiment()
            from .evaluate import evaluate_round1
            evaluate_round1(videos=not args.no_videos)
        if args.command in ('report','round1'):
            progress.stage('result verification and report')
            from .report import generate_report
            generate_report()
        progress.stages.append(dict(stage=progress.phase,elapsed_seconds=time.perf_counter()-progress.stage_started))
        progress.finished=args.command=='round1'
        progress.phase='complete' if progress.finished else f'{args.command} finished'
    except BaseException as exc:
        progress.failed=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        progress.stop.set(); thread.join(timeout=2); progress.write()
        timing=dict(command=args.command,pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(progress.started)),
                    elapsed_seconds=time.time()-progress.started,stages=progress.stages,
                    finished=progress.finished,error=progress.failed)
        atomic_json(ROOT/'artifacts'/f'invocation_{os.getpid()}.json',timing)
        if args.command=='round1':
            atomic_json(ROOT/'artifacts/pipeline_timing.json',timing)
        lock.close()


if __name__=='__main__':
    main()
