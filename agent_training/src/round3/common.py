"""Atomic experiment bookkeeping; no simulator, model, or designer oracle."""
import contextlib
import fcntl
import os
from datetime import datetime, timezone
from relative_dp.utils import ROOT, atomic_json, read_json, sha256, object_hash

R3 = ROOT / 'round3'
TASKS = ['pick-place-wall', 'assembly', 'drawer', 'door', 'peg-insert-side', 'stick-push']
NS = [2, 5, 10, 20]
CANDIDATES = ['B0', 'P1', 'P2', 'P3']
GPUS = [0, 2]
STATUSES = {'pending', 'pending-design', 'running', 'completed', 'invalid', 'blocked'}

def now():
    return datetime.now(timezone.utc).isoformat()

@contextlib.contextmanager
def lock(name, blocking=True):
    path = R3 / 'locks' / (name + '.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)

def run_id(task, candidate_id, n):
    return f'r3_{task}_{candidate_id}_n{n}_s0'

def initial_config():
    return dict(tasks=TASKS, demonstration_counts=NS, train_seeds=[0],
        candidates=CANDIDATES, agent_backend='codex_session', design_model='unavailable',
        session_id='unavailable', token_count='unavailable', api_cost='unavailable',
        initial_runs=96, maximum_formal_runs=108, updates_per_system=20000,
        batch_chunks_per_system_update=128, gpu_devices=GPUS, maximum_simultaneous_gpus=2,
        initial_design_exposure='D2 evidence bundle for isolated subagents; infrastructure root has historical exposure',
        revisions=dict(initial_feasibility=1, dev_feedback=1, feedback_feasibility=1),
        dev_episodes=dict(IID=10, C=20, E=20), test_episodes=dict(IID=20, C=40, E=40),
        split_version='round3-diagonal-v1', primary_checkpoint='20k EMA',
        maximum_episode_steps=500, specification_sha256=sha256(R3/'ROUND3_SPEC.txt'))

def initialize():
    with lock('manifest'):
        path = R3/'run_manifest.json'
        if path.exists():
            return read_json(path)
        runs = [dict(run_id=run_id(t,c,n), task=t, candidate_id=c, train_n=n,
            train_seed=0, status='pending-design' if c != 'B0' else 'pending',
            train_status='pending', dev_status='pending', test_status='pending',
            formal=True, feedback_revision=False) for t in TASKS for n in NS for c in CANDIDATES]
        state = dict(schema_version=1, created_at=now(), updated_at=now(), config=initial_config(), runs=runs)
        atomic_json(path,state)
        return state

def planned_runs():
    return initialize()['runs']

def run_record(identifier):
    return next(r for r in planned_runs() if r['run_id'] == identifier)

def update_run(identifier, **fields):
    if 'status' in fields and fields['status'] not in STATUSES:
        raise ValueError(fields['status'])
    initialize()
    with lock('manifest'):
        state = read_json(R3/'run_manifest.json')
        rec = next(r for r in state['runs'] if r['run_id'] == identifier)
        rec.update(fields, updated_at=now())
        state['updated_at'] = now()
        atomic_json(R3/'run_manifest.json',state)
        return rec

def add_revision(task):
    initialize()
    with lock('manifest'):
        state = read_json(R3/'run_manifest.json')
        for n in (5,20):
            identifier=run_id(task,'P4',n)
            if any(r['run_id']==identifier for r in state['runs']):
                continue
            state['runs'].append(dict(run_id=identifier, task=task, candidate_id='P4',train_n=n,
                train_seed=0,status='pending',train_status='pending',dev_status='pending',test_status='pending',
                formal=True,feedback_revision=True))
        assert len(state['runs']) <= 108
        state['updated_at']=now()
        atomic_json(R3/'run_manifest.json',state)

def checkpoint_dir(identifier):
    return R3/'checkpoints'/identifier

def implementation_path(task,candidate):
    return R3/'configs'/task/(candidate+'.json')

def event(kind, **fields):
    import json
    with lock('events'):
        with (R3/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=now(),pid=os.getpid(),kind=kind,**fields))+'\n')
