"""Execute the explicitly authorized 21-case continuation after the retained 502.

Reuse the frozen evaluator and transport unchanged. Never retry within this run.
"""
import argparse
from collections import deque
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from appl.io import ROOT, PIXI, MANIFEST, atomic, digest, object_hash, read
from . import recover_evaluation as previous
from . import runner

BASE = runner.BASE
PRIOR = BASE / 'evaluation_recovery_20260918'
OUT = BASE / 'evaluation_followup_20260918'
INCIDENT = BASE / 'incidents/evaluation_outage'
PROPOSAL = INCIDENT / 'followup_proposal_20260918.json'
AUTHORIZATION = INCIDENT / 'followup_authorization_20260918.json'
MODULE = 'experiments.exp2.astra.followup_evaluation'
GPUS = [1, 2, 3, 5]
key = previous.key


def bind():
    previous.select_round('20260918')
    previous.OUT = OUT


def authorized_cases():
    proposal = read(PROPOSAL)
    if object_hash({k: v for k, v in proposal.items() if k != 'proposal_sha256'}) != proposal['proposal_sha256']:
        raise ValueError('Follow-up proposal changed')
    authorization = read(AUTHORIZATION)
    execution = read(PRIOR / 'execution.json')
    if (authorization['status'] != 'authorized'
            or authorization['proposal_sha256'] != proposal['proposal_sha256']
            or authorization['previous_execution_sha256'] != digest(PRIOR / 'execution.json')
            or authorization['maximum_new_episodes'] != 21
            or authorization['maximum_attempts_per_case'] != 1):
        raise ValueError('Missing exact follow-up authorization')
    cases = proposal['cases']
    expected = {key(c) for c in execution['unstarted'] + execution['failed']}
    unknown = {key(c) for c in read(PRIOR / 'results.json')['episodes']
               if c['method'] == 'APPL_6_xhigh' and not c['completed']}
    if len(cases) != 21 or {key(c) for c in cases} != expected or expected != unknown:
        raise ValueError('Follow-up must match the exact 21 unknown cells')
    for case in cases:
        name, condition, seed = key(case)
        root = PRIOR / name / 'evaluation/APPL' / condition / str(seed)
        if case['operation'] == 'one_additional_reset_retest':
            if digest(root / 'failure.json') != case['latest_failure_sha256'] or (root / 'result.json').exists():
                raise ValueError('Retained interrupted attempt changed')
        elif case['operation'] == 'first_attempt_under_existing_20260918_authorization':
            if root.exists():
                raise ValueError('An unstarted cell was already attempted')
        else:
            raise ValueError('Unknown continuation operation')
    return cases


def inventory():
    records = previous.inventory()
    for path in sorted(PRIOR.rglob('*')):
        if path.is_file() and 'naive_DP' not in path.parts and not path.name.endswith(('-shm', '-wal')):
            records[str(path.relative_to(BASE))] = digest(path)
    return records


def preflight():
    bind()
    checked = previous.preflight()
    cases = authorized_cases()
    receipt = read(PRIOR / 'report_receipt.json')
    if (receipt['complete_outcomes'] != 279 or receipt['unknown'] != 21
            or receipt['resumed_outcomes_audited'] != 69
            or receipt['new_video_validations'] != 69):
        raise ValueError('Previous recovery has not been fully audited')
    return dict(checked, cases=len(cases), followup_source_sha256=digest(__file__),
                prior_report_receipt_sha256=digest(PRIOR / 'report_receipt.json'))


def evaluate(case):
    bind()
    if key(case) not in {key(c) for c in authorized_cases()}:
        raise ValueError('Cell outside the authorized follow-up')
    if digest(__file__) != read(OUT / 'preflight.json')['followup_source_sha256']:
        raise ValueError('Follow-up executor changed')
    return previous.evaluate(case)


def launch(case, gpu):
    from appl.gpu import identity
    name, condition, seed = key(case)
    job = OUT / 'jobs' / name / condition / str(seed)
    job.mkdir(parents=True, exist_ok=False)
    command = [PIXI, 'run', '--manifest-path', str(MANIFEST), '--locked', 'python', '-m', MODULE,
               'evaluate', '--task', name, '--condition', condition, '--seed', str(seed), '--gpu', str(gpu)]
    stdout, stderr = (job / 'stdout.log').open('w'), (job / 'stderr.log').open('w')
    occupancy = identity(gpu)
    process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
    atomic(job / 'process.json', dict(pid=process.pid, command=command, started=time.time(), occupancy=occupancy))
    return dict(case=case, gpu=gpu, job=job, process=process, stdout=stdout, stderr=stderr)


def run():
    from appl.gpu import identity
    bind()
    with (BASE / 'owner.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if OUT.exists():
            raise ValueError('Follow-up already exists; no automatic retry')
        checked = preflight()
        cases = authorized_cases()
        OUT.mkdir()
        atomic(OUT / 'preflight.json', checked)
        before = inventory()
        atomic(OUT / 'original_artifacts.json', before)
        atomic(OUT / 'allocation.json', dict(devices=GPUS, inspected=[identity(g) for g in GPUS],
            max_episode_slots=4, max_API_requests=4, first_episode_runs_alone=True))
        atomic(OUT / 'process.json', dict(pid=os.getpid(), started=time.time(), command=sys.argv,
            authorization_sha256=digest(AUTHORIZATION), proposal_sha256=read(PROPOSAL)['proposal_sha256']))
        shutil.copyfile(__file__, OUT / 'frozen_followup_executor.py')
        shutil.copyfile(previous.__file__, OUT / 'frozen_recovery_executor.py')
        shutil.copyfile(BASE / 'study_freeze.json', OUT / 'study_freeze.json')
        for name in runner.NAMES:
            folder = OUT / name
            folder.mkdir()
            shutil.copyfile(BASE / name / 'task.json', folder / 'task.json')
            (folder / 'evaluation').mkdir()
            (folder / 'evaluation/naive_DP').symlink_to((runner.OLD / name / 'evaluation/naive_DP').resolve())
        pending, active, finished, failed = deque(cases), {}, [], []
        while pending or active:
            for slot, job in list(active.items()):
                code = job['process'].poll()
                if code is None:
                    continue
                job['stdout'].close()
                job['stderr'].close()
                atomic(job['job'] / 'process_result.json', dict(returncode=code, finished=time.time()))
                record = dict(task=job['case']['task'], condition=job['case']['condition'],
                              seed=job['case']['seed'], returncode=code)
                finished.append(record)
                if code:
                    failed.append(record)
                del active[slot]
            for slot, gpu in enumerate(GPUS if finished else GPUS[:1]):
                if slot in active or not pending or failed or identity(gpu)['free_mib'] < 10000:
                    continue
                active[slot] = launch(pending.popleft(), gpu)
            atomic(OUT / 'status.json', dict(finished=finished, failed=failed, pending=list(pending),
                active=[dict(task=j['case']['task'], condition=j['case']['condition'], seed=j['case']['seed'],
                             gpu=j['gpu'], pid=j['process'].pid) for j in active.values()], time=time.time()))
            if failed and not active:
                break
            if pending or active:
                time.sleep(5)
        if before != inventory():
            raise ValueError('Earlier artifacts changed during the follow-up')
        atomic(OUT / 'execution.json', dict(finished=finished, failed=failed, unstarted=list(pending),
            all_21_completed=len(finished) == 21 and not failed, original_files_unchanged=len(before),
            authorization_sha256=digest(AUTHORIZATION), additional_training_updates=0,
            automatic_retries=0, ended=time.time()))
        print(dict(attempted=len(finished), failed=len(failed), unstarted=len(pending)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'run', 'evaluate'])
    parser.add_argument('--task', choices=runner.NAMES)
    parser.add_argument('--condition', choices=['ID', 'OOD'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--gpu', type=int, choices=GPUS)
    parser.add_argument('--device-isolated', action='store_true')
    args = parser.parse_args()
    if args.command == 'evaluate':
        if not args.device_isolated:
            from appl.gpu import launch as isolate
            return isolate(args.gpu, sys.argv[1:], module=MODULE)
        nodes = {v.name for v in Path('/dev').glob('nvidia*') if v.name.removeprefix('nvidia').isdigit()}
        if nodes != {'nvidia' + os.environ['APPL_GPU_MINOR']}:
            raise RuntimeError('Expected one isolated physical GPU')
        os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['APPL_GPU_UUID']
        os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'
        evaluate(dict(task=args.task, condition=args.condition, seed=args.seed))
    elif args.command == 'preflight':
        print(preflight(), flush=True)
    else:
        run()


if __name__ == '__main__':
    main()
