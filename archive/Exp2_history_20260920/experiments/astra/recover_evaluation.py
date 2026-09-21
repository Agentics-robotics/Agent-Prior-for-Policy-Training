"""One explicitly authorized reset retest per retained HTTP-interrupted cell.

Only output routing and fail-stop admission differ from the frozen evaluator.
Original scientific inputs, API prompts/outputs and all old attempts are retained.
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
from . import runner, transport

BASE = runner.BASE
OUT = BASE / 'evaluation_recovery'
INCIDENT = BASE / 'incidents/evaluation_outage'
MODULE = 'experiments.exp2.astra.recover_evaluation'
GPUS = [1, 2, 3, 5]
ROUND = 'initial'
AUTHORIZATION = INCIDENT / 'retest_authorization.json'


def select_round(value):
    global OUT, ROUND, AUTHORIZATION
    if value not in ('initial', '20260918'):
        raise ValueError('Unknown explicitly authorized recovery round')
    ROUND = value
    if value == '20260918':
        OUT = BASE / 'evaluation_recovery_20260918'
        AUTHORIZATION = INCIDENT / 'resume_authorization_20260918.json'


def key(case):
    return case['task'], case['condition'], case['seed']


def authorized_cases():
    proposal = read(INCIDENT / 'retest_proposal.json')
    body = {k: v for k, v in proposal.items() if k != 'proposal_sha256'}
    if object_hash(body) != proposal['proposal_sha256']:
        raise ValueError('Retest proposal changed')
    authorization = read(AUTHORIZATION)
    if (authorization['proposal_sha256'] != proposal['proposal_sha256']
            or authorization['maximum_new_episodes'] != 89
            or authorization['status'] != 'authorized'):
        raise ValueError('Missing exact bounded retest authorization')
    if ROUND == '20260918':
        previous = BASE / 'evaluation_recovery'
        if (authorization['previous_validation_sha256'] != digest(previous / 'validation.json')
                or authorization['previous_execution_sha256'] != digest(previous / 'execution.json')
                or read(previous / 'validation.json')['physical_steps_added'] != 0
                or read(previous / 'validation.json')['retests_completed'] != 0
                or authorization['resume_includes_previous_zero_step_case'] is not True):
            raise ValueError('Explicit zero-step resumption provenance mismatch')
    freeze = read(BASE / 'study_freeze.json')
    if (object_hash(freeze['study']) != freeze['study_sha256']
            or freeze['study_sha256'] != proposal['study_sha256']
            or digest(runner.__file__) != freeze['study']['runner_sha256']
            or runner.unchanged_framework() != freeze['study']['framework_source']):
        raise ValueError('Original frozen execution changed')
    if digest(transport.__file__) != '0296eea2c9148a12ab64f7323b367a31d0e85ab6f5f3539fe2b955a997652cb1':
        raise ValueError('Original request gate changed')
    cases = proposal['cases']
    unknown = {key(e) for e in read(BASE / 'results.json')['episodes']
               if e['method'] == 'APPL_6_xhigh' and not e['completed']}
    if len(cases) != 89 or len({key(c) for c in cases}) != 89 or {key(c) for c in cases} != unknown:
        raise ValueError('Retest cases do not exactly match the original unknown cells')
    for case in cases:
        old = Path(case['original_root'])
        if (digest(old / 'failure.json') != case['original_failure_sha256']
                or (old / 'result.json').exists()):
            raise ValueError('Original interruption changed')
    return cases


def preflight():
    from appl.prior_policies.deploy import library
    cases = authorized_cases()
    freeze = read(BASE / 'study_freeze.json')['study']
    models = 0
    for name in runner.NAMES:
        cfg = runner.config(name)
        expected = freeze['tasks'][name]
        for path, field in [(cfg['_path'], 'configuration_sha256'),
                            (BASE / name / 'task.json', 'task_spec_sha256'),
                            (cfg['completion_contract'], 'completion_contract_sha256'),
                            (cfg['dataset'] / 'manifest.json', 'dataset_manifest_sha256'),
                            (cfg['output'] / 'normalization.json', 'normalization_sha256')]:
            if digest(path) != expected[field]:
                raise ValueError('Frozen input changed: ' + str(path))
        policies = library(cfg)
        actual = {k: dict(version=p['version'], checkpoint_sha256=p['checkpoint_sha256'])
                  for k, p in policies.items()}
        if actual != expected['policies']:
            raise ValueError('Frozen policy library changed')
        models += len(actual)
    if models != 36:
        raise ValueError('Expected exactly 36 frozen policies')
    return dict(passed=True, cases=len(cases), frozen_models=models,
                expected_request_fields=dict(model=runner.MODEL, reasoning=dict(effort=runner.EFFORT)),
                API_requests=0, simulator_steps=0, source_sha256=digest(__file__))


def inventory():
    files = {}
    for name in runner.NAMES:
        for path in sorted((BASE / name / 'evaluation/APPL').rglob('*')):
            if path.is_file() and not path.name.endswith(('-shm', '-wal')):
                files[str(path.relative_to(BASE))] = digest(path)
    for path in sorted((BASE / 'jobs/evaluation').rglob('*')):
        if path.is_file():
            files[str(path.relative_to(BASE))] = digest(path)
    for name in ('study_freeze.json', 'completion.json', 'results.json', 'episodes.csv',
                 'REPORT.md', 'ANALYSIS.md', 'evaluation_audit.json', 'videos.json'):
        files[name] = digest(BASE / name)
    if ROUND == '20260918':
        for path in sorted((BASE / 'evaluation_recovery').rglob('*')):
            if path.is_file() and 'naive_DP' not in path.parts and not path.name.endswith(('-shm', '-wal')):
                files[str(path.relative_to(BASE))] = digest(path)
    return files


def evaluate(case):
    if key(case) not in {key(c) for c in authorized_cases()}:
        raise ValueError('Unapproved retest cell')
    if digest(__file__) != read(OUT / 'preflight.json')['source_sha256']:
        raise ValueError('Recovery executor changed after launch')
    name, condition, seed = key(case)
    root = OUT / name / 'evaluation/APPL' / condition / str(seed)
    if root.exists():
        raise ValueError('This retest was already attempted; no second retry')
    transport.install_episode_gate(BASE, root, slots=4)
    from appl.agent import ResponsesClient
    forward = ResponsesClient.respond

    def verify_request(client, request):
        if (request.get('model'), request.get('reasoning', {}).get('effort')) != ('gpt-6-astra', 'xhigh'):
            raise ValueError('Actual outgoing request violates the persistent API identity rule')
        return forward(client, request)

    ResponsesClient.respond = verify_request
    evaluator = runner.bind_evaluator()
    import appl.scaleup.protocol as protocol
    protocol.BASE = OUT
    evaluator.BASE = OUT
    result = evaluator.evaluate(name, 'APPL', condition, seed)
    atomic(root / 'model_identity.json', runner.api_identity(root / 'api/journal.sqlite'))
    return result


def launch(case, gpu):
    from appl.gpu import identity
    name, condition, seed = key(case)
    job = OUT / 'jobs' / name / condition / str(seed)
    job.mkdir(parents=True, exist_ok=False)
    command = [PIXI, 'run', '--manifest-path', str(MANIFEST), '--locked', 'python', '-m', MODULE,
               'evaluate', '--round', ROUND, '--task', name, '--condition', condition, '--seed', str(seed), '--gpu', str(gpu)]
    stdout = (job / 'stdout.log').open('w')
    stderr = (job / 'stderr.log').open('w')
    occupancy = identity(gpu)
    process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
    atomic(job / 'process.json', dict(pid=process.pid, command=command, started=time.time(), occupancy=occupancy))
    return dict(case=case, gpu=gpu, job=job, process=process, stdout=stdout, stderr=stderr)


def run():
    from appl.gpu import identity
    with (BASE / 'owner.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if OUT.exists():
            raise ValueError('Recovery run already exists; inspect it instead of retrying')
        checked = preflight()
        cases = authorized_cases()
        if read(Path(cases[0]['original_root']) / 'failure.json')['physical_steps'] != 0:
            raise ValueError('First recovery must be a zero-step interrupted cell')
        OUT.mkdir()
        atomic(OUT / 'preflight.json', checked)
        atomic(OUT / 'original_artifacts.json', inventory())
        atomic(OUT / 'allocation.json', dict(devices=GPUS, inspected=[identity(g) for g in GPUS],
            max_episode_slots=4, max_API_requests=4, first_episode_runs_alone=True))
        atomic(OUT / 'process.json', dict(pid=os.getpid(), started=time.time(), command=sys.argv,
            proposal_sha256=read(INCIDENT / 'retest_proposal.json')['proposal_sha256']))
        shutil.copyfile(__file__, OUT / 'frozen_recovery_executor.py')
        shutil.copyfile(BASE / 'study_freeze.json', OUT / 'study_freeze.json')
        for name in runner.NAMES:
            folder = OUT / name
            folder.mkdir()
            shutil.copyfile(BASE / name / 'task.json', folder / 'task.json')
            (folder / 'evaluation').mkdir()
            (folder / 'evaluation/naive_DP').symlink_to((runner.OLD / name / 'evaluation/naive_DP').resolve())
        pending = deque(cases)
        active = {}
        finished = []
        failed = []
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
            slots = GPUS if finished else GPUS[:1]
            for slot, gpu in enumerate(slots):
                if slot in active or not pending or failed:
                    continue
                if identity(gpu)['free_mib'] < 10000:
                    continue
                active[slot] = launch(pending.popleft(), gpu)
            atomic(OUT / 'status.json', dict(finished=finished, failed=failed, pending=list(pending),
                active=[dict(task=j['case']['task'], condition=j['case']['condition'], seed=j['case']['seed'],
                             gpu=j['gpu'], pid=j['process'].pid) for j in active.values()], time=time.time()))
            if failed and not active:
                break
            if pending or active:
                time.sleep(5)
        before = read(OUT / 'original_artifacts.json')
        after = inventory()
        if before != after:
            raise ValueError('Original evaluation artifacts changed during the retests')
        atomic(OUT / 'execution.json', dict(finished=finished, failed=failed, unstarted=list(pending),
            all_89_completed=len(finished) == 89 and not failed, original_files_unchanged=len(before),
            authorization_sha256=digest(AUTHORIZATION),
            additional_training_updates=0, automatic_retries=0, ended=time.time()))
        print(dict(attempted=len(finished), failed=len(failed), unstarted=len(pending)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'run', 'evaluate'])
    parser.add_argument('--round', choices=['initial', '20260918'], default='initial')
    parser.add_argument('--task', choices=runner.NAMES)
    parser.add_argument('--condition', choices=['ID', 'OOD'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--gpu', type=int, choices=GPUS)
    parser.add_argument('--device-isolated', action='store_true')
    args = parser.parse_args()
    select_round(args.round)
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
