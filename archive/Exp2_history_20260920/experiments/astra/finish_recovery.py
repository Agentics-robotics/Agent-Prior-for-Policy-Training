"""Supervise the four authorized recoveries, original training and test matrix."""
import argparse
import fcntl
import os
import shutil
import subprocess
import time

from appl.io import ROOT, PIXI, MANIFEST, atomic, read, digest
from appl.prior_policies.report import authorship
from appl.prior_policies.runner import train_submitted
from .runner import BASE, config, entries, api_identity, unchanged_framework, command, evaluate_study

CASES = [('tray_pack', 'deliver_block__h02', 1),
         ('two_block_sort', 'deliver_disengage__h01', 2),
         ('unstack_sort', 'acquire_lift__h02', 3),
         ('buffer_swap', 'buffer_red__h02', 5)]


def package(name, policy):
    entry, = [e for n,e in entries() if n == name and e['policy_id'] == policy]
    return config(name)['output'] / 'policies' / entry['skill_id'] / f"heuristic_{entry['heuristic_index']:02d}"


def train_case(name, policy, gpu):
    if (name, policy, gpu) not in CASES:
        raise ValueError('Unexpected recovery training assignment')
    unchanged_framework()
    folder = package(name, policy)
    submission = read(folder / 'submission.json')
    authorship(folder, submission)
    atomic(folder / 'model_identity.json', api_identity(folder / 'design/journal.sqlite'))
    if (folder / 'failure.json').exists():
        raise ValueError('A retained canonical failure still requires reconciliation')
    result = train_submitted(config(name), policy, gpu)
    subprocess.run(command(['check', '--task', name, '--policy-id', policy, '--gpu', str(gpu)]),
                   cwd=ROOT, check=True)
    print(dict(stage='trained_and_checked', task=name, policy=policy,
               updates=result['optimizer_steps']), flush=True)


def main():
    from appl.gpu import identity
    unchanged_framework()
    authorization = read(BASE / 'incidents/design_recovery_authorization.json')
    if not authorization['authorized'] or not read(BASE / 'incidents/buffer_red_h02_reload/recovery_authorization.json')['authorized']:
        raise ValueError('All four recoveries must be authorized')
    output = BASE / 'authorized_recovery/supervisor'
    output.mkdir(parents=True, exist_ok=False)
    with (BASE / 'owner.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        shutil.copyfile(BASE / 'coordinator.json', output / 'original_coordinator.json')
        atomic(BASE / 'coordinator.json', dict(pid=os.getpid(), started=time.time(),
            stage='authorized_recovery', source_sha256=digest(__file__),
            original_record=str(output / 'original_coordinator.json')))
        atomic(output / 'schedule.json', dict(cases=CASES, scientific_changes=False,
            formal_updates_per_policy=20000, automatic_retries=0,
            source_sha256=digest(__file__)))
        active = {}; finished = set(); failures = []; launched = set()
        while True:
            for key, job in list(active.items()):
                code = job['process'].poll()
                if code is None:
                    continue
                job['stdout'].close(); job['stderr'].close()
                atomic(job['folder'] / 'process_result.json', dict(returncode=code, finished=time.time()))
                if code:
                    failures.append(key)
                else:
                    finished.add(key)
                del active[key]
                print(dict(completed=key, returncode=code), flush=True)
            for name, policy, gpu in CASES:
                key = name + '/' + policy
                receipt = (BASE / 'incidents/buffer_red_h02_reload/continuation.json' if name == 'buffer_swap'
                           else BASE / 'incidents/design_recovery' / (name + '__' + policy + '.json'))
                if (key in launched or not receipt.exists() or read(receipt)['status'] != 'submitted'
                        or not (package(name, policy) / 'submission.json').exists()):
                    continue
                occupancy = identity(gpu)
                if occupancy['free_mib'] < 10000:
                    continue
                out = output / name / policy
                out.mkdir(parents=True, exist_ok=False)
                args = [PIXI, 'run', '--manifest-path', str(MANIFEST), '--locked', 'python',
                        '-m', 'experiments.exp2.astra.finish_recovery', '--train-task', name,
                        '--policy', policy, '--gpu', str(gpu)]
                stdout = (out / 'stdout.log').open('w'); stderr = (out / 'stderr.log').open('w')
                proc = subprocess.Popen(args, cwd=ROOT, stdout=stdout, stderr=stderr)
                atomic(out / 'process.json', dict(pid=proc.pid, command=args,
                    started=time.time(), gpu=gpu, occupancy=occupancy))
                active[key] = dict(process=proc, folder=out, stdout=stdout, stderr=stderr, gpu=gpu)
                launched.add(key)
                print(dict(stage='formal_training', policy=key, gpu=gpu), flush=True)
            design_results = {name:read(BASE / 'authorized_recovery' / name / 'process_result.json')
                for name in ('network', 'buffer')
                if (BASE / 'authorized_recovery' / name / 'process_result.json').exists()}
            atomic(BASE / 'status.json', dict(stage='authorized_recovery', time=time.time(),
                active=[dict(label=k, pid=j['process'].pid, gpu=j['gpu']) for k,j in active.items()],
                pending=4-len(launched), finished=len(finished), failed=failures,
                design_worker_results=design_results, original_ready_policies=32))
            if len(design_results) == 2 and not active:
                if failures or len(finished) != 4 or any(r['returncode'] != 0 for r in design_results.values()):
                    atomic(output / 'result.json', dict(status='retained_failure', finished=sorted(finished),
                        failures=failures, design_results=design_results, automatic_retry=False))
                    raise RuntimeError('Authorized recovery did not complete all four policies; receipts retained')
                break
            time.sleep(5)
        atomic(output / 'result.json', dict(status='all_four_trained_and_checked',
            finished=sorted(finished), updates=80000, design_results=design_results, time=time.time()))
        print(dict(stage='global_freeze_and_original_300_tests'), flush=True)
        evaluate_study()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-task'); parser.add_argument('--policy'); parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.train_task:
        train_case(args.train_task, args.policy, args.gpu)
    else:
        main()
