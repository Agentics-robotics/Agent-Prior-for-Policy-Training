"""One user-authorized reset of the historical APPL 5.5 interrupted cell.

Preserve the historical high effort for this exact reproduction. New experiment
defaults remain xhigh. Only output routing and the documented check-only engine
revision differ from the original frozen evaluator; no policy is retrained.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

from appl.io import ROOT, PIXI, MANIFEST, atomic, read, digest, object_hash

OLD = ROOT / 'runs/exp2/M1_scaleup'
OUT = OLD / 'evaluation_recovery_20260919'
NAME, CONDITION, SEED = 'buffer_swap', 'ID', 20116
ORIGINAL = OLD / NAME / 'evaluation/APPL' / CONDITION / str(SEED)
EPISODE = OUT / NAME / 'evaluation/APPL' / CONDITION / str(SEED)
MODULE = 'experiments.exp2.analysis.recover_scaleup_unknown'


def inventory():
    paths = list(ORIGINAL.rglob('*'))
    paths += [OLD / n for n in ('results.json', 'completion.json', 'study_freeze.json')]
    paths += [ROOT / 'runs/exp2/single_policy_astra_xhigh' / n
              for n in ('results.json', 'completion.json')]
    return {str(p): digest(p) for p in paths if p.is_file()
            and not p.name.endswith(('-shm', '-wal'))}


def preflight():
    from appl.scaleup.protocol import config
    from appl.prior_policies.deploy import library
    from experiments.exp2.astra.runner import unchanged_framework
    cfg = config(NAME)
    frozen = read(OLD / 'study_freeze.json')
    assert object_hash(frozen['study']) == frozen['study_sha256']
    assert not (ORIGINAL / 'result.json').exists()
    assert read(ORIGINAL / 'failure.json')['physical_steps'] == 650
    assert (cfg['api']['model'], cfg['api']['reasoning_effort']) == ('gpt-5.5', 'high')
    task = frozen['study']['tasks'][NAME]
    for path, key in ((cfg['_path'], 'configuration_sha256'),
                      (cfg['completion_contract'], 'completion_contract_sha256'),
                      (cfg['dataset'] / 'manifest.json', 'dataset_manifest_sha256'),
                      (cfg['output'] / 'normalization.json', 'normalization_sha256'),
                      (OLD / NAME / 'task.json', 'task_spec_sha256')):
        assert digest(path) == task[key], str(path)
    policies = library(cfg)
    assert {k: dict(version=p['version'], checkpoint_sha256=p['checkpoint_sha256'])
            for k, p in policies.items()} == task['policies']
    return dict(passed=True, framework_source=unchanged_framework(),
                source_sha256=digest(__file__), frozen_policies=len(policies),
                original_study_sha256=frozen['study_sha256'],
                model='gpt-5.5', effort='high', training_updates=0)


def evaluate():
    assert digest(__file__) == read(OUT / 'preflight.json')['source_sha256']
    assert not EPISODE.exists(), 'One authorized attempt only; no automatic retry'
    from appl.agent import ResponsesClient
    original_respond = ResponsesClient.respond
    first = True

    def checked_respond(client, request):
        nonlocal first
        assert (request['model'], request['reasoning']['effort']) == ('gpt-5.5', 'high')
        if first:
            assert request == read(ORIGINAL / 'api/api/0001.request.json')
            first = False
        return original_respond(client, request)

    ResponsesClient.respond = checked_respond
    import appl.scaleup.protocol as protocol
    from appl.scaleup import evaluate as evaluator
    protocol.BASE = OUT
    evaluator.BASE = OUT
    evaluator.evaluate(NAME, 'APPL', CONDITION, SEED)


def audit():
    from appl.scaleup.protocol import config
    from appl.scaleup.report import audit_episode, video
    from appl.prior_policies.report import api_accounting
    from experiments.exp2.analysis.audit_scaleup import audit_api
    from experiments.exp2.astra.report import verify_video
    assert read(OUT / 'process_result.json')['returncode'] == 0
    assert inventory() == read(OUT / 'original_artifacts.json')
    result = read(EPISODE / 'result.json')
    cfg = config(NAME)
    contract = read(cfg['completion_contract'])
    measured = audit_episode(EPISODE, contract, result)
    initial = read(EPISODE / 'initial_state.json')
    assert initial == read(ORIGINAL / 'initial_state.json')
    assert initial == read(OLD / NAME / 'evaluation/naive_DP' / CONDITION / str(SEED) / 'initial_state.json')
    assert read(EPISODE / 'prompt.json') == read(ORIGINAL / 'prompt.json')
    assert read(EPISODE / 'api/api/0001.request.json') == read(ORIGINAL / 'api/api/0001.request.json')
    trace = [json.loads(s) for s in (EPISODE / 'trace.jsonl').read_text().splitlines()]
    evidence = audit_api(EPISODE, contract, trace, initial, result,
                         read(OLD / 'study_freeze.json')['study']['tasks'][NAME])
    db = sqlite3.connect((EPISODE / 'api/journal.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
    rows = db.execute('SELECT status,request,response FROM api').fetchall()
    db.close()
    for state, request, response in rows:
        req, res = json.loads(request), json.loads(response)
        assert state == 'consumed'
        assert (req['model'], req['reasoning']['effort']) == ('gpt-5.5', 'high')
        assert (res['model'], res['reasoning']['effort']) == ('gpt-5.5', 'high')
    assert result['steps'] <= 5000
    assert not result['success'] or result['steps'] == measured['first_success']
    video(EPISODE)
    clip = verify_video(EPISODE)
    row = dict(task=NAME, condition=CONDITION, seed=SEED, method='APPL_5_5_high',
               completed=True, success=result['success'], status=result['status'], **measured,
               reused_original_result=False, selected_attempt='authorized_reset_retest_20260919',
               original_episode_root=str(ORIGINAL), episode_root=str(EPISODE))
    for cap in (1500, 3000, 5000):
        row['success_by_' + str(cap)] = measured['first_success'] is not None and measured['first_success'] <= cap
    value = dict(status='complete', passed=True, replacement=row, API_audit=evidence,
                 API_accounting=api_accounting(EPISODE / 'api/journal.sqlite'), video=clip,
                 original_files_unchanged=len(inventory()), initial_state_identical=True,
                 initial_prompt_identical=True, first_request_identical=True,
                 actual_model='gpt-5.5', actual_effort='high', additional_training_updates=0,
                 physical_reset_attempts=1, automatic_retries=0,
                 result_sha256=digest(EPISODE / 'result.json'), trace_sha256=digest(EPISODE / 'trace.jsonl'),
                 video_sha256=digest(EPISODE / 'replay.mp4'), ended=time.time())
    atomic(OUT / 'completion.json', value)
    print(dict(status='complete', success=result['success'], steps=result['steps'], API_calls=len(rows)), flush=True)


def run():
    from appl.gpu import identity
    assert not OUT.exists(), 'This authorization is bounded to one reset attempt'
    checked = preflight()
    occupancy = identity(3)
    OUT.mkdir()
    atomic(OUT / 'preflight.json', checked)
    atomic(OUT / 'authorization.json', dict(status='authorized', user_message='那个unkown outcome是什么意思？断了吗？那重新把那个跑了覆盖掉？',
        scope='One reset of buffer_swap/ID/20116 using the historical GPT-5.5/high configuration; update selected reporting outcome, preserve original interruption.',
        maximum_new_episodes=1, automatic_retries=0, time=time.time()))
    atomic(OUT / 'original_artifacts.json', inventory())
    atomic(OUT / 'allocation.json', dict(devices=[3], occupancy=occupancy, maximum_physical_devices=5))
    shutil.copyfile(__file__, OUT / 'frozen_recovery_executor.py')
    freeze = copy.deepcopy(read(OLD / 'study_freeze.json'))
    # The sole historical revision disabled gradients during a zero-update
    # reload check. Verify its exact provenance, then record a derived manifest.
    freeze['study']['framework_source'] = checked['framework_source']
    freeze['study_sha256'] = object_hash(freeze['study'])
    freeze['recovery_provenance'] = dict(original=str(OLD / 'study_freeze.json'),
        original_sha256=digest(OLD / 'study_freeze.json'),
        revision='runs/exp2/M1_astra_xhigh/incidents/buffer_red_h02_reload/framework_revision.json',
        changed_study_fields=['framework_source'])
    atomic(OUT / 'study_freeze.json', freeze)
    folder = OUT / NAME
    folder.mkdir()
    shutil.copyfile(OLD / NAME / 'task.json', folder / 'task.json')
    (folder / 'evaluation').mkdir()
    (folder / 'evaluation/naive_DP').symlink_to((OLD / NAME / 'evaluation/naive_DP').resolve())
    cmd = [PIXI, 'run', '--manifest-path', str(MANIFEST), '--locked', '--no-install',
           'python', '-m', MODULE, 'evaluate']
    with (OUT / 'stdout.log').open('w') as stdout, (OUT / 'stderr.log').open('w') as stderr:
        process = subprocess.Popen(cmd, cwd=ROOT, stdout=stdout, stderr=stderr)
        atomic(OUT / 'process.json', dict(pid=process.pid, supervisor_pid=os.getpid(), command=cmd, started=time.time()))
        code = process.wait()
    atomic(OUT / 'process_result.json', dict(returncode=code, ended=time.time()))
    assert inventory() == read(OUT / 'original_artifacts.json')
    if code:
        atomic(OUT / 'stopped.json', dict(returncode=code, automatic_retry=False, original_preserved=True))
        raise RuntimeError('Authorized attempt stopped; inspect the retained failure, no retry')
    audit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'run', 'evaluate', 'audit'])
    parser.add_argument('--device-isolated', action='store_true')
    args = parser.parse_args()
    if args.command == 'evaluate' and not args.device_isolated:
        from appl.gpu import launch
        launch(3, sys.argv[1:], module=MODULE)
    elif args.command == 'evaluate':
        assert {p.name for p in Path('/dev').glob('nvidia[0-9]*')} == {'nvidia' + os.environ['APPL_GPU_MINOR']}
        evaluate()
    elif args.command == 'preflight':
        print(preflight())
    elif args.command == 'audit':
        audit()
    else:
        run()
