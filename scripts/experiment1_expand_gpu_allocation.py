"""Apply the user-authorized eight-GPU resource amendment after live workers drain.

Run with locked Pixi. No candidate edits, training retries or protocol rewriting.
The exact proposed changes and original files are retained before any mutation.
"""
from pathlib import Path
import argparse
import importlib
import json
import os
import signal
import subprocess
import time

from experiment1.records import ROOT, EXPERIMENT, PIXI, Records, atomic_json, file_hash, immutable_json, now, read_json

DEST = EXPERIMENT / 'allocation_20260913'
NAMES = ('src/experiment1/cli.py', 'src/experiment1/isolation.py',
         'src/experiment1/protocol.py', 'src/experiment1/runner.py')


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Expected exact single resource-change target: ' + old)
    return text.replace(old, new)


def prepare():
    from experiment1.protocol import verify_frozen
    verify_frozen()
    changes = {}
    for name in NAMES:
        text = (ROOT / name).read_text()
        old = text
        if name.endswith('cli.py'):
            text = replace_once(text, 'choices=(4, 5, 6, 7)', 'choices=tuple(range(8))')
        elif name.endswith('isolation.py'):
            text = replace_once(text, 'gpu not in (4, 5, 6, 7)', 'gpu not in range(8)')
            text = replace_once(text, 'Only physical GPUs 4, 5, 6, 7 are authorized', 'Only physical GPUs 0 through 7 are authorized')
            text = replace_once(text, 'by the outer scheduler; no grants for the released devices 0 through 3.', 'by the outer scheduler; no grants for any other physical device.')
        elif name.endswith('runner.py'):
            text = replace_once(text, 'AUTHORIZED_GPUS = (4, 5, 6, 7)', 'AUTHORIZED_GPUS = tuple(range(8))')
            text = replace_once(text, 'Only physical GPUs 4, 5, 6, 7 are authorized', 'Only physical GPUs 0 through 7 are authorized')
            text = replace_once(text, 'Four independently resumable Pixi workers', 'Allocation-sized independently resumable Pixi workers')
            text = replace_once(text, 'if gpus != AUTHORIZED_GPUS:', 'if not gpus or len(set(gpus)) != len(gpus) or any(type(gpu) is not int or gpu not in AUTHORIZED_GPUS for gpu in gpus):')
            text = replace_once(text, 'Frozen runner allocation must be physical GPUs 4, 5, 6, 7', 'Verified runner allocation must contain unique authorized physical GPUs')
            text = replace_once(text, 'ThreadPoolExecutor(max_workers=4)', 'ThreadPoolExecutor(max_workers=len(gpus))')
            text = replace_once(text, 'files = expected + [root / "protocol.lock.json", root / "matrix.jsonl"]', 'files = expected + [root / "protocol.lock.json", root / "matrix.jsonl"]\n    if (root / "execution_allocation.json").exists():\n        files.append(root / "execution_allocation.json")')
        else:
            old_verify = '''    for name, sha in protocol["inputs"].items():
        if file_hash(ROOT / name) != sha:
            raise ValueError("Frozen experiment input changed: " + name)'''
            new_verify = '''    amendment_path = Path(root) / "execution_allocation.json"
    amended = {}
    if amendment_path.exists():
        amendment = read_json(amendment_path)
        allowed = {"src/experiment1/cli.py", "src/experiment1/isolation.py",
                   "src/experiment1/protocol.py", "src/experiment1/runner.py"}
        if (amendment["schema_version"] != "experiment1.resource-amendment.v1"
                or amendment["protocol_sha256"] != file_hash(Path(root) / "protocol.lock.json")
                or amendment["previous_gpus"] != protocol["gpus"]
                or amendment["gpus"] != list(range(8))
                or set(amendment["changes"]) != allowed):
            raise ValueError("Invalid user-authorized GPU allocation amendment")
        for name, change in amendment["changes"].items():
            original = Path(root) / "allocation_20260913" / "original" / name
            if change["before"] != protocol["inputs"][name] or file_hash(original) != change["before"]:
                raise ValueError("Original frozen infrastructure snapshot changed: " + name)
            amended[name] = change["after"]
    for name, sha in protocol["inputs"].items():
        if file_hash(ROOT / name) != amended.get(name, sha):
            raise ValueError("Frozen experiment input changed: " + name)
    if amended:
        protocol = dict(protocol, gpus=amendment["gpus"])'''
            text = replace_once(text, old_verify, new_verify)
        for folder, content in [('original', old), ('proposed', text)]:
            path = DEST / folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_text() != content:
                raise ValueError('Prepared allocation source differs')
            path.write_text(content)
        compile(text, name, 'exec')
        changes[name] = dict(before=file_hash(DEST/'original'/name), after=file_hash(DEST/'proposed'/name))
    amendment = dict(schema_version='experiment1.resource-amendment.v1',
        protocol_sha256=file_hash(EXPERIMENT/'protocol.lock.json'), previous_gpus=[4,5,6,7],
        gpus=list(range(8)), changes=changes,
        authority='User explicitly authorized all eight physical GPUs on 2026-09-13',
        scope='Resource authorization and scheduler concurrency only; original scientific protocol retained')
    immutable_json(DEST/'proposed_amendment.json', amendment)
    return amendment


def live_instances():
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            args = (path/'cmdline').read_bytes().split(b'\0')
            cwd = (path/'cwd').resolve()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if cwd == ROOT and b'experiment1.cli' in args and b'instance' in args:
            found.append(int(path.name))
    return found


def apply_and_run():
    records = Records()
    while True:
        active = live_instances()
        atomic_json(DEST/'transition_status.json', dict(time=now(),status='draining' if active else 'applying',active_instance_pids=active))
        if not active:
            break
        time.sleep(10)
    amendment = read_json(DEST/'proposed_amendment.json')
    if file_hash(EXPERIMENT/'protocol.lock.json') != amendment['protocol_sha256']:
        raise ValueError('Original protocol changed during worker drain')
    for name, value in amendment['changes'].items():
        if file_hash(ROOT/name) != value['before'] or file_hash(DEST/'proposed'/name) != value['after']:
            raise ValueError('Staged or current source changed during drain')
    # No live instance can observe the multi-file transition. Never rewrite protocol.lock.json.
    for name in amendment['changes']:
        (ROOT/name).write_bytes((DEST/'proposed'/name).read_bytes())
    immutable_json(EXPERIMENT/'execution_allocation.json', amendment)
    tests = ROOT/'tests/experiment1/test_isolation.py'
    original_tests = tests.read_text()
    immutable_json(DEST/'test_update.json', dict(original=original_tests,
        reason='GPUs 0–3 are now authorized; unauthorized boundary becomes -1, 8 and bool'))
    tests.write_text(replace_once(original_tests, '[0, 1, 2, 3, -1, True]', '[-1, 8, True]'))
    command = [str(PIXI),'run','--locked','python','-m','pytest','-q',
        'tests/experiment1/test_runner.py','tests/experiment1/test_isolation.py',
        'tests/experiment1/test_preflight_launch.py','tests/experiment1/test_allocation_amendment.py']
    with (DEST/'tests.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    import experiment1.protocol as protocol
    import experiment1.isolation as isolation
    importlib.reload(protocol); importlib.reload(isolation)
    assert protocol.verify_frozen()['gpus'] == list(range(8))
    for gpu in range(8):
        public = DEST/f'gpu{gpu}-public.txt'
        public.write_text('GPU allocation interface probe\n')
        output = DEST/f'gpu{gpu}-probe'
        result = isolation.run_isolated('probe',dict(operation='allowed',public_file=str(public),
            write_file=str(output/'ok.txt')),read_only_paths=[public],output_dir=output,gpu=gpu,timeout_seconds=60)
        records.cost('gpu_allocation_probe',dict(gpu=gpu,formal_slots=0,optimizer_updates=0,result=result))
        if result['returncode'] != 0:
            raise ValueError('GPU allocation probe failed; inspect retained record')
    records.event('gpu_allocation_applied',gpus=list(range(8)),amendment=str(EXPERIMENT/'execution_allocation.json'))
    command = [str(PIXI),'run','--locked','python',str(Path.home()/'.config/experiment1/launch.py')]
    with (DEST/'launch.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    atomic_json(DEST/'transition_status.json',dict(time=now(),status='eight_gpu_runner_launched',gpus=list(range(8))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--apply-after-drain',action='store_true')
    args=parser.parse_args()
    if args.apply_after_drain:
        try:
            apply_and_run()
        except Exception as error:
            atomic_json(DEST/'transition_status.json',dict(time=now(),status='blocked',reason=str(error),automatic_retry=False))
            raise
    else:
        prepare()
        print('Exact eight-GPU changes prepared; live frozen sources unchanged')
