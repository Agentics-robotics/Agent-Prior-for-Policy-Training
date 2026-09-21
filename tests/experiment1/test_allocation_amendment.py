"""Resource-only amendment and eight-GPU scheduling; synthetic, no API/training."""
import importlib.util
import threading

import pytest

from experiment1.records import ROOT, atomic_json, digest, file_hash


def staged_module(name):
    source = ROOT / 'experiments/experiment1/allocation_20260913/proposed/src/experiment1' / (name + '.py')
    if not source.exists():
        source = ROOT / 'src/experiment1' / (name + '.py')
    spec = importlib.util.spec_from_file_location('experiment1._allocation_test_' + name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_protocol(tmp_path, monkeypatch):
    module = staged_module('protocol')
    repository, root = tmp_path/'repo', tmp_path/'records'
    names = ['src/experiment1/' + name + '.py' for name in ('cli','isolation','protocol','runner')]
    inputs, changes = {}, {}
    for name in names:
        original = root/'allocation_20260913/original'/name
        original.parent.mkdir(parents=True,exist_ok=True)
        original.write_text('synthetic original ' + name)
        current = repository/name
        current.parent.mkdir(parents=True,exist_ok=True)
        current.write_text('synthetic resource update ' + name)
        inputs[name] = file_hash(original)
        changes[name] = dict(before=inputs[name],after=file_hash(current))
    (repository/'trainer.py').write_text('fixed scientific source')
    inputs['trainer.py'] = file_hash(repository/'trainer.py')
    atomic_json(root/'protocol.lock.json',dict(inputs=inputs,gpus=[4,5,6,7],
        prompt_hash=digest(module.SYSTEM_PROMPT),api_limits=module.LIMITS))
    amendment=dict(schema_version='experiment1.resource-amendment.v1',
        protocol_sha256=file_hash(root/'protocol.lock.json'),previous_gpus=[4,5,6,7],
        gpus=list(range(8)),changes=changes)
    atomic_json(root/'execution_allocation.json',amendment)
    monkeypatch.setattr(module,'ROOT',repository)
    return module,repository,root,amendment


def test_resource_amendment_retains_protocol_identity(tmp_path,monkeypatch):
    module,repository,root,amendment=fixture_protocol(tmp_path,monkeypatch)
    before=file_hash(root/'protocol.lock.json')
    assert module.verify_frozen(root)['gpus'] == list(range(8))
    assert file_hash(root/'protocol.lock.json') == before


@pytest.mark.parametrize('target',['trainer.py','src/experiment1/isolation.py'])
def test_amendment_does_not_allow_unrecorded_source_edits(tmp_path,monkeypatch,target):
    module,repository,root,amendment=fixture_protocol(tmp_path,monkeypatch)
    (repository/target).write_text('unauthorized change')
    with pytest.raises(ValueError,match='Frozen experiment input changed'):
        module.verify_frozen(root)


def test_amendment_requires_original_frozen_snapshot(tmp_path,monkeypatch):
    module,repository,root,amendment=fixture_protocol(tmp_path,monkeypatch)
    (root/'allocation_20260913/original/src/experiment1/isolation.py').write_text('changed')
    with pytest.raises(ValueError,match='Original frozen infrastructure snapshot changed'):
        module.verify_frozen(root)


def test_amendment_cannot_change_scientific_source(tmp_path,monkeypatch):
    module,repository,root,amendment=fixture_protocol(tmp_path,monkeypatch)
    amendment['changes']['trainer.py']=dict(before='x',after='y')
    atomic_json(root/'execution_allocation.json',amendment)
    with pytest.raises(ValueError,match='Invalid user-authorized'):
        module.verify_frozen(root)


def test_eight_gpu_schedule_24_instances_exactly_once(tmp_path,monkeypatch):
    runner=staged_module('runner')
    import experiment1.reporting as reporting
    monkeypatch.setattr(runner,'verify_frozen',lambda root:dict(gpus=list(range(8)),api={'api_key_env':'SYNTHETIC_KEY'}))
    monkeypatch.setattr(reporting,'report',lambda root:None)
    atomic_json(tmp_path/'preflight/status.json',dict(passed=True))
    calls=[]
    lock=threading.Lock()
    class Process:
        def __init__(self,command,**kwargs):
            self.pid=123
            with lock:
                calls.append(command)
        def wait(self):
            return 0
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    assert runner.run(tmp_path)['status']=='blocked'
    assignments=[(c[c.index('--task')+1],c[c.index('--n')+1],int(c[c.index('--gpu')+1])) for c in calls]
    assert len(assignments)==len({(t,n) for t,n,g in assignments})==24
    assert {g for t,n,g in assignments}==set(range(8))
    assert all(sum(g==gpu for t,n,g in assignments)==3 for gpu in range(8))
