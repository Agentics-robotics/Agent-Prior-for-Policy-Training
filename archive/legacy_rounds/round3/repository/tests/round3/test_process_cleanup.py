"""Owned-process cleanup checks without launching workers or using a GPU."""
import signal
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import pytest

from round3 import cli


@pytest.mark.parametrize('failure', [KeyboardInterrupt('original'), RuntimeError('original')])
def test_child_preserves_exception_and_waits_for_worker_group(tmp_path, monkeypatch, failure):
    trace=[]
    waits=iter([failure, KeyboardInterrupt('second interrupt'), 130])
    proc=Mock(pid=1234)
    def wait():
        value=next(waits)
        trace.append('wait')
        if isinstance(value,BaseException):
            raise value
        return value
    proc.wait.side_effect=wait
    proc.poll.return_value=130
    group_states=iter([True,False])
    def group_live(pgid):
        assert pgid==proc.pid
        live=next(group_states)
        trace.append('worker-live' if live else 'worker-exited')
        return live
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setattr(cli,'R3',tmp_path/'round3')
    monkeypatch.setattr(cli,'pixi_binary',lambda: '/synthetic/pixi')
    monkeypatch.setattr(cli.subprocess,'Popen',lambda *a,**k: proc)
    monkeypatch.setattr(cli,'event',lambda *a,**k: None)
    monkeypatch.setattr(cli,'lock',lambda *a,**k: nullcontext())
    monkeypatch.setattr(cli,'run_record',lambda rid: {'launcher_pid': proc.pid})
    monkeypatch.setattr(cli,'update_run',lambda rid,**fields: trace.append(
        'claim-cleared' if fields.get('launcher_pid','missing') is None else 'claim-set'))
    monkeypatch.setattr(cli.os,'killpg',lambda pid,sig: trace.append((pid,sig)))
    monkeypatch.setattr(cli,'_process_group_live',group_live)
    monkeypatch.setattr(cli.time,'sleep',lambda seconds: trace.append('sleep'))
    with pytest.raises(type(failure)) as caught:
        cli._child('train-one','synthetic',0)
    trace.append('caller-released')
    assert caught.value is failure
    assert trace==['claim-set','wait',(1234,signal.SIGINT),'wait','wait',
                   'worker-live','sleep','worker-exited','claim-cleared','caller-released']


def test_interrupt_race_with_already_exited_process_still_reaps_launcher(monkeypatch):
    proc=Mock(pid=1234)
    monkeypatch.setattr(cli.os,'killpg',Mock(side_effect=ProcessLookupError))
    monkeypatch.setattr(cli,'_process_group_live',lambda pgid: False)
    cli._interrupt_and_wait(proc)
    proc.wait.assert_called_once_with()


def test_finished_child_preserves_a_new_stage_owner(tmp_path, monkeypatch):
    proc=Mock(pid=1234)
    proc.wait.return_value=0
    proc.poll.return_value=0
    changes=[]
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setattr(cli,'R3',tmp_path/'round3')
    monkeypatch.setattr(cli,'pixi_binary',lambda: '/synthetic/pixi')
    monkeypatch.setattr(cli.subprocess,'Popen',lambda *a,**k: proc)
    monkeypatch.setattr(cli,'event',lambda *a,**k: None)
    monkeypatch.setattr(cli,'lock',lambda *a,**k: nullcontext())
    monkeypatch.setattr(cli,'run_record',lambda rid: {'launcher_pid': 5678})
    monkeypatch.setattr(cli,'update_run',lambda rid,**fields: changes.append(fields))
    cli._child('train-one','synthetic',0)
    assert len(changes)==1 and changes[0]['launcher_pid']==1234


def test_completed_stage_waits_for_its_live_launcher_before_reclaim(monkeypatch):
    monkeypatch.setattr(cli,'_pid_live',lambda pid: pid==1234)
    rec=dict(status='pending',train_status='completed',dev_status='pending',launcher_pid=1234)
    assert not cli._eligible(rec,'dev')
    rec['launcher_pid']=None
    assert cli._eligible(rec,'dev')


def test_owned_group_ignores_zombies_and_unrelated_processes(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.os,'killpg',lambda *args: None)
    monkeypatch.setattr(cli,'Path',lambda path: tmp_path if path=='/proc' else Path(path))
    worker=tmp_path/'1234'
    unrelated=tmp_path/'5678'
    worker.mkdir()
    unrelated.mkdir()
    (worker/'stat').write_text('1234 (worker (with spaces)) Z 1 1234 0')
    (unrelated/'stat').write_text('5678 (other) R 1 5678 0')
    assert not cli._process_group_live(1234)
    (worker/'stat').write_text('1234 (worker (with spaces)) S 1 1234 0')
    assert cli._process_group_live(1234)
