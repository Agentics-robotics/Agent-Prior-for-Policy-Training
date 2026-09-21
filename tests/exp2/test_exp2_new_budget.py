"""Cost admission and executor invariants; all API data here are test fixtures."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from appl.io import read
from appl.journal import Journal, encode
from experiments.exp2.exp2_new import budget
from experiments.exp2.exp2_new.budget import Ledger, BudgetClient, BudgetStop, reserve_cost, usage_cost
from experiments.exp2.exp2_new.run import binary_step


def test_reservations_prevent_overspend_and_duplicate_attempts(tmp_path):
    ledger=Ledger(tmp_path/'budget');ledger.initialize('0.3','Test fixture, no actual spend')
    ledger.reserve('a',1000,4096)
    with pytest.raises(ValueError):ledger.reserve('a',1000,4096)
    with pytest.raises(BudgetStop):ledger.reserve('b',1000,4096)
    v=read(ledger.path)
    assert len(v['records'])==1 and v['stopped']
    assert v['records'][0]['reserved_nano_usd']==217300000


def test_cache_and_reasoning_output_accounting():
    # output_tokens already includes reasoning: do not add reasoning twice.
    n,assumed=usage_cost(dict(input_tokens=1000,output_tokens=100,
        input_tokens_details=dict(cached_tokens=200,cache_write_tokens=300),
        output_tokens_details=dict(reasoning_tokens=70)))
    assert n==13950000 and not assumed
    conservative,assumed=usage_cost(dict(input_tokens=1000,output_tokens=100,
        input_tokens_details=dict(cached_tokens=200)))
    assert conservative==15200000 and assumed
    assert reserve_cost(272001,4096)==272001*25000+4096*75000


def test_unknown_transport_charge_is_not_released(tmp_path):
    ledger=Ledger(tmp_path/'budget');ledger.initialize('1','Test fixture')
    ledger.reserve('unknown',1000,4096)
    ledger.settle('unknown',dict(error=dict(type='server_error')),502)
    v=read(ledger.path)
    assert v['stopped'] and 'settled_nano_usd' not in v['records'][0]
    with pytest.raises(BudgetStop):ledger.reserve('next',1,1)


@pytest.mark.parametrize('raw',[-9.,-0.001,0.,0.001,9.])
def test_step_adapter_changes_only_executed_gripper(raw):
    received=[]
    fake_env=SimpleNamespace(unwrapped=SimpleNamespace(single_action_space=SimpleNamespace(
        low=np.array([-10.]*7+[-1.]),high=np.array([10.]*7+[1.]))))
    predicted=np.array([.1]*7+[raw],np.float32)
    executed=np.clip(predicted,fake_env.unwrapped.single_action_space.low,fake_env.unwrapped.single_action_space.high)
    binary_step(lambda env,action:received.append(action.copy()))(fake_env,executed)
    assert predicted[7]==np.float32(raw)
    assert np.array_equal(received[0][:7],predicted[:7])
    assert received[0][7]==(1 if raw>=0 else -1)
    assert np.array_equal(received[0],executed)


def test_budget_rejection_never_sends_generation(tmp_path,monkeypatch):
    monkeypatch.setattr(budget,'BASE',tmp_path)
    ledger=Ledger(tmp_path/'budget');ledger.initialize('.01','Test fixture')
    j=Journal(tmp_path/'episode/api')
    request=dict(model='gpt-6-astra',reasoning=dict(effort='xhigh'),instructions='fixture',
                 input=[],tools=[],tool_choice='auto',parallel_tool_calls=False,max_output_tokens=4096)
    with j.db:j.db.execute('INSERT INTO api VALUES(?,?,?,?)',(1,'started',encode(request),None))
    client=BudgetClient.__new__(BudgetClient);client.j=j;client.ledger=ledger
    sent=[]
    def fake_post(suffix,payload):
        sent.append(suffix)
        assert suffix=='/input_tokens'
        return 200,dict(input_tokens=1000)
    client.post=fake_post
    with pytest.raises(BudgetStop):client.respond(request)
    assert sent==['/input_tokens']
    assert j.db.execute('SELECT status FROM api').fetchone()[0]=='not_sent'
    assert read(ledger.path)['records']==[]
    saved=json.loads(j.db.execute('SELECT request FROM api').fetchone()[0])
    assert saved['service_tier']=='default' and saved['reasoning']['effort']=='xhigh'
    j.db.close()


def test_success_preserves_api_response_and_settles_reservation(tmp_path,monkeypatch):
    monkeypatch.setattr(budget,'BASE',tmp_path)
    ledger=Ledger(tmp_path/'budget');ledger.initialize('1','Test fixture')
    j=Journal(tmp_path/'episode/api')
    request=dict(model='gpt-6-astra',reasoning=dict(effort='xhigh'),instructions='fixture',
                 input=[],tools=[],tool_choice='auto',parallel_tool_calls=False,max_output_tokens=4096)
    with j.db:j.db.execute('INSERT INTO api VALUES(?,?,?,?)',(1,'started',encode(request),None))
    body=dict(id='fixture_response',model='gpt-6-astra',reasoning=dict(effort='xhigh'),
              output=[dict(type='message',content='fixture only')],
              usage=dict(input_tokens=1000,output_tokens=100,input_tokens_details=dict(cached_tokens=0,cache_write_tokens=0)))
    client=BudgetClient.__new__(BudgetClient);client.j=j;client.ledger=ledger;calls=[]
    def fake_post(suffix,payload):
        calls.append(suffix)
        return (200,dict(input_tokens=1000)) if suffix else (200,body)
    client.post=fake_post
    status,response=client.respond(request)
    assert status==200 and response is body and calls==['/input_tokens','']
    record=read(ledger.path)['records'][0]
    assert record['settled_nano_usd']==15000000 and record['status']=='usage_reported'
    assert read(j.root/'transport/0001/response.json')['response']==body
    j.db.close()


def test_generation_transport_failure_holds_reservation_and_does_not_retry(tmp_path,monkeypatch):
    monkeypatch.setattr(budget,'BASE',tmp_path)
    ledger=Ledger(tmp_path/'budget');ledger.initialize('1','Test fixture')
    j=Journal(tmp_path/'episode/api')
    request=dict(model='gpt-6-astra',reasoning=dict(effort='xhigh'),instructions='fixture',
                 input=[],tools=[],tool_choice='auto',parallel_tool_calls=False,max_output_tokens=4096)
    with j.db:j.db.execute('INSERT INTO api VALUES(?,?,?,?)',(1,'started',encode(request),None))
    client=BudgetClient.__new__(BudgetClient);client.j=j;client.ledger=ledger;calls=[]
    def fake_post(suffix,payload):
        calls.append(suffix)
        if suffix:return 200,dict(input_tokens=1000)
        raise TimeoutError('fixture transport timeout')
    client.post=fake_post
    with pytest.raises(TimeoutError):client.respond(request)
    assert calls==['/input_tokens','']
    record=read(ledger.path)['records'][0]
    assert 'settled_nano_usd' not in record and read(ledger.path)['stopped']=='TimeoutError'
    assert j.db.execute('SELECT status FROM api').fetchone()[0]=='started'
    j.db.close()


def test_request_identifier_handles_storage_symlink(tmp_path,monkeypatch):
    storage=tmp_path/'physical';storage.mkdir()
    logical=tmp_path/'repository_link';logical.symlink_to(storage,target_is_directory=True)
    monkeypatch.setattr(budget,'BASE',logical)
    ledger=Ledger(logical/'budget');ledger.initialize('.01','Symlink regression fixture')
    j=Journal(logical/'episode/api')
    request=dict(model='gpt-6-astra',reasoning=dict(effort='xhigh'),instructions='fixture',
                 input=[],tools=[],tool_choice='auto',parallel_tool_calls=False,max_output_tokens=4096)
    with j.db:j.db.execute('INSERT INTO api VALUES(?,?,?,?)',(1,'started',encode(request),None))
    client=BudgetClient.__new__(BudgetClient);client.j=j;client.ledger=ledger
    calls=[]
    def fake_post(suffix,payload):
        calls.append(suffix)
        assert suffix=='/input_tokens'
        return 200,dict(input_tokens=1000)
    client.post=fake_post
    with pytest.raises(BudgetStop):client.respond(request)
    assert calls==['/input_tokens'] and read(ledger.path)['records']==[]
    j.db.close()
