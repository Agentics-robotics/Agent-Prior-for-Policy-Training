"""Synthetic protocol fixtures, never experimental policy designs or rollouts."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from appl.io import ROOT,read
from appl.journal import Journal,encode
from appl.prior_policies import deploy,feedback


def condition(metric,value,comparison='ge',reference='absolute'):
    return dict(metric=metric,value=value,comparison=comparison,reference=reference)


def fixture_tools(tmp_path,monkeypatch,limit=5000):
    import torch
    state=dict(qpos=[0.]*7+[.04,.04],qvel=[0.]*9,
        tcp_pose=[-.4,.3,.4,1.,0.,0.,0.],red_pose=[.1,0.,.063,1.,0.,0.,0.],
        blue_pose=[-.4,.3,.02,1.,0.,0.,0.],drawer_position=[.3],drawer_velocity=[0.])
    obs=dict(sensor_data=dict(front=dict(rgb=torch.zeros((1,2,2,3),dtype=torch.uint8))))
    env=SimpleNamespace(unwrapped=SimpleNamespace(single_action_space=SimpleNamespace(low=-np.ones(8),high=np.ones(8))))
    cfg=dict(completion_contract=ROOT/'experiments/exp2/configs/demonstration_goals.json',
        evaluation=dict(max_steps=limit,max_api_calls=128,max_invocation_steps=300,
                        feedback_protocol='api_conditions_v1',hide_step_budget=True,context_recent_turns=2))
    policies=dict(fixture=dict(folder=tmp_path/'fixture',version='synthetic',checkpoint_sha256='synthetic',
                               metadata={},document='Synthetic protocol fixture.',source_heuristic='None'))
    class Worker:
        def __init__(self,*args):self.resets=[]
        def action(self,state,reset=False):self.resets.append(reset);return np.zeros(8)
    index=[0]
    def step(env,action):
        index[0]+=1;result=copy.deepcopy(state)
        result['blue_pose'][2]=[.10,.21,.30,.02,.02][min(index[0]-1,4)]
        return obs,result
    monkeypatch.setattr(deploy,'PolicyProcess',Worker);monkeypatch.setattr(deploy,'step',step)
    tools=deploy.DeploymentTools(cfg,env,obs,state,policies,tmp_path,6200)
    tools.read_docs.add('fixture')
    return tools,index


def invoke(tools,steps,groups,cid='synthetic_call'):
    return tools.execute(dict(name='invoke_policy',call_id=cid,arguments=json.dumps(dict(
        policy_id='fixture',steps=steps,reason='Synthetic stopping contract check.',
        stop_when=groups,notebook='Synthetic fixture: retain this exact API-authored string.'))))


def test_api_condition_returns_control_before_lift_is_lost(tmp_path,monkeypatch):
    tools,index=fixture_tools(tmp_path,monkeypatch)
    try:
        groups=[dict(label='synthetic lift observation',conditions=[condition('blue_z',.20)])]
        output=invoke(tools,5,groups)
        assert tools.steps==index[0]==2
        assert output['state']['blue_pose'][2]==.21
        call=read(tmp_path/'invocations.json')[0]
        assert call['stop_when']==groups and call['stop_reason']=='api_condition'
        assert call['matched_stop_rules']==[0] and call['metric_ranges']['blue_z']['max']==.21
        assert output['last_invocation']['notebook']==call['notebook']
        # Staying with the same policy does not reset its action/history stream.
        invoke(tools,1,[],cid='continue')
        assert tools.workers['fixture'].resets==[True,False,False]
    finally:tools.j.db.close()


def test_requested_duration_returns_intermediate_extrema(tmp_path,monkeypatch):
    tools,index=fixture_tools(tmp_path,monkeypatch)
    try:
        output=invoke(tools,4,[])
        assert index[0]==4 and output['state']['blue_pose'][2]==.02
        assert output['last_invocation']['metric_ranges']['blue_z']==dict(min=.02,max=.30,min_step=0,max_step=3)
        assert output['last_invocation']['stop_reason']=='requested_duration'
    finally:tools.j.db.close()


def test_tabletop_goal_condition_uses_the_same_per_step_api_protocol(tmp_path,monkeypatch):
    tools,index=fixture_tools(tmp_path,monkeypatch)
    tools.goal_names=['red_at_goal','blue_at_goal','success']
    tools.goal_measure=lambda state,contract:dict(red_at_goal=False,
        blue_at_goal=state['blue_pose'][2]>=.20,success=False)
    try:
        groups=[dict(label='synthetic task predicate',conditions=[condition('blue_at_goal',1,'eq')])]
        result=invoke(tools,5,groups)
        assert tools.steps==index[0]==2
        assert result['observed_metrics']['blue_at_goal']==1
        assert 'drawer_open' not in result['observed_metrics']
        assert not result['task_goals']['success']
        assert tools.calls[-1]['stop_when']==groups and tools.calls[-1]['stop_reason']=='api_condition'
    finally:tools.j.db.close()


def test_private_cap_not_in_visible_state_or_schemas_and_still_enforced(tmp_path,monkeypatch):
    tools,index=fixture_tools(tmp_path,monkeypatch,limit=3)
    try:
        before=tools.visible();schemas=tools.schemas()
        tools.cfg['evaluation']['max_steps']=9123
        assert tools.visible()==before and tools.schemas()==schemas
        assert 'remaining_steps' not in before and 'max_steps' not in before
        assert '5000' not in feedback.PROMPT and 'remaining_steps' not in feedback.PROMPT
        tools.cfg['evaluation']['max_steps']=3
        output=invoke(tools,300,[])
        assert tools.done() and tools.steps==index[0]==3
        assert output['last_invocation']['stop_reason']=='executor_limit'
    finally:tools.j.db.close()


def test_conjunctions_and_relative_change_have_literal_meaning():
    initial=dict(blue_z=.02,drawer_position=.30,finger_width=.08)
    current=dict(blue_z=.21,drawer_position=.28,finger_width=.037)
    groups=[dict(label='lift',conditions=[condition('blue_z',.2),condition('finger_width',.04,'le')]),
        dict(label='drawer change',conditions=[condition('drawer_position',-.01,'le','invocation_start')])]
    assert feedback.matches(groups,current,initial)==[0,1]
    current['finger_width']=.08
    assert feedback.matches(groups,current,initial)==[1]
    groups[0]['conditions'][0]['value']=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):feedback.matches(groups,current,initial)


def test_context_keeps_original_complete_exchanges_and_full_journal(tmp_path):
    journal=Journal(tmp_path/'journal');history=[dict(role='user',content='Synthetic initial input.')]
    turns=[]
    try:
        operations=[('read_policy',dict(policy_id='a')),('invoke_policy',dict(notebook='API exact notebook')),
            ('read_policy',dict(policy_id='a')),('read_policy',dict(policy_id='b'))]+[('observe',{})]*8
        for seq,(name,args) in enumerate(operations,1):
            output=[dict(type='reasoning',id=f'r{seq}',encrypted_content=f'opaque_{seq}'),
                dict(type='function_call',name=name,call_id=f'c{seq}',arguments=encode(args))]
            additions=output+[dict(type='function_call_output',call_id=f'c{seq}',output=encode(dict(fixture=seq)))]
            turns.append(additions);history+=copy.deepcopy(additions)
            with journal.db:journal.db.execute('INSERT INTO api VALUES(?,?,?,?)',
                (seq,'consumed','{}',encode(dict(output=output))))
        journal.set('history',history);before=copy.deepcopy(history)
        result=feedback.project_context(journal,history,2)
        expected=[history[0]]+turns[1]+turns[2]+turns[3]+turns[-2]+turns[-1]
        assert result==expected
        assert history==before and journal.get('history')==before
        assert all(item in history for item in result)
        with pytest.raises(ValueError,match='Uncommitted'):
            feedback.project_context(journal,history+[dict(role='user',content='uncommitted')],2)
    finally:journal.db.close()


def test_legacy_protocol_keeps_original_visible_contract(tmp_path,monkeypatch):
    tools,_=fixture_tools(tmp_path,monkeypatch)
    try:
        tools.feedback=False
        assert tools.visible()['remaining_steps']==5000
        assert 'invocations' in tools.visible()
        assert 'stop_when' not in tools.schemas()[2]['parameters']['properties']
        assert len(tools.schemas())==4
    finally:tools.j.db.close()


def test_agent_loop_uses_projection_without_changing_raw_api_output(tmp_path,monkeypatch):
    from appl.agent import AgentLoop
    tools,_=fixture_tools(tmp_path,monkeypatch)
    script=[('read_policy',dict(policy_id='fixture')),
        ('invoke_policy',dict(policy_id='fixture',steps=5,reason='Synthetic tool loop check.',
            stop_when=[dict(label='synthetic lift',conditions=[condition('blue_z',.2)])],
            notebook='Exact synthetic API notebook.')),
        ('finish',dict(reason='Synthetic protocol test finished.'))]
    class Client:
        model='synthetic_fixture';reasoning=None;provenance='synthetic_fixture'
        def __init__(self):self.requests=[];self.responses=[]
        def configuration(self):return dict(model=self.model)
        def respond(self,request):
            i=len(self.requests);self.requests.append(copy.deepcopy(request));name,args=script[i]
            response=dict(status='completed',output=[dict(type='function_call',name=name,
                call_id=f'fixture_{i}',arguments=json.dumps(args))])
            self.responses.append(copy.deepcopy(response));return 200,response
    client=Client()
    try:
        AgentLoop(tools.j,tools,client,feedback.PROMPT,context_builder=tools.context_input).run(
            dict(observation=tools.visible()))
        assert len(client.requests)==3 and tools.steps==2 and tools.finished
        assert 'remaining_steps' not in encode(client.requests)
        assert 'max_steps' not in encode(client.requests)
        for (raw,),original in zip(tools.j.db.execute('SELECT response FROM api ORDER BY seq'),client.responses):
            assert json.loads(raw)==original
        assert tools.j.db.execute("SELECT COUNT(*) FROM api WHERE status='consumed'").fetchone()[0]==3
    finally:tools.j.db.close()
