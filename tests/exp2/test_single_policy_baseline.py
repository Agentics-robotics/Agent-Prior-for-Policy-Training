"""Full-task data exposure and offline-design contracts; no scientific candidate."""
import numpy as np
import pytest

from appl.io import atomic, digest, object_hash
from experiments.exp2.single_policy.common import describe_task, immutable
from experiments.exp2.single_policy.design import FullTaskTools, INTERFACE


def test_full_task_tools_reach_terminal_label_and_share_exact_normalizer(tmp_path):
    norm={'mean':[0.]*47,'std':[1.]*47,'action_mean':[0.]*8,'action_std':[1.]*8}
    atomic(tmp_path/'normalizer.json',dict(normalizer=norm))
    atomic(tmp_path/'demo.json',dict(observations=[dict(state={'index':i}) for i in range(41)],actions=[[i]*8 for i in range(40)]))
    atomic(tmp_path/'dataset.json',dict(segments=[dict(trajectory_id='train',start=0,stop=40,file='demo.json')]))
    atomic(tmp_path/'goals.json',dict(public_goal='fixture'))
    atomic(tmp_path/'task_description.json',dict(description='Synthetic tool test'))
    atomic(tmp_path/'assignment.json',dict(policy_id='fixture',skill_id='full_task',heuristic_index=1,
        experiment_version='Exp2_single_policy',dataset=str(tmp_path/'dataset.json'),
        normalization=dict(path=str(tmp_path/'normalizer.json'),sha256=digest(tmp_path/'normalizer.json'),normalizer_sha256=object_hash(norm))))
    cfg=dict(training={'updates':60000},completion_contract=tmp_path/'goals.json',
        design=dict(max_api_calls=1,max_tool_calls=10,max_checks=1,max_output_tokens=100))
    tools=FullTaskTools(cfg,tmp_path,0)
    try:
        assignment=tools.dispatch('read_assignment',{})
        assert assignment['shared_normalizer']==norm and 'heuristic' not in assignment
        assert assignment['training']['updates']==60000
        overview=tools.dispatch('read_overview',dict(trajectory_id='train',frames=8))
        assert overview['rows'][0]['index']==0 and overview['rows'][-1]['index']==40
        assert overview['rows'][-1]['action'] is None
        with pytest.raises(ValueError,match='complete training trajectory'):
            tools.dispatch('read_overview',dict(trajectory_id='hidden_test',frames=8))
        with pytest.raises(ValueError,match='outside'):
            tools.dispatch('read_steps',dict(trajectory_id='train',indices=[41]))
        assert tools.dispatch('read_public',dict(name='INTERFACE.md'))['content']==INTERFACE
    finally: tools.j.db.close()


def test_environment_description_does_not_expose_expert_program_or_test_layouts():
    supplied=dict(task_id='two_block_sort',description='Public task',goals={'red':[0,0,0]},
        program=[['red','red']],ood_inner_m=.022,initial={'red':[1,2,3]})
    assert describe_task('two_block_sort',supplied)==dict(task_id='two_block_sort',description='Public task',goals={'red':[0,0,0]})


def test_preparation_cannot_silently_overwrite_an_executed_setting(tmp_path):
    path=tmp_path/'protocol.json'
    immutable(path,dict(updates=60000));before=path.read_bytes()
    immutable(path,dict(updates=60000))
    with pytest.raises(ValueError,match='Immutable record changed'):
        immutable(path,dict(updates=20000))
    assert path.read_bytes()==before
