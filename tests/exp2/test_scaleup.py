"""Task/distribution and fair-comparison contracts; synthetic states only."""
import copy
import numpy as np
import pytest
from appl.scaleup.tasks import TASKS,task,layout,contract,measure
from appl.prior_policies.data import vector
from appl.dp_baseline.data import vector as old_vector,FIELDS
from appl.prior_policies.feedback import measurements,stop_schema,matches


@pytest.mark.parametrize('name',list(TASKS))
def test_predeclared_ood_is_outside_training_position_support(name):
    spec=task(name)
    for seed in range(100):
        for condition in ('ID','OOD'):
            values=layout(spec,seed,condition)
            assert values==layout(spec,seed,condition)
            offsets=[]
            for color in ('red','blue'):
                delta=np.asarray(values[color])-spec['initial'][color];offsets.append(delta)
                assert delta[2]==0
                if condition=='ID':assert abs(delta[:2]).max()<=.012+1e-10
                else:assert .022-1e-10<=abs(delta[:2]).max()<=.04+1e-10
            if name=='unstack_sort':np.testing.assert_allclose(offsets[0],offsets[1])


def test_task_goal_uses_full_rotated_containment_without_terminal_extras():
    spec=task('two_block_sort');c=contract(spec)
    state={color+'_pose':spec['goals'][color]+[1,0,0,0] for color in ('red','blue')}
    assert measure(state,c)['success']
    state['red_pose'][0]+=.035
    assert measure(state,c)['success']
    state['red_pose'][3:]=[np.cos(np.pi/8),0,0,np.sin(np.pi/8)]
    assert not measure(state,c)['success']
    state['red_pose']=spec['goals']['red']+[1,0,0,0];state['blue_pose'][2]+=.012
    assert not measure(state,c)['success']


def test_new_targets_enter_the_model_and_drawer_inputs_remain_identical():
    state={name:[0.]*width for name,width in FIELDS}
    np.testing.assert_array_equal(vector(state),old_vector(state))
    original=copy.deepcopy(state);state.update(red_goal=[-.2,-.2,.02],blue_goal=[-.2,.2,.02])
    np.testing.assert_array_equal(vector(state)[:41],old_vector(original)[:41])
    np.testing.assert_allclose(vector(state)[41:],state['red_goal']+state['blue_goal'])
    state['red_goal']=[float('nan'),0,0]
    with pytest.raises(ValueError,match='observed target'):vector(state)


def test_feedback_uses_the_task_predicates_instead_of_drawer_goals():
    state={name:[0.]*width for name,width in FIELDS}
    m=measurements(state,dict(red_at_goal=True,blue_at_goal=False,success=False))
    assert m['red_at_goal']==1 and m['blue_at_goal']==0 and 'drawer_open' not in m
    schema=stop_schema(['red_at_goal','blue_at_goal','success'])
    names=schema['items']['properties']['conditions']['items']['properties']['metric']['enum']
    assert 'red_at_goal' in names and 'drawer_open' not in names
    assert matches([dict(conditions=[dict(metric='red_at_goal',comparison='eq',value=1,reference='absolute')])],m,m)==[0]


def test_naive_wrapper_matches_frozen_baseline_computation_and_objective():
    import torch
    from appl import baseline
    from appl.public import Factory
    from appl.scaleup.naive_policy import build_model,compute_loss
    torch.set_num_threads(2)
    spec=dict(observation_dimension=47,training=dict(observation_steps=2,timestep_embed_dim=8,
        down_dims=[8,16],kernel_size=3,groups=2),normalizer=dict(mean=[0.]*47,std=[1.]*47))
    torch.manual_seed(0);reference=baseline.build_model(Factory(spec['training']),spec)
    torch.manual_seed(0);wrapped=build_model(spec)
    assert all(torch.equal(a,b) for a,b in zip(reference.parameters(),wrapped.parameters()))
    batch=dict(raw_obs=torch.randn(2,2,47),noisy_action=torch.randn(2,16,8),
        timesteps=torch.tensor([2,7]),noise=torch.randn(2,16,8),mask=torch.ones(2,16,1))
    expected=baseline.compute_loss(reference,batch,spec)['loss']
    actual=compute_loss(wrapped,batch,spec)
    torch.testing.assert_close(actual['loss'],expected,rtol=0,atol=0)
    assert actual['prior_loss']==0
    actual['loss'].backward()
    assert sum(float(p.grad.square().sum()) for p in wrapped.parameters() if p.grad is not None)>0


def test_new_inference_prompt_matches_goal_names_and_keeps_budget_private():
    from appl.scaleup.evaluate import prompt
    from appl.prior_policies.feedback import PROMPT
    assert prompt('drawer_exchange')==PROMPT
    text=prompt('tray_pack')
    assert 'red_at_goal' in text and 'blue_at_goal' in text
    assert 'drawer-open, red-on-pad' not in text
    assert '5000' not in text and 'remaining_steps' not in text and 'max_steps' not in text


def test_original_drawer_ood_offsets_are_outside_both_original_id_ranges():
    from appl.scaleup.protocol import initial_layout
    from appl.envs.scene import initial_state,DRAWER_ORIGIN
    for seed in range(6400,6430):
        actual=initial_layout('drawer_exchange',seed,'OOD')
        for name,nominal in [('red',np.array(DRAWER_ORIGIN)+[-.085,-.065,.028]),('blue',[-.4,.3,.02])]:
            assert .022-1e-10<=max(abs(np.array(actual[name][:2])-np.array(nominal[:2])))<=.04+1e-10
        old=initial_state(seed);same=initial_layout('drawer_exchange',seed,'ID')
        assert same==dict(red=old['red'],blue=old['blue'])


def test_report_retains_agent_finish_before_any_physical_action(tmp_path):
    from appl.io import atomic
    from appl.scaleup.report import audit_episode
    spec=task('two_block_sort');c=contract(spec)
    state={color+'_pose':spec['initial'][color]+[1,0,0,0] for color in ('red','blue')}
    atomic(tmp_path/'initial_state.json',state)
    result=dict(steps=0,success=False,final=measure(state,c),status='agent_finished')
    audit=audit_episode(tmp_path,c,result)
    assert audit['steps']==0 and audit['first_success'] is None
