"""Framework contracts; no API prior is designed by these fixtures."""
import numpy as np
import pytest
from appl.prior_policies.data import windows
from appl.prior_policies.design import DesignTools
from appl.io import atomic, read


def test_future_labels_are_after_the_action_and_stay_in_slice():
    es=[dict(obs=np.arange(5)[:,None].repeat(47,1),action=np.arange(4)[:,None].repeat(8,1))]
    result=windows(es,dict(horizon=4,observation_steps=2))
    assert result['raw_obs'][0,:,0].tolist()==[0,0]
    assert result['future_obs'][0,:,0].tolist()==[0,1,2,3]
    assert result['future_mask'][0,:,0].tolist()==[0,1,1,1]
    assert result['raw_obs'][-1,:,0].tolist()==[2,3]
    assert result['future_obs'][-1,:,0].tolist()==[3,4,4,4]
    assert result['future_mask'][-1,:,0].tolist()==[1,1,0,0]


def test_api_file_bytes_and_submission_identity_are_preserved(tmp_path):
    assignment=dict(policy_id='fixture',skill_id='fixture',heuristic_index=1)
    atomic(tmp_path/'assignment.json',assignment)
    tools=DesignTools(dict(design=dict(max_api_calls=1,max_tool_calls=5,max_checks=1,max_output_tokens=100)),tmp_path,4)
    try:
        content='# Synthetic interface fixture, not an experimental policy.\n'
        tools.dispatch('write_file',dict(path='policy.py',content=content))
        assert (tmp_path/'design/work/policy.py').read_text()==content
        assert tools.dispatch('read_file',dict(path='policy.py'))['content']==content
        tools.j.set('submitted',{'fixture':True})
        with pytest.raises(ValueError,match='immutable'):
            tools.dispatch('write_file',dict(path='policy.py',content='changed'))
    finally:tools.j.db.close()


def test_read_steps_supports_a_recurrent_skill_without_changing_source_indices(tmp_path):
    folder=tmp_path/'policy';dataset=tmp_path/'dataset.json'
    atomic(dataset,dict(segments=[dict(trajectory_id='demo',start=0,stop=2,file='a.json'),
        dict(trajectory_id='demo',start=4,stop=6,file='b.json')]))
    for filename,start in [('a.json',0),('b.json',4)]:
        atomic(tmp_path/filename,dict(observations=[dict(state={'index':i}) for i in range(start,start+3)],
            actions=[[i] for i in range(start,start+2)]))
    atomic(folder/'assignment.json',dict(policy_id='fixture',dataset=str(dataset)))
    tools=DesignTools(dict(design=dict(max_api_calls=1,max_tool_calls=5,max_checks=1,max_output_tokens=100)),folder,4)
    try:
        rows=tools.dispatch('read_steps',dict(trajectory_id='demo',indices=[0,2,4,6]))['rows']
        assert [r['state']['index'] for r in rows]==[0,2,4,6]
        assert [r['action'] for r in rows]==[[0],None,[4],None]
        with pytest.raises(ValueError,match='outside'):
            tools.dispatch('read_steps',dict(trajectory_id='demo',indices=[3]))
    finally:tools.j.db.close()


def test_new_goal_uses_three_spatial_predicates_without_terminal_hold():
    from appl.io import ROOT
    from appl.prior_policies.goals import measure
    contract=read(ROOT/'experiments/exp2/configs/demonstration_goals.json')
    state=dict(drawer_position=[.30],red_pose=[-.18,-.30,.02,1,0,0,0],
        blue_pose=[-.11,0,.063,1,0,0,0],qpos=[0]*9,tcp_pose=[-.11,0,.063,1,0,0,0])
    assert measure(state,contract)['success']
    state['drawer_position']=[.26]
    assert not measure(state,contract)['success']
    state['drawer_position']=[.30];state['red_pose'][0]=-.145
    assert measure(state,contract)['success']
    state['red_pose'][3:]=[np.cos(np.pi/8),0,0,np.sin(np.pi/8)]
    assert not measure(state,contract)['success']


def test_full_training_normalizer_is_shared_across_disjoint_skill_ranges(tmp_path,monkeypatch):
    from appl.prior_policies import data
    from appl.dp_baseline.data import FIELDS
    training=dict(observation_normalization='limits',quaternion_normalization='unit_component_bounds',
        action_scale_floor=.01,observation_std_floor=.001)
    source=tmp_path/'original';source.mkdir()
    def trajectory(identifier,values):
        observations=[]
        for x in values:
            state={name:[0.]*width for name,width in FIELDS};state['red_pose'][0]=x
            observations.append(dict(state=state))
        atomic(source/(identifier+'.json'),dict(trajectory_id=identifier,provenance='original_demonstration',
            observations=observations,actions=[[float(i)]*8 for i in range(len(values)-1)]))
    trajectory('train_a',[-.3,-.2,-.1,0.])
    trajectory('train_b',[.1,.2,.3,.4])
    trajectory('validation_unused',[-1000.,1000.])
    cfg=dict(output=tmp_path/'run',training=training,normalization=dict(scope='full_training_demonstrations',
        directory=str(source),train_ids=['train_a','train_b']))
    receipt=data.fit_full_normalizer(cfg)
    reference={k:receipt[k] for k in ('path','sha256','normalizer_sha256')}
    normal=read(receipt['path'])['normalizer']
    assert normal['fit_ids']==['train_a','train_b'] and normal['fit_samples']==6
    assert normal['mean'][25]==pytest.approx(0.)
    assert normal['std'][25]==pytest.approx(.3)
    assert normal['mean'][28:32]==[0.]*4 and normal['std'][28:32]==[1.]*4
    narrow=np.zeros((3,47),np.float32);narrow[:,25]=[-.1801,-.18,-.1799]
    monkeypatch.setattr(data,'episodes',lambda entry:[dict(id=entry['skill_id'],obs=narrow,action=np.zeros((2,8)))])
    specs=[]
    for skill in ('early_skill','late_skill'):
        entry=dict(experiment_version='M1_v2',skill_id=skill,normalization=reference)
        spec,_=data.specification(entry,training,dict(config={}));specs.append(spec)
    assert specs[0]['normalizer']==specs[1]['normalizer']==normal
    # A 2 cm shift in a nearly static skill feature stays ordinary under full-demo scaling.
    assert abs((-.20-normal['mean'][25])/normal['std'][25])<1
    assert data.fit_full_normalizer(cfg)==receipt
    value=read(receipt['path']);value['normalizer']['std'][25]=1e-5;atomic(receipt['path'],value)
    with pytest.raises(ValueError,match='artifact changed'):data.shared_normalizer(entry)


def test_missing_library_freeze_does_not_create_an_interrupted_episode(tmp_path,monkeypatch):
    from appl.prior_policies import deploy
    monkeypatch.setattr(deploy,'library',lambda cfg:{})
    cfg=dict(output=tmp_path,experiment_version='M1_v2',evaluation=dict(seeds=[6200],diagnostic_seeds=[]))
    with pytest.raises(FileNotFoundError,match='library_freeze'):
        deploy.evaluate(cfg,6200)
    assert not (tmp_path/'evaluation/6200').exists()


def test_budget_reevaluation_freezes_separately_and_rejects_changed_models(tmp_path,monkeypatch):
    from appl.prior_policies import deploy
    from appl.io import digest
    parent=tmp_path/'trained';output=parent/'budget_3000';dataset=tmp_path/'dataset'
    config=tmp_path/'new.json';contract=tmp_path/'goals.json'
    for path in (config,contract,dataset/'manifest.json',parent/'normalization.json'):atomic(path,{})
    policies={'fixture':dict(version='api_submission',checkpoint_sha256='original_checkpoint')}
    old=dict(policies=policies,completion_contract_sha256=digest(contract),
        dataset_manifest_sha256=digest(dataset/'manifest.json'),normalization_sha256=digest(parent/'normalization.json'))
    atomic(parent/'library_freeze.json',dict(library=old));before=(parent/'library_freeze.json').read_bytes()
    cfg=dict(output=parent,dataset=dataset,_path=str(config),completion_contract=contract,
        experiment_version='M1_v2',evaluation=dict(output=str(output),max_steps=3000))
    monkeypatch.setattr(deploy,'library',lambda cfg:policies)
    frozen=deploy.freeze(cfg)
    assert frozen['library']['policies']==policies
    assert (output/'library_freeze.json').exists()
    assert (parent/'library_freeze.json').read_bytes()==before
    policies['fixture']['checkpoint_sha256']='changed_checkpoint'
    with pytest.raises(ValueError,match='changed the frozen policy library'):deploy.freeze(cfg)


def test_reevaluation_missing_freeze_cannot_touch_original_episode(tmp_path,monkeypatch):
    from appl.prior_policies import deploy
    original=tmp_path/'evaluation/6200/result.json';atomic(original,dict(success=False))
    monkeypatch.setattr(deploy,'library',lambda cfg:{})
    study=tmp_path/'budget_3000'
    cfg=dict(output=tmp_path,experiment_version='M1_v2',
        evaluation=dict(output=str(study),seeds=[6200],diagnostic_seeds=[]))
    with pytest.raises(FileNotFoundError,match='library_freeze'):deploy.evaluate(cfg,6200)
    assert read(original)==dict(success=False)
    assert not (study/'evaluation/6200').exists()


@pytest.mark.parametrize('cap',[3000,5000])
def test_configured_maniskill_time_limit_really_allows_step_1501(tmp_path,monkeypatch,cap):
    import gymnasium as gym
    import torch
    from gymnasium.envs.registration import EnvSpec,WrapperSpec
    from appl.envs import native
    from appl.prior_policies.deploy import make_evaluation_environment
    class CounterEnvironment(gym.Env):
        metadata={}
        def __init__(self,**kwargs):
            self.num_envs=1;self.device='cpu';self.elapsed_steps=torch.zeros(1,dtype=torch.int64)
            self.action_space=gym.spaces.Box(-1.,1.,(1,));self.observation_space=gym.spaces.Box(-1.,1.,(1,))
        def reset(self,seed=None,options=None):self.elapsed_steps.zero_();return np.zeros(1),{}
        def step(self,action):
            self.elapsed_steps+=1
            return np.zeros(1),0.,torch.tensor([False]),torch.tensor([False]),{}
    spec=EnvSpec('APPLDrawerExchange-v2',entry_point=CounterEnvironment,max_episode_steps=1500,
        disable_env_checker=True,additional_wrappers=(WrapperSpec('TimeLimitWrapper',
            'mani_skill.utils.registration:TimeLimitWrapper',dict(max_episode_steps=1500)),))
    monkeypatch.setitem(gym.envs.registry,'APPLDrawerExchange-v2',spec)
    env=make_evaluation_environment(cap)
    try:
        env.reset()
        for step in range(1,cap+1):
            truncated=bool(env.step(np.zeros(1))[3].item())
            assert truncated==(step==cap)
    finally:env.close()
