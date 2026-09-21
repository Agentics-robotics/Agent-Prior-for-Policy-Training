import numpy as np
import pytest
from appl.config import load
from appl.data import windows,normalizer
from appl.envs.evaluator import SuccessTracker,extent_wxyz


def test_causal_history_boundary_mask_and_no_episode_crossing():
    cfg=dict(horizon=4,observation_steps=2)
    es=[dict(obs=np.arange(4)[:,None].repeat(47,1),action=np.arange(3)[:,None].repeat(8,1)),
        dict(obs=np.arange(100,104)[:,None].repeat(47,1),action=np.arange(100,103)[:,None].repeat(8,1))]
    w=windows(es,cfg)
    assert w['raw_obs'][0,:,0].tolist()==[0,0]
    assert w['native_action'][0,:,0].tolist()==[0,0,1,2]
    assert w['mask'][0,:,0].tolist()==[0,1,1,1]
    assert w['raw_obs'][3,:,0].tolist()==[100,100]
    assert w['native_action'][2,:,0].tolist()==[1,2,2,2]
    assert w['mask'][2,:,0].tolist()==[1,1,0,0]
    assert w['raw_obs'][2,-1,0]==2


def test_normalization_train_only_floor_and_native_roundtrip():
    e=dict(id='train',obs=np.ones((3,47)),action=np.stack([np.arange(8),np.arange(8)+.1]))
    norm=normalizer([e],dict(observation_std_floor=.001,action_scale_floor=.01))
    assert norm['fit_ids']==['train'] and norm['fit_samples']==2
    assert norm['std']==[.001]*47
    encoded=2*(e['action']-norm['action_min'])/norm['action_scale']-1
    assert np.allclose((encoded+1)*np.asarray(norm['action_scale'])/2+norm['action_min'],e['action'])


def test_release_and_physical_stability_are_required():
    tracker=SuccessTracker(load()['evaluation'])
    m=dict(drawer_open=True,gripper_released=True,red_on_pad=True,blue_inside=True,
           red_stable=True,blue_stable=True,red_clear=True,blue_clear=True,handoff_clear=True)
    for _ in range(9):assert not tracker.update(m)['success']
    assert tracker.update(m)['success']
    assert not tracker.update(dict(m,gripper_released=False))['success']
    for _ in range(30):assert not tracker.update(dict(m,red_stable=False))['success']
    assert not tracker.skill_succeeded('move_red')


def test_object_occupancy_uses_wxyz_rotation():
    assert np.allclose(extent_wxyz([1,0,0,0],.02),[.02]*3)
    e=extent_wxyz([np.cos(np.pi/8),0,0,np.sin(np.pi/8)],.02)
    assert np.allclose(e[:2],np.sqrt(2)*.02)


def test_gpu_authorization_rejects_devices_outside_latest_user_allocation():
    from appl.gpu import identity
    for index in (-1,8,9):
        with pytest.raises(ValueError):identity(index)


def test_unit_quaternion_scale_does_not_explode_on_small_tilt():
    from appl.data import SLICES
    e=dict(id='train',obs=np.zeros((3,47)),action=np.zeros((2,8)))
    for name in ('tcp_pose','red_pose','blue_pose'):e['obs'][:,SLICES[name][0]+3]=1.
    norm=normalizer([e],dict(observation_std_floor=.001,action_scale_floor=.01,quaternion_normalization='unit_component_bounds'))
    tilted=e['obs'][0].copy();tilted[29]=.01
    transformed=(tilted-norm['mean'])/norm['std']
    assert transformed[29]==.01 and norm['std'][0]==.001
