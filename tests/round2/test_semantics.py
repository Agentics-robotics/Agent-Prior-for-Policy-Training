import copy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from round2.geometry import GeometryEnv, rz
from round2.calibrate import record
from round2.representation import transform, action_transform, Normalizer, PASSTHROUGH
from relative_dp.dataset import WindowDataset


@pytest.mark.parametrize('task',['drawer','door'])
def test_rotated_physics_goal_history_and_reset(task):
    e=GeometryEnv(task)
    try:
        baseline,_=e.reset(record(task,29000000,0))
        base=e.base.copy();h0=baseline[4:7].copy();goal0=e.goal.copy()
        other=[i for i in range(e.model.nbody) if e.model.body(i).name in ('tablelink','base')]
        static=e.data.xpos[other].copy()
        for yaw in (-50,50,-10,10,0):
            rec=record(task,29000000,yaw)
            obs,info=e.reset(rec);snap=e.snapshot(obs)
            np.testing.assert_allclose(obs[4:7],base+rz(yaw)@(h0-base),atol=1e-9)
            np.testing.assert_allclose(e.goal,base+rz(yaw)@(goal0-base),atol=1e-9)
            np.testing.assert_allclose(e.data.body(task).xmat.reshape(3,3),rz(yaw),atol=1e-9)
            np.testing.assert_array_equal(obs[:18],obs[18:36])
            np.testing.assert_allclose(e.data.xpos[other],static,atol=1e-9)
            assert not info['success'] and abs(info['progress'])<1e-12
            for _ in range(12):
                nxt,_,_,_,info=e.step(np.zeros(4))
                np.testing.assert_allclose(nxt[18:36],obs[:18],atol=1e-12)
                obs=nxt
                assert not info['success'] and abs(info['progress'])<.02
            obs,_=e.reset(rec)
            for k,v in e.snapshot(obs).items():np.testing.assert_allclose(v,snap[k],atol=1e-9,rtol=0)
    finally:e.close()


def test_information_and_action_roundtrip_with_unbounded_frame_component():
    e=GeometryEnv('door')
    try:
        obs,_=e.reset(record('door',29000001,45))
        x=np.stack([obs,obs]);x[1,:3]+=[.02,-.04,.01]
        frame=transform(x,'frame');back=transform(frame,'frame',inverse=True)
        np.testing.assert_allclose(back,x,atol=1e-12)
        np.testing.assert_array_equal(frame[:,4:7],x[:,4:7])
        np.testing.assert_array_equal(frame[:,11:18],0*x[:,11:18])
        a=np.array([[1.,1.,.4,-.8]])
        local=action_transform(a,obs,'frame')
        assert local[0,0]>1.4
        np.testing.assert_allclose(action_transform(local,obs,'frame',inverse=True),a,atol=1e-12)
        assert not np.allclose(action_transform(np.clip(local,-1,1),obs,'frame',inverse=True),a)
    finally:e.close()


def test_normalizer_provenance_passthrough_and_window_alignment():
    e=GeometryEnv('drawer')
    try:
        obs,_=e.reset(record('drawer',29000002,-7))
        observations=np.repeat(obs[None],4,axis=0).astype(np.float32)
        observations[:,0]+=[0.,.01,.02,.9]
        episode=dict(obs=observations,actions=np.arange(12,dtype=np.float32).reshape(3,4)/12)
        n=Normalizer.fit([episode],'frame',['only_train'],{'only_train':'hash'})
        assert n.count==3 and n.source_ids==['only_train']
        z=n.normalize(observations)
        np.testing.assert_array_equal(z[:,PASSTHROUGH],transform(observations,'frame')[:,PASSTHROUGH])
        assert abs(z[-1,0])>10
        ds=WindowDataset([episode],n)
        np.testing.assert_array_equal(ds.observations[0,0],ds.observations[0,1])
        np.testing.assert_array_equal(ds.actions[1,:2],episode['actions'][1:])
        assert ds.valid_mask[1].sum()==2
        assert ds.actions[1,2:].count_nonzero()==0
        assert Normalizer.from_dict(n.as_dict()).as_dict()==n.as_dict()
    finally:e.close()


def test_success_requires_three_valid_steps():
    import mujoco
    e=GeometryEnv('drawer')
    try:
        e.reset(record('drawer',29000003,0))
        for i in range(3):
            e.data.qpos[e.qadr]=e.q_open*.8
            e.data.qvel[:]=0
            mujoco.mj_forward(e.model,e.data)
            _,_,_,_,info=e.step(np.zeros(4))
            assert info['success']==(i==2)
        e.data.qpos[e.qadr]=e.q_open*1.5
        assert not e.diagnostics()['valid_joint']
        e.streak=0
    finally:e.close()


def test_schedulers_and_clean_output_are_unclipped():
    import torch
    from round2.learning import initialized,config
    cfg=config('r2_drawer_frame_n20_s0')
    p=initialized(cfg,'cpu')
    assert not p.train_scheduler.config.clip_sample and not p.inference_scheduler.config.clip_sample
    assert not p.train_scheduler.config.thresholding and not p.inference_scheduler.config.thresholding
    # Run the real DDIM update with a bounded-cost analytical epsilon denoiser;
    # zero epsilon retains large clean values which the old final clamp erased.
    p.forward=lambda actions,t,obs:torch.zeros_like(actions)
    result=p.predict_action(torch.zeros(1,2,41),torch.Generator().manual_seed(41))
    assert result.abs().max()>1
