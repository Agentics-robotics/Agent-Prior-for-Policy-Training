"""Regression checks for the M0 numerical and deployment contracts."""
import numpy as np
import torch

from appl.dp_baseline.data import normalizer, windows
from appl.dp_baseline.train import sample


def test_limits_scaling_does_not_amplify_constant_dimensions():
    obs=np.zeros((3,47));obs[:,0]=[0.,2.,1.];obs[:,28]=1.
    episode=dict(id='train',obs=obs,action=np.zeros((2,8)))
    n=normalizer([episode],dict(observation_normalization='limits',
        quaternion_normalization='unit_component_bounds',action_scale_floor=.01))
    perturbation=obs[0].copy();perturbation[29]=.01;perturbation[41]=.01
    normalized=(perturbation-n['mean'])/n['std']
    assert normalized[0]==-1.
    assert normalized[29]==.01 and normalized[41]==.01
    assert n['fit_samples']==2  # The terminal observation is never fitted.


def test_first_deployed_action_is_the_current_action():
    episode=dict(obs=np.arange(6)[:,None].repeat(47,axis=1),
                 action=np.arange(5)[:,None].repeat(8,axis=1))
    cfg=dict(horizon=4,observation_steps=2)
    batch=windows([episode],cfg)
    assert np.array_equal(batch['raw_obs'][:,-1,0],batch['native_action'][:,1,0])


def test_both_samplers_reproduce_clean_action_with_an_exact_denoiser():
    # A fixed clean action is known independently. Correct epsilon prediction
    # must recover it through either reverse process, including the final step.
    from appl.dp_baseline.train import scheduler
    from appl.baseline import decode_action as decode_native
    cfg=dict(denoising_train_steps=100,denoising_inference_steps=100,
             horizon=16,clip_sample=True)
    spec=dict(training=cfg,normalizer=dict(action_min=[-1.]*8,action_scale=[2.]*8))
    class Mechanism:
        condition=staticmethod(lambda raw,spec:raw.flatten(1))
        decode_action=staticmethod(decode_native)
    alpha=scheduler(cfg).alphas_cumprod
    target=torch.linspace(-.8,.8,128).reshape(1,16,8)
    def exact(noisy,timestep,condition):
        a=alpha[timestep]
        return (noisy-a.sqrt()*target)/(1-a).sqrt()
    for kind in ('ddpm','ddim'):
        cfg['sampler']=kind
        result=sample(exact,Mechanism,torch.zeros(1,2,47),spec,torch.Generator().manual_seed(12))
        torch.testing.assert_close(result,target,atol=1e-5,rtol=0.)
