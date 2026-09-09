"""D2-only deterministic mathematical audit. No optimizer or environment rollout."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

from relative_dp.config import TRAIN_CONFIG
from relative_dp.dataset import WindowDataset
from relative_dp.model import masked_epsilon_loss
from round2.learning import Policy
from round3.door_plugin import DoorPlugin, _rotate, _yaw, _door_direction
from round3.learning import Normalizer
from round3.proposals import validate

def main():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[3]
    evidence = root / 'round3/evidence/door'
    proposal_path = root / 'round3/design_records/door/proposal.json'
    proposal = json.loads(proposal_path.read_text())
    schema = json.loads((evidence/'task_materials.json').read_text())['observation_schema']
    episodes, ids, hashes = [], [], {}
    for index in (1, 2):
        path = evidence/f'demo_{index}_trajectory.npz'
        with np.load(path, allow_pickle=False) as data:
            episodes.append({key: data[key].copy() for key in ('obs', 'actions')})
        ids.append(f'bundled_demo_{index}')
        hashes[ids[-1]] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(episodes) == 2
    validation = validate('door', freeze=False)
    assert validation['status'] == 'validated'
    cfg_base = dict(TRAIN_CONFIG, clip_predicted_clean_actions=False,
                    thresholding=False, torch_compile=False, inference_torch_compile=False)
    torch.manual_seed(0)
    base = Policy(41, cfg_base)
    baseline_parameters = sum(p.numel() for p in base.parameters())
    del base
    report = {'created_at': datetime.now(timezone.utc).isoformat(),
              'proposal_sha256': hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
              'plugin_sha256': hashlib.sha256((root/'src/round3/door_plugin.py').read_bytes()).hexdigest(),
              'test_scope': 'Exactly two bundled D2 trajectories, synthetic inputs, random untrained networks; no optimizer updates or environment episodes',
              'candidate_results': {}, 'baseline_parameters': baseline_parameters}
    for candidate in proposal['candidates']:
        cid = candidate['candidate_id']
        implementation = json.loads((root/f'round3/configs/door/{cid}.json').read_text())
        assert implementation['proposal_sha256'] == report['proposal_sha256']
        plugin = DoorPlugin('door', schema, implementation['plugin_config'])
        normalizer = Normalizer.fit(episodes, plugin, ids, hashes)
        assert np.array_equal(normalizer.mean, np.zeros_like(normalizer.mean))
        assert np.array_equal(normalizer.std, np.ones_like(normalizer.std))
        ds = WindowDataset(episodes, normalizer, 16)
        raw = np.stack([episodes[e]['obs'][t] for e,t in ds.index])
        histories = np.stack([episodes[e]['obs'][[max(0,t-1),t]] for e,t in ds.index])
        normalized = normalizer.normalize_history(histories)
        assert normalized.shape == (220,2,candidate['architecture']['observation_dim'])
        assert np.isfinite(normalized).all()
        np.testing.assert_array_equal(normalized[..., :41], histories)
        original = ds.actions.numpy().copy()
        encoded = plugin.action_encode(original, raw)
        decoded = plugin.action_decode(encoded, raw)
        np.testing.assert_allclose(decoded, original, atol=3e-7, rtol=3e-7)
        np.testing.assert_array_equal(encoded[...,3],original[...,3])
        targets = plugin.supervision(episodes,ds.index)
        packed = plugin.training_actions(encoded,targets)
        assert packed.shape == (220,16,candidate['architecture']['action_dim'])
        assert np.all(packed[ds.valid_mask.numpy()==0] == 0)
        np.testing.assert_allclose(plugin.action_decode(packed,raw),original,atol=3e-7,rtol=3e-7)
        cfg = dict(cfg_base, obs_dim=normalized.shape[-1])
        cfg.update(plugin.model_config(cfg))
        torch.manual_seed(0)
        policy = plugin.build_policy(cfg)
        parameters = sum(p.numel() for p in policy.parameters())
        assert parameters <= 3*baseline_parameters
        # Actual valid terminal transitions must be included, never fabricated tail labels.
        if plugin.joint:
            labels = targets['future_geometry']
            for row in range(len(ds)):
                e,t = ds.index[row]
                valid = min(16,len(episodes[e]['actions'])-t)
                future = episodes[e]['obs'][t+1:t+valid+1]
                expected_displacement = (future[:,4:7]-raw[row,4:7])/.1
                expected_gap = (future[:,:3]-future[:,4:7])/.1
                expected_direction = _door_direction(future)
                if plugin.cabinet:
                    expected_displacement = _rotate(expected_displacement,_yaw(raw[row]))
                    expected_gap = _rotate(expected_gap,_yaw(raw[row]))
                    expected_direction = _rotate(np.concatenate([expected_direction,np.zeros((valid,1),np.float32)],-1),_yaw(raw[row]))[:,:2]
                expected = np.concatenate([expected_displacement,expected_gap,expected_direction],axis=-1)
                np.testing.assert_allclose(labels[row,:valid],expected,atol=1e-6,rtol=1e-6)
            fake_output = torch.zeros(2,16,12,requires_grad=True)
            fake_noise = torch.zeros_like(fake_output)
            fake_noise[...,4:] = 2.
            mask = torch.zeros(2,16);mask[:,:3] = 1.
            extra,_ = plugin.extra_loss(fake_output,{'noise':fake_noise,'mask':mask},0)
            np.testing.assert_allclose(extra.detach().item(),.8,rtol=1e-6)
            extra.backward()
            assert torch.count_nonzero(fake_output.grad[...,:4]) == 0
            assert torch.count_nonzero(fake_output.grad[:,3:]) == 0
            fake_output2 = torch.zeros(2,16,12)
            fake_noise2 = fake_noise.detach().clone();fake_noise2[:,3:,:] = 1000.
            extra2,_ = plugin.extra_loss(fake_output2,{'noise':fake_noise2,'mask':mask},19000)
            np.testing.assert_allclose(extra2.item(),.8,rtol=1e-6)
        # A yaw-changing synthetic history checks coherent axes, not physical augmentation.
        synthetic = histories[:2].copy()
        synthetic[:,0,39:41] = np.array([0.,1.])
        synthetic[:,1,39:41] = np.array([1.,0.])
        synthetic_encoded = plugin.observation_history(synthetic)
        if plugin.cabinet:
            expected = _rotate(synthetic[...,:3]-synthetic[...,4:7],_yaw(synthetic[:,-1,:]))/.1
            np.testing.assert_allclose(synthetic_encoded[...,41:44],expected,atol=1e-6)
        # The action frame stays unbounded; inverse decode precedes external clipping.
        arbitrary = np.array([[[2.,-3.,4.,1.]]],np.float32)
        np.testing.assert_allclose(plugin.action_decode(plugin.action_encode(arbitrary,raw[:1]),raw[:1]),arbitrary,atol=1e-6)
        assert np.max(np.abs(plugin.action_decode(arbitrary,raw[:1]))) > 1
        # Shared diffusion noise, forward and gradients, without optimization.
        obs = torch.from_numpy(normalized[:2])
        clean = torch.from_numpy(packed[:2])
        mask = ds.valid_mask[:2]
        noisy,timestep,noise = policy.noise_targets(clean,torch.Generator().manual_seed(200))
        output = policy(noisy,timestep,obs)
        loss = masked_epsilon_loss(output[...,:4],noise[...,:4],mask)
        extra,_ = plugin.extra_loss(output,{'noise':noise,'mask':mask},0)
        if extra is not None:
            loss = loss+extra
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters())
        policy.zero_grad(set_to_none=True)
        policy.eval()
        sample = policy.predict_action(obs[:1],torch.Generator().manual_seed(800))
        assert sample.shape == (1,16,cfg['action_dim']) and torch.isfinite(sample).all()
        assert policy.inference_scheduler.timesteps.tolist() == list(range(90,-1,-6))
        assert policy.inference_scheduler.config.clip_sample is False
        if cid == 'P1':
            # Same parameter shapes/weights as inherited Policy; exact sampler equality.
            reference = Policy(cfg['obs_dim'],cfg)
            reference.load_state_dict(policy.state_dict())
            reference.eval()
            reference_sample = reference.predict_action(obs[:1],torch.Generator().manual_seed(800))
            torch.testing.assert_close(reference_sample,sample,rtol=0,atol=0)
            del reference
        report['candidate_results'][cid] = {
            'passed':True,'observation_shape':list(normalized.shape),'target_shape':list(packed.shape),
            'parameters':parameters,'parameter_ratio':parameters/baseline_parameters,
            'world_action_inverse_max_error':float(np.max(np.abs(decoded-original))),
            'tail_zero_padding_and_mask':True,'fixed_passthrough_normalization':True,
            'coherent_current_anchor_history':True,'next_state_labels_all_valid_slots_verified':plugin.joint,
            'auxiliary_weight_and_tail_gradient_verified':plugin.joint,
            'finite_random_forward_backward':True,'exact_DDIM_timesteps':policy.inference_scheduler.timesteps.tolist(),
            'DDIM_unbounded_sampling':True,'P1_sampler_equals_inherited_Policy':cid=='P1',
        }
        del policy
    report['passed'] = True
    output = root/'round3/audits/door_plugin_v1.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
