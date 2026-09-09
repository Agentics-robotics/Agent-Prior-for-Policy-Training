"""Reproducible implementation checks using only synthetic arrays and bundled D2.

Run with: /home/users/oscar/.pixi/bin/pixi run python round3/audits/peg_plugin_checks.py
No optimizer update, environment, learned checkpoint, or outcome is used.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from relative_dp.config import TRAIN_CONFIG
from relative_dp.dataset import WindowDataset
from relative_dp.model import masked_epsilon_loss
from relative_dp.utils import ROOT, object_hash
from round2.learning import Policy
from round3.learning import Normalizer
from round3.peg_plugin import PegJointPolicy, PegPlugin


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    evidence = ROOT / 'round3/evidence/peg-insert-side'
    schema = json.loads((evidence / 'task_materials.json').read_text())['observation_schema']
    paths = [evidence / f'demo_{i}_trajectory.npz' for i in (1, 2)]
    episodes = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            episodes.append(dict(obs=data['obs'].copy(), actions=data['actions'].copy()))
    ids = [p.name for p in paths]
    hashes = {p.name: digest(p) for p in paths}
    cfg_base = dict(TRAIN_CONFIG, clip_predicted_clean_actions=False, thresholding=False,
                    torch_compile=False, inference_torch_compile=False)
    torch.manual_seed(0)
    baseline = Policy(42, cfg_base)
    baseline_parameters = sum(p.numel() for p in baseline.parameters())
    report = {}
    for candidate in ('P1', 'P2', 'P3'):
        plugin = PegPlugin('peg-insert-side', schema, {'candidate_id': candidate})
        normalizer = Normalizer.fit(episodes, plugin, ids, hashes)
        ds = WindowDataset(episodes, normalizer, 16)
        assert len(ds) == 168
        np.testing.assert_array_equal(normalizer.mean, np.zeros_like(normalizer.mean))
        np.testing.assert_array_equal(normalizer.std, np.ones_like(normalizer.std))
        expected_obs_dim = 62 if candidate in ('P1', 'P3') else 42
        assert ds.observations.shape == (168, 2, expected_obs_dim)
        raw = np.stack([e['obs'][0] for e in episodes])
        encoded_obs = plugin.observation(raw)
        np.testing.assert_array_equal(encoded_obs[..., :42], raw)
        if plugin.relations:
            np.testing.assert_allclose(encoded_obs[:, 42:45], (raw[:, 4:7]-raw[:, :3])/.1)
            np.testing.assert_allclose(encoded_obs[:, 45:48], (raw[:, 36:39]-raw[:, 39:42])/.1)
            np.testing.assert_allclose(encoded_obs[:, 48:51], (raw[:, 39:42]-raw[:, 4:7])/.1)
            np.testing.assert_allclose(encoded_obs[:, 51:54], (raw[:, 36:39]-raw[:, :3])/.1)
            np.testing.assert_allclose(encoded_obs[:, 54:57], (raw[:, :3]-raw[:, 18:21])/.01)
            np.testing.assert_allclose(encoded_obs[:, 57:60], (raw[:, 4:7]-raw[:, 22:25])/.01)
            np.testing.assert_allclose(encoded_obs[:, 60], np.linalg.norm(raw[:, 4:6]-raw[:, :2], axis=-1)/.1)
            np.testing.assert_allclose(encoded_obs[:, 61], np.linalg.norm(raw[:, 37:39]-raw[:, 40:42], axis=-1)/.1)
        for row, (episode_index, start) in enumerate(ds.index):
            history = episodes[episode_index]['obs'][[max(0, start-1), start]]
            np.testing.assert_array_equal(normalizer.normalize_history(history), ds.observations[row].numpy())

        world = np.array([[[2., -3., .5, 4.]]], np.float32)
        np.testing.assert_array_equal(plugin.action_encode(world, raw[:1]), world)
        sample = np.concatenate((world, np.full((1, 1, 6), 99., np.float32)), axis=-1) if plugin.joint else world
        np.testing.assert_array_equal(plugin.action_decode(sample, raw[0]), world)
        # Clipping belongs to the caller and affects physical4 only after decoding.
        np.testing.assert_array_equal(np.clip(plugin.action_decode(sample, raw[0]), -1, 1),
                                      [[[1., -1., .5, 1.]]])
        targets = plugin.supervision(episodes, ds.index)
        actions = plugin.training_actions(ds.actions.numpy(), targets)
        np.testing.assert_array_equal(actions[..., :4], ds.actions.numpy())
        assert np.all(actions[ds.valid_mask.numpy() == 0] == 0)
        if plugin.joint:
            for row, (episode_index, start) in enumerate(ds.index):
                episode = episodes[episode_index]
                valid = min(16, len(episode['actions'])-start)
                for slot in range(valid):
                    current, future = episode['obs'][start], episode['obs'][start+slot+1]
                    expected = np.concatenate(((future[39:42]-current[39:42])/.1,
                                               (future[4:7]-future[:3])/.1))
                    np.testing.assert_array_equal(actions[row, slot, 4:], expected)
            # Padded errors and physical action errors cannot enter the auxiliary loss.
            prediction = torch.zeros((1, 2, 10), requires_grad=True)
            with torch.no_grad():
                prediction[:, 0, 4:] = 2
                prediction[:, 1, :] = 999
                prediction[:, 0, :4] = 33
            mask = torch.tensor([[1., 0.]])
            aux, _ = plugin.extra_loss(prediction, dict(noise=torch.zeros_like(prediction), mask=mask), 0)
            torch.testing.assert_close(aux, torch.tensor(.8))
            aux.backward()
            assert torch.all(prediction.grad[..., :4] == 0)
            assert torch.all(prediction.grad[:, 1] == 0)
        else:
            assert targets == {} and plugin.extra_loss(None, {}, 0) == (None, {})

        generator = torch.Generator(device='cpu').manual_seed(100)
        expected_generator = torch.Generator(device='cpu').manual_seed(100)
        _, _, _, indices = ds.sample(128, generator)
        torch.testing.assert_close(indices, torch.randint(168, (128,), generator=expected_generator))
        cfg = dict(cfg_base, obs_dim=expected_obs_dim)
        cfg.update(plugin.model_config(cfg))
        torch.manual_seed(0)
        policy = plugin.build_policy(cfg)
        parameter_count = sum(p.numel() for p in policy.parameters())
        assert parameter_count <= 3 * baseline_parameters
        clean = torch.from_numpy(actions[[0, 76]])
        noisy, timesteps, noise = policy.noise_targets(clean, torch.Generator().manual_seed(200))
        output = policy(noisy, timesteps, ds.observations[[0, 76]])
        assert output.shape == clean.shape
        batch = dict(noise=noise, mask=ds.valid_mask[[0, 76]])
        loss = masked_epsilon_loss(output[..., :4], noise[..., :4], batch['mask'])
        auxiliary, _ = plugin.extra_loss(output, batch, 0)
        if auxiliary is not None:
            loss = loss + auxiliary
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters())
        # Untrained structural inference only: no rollout, score, optimizer, or checkpoint.
        policy.eval()
        drawn = policy.predict_action(ds.observations[:1], torch.Generator().manual_seed(713))
        assert drawn.shape == (1, 16, cfg['action_dim']) and torch.isfinite(drawn).all()
        assert policy.inference_scheduler.timesteps.tolist() == list(range(90, -1, -6))
        assert not policy.inference_scheduler.config.clip_sample
        report[candidate] = dict(parameter_count=parameter_count,
                                 parameter_ratio=parameter_count/baseline_parameters,
                                 obs_dim=expected_obs_dim, diffusion_dim=cfg['action_dim'],
                                 feature_alignment=True, fixed_normalization=True,
                                 all_windows_checked=168, current_D2_only=True,
                                 tail_padding_and_terminal_next_state=True,
                                 action_unclipped_until_after_decode=True,
                                 auxiliary_loss_and_mask=True, finite_forward_backward=True,
                                 untrained_sampler_shape_and_finiteness=True,
                                 ddim_timesteps=list(range(90, -1, -6)),
                                 cpu_uniform_sampling=True, optimizer_updates=0)

    # The dimension-only sampler override is numerically the baseline for action_dim4.
    torch.manual_seed(0)
    joint4 = PegJointPolicy(42, cfg_base)
    joint4.load_state_dict(baseline.state_dict())
    obs = torch.from_numpy(episodes[0]['obs'][[0, 0]])[None]
    original = baseline.predict_action(obs, torch.Generator().manual_seed(41))
    dynamic = joint4.predict_action(obs, torch.Generator().manual_seed(41))
    torch.testing.assert_close(original, dynamic, rtol=0, atol=0)
    assert original.abs().max() > 1  # No hidden final-sample clamp.
    proposal = ROOT / 'round3/design_records/peg-insert-side/proposal.json'
    source = ROOT / 'src/round3/peg_plugin.py'
    result = dict(status='passed', created_at=datetime.now(timezone.utc).isoformat(),
                  task='peg-insert-side', proposal_sha256=digest(proposal),
                  plugin_sha256=digest(source), audit_script_sha256=digest(Path(__file__)),
                  evidence_trajectories=hashes, baseline_parameter_count=baseline_parameters,
                  four_channel_sampler_exact_equality=True, no_clean_or_final_sample_clipping=True,
                  candidates=report, outcome_exposure='None; no learned checkpoint or rollout.',
                  formal_training_complete=False, evaluation_complete=False,
                  implementation_source_reads=['src/round3/plugins.py', 'src/round3/learning.py',
                      'src/round3/proposals.py', 'src/round2/learning.py',
                      'src/relative_dp/config.py', 'src/relative_dp/dataset.py',
                      'src/relative_dp/model.py', 'src/round3/common.py selected path helper',
                      'src/relative_dp/utils.py selected object_hash helper'],
                  environment_interaction=False)
    for candidate in ('P1', 'P2', 'P3'):
        path = ROOT / f'round3/configs/peg-insert-side/{candidate}.json'
        config = json.loads(path.read_text())
        assert config['proposal_sha256'] == result['proposal_sha256']
        assert config['observation_schema_sha256'] == object_hash(schema)
        result['candidates'][candidate]['implementation_config_sha256'] = digest(path)
    target = ROOT / 'round3/audits/peg_plugin_implementation.json'
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
