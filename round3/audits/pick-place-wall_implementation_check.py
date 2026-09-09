"""CPU correctness checks using only the bundled D2, never model evaluation."""
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from relative_dp.config import TRAIN_CONFIG
from relative_dp.dataset import WindowDataset
from relative_dp.model import masked_epsilon_loss, state_hash
from relative_dp.train import runtime_setup
from round2.learning import Policy
from round3.learning import Normalizer, Trainer
from round3.pickwall_plugin import PickwallPlugin, RHO, persistent_events, phase_labels, sphere_clearance


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    runtime_setup()
    root = Path(__file__).resolve().parents[2]
    evidence = root / 'round3/evidence/pick-place-wall'
    design = root / 'round3/design_records/pick-place-wall'
    proposal = json.loads((design / 'proposal.json').read_text())
    amendment = json.loads((design / 'initial_feasibility_revision.json').read_text())
    assert proposal['version'] == 1 and amendment['version'] == 2
    assert amendment['substantive_feasibility_revisions_used'] == 1
    assert amendment['original_proposal_sha256'] == digest(design / 'proposal.json')
    assert json.loads((design / 'initial_freeze.json').read_text())['proposal_sha256'] == digest(design / 'proposal.json')
    assert all(digest(root / p) == h for p, h in proposal['evidence_hashes'].items())
    schema = json.loads((evidence / 'task_materials.json').read_text())['observation_schema']
    episodes = []
    for index in (1, 2):
        with np.load(evidence / f'demo_{index}_trajectory.npz', allow_pickle=False) as data:
            episodes.append({'obs': data['obs'].copy(), 'actions': data['actions'].copy()})
    events = [list(persistent_events(e['obs'])) for e in episodes]
    counts = [np.bincount(phase_labels(e['obs']), minlength=4).tolist() for e in episodes]
    assert events == [[33, 45, 76], [35, 47, 74]]
    assert counts == [[33, 12, 31, 3], [35, 12, 27, 10]]
    center = torch.tensor([0.1, 0.75, 0.06])
    half = torch.tensor([0.12, 0.01, 0.06])
    probes = torch.stack((center, center + torch.tensor([0.0, 0.0, 0.11])))
    torch.testing.assert_close(sphere_clearance(probes, center, half), torch.tensor([-0.01 - RHO, 0.05 - RHO]), atol=1e-7, rtol=1e-6)

    baseline_cfg = dict(TRAIN_CONFIG, clip_predicted_clean_actions=False, action_dim=4)
    baseline = Policy(45, baseline_cfg)
    baseline_count = sum(p.numel() for p in baseline.parameters())
    del baseline
    report = {
        'task': 'pick-place-wall', 'passed': False, 'device': 'cpu',
        'scope': 'Correctness-only D2 checks, no simulator rollout, no development/test states, no formal training or model-score comparison.',
        'proposal_sha256': digest(design / 'proposal.json'),
        'initial_feasibility_revision_sha256': digest(design / 'initial_feasibility_revision.json'),
        'plugin_sha256': digest(root / 'src/round3/pickwall_plugin.py'),
        'check_script_sha256': digest(__file__),
        'data_sha256': {f'demo_{i}_trajectory.npz': digest(evidence / f'demo_{i}_trajectory.npz') for i in (1, 2)},
        'baseline_parameter_count': baseline_count,
        'persistent_event_steps': events, 'phase_action_anchor_counts': counts,
        'common_checks': ['Original proposal/freeze and sole amendment identity', 'All23 evidence SHA256 values', 'Exact D2 event rules/counts', 'Signed box/sphere clearance analytic probes'],
        'candidates': {},
    }
    for candidate in ('P1', 'P2', 'P3'):
        implementation = json.loads((root / 'round3/configs/pick-place-wall' / f'{candidate}.json').read_text())
        assert implementation['proposal_sha256'] == report['proposal_sha256']
        assert implementation['initial_feasibility_revision_sha256'] == report['initial_feasibility_revision_sha256']
        plugin = PickwallPlugin('pick-place-wall', schema, implementation['plugin_config'])
        normalizer = Normalizer.fit(episodes, plugin, ['bundled_demo_1', 'bundled_demo_2'], report['data_sha256'])
        assert np.array_equal(normalizer.mean, np.zeros_like(normalizer.mean))
        assert np.array_equal(normalizer.std, np.ones_like(normalizer.std))
        ds = WindowDataset(episodes, normalizer, 16)
        ds.plugin = plugin
        raw = np.stack([episodes[e]['obs'][t] for e, t in ds.index])
        ds.raw_current = torch.from_numpy(raw)
        targets = plugin.supervision(episodes, ds.index)
        ds.extra_targets = {k: torch.as_tensor(v) for k, v in targets.items()}
        ds.actions = torch.from_numpy(plugin.training_actions(ds.actions.numpy(), targets))
        obs_dim = 45 if candidate == 'P2' else 80
        action_dim = 10 if candidate == 'P3' else 4
        assert ds.observations.shape == (163, 2, obs_dim)
        assert ds.actions.shape == (163, 16, action_dim)
        assert torch.count_nonzero(ds.actions[ds.valid_mask == 0]) == 0
        for index, (episode_index, t) in enumerate(ds.index):
            e = episodes[episode_index]
            valid = min(16, len(e['actions']) - t)
            np.testing.assert_array_equal(ds.actions[index, :valid, :4], e['actions'][t:t + valid])
            if candidate == 'P3':
                actual = ds.actions[index, :valid, 4:].numpy()
                expected = np.concatenate(((e['obs'][t + 1:t + valid + 1, :3] - e['obs'][t, :3]) / 0.10, (e['obs'][t + 1:t + valid + 1, 4:7] - e['obs'][t, 4:7]) / 0.10), axis=-1)
                np.testing.assert_array_equal(actual, expected)
        first_history = episodes[0]['obs'][[0, 0]]
        np.testing.assert_array_equal(ds.observations[0].numpy(), plugin.observation(first_history))
        opposite = first_history.copy()
        for start in (7, 14, 25, 32):
            opposite[..., start:start + 4] *= -1
        np.testing.assert_array_equal(plugin.observation(first_history), plugin.observation(opposite))
        extreme = np.zeros((16, action_dim), np.float32)
        extreme[:, :4] = [3, -4, 5, 2]
        if candidate == 'P3':
            extreme[:, 4:] = 1e6
        np.testing.assert_array_equal(plugin.action_decode(extreme, raw[0]), extreme[:, :4])
        cfg = dict(TRAIN_CONFIG, task='pick-place-wall', candidate_id=candidate, obs_dim=obs_dim, raw_dim=45,
                   clip_predicted_clean_actions=False, thresholding=False, implementation=implementation,
                   observation_schema=schema, custom_loss=candidate != 'P1', torch_compile=False,
                   inference_torch_compile=False, debug=True)
        cfg.update(plugin.model_config(cfg))
        trainer = Trainer(ds, cfg, 'cpu')
        count = sum(p.numel() for p in trainer.policy.parameters())
        assert count <= 3 * baseline_count
        xobs, action, mask = ds.observations[:2], ds.actions[:2], ds.valid_mask[:2]
        noisy, timestep, noise = trainer.policy.noise_targets(action, torch.Generator().manual_seed(999))
        expected_forward = trainer.policy(noisy, timestep, xobs)
        captured = torch.compile(trainer.policy, backend='eager', fullgraph=True)
        actual_forward = captured(noisy, timestep, xobs)
        if isinstance(expected_forward, dict):
            for key in expected_forward:
                torch.testing.assert_close(expected_forward[key], actual_forward[key], atol=0, rtol=0)
        else:
            torch.testing.assert_close(expected_forward, actual_forward, atol=0, rtol=0)

        if candidate == 'P3':
            selected = torch.tensor([0, 78])
            target_action, target_mask = ds.actions[selected], ds.valid_mask[selected]
            target_noise = torch.randn(target_action.shape, generator=torch.Generator().manual_seed(700))
            t = torch.tensor([0, 0])
            target_noisy = trainer.policy.train_scheduler.add_noise(target_action, target_noise, t)
            batch = dict(actions=target_action, mask=target_mask, noisy=target_noisy, noise=target_noise, timesteps=t,
                         raw_current=ds.raw_current[selected], targets={k: v[selected] for k, v in ds.extra_targets.items()}, policy=trainer.policy)
            perfect, perfect_metrics = plugin.extra_loss(target_noise, batch, 999)
            assert abs(float(perfect)) < 1e-8
            altered = target_noise.clone()
            altered[target_mask == 0] += 100
            padded, _ = plugin.extra_loss(altered, batch, 999)
            torch.testing.assert_close(perfect, padded, atol=1e-8, rtol=0)
            torch.testing.assert_close(masked_epsilon_loss(target_noise[..., :4], target_noise[..., :4], target_mask), masked_epsilon_loss(altered[..., :4], target_noise[..., :4], target_mask), atol=0, rtol=0)
            high_t = torch.tensor([99, 99])
            high_noisy = trainer.policy.train_scheduler.add_noise(target_action, target_noise, high_t)
            high_batch = dict(batch, timesteps=high_t, noisy=high_noisy)
            _, high_metrics = plugin.extra_loss(target_noise + 1, high_batch, 999)
            assert float(high_metrics['geometry_consistency']) == 0
            assert perfect_metrics['geometry_weight'] == 0.05

        # Three disposable updates total: original1->2 plus restored1->2.
        first = trainer.update()
        assert first['step'] == 1 and np.isfinite(first['loss']) and np.isfinite(first['grad_norm'])
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainer.policy.parameters())
        metadata = {'correctness_audit': candidate}
        checkpoint = copy.deepcopy(trainer.checkpoint(metadata))
        second = trainer.update()
        expected_model = state_hash(trainer.policy.state_dict())
        expected_ema = state_hash(trainer.ema.state_dict())
        restored = Trainer(ds, cfg, 'cpu')
        restored.restore(checkpoint, metadata)
        resumed = restored.update()
        assert second['pairing_digest'] == resumed['pairing_digest']
        assert expected_model == state_hash(restored.policy.state_dict())
        assert expected_ema == state_hash(restored.ema.state_dict())
        with torch.no_grad():
            sample = restored.policy.predict_action(ds.observations[:1], torch.Generator().manual_seed(1234))
        assert sample.shape == (1, 16, action_dim) and torch.isfinite(sample).all()
        assert restored.policy.inference_scheduler.timesteps.tolist() == list(range(90, -1, -6))
        decoded = plugin.action_decode(sample[0].numpy(), raw[0])
        assert decoded.shape == (16, 4) and np.isfinite(decoded).all()
        diag = plugin.diagnostics(first_history, restored.policy)
        if candidate == 'P2':
            assert len(diag['phase_probabilities']) == 4 and np.isclose(sum(diag['phase_probabilities']), 1)
        checks = ['Fixed normalization identity', 'All163 D2 windows and valid target alignment', 'Zero invalid-tail targets with masks', 'Initial history repeat', 'Quaternion sign identity', 'Unclipped action decode with P3 future discard', 'One shared model below3x exact B0', 'CPU fullgraph eager-backend capture equals eager forward', 'Full128-batch finite gradients', 'Disposable checkpoint resume bitwise model/EMA/pairing equality', 'Inherited16-step DDIM shape/finiteness/timesteps']
        if candidate == 'P3':
            checks += ['Perfect epsilon reconstructs zero auxiliary targets/loss', 'Invalid-tail auxiliary and action mask invariance', 'High-noise geometry excluded', 'Future target next-state/replan-anchor alignment for all valid windows']
        report['candidates'][candidate] = {'passed': True, 'parameter_count': count, 'ratio_to_B0': count / baseline_count,
            'observation_features_per_row': obs_dim, 'diffusion_channels': action_dim,
            'disposable_cpu_optimizer_updates': 3, 'checks': checks}
        print(candidate, 'passed', count, 'parameters', flush=True)
        del trainer, restored, checkpoint, captured
        torch._dynamo.reset()
    report['passed'] = True
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    report['limitations'] = ['No CUDA execution or CUDA-graph compile checked in this CPU audit.', 'No trained behavior, training completion or evaluation outcome claimed.']
    path = root / 'round3/audits/pick-place-wall_implementation.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print('AUDIT_SAVED', path, flush=True)


if __name__ == '__main__':
    main()
