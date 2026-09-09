"""CPU-only mathematical checks for the immutable peg P4 representation."""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from relative_dp.config import TRAIN_CONFIG
from relative_dp.dataset import WindowDataset
from relative_dp.model import masked_epsilon_loss
from round2.learning import Policy
from round3.learning import Normalizer, source_identity
from round3.peg_plugin import PegPlugin
from round3.plugins import load_plugin


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    task = 'peg-insert-side'
    revision_path = Path('round3/design_records/peg-insert-side/revision.json')
    config_path = Path('round3/configs/peg-insert-side/P4.json')
    implementation = json.loads(config_path.read_text())
    revision = json.loads(revision_path.read_text())
    schema = json.loads(Path('round3/evidence/peg-insert-side/task_materials.json').read_text())['observation_schema']
    assert implementation['proposal_sha256'] == digest(revision_path)
    assert revision['decision'] == 'P4'
    assert len(revision['candidate']['all_changes_from_base']) == 2
    plugin = load_plugin(task, schema, implementation)
    base = PegPlugin(task, schema, {'candidate_id': 'P1'})
    rng = np.random.default_rng(60704)
    raw = rng.normal(size=(7, 2, 42)).astype(np.float32)
    # Explicit absent object2 schema; all other test values are synthetic.
    raw[..., 11:18] = 0
    raw[..., 29:36] = 0
    original = raw.copy()
    encoded = plugin.observation(raw)
    p1_encoded = base.observation(raw)
    assert encoded.shape == (7, 2, 62)
    assert encoded.dtype == np.float32
    np.testing.assert_array_equal(raw, original)
    assert not np.shares_memory(encoded, raw)
    changed = [*range(4, 7), *range(22, 25), *range(36, 42)]
    unchanged = [i for i in range(62) if i not in changed]
    np.testing.assert_array_equal(encoded[..., unchanged], p1_encoded[..., unchanged])
    for output, source, anchor in ((slice(4, 7), slice(4, 7), slice(0, 3)),
                                   (slice(22, 25), slice(22, 25), slice(18, 21)),
                                   (slice(36, 39), slice(36, 39), slice(0, 3)),
                                   (slice(39, 42), slice(39, 42), slice(0, 3))):
        np.testing.assert_array_equal(encoded[..., output], (raw[..., source] - raw[..., anchor]) / .1)
    np.testing.assert_array_equal(plugin.observation(raw[0, 0]), encoded[0, 0])
    assert plugin.passthrough(62) == list(range(62))
    try:
        plugin.observation(np.zeros(41))
        raise AssertionError('Dimension validation failed')
    except ValueError:
        pass
    try:
        type(plugin)(task, schema, {'candidate_id': 'P1'})
        raise AssertionError('Candidate validation failed')
    except ValueError:
        pass

    # A coordinate identity audit, not training augmentation or a physics claim.
    shift = np.asarray([.031, -.027, .012], np.float32)
    shifted = raw.copy()
    for start in (0, 4, 18, 22, 36, 39):
        shifted[..., start:start + 3] += shift
    shifted_encoded = plugin.observation(shifted)
    np.testing.assert_allclose(shifted_encoded[..., changed], encoded[..., changed], atol=5e-6)
    np.testing.assert_allclose(shifted_encoded[..., 42:], encoded[..., 42:], atol=5e-5)
    np.testing.assert_allclose(shifted_encoded[..., :3], encoded[..., :3] + shift)
    np.testing.assert_allclose(shifted_encoded[..., 18:21], encoded[..., 18:21] + shift)

    actions = rng.uniform(-1.7, 1.7, (7, 16, 4)).astype(np.float32)
    action_copy = actions.copy()
    encoded_actions = plugin.action_encode(actions, raw[:, -1])
    decoded = plugin.action_decode(encoded_actions, raw[:, -1])
    np.testing.assert_array_equal(decoded, actions)
    np.testing.assert_array_equal(actions, action_copy)
    assert np.max(np.abs(decoded)) > 1  # shared evaluation clips after this codec
    assert plugin.supervision(None, None) == {}
    np.testing.assert_array_equal(plugin.training_actions(encoded_actions, {}), actions)
    assert plugin.extra_loss(None, None, 0) == (None, {})
    assert plugin.model_config({}) == {'action_dim': 4}
    diagnostics = plugin.diagnostics(raw[0], None)
    assert diagnostics['candidate_id'] == 'P4'
    assert diagnostics['execution_horizon'] == 4
    assert diagnostics['sampled_channels'] == 4

    episodes = []
    for length in (3, 19):
        obs = rng.normal(0, .1, (length + 1, 42)).astype(np.float32)
        obs[:, 11:18] = 0
        obs[:, 29:36] = 0
        episodes.append({'obs': obs, 'actions': rng.uniform(-1, 1, (length, 4)).astype(np.float32)})
    normalizer = Normalizer.fit(episodes, plugin, ['synthetic-0', 'synthetic-1'], {})
    np.testing.assert_array_equal(normalizer.mean, np.zeros(62))
    np.testing.assert_array_equal(normalizer.std, np.ones(62))
    dataset = WindowDataset(episodes, normalizer, 16)
    assert len(dataset) == 22
    for row, (episode, start) in enumerate(dataset.index):
        valid = min(16, len(episodes[episode]['actions']) - start)
        assert dataset.valid_mask[row].sum().item() == valid
        assert torch.count_nonzero(dataset.actions[row, valid:]).item() == 0
        expected_history = episodes[episode]['obs'][[max(0, start - 1), start]]
        np.testing.assert_array_equal(dataset.observations[row].numpy(), plugin.observation(expected_history))
    mask = torch.zeros(2, 16)
    mask[0, :3] = 1
    mask[1, :11] = 1
    prediction = torch.zeros(2, 16, 4, requires_grad=True)
    noise = torch.ones(2, 16, 4)
    loss = masked_epsilon_loss(prediction, noise, mask)
    assert loss.item() == 1
    loss.backward()
    assert torch.count_nonzero(prediction.grad[mask == 0]).item() == 0
    corrupted = prediction.detach().clone()
    corrupted[mask == 0] = 1234
    assert masked_epsilon_loss(corrupted, noise, mask).item() == 1

    cfg = dict(TRAIN_CONFIG, obs_dim=62, clip_predicted_clean_actions=False,
               thresholding=False, torch_compile=False, inference_torch_compile=False)
    assert (cfg['n_obs_steps'], cfg['prediction_horizon'], cfg['action_dim'], cfg['batch_size'],
            cfg['train_updates'], cfg['diffusion_train_timesteps'], cfg['inference_steps']) == (2, 16, 4, 128, 20000, 100, 16)
    torch.manual_seed(cfg['model_init_seed'])
    policy = plugin.build_policy(cfg)
    p4_parameters = sum(p.numel() for p in policy.parameters())
    p1_policy = base.build_policy(cfg)
    p1_parameters = sum(p.numel() for p in p1_policy.parameters())
    b0 = Policy(42, cfg)
    b0_parameters = sum(p.numel() for p in b0.parameters())
    assert p4_parameters == p1_parameters == 4861508
    assert b0_parameters == 4718148
    assert p4_parameters <= 3 * b0_parameters
    assert all(p.device.type == 'cpu' for p in policy.parameters())
    assert policy.train_scheduler.config.clip_sample is False
    assert policy.inference_scheduler.config.clip_sample is False
    del p1_policy, b0
    observation, target_actions, valid_mask, _ = dataset.sample(128, torch.Generator().manual_seed(0))
    noisy, timestep, target_noise = policy.noise_targets(target_actions, torch.Generator().manual_seed(0))
    epsilon = policy(noisy, timestep, observation)
    assert epsilon.shape == (128, 16, 4)
    training_loss = masked_epsilon_loss(epsilon, target_noise, valid_mask)
    training_loss.backward()
    assert torch.isfinite(training_loss)
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
    sampled = policy.predict_action(observation[:1], torch.Generator().manual_seed(31))
    assert sampled.shape == (1, 16, 4) and torch.isfinite(sampled).all()
    assert policy.inference_scheduler.timesteps.tolist() == [90, 84, 78, 72, 66, 60, 54, 48, 42, 36, 30, 24, 18, 12, 6, 0]
    assert not torch.cuda.is_initialized()

    sources = source_identity(implementation)
    assert 'src/round3/peg_revision.py' in sources
    assert 'src/round3/peg_plugin.py' in sources
    readiness = json.loads(Path(implementation['implementation_audit']).read_text())
    assert readiness['passed'] is False
    result = {
        'task': task, 'candidate_id': 'P4', 'passed': True,
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'CPU mathematical implementation audit only; all test arrays synthetic; no optimizer updates or simulator rollouts',
        'proposal_sha256': digest(revision_path), 'implementation_config_sha256': digest(config_path),
        'audit_script_sha256': digest(__file__), 'source_hashes': sources,
        'checks': ['proposal/config binding', 'exact four block replacement', 'raw nonmutation',
                   'P1 appended and all untouched features exactly retained', 'history/current-previous anchoring',
                   'relative coordinate identity with retained world hand context', 'identity unclipped world action codec',
                   'no extra labels/loss/channels', 'fixed full passthrough normalization',
                   'tail masks and episode boundary padding', 'padding loss and gradient exclusion',
                   'exact B0 and P1 parameter comparison', '128-chunk finite epsilon forward/backward',
                   'CPU DDIM16 sample with no clean clipping', 'source identity includes imported original plugin',
                   'final readiness remains pending CUDA'],
        'parameter_count': p4_parameters, 'baseline_parameters': b0_parameters,
        'parameter_ratio': p4_parameters / b0_parameters,
        'cpu_probe_loss': training_loss.item(), 'optimizer_updates': 0,
        'torch_threads': torch.get_num_threads(), 'cuda_initialized': torch.cuda.is_initialized(),
        'formal_readiness': False, 'cuda_debug': 'pending parent execution',
    }
    path = Path('round3/audits/peg_revision_cpu.json')
    with path.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('passed', 'parameter_count', 'baseline_parameters', 'parameter_ratio', 'cpu_probe_loss', 'formal_readiness')}))


if __name__ == '__main__':
    main()
