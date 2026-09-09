"""CPU mechanical audit using only the isolated D2 trajectories.

Run: pixi run env CUDA_VISIBLE_DEVICES='' python this_file.py
No task success evaluation or candidate selection is performed.
"""
from pathlib import Path
import gc
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import torch

from relative_dp.config import TRAIN_CONFIG
from relative_dp.dataset import WindowDataset
from relative_dp.model import masked_epsilon_loss, state_hash
from relative_dp.train import atomic_torch_save, runtime_setup
from round2.learning import Policy
from round3.learning import Normalizer, Trainer
from round3.stick_plugin import StickPlugin, goal_frame, relation_features


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / 'round3/evidence/stick-push'
DIRECTORY = ROOT / 'round3/design_records/stick-push'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nested_equal(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(nested_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def make_dataset(episodes, plugin):
    paths = [EVIDENCE / f'demo_{i}_trajectory.npz' for i in (1, 2)]
    normalizer = Normalizer.fit(episodes, plugin, [p.name for p in paths],
                                {p.name: sha(p) for p in paths})
    assert np.array_equal(normalizer.mean, np.zeros_like(normalizer.mean))
    assert np.array_equal(normalizer.std, np.ones_like(normalizer.std))
    ds = WindowDataset(episodes, normalizer, 16)
    raw = np.stack([episodes[e]['obs'][t] for e, t in ds.index])
    histories = np.stack([episodes[e]['obs'][[max(t-1, 0), t]] for e, t in ds.index])
    ds.observations = torch.from_numpy(normalizer.normalize_history(histories))
    ds.actions = torch.from_numpy(plugin.action_encode(ds.actions.numpy(), raw))
    ds.raw_current = torch.from_numpy(raw)
    ds.extra_targets = {k: torch.as_tensor(v) for k, v in plugin.supervision(episodes, ds.index).items()}
    ds.actions = torch.from_numpy(plugin.training_actions(
        ds.actions.numpy(), {k: v.numpy() for k, v in ds.extra_targets.items()}))
    ds.plugin = plugin
    assert len(ds) == 148
    assert torch.isfinite(ds.observations).all() and torch.isfinite(ds.actions).all()
    assert torch.count_nonzero(ds.actions[ds.valid_mask == 0]) == 0
    return ds, histories


def main():
    runtime_setup()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not torch.cuda.is_available(), 'Audit must not touch formal GPU processes'
    episodes = []
    for i in (1, 2):
        with np.load(EVIDENCE / f'demo_{i}_trajectory.npz', allow_pickle=False) as data:
            episodes.append({'obs': data['obs'].copy(), 'actions': data['actions'].copy()})
    materials = json.loads((EVIDENCE / 'task_materials.json').read_text())
    schema = materials['observation_schema']
    base = Policy(39, dict(TRAIN_CONFIG, clip_predicted_clean_actions=False))
    baseline_count = sum(p.numel() for p in base.parameters())
    del base
    results = {}
    rng = np.random.default_rng(518)
    for cid in ('P1', 'P2', 'P3'):
        print('Auditing', cid, flush=True)
        plugin = StickPlugin('stick-push', schema, {'candidate_id': cid})
        ds, histories = make_dataset(episodes, plugin)
        cfg = dict(TRAIN_CONFIG, obs_dim=ds.observations.shape[-1], raw_dim=39,
                   clip_predicted_clean_actions=False, thresholding=False,
                   torch_compile=False, inference_torch_compile=False,
                   custom_loss=cid == 'P3', candidate_id=cid,
                   task='stick-push', debug=True)
        cfg.update(plugin.model_config(cfg))
        assert cfg['batch_size'] == cfg['microbatch_size'] == 128
        assert cfg['train_updates'] == 20000
        assert cfg['action_dim'] == ds.actions.shape[-1]
        raw = ds.raw_current.numpy()
        actions = rng.uniform(-1.8, 1.8, (len(raw), 16, 4)).astype(np.float32)
        encoded = plugin.action_encode(actions, raw)
        decoded = plugin.action_decode(encoded, raw)
        roundtrip_error = float(np.max(np.abs(actions-decoded)))
        assert roundtrip_error < 1e-6
        assert np.any(np.abs(decoded) > 1), 'Plugin must defer world clipping to Loaded'
        assert np.all(np.abs(np.clip(decoded, -1, 1)) <= 1)
        values = dict(observation_dim=cfg['obs_dim'], action_dim=cfg['action_dim'],
                      action_roundtrip_max_abs=roundtrip_error,
                      fixed_passthrough_all=True, zero_padded_masked_tails=True,
                      uniform_windows=148, runtime_inputs='two numerical raw39 observations',
                      clipping='Shared Loaded clips native world cube after inverse decode')
        # Canonical xyzw, including absent zeros, applies to every candidate.
        flipped = raw.copy()
        flipped[:, 7:11] *= -1
        flipped[:, 25:29] *= -1
        np.testing.assert_allclose(plugin.observation(raw), plugin.observation(flipped), atol=1e-6)
        assert np.all(plugin.observation(raw)[:, 14:18] == 0)
        if cid == 'P2':
            origin, rotation = goal_frame(raw)
            np.testing.assert_allclose(np.swapaxes(rotation, -1, -2) @ rotation,
                                       np.broadcast_to(np.eye(3), rotation.shape), atol=2e-7)
            transformed = plugin.observation_history(histories)
            for i in (0, 32, 74, 75, 106, 147):
                # Independent scalar projection reference, not a call to relation_features.
                expected = ((histories[i, 0, :3]-origin[i]) @ rotation[i]) / .1
                np.testing.assert_allclose(transformed[i, 0, :3], expected, atol=2e-6)
                expected_latest = plugin.observation(raw[i])
                np.testing.assert_allclose(transformed[i, 1], expected_latest, atol=2e-6)
            degenerate = raw[:1].copy()
            degenerate[:, 36:38] = degenerate[:, 11:13]
            _, fallback = goal_frame(degenerate)
            np.testing.assert_array_equal(fallback[0], np.eye(3, dtype=np.float32))
            assert np.isfinite(plugin.observation(degenerate)).all()
            values.update(current_anchor_both_history_rows=True,
                          orthonormal_gravity_preserving_frame=True,
                          zero_goal_distance_fallback=True)
        if cid == 'P3':
            labels = ds.extra_targets['future_motion'].numpy()
            for i, (e, t) in enumerate(ds.index):
                valid = min(16, len(episodes[e]['actions'])-t)
                for j in range(valid):
                    obs = episodes[e]['obs']
                    expected = np.concatenate([obs[t+j+1, s]-obs[t, s]
                                               for s in (slice(0,3), slice(4,7), slice(11,14))]) / .1
                    np.testing.assert_array_equal(labels[i, j], expected)
                assert np.all(labels[i, valid:] == 0)
            # Explicit unequal channel errors prove independent mean denominators.
            prediction = torch.zeros((2,16,13))
            noise = torch.ones_like(prediction)
            prediction[..., :4] = 3.  # action squared error4
            prediction[..., 4:] = 4.  # motion squared error9
            mask = torch.zeros((2,16)); mask[0,:3] = 1; mask[1,:7] = 1
            prediction[mask == 0] = 1000.
            base_loss = masked_epsilon_loss(prediction[..., :4], noise[..., :4], mask)
            extra, _ = plugin.extra_loss(prediction, {'noise':noise, 'mask':mask}, 0)
            assert float(base_loss) == 4.
            assert abs(float(extra)-.15*9.) < 1e-6
            values.update(all_current_D2_future_labels_exact=True,
                          anchored_to_replan_not_previous_horizon=True,
                          terminal_observation_used_for_last_valid_action=True,
                          action_epsilon_weight=1., motion_epsilon_weight=.15,
                          separate_action4_motion9_denominators=True)
        trainer = Trainer(ds, cfg, 'cpu')
        parameters = sum(p.numel() for p in trainer.policy.parameters())
        assert parameters <= 3*baseline_count
        before = state_hash(trainer.policy.state_dict())
        first = trainer.update()
        assert trainer.step == 1
        assert state_hash(trainer.policy.state_dict()) != before
        assert np.isfinite(first['loss']) and np.isfinite(first['grad_norm'])
        values.update(parameter_count=parameters, baseline_parameter_count=baseline_count,
                      parameter_ratio=parameters/baseline_count, full_batch128_update=True)
        trainer.policy.eval()
        with torch.no_grad():
            sampled = trainer.policy.predict_action(ds.observations[:1], torch.Generator().manual_seed(615))
        assert sampled.shape == (1,16,4) and torch.isfinite(sampled).all()
        assert trainer.policy.inference_scheduler.timesteps.tolist() == list(range(90,-1,-6))
        assert not trainer.policy.inference_scheduler.config.clip_sample
        values.update(ddim16_timesteps=list(range(90,-1,-6)), ddim_eta=0., clean_clip=False,
                      finite_inference_shape=[1,16,4], generated_motion_not_runtime_input=True)
        if cid in ('P2', 'P3'):
            # Actual file serialization and continuation exercise optimizer + RNG + EMA.
            with tempfile.TemporaryDirectory(prefix='stick-audit-') as temporary:
                checkpoint = Path(temporary) / 'one_update.pt'
                metadata = {'audit': 'stick-implementation-only', 'candidate_id': cid}
                atomic_torch_save(checkpoint, trainer.checkpoint(metadata))
                values['serialized_checkpoint_sha256'] = sha(checkpoint)
                next_metric = trainer.update()
                continued_model = state_hash(trainer.policy.state_dict())
                continued_ema = state_hash(trainer.ema.state_dict())
                continued_optimizer = trainer.optimizer.state_dict()
                restored = Trainer(ds, cfg, 'cpu')
                restored.restore(torch.load(checkpoint, map_location='cpu', weights_only=False), metadata)
                restored_metric = restored.update()
                assert state_hash(restored.policy.state_dict()) == continued_model
                assert state_hash(restored.ema.state_dict()) == continued_ema
                assert nested_equal(restored.optimizer.state_dict(), continued_optimizer)
                assert torch.equal(restored.loader_rng.get_state(), trainer.loader_rng.get_state())
                assert torch.equal(restored.diffusion_rng.get_state(), trainer.diffusion_rng.get_state())
                assert restored_metric['pairing_digest'] == next_metric['pairing_digest']
                assert restored_metric['loss'] == next_metric['loss']
                values.update(serialized_restored_next_update_exact=True,
                              resumed_model_ema_optimizer_rng_pairing_exact=True,
                              transient_checkpoint_removed_after_audit=True)
                del restored, continued_optimizer
        results[cid] = values
        del trainer, ds
        gc.collect()
    report = dict(task='stick-push', passed=True, status='passed',
                  audited_at=datetime.now(timezone.utc).isoformat(),
                  audit_device='cpu; CUDA_VISIBLE_DEVICES empty inside Pixi command',
                  proposal_sha256=sha(DIRECTORY / 'proposal.json'),
                  plugin_sha256=sha(ROOT / 'src/round3/stick_plugin.py'),
                  audit_script_sha256=sha(__file__),
                  evidence_sha256={f.name:sha(f) for f in EVIDENCE.glob('*trajectory.npz')},
                  source_scope='Shared implementation interfaces and own plugin; exactly D2 evidence trajectories only',
                  source_change_reason='Initial implementation of original saved proposal; no feasibility revision',
                  implementation_mapping='P1 world relation features; P2 frozen current goal frame; P3 relation features plus joint13 diffusion',
                  no_formal_training_or_evaluation_claimed=True,
                  no_learned_performance_used_for_design=True,
                  candidates=results)
    (DIRECTORY / 'implementation_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
