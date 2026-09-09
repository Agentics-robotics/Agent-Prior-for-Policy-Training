"""D2-only CPU implementation checks; these are not formal training results."""
import gc
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Pixi activation may set its historical CUDA selection; this audit is CPU-only.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

from relative_dp.config import TRAIN_CONFIG
from relative_dp.dataset import WindowDataset
from relative_dp.model import state_hash
from relative_dp.utils import object_hash
from round2.learning import Policy
from round3.drawer_plugin import DrawerPlugin
from round3.learning import Normalizer, Trainer


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / 'round3/evidence/drawer'
AUDITS = ROOT / 'round3/audits/drawer'
CONFIGS = ROOT / 'round3/configs/drawer'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal_tree(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def main():
    if torch.cuda.is_available():
        raise RuntimeError('Run this audit with CUDA_VISIBLE_DEVICES empty')
    AUDITS.mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(parents=True, exist_ok=True)
    paths = sorted(EVIDENCE.glob('*trajectory.npz'))
    assert len(paths) == 2
    episodes = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            episodes.append(dict(obs=data['obs'].copy(), actions=data['actions'].copy()))
    materials = json.loads((EVIDENCE / 'task_materials.json').read_text())
    schema = materials['observation_schema']
    evidence_hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    raw = np.stack([ep['obs'][t] for ep in episodes for t in range(len(ep['actions']))])
    baseline = Policy(41, dict(TRAIN_CONFIG, clip_predicted_clean_actions=False, thresholding=False))
    baseline_count = sum(p.numel() for p in baseline.parameters())
    del baseline

    for candidate in ('P1', 'P2', 'P3'):
        plugin = DrawerPlugin('drawer', schema, {'candidate_id': candidate})
        norm = Normalizer.fit(episodes, plugin, [p.name for p in paths], evidence_hashes)
        assert np.array_equal(norm.mean, np.zeros(plugin.observation_dim))
        assert np.array_equal(norm.std, np.ones(plugin.observation_dim))
        ds = WindowDataset(episodes, norm, 16)
        histories = np.stack([episodes[e]['obs'][[max(0, t - 1), t]] for e, t in ds.index])
        encoded_history = norm.normalize_history(histories)
        assert encoded_history.shape == (178, 2, plugin.observation_dim)
        assert np.isfinite(encoded_history).all()
        # Independent matrix expression for both members' shared current anchor.
        if candidate != 'P2':
            for row, (e, t) in enumerate(ds.index):
                anchor = episodes[e]['obs'][t]
                s, c = anchor[39:41].astype(np.float64)
                length = np.hypot(s, c)
                s, c = s / length, c / length
                rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                for history_index in (0, 1):
                    member = histories[row, history_index]
                    expected = (member[:3] - anchor[4:7]) @ rotation / .2
                    np.testing.assert_allclose(encoded_history[row, history_index, :3], expected, atol=2e-7)
                np.testing.assert_array_equal(encoded_history[row, 1, 3:6], np.zeros(3))
        ds.observations = torch.from_numpy(encoded_history)
        native_actions = ds.actions.numpy().copy()
        encoded = plugin.action_encode(native_actions, raw)
        roundtrip = plugin.action_decode(encoded, raw)
        roundtrip_error = float(np.max(np.abs(roundtrip - native_actions)))
        assert roundtrip_error < 3e-7
        if candidate != 'P2':
            assert np.max(np.abs(encoded[..., :3])) > 1.01
        ds.actions = torch.from_numpy(encoded)
        ds.raw_current = torch.from_numpy(raw)
        targets = plugin.supervision(episodes, ds.index)
        ds.extra_targets = {k: torch.from_numpy(v) for k, v in targets.items()}
        packed = plugin.training_actions(encoded, targets)
        assert packed.shape == (178, 16, plugin.action_dim)
        assert np.all(packed[ds.valid_mask.numpy() == 0] == 0)
        np.testing.assert_array_equal(packed[..., :4], encoded)
        if candidate != 'P1':
            for row, (e, t) in enumerate(ds.index):
                anchor = episodes[e]['obs'][t]
                s, c = anchor[39:41].astype(np.float64)
                length = np.hypot(s, c)
                s, c = s / length, c / length
                rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                for k in range(min(16, len(episodes[e]['actions']) - t)):
                    future = episodes[e]['obs'][t + k + 1]
                    rail = np.dot(anchor[36:39] - future[4:7], rotation[:, 1]) / .16
                    np.testing.assert_allclose(packed[row, k, -1], rail, atol=2e-7)
                    if candidate == 'P3':
                        gap = (future[:3] - future[4:7]) @ rotation / .2
                        np.testing.assert_allclose(packed[row, k, 4:7], gap, atol=2e-7)
            # Auxiliary averaging must remain independent of action dimension,
            # and masked padding must not influence the loss.
            output = torch.zeros_like(torch.from_numpy(packed))
            output[..., 4:] = 1
            mask = ds.valid_mask
            output[mask == 0] = 10000
            extra, _ = plugin.extra_loss(output, {'noise': torch.zeros_like(output), 'mask': mask}, 0)
            assert extra.item() == .25
        ds.actions = torch.from_numpy(packed)
        ds.plugin = plugin
        ds.schema = schema
        ds.manifest_hash = object_hash(evidence_hashes)
        cfg = dict(TRAIN_CONFIG, task='drawer', candidate_id=candidate, train_n=2,
                   obs_dim=plugin.observation_dim, raw_dim=41, torch_compile=False,
                   inference_torch_compile=False, clip_predicted_clean_actions=False,
                   thresholding=False, custom_loss=candidate != 'P1', debug=True,
                   observation_schema=schema)
        cfg.update(plugin.model_config(cfg))
        assert cfg['batch_size'] == cfg['microbatch_size'] == 128
        trainer = Trainer(ds, cfg, 'cpu')
        count = sum(p.numel() for p in trainer.policy.parameters())
        assert count <= 3 * baseline_count
        initial_hash = trainer.initial_weights_hash
        # Sampling checks use the initial random policy, not a trained outcome.
        prediction = trainer.policy.predict_action(ds.observations[:1], torch.Generator().manual_seed(901))
        assert prediction.shape == (1, 16, plugin.action_dim)
        assert torch.isfinite(prediction).all()
        assert trainer.policy.inference_scheduler.timesteps.tolist() == list(range(90, -1, -6))
        assert trainer.policy.inference_scheduler.config.clip_sample is False
        assert plugin.action_decode(prediction[0].numpy(), raw[0]).shape == (16, 4)
        trainer.update()
        assert trainer.step == 1
        assert state_hash(trainer.policy.state_dict()) != initial_hash
        metadata = {'audit': 'drawer-D2-CPU', 'candidate_id': candidate, 'debug': True}
        serialized = io.BytesIO()
        torch.save(trainer.checkpoint(metadata), serialized)
        checkpoint_sha = hashlib.sha256(serialized.getbuffer()).hexdigest()
        serialized.seek(0)
        checkpoint = torch.load(serialized, map_location='cpu', weights_only=False)
        resumed = Trainer(ds, cfg, 'cpu')
        resumed.restore(checkpoint, metadata)
        assert equal_tree(trainer.policy.state_dict(), resumed.policy.state_dict())
        assert equal_tree(trainer.ema.state_dict(), resumed.ema.state_dict())
        assert equal_tree(trainer.optimizer.state_dict(), resumed.optimizer.state_dict())
        assert torch.equal(trainer.loader_rng.get_state(), resumed.loader_rng.get_state())
        assert torch.equal(trainer.diffusion_rng.get_state(), resumed.diffusion_rng.get_state())
        trainer.update()
        resumed.update()
        assert trainer.step == resumed.step == 2
        assert equal_tree(trainer.policy.state_dict(), resumed.policy.state_dict())
        assert equal_tree(trainer.ema.state_dict(), resumed.ema.state_dict())
        assert equal_tree(trainer.optimizer.state_dict(), resumed.optimizer.state_dict())
        assert trainer.pairing_digest == resumed.pairing_digest
        audit_path = AUDITS / f'{candidate}_implementation.json'
        audit = dict(
            task='drawer', candidate_id=candidate, passed=True, status='passed',
            created_at=datetime.now(timezone.utc).isoformat(), device='cpu',
            data_scope='Exactly the two drawer evidence NPZs; no larger data or development/test states.',
            proposal_sha256=sha(ROOT / 'round3/design_records/drawer/proposal.json'),
            source_sha256=sha(ROOT / 'src/round3/drawer_plugin.py'),
            audit_script_sha256=sha(Path(__file__)), evidence_hashes=evidence_hashes,
            observation_dimension=plugin.observation_dim, diffusion_channels=plugin.action_dim,
            windows_checked=178, frame_and_future_alignment_passed=True,
            fixed_scales_and_tail_masks_passed=True, action_roundtrip_max_error=roundtrip_error,
            world_clipping_owner='Shared round3.learning.Loaded.actions, after decode',
            auxiliary_loss_scale_passed=True, ddim16_joint_channel_shape_passed=True,
            parameter_count=count, baseline_parameter_count=baseline_count,
            parameter_ratio=count / baseline_count, batch_size=128,
            debug_updates_before_checkpoint=1, next_update_exact_after_restore=True,
            model_ema_optimizer_rng_restore_passed=True,
            serialized_debug_checkpoint_sha256=checkpoint_sha,
            formal_training_complete=False, model_scores_used_for_design=False,
            feasibility_revision_required=False)
        audit_path.write_text(json.dumps(audit, indent=2) + '\n')
        config = dict(
            candidate_id=candidate, plugin_module='round3.drawer_plugin', plugin_class='DrawerPlugin',
            plugin_config={'candidate_id': candidate}, proposal_sha256=audit['proposal_sha256'],
            observation_schema_sha256=object_hash(schema),
            rationale='Implements frozen drawer proposal version1; fixed scales and joint generated future channels as specified.',
            implementation_audit=str(audit_path.relative_to(ROOT)), custom_loss=candidate != 'P1')
        (CONFIGS / f'{candidate}.json').write_text(json.dumps(config, indent=2) + '\n')
        print(json.dumps({k: audit[k] for k in ('candidate_id', 'passed', 'parameter_count',
              'parameter_ratio', 'action_roundtrip_max_error', 'next_update_exact_after_restore')}), flush=True)
        del trainer, resumed, checkpoint, serialized, ds, prediction
        gc.collect()


if __name__ == '__main__':
    main()
