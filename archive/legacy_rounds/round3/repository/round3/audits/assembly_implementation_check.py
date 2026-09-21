"""Pretraining implementation checks; only assembly D2 and synthetic states.

Run: pixi run python round3/audits/assembly_implementation_check.py
No environment rollout, formal run or development/test score is produced.
"""
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from relative_dp.config import TRAIN_CONFIG
from relative_dp.dataset import WindowDataset
from relative_dp.model import state_hash
from relative_dp.utils import sha256
from round2.learning import Policy
from round3.assembly_plugin import AssemblyPlugin, transport_basis
from round3.learning import Normalizer, Trainer


def check():
    torch.set_num_threads(2)
    bundle = Path('round3/evidence/assembly')
    schema = json.loads((bundle/'task_materials.json').read_text())['observation_schema']
    episodes = []
    for index in (1, 2):
        with np.load(bundle/f'demo_{index}_trajectory.npz', allow_pickle=False) as values:
            episodes.append(dict(obs=values['obs'].copy(), actions=values['actions'].copy()))
    source_ids = ['allowed_assembly_D2_demo_1', 'allowed_assembly_D2_demo_2']
    source_hashes = {name: sha256(bundle/f'demo_{i+1}_trajectory.npz')
                     for i, name in enumerate(source_ids)}
    results = {}
    models = {}
    datasets = {}

    baseline_cfg = dict(TRAIN_CONFIG, clip_predicted_clean_actions=False,
                        torch_compile=False, inference_torch_compile=False)
    baseline = Policy(42, baseline_cfg)
    baseline_count = sum(p.numel() for p in baseline.parameters())
    del baseline

    for candidate in ('P1', 'P2', 'P3'):
        plugin = AssemblyPlugin('assembly', schema, dict(candidate_id=candidate))
        normalizer = Normalizer.fit(episodes, plugin, source_ids, source_hashes)
        assert np.array_equal(normalizer.mean, np.zeros_like(normalizer.mean))
        assert np.array_equal(normalizer.std, np.ones_like(normalizer.std))
        dataset = WindowDataset(episodes, normalizer, 16)
        raw_histories = np.stack([episodes[e]['obs'][[max(0, t-1), t]]
                                  for e, t in dataset.index])
        dataset.observations = torch.from_numpy(normalizer.normalize_history(raw_histories))
        raw_current = np.stack([episodes[e]['obs'][t] for e, t in dataset.index])
        native = dataset.actions.numpy().copy()
        encoded = plugin.action_encode(native, raw_current)
        decoded = plugin.action_decode(encoded, raw_current)
        error = float(np.max(np.abs(decoded-native)))
        assert error < 1e-6
        assert np.all(encoded[dataset.valid_mask.numpy() == 0] == 0)
        dataset.actions = torch.from_numpy(encoded)
        dataset.raw_current = torch.from_numpy(raw_current)
        dataset.extra_targets = {key: torch.as_tensor(value)
                                 for key, value in plugin.supervision(episodes, dataset.index).items()}
        dataset.plugin = plugin
        dataset.schema = schema
        cfg = dict(baseline_cfg, obs_dim=len(normalizer.mean), raw_dim=42,
                   task='assembly', candidate_id=candidate, custom_loss=candidate == 'P3',
                   batch_size=4, microbatch_size=4)
        trainer = Trainer(dataset, cfg, 'cpu')
        metric = trainer.update()
        assert np.isfinite(metric['loss']) and np.isfinite(metric['grad_norm'])
        parameter_count = sum(p.numel() for p in trainer.policy.parameters())
        assert parameter_count <= 3*baseline_count
        results[candidate] = dict(
            observation_shape=list(dataset.observations.shape),
            parameter_count=parameter_count, baseline_parameter_count=baseline_count,
            parameter_budget_passed=True, action_roundtrip_max_error=error,
            identity_fitted_normalizer=True, zero_tail_padding_masked=True,
            cpu_single_debug_update_finite=True)
        models[candidate] = trainer
        datasets[candidate] = dataset

    p2 = datasets['P2'].plugin
    artificial = episodes[0]['obs'][0].copy()
    artificial[39:42] = 0
    artificial[36:39] = [.03, .04, .1]
    basis = transport_basis(artificial)
    np.testing.assert_allclose(basis.T@basis, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(basis), 1, atol=1e-6)
    actions = np.array([[1, 1, .3, -.4], [-1, 1, -1, 1]], np.float32)
    encoded = p2.action_encode(actions, artificial)
    assert abs(encoded[0, 0]) > 1  # Native cube is not a local cube.
    np.testing.assert_allclose(p2.action_decode(encoded, artificial), actions, atol=1e-6)
    oversize = p2.action_decode(encoded*3, artificial)
    assert np.max(abs(oversize)) > 1  # Clipping belongs to shared world execution.
    for radius in (0., .01999, .02001):
        artificial[36:38] = [0., radius]
        b = transport_basis(artificial)
        np.testing.assert_allclose(b.T@b, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(np.linalg.det(b), 1, atol=1e-6)
    older = artificial.copy()
    older[39:42] = [.1, -.2, .03]
    older[36:39] = [.25, -.2, .1]
    history = np.stack([older, artificial])
    features = p2.observation_history(history)
    expected_old_hand = transport_basis(artificial).T@(older[:3]-artificial[39:42])/.1
    np.testing.assert_allclose(features[0, :3], expected_old_hand, atol=1e-6)
    assert not np.allclose(features[0], p2.observation(older))
    sign_history = history.copy()
    sign_history[:, 7:11] *= -1
    sign_history[:, 25:29] *= -1
    np.testing.assert_allclose(p2.observation_history(sign_history), features, atol=1e-6)
    results['P2'].update(current_anchor_shared_across_history=True,
                         basis_orthonormal_and_right_handed=True,
                         zero_and_threshold_basis_cases=True,
                         local_components_above_one_preserved=True,
                         world_decode_not_locally_clipped=True,
                         quaternion_double_cover_invariant=True)

    p3 = datasets['P3'].plugin
    starts = [list(p3.milestone_indices(e['obs'])) for e in episodes]
    assert starts == [[22, 45, 73], [25, 48, 78]]
    fake = np.repeat(episodes[0]['obs'][0:1], 6, axis=0)
    assert p3.milestone_indices(fake) == (6, 6, 6)
    last = len(episodes[0]['actions'])
    labels = p3.supervision(episodes, [(0, last-1), (0, last-4), (0, last-16)])
    np.testing.assert_array_equal(labels['future_valid'], [[0, 0], [1, 0], [1, 1]])
    future_changed = copy.deepcopy(episodes)
    future_changed[0]['obs'][4, 39] += .01
    original = p3.supervision(episodes, [(0, 0)])
    changed = p3.supervision(future_changed, [(0, 0)])
    assert not np.array_equal(original['future_relations'], changed['future_relations'])
    np.testing.assert_array_equal(p3.observation_history(episodes[0]['obs'][[0, 0]]),
                                  p3.observation_history(future_changed[0]['obs'][[0, 0]]))
    outputs = dict(event_logits=torch.zeros(2, 4), future_relations=torch.ones(2, 2, 6))
    batch = dict(targets=dict(event_class=torch.zeros(2, dtype=torch.long),
                             future_relations=torch.zeros(2, 2, 6),
                             future_valid=torch.zeros(2, 2)))
    loss, metrics = p3.extra_loss(outputs, batch, 0)
    assert metrics['future_relation_smooth_l1'].item() == 0
    np.testing.assert_allclose(loss.item(), .1*np.log(4)/2000, rtol=1e-6)
    loss_full, _ = p3.extra_loss(outputs, batch, 1999)
    np.testing.assert_allclose(loss_full.item(), loss.item()*2000, rtol=1e-6)
    results['P3'].update(milestone_starts=starts, missing_milestones_sentinel=True,
                         future_target_terminal_masks=True, no_future_runtime_input=True,
                         auxiliary_empty_mask_zero=True, auxiliary_weight_ramp_exact=True)

    # P3 resume includes every head and optimizer slot, then reproduces the next
    # sampled/noised update exactly. This is a CPU debug check, not formal training.
    trainer = models['P3']
    metadata = dict(audit='assembly_D2_cpu_debug')
    checkpoint = copy.deepcopy(trainer.checkpoint(metadata))
    restored = Trainer(datasets['P3'], trainer.config, 'cpu')
    restored.restore(checkpoint, metadata)
    left, right = trainer.update(), restored.update()
    assert left['pairing_digest'] == right['pairing_digest']
    assert state_hash(trainer.policy.state_dict()) == state_hash(restored.policy.state_dict())
    assert state_hash(trainer.ema.state_dict()) == state_hash(restored.ema.state_dict())
    results['P3']['cpu_checkpoint_next_update_bitwise_equal'] = True

    # Exercise the actual compiled-inference branch with a lightweight Dynamo
    # eager backend on CPU. CUDA/Inductor performance is not claimed here.
    real_compile = torch.compile
    for candidate, trainer in models.items():
        policy = trainer.ema.eval()
        history = datasets[candidate].observations[:1]
        eager = policy.predict_action(history, torch.Generator().manual_seed(700))
        assert eager.shape == (1, 16, 4) and torch.isfinite(eager).all()
        expected_steps = list(range(90, -1, -6))
        assert policy.inference_scheduler.timesteps.tolist() == expected_steps
        policy.config['inference_torch_compile'] = True
        with patch('torch.compile', side_effect=lambda module, **kwargs:
                   real_compile(module, backend='eager', dynamic=False)):
            policy.enable_inference_compilation()
        compiled = policy.predict_action(history, torch.Generator().manual_seed(700))
        torch.testing.assert_close(eager, compiled, rtol=1e-5, atol=1e-5)
        diagnostic = datasets[candidate].plugin.diagnostics(episodes[0]['obs'][[0, 0]], policy)
        assert diagnostic['execution_horizon'] == 4
        results[candidate].update(ddim16_leading_steps=expected_steps,
                                  eager_and_dynamo_eager_inference_agree=True,
                                  inference_finite_and_shape_correct=True)

    original = Path('round3/design_records/assembly/proposal.json')
    amendment = original.with_name('initial_feasibility_revision.json')
    freeze = json.loads(original.with_name('initial_freeze.json').read_text())
    assert freeze['proposal_sha256'] == sha256(original)
    for candidate in results:
        config = json.loads(Path(f'round3/configs/assembly/{candidate}.json').read_text())
        assert config['proposal_sha256'] == sha256(original)
        assert config['initial_feasibility_revision_sha256'] == sha256(amendment)
    report = dict(task='assembly', status='passed',
                  created_at=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                  scope='Frozen proposal plus sole feasibility amendment; assembly D2 and synthetic inputs only.',
                  formal_training_completed=False, evaluation_completed=False,
                  gpu_validation='Not exercised by this CPU audit; Dynamo eager backend validates inference routing, not CUDA performance.',
                  checks=results,
                  original_proposal_sha256=sha256(original),
                  initial_feasibility_revision_sha256=sha256(amendment),
                  source_sha256=sha256(Path('src/round3/assembly_plugin.py')),
                  audit_script_sha256=sha256(Path(__file__)))
    output = Path('round3/audits/assembly_implementation.json')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    check()
