"""Read-only equivalence audit; no optimizer updates, simulator or API calls."""
from datetime import datetime, timezone
import numpy as np
import torch

from appl import baseline
from appl.io import ROOT, read, atomic, digest, object_hash, source_manifest
from appl.public import Factory, normalize_action, denormalize_action
from appl.dp_baseline import data as original_data, train as original_train
from appl.prior_policies import data, engine
from appl.scaleup.protocol import BASE, NAMES, naive_paths


def main():
    torch.set_num_threads(2)
    frozen = read(BASE / 'study_freeze.json')
    assert object_hash(source_manifest()) == frozen['study']['framework_source']
    reference_config = read(ROOT / 'experiments/exp2/configs/m0.json')['training']
    results = []
    for name in NAMES[1:]:
        source, checkpoint = naive_paths(name)
        stored = torch.load(checkpoint, map_location='cpu', weights_only=True)
        spec = stored['spec']
        config = spec['training']
        differences = {k: {'original': reference_config.get(k), 'new': config.get(k)}
                       for k in reference_config.keys() | config.keys()
                       if reference_config.get(k) != config.get(k)}
        assert set(differences) <= {'checkpoint_every', 'seeds'}, differences
        assert stored['step'] == config['updates'] == 60000
        assert digest(checkpoint) == frozen['study']['tasks'][name]['naive_checkpoint_sha256']
        assert all(group['lr'] == 0 for group in stored['optimizer']['param_groups'])
        entry = read(source.parent / 'assignment.json')
        episodes = data.episodes(entry)
        current = data.windows(episodes, config)
        original = original_data.windows(episodes, config)
        for field in ('raw_obs', 'native_action', 'mask'):
            assert np.array_equal(current[field], original[field]), (name, field)
        assert spec['normalizer'] == original_data.normalizer(episodes, config)
        assert spec['normalizer'] == data.shared_normalizer(entry)
        actions = torch.as_tensor(np.concatenate([e['action'] for e in episodes]))
        inverse_error = float((denormalize_action(normalize_action(actions, spec), spec) - actions).abs().max())
        assert inverse_error < 1e-6
        model = engine.module_at(source).build_model(spec).eval()
        model.load_state_dict(stored['ema'])
        reference = baseline.build_model(Factory(config), spec).eval()
        reference.load_state_dict({k.removeprefix('backbone.'): v for k, v in stored['ema'].items()})
        indices = [0, len(current['raw_obs']) // 2, len(current['raw_obs']) - 1]
        history = torch.as_tensor(current['raw_obs'][indices])
        actual = engine.sample(model, history, spec, torch.Generator().manual_seed(913))
        expected = original_train.sample(reference, baseline, history, spec, torch.Generator().manual_seed(913))
        sample_error = float((actual - expected).abs().max())
        assert torch.equal(actual, expected), (name, sample_error)
        assert config['observation_steps'] - 1 == 1 and config['execution_steps'] == 8
        request = read(checkpoint.parent / 'request.json')
        assert request['config'] == config and request['seed'] == 0
        training = read(checkpoint.parent / 'result.json')
        assert training['optimizer_steps'] == 60000 and training['sample_exposures'] == 60000 * 128
        results.append(dict(task=name, checkpoint_sha256=digest(checkpoint),
            training_updates=stored['step'], sample_exposures=training['sample_exposures'],
            non_scientific_config_differences=differences,
            full_trajectory_windows=len(current['raw_obs']),
            causal_histories_actions_masks_exactly_equal=True,
            normalizer_equal_to_original_full_trajectory_fitter=True,
            action_encode_decode_max_error=inverse_error,
            final_optimizer_learning_rate=stored['optimizer']['param_groups'][0]['lr'],
            full_DDPM100_samples_exactly_equal=True, sample_max_error=sample_error,
            sampling_inputs='First, middle and final training windows; CPU; identical final EMA and RNG.',
            execution_slice=[1, 9], passed=True))
        del stored, current, original, model, reference, episodes
    result = dict(reviewed_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        scientific_framework_unchanged=True, optimizer_updates=0, physical_steps=0, api_requests=0,
        checks=results,
        limits='Checks implementation equivalence and recorded configuration, not policy competence or causal explanations of failures. No additional test rollouts or tuning.')
    target = ROOT / 'experiments/exp2/analysis/naive_equivalence_audit.json'
    atomic(target, result)
    print(dict(passed=True, tasks=len(results), output=str(target)), flush=True)


if __name__ == '__main__':
    main()
