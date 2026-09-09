"""Inventory exact current-D_N training tensors without optimization or rollouts."""
import json
from pathlib import Path

import numpy as np
import torch

from relative_dp.utils import atomic_json, read_json, sha256
from round3.common import R3, ROOT, implementation_path, now
from round3.learning import data_for_run


def main():
    torch.set_num_threads(1)
    rows = []
    for run in read_json(R3 / 'run_manifest.json')['runs']:
        config_path = implementation_path(run['task'], run['candidate_id'])
        if not config_path.exists() or run.get('status') == 'invalid':
            continue
        config = read_json(config_path)
        ds = data_for_run(run, config)
        mask = ds.valid_mask.numpy().astype(bool)
        assert mask.shape == (len(ds), 16)
        assert np.isfinite(ds.actions.numpy()).all()
        labels = {}
        for key, tensor in ds.extra_targets.items():
            value = tensor.numpy()
            assert len(value) == len(ds) and np.isfinite(value).all()
            info = dict(shape=list(value.shape), dtype=str(value.dtype),
                        stored_scalar_elements=int(value.size))
            if value.ndim == 1 and np.issubdtype(value.dtype, np.integer):
                classes, counts = np.unique(value, return_counts=True)
                info['class_counts'] = {str(int(k)): int(v) for k, v in zip(classes, counts)}
                info['valid_label_rows'] = len(value)
            elif key == 'future_valid':
                info['eligible_horizon_rows'] = int(value.sum())
                info['eligible_rows_by_horizon_index'] = value.sum(axis=0).astype(int).tolist()
            elif value.ndim >= 3:
                custom = ds.extra_targets.get('future_valid')
                if custom is not None and tuple(custom.shape) == tuple(value.shape[:2]):
                    valid = custom.numpy().astype(bool)
                    info['mask_source'] = 'extra_targets.future_valid'
                elif tuple(value.shape[:2]) == mask.shape:
                    valid = mask
                    info['mask_source'] = 'action_valid_mask'
                else:
                    raise ValueError(f'Unaccounted auxiliary mask: {run["run_id"]} {key}')
                info['valid_horizon_rows'] = int(valid.sum())
                info['valid_scalar_elements'] = int(valid.sum() * np.prod(value.shape[2:]))
            labels[key] = info
        rows.append(dict(
            run_id=run['run_id'], task=run['task'], candidate_id=run['candidate_id'],
            train_n=run['train_n'], train_seed=run['train_seed'],
            formal_train_status_at_inventory=run['train_status'],
            implementation_sha256=sha256(config_path), proposal_sha256=config['proposal_sha256'],
            data_manifest_sha256=ds.manifest_hash,
            source_episode_ids=ds.normalizer.source_ids,
            complete_demonstrations=len(ds.normalizer.source_ids),
            transitions_and_distinct_window_starts=len(ds),
            observation_tensor_shape=list(ds.observations.shape),
            diffusion_target_shape=list(ds.actions.shape),
            native_executed_action_channels=4,
            joint_auxiliary_channels=int(ds.actions.shape[-1] - 4),
            valid_action_horizon_rows=int(mask.sum()),
            stored_action_horizon_rows=int(mask.size),
            auxiliary_targets=labels,
        ))
    result = dict(
        created_at=now(), passed=True, runs_inventoried=len(rows), runs=rows,
        meaning='Exact tensors reconstructed with frozen current-D_N data_for_run; inventory does not imply training completion.',
        accounting='One window starts at each actual transition. Overlapping future labels and repeated optimizer draws are not new demonstrations. Zero-valued labels are counted through explicit validity masks, not nonzero tests.',
        limitations='Counts describe materialized supervision, not stochastic per-update auxiliary-mask exposure or FLOPs. Definitions/units/weights remain in frozen proposals and plugins.',
        optimizer_updates_performed=0, scored_rollouts_performed=0,
        locked_test_contents_read=False,
    )
    path = R3 / 'reports' / 'dataset_accounting.json'
    atomic_json(path, result)
    print(json.dumps(dict(path=str(path.relative_to(ROOT)), runs_inventoried=len(rows), passed=True)))


if __name__ == '__main__':
    main()
