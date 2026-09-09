"""One P4 correctness smoke: separate train/inference processes, no scored rollout."""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--infer', action='store_true')
    args = parser.parse_args()
    from round3.common import GPUS, R3, ROOT, now, run_record
    from relative_dp.utils import atomic_json, read_json, sha256
    if args.gpu not in GPUS:
        raise ValueError(f'GPU must be one of {GPUS}')
    identifier = f'r3_{args.task}_P4_n5_s0'
    rec = run_record(identifier)
    assert rec['feedback_revision'] and rec['train_status'] == 'pending'
    config = read_json(R3 / 'configs' / args.task / 'P4.json')
    readiness = config['implementation_audit']
    assert isinstance(readiness, str)
    assert read_json(ROOT / readiness).get('passed') is False
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(args.gpu)
    if not args.infer:
        from round3.cli import pixi_binary
        prefix = [pixi_binary(), 'run', 'env', f'CUDA_VISIBLE_DEVICES={args.gpu}',
                  f'MUJOCO_EGL_DEVICE_ID={args.gpu}', 'OPENBLAS_NUM_THREADS=1',
                  'OMP_NUM_THREADS=1', 'python']
        code = 'import sys; from round3.learning import train; train(sys.argv[1], debug_updates=20)'
        subprocess.run(prefix + ['-c', code, identifier], cwd=ROOT, check=True)
        subprocess.run(prefix + [str(Path(__file__).resolve()), '--task', args.task,
                                  '--gpu', str(args.gpu), '--infer'], cwd=ROOT, check=True)
        return
    import numpy as np
    import torch
    from round3.learning import Loaded, source_identity
    checkpoint = R3 / 'debug' / identifier / 'checkpoints' / 'step_000020.pt'
    loaded = Loaded(checkpoint)
    with np.load(R3 / 'evidence' / args.task / 'demo_1_trajectory.npz', allow_pickle=False) as trajectory:
        raw = trajectory['obs'][:2].copy()
    generator = torch.Generator(device='cuda').manual_seed(988)
    for _ in range(2):
        actions, _ = loaded.actions(raw, generator)
        assert actions.shape == (16, 4)
        assert np.isfinite(actions).all() and np.abs(actions).max() <= 1
    target = R3 / 'audits' / 'gpu_validation' / f'{identifier}.json'
    atomic_json(target, dict(time=now(), passed=True, task=args.task, candidate='P4',
        checkpoint_sha256=sha256(checkpoint), physical_gpu=args.gpu, debug_updates=20,
        compiled_inference_calls=2, formal_updates=0, scored_rollouts=0,
        implementation_sha256=sha256(R3 / 'configs' / args.task / 'P4.json'),
        source_hashes=source_identity(config),
        source='Two D2 states; no simulator, expert or dev/test inputs',
        validator_sha256=sha256(Path(__file__))))
    print(identifier, 'CUDA update and compiled inference passed', flush=True)


if __name__ == '__main__':
    main()
