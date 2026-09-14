"""Short correctness checks, never scored rollouts or formal warm starts."""
import argparse
import os
import subprocess
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--candidate',choices=['P1','P2','P3'])
    parser.add_argument('--infer',action='store_true')
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    os.environ['MUJOCO_EGL_DEVICE_ID']=str(args.gpu)
    from round3.common import R3,ROOT,now
    from round3.cli import pixi_binary
    from relative_dp.utils import atomic_json,read_json,sha256
    if not args.infer:
        rows=[]
        for candidate in ('P1','P2','P3'):
            identifier=f'r3_{args.task}_{candidate}_n5_s0'
            prefix=[pixi_binary(),'run','env',f'CUDA_VISIBLE_DEVICES={args.gpu}',f'MUJOCO_EGL_DEVICE_ID={args.gpu}','python']
            code='import sys; from round3.learning import train; train(sys.argv[1],debug_updates=20)'
            subprocess.run(prefix+['-c',code,identifier],cwd=ROOT,check=True)
            subprocess.run(prefix+[str(Path(__file__).resolve()),'--task',args.task,'--gpu',str(args.gpu),'--candidate',candidate,'--infer'],cwd=ROOT,check=True)
            rows.append(read_json(R3/'audits'/'gpu_validation'/f'{identifier}.json'))
        atomic_json(R3/'audits'/'gpu_validation'/f'{args.task}_complete.json',dict(time=now(),passed=True,rows=rows,formal_updates=0,debug_updates=60))
        return
    import torch
    import numpy as np
    from round3.learning import Loaded
    identifier=f'r3_{args.task}_{args.candidate}_n5_s0'
    checkpoint=R3/'debug'/identifier/'checkpoints'/'step_000020.pt'
    loaded=Loaded(checkpoint)
    raw=np.load(R3/'evidence'/args.task/'demo_1_trajectory.npz',allow_pickle=False)['obs'][:2]
    generator=torch.Generator(device='cuda').manual_seed(988)
    for _ in range(2):
        actions,diagnostic=loaded.actions(raw,generator)
        assert actions.shape==(16,4) and np.isfinite(actions).all() and np.abs(actions).max()<=1
    atomic_json(R3/'audits'/'gpu_validation'/f'{identifier}.json',dict(time=now(),passed=True,task=args.task,candidate=args.candidate,checkpoint_sha256=sha256(checkpoint),physical_gpu=args.gpu,debug_updates=20,compiled_inference_calls=2,formal_updates=0,scored_rollouts=0,source='Two D2 states; no simulator, expert or dev/test inputs',validator_sha256=sha256(Path(__file__))))
    print(identifier,'CUDA update and compiled inference passed',flush=True)


if __name__=='__main__':
    main()
