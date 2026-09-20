"""One entry for API design, independent training and API policy selection."""
import argparse
import json
import os
from pathlib import Path
import sys
from ..io import read
from .data import configuration


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['segment','initialize','queue','prepare','train','worker','policy-worker','freeze','evaluate','report'])
    p.add_argument('--config');p.add_argument('--gpu',type=int,default=4)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--request');p.add_argument('--seed',type=int)
    p.add_argument('--policy-id',help='Already submitted policy to train independently')
    p.add_argument('--device-isolated',action='store_true')
    p.add_argument('--verify',action='store_true',help='Audit API authorship and checkpoint hashes when reporting')
    args=p.parse_args();cfg=configuration(args.config)
    if args.command in ('queue','prepare','train','worker','policy-worker','evaluate') and args.gpu not in cfg['devices']:
        p.error('GPU is outside this experiment configuration allocation')
    if args.device_isolated:
        nodes={v.name for v in Path('/dev').glob('nvidia*') if v.name.removeprefix('nvidia').isdigit()}
        if nodes!={'nvidia'+os.environ['APPL_GPU_MINOR']}:raise RuntimeError('Expected exactly one GPU device')
        os.environ['CUDA_VISIBLE_DEVICES']=os.environ['APPL_GPU_UUID'];os.environ['MUJOCO_EGL_DEVICE_ID']='0'
    elif args.command not in ('segment','initialize','queue','prepare','train','freeze','report'):
        from ..gpu import launch
        return launch(args.gpu,sys.argv[1:],module='appl.prior_policies')
    if args.command=='segment':
        from .segment import run
        run(cfg)
    elif args.command=='initialize':
        from .data import fit_full_normalizer
        print(json.dumps(fit_full_normalizer(cfg)),flush=True)
    elif args.command=='freeze':
        from .deploy import freeze
        print(json.dumps(freeze(cfg)),flush=True)
    elif args.command=='report':
        from .report import generate
        result=generate(cfg,verify=args.verify)
        print(json.dumps(dict(policy_states=result['policy_states'],independent_ID=result['independent_ID'])),flush=True)
    elif args.command=='queue':
        from .runner import run_queue
        run_queue(cfg,args.gpu,args.shard,args.shards)
    elif args.command=='prepare':
        from .runner import prepare_queue
        prepare_queue(cfg,args.gpu,args.shard,args.shards)
    elif args.command=='train':
        from .runner import train_submitted
        train_submitted(cfg,args.policy_id,args.gpu)
    elif args.command=='worker':
        from .engine import fit
        r=read(args.request);fit(cfg,**r)
    elif args.command=='policy-worker':
        from .engine import LoadedPolicy
        from ..security import lockdown
        r=read(args.request);folder=Path(r['folder']);checkpoint=folder/'training/last.pt'
        from ..io import digest
        result=read(folder/'training/result.json');submission=read(folder/'submission.json')
        if digest(checkpoint)!=result['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        for name,sha in submission['files'].items():
            if digest(folder/'source'/name)!=sha:raise ValueError('API source changed')
        lockdown(folder/'source',r['output'],[checkpoint])
        policy=LoadedPolicy(folder/'source',checkpoint,r['seed'])
        print(json.dumps(dict(status='ready')),flush=True)
        for line in sys.stdin:
            value=json.loads(line)
            if value['reset']:policy.reset()
            before=len(policy.timings);action=policy.action(value['state'])
            print(json.dumps(dict(action=action,seconds=policy.timings[-1] if len(policy.timings)>before else 0)),flush=True)
    else:
        from .deploy import evaluate
        evaluate(cfg,args.seed)


if __name__=='__main__':main()
