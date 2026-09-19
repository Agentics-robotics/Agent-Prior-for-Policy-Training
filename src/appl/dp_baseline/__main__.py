"""Train and evaluate M0: complete demonstrations, native actions, one DP."""
import argparse
import os
from pathlib import Path
import sys

from ..io import ROOT, read, atomic, digest, archive_source, event


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['train','evaluate'])
    parser.add_argument('--config',default=str(ROOT/'experiments/exp2/configs/m0.json'))
    parser.add_argument('--gpu',type=int,default=6)
    parser.add_argument('--seeds',type=int,nargs='+',help='Evaluate a subset of the configuration\'s fixed seeds')
    parser.add_argument('--device-isolated',action='store_true')
    args=parser.parse_args()
    if not args.device_isolated:
        from ..gpu import launch
        return launch(args.gpu,sys.argv[1:],module='appl.dp_baseline')
    cfg=read(args.config)
    if cfg['schema']!='appl.dp_baseline.v1':raise ValueError('Expected M0 configuration')
    if cfg.get('inference_overrides') or cfg.get('reset_offsets'):
        raise ValueError('Setting-study overrides are archived; evaluate the trained M0 configuration')
    cfg['data']['directory']=(ROOT/cfg['data']['directory']).resolve()
    cfg['output']=(ROOT/cfg['output']).resolve()
    cfg['output'].mkdir(parents=True,exist_ok=True)
    record=cfg['output']/'configuration.json'
    if record.exists() and read(record)!=read(args.config):raise ValueError('Run ID already has a different configuration')
    atomic(record,read(args.config))
    source_version=archive_source(cfg['output'])
    event(cfg['output']/'invocations.jsonl','command_started',command=args.command,
          arguments=sys.argv[1:],source_version=source_version)
    from ..gpu import verify_cuda
    device=verify_cuda()
    nodes={p.name for p in Path('/dev').glob('nvidia*') if p.name.removeprefix('nvidia').isdigit()}
    if nodes!={'nvidia'+os.environ['APPL_GPU_MINOR']}:raise RuntimeError('Expected one GPU device')
    atomic(cfg['output']/(args.command+'_device.json'),device)
    if args.command=='train':
        from .train import fit
        result=fit(cfg,cfg['output']/'training',seed=cfg['seed'])
        print({k:result[k] for k in ('optimizer_steps','sampling_joint_rmse','sampling_gripper_mae','training_elapsed_seconds')},flush=True)
    else:
        from .train import LoadedPolicy
        from ..evaluate import episode
        from ..envs.native import make_environment
        checkpoint=(ROOT/cfg['checkpoint']).resolve() if 'checkpoint' in cfg else cfg['output']/'training/last.pt'
        planned=cfg['evaluation_seeds'];seeds=args.seeds if args.seeds is not None else planned
        if not planned or len(set(planned))!=len(planned):raise ValueError('Unique planned evaluation seeds required')
        if not seeds or len(set(seeds))!=len(seeds) or not set(seeds)<=set(planned):
            raise ValueError('Evaluation subset must contain unique predeclared seeds')
        if any(1000<=s<=1014 for s in seeds):
            raise ValueError('Demonstration reset must not be labelled independent ID')
        plan=dict(checkpoint=str(checkpoint),sha256=digest(checkpoint),
            seeds=planned,scope=cfg['scope'],overrides={})
        receipt=cfg['output']/'evaluation_plan.json'
        if receipt.exists() and read(receipt)!=plan:raise ValueError('Evaluation checkpoint or settings changed; use a new run ID')
        if read(checkpoint.parent/'result.json')['checkpoint_sha256']!=plan['sha256']:
            raise ValueError('Evaluation requires the validated final training checkpoint')
        atomic(receipt,plan)
        env=make_environment();results=[]
        try:
            for seed in seeds:
                policy=LoadedPolicy(checkpoint,seed=seed)
                directory=cfg['output']/'rollouts'/str(seed)
                completed_before=(directory/'result.json').exists()
                if not completed_before:
                    event(cfg['output']/'invocations.jsonl','episode_started',seed=seed,
                          device=device,source_version=source_version,checkpoint_sha256=plan['sha256'])
                result=episode(cfg,env,policy,directory,seed)
                if not completed_before:
                    atomic(directory/'execution.json',dict(device=device,source_version=source_version,
                        checkpoint_sha256=plan['sha256']))
                results.append(result)
                print({k:result[k] for k in ('seed','success','steps','maximum_drawer','first_geometry_steps')},flush=True)
                del policy
        finally:env.close()
        paths=[cfg['output']/'rollouts'/str(seed)/'result.json' for seed in planned]
        if not all(p.exists() for p in paths):
            print(dict(completed_episodes=sum(p.exists() for p in paths),planned_episodes=len(planned)),flush=True)
            return
        results=[read(p) for p in paths]
        summary=dict(successes=sum(r['success'] for r in results),trials=len(results),scope=cfg['scope'],
                     results=results,checkpoint_sha256=digest(checkpoint))
        atomic(cfg['output']/'evaluation.json',summary)
        print({k:summary[k] for k in ('successes','trials','scope')},flush=True)


if __name__=='__main__':main()
