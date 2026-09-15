"""Outer-runner-only matched training and final target evaluation."""
import concurrent.futures
from pathlib import Path
from .io import atomic,read,digest
from .protocol import frozen


def train(cfg):
    from .worker import launch
    freeze=frozen(cfg);output=cfg['output']/'formal';library=freeze['library']['policies'];jobs=[]
    for seed in freeze['training_seeds']:
        jobs.append(dict(method='M0',seed=seed,policy_id='full',operation='baseline_train',segments=None))
        for policy in library:
            for method in ('M1','M2'):
                jobs.append(dict(method=method,seed=seed,policy_id=policy['metadata']['policy_id'],
                    operation='baseline_train' if method=='M1' else 'train',segments=policy['metadata']['segments'],
                    candidate=policy['candidate'],version=policy['version']))
    def execute(job,gpu):
        d=output/'training'/job['method']/str(job['seed'])/job['policy_id']
        if job['method']=='M2' and job['seed']==0:
            policy=next(p for p in library if p['version']==job['version'])
            result=read(Path(policy['checkpoint']).parent/'result.json')
            if digest(policy['checkpoint'])!=policy['checkpoint_sha256']:raise ValueError('Submitted checkpoint changed')
            atomic(d/'reuse.json',dict(source_checkpoint=policy['checkpoint'],sha256=policy['checkpoint_sha256'],new_optimizer_steps=0))
        else:
            if (d/'result.json').exists():result=read(d/'result.json')
            else:
                process=launch(cfg,dict(**job,updates=cfg['training']['updates']),d,gpu)
                code=process.wait()
                if code:raise RuntimeError(f'Matched training job failed: {d}; no selective automatic retry')
                result=read(d/'result.json')
        if digest(result['checkpoint'])!=result['checkpoint_sha256']:raise ValueError('Training checkpoint mismatch')
        return dict(job=job,result=result)
    # Four fixed device queues; no two jobs from this runner share one GPU.
    groups=[jobs[i::4] for i in range(4)]
    def queue(index):return [execute(job,cfg['devices']['allowed'][index]) for job in groups[index]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=[row for group in pool.map(queue,range(4)) for row in group]
    result=dict(freeze_sha256=freeze['freeze_sha256'],jobs=results)
    atomic(output/'training.json',result);return result


def evaluate(cfg):
    freeze=frozen(cfg)
    from .gpu import verify_cuda
    from .envs.native import make_environment
    from .envs.adapter import reset
    from .evaluate import episode
    from .train import LoadedPolicy
    from .deployment import run
    verify_cuda();output=cfg['output']/'formal'
    if not (output/'training.json').exists():raise ValueError('Finish all paired methods/training seeds before final target tests')
    trained=read(output/'training.json')
    if trained['freeze_sha256']!=freeze['freeze_sha256']:raise ValueError('Training uses another formal freeze')
    results=[];env=make_environment()
    try:
        for seed in freeze['training_seeds']:
            for case in freeze['cases']:
                for method in ('M0','M1','M2'):
                    d=output/'evaluation'/method/str(seed)/case['condition']/str(case['seed'])
                    if (d/'result.json').exists():result=read(d/'result.json')
                    else:
                        if d.exists():raise RuntimeError('Interrupted target episode retained; explicit fresh-attempt reconciliation required')
                        jobs=[r for r in trained['jobs'] if r['job']['method']==method and r['job']['seed']==seed]
                        for r in jobs:
                            if digest(r['result']['checkpoint'])!=r['result']['checkpoint_sha256']:raise ValueError('Frozen policy weights changed')
                        if method=='M0':
                            policy=LoadedPolicy(jobs[0]['result']['checkpoint'],seed=case['seed'])
                            result=episode(cfg,env,policy,d,case['seed'],variant=case['variant'],offsets=case['offsets'])
                            del policy
                        else:
                            policies=[]
                            for original in freeze['library']['policies']:
                                r=next(r for r in jobs if r['job']['policy_id']==original['metadata']['policy_id'])
                                policies.append(dict(metadata=original['metadata'],checkpoint=r['result']['checkpoint'],
                                    candidate=original['candidate'] if method=='M2' else None))
                            _,state=reset(env,case['seed'],case['variant'],case['offsets'])
                            result=run(cfg,env,state,policies,d,case['seed'])
                        result.update(method=method,training_seed=seed,condition=case['condition'],seed=case['seed'],freeze_sha256=freeze['freeze_sha256'])
                        atomic(d/'result.json',result)
                    results.append(result)
        value=dict(freeze_sha256=freeze['freeze_sha256'],results=results)
        atomic(output/'evaluation.json',value);return value
    finally:env.close()
