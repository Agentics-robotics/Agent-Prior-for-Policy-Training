"""Preparation validation and exportable scientific artifacts."""
import numpy as np
from pathlib import Path
from ..io import ROOT,read,atomic,digest
from ..prior_policies.data import vector
from .protocol import BASE,config
from .tasks import TASKS,measure


def preparations():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    result={};fig,axes=plt.subplots(4,3,figsize=(10,12))
    for row,name in enumerate(TASKS):
        cfg=config(name);source=ROOT/cfg['normalization']['directory'];records=[]
        goals=read(cfg['completion_contract'])
        for identifier in cfg['normalization']['train_ids']:
            path=source/(identifier+'.json');d=read(path);actions=np.asarray(d['actions'])
            assert d['scope']=='training' and d['condition']=='ID' and d['provenance']=='original_demonstration'
            assert d['task_spec_sha256']==digest(BASE/name/'task.json')
            assert actions.shape[1:]==(8,) and len(d['observations'])==len(actions)+1
            states=np.asarray([vector(v['state']) for v in d['observations']])
            assert states.shape==(len(actions)+1,47) and np.isfinite(actions).all() and np.isfinite(states).all()
            assert measure(d['observations'][-1]['state'],goals)['success']
            for index,obs in enumerate(d['observations']):
                assert abs(obs['timestamp']-index/20)<1e-8 and (source/obs['images']['front']).is_file()
            records.append(dict(trajectory_id=identifier,steps=len(actions),sha256=digest(path),source_version=d['source_version']))
        first=read(source/(cfg['normalization']['train_ids'][0]+'.json'))
        for col,index in enumerate((0,len(first['actions'])//2,len(first['actions']))):
            axes[row,col].imshow(Image.open(source/first['observations'][index]['images']['front']))
            axes[row,col].axis('off');axes[row,col].set_title(name.replace('_',' ')+f' / step {index}',fontsize=10)
        result[name]=dict(count=len(records),trajectories=records,total_actions=sum(r['steps'] for r in records),
            source_versions=sorted({r['source_version'] for r in records}))
    fig.tight_layout();fig.savefig(BASE/'demonstrations.png',dpi=150);plt.close(fig)
    atomic(BASE/'data_validation.json',dict(tasks=result,passed=True,trajectories=sum(v['count'] for v in result.values()),
        goal_definition='Fixed geometric goals without terminal extras',learned_policy_evaluations=0))
    return result


def policy_workers(name):
    """Real B=1 deployment protocol, original training observations, zero physics."""
    import time
    import torch
    from ..gpu import verify_cuda
    from ..prior_policies.data import catalog
    from ..prior_policies.runner import PolicyProcess
    from ..prior_policies.engine import LoadedPolicy
    cfg=config(name);root=BASE/name/'inference_checks';root.mkdir(parents=True,exist_ok=True)
    device=verify_cuda();records=[]
    entries=catalog(cfg)
    while not (BASE/name/'naive_DP/training/result.json').exists():time.sleep(20)
    naive=BASE/name/'naive_DP';done=root/'naive_DP.json'
    if done.exists():records.append(read(done))
    else:
        model=LoadedPolicy(naive/'source',naive/'training/last.pt',913)
        original=ROOT/cfg['normalization']['directory']/(cfg['normalization']['train_ids'][0]+'.json')
        trajectory=read(original);actions=[]
        for obs in trajectory['observations'][:9]:
            value=model.action(obs['state'])
            if np.shape(value)!=(8,) or not np.isfinite(value).all():raise ValueError('Invalid baseline deployment action')
            actions.append(value)
        record=dict(policy_id='naive_DP',passed=True,batch_size=1,generated_actions=len(actions),
            denoising_chunks=len(model.timings),device=device,checkpoint_sha256=digest(naive/'training/last.pt'),
            optimizer_updates=0,simulator_steps=0,original_training_observations_only=True,source_sha256=digest(original))
        atomic(done,record);records.append(record);del model;torch.cuda.empty_cache()
    pending={e['policy_id']:e for e in entries}
    while pending:
        for key,entry in list(pending.items()):
            folder=cfg['output']/'policies'/entry['skill_id']/f"heuristic_{entry['heuristic_index']:02d}"
            out=root/key
            if (out/'result.json').exists():records.append(read(out/'result.json'));del pending[key];continue
            if (folder/'failure.json').exists() or (folder/'training/failure.json').exists():raise RuntimeError('Recorded terminal policy failure')
            if not (folder/'training/result.json').exists():continue
            if out.exists():raise RuntimeError('Interrupted deployment interface check requires reconciliation')
            out.mkdir();worker=None;started=time.monotonic()
            try:
                dataset=read(entry['dataset']);segment=dataset['segments'][0]
                source=Path(entry['dataset']).parent/segment['file']
                observations=read(source)['observations'][:9]
                atomic(out/'inputs.json',dict(source=str(source),source_sha256=digest(source),
                    start_index=segment['start'],observations=observations,seed=913,physical_steps=0))
                worker=PolicyProcess(cfg,folder,out/'worker',913);actions=[]
                for index,obs in enumerate(observations):
                    value=worker.action(obs['state'],reset=index==0)
                    if np.shape(value)!=(8,) or not np.isfinite(value).all():raise ValueError('Invalid prior deployment action')
                    actions.append(value)
                worker.close();worker=None
                closed=read(out/'worker/closed.json')
                if closed['returncode']!=0 or closed['inference_chunks']!=2:raise ValueError('Deployment worker contract failed')
                record=dict(policy_id=key,passed=True,batch_size=1,actual_worker=True,generated_actions=len(actions),
                    denoising_chunks=closed['inference_chunks'],elapsed_seconds=time.monotonic()-started,
                    checkpoint_sha256=digest(folder/'training/last.pt'),device=device,optimizer_updates=0,
                    simulator_steps=0,original_training_observations_only=True,independent_process_closed=True)
                atomic(out/'actions.json',actions);atomic(out/'result.json',record);records.append(record);del pending[key]
                print(dict(task=name,checked=key),flush=True)
            except Exception as error:
                atomic(out/'failure.json',dict(error=str(error),automatic_retry=False));raise
            finally:
                if worker is not None:worker.close()
        if pending:time.sleep(20)
    atomic(root/'summary.json',dict(passed=True,prior_policies=len(entries),naive_models=1,records=records,
        optimizer_updates=0,simulator_steps=0,all_policies_checked=True))


if __name__=='__main__':preparations()
