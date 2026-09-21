"""Full-trajectory baseline through the common DDPM trainer; no API provenance."""
import copy
import shutil
from ..io import ROOT,read,atomic,digest,object_hash
from ..prior_policies.data import fit_full_normalizer,configuration
from ..prior_policies.runner import gpu_job


def prepare(cfg):
    fit_full_normalizer(cfg)
    folder=cfg['output']/'naive_DP';source=folder/'source';source.mkdir(parents=True,exist_ok=True)
    dataset_root=folder/'full_trajectories';dataset=dataset_root/'datasets/full_task/dataset.json'
    files={};segments=[];sources={}
    for identifier in cfg['normalization']['train_ids']:
        original=ROOT/cfg['normalization']['directory']/(identifier+'.json');value=read(original)
        target=dataset.parent/(identifier+'.json');target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and digest(target)!=digest(original):raise ValueError('Baseline data changed')
        shutil.copyfile(original,target);files[str(target.relative_to(dataset_root))]=digest(target)
        sources[identifier]=dict(path=str(original),sha256=digest(original))
        segments.append(dict(trajectory_id=identifier,start=0,stop=len(value['actions']),file=target.name))
    atomic(dataset,dict(skill_id='full_task',owner='developer_naive_baseline',segments=segments))
    files[str(dataset.relative_to(dataset_root))]=digest(dataset)
    atomic(dataset_root/'manifest.json',dict(owner='developer_naive_baseline',sources=sources,files=files))
    norm=cfg['output']/'normalization.json'
    entry=dict(policy_id='naive_DP',skill_id='full_task',dataset=str(dataset),dataset_sha256=digest(dataset),
        experiment_version='framework_naive',owner='developer',normalization=dict(path=str(norm),sha256=digest(norm),
        normalizer_sha256=read(norm)['normalizer_sha256']))
    atomic(folder/'assignment.json',entry)
    shutil.copyfile(ROOT/'src/appl/scaleup/naive_policy.py',source/'policy.py')
    atomic(source/'pipeline.json',dict(policy_id='naive_DP',config={},owner='developer',
        architecture='Vanilla conditional action diffusion U-Net; flattened normalized two-frame state.',
        objective='Masked epsilon MSE only; no auxiliary objective.'))
    hashes={p.name:digest(p) for p in source.iterdir() if p.is_file()}
    atomic(folder/'framework_source.json',dict(owner='developer',api_authored=False,files=hashes,version=object_hash(hashes)))
    raw=read(cfg['_path']);raw['training']['updates']=60000
    path=folder/'config.json';atomic(path,raw)
    return configuration(path),folder


def train(cfg,gpu):
    folder=cfg['output']/'naive_DP'
    if (folder/'training/result.json').exists():
        result=read(folder/'training/result.json');authored=read(folder/'framework_source.json')
        if digest(folder/'training/last.pt')!=result['checkpoint_sha256'] or any(
            digest(folder/'source'/k)!=v for k,v in authored['files'].items()):raise ValueError('Frozen baseline changed')
        return result
    if (folder/'training').exists():raise ValueError('Existing training owner or partial attempt: no implicit restart')
    cfg,folder=prepare(cfg)
    return gpu_job(cfg,folder,folder/'source',folder/'training',gpu,60000)
