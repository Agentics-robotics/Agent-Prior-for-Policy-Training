"""Frozen inputs for the authorized inference-only binary-gripper comparison."""
import csv
from pathlib import Path
import time
from appl.io import ROOT,read,atomic,digest,object_hash,source_manifest

BASE=ROOT/'runs/exp2/binary_gripper_v1_20260919'
MODULE='experiments.exp2.binary_gripper.run'
METHODS=['naive_DP','SinglePrior_6_xhigh']
TASKS=['drawer_exchange','two_block_sort','buffer_swap','unstack_sort','tray_pack']


def local_source():
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def execute_action(raw,low,high):
    import numpy as np
    raw=np.asarray(raw,np.float32)
    if raw.shape!=(8,) or not np.isfinite(raw).all():raise ValueError('Invalid raw learned action')
    action=np.clip(raw,low,high)
    action[7]=1.0 if raw[7]>=0 else -1.0
    return action


def prepare():
    from appl.scaleup.protocol import config,naive_paths
    from experiments.exp2.single_policy.common import config as prior_config,folder,framework
    if BASE.exists():raise ValueError('Study already exists; never overwrite its plan')
    originals=list(csv.DictReader((ROOT/'experiments/exp2/paper/tables/episodes.csv').open()))
    original={(r['task'],r['method'],r['condition'],int(r['seed'])):r for r in originals}
    models={};cells=[];BASE.mkdir(parents=True)
    for task in TASKS:
        cfg=config(task);s,checkpoint=naive_paths(task)
        for method in METHODS:
            if method=='SinglePrior_6_xhigh':
                f=folder(task);s=f/'source';checkpoint=f/'training/last.pt';c=prior_config(task)
                resource=read(c['_path']);resource['devices']=list(range(8))
                path=BASE/'configurations'/(task+'.json');atomic(path,resource)
                source_hashes=read(f/'submission.json')['files']
                for name,sha in source_hashes.items():
                    if digest(s/name)!=sha:raise ValueError('API submission changed')
            else:
                f=None;s,checkpoint=naive_paths(task);path=None
                source_hashes={} if s is None else {p.name:digest(p) for p in s.iterdir() if p.is_file()}
            key=task+'/'+method
            receipt=read(checkpoint.parent/'result.json')
            if digest(checkpoint)!=receipt['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
            models[key]=dict(task=task,method=method,source=None if s is None else str(s),folder=None if f is None else str(f),
                checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),source_hashes=source_hashes,
                worker_config=None if path is None else str(path),worker_config_sha256=None if path is None else digest(path),
                goal_contract=str(cfg['completion_contract']),goal_sha256=digest(cfg['completion_contract']),
                original_configuration=cfg['_path'] if f is None else prior_config(task)['_path'])
    for index in range(30):
        for condition in ['ID','OOD']:
            for task in TASKS:
                seed=config(task)['evaluation']['seeds' if condition=='ID' else 'ood_seeds'][index]
                for method in METHODS:
                    old=original[(task,method,condition,seed)];root=Path(old['episode_root'])
                    assert old['completed']=='True'
                    cells.append(dict(index=len(cells),task=task,method=method,condition=condition,seed=seed,
                        reference_root=str(root),reference_result_sha256=digest(root/'result.json'),
                        reference_initial_sha256=digest(root/'initial_state.json'),reference_trace_sha256=digest(root/'trace.jsonl'),
                        original_success=old['success']=='True',original_steps=int(old['steps'])))
    plan=dict(study='binary_gripper_v1_20260919',created=time.time(),owner='developer inference executor',
        authorization='User requested complete new naive DP and single-policy Agentic Prior DP runs, with inference-only gripper correction; both APPL variants paused to avoid API cost.',
        intervention='Only final native action[7]: +1 for raw[7]>=0, else -1; seven arm targets receive their original physical bound clipping.',
        cells=cells,models=models,maximum_steps=5000,initial_active_devices=[4,5,6,7],allowed_devices=list(range(8)),maximum_active_devices=5,
        jobs_per_device=4,minimum_free_mib_to_admit=2500,API_calls_allowed=0,optimizer_updates_allowed=0,
        paired_previously_examined_layouts=True,APPL_reevaluation=False,automatic_retry=False,
        frozen_framework=source_manifest(),single_policy_framework=framework(),executor_source=local_source())
    atomic(BASE/'plan.json',plan);atomic(BASE/'plan_sha256.json',dict(sha256=digest(BASE/'plan.json')))
    print(dict(prepared=True,cells=len(cells),models=len(models),output=str(BASE)),flush=True)


def verify():
    from experiments.exp2.single_policy.common import framework
    p=read(BASE/'plan.json')
    if digest(BASE/'plan.json')!=read(BASE/'plan_sha256.json')['sha256']:raise ValueError('Study plan changed')
    if source_manifest()!=p['frozen_framework'] or local_source()!=p['executor_source']:raise ValueError('Frozen executor changed')
    if framework()!=p['single_policy_framework']:raise ValueError('Frozen single-policy framework changed')
    return p


def model_record(cell):
    record=verify()['models'][cell['task']+'/'+cell['method']]
    if digest(record['checkpoint'])!=record['checkpoint_sha256']:raise ValueError('Frozen checkpoint changed')
    for name,sha in record['source_hashes'].items():
        if digest(Path(record['source'])/name)!=sha:raise ValueError('Policy source changed')
    if digest(record['goal_contract'])!=record['goal_sha256']:raise ValueError('Task goal changed')
    if record['worker_config'] and digest(record['worker_config'])!=record['worker_config_sha256']:raise ValueError('Worker resource configuration changed')
    return record


def episode_root(cell):
    return BASE/cell['task']/'evaluation'/cell['method']/cell['condition']/str(cell['seed'])


def load_policy(cell,out):
    from appl.prior_policies.runner import PolicyProcess
    from appl.prior_policies.data import configuration
    record=model_record(cell)
    if cell['method']=='SinglePrior_6_xhigh':
        return PolicyProcess(configuration(record['worker_config']),Path(record['folder']),out/'worker',cell['seed'])
    if cell['task']=='drawer_exchange':
        from appl.dp_baseline.train import LoadedPolicy
        return LoadedPolicy(record['checkpoint'],seed=cell['seed'])
    from appl.prior_policies.engine import LoadedPolicy
    return LoadedPolicy(record['source'],record['checkpoint'],seed=cell['seed'])


def close_policy(policy,cell):
    if cell['method']=='SinglePrior_6_xhigh':policy.close()
