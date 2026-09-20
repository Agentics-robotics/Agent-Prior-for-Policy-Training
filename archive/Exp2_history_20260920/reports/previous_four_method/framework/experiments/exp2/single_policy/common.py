"""Preparation, immutable inputs and study-level contracts."""
import copy
from pathlib import Path
import shutil
import time

from appl.io import ROOT, atomic, digest, object_hash, read, source_manifest
from appl.prior_policies.data import configuration, fit_full_normalizer

BASE=ROOT/'runs/exp2/single_policy_astra_xhigh'
CONFIGS=ROOT/'experiments/exp2/configs/single_policy_astra_xhigh'
OLD=ROOT/'runs/exp2/M1_scaleup'
COMPARISON=ROOT/'runs/exp2/M1_astra_xhigh/evaluation_followup_20260918/results.json'
NAMES=['drawer_exchange','two_block_sort','buffer_swap','unstack_sort','tray_pack']
DEVICES=[0,1,7]
METHOD='SinglePrior_6_xhigh'
MODULE='experiments.exp2.single_policy.runner'


def immutable(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value: raise ValueError('Immutable record changed: '+str(path))
    else: atomic(path,value)


def config(name):
    if name not in NAMES: raise ValueError('Unknown task')
    cfg=configuration(CONFIGS/(name+'.json'))
    if (cfg['api']['model'],cfg['api']['reasoning_effort'],cfg['training']['updates']) != ('gpt-6-astra','xhigh',60000):
        raise ValueError('Single-policy design/training contract changed')
    if cfg['output']!=(BASE/name).resolve() or cfg['devices']!=DEVICES:
        raise ValueError('Output/allocation mismatch')
    return cfg


def folder(name): return BASE/name/'policy'


def framework():
    result=source_manifest()
    for p in Path(__file__).parent.glob('*.py'):
        result[str(p.relative_to(ROOT))]=digest(p)
    result['experiments/exp2/SINGLE_POLICY.md']=digest(ROOT/'experiments/exp2/SINGLE_POLICY.md')
    return result


def describe_task(name,spec):
    if name=='drawer_exchange':
        return dict(task_id=name,description='Open the drawer, move red from the drawer onto its pad, and move blue from the table into the drawer. The three supplied geometric goals must hold simultaneously.',
            control_hz=20,observation_and_action_interface='See INTERFACE.md; all complete original demonstrations are available.')
    keys=('task_id','description','control_hz','fixtures','goal_half_xy_m','goal_z_tolerance_m','goals','object_half_m')
    return {k:spec[k] for k in keys if k in spec}


def prepare():
    from appl.gpu import identity
    tasks={}
    for name in NAMES:
        previous=read(ROOT/f'experiments/exp2/configs/scaleup/{name}.json')
        cfg=copy.deepcopy(previous)
        cfg.update(experiment_version='Exp2_single_policy',task_id=name,devices=DEVICES,
            output=f'runs/exp2/single_policy_astra_xhigh/{name}',
            dataset=f'runs/exp2/single_policy_astra_xhigh/{name}/full_trajectories')
        cfg['api'].update(model='gpt-6-astra',reasoning_effort='xhigh')
        cfg['training']['updates']=60000
        cfg['evaluation']['output']=cfg['output']+'/evaluation'
        cfg['evaluation']['diagnostic_seeds']=[]
        immutable(CONFIGS/(name+'.json'),cfg)
        current=config(name);fit_full_normalizer(current)
        normalization=current['output']/'normalization.json'
        prior_norm=read(ROOT/previous['output']/'normalization.json')
        if read(normalization)!=prior_norm: raise ValueError('Original full-demo normalization changed')
        ds=current['dataset'];dataset=ds/'datasets/full_task/dataset.json'
        segments=[];files={};sources={}
        for identifier in cfg['normalization']['train_ids']:
            src=ROOT/cfg['normalization']['directory']/(identifier+'.json');v=read(src)
            target=dataset.parent/(identifier+'.json');target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if digest(target)!=digest(src): raise ValueError('Complete demonstration copy changed')
            else: shutil.copyfile(src,target)
            segments.append(dict(trajectory_id=identifier,start=0,stop=len(v['actions']),file=target.name))
            files[str(target.relative_to(ds))]=digest(target)
            sources[identifier]=dict(path=str(src),sha256=digest(src))
        immutable(dataset,dict(skill_id='full_task',owner='developer_full_trajectory_preparation',segments=segments))
        files[str(dataset.relative_to(ds))]=digest(dataset)
        immutable(ds/'manifest.json',dict(owner='developer_full_trajectory_preparation',sources=sources,files=files,
            original_demonstrations=len(segments),segmentation_performed=False,excluded_actions=0))
        entry=dict(policy_id=METHOD,skill_id='full_task',heuristic_index=1,
            experiment_version='Exp2_single_policy',scientific_design_owner='Runtime API',
            dataset=str(dataset),dataset_sha256=digest(dataset),
            normalization=dict(path=str(normalization),sha256=digest(normalization),normalizer_sha256=prior_norm['normalizer_sha256']))
        immutable(folder(name)/'assignment.json',entry)
        task=read(OLD/name/'task.json')
        immutable(BASE/name/'task.json',task)
        immutable(folder(name)/'task_description.json',describe_task(name,task))
        tasks[name]=dict(configuration_sha256=digest(current['_path']),sources=sources,
            dataset_manifest_sha256=digest(ds/'manifest.json'),normalization_sha256=digest(normalization),
            task_spec_sha256=digest(BASE/name/'task.json'),completion_contract_sha256=digest(current['completion_contract']),
            assignment_sha256=digest(folder(name)/'assignment.json'),
            task_description_sha256=digest(folder(name)/'task_description.json'))
    immutable(BASE/'preparation.json',dict(tasks=tasks,framework_files=framework(),
        comparator_results_sha256=digest(COMPARISON),model='gpt-6-astra',reasoning_effort='xhigh',
        candidates_per_task=1,formal_updates_per_task=60000,new_evaluation_episodes=300,
        deployment_API_calls=0,user_selected_one_candidate=True))
    if not (BASE/'allocation.json').exists():
        atomic(BASE/'allocation.json',dict(checked=time.time(),devices=DEVICES,
            occupancy={i:identity(i) for i in DEVICES},maximum_authorized_simultaneous_devices=5,
            note='Use three inspected idle GPUs, allowing multiple jobs per device. Unrelated processes are preserved.'))
    return read(BASE/'preparation.json')


def verify_preparation():
    prepared=read(BASE/'preparation.json')
    if framework()!=prepared['framework_files']: raise ValueError('Prepared scientific framework changed')
    if digest(COMPARISON)!=prepared['comparator_results_sha256']: raise ValueError('Frozen comparators changed')
    for name,p in prepared['tasks'].items():
        cfg=config(name)
        for path,key in ((cfg['_path'],'configuration_sha256'),(cfg['dataset']/'manifest.json','dataset_manifest_sha256'),
                         (cfg['output']/'normalization.json','normalization_sha256'),
                         (cfg['completion_contract'],'completion_contract_sha256'),(BASE/name/'task.json','task_spec_sha256'),
                         (folder(name)/'assignment.json','assignment_sha256'),(folder(name)/'task_description.json','task_description_sha256')):
            if digest(path)!=p[key]: raise ValueError('Prepared input changed: '+str(path))
        for src in p['sources'].values():
            if digest(src['path'])!=src['sha256']: raise ValueError('Original training data changed')
        for rel,sha in read(cfg['dataset']/'manifest.json')['files'].items():
            if digest(cfg['dataset']/rel)!=sha: raise ValueError('Training manifest data changed')
    return prepared


def freeze():
    from experiments.exp2.astra.runner import api_identity
    from appl.prior_policies.report import authorship
    from appl.scaleup.protocol import initial_layout
    verify_preparation();tasks={}
    for name in NAMES:
        f=folder(name);submitted=read(f/'submission.json');trained=read(f/'training/result.json')
        if trained['optimizer_steps']!=60000 or trained['interface_check']: raise ValueError('Formal training incomplete')
        if digest(f/'training/last.pt')!=trained['checkpoint_sha256']: raise ValueError('Checkpoint changed')
        api_identity(f/'design/journal.sqlite');authorship(f,submitted)
        checked=read(f/'deployment_check/result.json')
        if not checked['passed'] or checked['checkpoint_sha256']!=trained['checkpoint_sha256']:
            raise ValueError('Deployment check missing or stale')
        cfg=config(name)
        tasks[name]=dict(source_hashes=submitted['files'],version=submitted['version'],
            checkpoint_sha256=trained['checkpoint_sha256'],deployment_check_sha256=digest(f/'deployment_check/result.json'),
            layouts={c:[dict(seed=s,layout=initial_layout(name,s,c)) for s in cfg['evaluation']['seeds' if c=='ID' else 'ood_seeds']]
                for c in ('ID','OOD')})
    study=dict(tasks=tasks,preparation_sha256=digest(BASE/'preparation.json'),framework_files=framework(),
        max_steps=5000,episodes=300,policy_selection='One offline candidate per task; last EMA; no runtime agent')
    immutable(BASE/'study_freeze.json',dict(study=study,study_sha256=object_hash(study)))
    return read(BASE/'study_freeze.json')
