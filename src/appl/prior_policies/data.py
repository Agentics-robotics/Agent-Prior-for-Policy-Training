"""Read immutable API slices; construct causal inputs and training-only futures."""
from pathlib import Path
import numpy as np
from ..dp_baseline.data import vector as drawer_vector, normalizer, SLICES
from ..io import ROOT, read, digest, atomic, object_hash


def vector(state):
    """Preserve drawer inputs and use explicitly observed targets for new tasks."""
    value=drawer_vector(state)
    for name,start in (('red_goal',41),('blue_goal',44)):
        if name in state:
            goal=np.asarray(state[name],dtype=np.float32)
            if goal.shape!=(3,) or not np.isfinite(goal).all():raise ValueError('Invalid observed target')
            value[start:start+3]=goal
    return value


def configuration(path=None):
    path=Path(path or ROOT/'experiments/exp2/configs/m1_v2.json').resolve()
    cfg=read(path)
    if cfg['schema']!='appl.prior_policies.v1':raise ValueError('Wrong policy experiment configuration')
    cfg['_path']=str(path)
    for name in ('output','dataset','completion_contract'):cfg[name]=(ROOT/cfg[name]).resolve()
    return cfg


def evaluation_output(cfg):
    """An explicit reevaluation directory leaves the original trained run intact."""
    value=cfg['evaluation'].get('output')
    if value is None:return cfg['output']
    path=(ROOT/value).resolve()
    if path==cfg['output'].resolve():raise ValueError('Reevaluation output must differ from the trained run')
    return path


def catalog(cfg):
    manifest=read(cfg['dataset']/'manifest.json')
    shared=None
    if cfg.get('experiment_version')=='M1_v2':
        path=cfg['output']/'normalization.json';shared=read(path)
        expected={k:dict(path=str(ROOT/cfg['normalization']['directory']/(k+'.json')),
            sha256=manifest['sources'][k]['sha256']) for k in cfg['normalization']['train_ids']}
        if shared['sources']!=expected or set(manifest['sources'])!=set(expected):
            raise ValueError('Segmentation and shared normalization must use exactly the same original training demonstrations')
        shared=dict(path=str(path),sha256=digest(path),normalizer_sha256=shared['normalizer_sha256'])
    entries=[]
    for item in manifest['datasets']:
        path=cfg['dataset']/item['dataset'];dataset=read(path)
        if digest(path)!=manifest['files'][item['dataset']]:raise ValueError('Dataset changed')
        for i,heuristic in enumerate(dataset['heuristics'],1):
            entry=dict(policy_id=dataset['skill_id']+f'__h{i:02d}',skill_id=dataset['skill_id'],
                heuristic_index=i,heuristic=heuristic,subgoal=dataset['subgoal'],
                dataset=str(path),dataset_sha256=digest(path),plan_hash=manifest['plan_hash'])
            if shared is not None:entry.update(experiment_version='M1_v2',normalization=shared)
            entries.append(entry)
    return entries


def fit_full_normalizer(cfg):
    """Fit once on complete original train trajectories; no slices or test states."""
    settings=cfg['normalization']
    if settings['scope']!='full_training_demonstrations':raise ValueError('Unknown normalization scope')
    ids=settings['train_ids']
    if len(ids)!=len(set(ids)) or not ids:raise ValueError('Unique training IDs are required')
    es=[];sources={}
    for identifier in ids:
        path=ROOT/settings['directory']/(identifier+'.json');value=read(path)
        if value['trajectory_id']!=identifier or value['provenance']!='original_demonstration':
            raise ValueError('Expected an original authorized training demonstration')
        obs=np.asarray([vector(o['state']) for o in value['observations']],np.float32)
        action=np.asarray(value['actions'],np.float32)
        if obs.shape!=(len(action)+1,47) or action.shape[1:]!=(8,):raise ValueError('Misaligned full trajectory')
        if not np.isfinite(obs).all() or not np.isfinite(action).all():raise ValueError('Nonfinite training data')
        es.append(dict(id=identifier,obs=obs,action=action))
        sources[identifier]=dict(path=str(path),sha256=digest(path))
    norm=normalizer(es,cfg['training'])
    record=dict(schema='appl.shared_normalization.v1',scope=settings['scope'],sources=sources,
        normalizer=norm,normalizer_sha256=object_hash(norm),
        fitted_observations='All causal action-input observations from the complete original trajectories (T per trajectory); final T+1 observation is a label only.',
        fitted_actions='All original training actions once, before segmentation or overlap.',
        inference_or_validation_data_used=False)
    path=cfg['output']/'normalization.json'
    if path.exists() and read(path)!=record:raise ValueError('Shared normalization is immutable; use a new version')
    atomic(path,record)
    return dict(path=str(path),sha256=digest(path),normalizer_sha256=record['normalizer_sha256'],
        trajectories=len(es),fit_samples=norm['fit_samples'])


def shared_normalizer(entry):
    reference=entry['normalization'];path=Path(reference['path'])
    if digest(path)!=reference['sha256']:raise ValueError('Shared normalizer artifact changed')
    record=read(path)
    if object_hash(record['normalizer'])!=reference['normalizer_sha256']:raise ValueError('Normalizer payload changed')
    return record['normalizer']


def handoff_evidence(entry):
    """Measured support only; semantic transfer decisions remain API-owned."""
    path=Path(entry['dataset']);dataset=read(path);root=path.parents[2]
    manifest=read(root/'manifest.json');boundaries=[];overlaps=[]
    for segment in dataset['segments']:
        value=read(path.parent/segment['file'])
        boundaries.append(dict(trajectory_id=segment['trajectory_id'],start_index=segment['start'],
            stop_index=segment['stop'],start_state=value['observations'][0]['state'],
            end_state=value['observations'][-1]['state']))
        for item in manifest['datasets']:
            if item['skill_id']==entry['skill_id']:continue
            neighbor=read(root/item['dataset'])
            for other in neighbor['segments']:
                if other['trajectory_id']!=segment['trajectory_id']:continue
                start=max(segment['start'],other['start']);stop=min(segment['stop'],other['stop'])
                if stop>start:
                    overlaps.append(dict(trajectory_id=segment['trajectory_id'],other_skill_id=item['skill_id'],
                        start=start,stop=stop,actions=stop-start,
                        start_state=value['observations'][start-segment['start']]['state'],
                        end_state=value['observations'][stop-segment['start']]['state']))
    stats={}
    for side in ('start','end'):
        vectors=np.stack([vector(row[side+'_state']) for row in boundaries])
        width=vectors[:,7:9].sum(1)
        stats[side]=dict(min=vectors.min(0).tolist(),median=np.median(vectors,axis=0).tolist(),
            max=vectors.max(0).tolist(),finger_width_m=dict(min=float(width.min()),median=float(np.median(width)),max=float(width.max())))
    return dict(schema='appl.handoff_evidence.v1',owner='deterministic_framework_measurement',
        skill_id=entry['skill_id'],plan_hash=entry['plan_hash'],dataset_sha256=entry['dataset_sha256'],
        fields=SLICES,boundary_statistics=stats,boundaries=boundaries,overlaps=overlaps,
        interpretation='Observed training support, not hard applicability thresholds, proof of grasp/contact, or learned-policy success. API authors the semantic handoff conditions.')


def episodes(entry):
    path=Path(entry['dataset']);dataset=read(path)
    if digest(path)!=entry['dataset_sha256']:raise ValueError('Assigned API dataset changed')
    root=path.parents[2];manifest=read(root/'manifest.json')
    es=[]
    for segment in dataset['segments']:
        segment_path=path.parent/segment['file']
        if digest(segment_path)!=manifest['files'][str(segment_path.relative_to(root))]:
            raise ValueError('Published API segment changed')
        value=read(segment_path)
        obs=np.asarray([vector(o['state']) for o in value['observations']],np.float32)
        action=np.asarray(value['actions'],np.float32)
        if obs.shape!=(len(action)+1,47) or action.shape[1:]!=(8,):raise ValueError('Misaligned source slice')
        es.append(dict(id=segment['trajectory_id'],obs=obs,action=action,
            start=segment['start'],stop=segment['stop'],source_sha256=digest(path.parent/segment['file'])))
    return es


def windows(es,training):
    rows={k:[] for k in ('raw_obs','native_action','mask','future_obs','future_mask')}
    h,n=training['horizon'],training['observation_steps']
    for e in es:
        length=len(e['action'])
        for t in range(length):
            ai=np.arange(t-n+1,t-n+1+h)
            oi=np.maximum(np.arange(t-n+1,t+1),0)
            rows['raw_obs'].append(e['obs'][oi])
            rows['native_action'].append(e['action'][np.clip(ai,0,length-1)])
            rows['mask'].append(((ai>=0)&(ai<length))[:,None])
            rows['future_obs'].append(e['obs'][np.clip(ai+1,0,length)])
            rows['future_mask'].append(((ai>=0)&(ai<length))[:,None])
    return {k:np.asarray(v,np.float32) for k,v in rows.items()}


def specification(entry,training,metadata):
    es=episodes(entry)
    if 'normalization' in entry:
        norm=shared_normalizer(entry)
    else:norm=normalizer(es,training)  # Historical M1_v1/contract fixtures only.
    return dict(training=training,normalizer=norm,fields=SLICES,
        observation_dimension=47,skill_id=entry['skill_id'],candidate_config=metadata['config']),es
