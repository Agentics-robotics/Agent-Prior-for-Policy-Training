"""Fixed development gates and method-independent scene definitions."""
from .io import read,atomic,object_hash,source_manifest,digest,archive_source

ROLES=('open_drawer','move_red','move_blue')
CONDITIONS={
    'ID':dict(variant='standard',offsets=[0.,0.,0.,0.]),
    'OOD-position':dict(variant='standard',offsets=[0.,.03,.05,0.]),
    'OOD-state':dict(variant='blue_already_inside',offsets=[0.,0.,0.,0.]),
}


def gates(cfg):
    r=cfg['output'];values={}
    for name,path in [('native_replay','diagnostics/D0/result.json'),('batch_fit','diagnostics/D1/diagnostic.json'),
                      ('ID_development','diagnostics/D4/result.json')]:
        p=r/path;values[name]=p.exists() and read(p).get('success') is True
    values['single_trajectory_diagnosed']=(r/'diagnostics/D2/result.json').exists()
    values['all_skills_diagnosed']=all((r/'diagnostics/D3'/role/'result.json').exists() for role in ROLES)
    p=r/'reconstruction/submission.json'
    values['calibrated_MuJoCo']=p.exists() and read(p).get('calibration_passed') is True
    p=r/'api_capability/submission.json'
    values['actual_API_neural_training']=p.exists() and read(p)['training'].get('neural_training_verified') is True
    p=r/'reference/result.json'
    values['independent_scene_reachability']=p.exists() and read(p).get('success') is True
    return dict(passed=all(values.values()),checks=values,unmet=[k for k,v in values.items() if not v])


def require_design_gates(cfg):
    result=gates(cfg)
    if not result['passed']:raise ValueError('Formal design is blocked by declared gates: '+', '.join(result['unmet']))
    return result


def freeze(cfg,library):
    require_design_gates(cfg)
    archive_source(cfg['output'])
    # Test seeds are held by this fixed runner, never included in design tools.
    cases=[]
    for condition_index,(condition,options) in enumerate(CONDITIONS.items()):
        for index in range(cfg['evaluation']['paired_resets_per_condition']):
            cases.append(dict(condition=condition,seed=20000+condition_index*1000+index,**options))
    reconstruction=read(cfg['output']/'reconstruction/submission.json')
    value=dict(schema='appl.formal.freeze.v1',configuration_sha256=cfg['_hash'],source=source_manifest(),
        library=library,reconstruction_version=reconstruction['version'],
        training_seeds=cfg['training']['seeds'],cases=cases,training=cfg['training'],evaluation=cfg['evaluation'],
        checkpoint_selection=cfg['training']['selection'],methods=['M0','M1','M2'],
        data_hashes=read(cfg['output']/'audit.json')['data']['source_hashes'])
    value['freeze_sha256']=object_hash(value)
    path=cfg['output']/'formal/freeze.json'
    if path.exists() and read(path)!=value:raise ValueError('An incompatible formal freeze already exists')
    atomic(path,value);return value


def frozen(cfg):
    path=cfg['output']/'formal/freeze.json'
    if not path.exists():
        require_design_gates(cfg)
        raise ValueError('No explicitly frozen API second-version library')
    value=read(path);check=dict(value);h=check.pop('freeze_sha256')
    if h!=object_hash(check) or cfg['_hash']!=value['configuration_sha256'] or source_manifest()!=value['source']:
        raise ValueError('Frozen source/configuration hash mismatch; a new development cycle is required')
    for identifier,expected in value['data_hashes'].items():
        if digest(cfg['data']['directory']/(identifier+'.json'))!=expected:raise ValueError('Frozen demonstration changed')
    return value
