"""Fixed paired layouts and immutable learned-library manifest."""
import copy
import time
import numpy as np
from ..io import ROOT,read,atomic,digest,object_hash,archive_source
from ..prior_policies.data import configuration
from .tasks import TASKS,task,layout

BASE=ROOT/'runs/exp2/M1_scaleup'
CONFIGS=ROOT/'experiments/exp2/configs/scaleup'
NAMES=['drawer_exchange',*TASKS]


def config(name):return configuration(CONFIGS/(name+'.json'))


def initial_layout(name,seed,condition):
    if name!='drawer_exchange':return layout(read(BASE/name/'task.json'),seed,condition)
    from ..envs.scene import initial_state,DRAWER_ORIGIN
    initial=initial_state(seed)
    if condition=='ID':return dict(red=initial['red'],blue=initial['blue'])
    if condition!='OOD':raise ValueError('Unknown condition')
    spec=task('two_block_sort');spec['initial']=dict(red=(np.array(DRAWER_ORIGIN)+[-.085,-.065,.028]).tolist(),blue=[-.4,.3,.02])
    return layout(spec,seed,condition)


def prepare_drawer():
    cfg=read(ROOT/'experiments/exp2/configs/m1_v2_5000.json')
    if (BASE/'allocation/amendment.json').exists():cfg['devices']=read(BASE/'allocation/amendment.json')['current_pool']
    cfg.update(task_id='drawer_exchange',output='runs/exp2/M1_initial_test')
    cfg['evaluation'].update(output='runs/exp2/M1_scaleup/drawer_exchange/evaluation',
        seeds=list(range(6300,6330)),ood_seeds=list(range(6400,6430)),diagnostic_seeds=[])
    path=CONFIGS/'drawer_exchange.json'
    if path.exists() and read(path)!=cfg:raise ValueError('Drawer scale-up configuration changed')
    atomic(path,cfg)
    atomic(BASE/'drawer_exchange/task.json',dict(task_id='drawer_exchange',
        original_environment_unchanged=True,training_data='data/exp2/demonstrations_v2',
        naive_checkpoint='runs/exp2/m0/training/last.pt',prior_library='runs/exp2/M1_initial_test/policies',
        position_ood=dict(inner_m=.022,outer_m=.04,original_red_id_half_width_m=.012,original_blue_id_half_width_m=.015)))


def naive_paths(name):
    if name=='drawer_exchange':
        return None,ROOT/'runs/exp2/m0/training/last.pt'
    folder=BASE/name/'naive_DP';return folder/'source',folder/'training/last.pt'


def freeze():
    from ..prior_policies.deploy import library
    from ..prior_policies.report import authorship
    rows={}
    for name in NAMES:
        cfg=config(name);policies=library(cfg)
        for p in policies.values():
            # The original library's existing audit is retained byte-for-byte.
            if name=='drawer_exchange':
                if not read(p['folder']/'authorship_audit.json')['passed']:raise ValueError('Missing historical authorship audit')
            else:authorship(p['folder'],read(p['folder']/'submission.json'))
        source,checkpoint=naive_paths(name)
        check_root=cfg['output']/'inference_checks' if name=='drawer_exchange' else BASE/name/'inference_checks'
        checks=read(check_root/'summary.json')
        if not checks['passed'] or not checks['all_policies_checked']:raise ValueError('Deployment interface checks incomplete')
        trained=read(checkpoint.parent/'result.json')
        if digest(checkpoint)!=trained['checkpoint_sha256']:raise ValueError('Naive checkpoint changed')
        if source is not None:
            authored=read(source.parent/'framework_source.json')
            if authored['api_authored'] or any(digest(source/k)!=v for k,v in authored['files'].items()):
                raise ValueError('Baseline source ownership/hash mismatch')
        rows[name]=dict(configuration_sha256=digest(cfg['_path']),
            task_spec_sha256=digest(BASE/name/'task.json'),completion_contract_sha256=digest(cfg['completion_contract']),
            dataset_manifest_sha256=digest(cfg['dataset']/'manifest.json'),
            normalization_sha256=digest(cfg['output']/'normalization.json'),
            policies={k:dict(version=p['version'],checkpoint_sha256=p['checkpoint_sha256']) for k,p in policies.items()},
            naive_checkpoint_sha256=digest(checkpoint),
            deployment_check_sha256=digest(check_root/'summary.json'),
            layouts={condition:[dict(seed=seed,layout=initial_layout(name,seed,condition)) for seed in
                cfg['evaluation']['seeds' if condition=='ID' else 'ood_seeds']] for condition in ('ID','OOD')})
    value=dict(tasks=rows,framework_source=archive_source(BASE),max_steps=5000,episodes=600)
    path=BASE/'study_freeze.json'
    if path.exists():
        if read(path)['study']!=value:raise ValueError('Frozen study changed')
    else:atomic(path,dict(frozen=time.time(),study=value,study_sha256=object_hash(value)))
    return read(path)
