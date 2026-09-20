"""Freeze task preparation and evaluation seeds before any learned-policy testing."""
import copy
from ..io import ROOT,read,atomic
from .tasks import TASKS,task,contract,layout


def initialize():
    base=ROOT/'runs/exp2/M1_scaleup';configs=ROOT/'experiments/exp2/configs/scaleup'
    reference=read(ROOT/'experiments/exp2/configs/m1_v2_5000.json');rows=[]
    for index,name in enumerate(TASKS):
        root=base/name;spec=task(name)
        path=root/'task.json'
        if path.exists() and read(path)!=spec:raise ValueError('Task specification already frozen')
        atomic(path,spec);atomic(configs/(name+'_goals.json'),contract(spec))
        cfg=copy.deepcopy(reference);cfg.update(output=str(root.relative_to(ROOT)),
            task_id=name,dataset='data/exp2/scaleup/'+name+'/processed',
            completion_contract=str((configs/(name+'_goals.json')).relative_to(ROOT)))
        if (base/'allocation/amendment.json').exists():cfg['devices']=read(base/'allocation/amendment.json')['current_pool']
        train=list(range(10000+index*100,10012+index*100))
        cfg['normalization'].update(directory='data/exp2/scaleup/'+name+'/demonstrations',
            train_ids=['demo'+str(s) for s in train])
        cfg['segmentation'].pop('previous_dataset')
        cfg['evaluation'].update(output=str((root/'evaluation').relative_to(ROOT)),diagnostic_seeds=[],
            seeds=list(range(20000+index*100,20030+index*100)),
            ood_seeds=list(range(30000+index*100,30030+index*100)))
        config_path=configs/(name+'.json')
        if config_path.exists() and read(config_path)!=cfg:raise ValueError('Study configuration already exists')
        atomic(config_path,cfg)
        rows.append(dict(task_id=name,config=str(config_path.relative_to(ROOT)),training_seeds=train,
            ID=[dict(seed=s,layout=layout(spec,s,'ID')) for s in cfg['evaluation']['seeds']],
            OOD=[dict(seed=s,layout=layout(spec,s,'OOD')) for s in cfg['evaluation']['ood_seeds']]))
    atomic(base/'preparation.json',dict(schema='appl.scaleup.preparation.v1',tasks=rows,
        owners=dict(tasks='developer',demonstrations='developer_fixed_motion_planner',
            segmentation='Runtime_API',heuristics='Runtime_API',prior_source='Runtime_API',inference='Runtime_API'),
        evaluation=dict(conditions=['ID','OOD'],trials_per_condition=30,methods=['naive_DP','APPL'],
            max_physical_steps=5000,episode_budget_visible_to_API=False),
        original_drawer='Reuse frozen M0 and M1_v2 models; new paired tests with the relaxed geometric goals.',
        state='Preparing and validating environments; no learned-policy test results available.'))
    return rows


if __name__=='__main__':initialize()
