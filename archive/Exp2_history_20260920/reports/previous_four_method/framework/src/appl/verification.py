"""Fixed learned-policy verification suite in the frozen API reconstruction."""
import os
import time
import numpy as np
from pathlib import Path
from .io import read,atomic,event,digest
from .protocol import ROLES
from .surrogate import Simulator
from .envs.evaluator import measure,SuccessTracker
from .worker import PolicyProcess


def model_path(cfg,output):
    base=cfg['output']/'reconstruction';submitted=read(base/'submission.json')
    if not submitted['calibration_passed']:raise ValueError('Uncalibrated model cannot validate learned policies')
    last=submitted['latest_calibration'];folder=base/'calibration'/last['calibration_id']
    if digest(base/'versions'/submitted['version']/'model.xml')!=submitted['version']:raise ValueError('Frozen API model changed')
    # Re-materialize the exact hash-frozen XML instead of trusting a mutable
    # historical compiled-path file. Asset hashes are checked by materialize.
    from .model_contract import materialize,audit_model
    from .io import ROOT
    import mujoco
    request=read(folder/'worker_request.json')
    path=Path(output)/'resolved.xml'
    # Historical receipts retain their original storage paths. Load relocated
    # assets from the canonical data directory, checking the same frozen hashes.
    assets=(ROOT/'data/exp2/assets').resolve()
    materialize(base/'versions'/submitted['version']/'model.xml',assets,request['asset_manifest'],path)
    audit_model(mujoco.MjModel.from_xml_path(str(path)),read(ROOT/'data/exp2/reconstruction_public/interface.json'))
    return path


def run(cfg,request):
    output=Path(request['output']);policies=request['policies'];gpu=int(os.environ['APPL_PHYSICAL_GPU'])
    sim=Simulator(model_path(cfg,output));results=[]
    # Native physical bounds, fixed at the successful D0 controller audit.
    control=read(cfg['output']/'diagnostics/D0/controller.json')
    low=np.asarray(control['action_low']);high=np.asarray(control['action_high'])
    cases=[]
    for index,policy in enumerate(policies):
        for identifier in ('demo1000','demo1007'):
            cases.append(dict(name=f'skill_{index}_{identifier}',identifier=identifier,
                start=policy['metadata']['segments'][identifier][0],sequence=[index],full=False))
    if set(p['metadata']['skill'] for p in policies)==set(ROLES):
        primary=[next(i for i,p in enumerate(policies) if p['metadata']['skill']==role) for role in ROLES]
        for identifier in ('demo1000','demo1007'):
            cases.append(dict(name='full_'+identifier,identifier=identifier,start=0,sequence=primary,full=True))
    elif request.get('scope')!='framework_capability':raise ValueError('Full library verification requires all three subgoals')
    atomic(output/'suite.json',dict(cases=cases,scope='Original training-calibration states only; no target queries',
        reset_limitation='Recorded generalized states; no original native contact solver state'))
    for case in cases:
        directory=output/case['name']
        if (directory/'result.json').exists():results.append(read(directory/'result.json'));continue
        if directory.exists():raise RuntimeError('Partial MuJoCo execution requires explicit interrupted-attempt accounting')
        directory.mkdir();t=read(cfg['data']['directory']/(case['identifier']+'.json'));start=case['start']
        state=sim.reset(t['observations'][start]['state'],t['observations'][start-1]['state'] if start else None)
        tracker=SuccessTracker(cfg['evaluation']);total=0;started=time.monotonic();switches=[];inference=0.;chunks=0
        for selected_index in case['sequence']:
            selected=policies[selected_index];role=selected['metadata']['skill']
            if digest(selected['checkpoint'])!=selected['checkpoint_sha256']:raise ValueError('Verification checkpoint changed')
            policy=PolicyProcess(cfg,selected['candidate'],selected['checkpoint'],directory/('worker_'+str(selected_index)),gpu,int(case['identifier'][4:]))
            status='timeout';switches.append(dict(step=total,policy=selected['metadata']['policy_id'],skill=role))
            try:
                for _ in range(min(600,cfg['evaluation']['max_steps']-total)):
                    raw=np.asarray(policy.action(state),np.float32)
                    if raw.shape!=(8,) or not np.isfinite(raw).all():raise ValueError('Invalid action in verification')
                    action=np.clip(raw,low,high);state=sim.step(action);total+=1
                    metrics=tracker.update(measure(None,state,cfg['evaluation'],sim.speeds()))
                    event(directory/'trace.jsonl','step',step=total,state=state,action=action.tolist(),raw_action=raw.tolist(),metrics=metrics,skill=role)
                    if tracker.skill_succeeded(role):status='succeeded';break
                switches[-1].update(status=status,end_step=total)
                inference+=sum(policy.timings);chunks+=len(policy.timings)
            finally:policy.close()
            if status!='succeeded':break
        success=bool(tracker.last.get('success',False)) if case['full'] else status=='succeeded'
        result=dict(case=case,success=success,steps=total,final=tracker.last,switches=switches,
            inference_chunks=chunks,inference_seconds=inference,elapsed_seconds=time.monotonic()-started,
            model_version=read(cfg['output']/'reconstruction/submission.json')['version'])
        atomic(directory/'result.json',result);results.append(result)
    result=dict(success=all(r['success'] for r in results),round=request['round'],cases=results,
        physical_steps=sum(r['steps'] for r in results),target_calls=0,training_data_added=0)
    atomic(output/'result.json',result);return result
