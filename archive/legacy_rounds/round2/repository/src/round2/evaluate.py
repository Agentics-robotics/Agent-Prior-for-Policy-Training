import json
import time
import traceback
import numpy as np
import torch
from relative_dp.utils import ROOT, read_json, atomic_json, sha256, object_hash
from .geometry import GeometryEnv
from .learning import Loaded, RUNS
from .collect import SPLITS


def implementation_hash():
    from .learning import code_hash
    return object_hash(dict(core=code_hash(),evaluation=sha256(__file__)))


def run_episode(env, loaded, rec):
    start=time.monotonic()
    observations=[]; actions=[]; infos=[]; clipping=[]; exception=None
    try:
        obs,info=env.reset(rec)
        with np.load(ROOT/rec['path'],allow_pickle=False) as f:
            for key,value in env.snapshot(obs).items():
                np.testing.assert_allclose(value,f[key],atol=1e-9,rtol=0)
        observations.append(obs);infos.append(info)
        generator=torch.Generator(device='cuda').manual_seed(rec['policy_noise_seed'])
        for step in range(0,500,4):
            history=np.asarray([observations[max(0,len(observations)-2)],observations[-1]],np.float32)
            chunk=loaded.actions(history,generator)
            assert chunk.shape==(16,4) and np.isfinite(chunk).all()
            for proposed in chunk[:4]:
                clipping.append(np.abs(proposed)>1.)
                actual=np.clip(proposed,-1.,1.).astype(np.float32)
                obs,_,_,_,info=env.step(actual.copy())
                actions.append(actual);observations.append(obs);infos.append(info)
                if info['success']:break
            if info['success']:break
    except Exception:
        exception=traceback.format_exc()
    success=bool(infos and infos[-1]['success'] and exception is None and all(i['valid_joint'] for i in infos))
    first=lambda key:next((i for i,x in enumerate(infos) if x[key]),None)
    progress=[x['progress'] for x in infos]
    distance=[x['distance'] for x in infos]
    contact=any(x['direct_contact'] for x in infos)
    max_progress=max(progress,default=0.)
    if success:category='success'
    elif min(distance,default=float('inf'))>.08:category='no_approach'
    elif not contact:category='approach_no_contact'
    elif max_progress<.1:category='contact_no_progress'
    else:category='progress_not_completed'
    counts=np.asarray(clipping,bool)
    result=dict(episode_id=rec['episode_id'],split=rec['split'],success=success,steps=len(actions),
                first_success_step=first('success') if success else None,
                min_hand_handle_distance=min(distance) if distance else None,
                direct_contact=contact,first_contact_step=first('direct_contact'),
                first_opening_step=next((i for i,p in enumerate(progress) if p>=.1),None),
                opening_started=max_progress>=.1,max_progress=max_progress,
                final_progress=progress[-1] if progress else None, failure_category=category,
                invalid_joint=any(not i['valid_joint'] for i in infos),
                clipped_action_steps=int(np.any(counts,axis=-1).sum()) if len(counts) else 0,
                clipped_coordinates=counts.sum(0).tolist() if len(counts) else [0]*4,
                evaluated_action_steps=len(counts),exception=exception,
                yaw_degrees=rec['task_params']['yaw_degrees'],region=rec['region'],
                initial_state_hash=rec['initial_state_hash'],wall_seconds=time.monotonic()-start)
    arrays=dict(obs=np.asarray(observations,np.float32),actions=np.asarray(actions,np.float32).reshape(-1,4),
                progress=np.asarray(progress),distance=np.asarray(distance),
                direct_contact=np.asarray([i['direct_contact'] for i in infos],bool),clipping=counts)
    return result,arrays


def metrics(records):
    success=[r for r in records if r['success']]
    final_progress=[r['final_progress'] for r in records if r['final_progress'] is not None]
    return dict(n=len(records),successes=len(success),success_rate=len(success)/len(records),
        mean_first_success_step=float(np.mean([r['first_success_step'] for r in success])) if success else None,
        mean_max_progress=float(np.mean([r['max_progress'] for r in records])),
        mean_final_progress=float(np.mean(final_progress)) if final_progress else None,
        contact_rate=float(np.mean([r['direct_contact'] for r in records])),
        opening_rate=float(np.mean([r['opening_started'] for r in records])),
        clipping_step_rate=sum(r['clipped_action_steps'] for r in records)/max(1,sum(r['evaluated_action_steps'] for r in records)),
        failure_categories={c:sum(r['failure_category']==c for r in records) for c in
            ['success','no_approach','approach_no_contact','contact_no_progress','progress_not_completed']},
        exceptions=sum(r['exception'] is not None for r in records),
        invalid_joint_episodes=sum(r['invalid_joint'] for r in records))


def worker(run_id,stage,step=None):
    run=next(r for r in RUNS if r['run_id']==run_id)
    manifest=read_json(ROOT/'data/round2/manifest.json')
    if stage=='test':
        selection=read_json(ROOT/'results/round2/selection.json')
        assert len(selection['runs'])==4
        step=selection['runs'][run_id]['step']
        for item in selection['runs'].values():assert sha256(ROOT/item['checkpoint_path'])==item['checkpoint_hash']
    checkpoint=ROOT/f'runs/round2/{run_id}/checkpoints/step_{step:06d}.pt'
    directory=ROOT/f'results/round2/{stage}/{run_id}/step_{step:06d}'
    directory.mkdir(parents=True,exist_ok=True)
    identity=dict(run_id=run_id,step=step,stage=stage,checkpoint_hash=sha256(checkpoint),
                  manifest_hash=sha256(ROOT/'data/round2/manifest.json'),implementation_hash=implementation_hash())
    if (directory/'complete.json').exists():
        complete=read_json(directory/'complete.json')
        assert complete['identity']==identity
        for r in complete['records']:assert sha256(ROOT/r['trajectory_path'])==r['trajectory_hash']
        return complete
    loaded=Loaded(checkpoint)
    env=GeometryEnv(run['task'])
    records=[];started=time.monotonic()
    try:
        splits=['dev'] if stage=='dev' else SPLITS[1:]
        for split in splits:
            for rec in manifest['tasks'][run['task']][split]:
                p=directory/f'{rec["episode_id"]}.json'
                if p.exists():
                    r=read_json(p);assert r['identity']==identity
                    assert sha256(ROOT/r['trajectory_path'])==r['trajectory_hash']
                else:
                    r,arrays=run_episode(env,loaded,rec)
                    npz=p.with_suffix('.npz');np.savez_compressed(npz,**arrays)
                    r.update(identity=identity,trajectory_path=str(npz.relative_to(ROOT)),trajectory_hash=sha256(npz))
                    atomic_json(p,r)
                records.append(r)
                print(stage,run_id,step,rec['episode_id'],r['success'],r['steps'],round(r['max_progress'],3),flush=True)
    finally:env.close()
    result=dict(identity=identity,records=records,metrics={s:metrics([r for r in records if r['split']==s]) for s in splits},
                last_session_wall_seconds=time.monotonic()-started,
                contact_geom_ids=env.handle_geoms,gripper_geom_ids=env.gripper_geoms)
    if stage=='test':
        yaw=[r for r in records if r['split'] in ('test_yaw_near','test_yaw_mid','test_yaw_far')]
        result['yaw_primary']=metrics(yaw)
        result['yaw_by_sign']={f'{s}:{sign}':metrics([r for r in records if r['split']==s and np.sign(r['yaw_degrees'])==sign])
                              for s in ('test_yaw_near','test_yaw_mid','test_yaw_far') for sign in (-1,1)}
    atomic_json(directory/'complete.json',result)
    return result


def freeze_selection():
    output=ROOT/'results/round2/selection.json'
    selection=dict(runs={},rule='highest dev successes, then lowest conditional mean first success, then earliest checkpoint',
                   manifest_hash=sha256(ROOT/'data/round2/manifest.json'))
    for run in RUNS:
        run_id=run['run_id'];candidates=[]
        for step in (5000,10000,20000):
            p=ROOT/f'results/round2/dev/{run_id}/step_{step:06d}/complete.json'
            result=read_json(p);m=result['metrics']['dev']
            assert m['n']==20
            candidates.append(dict(step=step,**m,dev_results_hash=sha256(p)))
        best=min(candidates,key=lambda c:(-c['successes'],c['mean_first_success_step'] if c['successes'] else float('inf'),c['step']))
        ckpt=ROOT/f'runs/round2/{run_id}/checkpoints/step_{best["step"]:06d}.pt'
        selection['runs'][run_id]=dict(step=best['step'],candidates=candidates,checkpoint_path=str(ckpt.relative_to(ROOT)),checkpoint_hash=sha256(ckpt))
    if output.exists():assert read_json(output)==selection
    else:atomic_json(output,selection)
    return selection


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('run_id');p.add_argument('stage',choices=['dev','test']);p.add_argument('--step',type=int)
    a=p.parse_args();worker(a.run_id,a.stage,a.step)
