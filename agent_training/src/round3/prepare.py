"""Calibrate real task resets, collect nested demonstrations, and seal states.

This module is infrastructure, not the design agent. Designers read only the
independent D2 evidence directory. Evaluation state values never enter it.
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
import numpy as np
from relative_dp.utils import ROOT,atomic_json,read_json,sha256,object_hash
from relative_dp.environment import initial_state_hash,METAWORLD_COMMIT
from .tasks import TASKS,NATIVE_IDS,SOURCE_FILES,distribution,make_record
from .environment import TaskEnv,make_expert,observation_schema

BASE=ROOT/'round3'
def jsonable(value):
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value

def save_npz(path,arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.partial.npz')
    np.savez_compressed(tmp,**arrays)
    os.replace(tmp,path)

def episode(env,rec,store=False):
    start=time.monotonic();obs,info=env.reset(rec)
    snap=env.snapshot(obs);expert=make_expert(env)
    observations=[obs.copy()];actions=[];rewards=[];infos=[info];terms=[];truncs=[]
    error=None;reason='timeout'
    for _ in range(500):
        try:
            before=obs.copy()
            action=np.clip(expert.action(obs.copy()),-1,1).astype(np.float32)
            np.testing.assert_array_equal(obs,before)
            obs,reward,term,trunc,info=env.step(action.copy())
            actions.append(action);observations.append(obs.copy());rewards.append(reward);infos.append(info);terms.append(term);truncs.append(trunc)
            if info['success']:reason='success';break
            if term:reason='native_terminated';break
            if trunc:reason='native_truncated';break
        except Exception as exc:error=repr(exc);reason='simulation_exception';break
    invalid=any(not i.get('valid_joint',True) for i in infos)
    result=dict(episode_id=rec['episode_id'],record=rec,success=bool(info['success']) and not invalid and error is None,steps=len(actions),termination_reason='invalid_joint' if invalid else reason,invalid_joint=invalid,exception=error,wall_seconds=time.monotonic()-start,final_info=jsonable(info))
    if not store:return result
    arrays=dict(obs=np.asarray(observations,np.float32),actions=np.asarray(actions,np.float32),rewards=np.asarray(rewards,np.float32),success=np.asarray([bool(i['success']) for i in infos[1:]],bool),valid_joint=np.asarray([i.get('valid_joint',True) for i in infos],bool),terminated=np.asarray(terms,bool),truncated=np.asarray(truncs,bool),**snap)
    for key in ('q','progress','distance','direct_contact','obj_to_target','grasp_success'):
        if key in infos[0]:arrays[key]=np.asarray([i.get(key,0) for i in infos])
    return result,arrays

def static_knowledge(env):
    m=env.model;d=env.data
    if env.task=='pick-place-wall':return dict(wall_center=d.body('wall').xpos.tolist(),wall_half_size=[.12,.01,.06],object_cylinder_radius=.02,object_cylinder_halfheight=.02,gravity=m.opt.gravity.tolist(),symmetry_caution='Fixed robot, gravity, table, and wall prevent claiming full environment rotation equivariance.')
    if env.task=='assembly':return dict(task='Lift the ring-shaped nut and place it over a fixed vertical peg.',peg_radius=.02,peg_halfheight=.05,peg_axis=[0,0,1],ring_center_observed=True,nut_quaternion='native wxyz',symmetry_caution='Fixed robot and gravity remain in world coordinates.')
    if env.task=='peg-insert-side':return dict(task='Grasp the peg and insert its head into the side opening of a fixed box.',insertion_axis=[1,0,0],box_position_from_goal='goal - [.03,0,.13] metres',peg_head_observed=True,symmetry_caution='Fixed robot and gravity; insertion axis is fixed by native geometry.')
    if env.task=='stick-push':return dict(task='Pick up the stick and push the separate container toward the target.',pushed_object_initial_position='native fixed; independent factors are tool x and target y',object2_quaternion='absent zero placeholder',symmetry_caution='Fixed robot, gravity, container start, and tool geometry.')
    return dict(task='Open the mechanism with fixed end-effector orientation.',joint_type='prismatic' if env.task=='drawer' else 'hinge',cabinet_yaw_from_shared_fields=[39,41],gravity=m.opt.gravity.tolist(),mechanism_axis='cabinet-local y' if env.task=='drawer' else 'world z',joint_range=env.geometry.model.jnt_range[env.geometry.joint].tolist(),rear_rail_repair='Round2 unchanged: retaining rail y=1.15 for both tasks/all yaw.',symmetry_caution='Only cabinet rotates; robot, gravity and table remain fixed.')

def task_manifest(task,env):
    import metaworld
    source=Path(metaworld.__file__).parent/'envs'/SOURCE_FILES[task]
    success={
      'pick-place-wall':'Native Euclidean distance between objGeom center and target <= .07 m.',
      'assembly':'Native _reward_pos: xy distance from RoundNut ring-center site to target < .02 m AND target.z - ring_center.z > 0. There is no additional grasp or hold condition in native success.',
      'peg-insert-side':'Native scaled Euclidean distance between pegHead and target with axis weights [1,2,2] <= .07 m.',
      'stick-push':'Native conjunction: touching tool, aperture >0, tool lifted >.01 m above initial z, and container observation point within .12 m of target.',
      'drawer':'Round2: progress in [.75,1.01] for 3 successive steps, with no progress outside [-.01,1.01] anywhere in episode.',
      'door':'Round2: progress in [.75,1.01] for 3 successive steps, with no progress outside [-.01,1.01] anywhere in episode.',
    }[task]
    return dict(task=task,native_id=NATIVE_IDS[task],version='round3-task-v1',metaworld_commit=METAWORLD_COMMIT,source_file=str(source),source_sha256=sha256(source),source_url=f'https://github.com/Farama-Foundation/Metaworld/blob/{METAWORLD_COMMIT}/metaworld/envs/{source.name}',source_knowledge=static_knowledge(env),distribution=distribution(task),observation_schema=observation_schema(task),action_schema=dict(dim=4,fields=['world_dx','world_dy','world_dz','gripper'],low=[-1]*4,high=[1]*4,native_xyz_scale_m=float(env.native.action_scale),control_dt_seconds=float(env.native.dt),end_effector_rotation='fixed native orientation',world_clip='after policy action decode, before env.step'),success_definition=success,success_hold_steps=3 if task in ('drawer','door') else 1,max_steps=500,termination_policy='Check success after every env.step; stop at first success, native terminated or truncated, or 500 steps. Timeout alone is failure. Numerical exceptions are failure. Any joint validity violation makes the full cabinet episode failure.',reset_method='Canonical native Task+seed reset; fixed body placement precompiled to native requested position for assembly/peg box, no shape/dynamics changes.',expert_version=dict(backend='Round2 CanonicalExpert' if env.geometry else 'official pinned MetaWorld scripted policy',commit=METAWORLD_COMMIT,wrapper_sha256=sha256(Path(__file__).with_name('environment.py'))),static_geometry_source=dict(xml=env.native.model_name,sha256=sha256(Path(env.native.model_name))),stats_scope='Only the current D_N observations/actions; never larger subsets.',seeds=dict(train_base=31000000+TASKS.index(task)*100000,calibration_base=30000000+TASKS.index(task)*100000,dev_base=32000000+TASKS.index(task)*100000,test_base='sealed, independent'),eval_counts=dict(dev=dict(IID=10,C=20,E=20),test=dict(IID=20,C=40,E=40)))

def calibrate(task,env):
    path=BASE/'calibration'/task/'results.json';path.parent.mkdir(parents=True,exist_ok=True)
    results=read_json(path) if path.exists() else []
    done={r['record']['episode_id'] for r in results}
    start=time.monotonic()
    for si,split in enumerate(('IID','C','E')):
        for i in range(20):
            rec=make_record(task,30000000+TASKS.index(task)*100000+si*1000+i,split,i,'calibration')
            if rec['episode_id'] in done:continue
            o,info=env.reset(rec);before=o.copy();zero_success=False
            for _ in range(20):
                o,_,term,trunc,info=env.step(np.zeros(4,np.float32));zero_success|=bool(info['success'])
                if term or trunc:break
            result=episode(env,rec)
            result['zero_action']=dict(steps=20,became_successful=zero_success,object_displacement=float(np.linalg.norm(o[4:7]-before[4:7])),finite=bool(np.isfinite(o).all()),final_valid_joint=info.get('valid_joint',True))
            assert not zero_success,'Spontaneous success needs calibration review'
            results.append(result);atomic_json(path,jsonable(results))
        print('CALIBRATION',task,split,sum(r['success'] for r in results if r['record']['split']==split),'/20',flush=True)
    summary=dict(task=task,counts={s:dict(episodes=sum(r['record']['split']==s for r in results),expert_success=sum(r['record']['split']==s and r['success'] for r in results)) for s in ('IID','C','E')},independent_of_models=True,expert_failure_not_impossibility=True,zero_action_checks=60,source_audit_required=True,wall_seconds_this_call=time.monotonic()-start,records_hash=sha256(path))
    atomic_json(path.with_name('summary.json'),summary)
    return summary

def build_evidence(task,records,env):
    out=BASE/'evidence'/task
    if (out/'bundle.json').exists():return read_json(out/'bundle.json')
    out.mkdir(parents=True,exist_ok=True)
    manifest=read_json(BASE/'protocol_and_task_manifests'/f'{task}.json')
    atomic_json(out/'task_materials.json',{k:manifest[k] for k in ('task','native_id','source_url','source_knowledge','distribution','observation_schema','action_schema','static_geometry_source')})
    summaries=[];image_records=[]
    from PIL import Image,ImageDraw
    render_env=TaskEnv(task,render_mode='rgb_array')
    try:
        for rank,rec in enumerate(records[:2]):
            with np.load(ROOT/rec['path']) as f:obs=f['obs'].copy();actions=f['actions'].copy()
            n=len(actions);indices=np.rint(np.linspace(0,n,8)).astype(int)
            numeric=out/f'demo_{rank+1}_trajectory.npz';save_npz(numeric,dict(obs=obs,actions=actions,time_seconds=np.arange(n+1)*env.native.dt))
            velocity=np.diff(obs[:,:3],axis=0)/env.native.dt
            summaries.append(dict(episode_id=rec['episode_id'],transitions=n,layout_cell=rec['cell'],factors=rec['factors'],trajectory_file=numeric.name,observation_min=obs.min(0).tolist(),observation_max=obs.max(0).tolist(),action_min=actions.min(0).tolist(),action_max=actions.max(0).tolist(),hand_speed_mps=dict(mean=float(np.linalg.norm(velocity,axis=1).mean()),max=float(np.linalg.norm(velocity,axis=1).max())),sampled_states=[dict(step=int(t),time_seconds=float(t*env.native.dt),obs=obs[t].tolist(),action=actions[t].tolist() if t<n else None) for t in indices]))
            current,_=render_env.reset(rec)
            for t in range(n+1):
                if t in indices:
                    np.testing.assert_allclose(current,obs[t],atol=1e-6,rtol=0)
                    frame=np.flipud(render_env.native.render()).copy()
                    im=Image.fromarray(frame);draw=ImageDraw.Draw(im)
                    label=f'D2 demo {rank+1} | {task} | step {t}/{n} | {t*env.native.dt:.3f}s'
                    draw.rectangle((0,0,480,25),fill='black');draw.text((6,6),label,fill='white')
                    name=f'demo_{rank+1}_frame_{list(indices).index(t):02d}_step_{t:03d}.png';im.save(out/name)
                    image_records.append(dict(file=name,episode_id=rec['episode_id'],step=t,time_seconds=t*env.native.dt,camera='corner2',display_flipud=True))
                if t<n:current,*_=render_env.step(actions[t])
    finally:render_env.close()
    atomic_json(out/'D2_summary.json',summaries)
    atomic_json(out/'frames.json',image_records)
    (out/'DESIGN_INTERFACE.md').write_text('''You are the isolated Codex design agent. Read only this evidence directory and the parent-provided design specification. Do not inspect experts, source reward rules, larger demonstration subsets, old results, or concrete development/test states.\n\nThe policy is state-input. Raw deployment observation/action schema and static physical geometry are in task_materials.json. D2_summary.json plus both trajectory NPZ files contain exactly the first two whole successful demonstrations, one LL and one HH. Each demonstration has 8 uniform real simulator frames. Images inform design; policy actions use state only.\n\nPropose exactly three meaningfully different candidates P1/P2/P3 and seriously consider a combination without forcing one. All primary policies use diffusion action learning, whole-system 20,000 updates, batch128 total chunks/update, 2-observation history, prediction16/execution4 default; widths64/128/256,time embedding128,kernel5,groups8, AdamW lr1e-4 wd1e-6, epsilon DDPM100 cosine, DDIM16 eta0, EMA.995. Auxiliary heads may use this same batch. Total parameters <=3 times B0. No expert calls, new demonstrations, runtime success/future/hidden phases, geometric augmentation that changes fixed robot/gravity/obstacle physics, or extra controller capabilities. Dynamic frames must fix the transform at the current replan observation throughout the chunk, decode physical xyz back to world, then clip the native world action cube.\n\nImplementation interfaces: observation_transform(raw history), action_encode(world chunks,current observation), action_decode(encoded chunk,current observation), training_supervision(current D_N only), optional trainable torch encoder/auxiliary head/router. Describe needed new interfaces precisely if these are insufficient. Loss targets, scales, weights, applicability, routing/termination, and chunk transition handling must be exact. Save proposal before implementation. For every knowledge hypothesis label source external geometry, observed D2, or unverified inference.\n''')
    files={str(p.relative_to(out)):sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='bundle.json'}
    bundle=dict(task=task,backend='codex_session',demo_count=2,episode_ids=[r['episode_id'] for r in records[:2]],frame_count=len(image_records),contains_expert_source=False,contains_reward_rules=False,contains_dev_test_states=False,contains_larger_datasets=False,files=files,hash=object_hash(files),created_unix=time.time())
    assert len(image_records)==16
    atomic_json(out/'bundle.json',bundle)
    print('D2_EVIDENCE_READY',task,str(out),flush=True)
    return bundle

def prepare_task(task):
    start=time.monotonic();base=BASE/'data'/task;base.mkdir(parents=True,exist_ok=True)
    path=base/'manifest.json'
    manifest=read_json(path) if path.exists() else dict(task=task,version='round3-data-v1',train20=[],dev={},complete=False)
    if manifest.get('complete'):
        for rec in manifest['train20']+[r for rs in manifest['dev'].values() for r in rs]:assert sha256(ROOT/rec['path'])==rec['sha256']
        return manifest
    env=TaskEnv(task)
    try:
        probe=make_record(task,30000000+TASKS.index(task)*100000,'IID',0,'calibration')
        env.reset(probe)
        mp=BASE/'protocol_and_task_manifests'/f'{task}.json'
        definition=task_manifest(task,env)
        if not mp.exists():atomic_json(mp,definition)
        else:
            previous=read_json(mp)
            for key in ('distribution','observation_schema','success_definition'):assert previous[key]==definition[key],f'Frozen task changed: {key}'
        calibration=calibrate(task,env)
        attempts_path=base/'collection_attempts.json'
        attempts=read_json(attempts_path) if attempts_path.exists() else []
        for i in range(len(manifest['train20']),20):
            for attempt in range(100):
                seed=31000000+TASKS.index(task)*100000+i*100+attempt
                rec=make_record(task,seed,'IID',i,'train')
                old=next((r for r in attempts if r['episode_id']==rec['episode_id']),None)
                if old and not old['success']:continue
                result,arrays=episode(env,rec,store=True)
                if old is None:attempts.append(result);atomic_json(attempts_path,jsonable(attempts))
                if not result['success']:continue
                p=base/'train20'/f'{rec["episode_id"]}.npz';save_npz(p,arrays)
                rec.update(path=str(p.relative_to(ROOT)),sha256=sha256(p),transitions=len(arrays['actions']),expert={k:v for k,v in result.items() if k!='record'},initial_state_hash=initial_state_hash({k:v for k,v in arrays.items() if k.startswith('snapshot_')}))
                manifest['train20'].append(rec);atomic_json(path,jsonable(manifest))
                print('COLLECT',task,i+1,rec['cell'],len(arrays['actions']),flush=True)
                if i==1:build_evidence(task,manifest['train20'],env)
                break
            else:raise RuntimeError(f'{task} training cell {i} failed 100 expert attempts; calibration investigation required')
        build_evidence(task,manifest['train20'],env)
        # Test values are kept outside both evidence and public train/dev manifest.
        sealed_path=BASE/'locked_test_states'/task/'manifest.json'
        sealed=read_json(sealed_path) if sealed_path.exists() else dict(task=task,test={},sealed=True)
        for phase,counts,target,rootdir in [('dev',dict(IID=10,C=20,E=20),manifest['dev'],base/'dev'),('test',dict(IID=20,C=40,E=40),sealed['test'],sealed_path.parent)]:
            for si,(split,count) in enumerate(counts.items()):
                if len(target.get(split,[]))==count:continue
                target[split]=[]
                for i in range(count):
                    seed=(32000000 if phase=='dev' else 33000000)+TASKS.index(task)*100000+si*1000+i
                    rec=make_record(task,seed,split,i,phase)
                    obs,info=env.reset(rec);snap=env.snapshot(obs)
                    p=rootdir/split/f'{rec["episode_id"]}.npz';save_npz(p,snap)
                    rec.update(path=str(p.relative_to(ROOT)),sha256=sha256(p),initial_state_hash=initial_state_hash(snap))
                    target[split].append(rec)
                atomic_json(path if phase=='dev' else sealed_path,jsonable(manifest if phase=='dev' else sealed))
                print('FROZEN_STATES',task,phase,split,count,flush=True)
        replay=[]
        for rec in manifest['train20']:
            with np.load(ROOT/rec['path']) as f:
                obs,info=env.reset(rec)
                for key,value in env.snapshot(obs).items():np.testing.assert_allclose(value,f[key],atol=1e-9,rtol=0,err_msg=key)
                maxerr=0.;invalid=False
                np.testing.assert_allclose(obs,f['obs'][0],atol=1e-6,rtol=0)
                for t,action in enumerate(f['actions']):
                    obs,_,_,_,info=env.step(action);invalid|=not info.get('valid_joint',True)
                    maxerr=max(maxerr,float(np.max(np.abs(obs-f['obs'][t+1]))))
                    np.testing.assert_allclose(obs,f['obs'][t+1],atol=1e-6,rtol=0)
                assert info['success'] and not invalid
                replay.append(dict(episode_id=rec['episode_id'],steps=len(f['actions']),max_error=maxerr,passed=True))
        atomic_json(base/'replay_audit.json',dict(passed=True,full_demos=20,records=replay))
        manifest.update(complete=True,calibration_hash=sha256(BASE/'calibration'/task/'results.json'),task_manifest_hash=sha256(mp),evidence_bundle_hash=sha256(BASE/'evidence'/task/'bundle.json'),sealed_test_manifest_hash=sha256(sealed_path),attempts=len(attempts),failed_attempts=sum(not r['success'] for r in attempts),subsets={str(n):dict(episode_ids=[r['episode_id'] for r in manifest['train20'][:n]],transitions=sum(r['transitions'] for r in manifest['train20'][:n]),cell_counts={c:sum(r['cell']==c for r in manifest['train20'][:n]) for c in ('LL','HH')}) for n in (2,5,10,20)},wall_seconds=time.monotonic()-start)
        atomic_json(path,jsonable(manifest));print('TASK_DATA_READY',task,'20 demos; 50 dev; 100 sealed test; all demos replayed',flush=True)
        return manifest
    finally:env.close()

def prepare(tasks=None):return {task:prepare_task(task) for task in (tasks or TASKS)}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=TASKS,action='append');args=parser.parse_args();prepare(args.task)
