import time
import copy
from pathlib import Path
import numpy as np
from relative_dp.utils import ROOT, read_json, atomic_json, sha256, object_hash
from relative_dp.environment import initial_state_hash
from .calibrate import record, rollout
from .geometry import GeometryEnv

SPLITS = ['dev','test_iid','test_position','test_yaw_near','test_yaw_mid','test_yaw_far','test_combined']


def collect():
    from .learning import code_hash
    path=ROOT/'data/round2/manifest.json'
    if path.exists():
        manifest=read_json(path)
        for group in manifest['tasks'].values():
            for records in group.values():
                if isinstance(records,list):
                    for r in records:
                        assert sha256(ROOT/r['path'])==r['sha256']
        audit_path=ROOT/'artifacts/round2/data_audit.json'
        if not audit_path.exists() or read_json(audit_path)['manifest_hash']!=sha256(path):audit()
        return manifest
    started=time.monotonic()
    cal=read_json(ROOT/'configs/round2/calibration.json')
    manifest=dict(protocol_version='round2-v1',calibration_hash=sha256(ROOT/'configs/round2/calibration.json'),
                  environment_expert_code_hash=code_hash(),tasks={})
    attempts=[]
    for ti,task in enumerate(('drawer','door')):
        chosen=cal['tasks'][task]
        if chosen.get('blocked'):
            manifest['tasks'][task]=dict(blocked=True)
            continue
        env=GeometryEnv(task)
        angles=chosen['angles']
        group={'train20':[]}
        try:
            for bin_id in range(4):
                rng=np.random.default_rng(24000000+ti*100000+bin_id*1000)
                for attempt in range(100):
                    if len(group['train20']) >= (bin_id+1)*5:
                        break
                    seed=24000000+ti*100000+bin_id*1000+attempt
                    yaw=rng.uniform(-angles[0]+bin_id*angles[0]/2,-angles[0]+(bin_id+1)*angles[0]/2)
                    rec=record(task,seed,yaw,episode_id=f'r2_{task}_train_b{bin_id}_a{attempt:03d}')
                    rec['yaw_bin']=bin_id
                    result,arrays=rollout(env,rec,store=True)
                    attempts.append(copy.deepcopy(result))
                    atomic_json(ROOT/'artifacts/round2/collection_attempts.json',attempts)
                    if not result['success'] or result['invalid_joint']:
                        continue
                    p=ROOT/f'data/round2/{task}/train20/{rec["episode_id"]}.npz'
                    p.parent.mkdir(parents=True,exist_ok=True)
                    if p.exists():
                        with np.load(p) as old:
                            for key in arrays:np.testing.assert_array_equal(arrays[key],old[key])
                    else:np.savez_compressed(p,**arrays)
                    rec.update(path=str(p.relative_to(ROOT)),sha256=sha256(p),expert={k:v for k,v in result.items() if k!='record'},
                               initial_state_hash=initial_state_hash(env.snapshot(env.reset(rec)[0])) )
                    group['train20'].append(rec)
                    print('COLLECT',rec['episode_id'],len(group['train20']),result['steps'],flush=True)
                if len(group['train20'])!=(bin_id+1)*5:
                    raise RuntimeError(f'{task} bin {bin_id} did not yield 5 successes in 100 attempts')
            # Pre-generate every dev/test record before any policy is trained.
            for si,split in enumerate(SPLITS):
                records=[]
                rng=np.random.default_rng(25000000+ti*100000+si*1000)
                for i in range(20):
                    seed=25000000+ti*100000+si*1000+i
                    region='train'
                    if split=='test_position':region='left' if i<10 else 'right'
                    if split=='test_combined':region='left' if i//5 in (0,1) else 'right'
                    if split in ('dev','test_iid','test_position'):
                        yaw=rng.uniform(-angles[0],angles[0])
                    else:
                        mag=angles[{'test_yaw_near':1,'test_yaw_mid':2,'test_yaw_far':3,'test_combined':3}[split]]
                        sign=(-1 if i<10 else 1) if split!='test_combined' else (-1 if i//5 in (0,2) else 1)
                        yaw=sign*mag+rng.uniform(-2,2)
                    rec=record(task,seed,yaw,region,episode_id=f'r2_{task}_{split}_{i:03d}')
                    rec.update(split=split,policy_noise_seed=26000000+ti*100000+si*1000+i)
                    records.append(rec)
                group[split]=records
            atomic_json(ROOT/f'data/round2/{task}/eval_plan.json',{s:group[s] for s in SPLITS})
            for split in SPLITS:
                for rec in group[split]:
                    obs,info=env.reset(rec)
                    snap=env.snapshot(obs)
                    assert not info['success'] and abs(info['progress'])<1e-12
                    p=ROOT/f'data/round2/{task}/{split}/{rec["episode_id"]}.npz'
                    p.parent.mkdir(parents=True,exist_ok=True)
                    if p.exists():
                        with np.load(p) as old:
                            for key in snap:np.testing.assert_array_equal(snap[key],old[key])
                    else:np.savez_compressed(p,**snap)
                    rec.update(path=str(p.relative_to(ROOT)),sha256=sha256(p),initial_state_hash=initial_state_hash(snap),
                               initial_handle_pose=obs[4:11].tolist(),actual_base_position=env.base.tolist(),
                               actual_base_quat=env.data.body(task).xquat.tolist(),goal_pose=np.r_[env.goal,env.goal_quat].tolist())
                print('FROZEN',task,split,len(group[split]),flush=True)
            manifest['tasks'][task]=group
        finally:env.close()
    manifest['collection_wall_seconds']=time.monotonic()-started
    atomic_json(path,manifest)
    audit()
    return manifest


def audit():
    manifest=read_json(ROOT/'data/round2/manifest.json')
    results=[]; started=time.monotonic()
    seeds=[]
    for task,group in manifest['tasks'].items():
        if group.get('blocked'):continue
        env=GeometryEnv(task)
        try:
            for split in ['train20']+SPLITS:
                for rec in group[split]:
                    seeds.append(rec['seed'])
                    with np.load(ROOT/rec['path'],allow_pickle=False) as f:
                        obs,info=env.reset(rec)
                        snap=env.snapshot(obs)
                        for key in snap:np.testing.assert_allclose(snap[key],f[key],atol=1e-9,rtol=0)
                        maxerror=0.
                        if split=='train20':
                            np.testing.assert_allclose(obs,f['obs'][0],atol=1e-6,rtol=0)
                            for t,action in enumerate(f['actions']):
                                obs,_,_,_,info=env.step(action)
                                error=float(np.max(np.abs(obs-f['obs'][t+1])))
                                maxerror=max(maxerror,error)
                                np.testing.assert_allclose(obs,f['obs'][t+1],atol=1e-6,rtol=0)
                            assert info['success']
                        results.append(dict(episode_id=rec['episode_id'],split=split,replay_steps=len(f['actions']) if split=='train20' else 0,max_error=maxerror))
        finally:env.close()
    assert len(seeds)==len(set(seeds)), 'No base reset seed can cross splits'
    result=dict(passed=True,manifest_hash=sha256(ROOT/'data/round2/manifest.json'),
                full_demo_replays=sum(r['split']=='train20' for r in results),
                snapshot_checks=len(results),records=results,wall_seconds=time.monotonic()-started)
    atomic_json(ROOT/'artifacts/round2/data_audit.json',result)
    return result


if __name__=='__main__':collect()
