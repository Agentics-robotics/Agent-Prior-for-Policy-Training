"""Artifact acceptance without optimizer updates or new learned-policy tests."""
import json
from pathlib import Path
import numpy as np
import torch
from relative_dp.utils import ROOT, read_json, sha256, atomic_json, object_hash
from round2.learning import code_hash, RUNS, config


def main():
    historical=read_json(ROOT/'artifacts/experiment_freeze.json')
    for path,expected in historical['file_hashes'].items():
        actual=ROOT/'artifacts/round2/round1_archive/pixi.toml' if path=='pixi.toml' else ROOT/path
        assert sha256(actual)==expected
    import metaworld
    resources=read_json(ROOT/'artifacts/round2/resources.json')
    source=Path(metaworld.__file__).parent
    assert resources['lock_hash']==sha256(ROOT/'pixi.lock')
    for path,expected in resources['files'].items():assert sha256(source/path)==expected
    from round2.evaluate import implementation_hash
    evaluation_hash=implementation_hash()
    frozen=read_json(ROOT/'configs/round2/freeze.json')
    manifest=read_json(ROOT/'data/round2/manifest.json')
    manifest_hash=sha256(ROOT/'data/round2/manifest.json')
    assert frozen['manifest_hash']==manifest_hash and frozen['code_hash']==code_hash()
    assert frozen['protocol_hash']==sha256(ROOT/'ROUND2_PROTOCOL.md')
    calibration=read_json(ROOT/'configs/round2/calibration.json')
    assert sha256(ROOT/'configs/round2/calibration.json')==manifest['calibration_hash']
    ledger=read_json(ROOT/'artifacts/round2/calibration_rollouts.json')
    for task in ('drawer','door'):
        assert sum(r['record']['task_name']==task for r in ledger)<=600
        assert calibration['tasks'][task]['feasible'] and calibration['tasks'][task]['tier']=='A'
    all_seeds=[];all_positions=[];data_files=0
    for task,group in manifest['tasks'].items():
        for split,records in group.items():
            assert len(records)==20
            if split=='train20':
                assert [sum(r['yaw_bin']==b for r in records) for b in range(4)]==[5]*4
            if split in ('test_position','test_combined'):
                assert sum(r['region']=='left' for r in records)==10
                assert sum(r['region']=='right' for r in records)==10
            if split.startswith('test_yaw') or split=='test_combined':
                assert sum(r['task_params']['yaw_degrees']<0 for r in records)==10
                assert sum(r['task_params']['yaw_degrees']>0 for r in records)==10
            if split=='test_combined':
                assert all(sum(r['region']==side and np.sign(r['task_params']['yaw_degrees'])==sign for r in records)==5 for side in ('left','right') for sign in (-1,1))
            for rec in records:
                assert sha256(ROOT/rec['path'])==rec['sha256']
                all_seeds.append(rec['seed']);all_positions.append((task,tuple(rec['task_params']['base_position'])))
                with np.load(ROOT/rec['path'],allow_pickle=False) as data:
                    assert data['snapshot_initial_obs'].shape==(41,)
                    if split=='train20':
                        assert data['obs'].shape==(len(data['actions'])+1,41)
                        assert np.isfinite(data['obs']).all() and np.max(np.abs(data['actions']))<=1
                        assert data['success'][-1]
                data_files+=1
    assert len(set(all_seeds))==len(all_seeds)
    assert len(set(all_positions))==len(all_positions)
    # Disjoint from every historic Round 1 reset seed.
    old=read_json(ROOT/'data/dataset_manifest.json')
    old_seeds={r['seed'] for g in old['tasks'].values() for records in g['splits'].values() for r in records}
    assert set(all_seeds).isdisjoint(old_seeds)
    data_audit=read_json(ROOT/'artifacts/round2/data_audit.json')
    assert data_audit['passed'] and data_audit['full_demo_replays']==40 and data_audit['snapshot_checks']==320
    assert read_json(ROOT/'artifacts/round2/short_replay.json')['passed']
    models={};debug_updates=0;checkpoint_count=0
    for run in RUNS:
        rid=run['run_id'];directory=ROOT/f'runs/round2/{rid}'
        done=read_json(directory/'complete.json');cfg=config(rid)
        assert done['step']==20000 and done['config_hash']==object_hash(cfg)
        assert done['code_hash']==frozen['code_hash'] and done['manifest_hash']==manifest_hash and not done['debug']
        for step in (5000,10000,20000):
            item=done['checkpoints'][str(step)]
            assert sha256(ROOT/item['path'])==item['sha256']
            ckpt=torch.load(ROOT/item['path'],map_location='cpu',weights_only=False)
            assert ckpt['step']==step and ckpt['samples_drawn']==step*128
            assert len(ckpt['optimizer']['state'])>0 and all(int(x['step'])==step for x in ckpt['optimizer']['state'].values())
            assert ckpt['config']==cfg and not cfg['clip_predicted_clean_actions'] and not cfg['thresholding']
            norm=ckpt['normalizer'];assert len(norm['mean'])==41
            assert norm['source_ids']==[r['episode_id'] for r in manifest['tasks'][run['task']]['train20']]
            for rec in manifest['tasks'][run['task']]['train20']:assert norm['source_hashes'][rec['episode_id']]==rec['sha256']
            for i in list(range(7,18))+list(range(25,36))+[39,40]:assert norm['mean'][i]==0 and norm['std'][i]==1
            assert set(ckpt['rng'])=={'loader','diffusion','torch','numpy','python','cuda'}
            checkpoint_count+=1
            del ckpt
        models[rid]=done
        debug=read_json(ROOT/f'artifacts/round2/debug/{rid}/complete.json')
        assert debug['step']==200 and debug['debug']
        resume=read_json(ROOT/f'artifacts/round2/debug/{rid}/resume_audit.json')
        assert resume['passed'] and resume['restored_step']==100
        debug_updates+=debug['step']
    assert debug_updates==800
    for task in ('drawer','door'):
        a,b=[models[f'r2_{task}_{rep}_n20_s0'] for rep in ('world','frame')]
        for key in ('initial_weights_hash','pairing_digest','parameter_count'):assert a[key]==b[key]
    selection=read_json(ROOT/'results/round2/selection.json');dev_count=0;test_count=0
    test_ids={};rates={}
    for run in RUNS:
        rid=run['run_id'];candidates=[]
        for step in (5000,10000,20000):
            result=read_json(ROOT/f'results/round2/dev/{rid}/step_{step:06d}/complete.json')
            assert result['identity']['implementation_hash']==evaluation_hash
            rows=result['records'];assert len(rows)==20
            assert {r['episode_id'] for r in rows}=={r['episode_id'] for r in manifest['tasks'][run['task']]['dev']}
            successes=[r for r in rows if r['success']]
            mean=float(np.mean([r['first_success_step'] for r in successes])) if successes else float('inf')
            candidates.append((-len(successes),mean,step));dev_count+=len(rows)
        selected=selection['runs'][rid];assert min(candidates)[2]==selected['step']
        assert sha256(ROOT/selected['checkpoint_path'])==selected['checkpoint_hash']
        result=read_json(ROOT/f'results/round2/test/{rid}/step_{selected["step"]:06d}/complete.json')
        assert result['identity']['implementation_hash']==evaluation_hash
        assert len(result['records'])==120
        test_ids[rid]={r['episode_id']:r['initial_state_hash'] for r in result['records']}
        for r in result['records']:
            assert sha256(ROOT/r['trajectory_path'])==r['trajectory_hash']
            assert r['identity']['manifest_hash']==manifest_hash and r['identity']['step']==selected['step']
            with np.load(ROOT/r['trajectory_path'],allow_pickle=False) as f:
                actions=f['actions'];progress=f['progress'];assert len(actions)<=500
                assert np.isfinite(actions).all() and (not len(actions) or np.max(np.abs(actions))<=1)
                if not r['exception']:assert len(progress)==len(actions)+1
                if r['success']:
                    assert not r['exception'] and not r['invalid_joint'] and len(progress)>=4
                    assert (progress[-3:]>=.75).all() and (progress<=1.01).all() and (progress>=-.01).all()
                    assert r['first_success_step']==len(actions)
            test_count+=1
        for split in manifest['tasks'][run['task']]:
            if not split.startswith('test_'):continue
            rows=[r for r in result['records'] if r['split']==split]
            assert len(rows)==20
            assert {r['episode_id'] for r in rows}=={r['episode_id'] for r in manifest['tasks'][run['task']][split]}
            assert result['metrics'][split]['successes']==sum(r['success'] for r in rows)
        yaw=[r for r in result['records'] if r['split'] in ('test_yaw_near','test_yaw_mid','test_yaw_far')]
        assert len(yaw)==60
        rates[rid]=sum(r['success'] for r in yaw)/60
        assert result['yaw_primary']['success_rate']==rates[rid]
    for task in ('drawer','door'):assert test_ids[f'r2_{task}_world_n20_s0']==test_ids[f'r2_{task}_frame_n20_s0']
    assert dev_count==240 and test_count==480
    summary=read_json(ROOT/'results/round2/summary.json')
    for row in summary['rows']:assert row['yaw_primary']==rates[row['run_id']]
    reference=read_json(ROOT/'results/round2/expert_reference.json')
    assert reference['identity']['manifest_hash']==manifest_hash and reference['identity']['code_hash']==code_hash()
    for task,group in reference['tasks'].items():
        assert len(group['records'])==120
        for split,m in group['metrics'].items():
            subset=[r for r in group['records'] if r['record']['split']==split]
            assert len(subset)==20 and m['valid_successes']==sum(r['success'] and not r['invalid_joint'] for r in subset)
            assert {r['record']['episode_id'] for r in subset}=={r['episode_id'] for r in manifest['tasks'][task][split]}
    videos=read_json(ROOT/'artifacts/round2/videos/manifest.json');assert videos['complete']
    for v in videos['records']:
        if v.get('path'):assert sha256(ROOT/v['path'])==v['sha256']
        else:assert v.get('absent') or v.get('render_exception')
    report=ROOT/'ROUND2_REPORT.md';assert report.exists()
    result=dict(passed=True,data_files=data_files,formal_checkpoints=checkpoint_count,formal_updates=80000,
                debug_updates=debug_updates,dev_episodes=dev_count,test_episodes=test_count,
                manifest_hash=manifest_hash,report_hash=sha256(report),primary_rates=rates)
    atomic_json(ROOT/'artifacts/round2/delivery_audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
