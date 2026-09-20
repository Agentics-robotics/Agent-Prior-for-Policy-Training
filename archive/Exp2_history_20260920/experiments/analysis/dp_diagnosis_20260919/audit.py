"""Validate completed diagnostic artifacts, frozen inputs and worker closure."""
import json
import re
import subprocess
import time
import numpy as np
from appl.io import read,atomic,digest,source_manifest
from appl.scaleup.tasks import measure
from .run import BASE,TASKS


def main():
    plan=read(BASE/'plan.json');assert plan['framework_source']==source_manifest()
    groups={'original_DP':0,'binary_gripper':0,'takeover':0};steps={k:0 for k in groups};steps['expert_replay']=0
    videos=[];incidents=[]
    for task in TASKS:
        entry=plan['tasks'][task];assert digest(entry['checkpoint'])==entry['checkpoint_sha256']
        goal=read(entry['completion_contract']);assert digest(entry['completion_contract'])==entry['completion_contract_sha256']
        for v in entry['demonstrations'].values():assert digest(v['path'])==v['sha256']
        for group,label in [('original_DP','runs'),('binary_gripper','binary_gripper'),('takeover','takeover'),('takeover','takeover_physical_replay')]:
            for path in sorted((BASE/label/task).glob('*/result.json') if label!='runs' else (BASE/label/task).glob('*/frozen_DP/result.json')):
                result=read(path);trace=[json.loads(l) for l in (path.parent/'trace.jsonl').read_text().splitlines()]
                count=result['policy_steps'] if group=='takeover' else result['steps']
                assert len(trace)==count and [r['step'] for r in trace]==list(range(1,count+1))
                for r in trace:
                    assert r['metrics']==measure(r['state'],goal)
                    assert np.isfinite(np.array(r['raw_action'])).all()
                    if group=='binary_gripper':
                        assert r['action'][7]==(1 if r['raw_action'][7]>=0 else -1)
                        assert r['action'][:7]==r['raw_action'][:7]
                    else:assert r['action']==r['raw_action']
                assert result['success']==trace[-1]['metrics']['success']
                if not result['success']:assert count==1500
                groups[group]+=1;steps[group]+=count
                if group=='takeover':steps[group]+=result['expert_prefix_steps']
                videos.append(path.parent/'replay.mp4')
        replay=list((BASE/'runs'/task).glob('*/expert_replay/result.json'));assert len(replay)==12
        for p in replay:
            r=read(p);assert r['success'] and max(r['initial_errors'].values())==0
            steps['expert_replay']+=r['steps'];videos.append(p.parent/'replay.mp4')
        for p in (BASE/'takeover'/task).glob('*/failure.json'):
            assert read(p)['error']=='Prefix replay mismatch'
            demo=read(entry['demonstrations'][p.parent.name]['path']);g=np.array(demo['actions'])[:,-1]
            closes=np.flatnonzero((g[1:]<0)&(g[:-1]>=0))+1
            count=int(closes[2 if task=='drawer_exchange' else 1])-32
            incidents.append(dict(task=task,demonstration=p.parent.name,steps=count,policy_steps=0,reason='Exact-state guard stopped after physical expert prefix; retained follow-up records replay differences.'))
    assert groups==dict(original_DP=60,binary_gripper=15,takeover=15)
    assert len(videos)==150
    for row in read(BASE/'videos.json'):videos.append(BASE/row['path']);assert digest(BASE/row['path'])==row['sha256']
    for path in videos:
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=codec_name,nb_frames,width,height','-of','json',str(path)],text=True))
        assert info['streams'] and int(info['streams'][0]['nb_frames'])>0
    processes=subprocess.check_output(['ps','-eo','pid=,comm=,args='],text=True).splitlines()
    pattern=re.compile(r' -m experiments\.exp2\.analysis\.dp_diagnosis_20260919\.(run|teacher|takeover|gripper|conditioning|grip_sensitivity)( |$)')
    active=[line.split(maxsplit=2) for line in processes if len(line.split(maxsplit=2))==3 and line.split(maxsplit=2)[1].startswith('python') and pattern.search(line.split(maxsplit=2)[2])]
    assert not active,active
    occupancy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory,process_name','--format=csv'],text=True)
    atomic(BASE/'completion.json',dict(finished=time.time(),validated=True,completed_policy_rollouts=groups,
        completed_expert_replays=60,physical_steps_by_completed_operation=steps,
        guarded_prefix_attempts=incidents,guarded_prefix_steps=sum(r['steps'] for r in incidents),
        total_physical_steps=sum(steps.values())+sum(r['steps'] for r in incidents),
        zero_action_setup_incident='incidents/instrumentation_before_physics',
        validated_videos=len(videos),validated_stepwise_goals=True,
        model_source_training_data_goal_hashes_unchanged=True,optimizer_updates=0,Runtime_API_requests=0,
        final_occupancy=occupancy,active_owned_scientific_workers=active,
        allocated_physical_devices=plan['devices'],diagnostic_source_hashes={p.name:digest(p) for p in BASE.glob('*.py')}))
    print(dict(validated=True,rollouts=groups,videos=len(videos),physical_steps=sum(steps.values())+sum(r['steps'] for r in incidents),active_owned_workers=len(active)))


if __name__=='__main__':main()
