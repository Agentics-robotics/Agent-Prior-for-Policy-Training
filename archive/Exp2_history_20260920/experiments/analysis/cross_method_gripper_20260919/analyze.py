"""Read-only gripper execution/trajectory screening across all four Exp2 methods."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from appl.io import ROOT,read,atomic,digest,source_manifest

BASE=Path(__file__).resolve().parent


def screen(row):
    root=Path(row['episode_root']);initial=read(root/'initial_state.json')
    trace=[json.loads(l) for l in (root/'trace.jsonl').read_text().splitlines() if l.strip()]
    assert len(trace)==int(row['steps'])
    grip=np.asarray([r['action'][7] for r in trace]);raw=np.asarray([r['raw_action'][7] for r in trace])
    states=[initial]+[r['state'] for r in trace]
    width=np.asarray([sum(s['qpos'][7:9]) for s in states])
    red=np.asarray([s['red_pose'][:3] for s in states]);blue=np.asarray([s['blue_pose'][:3] for s in states])
    tcp=np.asarray([s['tcp_pose'][:3] for s in states]);pid=np.asarray([r.get('policy_id','naive_DP') for r in trace])
    closes=np.flatnonzero((grip[1:]<0)&(grip[:-1]>=0))+1
    events=[]
    # This conservative screen excludes drawer, whose handle is another intended grasp target.
    if row['task']!='drawer_exchange':
        for index in closes:
            index=int(index)
            if index<24 or index+20>=len(trace):continue
            start=index-24;end=index+20
            if len(set(pid[start:end+1]))!=1:continue
            if np.any(grip[start:index]<0):continue
            if int(((grip[start:index]>0)&(grip[start:index]<.95)).sum())<8:continue
            if width[start:index+1].max()<.07 or width[index]>.065:continue
            if width[start:index+1].max()-width[index]<.012:continue
            clearance=tcp[index,2]-max(red[index,2],blue[index,2])
            if clearance<=.08:continue
            # Exclude a visibly moving/held cube and require the closing event to end in a very narrow opening.
            if max(np.linalg.norm(red[start:index+1]-red[index],axis=1).max(),np.linalg.norm(blue[start:index+1]-blue[index],axis=1).max())>.01:continue
            if min(width[index+1:end+2])>.005:continue
            if max(np.linalg.norm(red[index:end+2]-red[index],axis=1).max(),np.linalg.norm(blue[index:end+2]-blue[index],axis=1).max())>.01:continue
            events.append(dict(action_step=index+1,policy_id=str(pid[index]),
                pre_action_clearance_above_both_cubes_m=float(clearance),pre_action_finger_width_m=float(width[index]),
                gripper_command=float(grip[index]),previous_24_opening_max_m=float(width[start:index+1].max()),
                subsequent_20_opening_min_m=float(width[index+1:end+2].min()),
                window=[dict(step=j+1,policy_id=str(pid[j]),gripper=float(grip[j]),input_width_m=float(width[j]),
                    input_TCP_z_m=float(tcp[j,2]),red_z_m=float(red[j,2]),blue_z_m=float(blue[j,2])) for j in range(start,end+1)]))
    narrow=width[1:]<=.005;longest=0;current=0
    for x in narrow:current=current+1 if x else 0;longest=max(longest,current)
    return dict(task=row['task'],method=row['method'],condition=row['condition'],seed=int(row['seed']),
        episode_root=str(root.relative_to(ROOT)),success=row['success']=='True',steps=len(trace),
        trace_sha256=digest(root/'trace.jsonl'),result_sha256=digest(root/'result.json'),
        gripper_changed_by_executor_steps=int((raw!=grip).sum()),
        materially_intermediate_gripper_steps=int((abs(grip)<.95).sum()),
        non_endpoint_gripper_steps=int((abs(grip)<.999).sum()),longest_empty_width_run=longest,
        first_red_3cm_rise=next((i for i,x in enumerate(red[:,2]) if x>red[0,2]+.03),None),
        first_blue_3cm_rise=next((i for i,x in enumerate(blue[:,2]) if x>blue[0,2]+.03),None),
        progressive_air_closure_events=events)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--finalize-saved',action='store_true');args=parser.parse_args()
    rows=list(csv.DictReader((ROOT/'experiments/exp2/paper/tables/episodes.csv').open()));assert len(rows)==1200
    framework=source_manifest();started=time.time();BASE.mkdir(exist_ok=True)
    if args.finalize_saved:
        prior=read(BASE/'plan.json');assert prior['framework']==framework
        assert prior['episode_table_sha256']==digest(ROOT/'experiments/exp2/paper/tables/episodes.csv')
        out=read(BASE/'episodes.json');assert len(out)==1200
        for row,record in zip(rows,out,strict=True):
            assert (row['task'],row['method'],row['condition'],int(row['seed']))==(record['task'],record['method'],record['condition'],record['seed'])
            root=Path(row['episode_root'])
            assert digest(root/'trace.jsonl')==record['trace_sha256'] and digest(root/'result.json')==record['result_sha256']
        return finalize(rows,out,framework,started)
    atomic(BASE/'plan.json',dict(created=started,source_sha256=digest(__file__),framework=framework,
        episode_table_sha256=digest(ROOT/'experiments/exp2/paper/tables/episodes.csv'),
        mode='Read-only analysis of the 1200 currently selected formal outcomes; no new rollouts, training, or API requests.',
        detector='Non-drawer: positive-to-negative grip transition; 24 preceding nonnegative steps on one policy, >=8 intermediate positive commands below .95, opening falls >=12mm from >=70mm to <=65mm; TCP >80mm above both stationary cubes; within 20 further same-policy steps opening reaches <=5mm and cubes remain within 10mm of their event positions.',
        interpretation='A conservative progressive-air-closure screen, not contact truth, proof of causal feedback, or a failure attribution. Intentional empty-hand closes can be flagged; other failures can be missed.'))
    out=[]
    for i,row in enumerate(rows,1):
        out.append(screen(row))
        if i%100==0:print(dict(processed=i,total=1200,elapsed_seconds=round(time.time()-started,1)),flush=True)
    atomic(BASE/'episodes.json',out)
    finalize(rows,out,framework,started)


def finalize(rows,out,framework,started):
    summaries=[]
    for method in sorted({r['method'] for r in rows}):
        for condition in ['ID','OOD']:
            for task in sorted({r['task'] for r in rows})+['all']:
                subset=[r for r in out if r['method']==method and r['condition']==condition and (task=='all' or r['task']==task)]
                events=[r for r in subset if r['progressive_air_closure_events']]
                summaries.append(dict(method=method,condition=condition,task=task,episodes=len(subset),successes=sum(r['success'] for r in subset),
                    gripper_changed_by_executor_steps=sum(r['gripper_changed_by_executor_steps'] for r in subset),
                    intermediate_command_fraction=sum(r['materially_intermediate_gripper_steps'] for r in subset)/sum(r['steps'] for r in subset),
                    flagged_episodes=len(events),flagged_successes=sum(r['success'] for r in events),
                    flagged_events=sum(len(r['progressive_air_closure_events']) for r in subset),
                    episodes_with_100step_empty_width=sum(r['longest_empty_width_run']>=100 for r in subset)))
    atomic(BASE/'summary.json',summaries)
    with (BASE/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    policies=list(csv.DictReader((ROOT/'experiments/exp2/paper/POLICY_INDEX.csv').open()))
    hashes={}
    for r in policies:
        if r['owner']=='Runtime API':
            path=ROOT/r['source']/'policy.py';hashes[str(path.relative_to(ROOT))]=digest(path)
            bundled=ROOT/'experiments/exp2/paper'/r['bundled_API_source']/'policy.py'
            assert digest(bundled)==digest(path)
    assert source_manifest()==framework
    atomic(BASE/'completion.json',dict(completed=True,episodes=len(out),runtime_API_requests=0,training_updates=0,simulation_steps=0,
        source_matches_paper_bundle=len(hashes),API_policy_source_sha256=hashes,framework_unchanged=True,
        finalization_source_sha256=digest(__file__),elapsed_seconds=time.time()-started,
        analysis_repairs='Fixed a NumPy index JSON serialization error and excluded developer baseline rows from the API bundle-source check; no scientific run or source was altered.'))
    print([r for r in summaries if r['task']=='all'])


if __name__=='__main__':main()
