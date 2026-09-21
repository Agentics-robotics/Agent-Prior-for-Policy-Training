"""Frozen-cut preprocessing. Future anchor access is restricted to fixed training goals/identity labels."""
import json
import numpy as np
from real_robot.policy_training import public
from perception import current,features,box_mask,registration
from geometry import graph,goal_image

# The frozen original RAW third-image boxes. These constants are TRAINING LABELS only.
GOAL_BOXES={
'a_w':[500,490,695,640],'a_i_stage1':[275,310,465,432],'a_o':[505,365,680,490],
'a_r_stage':[405,135,545,245],'a_i_stage2':[250,245,425,385],'a_h_stage':[250,462,448,605],
'a_d_stage':[528,150,645,260],'a_r':[490,263,642,370],'a_d_align':[500,135,635,217],
'a_l':[490,190,635,284],'a_i_align':[315,238,477,343],'a_h_align':[290,333,470,452],
'b_w':[598,492,805,651],'b_i_stage':[408,312,571,430],'b_d_stage':[590,148,713,252],
'b_o':[592,363,762,490],'b_r':[580,258,740,370],'b_d_align':[587,123,716,206],
'b_l':[580,186,726,278],'b_h_stage':[201,387,415,529],'b_i_align':[368,232,529,333],
'b_h_align':[352,370,529,493]}
HISTORY=8
STRIDE=3

def scalar(a):
    b=np.asarray(a)
    return b.item() if b.size==1 else a

def observation(records,i,meta,rgb=True,trajectory_id=None):
    names=('q','dq','tau_ext','T_base_ee','T_base_flange','gripper_position','gripper_width_m',
           't','recv_time','sample_index','img_t_wrist','img_t_third','img_age_wrist','img_age_third')
    ob={key:scalar(records[key][i]) for key in names if key in records}
    ob['metadata']=meta;ob['sample_index']=int(i)
    if 't' not in ob: raise ValueError('original robot t is required; timestamp reconstruction forbidden')
    if rgb:
        ob['third_rgb']=public.rgb(trajectory_id,i,'third');ob['wrist_rgb']=public.rgb(trajectory_id,i,'wrist')
    return ob

def command(records,i):
    raw=scalar(records['action_json'][i])
    if isinstance(raw,bytes): raw=raw.decode('utf-8')
    act=json.loads(raw) if isinstance(raw,str) else raw
    dq=act.get('dq')
    if dq is None: return None
    a=np.asarray(dq,np.float64)
    return a if a.shape==(7,) and np.isfinite(a).all() else None

def supervised(seg,i):
    if not seg['supervised_start']<=i<seg['supervised_stop']: return False
    return not any(e['start']<=i<e['stop'] for e in seg['supervision_exclusions'])

def prepare_data(spec):
    segments=public.segments(); eps=spec['trajectory_ids']
    # One cache slot per ORIGINAL materialized row, never a synthesized terminal sample.
    total=sum(s['stop']-s['start'] for s in segments)
    a={
      'frame_state':np.zeros((total,142),np.float32),
      'frame_graph':np.zeros((total,256,20),np.float32),
      'frame_edges':np.zeros((total,256,2),np.int16),
      'frame_node_mask':np.zeros((total,256),np.uint8),
      'frame_global':np.zeros((total,180,320,4),np.uint8),
      'frame_wrist':np.zeros((total,144,256,3),np.uint8),
      'frame_detail':np.zeros((total,160,160,4),np.uint8),
      'frame_crop_transform':np.zeros((total,4),np.float32),
      'frame_valid':np.zeros(total,np.uint8),
      'frame_center':np.zeros((total,2),np.float32),
      'segment_goal':np.zeros((len(segments),90,160,4),np.uint8)}
    ee=[];ss=[];sg=[];actions=[];weights=[];frame_indices=[];histories=[];aux=[];aux_valid=[];coverage=[]
    offset=0;last_episode=None;rec=None;meta=None
    for ordinal,seg in enumerate(segments):
        trajectory=seg['trajectory_id'];start,stop=seg['start'],seg['stop']; sid=seg['segment_id']
        if trajectory!=last_episode:
            rec=public.records(trajectory);meta=public.metadata(trajectory);last_episode=trajectory
        frames=[];old=None;observations=[]
        for i in range(start,stop):
            ob=observation(rec,i,meta,True,trajectory)
            fr=current(ob,old);old={'tracker':fr['tracker']}
            # No future frames, goal box or goal pose enters this detector/association pass.
            frames.append({k:fr[k] for k in ('rgb','wrist','proposals','assigned')})
            observations.append({k:v for k,v in ob.items() if k not in ('third_rgb','wrist_rgb')})
        report={'segment_id':sid,'segment_ordinal':ordinal,'trajectory_id':trajectory,
                'start':start,'stop':stop,'goal_anchor':stop-1,'materialized':stop-start,
                'edge_or_range_excluded':0,'missing_command':0,'invalid_perception':0,
                'invalid_state':0,'eligible':0,'goal_derivation':None}
        hint=box_mask(GOAL_BOXES[sid],meta)
        choices=[]
        for tid,p in frames[-1]['assigned'].items():
            overlap=float((p['mask']*hint).sum()/max(1,p['area']))
            if overlap>.55: choices.append((overlap,tid,p))
        choices.sort(key=lambda z:z[0],reverse=True)
        selected=None;goal=None;first=None
        if not choices or (len(choices)>1 and choices[0][0]-choices[1][0]<.15):
            report['goal_derivation']='failed: anchor foreground/box association missing or ambiguous'
        else:
            selected=choices[0][1];goal=choices[0][2]['mask']
            first=next((fr['assigned'][selected] for fr in frames if selected in fr['assigned'] and fr['assigned'][selected]['confidence']>=.3),None)
            reg=registration(first['mask'],goal) if first is not None else {'valid':False}
            report['goal_registration']=reg
            if not reg['valid']:
                report['goal_derivation']='failed: foreground not similarity-consistent';selected=None
            else:
                report['goal_derivation']='derived: material foreground, causal track selected retrospectively for LABEL identity only'
                report['causal_track_id_training_audit']=int(selected)
                a['segment_goal'][ordinal]=goal_image(goal,frames[-1]['rgb'])
        candidates=[];native=[];delta=[]
        for local,(fr,ob) in enumerate(zip(frames,observations)):
            i=start+local; slot=offset+local
            p=fr['assigned'].get(selected) if selected is not None else None
            state_ok=all(np.asarray(ob.get(k,[])).size==n and np.isfinite(np.asarray(ob[k],float)).all()
                         for k,n in [('q',7),('dq',7),('tau_ext',7),('T_base_ee',16),('T_base_flange',16)])
            perception_ok=p is not None and p['confidence']>=.3
            if state_ok and perception_ok:
                f=features(ob,fr,p,observations[local-1] if local else None)
                others=[pp['mask'] for k,pp in fr['assigned'].items() if k!=selected]
                nodes,edges,mask=graph(p['mask'],goal,others,f['tool'],f['tool_valid'],p['confidence'])
                a['frame_state'][slot]=f['state'];a['frame_graph'][slot]=nodes
                a['frame_edges'][slot]=edges;a['frame_node_mask'][slot]=mask
                for key in ('global','wrist','detail','crop_transform'): a['frame_'+key][slot]=f[key]
                a['frame_valid'][slot]=1;a['frame_center'][slot]=p['center']/[640,360]
            if not supervised(seg,i): report['edge_or_range_excluded']+=1;continue
            cmd=command(rec,i)
            if cmd is None: report['missing_command']+=1;continue
            if not state_ok: report['invalid_state']+=1;continue
            if not perception_ok: report['invalid_perception']+=1;continue
            candidates.append(local);native.append(cmd)
            dt=float(ob['t'])-float(observations[local-1]['t']) if local else (float(observations[1]['t'])-float(ob['t']) if len(observations)>1 else 1/30.)
            delta.append(float(np.clip(dt,1e-4,.2)))
        report['eligible']=len(candidates)
        # Time weighting, cap contiguous exactly repeated-command runs at 0.5 seconds,
        # then balance each original segment equally (not each letter/class).
        ww=np.asarray(delta,np.float64)
        begin=0
        while begin<len(candidates):
            end=begin+1
            while end<len(candidates) and candidates[end]==candidates[end-1]+1 and np.array_equal(native[end],native[begin]): end+=1
            mass=ww[begin:end].sum();ww[begin:end]*=min(1.,.5/max(mass,1e-10));begin=end
        if len(ww): ww/=ww.sum()
        eligible_set=set(candidates)
        for j,local in enumerate(candidates):
            slot=offset+local
            ee.append(eps.index(trajectory));ss.append(start+local);sg.append(ordinal)
            actions.append(native[j]);weights.append(ww[j]);frame_indices.append(slot)
            history=[offset+k if k>=0 and a['frame_valid'][offset+k] else -1 for k in range(local-STRIDE*(HISTORY-1),local+1,STRIDE)]
            histories.append(history)
            future=local+STRIDE
            ok=future in eligible_set
            aux_valid.append(float(ok))
            aux.append((a['frame_center'][offset+future]-a['frame_center'][slot]) if ok else np.zeros(2))
        coverage.append(report)
        public.progress({'stage':'prepare','segment':sid,'coverage':report})
        offset+=stop-start
        del frames,observations
    if not actions: raise ValueError('No valid goal-conditioned action examples; see coverage. No untrained fallback.')
    k=len(actions)
    a.update({'example_episode':np.asarray(ee,np.int64),'example_source_index':np.asarray(ss,np.int64),
              'example_segment':np.asarray(sg,np.int64),'native_action':np.asarray(actions,np.float64),
              'loss_weight':np.asarray(weights,np.float64)*k/max(1,sum(c['eligible']>0 for c in coverage)),
              'example_frame':np.asarray(frame_indices,np.int64),'history':np.asarray(histories,np.int64),
              'future_displacement':np.asarray(aux,np.float32),'future_valid':np.asarray(aux_valid,np.float32)})
    valid=a['frame_valid']>0;st=a['frame_state'][valid]
    mean=st.mean(0);std=np.maximum(st.std(0),.01);mean[71:]=0;std[71:]=1
    am=a['native_action'].mean(0);ast=np.maximum(a['native_action'].std(0),.015)
    md={'num_examples':k,'coverage':coverage,'num_materialized':total,
        'action_mean':am.tolist(),'action_std':ast.tolist(),'state_mean':mean.tolist(),'state_std':std.tolist(),
        'history_length':HISTORY,'history_stride_original_rows':STRIDE,
        'normalization':'Training-only finite statistics; saved in model buffers. Native commands cached float64, never observation dq.',
        'conversion':'Half-resolution Brown-Conrady chart (K unchanged in original coordinates); no time realignment; no depth.',
        'perception':'Untrained broad brown/cardboard chromaticity, connected components and causal Hungarian association; not a generic segmenter.',
        'weighting':'Recorded dt, exact repeated-command run mass capped at .5 s, equal segment mass.',
        'assets':[],'approximation':'2-pixel foreground raster; goal similarity consistency is heuristic, not ground truth.'}
    return {'arrays':a,'metadata':md}

def make_batch(policy_id,arrays,metadata,indices,spec):
    import torch
    idx=np.asarray(indices,np.int64);hh=np.asarray(arrays['history'][idx]);safe=np.maximum(hh,0)
    def tensor(x,dtype=torch.float32): return torch.as_tensor(np.array(x,copy=True),device=spec['device'],dtype=dtype)
    batch={'state':tensor(arrays['frame_state'][safe]),'history_mask':tensor(hh>=0),
           'action':tensor(arrays['native_action'][idx]),'future':tensor(arrays['future_displacement'][idx]),
           'future_valid':tensor(arrays['future_valid'][idx])}
    if policy_id=='contour_push':
        batch.update({'nodes':tensor(arrays['frame_graph'][safe]),'edges':tensor(arrays['frame_edges'][safe],torch.long),
                      'node_mask':tensor(arrays['frame_node_mask'][safe])})
    elif policy_id=='visual_push':
        for name in ('global','wrist','detail'):
            batch[name]=tensor(arrays['frame_'+name][safe],torch.uint8).permute(0,1,4,2,3)
        batch['goal']=tensor(arrays['segment_goal'][np.asarray(arrays['example_segment'][idx])],torch.uint8).permute(0,3,1,2)
    else: raise ValueError('unknown policy')
    return batch
