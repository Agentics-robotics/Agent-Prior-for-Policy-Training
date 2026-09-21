"""Single full-data preparation and bounded mmap minibatch assembly."""
import json
import numpy as np
import torch
from real_robot.policy_training import public
from perception import Chart,convert_observation,selected_features,full_mask,mask_iou,goal_pack,finite,GW,GH,NODE_COUNT,NODE_DIM
from goals import ANCHOR_BOXES,anchor_goal,registration

RAW_KEYS=('q','dq','tau_ext','T_base_ee','T_base_flange','gripper_position','gripper_width_m',
          't','recv_time','img_t_wrist','img_t_third','img_age_wrist','img_age_third','sample_index')
# Frozen physical referent grouping solely for loss weighting (never neural input).
INSTANCE_GROUPS=[0,1,2,3,1,4,5,3,5,6,1,4,0,1,5,2,3,5,6,4,1,4]
GRAPH_OFFSETS=[7,6,5,4,3,2,1,0]
VISUAL_OFFSETS=[14,8,4,2,0]


def observation_at(records, tid, i, metadata):
    obs={'metadata':metadata}
    for key in RAW_KEYS:
        if key in records:
            value=records[key][i]
            if key.startswith('T_base_') and np.asarray(value).size==16:
                value=np.asarray(value).reshape(4,4)
            obs[key]=value
    obs['sample_index']=i
    obs['third_rgb']=public.rgb(tid,i,'third')
    obs['wrist_rgb']=public.rgb(tid,i,'wrist')
    return obs


def action_at(records,i):
    v=records['action_json'][i]
    if isinstance(v,(bytes,np.bytes_)): v=v.decode('utf-8')
    if isinstance(v,(str,np.str_)): v=json.loads(v)
    if not isinstance(v,dict): return None
    dq=v.get('dq')
    if not finite(dq,(7,)): return None
    return np.asarray(dq,np.float64)


def allowed(segment,i):
    if not segment['supervised_start']<=i<segment['supervised_stop']: return False
    return not any(e['start']<=i<e['stop'] for e in segment.get('supervision_exclusions',[]))


def prepare(spec):
    segments=public.segments()
    if len(segments)!=22: raise ValueError('Frozen cut_v3 requires exactly 22 segments')
    tids=list(spec['trajectory_ids'])
    records={t:public.records(t) for t in tids}
    recording={t:public.metadata(t) for t in tids}
    n=sum(s['stop']-s['start'] for s in segments)
    arrays={
      'frame_episode':np.zeros(n,np.int16),'frame_source_index':np.zeros(n,np.int64),
      'frame_segment':np.zeros(n,np.int16),'third':np.zeros((n,3,GH,GW),np.uint8),
      'wrist':np.zeros((n,3,128,224),np.uint8),'local':np.zeros((n,4,128,128),np.uint8),
      'marker':np.zeros((n,1,GH,GW),np.uint8),'nodes':np.zeros((n,NODE_COUNT,NODE_DIM),np.float32),
      'edges':np.zeros((n,NODE_COUNT,2),np.int16),'node_mask':np.zeros((n,NODE_COUNT),np.uint8),
      'goal':np.zeros((22,4,GH,GW),np.uint8),'center':np.zeros((n,2),np.float32),
      'perception_valid':np.zeros(n,np.uint8),'aux':np.zeros((n,2),np.float32),
      'aux_valid':np.zeros(n,np.float32)}
    example_frame=[]; example_ep=[]; example_idx=[]; example_seg=[]; commands=[]; weights=[]
    coverage=[]; offset=0; segment_limits=[]
    for si,s in enumerate(segments):
        tid=s['trajectory_id']; ep=tids.index(tid); rec=records[tid]; md=recording[tid]; chart=Chart(md)
        start,stop=s['start'],s['stop']; length=stop-start
        if stop>len(rec['action_json']): raise ValueError('Segment outside original paired data')
        frames=[]; tracker=None; previous=None; times=[]
        # All current features are generated FORWARD before anchor association.
        for i in range(start,stop):
            obs=observation_at(rec,tid,i,md)
            frame,tracker=convert_observation(obs,chart,tracker,previous)
            frames.append(frame); times.append(float(obs['t']))
            previous={k:obs.get(k) for k in RAW_KEYS}
        arrays['frame_episode'][offset:offset+length]=ep
        arrays['frame_source_index'][offset:offset+length]=np.arange(start,stop)
        arrays['frame_segment'][offset:offset+length]=si
        segment_limits.append([offset,offset+length])
        derivation={}; selected_id=None; failure=None
        goal_mask=np.zeros((720,1280),np.uint8); goal_center=np.zeros(2,np.float32)
        try:
            raw=public.rgb(tid,stop-1,'third')
            goal_mask,goal_rgb,gp=anchor_goal(raw,chart,ANCHOR_BOXES[si])
            candidates=sorted([(mask_iou(p,gp),k) for k,p in frames[-1]['observed'].items()],reverse=True)
            if not candidates or candidates[0][0]<.2: raise ValueError('anchor_track_unassociated')
            if len(candidates)>1 and candidates[0][0]-candidates[1][0]<.12:
                raise ValueError('anchor_track_ambiguous')
            selected_id=candidates[0][1]
            refs=[f['observed'][selected_id] for f in frames if selected_id in f['observed'] and
                  f['observed'][selected_id]['area']>=.65*gp['area']]
            if not refs: raise ValueError('no_causal_reference_for_goal')
            derivation=registration(refs[0],gp)
            if derivation['iou']<.38 or not .25<=derivation['area_ratio']<=4.:
                raise ValueError('anchor_registration_inconsistent')
            goal_center=gp['center']; arrays['goal'][si]=goal_pack(goal_mask,goal_rgb)
            derivation['anchor_iou']=candidates[0][0]
            derivation['goal_confidence']=gp['confidence']
        except ValueError as exc:
            failure=str(exc); selected_id=None
        counts={'materialized':length,'outside_supervision':0,'missing_command':0,'invalid_perception':0,
                'invalid_state_or_timing':0,'goal_derivation_failure':0,'eligible':0}
        eligible=np.zeros(length,bool); segment_commands={}; dts=np.diff(np.asarray(times))
        positive=dts[dts>0]; first_dt=float(np.median(positive)) if len(positive) else .0333333333
        duration_weights=np.clip(np.r_[first_dt,dts],.0001,.2)
        state_valid=[]
        for j,frame in enumerate(frames):
            i=start+j; fidx=offset+j
            selected=frame['observed'].get(selected_id)
            feat=selected_features(frame,selected,goal_mask,goal_center)
            if 'state' not in arrays:
                arrays['state']=np.zeros((n,len(feat['state'])),np.float32)
            for key in ('state','nodes','edges','node_mask','marker','local','center'):
                arrays[key][fidx]=feat[key]
            arrays['third'][fidx]=frame['third'].transpose(2,0,1)
            arrays['wrist'][fidx]=frame['wrist'].transpose(2,0,1)
            arrays['perception_valid'][fidx]=int(feat['valid'])
            cmd=action_at(rec,i)
            # State q/transforms are required; measured dq/tau may be missing with explicit masks.
            ok_state=all(key in rec and finite(np.asarray(rec[key][i]).reshape(-1)) for key in ('q','T_base_ee','T_base_flange','t'))
            ages=[float(rec[key][i]) if key in rec and finite(rec[key][i]) else np.inf for key in ('img_age_third','img_age_wrist')]
            ok_state=ok_state and all(abs(x)<=.5 for x in ages) and (j==0 or times[j]>times[j-1])
            state_valid.append(ok_state)
            if not allowed(s,i): counts['outside_supervision']+=1
            elif cmd is None: counts['missing_command']+=1
            elif failure is not None: counts['goal_derivation_failure']+=1
            elif not feat['valid']: counts['invalid_perception']+=1
            elif not ok_state: counts['invalid_state_or_timing']+=1
            else:
                eligible[j]=True; segment_commands[j]=cmd; counts['eligible']+=1
        # Each physical referent within each episode gets equal total weight, divided over its calls.
        group=INSTANCE_GROUPS[si]
        calls=sum(1 for sj,ss in enumerate(segments) if ss['trajectory_id']==tid and INSTANCE_GROUPS[sj]==group)
        denom=max(float(duration_weights[eligible].sum()),1e-10)*calls
        for j in np.flatnonzero(eligible):
            fidx=offset+j
            example_frame.append(fidx); example_ep.append(ep); example_idx.append(start+j); example_seg.append(si)
            commands.append(segment_commands[j]); weights.append(float(duration_weights[j]/denom))
            # Future target is training-only; all intervening rows satisfy every action mask.
            if j+3<length and eligible[j:j+4].all():
                arrays['aux'][fidx]=(arrays['center'][fidx+3]-arrays['center'][fidx])/[1280.,720.]
                arrays['aux_valid'][fidx]=1.
        coverage.append({'segment_id':s['segment_id'],'segment_ordinal':si,'trajectory_id':tid,
                         'start':start,'stop':stop,'goal_anchor':stop-1,'original_goal_box':ANCHOR_BOXES[si],
                         'counts':counts,'goal_failure':failure,'registration':derivation})
        public.progress({'phase':'preparation','segment':s['segment_id'],'counts':counts,'goal_failure':failure})
        offset+=length
        del frames
    if not commands: raise ValueError('No eligible observations; inspect coverage, do not train a stub')
    arrays['example_frame']=np.asarray(example_frame,np.int64)
    arrays['example_episode']=np.asarray(example_ep,np.int16)
    arrays['example_source_index']=np.asarray(example_idx,np.int64)
    arrays['example_segment']=np.asarray(example_seg,np.int16)
    arrays['native_action']=np.asarray(commands,np.float64)
    arrays['loss_weight']=np.asarray(weights,np.float64)
    # Causal map: zero-validity padded prefix. No frame from a different segment is borrowed.
    for name,offsets in [('graph',GRAPH_OFFSETS),('visual',VISUAL_OFFSETS)]:
        hist=np.zeros((len(commands),len(offsets)),np.int64); mask=np.zeros(hist.shape,np.float32)
        for k,(fidx,si) in enumerate(zip(example_frame,example_seg)):
            lo,hi=segment_limits[si]
            for j,lag in enumerate(offsets):
                ix=fidx-lag; hist[k,j]=max(ix,lo); mask[k,j]=float(ix>=lo)
        arrays[name+'_history']=hist; arrays[name+'_history_mask']=mask
    x=arrays['state'][arrays['example_frame']].astype(np.float64)
    y=arrays['native_action']; w=arrays['loss_weight']; w=w/w.sum()
    xm=(x*w[:,None]).sum(0); xs=np.maximum(np.sqrt(((x-xm)**2*w[:,None]).sum(0)),.05)
    ym=(y*w[:,None]).sum(0); ys=np.maximum(np.sqrt(((y-ym)**2*w[:,None]).sum(0)),.02)
    meta={'num_examples':len(commands),'num_original_frames':n,'coverage':coverage,'state_dim':x.shape[1],
          'state_mean':xm.tolist(),'state_std':xs.tolist(),'action_mean':ym.tolist(),'action_std':ys.tolist(),
          'normalization':'Weighted training statistics; command dq native units, no rad/s assertion.',
          'conversion':'Brown-Conrady K unchanged; warm-chroma foreground, causal Hungarian identity; anchor boxes only goals/track selection.',
          'history':{'graph_offsets':GRAPH_OFFSETS,'visual_offsets':VISUAL_OFFSETS,'no_cross_segment':True},
          'weighting':'Elapsed robot time clipped [0.0001,0.2] seconds, equal physical referent/episode totals divided among its segments; first-row weight uses within-segment median dt only on label side.',
          'auxiliary':'Three original rows ahead, only if all four rows are eligible; normalized image centroid displacement.',
          'perception_assets':'None; deterministic material-contrast adaptation, not generic learned segmentation.',
          'cache_bytes':int(sum(a.nbytes for a in arrays.values()))}
    if meta['cache_bytes']>64000000000: raise ValueError('Cache limit exceeded')
    return {'arrays':arrays,'metadata':meta}


def tensor(a,device,dtype=None):
    return torch.as_tensor(np.array(a,copy=True),device=device,dtype=dtype)


def batch(policy_id,arrays,metadata,indices,spec):
    idx=np.asarray(indices,np.int64); dev=spec['device']
    mode='graph' if policy_id=='contour_push' else 'visual'
    h=np.asarray(arrays[mode+'_history'][idx]); f=np.asarray(arrays['example_frame'][idx]); s=np.asarray(arrays['example_segment'][idx])
    out={'state':tensor(arrays['state'][h],dev,torch.float32),
         'history_mask':tensor(arrays[mode+'_history_mask'][idx],dev,torch.float32),
         'action':tensor(arrays['native_action'][idx],dev,torch.float32),
         'aux':tensor(arrays['aux'][f],dev,torch.float32),'aux_valid':tensor(arrays['aux_valid'][f],dev,torch.float32)}
    if mode=='graph':
        for key,dtype in [('nodes',torch.float32),('edges',torch.long),('node_mask',torch.float32)]:
            out[key]=tensor(arrays[key][h],dev,dtype)
    else:
        for key in ('third','wrist','local','marker'):
            out[key]=tensor(arrays[key][h],dev,torch.float32)/255.
        out['goal']=tensor(arrays['goal'][s],dev,torch.float32)/255.
    return out
