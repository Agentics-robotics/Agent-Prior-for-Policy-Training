"""Immutable feature cache, exact paired provenance and loss-specific eligibility."""
import json
import numpy as np
import torch
from real_robot.policy_training_v3 import public
from geometry import workspace, perceive, project_pixels, init_tracks, update_tracks, scene_labels, features, HISTORY, center, DX, register_masks, finite

SEEDS = {
 'episode_2026091922002701': [[280,192],[385,225],[326,306],[492,285],[451,390],[227,432],[380,502]],
 'episode_2026091922062101': [[518,145],[518,635],[280,380],[316,204],[624,430],[315,548],[574,260]]}
# Initial visual handles only. Shape aliases are never learned input features.
def target_handle(segment):
    s=segment['segment_id']
    if segment['dataset_group']=='tool_position': return 0
    if 'wedge' in s: return 2
    if 'upper_bar' in s: return 4
    if '_bar_' in s: return 5
    if '_ring_' in s: return 7
    if '_r_' in s: return 3
    if '_d_' in s: return 6
    if '_l_' in s: return 1
    raise ValueError('Unassigned frozen segment '+s)

def command(raw):
    if isinstance(raw, bytes): raw=raw.decode('utf-8')
    a=json.loads(str(raw)) if not isinstance(raw,dict) else raw
    v=a.get('v'); w=a.get('w')
    ok=v is not None and w is not None
    u=np.asarray(list(v)+list(w),np.float64) if ok else np.full(6,np.nan)
    ok=u.shape==(6,) and np.isfinite(u).all()
    dq=a.get('dq'); valid_dq=dq is not None and np.asarray(dq).shape==(7,) and np.isfinite(np.asarray(dq,dtype=float)).all()
    return u,bool(ok),finite(dq if valid_dq else np.zeros(7)),bool(valid_dq)

def row_observation(r,i,metadata,images=False,trajectory_id=None):
    o={'metadata':metadata}
    for key in ['q','dq','tau_ext','T_base_ee','T_base_flange','gripper_position','gripper_width_m','t','recv_time','img_t_wrist','img_t_third','img_age_wrist','img_age_third']:
        if key in r: o[key]=r[key][i]
    o['sample_index']=int(i)
    o['T_base_ee']=np.asarray(o['T_base_ee']).reshape(4,4)
    o['T_base_flange']=np.asarray(o['T_base_flange']).reshape(4,4)
    if images:
        for camera in ['third','wrist']: o[camera+'_rgb']=public.rgb(trajectory_id,int(i),camera)
    return o

def excluded(segment,i):
    for e in segment.get('supervision_exclusions',[]):
        if isinstance(e,dict) and int(e.get('start',i+1))<=i<int(e.get('stop',i)): return True
        if isinstance(e,(list,tuple)) and len(e)>=2 and e[0]<=i<e[1]: return True
    return False

def prepare(spec):
    segments=public.segments(); episodes=spec['trajectory_ids']; ids=spec['policy_ids']
    source={}; conversion={}; perception_summary={}
    # Decode each original paired RGB only once, shared across segment reuse.
    for ep in episodes:
        r=public.records(ep); md=public.metadata(ep); T,sigma,_=workspace(md)
        N=len(r['action_json']); labels=np.zeros((N,256,256),np.uint8)
        confidence=np.zeros((N,7),np.float32); angles=np.zeros_like(confidence)
        disagreement=np.zeros(N,np.float32); tracks=None
        for i in range(N):
            o=row_observation(r,i,md,True,ep)
            fg,valid,dis=perceive(o,T)
            if tracks is None:
                seed=project_pixels(SEEDS[ep],o,'third',T)
                tracks=init_tracks(fg,seed)
                if len(tracks)!=7: raise ValueError('No initial track inventory for '+ep)
            else: tracks=update_tracks(tracks,fg)
            labels[i]=scene_labels(tracks)
            confidence[i]=[t['confidence'] for t in tracks]; angles[i]=[t['angle'] for t in tracks]
            disagreement[i]=dis
            if i%1000==0: public.progress({'stage':'causal_perception','episode':ep,'source_index':i,'confidence':confidence[i].tolist()})
        source[ep]=(r,md,T,labels,confidence,angles,disagreement)
        conversion[ep]={'T_base_W_proxy':T.tolist(),'plane_sigma_design_margin_m':sigma,'plane_verified':False,'initial_third_pixels':SEEDS[ep]}
        perception_summary[ep]={'mean_confidence':confidence.mean(0).tolist(),'mean_dual_view_disagreement':float(disagreement.mean()),'unique_frames':N}
    frames={k:[] for k in ['state','maps','nodes','basis','base','score']}
    anchors={k:[] for k in ['example_episode','example_source_index','example_segment','native_action','policy_weight','history','history_valid','phase','response','response_valid','contact_index','geometry_valid','command_dq','command_dq_valid','goal_valid','reuse_count']}
    coverage=[]; reuse={}
    for seg in segments:
        for i in range(seg['supervised_start'],seg['supervised_stop']):
            key=(seg['trajectory_id'],i); reuse[key]=reuse.get(key,0)+1
    for si,seg in enumerate(segments):
        ep=seg['trajectory_id']; ei=episodes.index(ep)
        r,md,T,labs,conf,angles,dis=source[ep]; target=target_handle(seg)
        lo,hi=seg['start'],seg['stop']; s,e=seg['supervised_start'],seg['supervised_stop']
        endpoint=e-1; goaltool=np.asarray(r['T_base_ee'][endpoint]).reshape(4,4)
        term,termok,_,_=command(r['action_json'][endpoint]); term=finite(term) if termok else np.zeros(6,np.float32)
        goalmask=(labs[endpoint]==target).astype(np.uint8) if target else np.zeros((256,256),np.uint8)
        goalvalid=bool(target and conf[endpoint,target-1]>=.35 and goalmask.sum()>=12)
        goalfit=register_masks((labs[s]==target).astype(np.uint8),goalmask)[1] if target else 1.
        goalvalid=goalvalid and goalfit>=.20 if target else False
        if not goalvalid: goalmask=np.zeros_like(goalmask)
        start_frame=len(frames['state']); fs=[]
        for i in range(lo,hi):
            o=row_observation(r,i,md)
            prev=command(r['action_json'][i-1]) if i>lo else (np.zeros(6),False,None,False)
            last=finite(prev[0]) if prev[1] else np.zeros(6,np.float32)
            quality=float(conf[i,target-1]) if target else float(np.mean(conf[i]))
            f=features(o,labs[i],target,goalmask,goalvalid,quality,T,goaltool,term,last,.03)
            fs.append(f)
            for k in frames: frames[k].append(f[k])
        kept=[]; bad=0; phase_counts=[0]*5; geometry_count=0; response_count=0; null_dq=0
        for i in range(s,e):
            u,ok,adq,adqok=command(r['action_json'][i])
            if excluded(seg,i) or not ok: bad+=1; continue
            f=fs[i-lo]; phase=4 if np.linalg.norm(u)<1e-9 else (3 if f['contour_error']<.012 and goalvalid else (2 if abs(u[2])>.012 else (0 if f['proximity']>.025 else 1)))
            if not target: phase=4 if np.linalg.norm(u)<1e-9 else (2 if abs(u[2])>.012 else 0)
            phase_counts[phase]+=1
            geo=bool(target and conf[i,target-1]>.35)
            resp=np.zeros(3,np.float32); respok=False
            if target and i+1<e and not excluded(seg,i+1) and geo and conf[i+1,target-1]>.35:
                ca=center(labs[i]==target); cb=center(labs[i+1]==target)
                # Weak motion is actual in-segment successor, never an online feature.
                v=(cb-ca)*DX; b=f['basis']; world=T[:3,:3]@np.array([v[0],v[1],0.])
                loc=b.T@world
                yaw=angles[i+1,target-1]-angles[i,target-1]
                resp=np.array([loc[0]/.01,loc[1]/.01,yaw/.1],np.float32)
                respok=bool(np.linalg.norm(v)<.03 and abs(yaw)<.3)
            geometry_count+=int(geo); response_count+=int(respok); null_dq+=int(not adqok)
            history=[start_frame+max(0,i-lo-h) for h in HISTORY]
            hvalid=[float(i-lo-h>=0) for h in HISTORY]
            weight=np.zeros(len(ids),np.float32)
            for j,pid in enumerate(ids):
                group='tool_position' if pid=='tool_waypoint_v1' else 'piece_relocation'
                if seg['dataset_group']==group: weight[j]=1.
            row={'example_episode':ei,'example_source_index':i,'example_segment':si,'native_action':u,'policy_weight':weight,'history':history,'history_valid':hvalid,'phase':phase,'response':resp,'response_valid':float(respok),'contact_index':int(np.argmin(np.linalg.norm(f['nodes'][:,6:8],axis=1))),'geometry_valid':float(geo),'command_dq':adq,'command_dq_valid':float(adqok),'goal_valid':float(goalvalid),'reuse_count':reuse[(ep,i)]}
            kept.append(len(anchors['phase']))
            for k in anchors: anchors[k].append(row[k])
        # Equal episode, equal invocation, then phase-balanced; holds at most 5%.
        present=[p for p in range(4) if phase_counts[p]>0]
        for a in kept:
            ph=anchors['phase'][a]; mass=.05 if ph==4 else .95/max(1,len(present))
            if not present: mass=1.
            nseg=sum(1 for g in segments if g['trajectory_id']==ep and g['dataset_group']==seg['dataset_group'])
            anchors['policy_weight'][a]*=mass/max(1,phase_counts[ph])/nseg
        record={'segment_id':seg['segment_id'],'segment_ordinal':si,'trajectory_id':ep,'context_range':[lo,hi],'supervised_range':[s,e],'declared_exclusions':seg.get('supervision_exclusions',[]),'retained_action':len(kept),'excluded_action':bad,'excluded_reason':'Only missing/nonfinite v,w or declared exclusions; null command dq retained.','target_visual_handle':target,'phase_counts_approach_contact_reset_exit_hold':phase_counts,'geometry_retained':geometry_count,'response_retained':response_count,'goal_eligible':goalvalid,'goal_fit_IoU':float(goalfit),'goal_source_index':endpoint,'goal_provenance':'Unreviewed causal tracked endpoint on board-plane proxy; no verified intent/success.','missing_dq_retained':null_dq,'policies':{}}
        for pid in ids:
            applicable=(pid=='tool_waypoint_v1')==(seg['dataset_group']=='tool_position')
            record['policies'][pid]={'retained':len(kept) if applicable else 0,'excluded':bad if applicable else e-s,'reason':'assigned group; partial geometry and endpoint masks apply per loss' if applicable else 'different frozen dataset group','goal_conditioned_piece_rows':len(kept) if applicable and goalvalid else 0}
        coverage.append(record); public.progress({'stage':'segment_cache','coverage':record})
    arrays={}
    for k,v in frames.items(): arrays['f_'+k]=np.asarray(v,dtype=np.float16 if k in ['maps','nodes'] else np.float32)
    ints=['example_episode','example_source_index','example_segment','history','phase','contact_index','reuse_count']
    for k,v in anchors.items(): arrays[k]=np.asarray(v,dtype=np.int64 if k in ints else (np.float64 if k=='native_action' else np.float32))
    scales={}
    for j,pid in enumerate(ids):
        u=arrays['native_action'][arrays['policy_weight'][:,j]>0]
        linear=max(.005,float(np.quantile(np.abs(u[:,:3]),.95)))
        angular=max(.025,float(np.quantile(np.abs(u[:,3:]),.98)))
        scales[pid]=[linear]*3+[angular]*3
    meta={'num_examples':len(arrays['native_action']),'coverage':coverage,'normalization':scales,'conversion':conversion,'perception':perception_summary,'history_offsets':HISTORY,'state_dim':112,'map_channels':8,'node_dim':16,'cache_bytes':sum(a.nbytes for a in arrays.values()),'unique_original_records':sum(len(source[e][0]['action_json']) for e in episodes),'source_segments':segments,'goal_missing_adaptation':'All finite action labels retained; missing object goal/track encoded explicitly. Those rows train command/mode with endpoint tool goal, but not confident object-goal or motion loss.','phase_labels':'Weak numeric command and geometric-proximity labels, not contact truth.','raw_command_preservation':'Native v/w exact float64; command dq separate from measured dq. Complete raw original records and media remain public/source manifest; no realignment.','metric_aggregation':'Use inverse reuse_count for pooled metrics, not independent-observation claims.'}
    if meta['cache_bytes']>64000000000: raise ValueError('cache budget exceeded')
    return {'arrays':arrays,'metadata':meta}

def batch(policy_id, arrays, metadata, indices, spec):
    ix=np.asarray(indices,dtype=np.int64); h=np.asarray(arrays['history'][ix]); out={}
    for k in ['state','maps','nodes']:
        out[k]=torch.as_tensor(np.array(arrays['f_'+k][h],dtype=np.float32),device=spec['device'])
    for k in ['basis','base','score']:
        out[k]=torch.as_tensor(np.array(arrays['f_'+k][h[:,-1]],dtype=np.float32),device=spec['device'])
    for k in ['native_action','history_valid','response','response_valid','geometry_valid','goal_valid','phase','contact_index']:
        dtype=torch.long if k in ['phase','contact_index'] else torch.float32
        out[k]=torch.as_tensor(np.array(arrays[k][ix]),dtype=dtype,device=spec['device'])
    return out
