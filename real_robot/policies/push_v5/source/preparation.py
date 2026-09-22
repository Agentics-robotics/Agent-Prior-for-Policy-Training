import numpy as np
import cv2
from collections import Counter
from real_robot.training_pipeline import public
from geometry import camera, project, fit_tool, stereo_pair, boundary, register, features
from recovery import MaskRecovery, fit_plane


def prepare(spec):
    public.write_report('original_source_plan.json',spec['source_plan'])
    ann=public.source_json('annotations.json')
    tip,tool_report=fit_tool(spec)
    public.write_report('tool_recovery.json',tool_report)
    pts=[];stereo=[]
    for ep,tid in enumerate(spec['trajectory_ids']):
        for i in ann['review_indices'][ep][::2]:
            x,r=stereo_pair(tid,i);pts.extend(x);r['episode']=ep;stereo.append(r)
    plane,plane_report=fit_plane(pts)
    public.write_report('plane_recovery.json',{'fit':plane_report,'pairs':stereo})
    engine=MaskRecovery(spec['device'])
    cache={};audits=[];examples=[];reason_counts=Counter();segment_counts=Counter()
    def scene(tid,i):
        key=(tid,int(i))
        if key not in cache:
            rgb=public.rgb(tid,int(i),'third')
            masks,seed=engine.run(rgb)
            cam=camera(public.metadata(tid)['calibration'],'third')
            out=[]
            if plane is not None:
                for m in masks:
                    s=boundary(m['loops'],cam,plane)
                    if s is not None:
                        m['shape']=s;out.append(m)
            else:
                out=masks
            cache[key]=(out,seed)
        return cache[key]
    seen=set()
    for si,seg in enumerate(spec['data_plan']['segments']):
        tid=seg['trajectory_id'];ep=spec['trajectory_ids'].index(tid)
        rec=public.records(tid);start=seg['start'];stop=seg['stop']
        old=[i for i in ann['review_indices'][ep] if start<=i<stop]
        anchors=sorted(set(range(start,stop-1,30))|set(old))
        Rf=np.asarray(rec['T_base_flange'][:,:3,:3]);pf=np.asarray(rec['T_base_flange'][:,:3,3])
        tool=np.einsum('nij,j->ni',Rf,tip)+pf
        cam=camera(public.metadata(tid)['calibration'],'third')
        for a in anchors:
            if (ep,a) in seen:
                continue
            seen.add((ep,a))
            audit={'episode':ep,'segment':seg['segment_id'],'index':a,'prior_anchor':a in ann['review_indices'][ep],'validity':{'contact':False,'direction':False,'length':False},'reason':None}
            fail=None;target_end=min(a+30,stop-1);e=a
            if plane is None:
                fail='no_top_plane'
            if fail is None:
                top=float(np.r_[tool[a,:2],1.]@plane)
                audit['tip_minus_top_m']=float(tool[a,2]-top)
                if tool[a,2]>top+.012 or tool[a,2]<top-.04:
                    fail='tool_height_outside_unverified_contact_band'
            if fail is None:
                initial=None
                for j in range(a+1,target_end+1):
                    delta=tool[j,:2]-tool[a,:2];length=np.linalg.norm(delta)
                    if length>.02 or abs(tool[j,2]-tool[a,2])>.003:
                        break
                    rot=Rf[a].T@Rf[j]
                    angle=np.arccos(np.clip((np.trace(rot)-1)/2,-1,1))
                    if angle>.0524:
                        break
                    if length>.003 and initial is None:
                        initial=delta/length
                    if initial is not None and length>.004 and np.dot(delta/length,initial)<.966:
                        break
                    e=j
                delta=tool[e,:2]-tool[a,:2];length=float(np.linalg.norm(delta))
                audit['prefix_length_m']=length;audit['prefix_stop']=e
                if length<.004 or e<=a:
                    fail='no_supported_linear_prefix'
            cur=[];seed=None
            if fail is None or audit['prior_anchor']:
                cur,seed=scene(tid,a)
                audit['scene_masks']=len(cur)
                audit['mask_pixels']=[m['mask_area'] for m in cur]
                audit['seed_pixels']=[m['seed_area'] for m in cur]
                audit['visible_loop_counts']=[len(m['loops']) for m in cur]
            chosen=None;future=None;fit=None
            if fail is None:
                if not cur:
                    fail='no_current_mask'
                else:
                    distance=[float(np.min(np.linalg.norm(m['shape'][:,:2]-tool[a,:2],axis=1))) for m in cur]
                    ix=int(np.argmin(distance));chosen=cur[ix]
                    audit['nearest_boundary_m']=distance[ix]
                    if distance[ix]>.018:
                        fail='no_contact_association'
                    elif len(distance)>1 and np.sort(distance)[1]<distance[ix]+.008:
                        fail='ambiguous_actor'
            if fail is None:
                fut,_=scene(tid,e)
                choices=[]
                for m in fut:
                    s=m['shape'];oldshape=chosen['shape']
                    if np.linalg.norm(s[:,:2].mean(0)-oldshape[:,:2].mean(0))>.07:
                        continue
                    ratio=np.ptp(s[:,:2],axis=0).prod()/max(1e-8,np.ptp(oldshape[:,:2],axis=0).prod())
                    if ratio<.5 or ratio>2:
                        continue
                    R,t,err=register(oldshape[:,:2],s[:,:2]);choices.append((err,m,R,t))
                if not choices:
                    fail='no_future_actor'
                else:
                    choices.sort(key=lambda x:x[0]);err,future,R,t=choices[0];fit=(R,t)
                    audit['registration_m']=err
                    motion=np.linalg.norm(chosen['shape'][:,:2]@R.T+t-chosen['shape'][:,:2],axis=1)
                    audit['object_motion_m']=float(np.median(motion))
                    if err>.006:
                        fail='registration_uncertain'
                    elif np.median(motion)<max(.004,err*1.5):
                        fail='object_motion_not_resolved'
                    elif len(choices)>1 and choices[1][0]<err+.001:
                        fail='future_association_ambiguous'
            if fail is None:
                s=chosen['shape'];R,t=fit
                cp=s[np.argmin(np.linalg.norm(s[:,:2]-tool[a,:2],axis=1)),:2]
                fend=float(np.min(np.linalg.norm(future['shape'][:,:2]-tool[e,:2],axis=1)))
                audit['end_boundary_distance_m']=fend
                if fend>.020:
                    fail='end_contact_not_supported'
                elif tool_report['uncertainty_floor_m']>.008:
                    fail='tool_calibration_uncertain'
                else:
                    actor,cand,geom=features(s,R,t,uncertainty=max(.005,tool_report['uncertainty_floor_m']))
                    if length>.2*geom['diameter']:
                        fail='stroke_exceeds_shape_fraction'
                    else:
                        sigma=max(.005,tool_report['uncertainty_floor_m'],err)
                        dc=np.linalg.norm(geom['points']-cp,axis=1)
                        dv=np.clip(geom['directions']@(delta/length),-1,1)
                        if np.max(dv[dc<.012])<.5:
                            fail='outward_or_unsupported_contact'
                        else:
                            valid=np.array([1.,float(length>sigma),float(length>2*sigma)],np.float32)
                            penalties=np.stack([.5*(dc/sigma)**2,(1-dv)/.1,.5*((geom['lengths']-length)/.005)**2],axis=1)
                            kernels=np.exp(-np.minimum(penalties,60)).astype(np.float32)
                            examples.append({'ep':ep,'seg':si,'a':a,'e':e,'actor':actor,'cand':cand,'kernels':kernels,'valid':valid,'shape':s.astype(np.float32),'R':R.astype(np.float32),'t':t.astype(np.float32)})
                            audit['validity']={k:bool(v) for k,v in zip(['contact','direction','length'],valid)}
                            segment_counts[seg['segment_id']]+=1
            audit['reason']=fail if fail else 'retained_inferred_weak_contact'
            reason_counts[audit['reason']]+=1;audits.append(audit)
            if audit['prior_anchor'] or (fail is None and segment_counts[seg['segment_id']]<=2):
                rgb=public.rgb(tid,a,'third').copy()
                for m in cur:
                    for p,h in m['loops']:
                        cv2.polylines(rgb,[p.astype(np.int32)],True,(0,190,255) if h else (40,220,30),2)
                uv=project([tool[a]],cam)[0]
                if np.isfinite(uv).all() and np.max(np.abs(uv))<10000:
                    cv2.circle(rgb,tuple(uv.astype(int)),8,(255,30,30),2)
                cv2.putText(rgb,audit['reason'],(35,680),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,0,0),2)
                public.write_image('recovery_'+str(ep)+'_'+str(a)+'.png',rgb)
            if len(audits)%40==0:
                public.progress({'stage':'recovery','examined':len(audits),'retained':len(examples)})
        public.write_report('audit_'+seg['segment_id']+'.json',[r for r in audits if r['segment']==seg['segment_id']])
    K=len(examples)
    def stack(key,shape,dtype=np.float32):
        return np.stack([e[key] for e in examples]).astype(dtype) if examples else np.zeros((0,)+shape,dtype)
    arrays={'example_episode':np.array([e['ep'] for e in examples],np.int64),'example_segment':np.array([e['seg'] for e in examples],np.int64),'example_source_index':np.array([e['a'] for e in examples],np.int64),'example_variant':np.zeros(K,np.int64),'observation_start':np.array([e['a'] for e in examples],np.int64),'observation_stop':np.array([e['a']+1 for e in examples],np.int64),'target_start':np.array([e['a'] for e in examples],np.int64),'target_stop':np.array([e['e']+1 for e in examples],np.int64),'policy_weight':np.ones((K,1),np.float32),'actor':stack('actor',(64,7)),'candidate':stack('cand',(960,18)),'kernels':stack('kernels',(960,3)),'valid':stack('valid',(3,)),'shape':stack('shape',(64,5)),'goal_R':stack('R',(2,2)),'goal_t':stack('t',(2,))}
    accounting=[]
    for tid in spec['trajectory_ids']:
        accounting.append({'trajectory_id':tid,'start':0,'stop':len(public.records(tid)['sample_index']),'use':'context','reason':'Full original paired executor evidence retained; never replayed or used for motor loss.'})
    for seg in spec['data_plan']['segments']:
        accounting.append({'trajectory_id':seg['trajectory_id'],'start':seg['start'],'stop':seg['stop'],'use':'supervision','reason':'Expanded contact-recovery search domain. Only explicitly retained anchors contribute masked choice loss; rejected rows remain context.'})
    coverage={'attempted':len(audits),'retained':K,'original_anchors_attempted':sum(r['prior_anchor'] for r in audits),'original_anchors_retained':sum(r['prior_anchor'] and r['reason']=='retained_inferred_weak_contact' for r in audits),'segments':dict(segment_counts),'reasons':dict(reason_counts),'episode_counts':[sum(e['ep']==ep for e in examples) for ep in range(len(spec['trajectory_ids']))],'partial_valid_counts':arrays['valid'].sum(0).tolist(),'independence':'Two episodes of the same inventory, at most 12 interaction groups; adjacent windows are correlated, not new independent trials.'}
    metadata={'num_examples':K,'coverage':coverage,'label_provenance':{'tool':tool_report,'plane':plane_report,'targets':'Inferred SAM silhouettes, local rigid registration and measured linear tool-point prefixes. Not hand-verified force/contact or intended goals. Future geometry is label-only. Contact is nearest boundary conditioned on coincident motion, with soft uncertainty kernels.','inputs':'Current observed shape plus explicit numerical goal; training goal inferred from future e only. No episode index, semantic ID, RGB color or future mask is a model feature.'},'data_audit':{'reason_counts':dict(reason_counts),'recovery':'Expanded 1Hz grid plus all 47 source anchors; frozen SAM 2.1 replaces raw color contours. SIFT stereo followed by independent cross-view silhouette-height sweep if SIFT fails. API visible-cap annotation attempts tool recovery. Partial channels masked; topology mismatch alone does not discard.','unresolved':'Full tool support/collision geometry, table plane versus piece-top separation, temporal image-to-state offset, semantic autonomy and independent calibration are not certified.'},'variant_definition':'0: first bounded linear prefix with same-instance achieved local goal; no augmented duplicate variants','source_accounting':accounting,'split':'Episode 0 training, episode 1 development (already inspected and used in preparation; not an untouched test).'}
    public.write_report('coverage.json',metadata)
    return {'arrays':arrays,'metadata':metadata}
