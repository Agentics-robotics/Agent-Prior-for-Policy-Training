import numpy as np
import cv2
import torch
from collections import Counter
from real_robot.training_pipeline import public
from geometry import make_sam,SAM2ImagePredictor,project,unproject,shape_from_mask,match,features
from perception import proposals,visible_tip
from calibration import infer_calibration
from tracking import material_area,restore_causal_holes


def prepare(spec):
    public.write_report('original_source_plan.json',spec['source_plan']);public.write_report('original_recording_metadata.json',spec['recording_metadata'])
    tids=spec['trajectory_ids'];segments=spec['data_plan']['segments']
    top,tip,calreport=infer_calibration(tids[0],public.records(tids[0]),public.metadata(tids[0]))
    net=make_sam(spec['device']);predictor=SAM2ImagePredictor(net)
    fields=['point','candidate','feasible','action','target_contact','target_direction','target_length','target_validity','xy','normal','loop','goal_xy','obstacles','example_episode','example_segment','example_source_index','example_variant','observation_start','observation_stop','target_start','target_stop','policy_weight']
    data={k:[] for k in fields};audits=[];summaries=[];fixtures=[];original={}
    for skill in spec['source_plan']['skills']:
        for s in skill['segments']:
            original[s['segment_id']]=s.get('decision_indices',[])
    for si,seg in enumerate(segments):
        if 'executor' in seg['segment_id']:
            continue
        tid=seg['trajectory_id'];ei=tids.index(tid);rec=public.records(tid);cam=public.metadata(tid)['calibration']['third'];T=np.array(cam['T_base_cam']);start=seg['start'];stop=seg['stop']
        anchors=sorted(set(list(range(start+60,stop-30,15))+original.get(seg['segment_id'],[])));cache={};why=Counter();retained=0;partial=0;images=0
        Ts=rec['T_base_flange'];tips=Ts[:,:3,3]+np.einsum('nij,j->ni',Ts[:,:3,:3],tip)
        def scene_at(a):
            if a not in cache:
                rgb=public.rgb(tid,int(a),'third')
                with torch.no_grad():
                    masks=proposals(rgb,predictor)
                shapes=[]
                for mask in masks:
                    s=shape_from_mask(mask,cam,T,top)
                    if s is not None:
                        s['metric_area']=material_area(s);shapes.append(s)
                if a!=start:
                    shapes=restore_causal_holes(shapes,reference)
                uv=visible_tip(rgb,project([tips[a]],cam,T)[0]);cache[a]=(shapes,uv)
            return cache[a]
        reference,_=scene_at(start);max_area=max([s['metric_area'] for s in reference]+[.003])
        for a in anchors:
            row={'episode':tid,'segment':seg['segment_id'],'anchor':int(a),'original_anchor':a in original.get(seg['segment_id'],[])};reason=None
            if a+15>=stop:
                reason='no_future_inside_cut'
            if reason is None and not top-.045<tips[a,2]<top+.022:
                reason='tool_not_at_piece_height'
            shapes=[]
            if reason is None:
                shapes,uv=scene_at(a)
                if not shapes:
                    reason='current_mask_geometry_missing'
                elif uv is None:
                    reason='visual_tool_contact_unresolved'
            if reason is None:
                visualxy=unproject([uv],cam,T,top)[0];distances=[float(np.min(np.linalg.norm(s['xy']-visualxy,axis=1))) for s in shapes]
                j=int(np.argmin(distances));actor=shapes[j];cp_id=int(np.argmin(np.linalg.norm(actor['xy']-visualxy,axis=1)));cp=actor['xy'][cp_id]
                row.update({'visual_tip_uv':uv.tolist(),'tip_boundary_distance_m':distances[j],'nominal_visual_tip_discrepancy_px':float(np.linalg.norm(uv-project([tips[a]],cam,T)[0])),'hole_restoration_m':actor.get('causal_hole_restoration_error_m')})
                if distances[j]>.018:
                    reason='no_supported_near_tool_contact'
                elif actor['metric_area']>1.4*max_area:
                    reason='suspected_compound_mask'
            if reason is None:
                last=min(a+30,stop-1);path=tips[a:last+1];arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1))];ok=np.where(arc<=min(.020,.2*actor['diameter']))[0]
                e=a+int(ok[-1]);delta=tips[e,:2]-tips[a,:2];length=float(np.linalg.norm(delta))
                if length<.003 or e-a<3:
                    reason='insufficient_short_prefix_motion'
            if reason is None:
                f=min(a+30,stop-1);fut,fuv=scene_at(f);m=match(actor,fut)
                if m is None or m[2]>.009:
                    reason='response_registration_unresolved'
                else:
                    _,jf,err,R,t=m;goal=actor['xy']@R.T+t;motion=float(np.sqrt(np.mean(np.sum((goal-actor['xy'])**2,axis=1))));response=goal[cp_id]-cp;ratio=fut[jf]['metric_area']/max(1.e-6,actor['metric_area'])
                    row.update({'goal_index':f,'goal_registration_m':err,'observed_shape_motion_m':motion,'prefix_end':e,'shape_area_ratio':float(ratio)})
                    if not .78<ratio<1.28:
                        reason='visibility_or_instance_area_inconsistent'
                    elif motion<max(.006,err*1.5):
                        reason='object_response_below_uncertainty'
                    elif np.dot(response,delta)<0:
                        reason='tool_response_direction_inconsistent'
            if reason is None:
                direction=delta/length;line=tips[a:e+1,:2]-tips[a,:2];perp=np.abs(line[:,0]*direction[1]-line[:,1]*direction[0]);Rdiff=Ts[a,:3,:3].T@Ts[e,:3,:3];angle=float(np.arccos(np.clip((np.trace(Rdiff)-1)/2,-1,1)));visual_agreement=True
                if fuv is not None:
                    vd=unproject([fuv],cam,T,top)[0]-visualxy
                    if np.linalg.norm(vd)>.005:
                        visual_agreement=bool(np.dot(vd,direction)/np.linalg.norm(vd)>.8)
                direction_valid=bool(perp.max()<.004 and np.ptp(tips[a:e+1,2])<.006 and angle<.14 and np.dot(direction,-actor['normal'][cp_id])>-.15 and visual_agreement);length_valid=direction_valid and length>.005
                obslist=[s['xy'] for k,s in enumerate(shapes) if k!=j];obs=np.concatenate(obslist) if obslist else np.zeros((0,2));packed=np.full((576,2),5.,np.float32);packed[:min(576,len(obs))]=obs[:576]
                point,cand,valid,acts,ids=features(actor['xy'],actor['normal'],actor['loop'],goal,obs)
                if not np.any(valid & (np.linalg.norm(acts[:,:2]-cp,axis=1)<.014)):
                    reason='inferred_obstacle_or_access_conflict'
            if reason is None:
                vals={'point':point,'candidate':cand,'feasible':valid.astype(np.float32),'action':acts,'target_contact':cp,'target_direction':direction,'target_length':length,'target_validity':np.array([1,direction_valid,length_valid],np.float32),'xy':actor['xy'],'normal':actor['normal'],'loop':actor['loop'],'goal_xy':goal,'obstacles':packed,'example_episode':ei,'example_segment':si,'example_source_index':a,'example_variant':0,'observation_start':start,'observation_stop':a+1,'target_start':a,'target_stop':f+1,'policy_weight':[1.]}
                for key in fields:
                    data[key].append(vals[key])
                row.update({'status':'retained','target_validity':[True,direction_valid,length_valid],'contact_m':cp.tolist(),'direction':direction.tolist(),'length_m':length,'tool_rotation_rad':angle,'line_residual_m':float(perp.max()),'loop':int(actor['loop'][cp_id])});retained+=1;partial+=int(not length_valid)
                if images<4 or row['original_anchor']:
                    rgb=public.rgb(tid,a,'third').copy()
                    for sh in shapes:
                        pixels=project(np.column_stack([sh['xy'],np.full(96,top)]),cam,T).astype(int)
                        for u in pixels:
                            cv2.circle(rgb,tuple(u),2,(0,220,0),-1)
                    u=project([[cp[0],cp[1],top]],cam,T)[0].astype(int);v=project([[cp[0]+direction[0]*length,cp[1]+direction[1]*length,top]],cam,T)[0].astype(int)
                    cv2.arrowedLine(rgb,tuple(u),tuple(v),(255,0,0),3);cv2.circle(rgb,tuple(uv.astype(int)),5,(0,80,255),2);name='label_'+str(ei)+'_'+str(a)+'.png';public.write_image(name,rgb);row['image']=name;images+=1
                if retained==1:
                    fixtures.append({'array_row':len(data['point'])-1,'segment':seg['segment_id']})
            else:
                row['status']='excluded_from_action_loss';row['reason']=reason;why[reason]+=1
            audits.append(row)
        summaries.append({'segment_id':seg['segment_id'],'attempted':len(anchors),'retained':retained,'partial_contact_only':partial,'reasons':dict(why),'sampled_frames':len(cache)})
        public.progress({'segment':seg['segment_id'],'retained':retained,'attempted':len(anchors)});public.write_report('coverage_progress.json',summaries)
    K=len(data['point']);ints={'example_episode','example_segment','example_source_index','example_variant','observation_start','observation_stop','target_start','target_stop','loop'};arrays={key:np.asarray(data[key],dtype=np.int64 if key in ints else np.float32) for key in fields}
    if K:
        for si in np.unique(arrays['example_segment']):
            idx=arrays['example_segment']==si;arrays['policy_weight'][idx,0]=1./int(idx.sum())
    accounting=[{'trajectory_id':tid,'start':0,'stop':len(public.records(tid)['sample_index']),'use':'context','reason':'Complete original execution trace retained; not motor imitation.'} for tid in tids]
    for s in segments:
        if 'executor' not in s['segment_id']:
            accounting.append({'trajectory_id':s['trajectory_id'],'start':s['start'],'stop':s['stop'],'use':'supervision','reason':'Dense candidates; individually audited rows and masks only.'})
    rows=[r for r in audits if r['status']=='retained']
    stats={'attempted':len(audits),'retained':K,'partial_target_rows':int(np.sum(arrays['target_validity'][:,2]==0)) if K else 0,'original_anchors_retained':sum(r['original_anchor'] for r in rows),'inner_loop_labels':sum(r['loop']>0 for r in rows),'causal_hole_restored_labels':sum(r.get('hole_restoration_m') is not None for r in rows),'rejection_counts':dict(Counter(r['reason'] for r in audits if r['status']!='retained')),'recovery':'Point-prompt SAM, enclosed neutral holes, current visual shaft localization, causal complete-hole reference registration (85% bidirectional visible support), future area consistency, true marginal likelihood for partial channels. Approximate stereo height not commissioning. No failed row declared physically useless.'}
    metadata={'num_examples':K,'coverage':{'segments':summaries,'independence':'Two episodes, same seven-piece inventory, twelve correlated cuts; dense samples not independent shapes.'},'label_provenance':{'calibration':calreport,'masks':'Frozen SAM2.1 points offline/grid online; visible neutral holes plus same-cut first-frame causal hole references. Inferred pseudo-labels, not ground truth.','online_inputs':'Current shape/obstacles, caller SE2, commissioned geometry and optional causal shape references.','training_only':'Future registration and robot prefix; no future features or source indices in model.'},'data_audit':stats,'variant_definition':'0=one local achieved goal and bounded prefix per source anchor','source_accounting':accounting,'fixtures':fixtures}
    public.write_report('anchor_audit.json',audits);public.write_report('coverage.json',metadata);public.write_report('quality_summary.json',stats)
    return {'arrays':arrays,'metadata':metadata}
