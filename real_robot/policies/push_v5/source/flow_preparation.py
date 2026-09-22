import numpy as np
import cv2
from collections import Counter
from real_robot.training_pipeline import public
from geometry import camera, project, to_plane
from flow_geometry import support_candidates, track_pair, encode


def prepare(spec):
    public.write_report('original_source_plan.json',spec['source_plan'])
    prior=spec['source_plan']['skills'][0]['segments'];ann=public.source_json('annotations.json')
    audits=[];examples=[];counts=Counter();bygroup=Counter();plane=np.array([0.,0.,.068])
    for ep,tid in enumerate(spec['trajectory_ids']):
        rec=public.records(tid);N=len(rec['sample_index']);cam=camera(public.metadata(tid)['calibration'],'third')
        ee=np.asarray(rec['T_base_ee']);xyz=ee[:,:3,3];old=ann['review_indices'][ep]
        anchors=sorted(set(range(240 if ep else 400,N-2,15))|set(old))
        for a in anchors:
            owners=[j for j,s in enumerate(prior) if s['trajectory_id']==tid and s['supervised_start']<=a<s['supervised_stop']]
            if not owners:
                continue
            group=max(owners,key=lambda j:prior[j]['supervised_start'])
            seg=prior[group];stop=seg['stop'];e=a;fail=None
            audit={'episode':ep,'index':a,'group':seg['segment_id'],'original_anchor':a in old}
            if not .025<xyz[a,2]<.095:
                fail='not_low_tool_interaction'
            if fail is None:
                initial=None
                for j in range(a+1,min(a+31,stop)):
                    delta=xyz[j]-xyz[a];length=np.linalg.norm(delta[:2])
                    angle=np.arccos(np.clip((np.trace(ee[a,:3,:3].T@ee[j,:3,:3])-1)/2,-1,1))
                    if length>.018 or abs(delta[2])>.0025 or angle>.0524:
                        break
                    if length>.003 and initial is None:
                        initial=delta[:2]/length
                    if initial is not None and length>.004 and np.dot(delta[:2]/length,initial)<.966:
                        break
                    e=j
                delta=xyz[e,:2]-xyz[a,:2];length=float(np.linalg.norm(delta))
                if e-a<4 or length<.005:
                    fail='insufficient_planar_prefix'
                else:
                    path=xyz[a:e+1,:2]-xyz[a,:2]
                    cross=np.abs(path[:,0]*delta[1]-path[:,1]*delta[0])/length
                    if cross.max()>.002:
                        fail='curved_prefix'
            audit['end']=e
            if fail is None and max(abs(float(rec['img_age_third'][a])),abs(float(rec['img_age_third'][e])))>.10:
                fail='image_age_outside_band'
            chosen=None;fit=None;rgb0=None;rgb1=None;candidates=[]
            if fail is None:
                rgb0=public.rgb(tid,a,'third');rgb1=public.rgb(tid,e,'third')
                candidates=support_candidates(rgb0,cam,plane);choices=[]
                for c in candidates:
                    proximity=float(np.min(np.linalg.norm(c['occupied']-xyz[a,:2],axis=1)))
                    if proximity>.04:
                        continue
                    f=track_pair(rgb0,rgb1,c['uv'],cam,plane)
                    if f is None or f['motion']<.004:
                        continue
                    dr=np.median(f['xy1']-f['xy0'],axis=0)
                    coherence=float(dr@delta/max(1e-8,np.linalg.norm(dr)*length))
                    if coherence<.2 or f['motion']>length*3+.005:
                        continue
                    choices.append((proximity,c,f,coherence))
                choices.sort(key=lambda x:x[0])
                if not choices:
                    fail='no_resolved_rigid_visible_support'
                elif len(choices)>1 and choices[1][0]<choices[0][0]+.008:
                    fail='ambiguous_moving_support'
                else:
                    proximity,chosen,fit,coherence=choices[0]
                    audit.update({'points':fit['count'],'rigid_rms_m':fit['rms'],'split_spread_m':fit['spread'],'object_motion_m':fit['motion'],'robot_displacement_m':length,'scale':fit['scale'],'coherence':coherence,'proximity_m':proximity,'bbox':chosen['bbox'],'support_coverage':fit['support_coverage'],'discordant_fraction':fit['discordant_fraction']})
            if fail is None:
                changes=[];base_flow=fit['xy1']-fit['xy0']
                for z in [.060,.076]:
                    x=to_plane(fit['uv0'],cam,np.array([0.,0.,z]));y=to_plane(fit['uv1'],cam,np.array([0.,0.,z]))
                    changes.append(float(np.max(np.linalg.norm((y-x)-base_flow,axis=1))))
                elapsed=float(rec['t'][e]-rec['t'][a])
                speed=length/max(elapsed,.001)
                # Preserve paired samples. Budget endpoint ages plus one30Hz
                # period for unverified exposure/state alignment; no re-pairing.
                timing_sigma=speed*(abs(float(rec['img_age_third'][a]))+abs(float(rec['img_age_third'][e]))+1/30.)
                differential=max(.0015,fit['rms'],fit['spread'],max(changes),timing_sigma)
                audit.update({'height_sensitivity_m':max(changes),'timing_budget_m':timing_sigma,'differential_uncertainty_m':differential})
                if fit['motion']<2*differential:
                    fail='motion_below_differential_uncertainty'
                else:
                    prev=xyz[a,:2]-xyz[a-6,:2]
                    actor,state,F=encode(chosen['occupied'],fit['R'],fit['t'],xyz[a],ee[a,:3,2],prev,.008)
                    examples.append({'ep':ep,'a':a,'e':e,'group':group,'actor':actor,'state':state,'target':(delta@F/.02).astype(np.float32),'support':chosen['occupied'],'R':fit['R'],'t':fit['t'],'sigma':differential})
                    bygroup[seg['segment_id']]+=1
            audit['reason']=fail or 'retained_visible_motion_and_measured_EE';counts[audit['reason']]+=1;audits.append(audit)
            if fail is None:
                view=rgb0.copy();mask=chosen['mask']>0
                view[mask]=(view[mask]*.6+np.array([0,160,30])*.4).astype(np.uint8)
                for p,q in zip(fit['uv0'],fit['uv1']):
                    cv2.circle(view,tuple(p.astype(int)),3,(255,30,30),-1)
                    cv2.line(view,tuple(p.astype(int)),tuple(q.astype(int)),(255,255,0),2)
                uv=project(np.array([xyz[a],xyz[e]]),cam)
                cv2.line(view,tuple(uv[0].astype(int)),tuple(uv[1].astype(int)),(20,100,255),3)
                cv2.putText(view,str(ep)+':'+str(a)+'-'+str(e)+' partial support, NOT contact',(25,690),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,0,0),2)
                if bygroup[seg['segment_id']]<=2 or a in old:
                    public.write_image('flow_'+str(ep)+'_'+str(a)+'.png',np.concatenate([view,rgb1],axis=1))
                examples[-1]['thumb']=cv2.resize(view,(320,180))
            if len(audits)%80==0:
                public.progress({'stage':'rigid_flow_recovery','attempted':len(audits),'retained':len(examples)})
        public.write_report('flow_audit_'+str(ep)+'.json',[r for r in audits if r['episode']==ep])
    for page in range((len(examples)+23)//24):
        canvas=np.zeros((4*180,6*320,3),np.uint8)
        for j,x in enumerate(examples[page*24:(page+1)*24]):
            y=j//6;col=j%6;canvas[y*180:(y+1)*180,col*320:(col+1)*320]=x['thumb']
        public.write_image('flow_sheet_'+str(page)+'.png',canvas)
    K=len(examples)
    def arr(key,shape,dtype=np.float32):
        return np.stack([x[key] for x in examples]).astype(dtype) if K else np.zeros((0,)+shape,dtype)
    arrays={'example_episode':arr('ep',(),np.int64),'example_segment':arr('ep',(),np.int64),'example_source_index':arr('a',(),np.int64),'example_variant':np.zeros(K,np.int64),'observation_start':np.array([x['a']-6 for x in examples],np.int64),'observation_stop':np.array([x['a']+1 for x in examples],np.int64),'target_start':arr('a',(),np.int64),'target_stop':np.array([x['e']+1 for x in examples],np.int64),'policy_weight':np.ones((K,1),np.float32),'actor':arr('actor',(64,4)),'state':arr('state',(11,)),'target':arr('target',(2,)),'group':arr('group',(),np.int64),'goal_sigma':arr('sigma',())}
    coverage={'attempted':len(audits),'retained':K,'episode_counts':[sum(x['ep']==ep for x in examples) for ep in range(2)],'groups':dict(bygroup),'original_anchors_attempted':sum(x['original_anchor'] for x in audits),'original_anchors_retained':sum(x['original_anchor'] and x['reason'].startswith('retained') for x in audits),'reasons':dict(counts),'independence':'Two same-inventory episodes, up to12 broad original groups. Overlapping0.5s windows are highly correlated; not K independent pushes.'}
    accounting=[]
    for seg in spec['data_plan']['segments']:
        for use,reason in [('context','All original states, RGB, actions and terminal rows retained. No motor replay or success labels.'),('supervision','Search domain only. Retained anchors contribute recorded planar EE displacement loss with inferred visual-motion goals; exclusions audited by original index.')]:
            accounting.append(dict(trajectory_id=seg['trajectory_id'],start=seg['start'],stop=seg['stop'],use=use,reason=reason))
    metadata={'num_examples':K,'coverage':coverage,'label_provenance':{'target':'Recorded T_base_ee[e].translation_xy minus T_base_ee[a].translation_xy. No inferred contact point/normal/force, tip offset, gripper or dq label. Rotation/z/straightness bounded for compatibility.','goal':'Rigid consensus of bidirectionally tracked visible surface corners, with spatial coverage of CURRENT support and low discordant-motion fraction. Future pixels only goal/eligibility/actor association, never causal shape completion.','input':'Current occupied support cloud, EE xyz/axis and last6row displacement, plus caller-equivalent numeric goal. Current support is never replaced with future inlier positions or inferred full-instance topology.','chart':'Base XY at provisional z=.068, absolute input-chart uncertainty8mm; differential motion sensitivity at .060/.076 and one-frame-plus-endpoint-age timing budget. Direct robot delta bypasses physical tip conversion.'},'data_audit':{'recovery':'2Hz plus47 source anchors; direct EE labels and visible rigid-flow goals replace unsound contact truth. After205-row review, added general85% rigid-hull input-coverage and<=12% discordant-track gates against merged/stationary support, and explicit timing budget. No index whitelist.','excluded':'High/rotating/curved, small movement, ambiguous support, poor temporal/spatial rigid consensus remain context. Before-contact travel is deterministic initialization, not imitated. Full geometry needed for physical initial contact but not for this loss. No training on old contact cache.','prior_attempts':'Original contact recovery source and assessment plus all source versions retained; first205-row flow cache remains separate failed/revisable evidence. Original source_plan copied verbatim.'},'variant_definition':'0: first <=1s measured planar EE prefix with achieved visible-motion goal. One source-key example; original interaction boundary decisions owned by later group.','source_accounting':accounting,'split':'A train/B development; B informed preparation, never untouched test.','actual_differential_sigma_quantiles':np.quantile(arr('sigma',()),[0,.5,1]).tolist() if K else []}
    retained=[x for x in audits if x['reason'].startswith('retained')]
    metadata['label_quality_quantiles']={key:np.quantile([x[key] for x in retained],[0,.5,1]).tolist() for key in ['points','rigid_rms_m','support_coverage','discordant_fraction','timing_budget_m','object_motion_m','robot_displacement_m']} if retained else {}
    public.write_report('flow_coverage.json',metadata)
    return {'arrays':arrays,'metadata':metadata}
