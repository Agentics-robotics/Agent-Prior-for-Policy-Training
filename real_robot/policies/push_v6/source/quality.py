import numpy as np
import torch
import cv2
from scipy.spatial import cKDTree
from real_robot.training_pipeline import public
from geometry import make_sam,SAM2ImagePredictor,shape_from_mask,rotation,register,project
from perception import reference_proposals
from tracking import material_area


def qualify_actor(actor,references):
    area=material_area(actor);best=1.
    for ref in references:
        if not .85<area/max(material_area(ref),1.e-7)<1.18:
            continue
        for angle in [0,np.pi/2,np.pi,3*np.pi/2]:
            p=ref['xy']@rotation(angle).T;error,R,t=register(p,actor['xy']);fit=p@R.T+t
            a=cKDTree(fit).query(actor['xy'])[0];b=cKDTree(actor['xy']).query(fit)[0];best=min(best,error)
            if error<.006 and np.mean(a<.008)>.85 and np.mean(b<.008)>.85:
                return True,float(error)
    return False,float(best)


def filter_prepared(prepared,spec):
    arrays=prepared['arrays'];meta=prepared['metadata'];top=meta['label_provenance']['calibration']['top_z_base_m'];segments=spec['data_plan']['segments'];predictor=SAM2ImagePredictor(make_sam(spec['device']));keep=[];audit=[];removed={}
    for si in np.unique(arrays['example_segment']):
        seg=segments[int(si)];tid=seg['trajectory_id'];cam=public.metadata(tid)['calibration']['third'];T=np.array(cam['T_base_cam']);rgb=public.rgb(tid,seg['start'],'third')
        with torch.no_grad():
            masks=reference_proposals(rgb,predictor)
        refs=[]
        for m in masks:
            sh=shape_from_mask(m,cam,T,top)
            if sh is not None:
                refs.append(sh)
        retained=0;removed[seg['segment_id']]=0
        for k in np.where(arrays['example_segment']==si)[0]:
            actor={'xy':arrays['xy'][k],'normal':arrays['normal'][k],'loop':arrays['loop'][k]};ok,error=qualify_actor(actor,refs);a=int(arrays['example_source_index'][k])
            entry={'episode':tid,'segment':seg['segment_id'],'anchor':a,'keep':ok,'best_complete_causal_reference_residual_m':error,'reason':'supported_complete_shape' if ok else 'incomplete_or_unresolved_actor_reference'};audit.append(entry)
            if ok:
                keep.append(int(k));retained+=1
                if retained<=2:
                    im=public.rgb(tid,a,'third').copy();xy=actor['xy'];uv=project(np.column_stack([xy,np.full(len(xy),top)]),cam,T).astype(int)
                    for u in uv:
                        cv2.circle(im,tuple(u),2,(0,220,0),-1)
                    cp=arrays['target_contact'][k];d=arrays['target_direction'][k];L=float(arrays['target_length'][k]);uv=project([[cp[0],cp[1],top],[cp[0]+d[0]*L,cp[1]+d[1]*L,top]],cam,T).astype(int)
                    cv2.arrowedLine(im,tuple(uv[0]),tuple(uv[1]),(255,0,0),3);name='final_'+str(int(si))+'_'+str(a)+'.png';public.write_image(name,im);entry['image']=name
            else:
                removed[seg['segment_id']]+=1
    keep=np.array(sorted(keep),np.int64);out={k:v[keep] for k,v in arrays.items()}
    for si in np.unique(out['example_segment']):
        ix=out['example_segment']==si;out['policy_weight'][ix,0]=1./int(ix.sum())
    meta['num_examples']=len(keep)
    for s in meta['coverage']['segments']:
        n=removed.get(s['segment_id'],0);s['retained']-=n;s['reasons']['incomplete_or_unresolved_actor_reference']=n;sid=[i for i,v in enumerate(segments) if v['segment_id']==s['segment_id']][0];ix=out['example_segment']==sid;s['partial_contact_only']=int(np.sum(out['target_validity'][ix,2]==0))
    old=dict(meta['data_audit']);meta['data_audit']={'pre_completeness_audit':old,'retained':len(keep),'removed_incomplete_or_unresolved':len(arrays['point'])-len(keep),'partial_target_rows':int(np.sum(out['target_validity'][:,2]==0)),'inner_loop_labels':int(np.sum([out['loop'][i,np.argmin(np.linalg.norm(out['xy'][i]-out['target_contact'][i],axis=1))]>0 for i in range(len(keep))])),'recovery':'All actors require a complete causal initial-shape match: area .85..1.18, residual <6mm, >85% bidirectional support within 8mm. Reference masks use whole foreground-component boxes, since point proposals can be subparts even before contact. Current masks remain separate point proposals. No whitelist or future reference. Shaft localization allows gray tool over darker brown material instead of stopping at its upper edge. Unresolved shapes remain context, not gradients.'}
    meta['fixtures']=[];meta['label_provenance']['completeness']='Final actor-only completeness uses cut-start whole-component SAM references. Obstacle masks remain uncertain contextual cues, not collision certification.'
    public.write_report('final_quality_audit.json',audit);public.write_report('quality_summary.json',meta['data_audit']);public.write_report('coverage.json',meta)
    return {'arrays':out,'metadata':meta}
