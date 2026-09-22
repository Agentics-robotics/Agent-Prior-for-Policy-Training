import numpy as np
import cv2
from real_robot.training_pipeline import public
from geometry import brown, camera, fit_plane as stereo_plane, MaskRecovery as OriginalMaskRecovery


class MaskRecovery(OriginalMaskRecovery):
    def run(self,rgb):
        seed=brown(rgb)
        roi=np.zeros_like(seed)
        cv2.fillPoly(roi,[np.array([[80,690],[250,25],[790,25],[1070,690]],np.int32)],1)
        seed*=roi
        n,lab,stats,cent=cv2.connectedComponentsWithStats(seed)
        self.predictor.set_image(rgb)
        results=[]
        for k in range(1,n):
            x,y,w,h,area=stats[k]
            if not (500<int(area)<70000 and w>14 and h>14):
                continue
            inside=(lab[y:y+h,x:x+w]==k).astype(np.uint8)
            dt=cv2.distanceTransform(inside,cv2.DIST_L2,3)
            yy,xx=np.unravel_index(np.argmax(dt),dt.shape)
            masks,scores,_=self.predictor.predict(point_coords=np.array([[x+xx,y+yy]],float),point_labels=np.ones(1,int),box=np.array([x-8,y-8,x+w+8,y+h+8]),multimask_output=True)
            masks=np.asarray(masks)>0
            rank=[];original=(lab==k)
            for m,s in zip(masks,scores):
                a=int(m.sum());over=int((m&original).sum())
                rank.append(over/max(int(area),1)-.15*abs(np.log(max(a,1)/area))+float(s)*.1 if 400<a<85000 else -100.)
            best=int(np.argmax(rank));m=masks[best].astype(np.uint8);score=float(scores[best])
            if rank[best]<.4 or any((m&r['mask']).sum()/max(1,min(m.sum(),r['mask'].sum()))>.8 for r in results):
                continue
            contours,hierarchy=cv2.findContours(m,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
            loops=[]
            if hierarchy is not None:
                for j,c in enumerate(contours):
                    if cv2.contourArea(c)>80:
                        loops.append((c[:,0,:].astype(float),int(hierarchy[0,j,3]>=0)))
            results.append({'mask':m,'loops':loops,'score':score,'seed_area':int(area),'mask_area':int(m.sum())})
        return results,seed


def uv_undistorted(X,cam,scale):
    K,d,T=cam
    q=(X-T[:3,3])@T[:3,:3]
    pix=q@K.T
    return (pix[:,:2]/pix[:,2:3]*scale).astype(np.float32)


def silhouette_sweep(tid,i):
    rec=public.records(tid);cal=public.metadata(tid)['calibration']
    cams=[camera(cal,'third'),camera(cal,'wrist',rec['T_base_flange'][i])]
    size=(320,180);scale=.25
    masks=[]
    for name,cam in zip(['third','wrist'],cams):
        img=public.rgb(tid,i,name)
        und=cv2.undistort(img,cam[0],cam[1])
        masks.append(cv2.resize(brown(und),size,interpolation=cv2.INTER_NEAREST))
    # Evaluate silhouette correspondence over actual overlapping FOV, not texture matches.
    scores=[];heights=np.arange(.0,.122,.002)
    xy=np.array([[.25,-.4],[.85,-.4],[.85,.4],[.25,.4]],float)
    valid0=np.ones((180,320),np.uint8)
    for z in heights:
        X=np.c_[xy,np.full(4,z)]
        H=cv2.getPerspectiveTransform(uv_undistorted(X,cams[0],scale),uv_undistorted(X,cams[1],scale))
        warped=cv2.warpPerspective(masks[0],H,size,flags=cv2.INTER_NEAREST)
        valid=cv2.warpPerspective(valid0,H,size,flags=cv2.INTER_NEAREST)>0
        valid[:8]=False;valid[-8:]=False;valid[:,:8]=False;valid[:,-8:]=False
        a=warped>0;b=masks[1]>0
        union=((a|b)&valid).sum();intersection=((a&b)&valid).sum()
        scores.append(float(intersection/max(1,union)))
    k=int(np.argmax(scores));z=float(heights[k])
    near=np.where(np.asarray(scores)>scores[k]-.025)[0]
    width=float(np.ptp(heights[near])) if len(near) else .12
    reliable=scores[k]>.55 and width<.02 and 0<k<len(heights)-1
    return {'index':i,'height_m':z,'IoU':scores[k],'near_peak_width_m':width,'accepted':bool(reliable),'scores':scores}


def fit_plane(points):
    plane,report=stereo_plane(points)
    if plane is not None:
        return plane,report
    tids=[]
    for s in public.segments():
        if s['trajectory_id'] not in tids:
            tids.append(s['trajectory_id'])
    ann=public.source_json('annotations.json');rows=[]
    for ep,tid in enumerate(tids):
        for i in ann['review_indices'][ep][::3]:
            r=silhouette_sweep(tid,i);r['episode']=ep;rows.append(r)
    good=[r for r in rows if r['accepted']]
    summary={'stereo_attempt':report,'method':'cross-view silhouette plane-height sweep, not board/table assumption','assumption':'piece top surfaces near base-horizontal; slope/table height remain unverified','pairs':rows,'accepted_pairs':len(good)}
    if len(good)<6:
        summary['status']='insufficient_cross_view_silhouette_agreement'
        public.write_report('silhouette_plane_recovery.json',summary)
        return None,summary
    zs=np.array([r['height_m'] for r in good]);med=float(np.median(zs));spread=float(np.median(np.abs(zs-med)))
    summary['median_height_m']=med;summary['MAD_m']=spread
    summary['plane_z_ax_by_c']=[0.,0.,med]
    summary['uncertainty_floor_m']=max(.005,2*spread)
    # A scattered plane fit is not repaired by just loosening contact gates.
    if spread>.008:
        summary['status']='inconsistent_surface_heights'
        public.write_report('silhouette_plane_recovery.json',summary)
        return None,summary
    summary['status']='weak_horizontal_surface_hypothesis_requires_review'
    public.write_report('silhouette_plane_recovery.json',summary)
    return np.array([0.,0.,med]),summary
