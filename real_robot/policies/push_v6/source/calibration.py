import numpy as np
import cv2
from real_robot.training_pipeline import public
from geometry import brown, table_roi, rays, project, unproject


def infer_calibration(tid,rec,meta):
    cal=meta['calibration'];c3=cal['third'];cw=cal['wrist']
    T3=np.array(c3['T_base_cam']);Tf=np.array(cw['T_flange_cam'])
    sift=cv2.SIFT_create(nfeatures=5000);evidence=[];counts=[]
    for a in [800,1600,3500,4800,5100,6900,8250]:
        im=public.rgb(tid,a,'third');iw=public.rgb(tid,a,'wrist');Tw=rec['T_base_flange'][a]@Tf
        mask=cv2.dilate(brown(im)*table_roi(im.shape),np.ones((11,11),np.uint8))*255
        kw,dw=sift.detectAndCompute(cv2.cvtColor(iw,cv2.COLOR_RGB2GRAY),None);seen=[];count=0
        for z0 in [None,.04,.06,.08]:
            if z0 is None:
                warp=im;wm=mask;Hi=np.eye(3)
            else:
                uv=np.array([[230,100],[800,100],[850,650],[100,650]],np.float32)
                xy=unproject(uv,c3,T3,z0);v=project(np.column_stack([xy,np.full(4,z0)]),cw,Tw).astype(np.float32)
                H=cv2.getPerspectiveTransform(uv,v);Hi=np.linalg.inv(H)
                warp=cv2.warpPerspective(im,H,(1280,720));wm=cv2.warpPerspective(mask,H,(1280,720))
            k,d=sift.detectAndCompute(cv2.cvtColor(warp,cv2.COLOR_RGB2GRAY),wm)
            if d is None or dw is None:
                continue
            for pair in cv2.BFMatcher().knnMatch(d,dw,k=2):
                if len(pair)<2:
                    continue
                m,n=pair
                if m.distance>.78*n.distance:
                    continue
                p=np.r_[k[m.queryIdx].pt,1.]@Hi.T;u=p[:2]/p[2];v=np.array(kw[m.trainIdx].pt)
                if any(np.linalg.norm(u-old)<5 for old in seen):
                    continue
                o,d1=rays([u],c3,T3);o2,d2=rays([v],cw,Tw)
                ab=np.linalg.lstsq(np.column_stack([d1[0],-d2[0]]),o2[0]-o[0],rcond=None)[0]
                x=o[0]+ab[0]*d1[0];y=o2[0]+ab[1]*d2[0];pt=(x+y)/2;err=float(np.linalg.norm(x-y))
                if err<.004 and -.02<pt[2]<.12 and min(ab)>0:
                    seen.append(u);count+=1
                    evidence.append({'index':a,'uv_third':u.tolist(),'uv_wrist':v.tolist(),'xyz_base':pt.tolist(),'ray_residual_m':err,'rectification_hypothesis_m':z0})
        counts.append({'index':a,'stereo_matches':count})
    z=np.array([p['xyz_base'][2] for p in evidence])
    if len(z)<2:
        public.write_report('calibration_inference.json',{'status':'insufficient_stereo','counts':counts,'matches':evidence})
        raise ValueError('Missing two-view height evidence')
    hist,bins=np.histogram(z,bins=np.arange(-.02,.122,.003));peak=(bins[np.argmax(hist)]+bins[np.argmax(hist)+1])/2
    inliers=abs(z-peak)<.008;zz=z[inliers];top=float(np.median(zz));sigma=max(.006,float(np.median(abs(zz-top))*1.4826))
    poses=sorted(set(p['index'] for p,ok in zip(evidence,inliers) if ok))
    if len(poses)<2:
        public.write_report('calibration_inference.json',{'status':'single_pose_height_only','counts':counts,'matches':evidence})
        raise ValueError('Height needs corroboration across poses')
    marks=[(800,[322,215]),(1600,[620,468]),(4800,[381,397]),(6900,[386,169])];As=[];bs=[]
    for a,u in marks:
        o,d=rays([u],c3,T3);T=rec['T_base_flange'][a];lo=(o[0]-T[:3,3])@T[:3,:3];ld=d[0]@T[:3,:3]
        M=np.eye(3)-np.outer(ld,ld);As.append(M);bs.append(M@lo)
    o,d=rays([[670,480]],cw,Tf);M=np.eye(3)-np.outer(d[0],d[0]);As.append(M);bs.append(M@o[0])
    tip=np.linalg.lstsq(np.concatenate(As),np.concatenate(bs),rcond=None)[0];tiperr=[]
    for a,u in marks:
        T=rec['T_base_flange'][a];x=T[:3,:3]@tip+T[:3,3];pred=project([x],c3,T3)[0]
        tiperr.append({'index':a,'marked_uv':u,'projected_uv':pred.tolist(),'error_px':float(np.linalg.norm(pred-u)),'tip_base':x.tolist()})
    report={'top_z_base_m':top,'top_sigma_m':sigma,'top_num_inliers':len(zz),'top_pose_support':poses,'stereo_counts':counts,'matches':evidence,'inferred_tip_flange_m':tip.tolist(),'tip_projection':tiperr,'status':'weak_inferred_not_commissioned','assumptions':'Common horizontal top-height hypothesis from sparse triangulated textured-label correspondences, not a fitted arbitrary plane. Multiple feature matches at one label are correlated. Minimum 6 mm uncertainty. Rectification aids appearance only; depths come from original rays. Tip endpoints visually marked. This approximate relative-geometry training reconstruction is not a deployment calibration.'}
    public.write_report('calibration_inference.json',report)
    for a in [800,4800,6900]:
        rgb=public.rgb(tid,a,'third').copy()
        for p,valid in zip(evidence,inliers):
            if p['index']==a and valid:
                cv2.circle(rgb,tuple(np.array(p['uv_third']).astype(int)),4,(0,255,0),-1)
        public.write_image('stereo_'+str(a)+'.png',rgb)
    return top,tip,report
