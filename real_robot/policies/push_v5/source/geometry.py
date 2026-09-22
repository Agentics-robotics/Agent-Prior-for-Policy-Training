import math
import numpy as np
import cv2
import torch
from scipy.spatial import cKDTree
from real_robot.training_pipeline import public


def camera(cal, name, flange=None):
    c = cal[name]
    T = np.asarray(c['T_base_cam'] if name == 'third' else flange @ np.asarray(c['T_flange_cam']), dtype=float)
    return np.asarray(c['K'], float), np.asarray(c['dist'], float), T


def rays(uv, cam):
    K,d,T = cam
    xy = cv2.undistortPoints(np.asarray(uv,float).reshape(-1,1,2),K,d).reshape(-1,2)
    v = np.c_[xy, np.ones(len(xy))] @ T[:3,:3].T
    v /= np.linalg.norm(v,axis=1,keepdims=True)
    return T[:3,3],v


def project(xyz,cam):
    K,d,T=cam
    q=(np.asarray(xyz)-T[:3,3]) @ T[:3,:3]
    return cv2.projectPoints(q,np.zeros(3),np.zeros(3),K,d)[0].reshape(-1,2)


def to_plane(uv,cam,plane):
    o,v=rays(uv,cam)
    n=np.r_[-np.asarray(plane)[:2],1.]
    t=(plane[2]-o @ n)/(v @ n)
    return (o+v*t[:,None])[:,:2]


def brown(rgb):
    x=rgb.astype(float)/255.
    # Broad proposal only; not a final mask or identity classifier.
    b=((x[:,:,0]-x[:,:,2]>.045)&(x[:,:,1]-x[:,:,2]>.025)&(x[:,:,0]>x[:,:,1]*.95)).astype(np.uint8)
    return cv2.morphologyEx(b,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))


def fit_tool(spec):
    ann=public.source_json('annotations.json')['tool_points']
    A=[];b=[];entries=[]
    for p in ann:
        tid=spec['trajectory_ids'][p['episode']]
        rec=public.records(tid);cal=public.metadata(tid)['calibration']
        F=np.asarray(rec['T_base_flange'][p['index']])
        for name in ['third','wrist']:
            cam=camera(cal,name,F)
            o,v=rays([p[name]],cam)
            of=(o-F[:3,3]) @ F[:3,:3]
            vf=v[0] @ F[:3,:3]
            Q=np.eye(3)-np.outer(vf,vf)
            A.append(Q);b.append(Q@of)
            entries.append((p,name,cam,F))
    M=np.concatenate(A);y=np.concatenate(b)
    pt=np.linalg.lstsq(M,y,rcond=None)[0]
    residual=[];loo=[]
    for k,(p,name,cam,F) in enumerate(entries):
        pred=project([F[:3,:3]@pt+F[:3,3]],cam)[0]
        residual.append({'episode':p['episode'],'index':p['index'],'camera':name,'predicted_uv':pred.tolist(),'annotated_uv':p[name],'error_px':float(np.linalg.norm(pred-p[name]))})
    for j in range(len(ann)):
        keep=[k for k in range(len(A)) if k//2!=j]
        loo.append(np.linalg.lstsq(np.concatenate([A[k] for k in keep]),np.concatenate([b[k] for k in keep]),rcond=None)[0])
    rms=float(np.sqrt(np.mean((M@pt-y)**2))*np.sqrt(3))
    spread=float(np.max(np.linalg.norm(np.asarray(loo)-pt,axis=1)))
    report={'flange_point_m':pt.tolist(),'ray_rms_m':rms,'leave_pair_out_max_m':spread,'projection_residuals':residual,'certified':False,'uncertainty_floor_m':max(.004,rms,spread),'meaning':'Inferred visible cap point only. Does not recover support surface, shaft dimensions, full envelope or table clearance.'}
    return pt,report


def stereo_pair(tid,i):
    cal=public.metadata(tid)['calibration'];rec=public.records(tid)
    cams=[camera(cal,'third'),camera(cal,'wrist',rec['T_base_flange'][i])]
    imgs=[public.rgb(tid,i,c) for c in ['third','wrist']]
    sift=cv2.SIFT_create(nfeatures=3500,contrastThreshold=.012)
    kp=[];ds=[]
    for img in imgs:
        mask=brown(img)*255
        k,d=sift.detectAndCompute(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY),mask)
        kp.append(k);ds.append(d)
    if ds[0] is None or ds[1] is None:
        return np.zeros((0,3)),{'index':i,'matches':0,'accepted':0}
    matches=cv2.BFMatcher().knnMatch(ds[0],ds[1],k=2)
    pairs=[m[0] for m in matches if len(m)==2 and m[0].distance<.72*m[1].distance]
    if len(pairs)<4:
        return np.zeros((0,3)),{'index':i,'matches':len(pairs),'accepted':0}
    u=[np.array([kp[0][m.queryIdx].pt for m in pairs]),np.array([kp[1][m.trainIdx].pt for m in pairs])]
    norm=[cv2.undistortPoints(u[k].reshape(-1,1,2),cams[k][0],cams[k][1]).reshape(-1,2) for k in range(2)]
    ext=[np.linalg.inv(c[2])[:3] for c in cams]
    X=cv2.triangulatePoints(ext[0],ext[1],norm[0].T,norm[1].T).T
    X=X[:,:3]/X[:,3:4]
    err=np.maximum(np.linalg.norm(project(X,cams[0])-u[0],axis=1),np.linalg.norm(project(X,cams[1])-u[1],axis=1))
    keep=np.isfinite(X).all(1)&(X[:,2]>-.03)&(X[:,2]<.13)&(X[:,0]>.2)&(X[:,0]<.9)&(np.abs(X[:,1])<.5)&(err<4.5)
    return X[keep],{'index':i,'matches':len(pairs),'accepted':int(keep.sum()),'median_reprojection_px':float(np.median(err[keep])) if keep.any() else None}


def fit_plane(points):
    x=np.asarray(points,float)
    if len(x)<20:
        return None,{'points':len(x),'status':'insufficient_stereo'}
    rng=np.random.default_rng(1809);best=np.zeros(len(x),bool)
    for k in range(400):
        take=rng.choice(len(x),3,replace=False)
        ab=np.linalg.lstsq(np.c_[x[take,:2],np.ones(3)],x[take,2],rcond=None)[0]
        if np.linalg.norm(ab[:2])>.15:
            continue
        res=np.abs(np.c_[x[:,:2],np.ones(len(x))]@ab-x[:,2])
        mask=res<.004
        if mask.sum()>best.sum():
            best=mask
    if best.sum()<20:
        return None,{'points':len(x),'status':'no_supported_plane'}
    p=np.linalg.lstsq(np.c_[x[best,:2],np.ones(best.sum())],x[best,2],rcond=None)[0]
    residual=np.abs(np.c_[x[:,:2],np.ones(len(x))]@p-x[:,2])
    return p,{'points':len(x),'inliers':int(best.sum()),'plane_z_ax_by_c':p.tolist(),'inlier_rms_m':float(np.sqrt(np.mean(residual[best]**2))),'xy_span_m':np.ptp(x[best,:2],axis=0).tolist(),'status':'inferred_top_surface_not_table','all_z_quantiles':np.quantile(x[:,2],[0,.1,.5,.9,1]).tolist(),'uncertainty_floor_m':.005}


class MaskRecovery:
    def __init__(self,device):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        net=build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',public.asset_path('sam2.1_hiera_small.pt'),device=device,apply_postprocessing=False)
        self.predictor=SAM2ImagePredictor(net)

    def run(self,rgb):
        seed=brown(rgb)
        # The authorized scene's tabletop, not a glyph template.
        roi=np.zeros_like(seed)
        cv2.fillPoly(roi,[np.array([[80,690],[250,25],[790,25],[1070,690]],np.int32)],1)
        seed*=roi
        n,lab,stats,cent=cv2.connectedComponentsWithStats(seed)
        parts=[]
        for k in range(1,n):
            area=int(stats[k,4]);x,y,w,h=stats[k,:4]
            if 500<area<70000 and w>14 and h>14:
                parts.append((k,x,y,w,h,area))
        self.predictor.set_image(rgb)
        results=[]
        for k,x,y,w,h,area in parts:
            inside=(lab[y:y+h,x:x+w]==k).astype(np.uint8)
            dt=cv2.distanceTransform(inside,cv2.DIST_L2,3)
            yy,xx=np.unravel_index(np.argmax(dt),dt.shape)
            masks,scores,_=self.predictor.predict(point_coords=np.array([[x+xx,y+yy]],float),point_labels=np.ones(1,int),box=np.array([x-8,y-8,x+w+8,y+h+8]),multimask_output=True)
            # Select by proposal support and bounded expansion, not SAM confidence alone.
            rank=[]
            original=(lab==k)
            for m,s in zip(masks,scores):
                a=int(m.sum());over=int((m&original).sum())
                rank.append(over/max(area,1)-.15*abs(math.log(max(a,1)/area))+float(s)*.1 if 400<a<85000 else -100.)
            j=int(np.argmax(rank));m=masks[j].astype(np.uint8)
            if rank[j]<.4:
                continue
            if any((m&r['mask']).sum()/max(1,min(m.sum(),r['mask'].sum()))>.8 for r in results):
                continue
            contours,hierarchy=cv2.findContours(m,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
            loops=[]
            if hierarchy is not None:
                for j,c in enumerate(contours):
                    if cv2.contourArea(c)>80:
                        loops.append((c[:,0,:].astype(float),int(hierarchy[0,j,3]>=0)))
            results.append({'mask':m,'loops':loops,'score':float(scores[j]) if j<len(scores) else 0.,'seed_area':area,'mask_area':int(m.sum())})
        return results,seed


def boundary(loops,cam,plane,n=64):
    metric=[(to_plane(p,cam,plane),hole) for p,hole in loops]
    lengths=[np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1).sum() for p,h in metric]
    if not lengths or sum(lengths)<.04:
        return None
    counts=np.maximum(4,np.array(lengths)/sum(lengths)*n).astype(int)
    while counts.sum()>n:
        k=int(np.argmax(counts));counts[k]-=1
    while counts.sum()<n:
        k=int(np.argmax(np.array(lengths)/counts));counts[k]+=1
    points=[];normal=[];flags=[]
    for (p,h),cnt in zip(metric,counts):
        delta=np.roll(p,-1,axis=0)-p;dist=np.linalg.norm(delta,axis=1)
        cum=np.r_[0,np.cumsum(dist)]
        idx=np.minimum(np.searchsorted(cum,np.arange(cnt)*cum[-1]/cnt,side='right')-1,len(p)-1)
        p2=p[idx]
        tangent=np.roll(p2,-1,axis=0)-np.roll(p2,1,axis=0)
        signed=np.sum(p2[:,0]*np.roll(p2[:,1],-1)-p2[:,1]*np.roll(p2[:,0],-1))
        # Outward toward free space: exterior CCW -> right; inner -> left.
        nor=np.c_[tangent[:,1],-tangent[:,0]]* (1 if signed>=0 else -1) * (-1 if h else 1)
        nor/=np.maximum(np.linalg.norm(nor,axis=1,keepdims=True),1e-8)
        points.extend(p2);normal.extend(nor);flags.extend([h]*cnt)
    return np.c_[np.array(points),np.array(normal),np.array(flags)]


def register(a,b):
    # Partial silhouette registration; topology mismatch alone is not an exclusion.
    best=None
    ta=cKDTree(b)
    for angle in [0.,-.15,.15,-.35,.35]:
        R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        t=np.mean(b,axis=0)-np.mean(a@R.T,axis=0)
        for k in range(12):
            moved=a@R.T+t
            d,ix=ta.query(moved)
            use=d<=np.quantile(d,.8)
            x=moved[use];y=b[ix[use]]
            xc=x-x.mean(0);yc=y-y.mean(0)
            U,S,V=np.linalg.svd(xc.T@yc)
            Q=V.T@U.T
            if np.linalg.det(Q)<0:
                V[-1]*=-1;Q=V.T@U.T
            dt=y.mean(0)-x.mean(0)@Q.T
            R=Q@R;t=t@Q.T+dt
        d,_=ta.query(a@R.T+t)
        rev,_=cKDTree(a@R.T+t).query(b)
        err=float(.5*(np.mean(np.sort(d)[:int(.85*len(d))])+np.mean(np.sort(rev)[:int(.85*len(rev))])))
        if best is None or err<best[2]:
            best=(R,t,err)
    return best


def features(shape,R,t,tool_radius=.005,uncertainty=.005):
    p=shape[:,:2];normal=shape[:,2:4];flags=shape[:,4]
    center=p.mean(0);diam=max(float(np.linalg.norm(np.ptp(p,axis=0))),.03)
    goal=p@R.T+t
    gd=goal.mean(0)-center
    theta=math.atan2(gd[1],gd[0]) if np.linalg.norm(gd)>.005 else 0.
    F=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    pc=(p-center)@F/diam;nc=normal@F;gv=(goal-p)@F/diam
    actor=np.c_[pc,nc,gv,flags].astype(np.float32)
    directions=[];contacts=[];lengths=[];rows=[];loop=[]
    yaw=math.atan2(R[1,0],R[0,0])
    for i in range(len(p)):
        for ang in [-.7,-.35,0.,.35,.7]:
            C=np.array([[math.cos(ang),-math.sin(ang)],[math.sin(ang),math.cos(ang)]])
            v=-normal[i]@C.T;vc=v@F
            for length in [.005,.01,.02]:
                length=min(length,.2*diam)
                feat=np.r_[pc[i],nc[i],gv[i],gd@F/diam,math.sin(yaw),math.cos(yaw),vc,length/diam,np.cross(pc[i],vc),diam,tool_radius,uncertainty,flags[i]]
                rows.append(feat);contacts.append(p[i]);directions.append(v);lengths.append(length);loop.append(int(flags[i]))
    return actor,np.asarray(rows,np.float32),{'points':np.array(contacts),'directions':np.array(directions),'lengths':np.array(lengths),'loops':np.array(loop),'diameter':diam}
