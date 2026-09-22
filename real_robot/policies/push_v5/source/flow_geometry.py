import numpy as np
import cv2
from geometry import camera, to_plane, project, brown


def support_candidates(rgb,cam,plane):
    # Observed occupied support, never complete actor boundary or topology.
    seed=brown(rgb);roi=np.zeros(seed.shape,np.uint8)
    cv2.fillPoly(roi,[np.array([[80,705],[235,25],[810,25],[1080,705]],np.int32)],1)
    seed*=roi
    n,lab,stats,cent=cv2.connectedComponentsWithStats(seed);out=[]
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    for j in range(1,n):
        x,y,w,h,area=stats[j]
        if not 250<area<70000 or w<10 or h<10:
            continue
        mask=(lab==j).astype(np.uint8)
        k=cv2.goodFeaturesToTrack(gray,100,.006,4,mask=mask*255,blockSize=5)
        if k is None or len(k)<5:
            continue
        uv=k[:,0,:];xy=to_plane(uv,cam,plane)
        yy,xx=np.where(mask>0);take=np.linspace(0,len(xx)-1,min(256,len(xx))).astype(int)
        occupied=to_plane(np.c_[xx[take],yy[take]],cam,plane)
        out.append({'mask':mask,'uv':uv,'xy':xy,'occupied':occupied,'bbox':[int(x),int(y),int(w),int(h)]})
    return out


def rigid(x,y):
    U,S,V=np.linalg.svd((x-x.mean(0)).T@(y-y.mean(0)))
    R=V.T@U.T
    if np.linalg.det(R)<0:
        V[-1]*=-1;R=V.T@U.T
    return R,y.mean(0)-x.mean(0)@R.T


def track_pair(rgb0,rgb1,uv,cam,plane):
    g0=cv2.cvtColor(rgb0,cv2.COLOR_RGB2GRAY);g1=cv2.cvtColor(rgb1,cv2.COLOR_RGB2GRAY)
    p=np.asarray(uv,np.float32).reshape(-1,1,2)
    q,ok,err=cv2.calcOpticalFlowPyrLK(g0,g1,p,None,winSize=(25,25),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
    if q is None:
        return None
    back,ok2,err2=cv2.calcOpticalFlowPyrLK(g1,g0,q,None,winSize=(25,25),maxLevel=4,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.01))
    good=(ok[:,0]>0)&(ok2[:,0]>0)&(np.linalg.norm(back[:,0]-p[:,0],axis=1)<.8)&(err[:,0]<22)
    x=to_plane(p[:,0],cam,plane);y=to_plane(q[:,0],cam,plane)
    if good.sum()<5:
        return None
    X=x[good];Y=y[good]
    M,inliers=cv2.estimateAffinePartial2D(X,Y,method=cv2.RANSAC,ransacReprojThreshold=.0025,maxIters=1000,confidence=.995)
    if M is None:
        return None
    use=inliers[:,0]>0
    if use.sum()<5 or use.mean()<.65:
        return None
    X=X[use];Y=Y[use];scale=float(np.linalg.norm(M[:,:2],axis=0).mean())
    if not .975<scale<1.025 or np.linalg.norm(np.ptp(X,axis=0))<.018:
        return None
    R,t=rigid(X,Y);errors=np.linalg.norm(X@R.T+t-Y,axis=1)
    rms=float(np.sqrt(np.mean(errors**2)))
    if rms>.002:
        return None
    selected=np.where(good)[0][use];spread=.004
    if len(X)>=8:
        r1,t1=rigid(X[::2],Y[::2]);r2,t2=rigid(X[1::2],Y[1::2])
        spread=float(np.max(np.linalg.norm(X@r1.T+t1-X@r2.T-t2,axis=1)))
        if spread>.005:
            return None
    # NEW general merged-input/unsupported extrapolation gate. Input occupancy
    # must lie near the span of rigidly co-moving corners, not merely share a
    # color component. Infer the entire CURRENT component(s) touched by uv;
    # never replace the input with future inliers. This gate applies identically
    # to every candidate, not a reviewed-index whitelist.
    mask=brown(rgb0)
    n,lab,stats,cent=cv2.connectedComponentsWithStats(mask)
    pix=np.rint(p[:,0]).astype(int)
    pix[:,0]=np.clip(pix[:,0],0,lab.shape[1]-1);pix[:,1]=np.clip(pix[:,1],0,lab.shape[0]-1)
    labels=lab[pix[:,1],pix[:,0]];labels=labels[labels>0]
    coverage=0.
    if len(labels):
        chosen_label=int(np.bincount(labels).argmax())
        yy,xx=np.where(lab==chosen_label)
        take=np.linspace(0,len(xx)-1,min(256,len(xx))).astype(int)
        occupied=to_plane(np.c_[xx[take],yy[take]],cam,plane)
        hull=cv2.convexHull(X.astype(np.float32).reshape(-1,1,2))
        distances=np.array([cv2.pointPolygonTest(hull,(float(v[0]),float(v[1])),True) for v in occupied])
        coverage=float(np.mean(distances>=-.006))
    if coverage<.85:
        return None
    # Static or independently moving tracked corners outside the consensus are
    # evidence against treating a color component as one rigid support.
    all_residual=np.linalg.norm(x[good]@R.T+t-y[good],axis=1)
    discordant=float(np.mean(all_residual>.004))
    if discordant>.12:
        return None
    motion=float(np.median(np.linalg.norm(Y-X,axis=1)))
    return {'R':R,'t':t,'rms':rms,'spread':spread,'motion':motion,'count':len(X),'uv0':p[selected,0],'uv1':q[selected,0],'xy0':X,'xy1':Y,'scale':scale,'support_coverage':coverage,'discordant_fraction':discordant}


def subsample(points,n=64):
    p=np.asarray(points,float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<5 or not np.isfinite(p).all():
        raise ValueError('Need visible occupied support points, not boundary completion')
    first=int(np.argmin(p[:,0]+.001*p[:,1]));chosen=[first]
    dist=np.sum((p-p[first])**2,axis=1)
    for j in range(1,n):
        k=int(np.argmax(dist));chosen.append(k);dist=np.minimum(dist,np.sum((p-p[k])**2,axis=1))
    return p[chosen]


def encode(points,R,t,ee_xyz,ee_axis,previous_delta=None,uncertainty=.008):
    p=subsample(points);center=p.mean(0);goal=p@R.T+t;g=goal.mean(0)-center
    theta=np.arctan2(g[1],g[0]) if np.linalg.norm(g)>.002 else 0.
    F=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    diameter=max(.025,float(np.linalg.norm(np.ptp(p,axis=0))))
    actor=np.c_[(p-center)@F/.1,(goal-p)@F/.05].astype(np.float32)
    prev=np.zeros(2) if previous_delta is None else np.asarray(previous_delta)
    ee=np.asarray(ee_xyz);axis=np.asarray(ee_axis)
    state=np.r_[(ee[:2]-center)@F/.1,(ee[2]-.068)/.1,axis[:2]@F,axis[2],prev@F/.02,float(previous_delta is not None),diameter,uncertainty/.01]
    return actor,np.clip(state,-10,10).astype(np.float32),F
