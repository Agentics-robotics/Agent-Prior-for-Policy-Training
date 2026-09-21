"""Shared causal geometry. Board-plane results are proxies, not metrology."""
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
S=256
BOUNDS=[-.2,-.35,.8,.65]
DX=1./S
HISTORY=[11,7,3,0]

def finite(x,shape=None):
    a=np.asarray(x,dtype=np.float32)
    if shape is not None: a=a.reshape(shape)
    return np.nan_to_num(a,nan=0.,posinf=0.,neginf=0.)

def workspace(metadata,supplied=None):
    if supplied is not None:
        T=np.asarray(supplied['T_base_W'],np.float64).reshape(4,4)
        if not np.isfinite(T).all() or np.linalg.norm(T[3]-[0,0,0,1])>1e-6 or np.linalg.norm(T[:3,:3].T@T[:3,:3]-np.eye(3))>.01 or np.linalg.det(T[:3,:3])<.99:
            raise ValueError('T_base_W must be a finite right-handed rigid transform')
        sigma=float(supplied.get('plane_sigma_m',.03))
        if not np.isfinite(sigma) or sigma<0: raise ValueError('Invalid plane uncertainty')
        return T,sigma,bool(supplied.get('verified',False))
    B=np.asarray(metadata['calibration']['T_base_board'],np.float64); z=B[:3,2].copy()
    if z[2]<0: z=-z
    x=B[:3,0]-z*np.dot(z,B[:3,0]); x/=np.linalg.norm(x)
    T=np.eye(4); T[:3,:3]=np.stack([x,np.cross(z,x),z],1); T[:3,3]=B[:3,3]
    return T,.03,False

def pix_to_w(p): return np.asarray(p)*DX+np.asarray(BOUNDS[:2])
def w_to_pix(p): return (np.asarray(p)-np.asarray(BOUNDS[:2]))/DX
def base_to_w(p,T): return (np.asarray(p)-T[:3,3])@T[:3,:3]

def camera(obs,name):
    c=obs['metadata']['calibration'][name]
    C=np.asarray(c['T_base_cam']) if name=='third' else np.asarray(obs['T_base_flange']).reshape(4,4)@np.asarray(c['T_flange_cam'])
    return c,C

def project_pixels(points,obs,name,T):
    c,C=camera(obs,name); uv=cv2.undistortPoints(np.asarray(points,np.float64).reshape(-1,1,2),np.asarray(c['K']),np.asarray(c['dist'])).reshape(-1,2)
    rays=np.concatenate([uv,np.ones((len(uv),1))],1)@C[:3,:3].T; n=T[:3,2]; den=rays@n
    s=((T[:3,3]-C[:3,3])@n)/np.where(np.abs(den)>1e-8,den,1e-8)
    return w_to_pix(base_to_w(C[:3,3]+rays*s[:,None],T)[:,:2])

def warp_view(obs,name,T):
    # Inverse-distortion remapping implements undistort then plane warp in one
    # interpolation. Sampling coordinates refer to original RGB pixels.
    c,C=camera(obs,name); yy,xx=np.mgrid[:S,:S]
    p=np.stack([xx*DX+BOUNDS[0],yy*DX+BOUNDS[1],np.zeros_like(xx)],-1)
    b=p.reshape(-1,3)@T[:3,:3].T+T[:3,3]; q=(b-C[:3,3])@C[:3,:3]
    xy,_=cv2.projectPoints(q,np.zeros(3),np.zeros(3),np.asarray(c['K']),np.asarray(c['dist'])); xy=xy.reshape(S,S,2).astype(np.float32)
    rgb=np.asarray(obs[name+'_rgb'],np.uint8)
    valid=(q[:,2].reshape(S,S)>.025)&(xy[:,:,0]>=1)&(xy[:,:,0]<rgb.shape[1]-2)&(xy[:,:,1]>=1)&(xy[:,:,1]<rgb.shape[0]-2)
    im=cv2.remap(rgb,xy[:,:,0],xy[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT); hsv=cv2.cvtColor(im,cv2.COLOR_RGB2HSV)
    wood=(hsv[:,:,0]>=7)&(hsv[:,:,0]<=39)&(hsv[:,:,1]>=45)&(hsv[:,:,2]>=32)&(hsv[:,:,2]<=235)&valid
    return wood.astype(np.uint8),valid.astype(np.uint8)

def perceive(obs,T,external=None):
    if external is not None:
        m=np.asarray(external,np.uint8)
        if m.shape!=(S,S): raise ValueError('live_mask must be 256x256')
        return (m>0).astype(np.uint8),np.ones((S,S),np.uint8),0.
    a,va=warp_view(obs,'third',T); b,vb=warp_view(obs,'wrist',T)
    support=cv2.dilate(a,np.ones((7,7),np.uint8)); merged=a|(b&support)
    merged=cv2.morphologyEx(merged,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,lab,st,_=cv2.connectedComponentsWithStats(merged,connectivity=8); out=np.zeros_like(merged)
    for j in range(1,n):
        if 12<=st[j,cv2.CC_STAT_AREA]<=4500: out[lab==j]=1
    union=np.maximum(a,b).sum(); dis=float(((a!=b)&(va>0)&(vb>0)).sum()/max(1,union))
    return out,va|vb,min(1.,dis)

def components(mask):
    n,lab,st,c=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
    return [(lab==i).astype(np.uint8) for i in range(1,n) if 12<=st[i,4]<=4500]

def center(mask):
    y,x=np.nonzero(mask)
    return np.array([x.mean(),y.mean()],np.float32) if len(x) else np.zeros(2,np.float32)

def boundary(mask,n=64):
    cs,h=cv2.findContours(mask.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE); pts=[]; holes=[]
    for j,c in enumerate(cs):
        if len(c)<4: continue
        pts.extend(c[:,0,:].tolist()); holes.extend([int(h[0,j,3]>=0)]*len(c))
    if not pts: return np.zeros((n,2),np.float32),np.zeros(n,np.float32)
    ix=np.linspace(0,len(pts)-1,n).astype(int)
    return np.asarray(pts,np.float32)[ix],np.asarray(holes,np.float32)[ix]

def moved(mask,dx,dy,angle=0.):
    M=cv2.getRotationMatrix2D(tuple(float(v) for v in center(mask)),float(angle),1.); M[:,2]+=np.array([dx,dy])
    return cv2.warpAffine(mask,M,(S,S),flags=cv2.INTER_NEAREST)

def init_tracks(mask,seeds=None):
    cs=components(mask)
    if not cs: return []
    if seeds is None: cs=[cs[i] for i in np.argsort([center(c)[0] for c in cs])]
    else:
        cost=np.array([[np.linalg.norm(center(c)-p) for c in cs] for p in seeds]); ri,ci=linear_sum_assignment(cost)
        assigned={int(i):cs[j] for i,j in zip(ri,ci)}; cs=[assigned.get(i,np.zeros((S,S),np.uint8)) for i in range(len(seeds))]
    return [{'mask':c,'reference':c.copy(),'pose':np.eye(3),'confidence':float(c.sum()>12),'miss':0,'angle':0.} for c in cs]

def update_tracks(tracks,fg):
    claimed=np.zeros_like(fg); updated=[]
    for tr in tracks:
        old=np.asarray(tr['mask'],np.uint8); area=int(old.sum()); ref=tr['reference']; A=tr['pose']
        if area<12:
            updated.append({'mask':old,'reference':ref,'pose':A,'confidence':0.,'miss':tr['miss']+1,'angle':tr['angle']}); continue
        ys,xs=np.nonzero(old); pad=6+min(6,tr['miss']//3)
        l=max(0,int(xs.min())-pad); r=min(S,int(xs.max())+pad+1); t=max(0,int(ys.min())-pad); b=min(S,int(ys.max())+pad+1)
        free=(fg*(1-claimed)).astype(np.float32); best=(-1.,0.,old,A)
        for ang in [-4.,0.,4.]:
            M=np.eye(3); M[:2]=cv2.getRotationMatrix2D(tuple(float(v) for v in center(old)),float(ang),1.)
            Ar=M@A; rot=cv2.warpAffine(ref,Ar[:2],(S,S),flags=cv2.INTER_NEAREST); yy,xx=np.nonzero(rot)
            if len(xx)<12: continue
            xl,xr=int(xx.min()),int(xx.max())+1; yt,yb=int(yy.min()),int(yy.max())+1
            tmp=rot[yt:yb,xl:xr].astype(np.float32); region=free[t:b,l:r]
            if region.shape[0]<tmp.shape[0] or region.shape[1]<tmp.shape[1]: continue
            scores=cv2.matchTemplate(region,tmp,cv2.TM_CCORR)/max(1.,tmp.sum()); sy,sx=np.mgrid[:scores.shape[0],:scores.shape[1]]
            scores-=.006*((sx+l-xl)**2+(sy+t-yt)**2)+.001*abs(ang)
            _,val,_,loc=cv2.minMaxLoc(scores); Ac=Ar.copy(); Ac[:2,2]+=np.array([loc[0]+l-xl,loc[1]+t-yt])
            cand=cv2.warpAffine(ref,Ac[:2],(S,S),flags=cv2.INTER_NEAREST)
            if val>best[0]: best=(val,ang,cand,Ac)
        val,ang,cand,Ac=best; visible=float((cand*free).sum()/max(1,area)); conf=float(np.clip((visible-.2)/.7,0,1)); miss=0 if conf>.45 else tr['miss']+1
        if conf<.25: cand=old; ang=0.; Ac=A
        # Render the acquisition template through accumulated SE2, avoiding
        # repeated raster resampling/erosion. No future or shape-class template.
        conf*=float(np.exp(-miss/12.)); claimed|=cand
        updated.append({'mask':cand,'reference':ref,'pose':Ac,'confidence':conf,'miss':miss,'angle':float(tr['angle']-ang*np.pi/180.)})
    return updated

def scene_labels(tracks):
    labels=np.zeros((S,S),np.uint8)
    for j,tr in enumerate(tracks): labels[np.asarray(tr['mask'])>0]=j+1
    return labels

def sdf(m):
    m=np.asarray(m)>0
    if not m.any(): return np.ones(m.shape,np.float32)
    return np.clip((distance_transform_edt(~m)-distance_transform_edt(m))*DX/.08,-1,1).astype(np.float32)

def register_masks(a,b):
    ca,cb=center(a),center(b)
    if a.sum()<12 or b.sum()<12: return np.zeros(3,np.float32),0.,True
    vals=[]
    for theta in np.arange(-180,180,5):
        w=moved(a,*(cb-ca),theta); vals.append(float((w*b).sum()/max(1,(w|b).sum())))
    best=max(vals); ties=[i for i,v in enumerate(vals) if v>=best-.015]; chosen=min(ties,key=lambda i:abs(i*5-180)); theta=chosen*5-180
    amb=any(abs(((i*5-180-theta+180)%360)-180)>25 for i in ties)
    return np.array([*(pix_to_w(cb)-pix_to_w(ca)),-theta*np.pi/180],np.float32),best,amb

def rotation_error(R,G):
    v,_=cv2.Rodrigues(np.asarray(G)@np.asarray(R).T); return v.reshape(3).astype(np.float32)

def robot_vector(obs,last):
    E=finite(obs['T_base_ee'],(4,4)); F=finite(obs['T_base_flange'],(4,4)); vals=[]
    for key,scale in [('q',3.),('dq',1.),('tau_ext',10.)]:
        raw=np.asarray(obs.get(key,np.zeros(7)),np.float32).reshape(-1)[:7]; vals.extend((finite(raw)/scale).tolist()); vals.extend(np.isfinite(raw).astype(float).tolist())
    vals.extend(E[:3,:3].reshape(-1).tolist()); vals.extend((F[:3,3]-E[:3,3]).tolist()); vals.extend(finite(last,(6,)).tolist())
    vals.extend([float(abs(obs.get('img_age_third',0.))),float(abs(obs.get('img_age_wrist',0.)))])
    gp=obs.get('gripper_position',-1); gw=obs.get('gripper_width_m',None)
    vals.extend([float(gp is not None and gp>=0),float(gw is not None and np.isfinite(gw))]); return np.asarray(vals,np.float32)

def features(obs,labels,target,goalmask,goalvalid,confidence,T,goaltool,terminal,last,sigma=.03,clearance_m=.10):
    E=finite(obs['T_base_ee'],(4,4)); tcp=base_to_w(E[:3,3],T); tp=w_to_pix(tcp[:2]); cur=(labels==target).astype(np.uint8) if target else np.zeros((S,S),np.uint8); other=((labels>0)&(labels!=target)).astype(np.uint8)
    c=center(cur) if cur.sum() else tp; cp=pix_to_w(c); g=center(goalmask) if goalmask.sum() else c; displacement=pix_to_w(g)-cp if goalvalid else np.zeros(2)
    reg,fit,ambiguous=register_masks(cur,goalmask) if goalvalid else (np.zeros(3),0.,True); theta=reg[2]; R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    yy,xx=np.mgrid[:S,:S]; pts=np.stack([xx,yy],-1); relative=(pts-c)*DX
    flow=(relative@R.T-relative+displacement)/.2 if goalvalid else np.zeros((S,S,2)); heat=np.exp(-((xx-tp[0])**2+(yy-tp[1])**2)/8.); sd=sdf(cur); gd=sdf(goalmask)
    maps=[]
    for m in [sd,gd,other.astype(float),heat,flow[:,:,0],flow[:,:,1]]:
        maps.append(cv2.getRectSubPix(m.astype(np.float32),(64,64),tuple(float(v) for v in c)))
    maps.extend([cv2.resize((labels>0).astype(np.float32),(64,64),interpolation=cv2.INTER_AREA),np.full((64,64),float(goalvalid),np.float32)])
    bp,hole=boundary(cur,32); bw=pix_to_w(bp); off=bw-cp; gy,gx=np.gradient(sd); ij=np.clip(np.rint(bp).astype(int),0,S-1)
    normal=-np.stack([gx[ij[:,1],ij[:,0]],gy[ij[:,1],ij[:,0]]],1); normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-6)
    clearance=distance_transform_edt(other==0)[ij[:,1],ij[:,0]]*DX; flowp=off@R.T-off+displacement if goalvalid else np.zeros_like(off); near=tcp[:2]-bw; feasible=(clearance>max(.008,sigma*.25)).astype(np.float32)*(cur.sum()>12)
    nodes=np.concatenate([off/.15,normal,flowp/.2,near/.2,clearance[:,None]/.1,hole[:,None],feasible[:,None],np.full((32,1),confidence),np.full((32,1),np.sqrt(cur.sum())*DX/.1),np.linalg.norm(near,axis=1)[:,None]/.2,np.full((32,1),float(goalvalid)),np.full((32,1),float(ambiguous))],1).astype(np.float32)
    score=(normal*flowp).sum(1)/.1-np.linalg.norm(near,axis=1)/.3-5*(1-feasible); idx=int(np.argmax(score)); n=normal[idx]
    if np.linalg.norm(n)<.5: n=np.array([1.,0.])
    Wbasis=np.array([[n[0],-n[1],0.],[n[1],n[0],0.],[0.,0.,1.]]); basis=(T[:3,:3]@Wbasis).astype(np.float32)
    goaltool=finite(goaltool,(4,4)); e=goaltool[:3,3]-E[:3,3]; rot=rotation_error(E[:3,:3],goaltool[:3,:3]); err=float(np.linalg.norm(displacement))
    if cur.any() and goalvalid:
        gb,_=boundary(goalmask,64); gi=np.clip(np.rint(gb).astype(int),0,S-1)
        cb=cv2.morphologyEx(cur,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)); gg=cv2.morphologyEx(goalmask,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)); dc=distance_transform_edt(cb==0)*DX; dg=distance_transform_edt(gg==0)*DX
        contour=float((dg[ij[:,1],ij[:,0]].mean()+dc[gi[:,1],gi[:,0]].mean())*.5)
    else: contour=1.
    route=goaltool[:3,3].copy(); prox=float(np.linalg.norm(near[idx])) if cur.any() else 1.; exitmode=bool(target and goalvalid and contour<.012)
    if target and not exitmode and cur.any():
        wp=bw[idx]-normal[idx]*.012 if prox>.018 else bw[idx]+flowp[idx]*.25; route=T[:3,:3]@np.array([wp[0],wp[1],tcp[2]])+T[:3,3]
    line=np.clip(np.linspace(tp,w_to_pix(base_to_w(route,T)[:2]),24).astype(int),0,S-1); blocked=bool(np.any(other[line[:,1],line[:,0]]))
    if blocked and tcp[2]<clearance_m: route=E[:3,3]+T[:3,2]*max(.02,clearance_m-tcp[2])
    vec=basis.T@(route-E[:3,3]); base=np.concatenate([vec/max(.06,np.linalg.norm(vec)),np.zeros(3)]).astype(np.float32)
    rv=robot_vector(obs,last); extra=np.concatenate([(basis.T@e)/.3,rot,np.asarray(terminal),tcp[2:3],(tcp[:2]-cp)/.2,displacement/.2,[np.sin(theta),np.cos(theta),confidence,float(goalvalid),fit,float(ambiguous),sigma/.03,prox/.1,float(blocked),float(exitmode)]])
    state=np.concatenate([rv,extra]).astype(np.float32); out=np.zeros(112,np.float32); out[:len(state)]=state
    return {'state':out,'maps':np.clip(np.stack(maps),-2,2).astype(np.float32),'nodes':nodes,'basis':basis,'base':base,'score':score.astype(np.float32),'error':err,'contour_error':contour,'proximity':prox,'blocked':blocked,'ambiguous':ambiguous,'confidence':confidence,'candidate':idx,'tcp_W':tcp,'center_W':cp}
