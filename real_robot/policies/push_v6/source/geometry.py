import numpy as np
import cv2
import torch
from scipy.spatial import cKDTree
from real_robot.training_pipeline import public
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from perception import proposals


def make_sam(device):
    net=build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',public.asset_path('sam2.1_hiera_small.pt'),device=str(device))
    for p in net.parameters():
        p.requires_grad_(False)
    net.eval();return net


def rays(uv,cam,T):
    u=cv2.undistortPoints(np.asarray(uv,np.float64).reshape(-1,1,2),np.array(cam['K']),np.array(cam['dist'])).reshape(-1,2)
    d=np.column_stack([u,np.ones(len(u))])@T[:3,:3].T;d/=np.linalg.norm(d,axis=1,keepdims=True)
    return np.repeat(T[None,:3,3],len(u),axis=0),d


def unproject(uv,cam,T,z):
    o,d=rays(uv,cam,T)
    return (o+d*((z-o[:,2])/d[:,2])[:,None])[:,:2]


def project(xyz,cam,T):
    x=(np.asarray(xyz)-T[:3,3])@T[:3,:3]
    u,_=cv2.projectPoints(x.reshape(-1,1,3),np.zeros(3),np.zeros(3),np.array(cam['K']),np.array(cam['dist']))
    return u.reshape(-1,2)


def brown(rgb):
    x=rgb.astype(np.float32)
    return ((x[:,:,0]>x[:,:,1]*1.025)&(x[:,:,1]>x[:,:,2]*1.11)&(x[:,:,0]-x[:,:,2]>12)&(x[:,:,0]>22)&(x[:,:,0]<225)).astype(np.uint8)


def table_roi(shape):
    m=np.zeros(shape[:2],np.uint8);h,w=shape[:2]
    v=np.array([[.212,.058],[.605,.035],[.828,.998],[.028,.998]],np.float32)
    cv2.fillPoly(m,[(v*np.array([w,h])).astype(np.int32)],1);return m


def sample_loop(p,n):
    e=np.roll(p,-1,axis=0)-p;l=np.linalg.norm(e,axis=1);cum=np.r_[0,np.cumsum(l)]
    if cum[-1]<1e-6:
        return np.repeat(p[:1],n,axis=0)
    s=np.linspace(0,cum[-1],n,endpoint=False);j=np.minimum(np.searchsorted(cum,s,side='right')-1,len(p)-1)
    return p[j]+e[j]*((s-cum[j])/np.maximum(l[j],1e-8))[:,None]


def shape_from_mask(mask,cam,T,z,n=96):
    cs,hi=cv2.findContours(mask.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    if hi is None:
        return None
    outer=[i for i,c in enumerate(cs) if hi[0,i,3]<0 and cv2.contourArea(c)>180]
    if not outer:
        return None
    root=max(outer,key=lambda i:cv2.contourArea(cs[i]));ids=[root]+[i for i,c in enumerate(cs) if hi[0,i,3]==root and cv2.contourArea(c)>100]
    loops=[]
    for k in ids:
        p=unproject(cs[k].reshape(-1,2),cam,T,z)
        if len(p)>=3:
            loops.append(p)
    if not loops:
        return None
    ls=np.array([np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1).sum() for p in loops]);sizes=np.maximum(8,(n*ls/ls.sum()).astype(int));sizes[0]+=n-int(sizes.sum())
    if sizes[0]<24:
        return None
    pts=[];normals=[];flags=[]
    for j,(p,size) in enumerate(zip(loops,sizes)):
        p=sample_loop(p,int(size));tan=np.roll(p,-2,axis=0)-np.roll(p,2,axis=0)
        area=np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))
        nn=np.column_stack([tan[:,1],-tan[:,0]])*(1 if area>0 else -1)*(1 if j==0 else -1)
        nn/=np.maximum(np.linalg.norm(nn,axis=1,keepdims=True),1e-8);pts.append(p);normals.append(nn);flags.extend([j]*int(size))
    xy=np.concatenate(pts);nn=np.concatenate(normals);dia=float(np.linalg.norm(np.ptp(xy,axis=0)))
    if dia<.04 or dia>.26:
        return None
    return {'xy':xy,'normal':nn,'loop':np.array(flags),'diameter':dia,'area':float(cv2.contourArea(cs[root])),'pixel_center':np.mean(cs[root].reshape(-1,2),axis=0)}


def rotation(theta):
    return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])


def register(p,q):
    cp=p.mean(0);cq=q.mean(0);best=None;tree=cKDTree(q)
    for angle in [-.6,-.3,0,.3,.6]:
        R=rotation(angle);t=cq-cp@R.T
        for it in range(12):
            x=p@R.T+t;d,j=tree.query(x);keep=d<=np.quantile(d,.85);u=p[keep];v=q[j[keep]];uc=u.mean(0);vc=v.mean(0)
            U,S,V=np.linalg.svd((u-uc).T@(v-vc));rr=V.T@U.T
            if np.linalg.det(rr)<0:
                V[-1]*=-1;rr=V.T@U.T
            R=rr;t=vc-uc@R.T
        x=p@R.T+t;d=tree.query(x)[0];d2=cKDTree(x).query(q)[0]
        err=float(np.sqrt((np.mean(np.sort(d)[:int(.9*len(d))]**2)+np.mean(np.sort(d2)[:int(.9*len(d2))]**2))/2))
        if best is None or err<best[0]:
            best=(err,R,t)
    return best


def match(actor,future):
    best=None
    for j,s in enumerate(future):
        if np.linalg.norm(s['xy'].mean(0)-actor['xy'].mean(0))>.14 or not .65<s['diameter']/actor['diameter']<1.45:
            continue
        err,R,t=register(actor['xy'],s['xy']);score=err+.02*abs(np.log(s['diameter']/actor['diameter']))
        if best is None or score<best[0]:
            best=(score,j,err,R,t)
    return best


def features(xy,normal,loop,goal_xy,obstacles,tool_radius=.003,workspace=None,max_length=.02):
    center=xy.mean(0);delta=(goal_xy-xy).mean(0);theta=float(np.arctan2(delta[1],delta[0])) if np.linalg.norm(delta)>.004 else 0.;F=rotation(theta)
    dia=max(.04,float(np.linalg.norm(np.ptp(xy,axis=0))));p=(xy-center)@F/dia;n=normal@F;g=(goal_xy-xy)@F/dia
    curvature=np.clip(np.linalg.norm(np.roll(normal,2,axis=0)-np.roll(normal,-2,axis=0),axis=1),0,2)
    point=np.column_stack([p,n,g,(loop>0).astype(float),curvature]).astype(np.float32)
    lengths=np.minimum(np.array([.005,.010,.020]),min(max_length,.2*dia));dirs=[];contacts=[];lens=[];ns=[];ids=[]
    for i in range(len(xy)):
        for a in [-.6981317,0,.6981317]:
            d=rotation(a)@(-normal[i])
            for L in lengths:
                contacts.append(xy[i]);dirs.append(d);lens.append(L);ns.append(normal[i]);ids.append(i)
    c=np.array(contacts);d=np.array(dirs);L=np.array(lens);nn=np.array(ns);ids=np.array(ids);cf=(c-center)@F/dia;df=d@F;nf=nn@F
    desired=g[ids];align=np.sum(df*desired,axis=1);torque=cf[:,0]*df[:,1]-cf[:,1]*df[:,0];clear=np.full(len(c),.15);valid=np.ones(len(c),bool);start=c+nn*(tool_radius+.002)
    if len(obstacles)>0:
        tree=cKDTree(obstacles)
        for alpha in [0,.5,1]:
            clear=np.minimum(clear,tree.query(start+d*(L*alpha)[:,None])[0])
        valid &= clear>tool_radius+.003
    dist=np.linalg.norm(start[:,None,:]-xy[None,:,:],axis=2);far=np.linalg.norm(c[:,None,:]-xy[None,:,:],axis=2)>tool_radius*2.5+.003
    opposite=np.min(np.where(far,dist,1),axis=1);valid &= opposite>tool_radius+.002
    if workspace is not None:
        poly=np.array(workspace,np.float32).reshape(-1,1,2)
        for i in range(len(c)):
            if cv2.pointPolygonTest(poly,tuple(start[i]),True)<tool_radius+.003 or cv2.pointPolygonTest(poly,tuple(start[i]+d[i]*L[i]),True)<tool_radius+.003:
                valid[i]=False
    feats=np.column_stack([cf,nf,df,desired,align,torque,L/dia,L*50,np.full(len(c),dia*10),np.full(len(c),tool_radius*100),np.minimum(clear/dia,2),np.minimum(opposite/dia,2),(loop[ids]>0).astype(float),curvature[ids],np.linalg.norm(desired,axis=1),np.sum(nf*desired,axis=1)]).astype(np.float32)
    return point,feats,valid,np.column_stack([c,d,L]).astype(np.float32),ids
