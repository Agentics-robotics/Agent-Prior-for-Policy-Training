"""Image-chart goals and padded variable-topology contour graphs."""
import cv2
import numpy as np
from perception import W,H,center,packed,unpacked,template_rgb,registration

NODES=256
NODE_DIM=20

def render_goal(template, spatial_goal, workspace=None):
    if spatial_goal.get('frame')!='third_undistorted_original_pixels':
        raise ValueError('calibration_required: only the undistorted original image chart is supported')
    m=unpacked(template); rgb=template_rgb(template); c=center(m)
    uv=np.asarray(spatial_goal['center_uv'],np.float64)
    angle=float(spatial_goal['theta_deg_clockwise'])
    if uv.shape!=(2,) or not np.isfinite(uv).all() or not np.isfinite(angle):
        raise ValueError('nonfinite image placement')
    if not spatial_goal.get('accept_similarity_approximation',False):
        raise ValueError('calibration_required: similarity approximation must be acknowledged')
    # No fabricated plane from T_base_board. Large changes need a different commissioned renderer.
    if np.linalg.norm(uv/2-c)>260:
        raise ValueError('calibration_required: translation exceeds similarity-renderer support (520 original pixels)')
    mat=cv2.getRotationMatrix2D(tuple(c.astype(float)),-angle,1.)
    mat[:,2]+=uv/2-c
    goal=cv2.warpAffine(m,mat,(W,H),flags=cv2.INTER_NEAREST)
    grgb=cv2.warpAffine(rgb,mat,(W,H),flags=cv2.INTER_LINEAR)*goal[:,:,None]
    if goal.sum()<.97*m.sum(): raise ValueError('goal is clipped or cannot be realized in chart')
    if workspace is not None:
        ws=np.asarray(workspace,np.uint8)
        if ws.shape!=(H,W) or np.any((goal>0)&(ws==0)): raise ValueError('goal outside supplied image workspace')
    return goal,grgb,{'renderer':'similarity_approximation','angle_deg_clockwise':angle,'metric_valid':False}

def goal_image(mask,rgb):
    return cv2.resize(np.concatenate([rgb*mask[:,:,None],mask[:,:,None]*255],axis=2),(160,90))

def sdf(mask):
    return (cv2.distanceTransform(1-mask,cv2.DIST_L2,3)-cv2.distanceTransform(mask,cv2.DIST_L2,3))/W

def sample_ring(contour,count):
    pts=contour[:,0,:].astype(np.float32)
    if len(pts)<3: return None
    close=np.concatenate([pts,pts[:1]])
    distance=np.linalg.norm(np.diff(close,axis=0),axis=1)
    cumulative=np.concatenate([[0],np.cumsum(distance)])
    if cumulative[-1]<5: return None
    at=np.linspace(0,cumulative[-1],count,endpoint=False)
    return np.stack([np.interp(at,cumulative,close[:,0]),np.interp(at,cumulative,close[:,1])],axis=1).astype(np.float32)

def rings(mask,budget):
    cs,hh=cv2.findContours(mask.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    if hh is None: return []
    chosen=[i for i,c in enumerate(cs) if cv2.arcLength(c,True)>=12]
    chosen=sorted(chosen,key=lambda i:cv2.arcLength(cs[i],True),reverse=True)[:max(1,budget//8)]
    if not chosen: return []
    lengths=np.array([cv2.arcLength(cs[i],True) for i in chosen]); left=budget
    out=[]
    for j,i in enumerate(chosen):
        count=left if j==len(chosen)-1 else int(np.clip(round(budget*lengths[j]/lengths.sum()),8,left-8*(len(chosen)-j-1)))
        left-=count
        pp=sample_ring(cs[i],count)
        if pp is not None: out.append((pp, bool(hh[0,i,3]>=0)))
    return out

def graph(selected,goal,other_masks,tool,tool_valid,confidence,workspace=None):
    nodes=np.zeros((NODES,NODE_DIM),np.float32); edges=np.zeros((NODES,2),np.int64); valid=np.zeros(NODES,np.float32)
    gdist=sdf(goal); c=center(selected); pos=0
    sets=[(selected,64,0,confidence),(goal,64,1,1.)]
    # Retain all ordinary scene proposals up to a bounded obstacle-node budget.
    others=sorted(other_masks,key=lambda x:float(x.sum()),reverse=True)[:8]
    for om in others: sets.append((om,max(8,96//max(1,len(others))),2,1.))
    if workspace is None:
        workspace=np.ones((H,W),np.uint8); workspace[[0,-1]]=0;workspace[:,[0,-1]]=0
    sets.append((workspace,16,3,1.))
    for mask,budget,role,conf in sets:
        for xy,hole in rings(mask,budget):
            count=len(xy)
            if pos+count>=NODES: break
            tangent=np.roll(xy,-1,axis=0)-np.roll(xy,1,axis=0)
            tangent/=np.maximum(1e-4,np.linalg.norm(tangent,axis=1,keepdims=True))
            normal=np.stack([-tangent[:,1],tangent[:,0]],1)
            # Orient normals toward background using the observed binary mask, including holes.
            near=np.rint(xy+normal*2).astype(int); near[:,0]=np.clip(near[:,0],0,W-1);near[:,1]=np.clip(near[:,1],0,H-1)
            normal[mask[near[:,1],near[:,0]]>0]*=-1
            curvature=np.sum((np.roll(tangent,-1,axis=0)-np.roll(tangent,1,axis=0))*normal,axis=1,keepdims=True)/2
            identity=np.zeros((count,5),np.float32);identity[:,role]=1
            px=np.rint(xy).astype(int);px[:,0]=np.clip(px[:,0],0,W-1);px[:,1]=np.clip(px[:,1],0,H-1)
            feat=np.concatenate([xy/[W,H],(xy-c)/[W,H],normal,tangent,curvature,identity,
                 np.full((count,1),float(hole)),np.full((count,1),conf),
                 (xy-tool)/[W,H],gdist[px[:,1],px[:,0],None],np.full((count,1),float(tool_valid))],axis=1)
            nodes[pos:pos+count]=feat
            ids=np.arange(pos,pos+count);edges[ids,0]=np.roll(ids,1);edges[ids,1]=np.roll(ids,-1);valid[ids]=1;pos+=count
    if pos<NODES:
        nodes[pos,:2]=tool/[W,H];nodes[pos,2:4]=(tool-c)/[W,H];nodes[pos,13]=1
        nodes[pos,15]=float(tool_valid);nodes[pos,19]=float(tool_valid);edges[pos]=pos;valid[pos]=float(tool_valid)
    return nodes,edges,valid

def discrepancy(current,goal):
    cc,gc=center(current),center(goal)
    intersection=np.logical_and(current,goal).sum();union=np.logical_or(current,goal).sum()
    return {'center_error_px':float(np.linalg.norm(cc-gc)*2),
            'foreground_iou':float(intersection/max(1,union))}
