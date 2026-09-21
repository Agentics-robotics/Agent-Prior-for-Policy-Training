"""Causal, class-free, material-limited foreground and camera conversion.
No learned/third-party perception assets and no alphabet templates are used.
All internal raster coordinates are half of the original undistorted chart.
"""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

W, H = 640, 360
_MAPS = {}

def calibration(metadata):
    c = metadata['calibration']
    for view in ('third', 'wrist'):
        if np.asarray(c[view]['K']).shape != (3, 3):
            raise ValueError('invalid camera calibration')
    return c

def remap(rgb, metadata, view, size):
    c = calibration(metadata)[view]
    if rgb is None or np.asarray(rgb).shape != (720, 1280, 3) or np.asarray(rgb).dtype != np.uint8:
        raise ValueError('original paired 1280x720 uint8 RGB required')
    key = (view, tuple(np.asarray(c['K']).ravel()), tuple(c['dist']), tuple(size))
    if key not in _MAPS:
        k = np.asarray(c['K'], np.float64)
        nk = k.copy(); nk[0] *= size[0] / 1280.; nk[1] *= size[1] / 720.
        _MAPS[key] = cv2.initUndistortRectifyMap(k, np.asarray(c['dist']), None, nk, tuple(size), cv2.CV_32FC1)
    return cv2.remap(rgb, *_MAPS[key], cv2.INTER_LINEAR)

def box_mask(box, metadata):
    raw = np.zeros((720,1280,3), np.uint8)
    l,t,r,b = box; raw[t:b,l:r] = 255
    return remap(raw, metadata, 'third', (W,H))[:,:,0] > 127

def center(mask):
    y,x = np.nonzero(mask)
    return np.array([x.mean(), y.mean()], np.float32) if len(x) else np.array([0.,0.],np.float32)

def proposals(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    r,g,b = [rgb[:,:,i].astype(np.float32) for i in range(3)]
    # Broad matte cardboard chromaticity. This is not generic semantic segmentation.
    fg = ((hsv[:,:,0] >= 7) & (hsv[:,:,0] <= 39) & (hsv[:,:,1] >= 38)
          & (hsv[:,:,2] >= 38) & (hsv[:,:,2] <= 242) & (r > b*1.12) & (g > b*1.035))
    white = ((hsv[:,:,1] < 72) & (hsv[:,:,2] > 95)).astype(np.uint8)
    cs,_ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    support = np.zeros((H,W),np.uint8)
    if cs:
        cv2.fillConvexPoly(support, cv2.convexHull(max(cs,key=cv2.contourArea)),1)
    fg = (fg & (support > 0)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    n,lab,stats,_ = cv2.connectedComponentsWithStats(fg,8)
    out=[]
    for k in range(1,n):
        x,y,w,h,area = stats[k]
        if area < 180 or area > 14000 or min(w,h)<9 or x<=1 or y<=1 or x+w>=W-1 or y+h>=H-1:
            continue
        m=(lab==k).astype(np.uint8)
        # Fill tiny sticker/chromaticity gaps only; retain substantial holes.
        inv=1-m[y:y+h,x:x+w]
        nn,ll,ss,_=cv2.connectedComponentsWithStats(inv,8)
        for j in range(1,nn):
            xx,yy,ww,hh,aa=ss[j]
            if aa<=18 and xx>0 and yy>0 and xx+ww<w and yy+hh<h:
                m[y:y+h,x:x+w][ll==j]=1
        out.append({'mask':m,'center':center(m),'area':float(m.sum()),
                    'color':rgb[m>0].mean(0)/255., 'bbox':[int(x),int(y),int(x+w),int(y+h)]})
    return sorted(out,key=lambda p:(float(p['center'][0]),float(p['center'][1])))

def track(proposed, memory=None):
    """Forward-only multi-instance association. Lost tracks are never emitted as observed masks."""
    mem = {'tracks':{},'next_id':0} if memory is None else memory
    tracks=mem['tracks']; keys=list(tracks)
    cost=np.full((len(keys),len(proposed)),1e6,np.float64)
    for a,k in enumerate(keys):
        old=tracks[k]
        for b,p in enumerate(proposed):
            ratio=p['area']/old['area']
            d=np.linalg.norm(p['center']-old['center']-old['velocity'])
            color=np.linalg.norm(p['color']-old['color'])
            if .32<ratio<2.15 and d<min(95,25+old['missed']*3):
                cost[a,b]=d+18*abs(np.log(ratio))+30*color
    assigned={}; used=set()
    if cost.size:
        rr,cc=linear_sum_assignment(cost)
        for a,b in zip(rr,cc):
            if cost[a,b]>=1e5: continue
            k=keys[a]; old=tracks[k]; p=proposed[b]
            alternatives=sorted(cost[:,b].tolist())
            row=sorted(cost[a].tolist())
            gap=min(alternatives[1]-alternatives[0] if len(alternatives)>1 else 99,
                    row[1]-row[0] if len(row)>1 else 99)
            # Ambiguous assignments do not become valid selected-object evidence.
            confidence=float(np.clip(gap/12.,0,1)*min(1.,p['area']/old['reference_area']))
            velocity=.6*old['velocity']+.4*(p['center']-old['center'])/max(1,old['missed']+1)
            tracks[k]={**p,'velocity':np.clip(velocity,-12,12),'missed':0,
                       'reference_area':old['reference_area'],'confidence':confidence}
            assigned[k]={**p,'confidence':confidence}; used.add(b)
    for k in keys:
        if k not in assigned:
            tracks[k]['missed']+=1
            tracks[k]['velocity']*=.8
            if tracks[k]['missed']>90: del tracks[k]
    for b,p in enumerate(proposed):
        if b in used: continue
        k=mem['next_id']; mem['next_id']+=1
        tracks[k]={**p,'velocity':np.zeros(2),'missed':0,'reference_area':p['area'],'confidence':1.}
        assigned[k]={**p,'confidence':1.}
    return assigned,mem

def packed(mask):
    y,x=np.nonzero(mask)
    if not len(x): raise ValueError('empty foreground')
    l,t,r,b=int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)
    return {'bbox_half':[l,t,r,b],'mask':mask[t:b,l:r].astype(np.uint8).tolist()}

def unpacked(p):
    l,t,r,b=[int(v) for v in p['bbox_half']]
    if not (0<=l<r<=W and 0<=t<b<=H): raise ValueError('invalid template bounds')
    a=np.asarray(p['mask'],np.uint8)
    if a.shape!=(b-t,r-l) or not np.isin(a,[0,1]).all(): raise ValueError('invalid foreground')
    m=np.zeros((H,W),np.uint8); m[t:b,l:r]=a
    if m.sum()<180: raise ValueError('insufficient foreground')
    return m

def template(p,rgb,scene_version,instance_id,chart_id,sample_index):
    enc=packed(p['mask']); l,t,r,b=enc['bbox_half']
    return {**enc,'rgb':(rgb[t:b,l:r]*p['mask'][t:b,l:r,None]).tolist(),
            'scene_version':scene_version,'instance_id':instance_id,'chart_id':chart_id,
            'sample_index':int(sample_index),'center_uv':(2*p['center']).tolist(),
            'confidence':float(p['confidence'])}

def template_rgb(t):
    m=unpacked(t); l,top,r,b=t['bbox_half']; out=np.zeros((H,W,3),np.uint8)
    a=np.asarray(t['rgb'],np.uint8)
    if a.shape!=(b-top,r-l,3): raise ValueError('invalid template appearance')
    out[top:b,l:r]=a*m[top:b,l:r,None]
    return out

def registration(a,b):
    """Similarity foreground registration with explicit symmetry ties, not semantic upright yaw."""
    ca,cb=center(a),center(b)
    scale=float(np.sqrt(max(1,b.sum())/max(1,a.sum())))
    # Canonical 96px rasters keep bounded registration cost.
    def norm(m,c):
        y,x=np.nonzero(m); span=max(float(x.max()-x.min()+1),float(y.max()-y.min()+1))
        s=70./max(span,1.)
        mat=np.array([[s,0,48-s*c[0]],[0,s,48-s*c[1]]],np.float32)
        return cv2.warpAffine(m,mat,(96,96),flags=cv2.INTER_NEAREST)
    aa,bb=norm(a,ca),norm(b,cb)
    scores=[]
    for angle in range(0,360,10):
        rot=cv2.warpAffine(aa,cv2.getRotationMatrix2D((48,48),-angle,1),(96,96),flags=cv2.INTER_NEAREST)
        scores.append(float(np.logical_and(rot,bb).sum()/max(1,np.logical_or(rot,bb).sum())))
    best=max(scores)
    angles=[i*10 for i,v in enumerate(scores) if v>=best-.035]
    return {'iou':best,'angles_deg_clockwise':angles,'scale':scale,
            'valid':bool(best>=.32 and .5<=scale<=2.)}

def current(observation, previous=None):
    meta=observation['metadata']; c=calibration(meta)
    rgb=remap(observation['third_rgb'],meta,'third',(W,H))
    wrist=remap(observation['wrist_rgb'],meta,'wrist',(256,144))
    ps=proposals(rgb)
    assoc,tm=track(ps,None if previous is None else previous['tracker'])
    return {'rgb':rgb,'wrist':wrist,'proposals':ps,'assigned':assoc,'tracker':tm}

def project_tool(obs):
    c=calibration(obs['metadata'])
    t=np.asarray(obs['T_base_ee'],np.float64).reshape(4,4)
    cam=np.linalg.inv(np.asarray(c['third']['T_base_cam'])) @ t[:,3]
    uv=np.asarray(c['third']['K'])@cam[:3]
    if not np.isfinite(uv).all() or cam[2]<=0: return np.zeros(2,np.float32),False
    uv=uv[:2]/uv[2]/2.
    return uv.astype(np.float32),bool(0<=uv[0]<W and 0<=uv[1]<H)

def crop_detail(rgb,mask,tool):
    c=center(mask); d=np.linalg.norm(tool-c)
    # Bounded target/tool neighborhood; global branch retains more distant tool.
    cc=(c+tool)/2 if d<160 else c
    size=int(np.clip(max(100.,d+90.),100,220)) if d<160 else 140
    x=int(np.clip(cc[0]-size/2,0,W-size)); y=int(np.clip(cc[1]-size/2,0,H-size))
    image=np.concatenate([rgb,mask[:,:,None]*255],axis=2)
    return cv2.resize(image[y:y+size,x:x+size],(160,160)),np.array([2*x,2*y,2*size/160,2*size/160],np.float32)

def state(obs,p,previous_obs,tool,tool_valid):
    def val(name,n):
        a=np.asarray(obs.get(name,np.full(n,np.nan)),np.float64).reshape(-1)
        if len(a)!=n: raise ValueError('bad state shape: '+name)
        return a
    values=[]
    for name in ('q','dq','tau_ext'): values.extend(val(name,7))
    for name in ('T_base_ee','T_base_flange'): values.extend(val(name,16).reshape(4,4)[:3].ravel())
    values.extend([obs.get('gripper_position',np.nan),obs.get('gripper_width_m',np.nan)])
    if values[-2] is None or values[-2]<0: values[-2]=np.nan
    dt=0. if previous_obs is None else float(obs['t'])-float(previous_obs['t'])
    ages=[obs.get('img_age_third',np.nan),obs.get('img_age_wrist',np.nan)]
    repeats=[float(previous_obs is not None and obs.get('img_t_'+v)!=None and obs.get('img_t_'+v)==previous_obs.get('img_t_'+v)) for v in ('third','wrist')]
    values.extend([dt]+ages+repeats)
    flange=val('T_base_flange',16).reshape(4,4)
    wrist_pose=flange @ np.asarray(calibration(obs['metadata'])['wrist']['T_flange_cam'])
    values.extend(wrist_pose[:3].ravel())
    values.extend([tool[0]/W,tool[1]/H,float(tool_valid),p['center'][0]/W,p['center'][1]/H,p['area']/(W*H),p['confidence']])
    a=np.asarray(values,np.float32); valid=np.isfinite(a)
    a=np.where(valid,a,0.)
    if a.shape!=(71,): raise ValueError('state converter dimension')
    return np.concatenate([a,valid.astype(np.float32)])

def features(obs,frame,p,previous_obs):
    tool,tv=project_tool(obs)
    detail,transform=crop_detail(frame['rgb'],p['mask'],tool)
    global_image=cv2.resize(np.concatenate([frame['rgb'],p['mask'][:,:,None]*255],2),(320,180))
    return {'global':global_image,'wrist':frame['wrist'],'detail':detail,'crop_transform':transform,
            'state':state(obs,p,previous_obs,tool,tv),'tool':tool,'tool_valid':tv}
