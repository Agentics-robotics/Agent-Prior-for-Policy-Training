"""Causal, shape-agnostic perception in the calibrated image chart.
No trained perception weights, alphabet, depth, or future observations are used.
"""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

W, H = 1280, 720
GW, GH = 256, 144
NODE_COUNT = 192
NODE_DIM = 19


def finite(x, shape=None):
    try:
        a = np.asarray(x, dtype=np.float64)
        return bool(np.isfinite(a).all() and (shape is None or a.shape == shape))
    except (ValueError, TypeError):
        return False


class Chart:
    def __init__(self, metadata):
        self.cal = metadata['calibration']
        self.maps = {}
        for name in ('third', 'wrist'):
            c = self.cal[name]
            if (c['width'], c['height']) != (W, H):
                raise ValueError('Unsupported camera size; original 1280x720 required')
            k = np.asarray(c['K'], np.float64)
            d = np.asarray(c['dist'], np.float64)
            self.maps[name] = cv2.initUndistortRectifyMap(k, d, None, k, (W, H), cv2.CV_32FC1)
        self.base_cam = np.asarray(self.cal['third']['T_base_cam'], np.float64)
        self.flange_cam = np.asarray(self.cal['wrist']['T_flange_cam'], np.float64)
        self.k = np.asarray(self.cal['third']['K'], np.float64)

    def image(self, rgb, name, mask=False):
        a = np.asarray(rgb)
        if a.shape[:2] != (H, W):
            raise ValueError('Image is not in the original camera coordinates')
        return cv2.remap(a, *self.maps[name], cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR)

    def project(self, xyz):
        p = self.base_cam[:3, :3].T @ (np.asarray(xyz) - self.base_cam[:3, 3])
        if not finite(p) or p[2] <= .02:
            return np.zeros(2, np.float32), 0.
        uv = (self.k @ p)[:2] / p[2]
        valid = float(0 <= uv[0] < W and 0 <= uv[1] < H)
        return np.clip(uv, [-W, -H], [2*W, 2*H]).astype(np.float32), valid


def full_mask(piece):
    m = np.zeros((H, W), np.uint8)
    if piece is not None:
        x, y, w, h = piece['box']
        m[y:y+h, x:x+w] = piece['mask']
    return m


def compact(mask, rgb=None):
    yy, xx = np.nonzero(mask)
    if len(xx) < 1:
        return None
    x, y, r, b = int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)
    m = np.asarray(mask[y:b, x:r] > 0, np.uint8)
    color = np.zeros(3, np.float32) if rgb is None else np.median(rgb[mask > 0], axis=0).astype(np.float32)/255.
    # Rotation-independent shape statistics are soft association cues, not letter IDs.
    hu = cv2.HuMoments(cv2.moments(m)).flatten()[:3]
    desc = -np.sign(hu)*np.log10(np.abs(hu)+1e-12)
    return {'box': (x, y, r-x, b-y), 'mask': m, 'center': np.array([xx.mean(), yy.mean()], np.float32),
            'area': float(len(xx)), 'color': color, 'desc': desc.astype(np.float32)}


def detect(rgb):
    """Class-agnostic warm chroma components with neutral-surround evidence.
    Holes are NOT filled. Closing only bridges 1-2 pixel JPEG noise. Tool/robot
    neutral pixels are not foreground; contact occlusions remain missing.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    warm = ((hsv[:,:,0] >= 4) & (hsv[:,:,0] <= 42) & (hsv[:,:,1] >= 38) &
            (hsv[:,:,2] >= 32) & (hsv[:,:,2] <= 246)).astype(np.uint8)
    warm = cv2.morphologyEx(warm, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(warm, 8)
    proposals = []
    for j in range(1, n):
        x, y, w, h, area = stats[j]
        if not (350 <= area <= 42000 and 15 <= w <= 340 and 15 <= h <= 300):
            continue
        if x <= 1 or y <= 1 or x+w >= W-1 or y+h >= H-1:
            continue
        l, t, r, b = max(0,x-8), max(0,y-8), min(W,x+w+8), min(H,y+h+8)
        local = (labels[t:b,l:r] == j).astype(np.uint8)
        ring = (cv2.dilate(local, np.ones((13,13),np.uint8)) > 0) & (local == 0)
        neutral = (hsv[t:b,l:r,1] < 70) & (hsv[t:b,l:r,2] > 65)
        if ring.sum() == 0 or (neutral & ring).sum()/ring.sum() < .18:
            continue
        p = compact((labels == j).astype(np.uint8), rgb)
        p['confidence'] = float(min(1., .5 + .5*(neutral & ring).sum()/ring.sum()))
        proposals.append(p)
    return proposals


def mask_iou(a, b):
    ax, ay, aw, ah = a['box']; bx, by, bw, bh = b['box']
    x, y, r, d = max(ax,bx), max(ay,by), min(ax+aw,bx+bw), min(ay+ah,by+bh)
    if r <= x or d <= y:
        return 0.
    inter = np.logical_and(a['mask'][y-ay:d-ay,x-ax:r-ax], b['mask'][y-by:d-by,x-bx:r-bx]).sum()
    return float(inter / max(a['area']+b['area']-inter, 1.))


def update_tracks(proposals, state, timestamp):
    """Forward-only Hungarian association; ambiguous matches are not accepted.
    Tracks persist for three seconds without being emitted as observed masks.
    """
    if state is None:
        state = {'tracks': {}, 'next': 0, 'time': float(timestamp)}
    tracks = state['tracks']
    ids = [k for k, v in tracks.items() if timestamp-v['last'] <= 3.]
    cost = np.full((len(ids),len(proposals)), 1000., np.float64)
    for i, tid in enumerate(ids):
        tr = tracks[tid]; p = tr['piece']
        dt = min(max(timestamp-tr['last'],0.), .3)
        pred = p['center'] + tr['velocity']*dt
        for j, q in enumerate(proposals):
            dist = np.linalg.norm(q['center']-pred)
            ratio = q['area']/max(p['area'],1.)
            if dist > 100. or not .22 <= ratio <= 3.5:
                continue
            shape = min(np.linalg.norm(q['desc']-p['desc']), 8.)/8.
            col = min(np.linalg.norm(q['color']-p['color']),1.)
            cost[i,j] = dist/65. + .3*abs(np.log(ratio)) + .35*shape + .3*col - .4*mask_iou(p,q)
    matched = set(); observed = {}
    if cost.size:
        rows, cols = linear_sum_assignment(cost)
        for i,j in zip(rows,cols):
            c = cost[i,j]
            row = np.sort(cost[i]); col = np.sort(cost[:,j])
            margin = min(row[1]-c if len(row)>1 else 99., col[1]-c if len(col)>1 else 99.)
            if c > 1.65 or margin < .12:
                continue
            tid = ids[i]; old = tracks[tid]
            dt = max(timestamp-old['last'], .001)
            v = np.clip((proposals[j]['center']-old['piece']['center'])/dt,-400.,400.)
            q = dict(proposals[j]); q['confidence'] *= float(np.clip(1.-max(c,0.)*.25,.35,1.))
            tracks[tid] = {'piece': q, 'last': float(timestamp), 'velocity': .5*old['velocity']+.5*v}
            observed[tid] = q; matched.add(j)
    for j,p in enumerate(proposals):
        if j in matched:
            continue
        # Do not create a new identity alongside a nearby unresolved live track.
        near = any(np.linalg.norm(p['center']-tracks[k]['piece']['center']) < 75. and
                   timestamp-tracks[k]['last'] < .7 for k in ids if k not in observed)
        if near:
            continue
        tid = state['next']; state['next'] += 1
        tracks[tid] = {'piece': p, 'last': float(timestamp), 'velocity': np.zeros(2,np.float32)}
        observed[tid] = p
    state['tracks'] = {k:v for k,v in tracks.items() if timestamp-v['last'] <= 3.}
    state['time'] = float(timestamp)
    return observed, state


def scalar(obs, key):
    v = obs.get(key)
    return (float(v), 1.) if finite(v) and np.asarray(v).size == 1 else (0., 0.)


def state_vector(obs, chart, previous, rgb):
    vals, masks = [], []
    for name, shape, scale in [('q',(7,),np.pi),('dq',(7,),1.),('tau_ext',(7,),10.),
                                ('T_base_ee',(4,4),1.),('T_base_flange',(4,4),1.)]:
        a = np.asarray(obs.get(name, np.full(shape,np.nan)), np.float64)
        if a.shape != shape:
            a = np.full(shape,np.nan)
        if len(shape) == 2:
            a = a[:3,:]
        ok = np.isfinite(a)
        vals.extend(np.where(ok,a/scale,0.).flatten()); masks.extend(ok.flatten())
    fl = np.asarray(obs.get('T_base_flange',np.full((4,4),np.nan)))
    wc = fl @ chart.flange_cam if finite(fl,(4,4)) else np.full((4,4),np.nan)
    ok = np.isfinite(wc[:3]); vals.extend(np.where(ok,wc[:3],0.).flatten()); masks.extend(ok.flatten())
    for name in ('gripper_position','gripper_width_m'):
        v, m = scalar(obs,name)
        if v < 0: v,m = 0.,0.
        vals.append(v); masks.append(m)
    temporal = []
    for name in ('t','recv_time','img_t_third','img_t_wrist'):
        v, valid = scalar(obs,name)
        pv, pok = scalar(previous or {},name)
        temporal.extend([float(np.clip(v-pv, -1.,1.)) if valid and pok else 0.,valid*pok,
                         float(valid and pok and v == pv)])
    for name in ('img_age_third','img_age_wrist'):
        v,m = scalar(obs,name); temporal.extend([float(np.clip(v,-2.,2.)),m])
    ee = obs.get('T_base_ee')
    uv, valid = chart.project(np.asarray(ee)[:3,3]) if finite(ee,(4,4)) else (np.zeros(2,np.float32),0.)
    support = 0.
    if valid:
        gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray,40,100)
        x,y = np.rint(uv).astype(int); x=min(x,W-1); y=min(y,H-1)
        roi = edges[max(0,y-20):min(H,y+21),max(0,x-20):min(W,x+21)]
        support = float(roi.size > 0 and roi.max() > 0)
    state = np.array(vals+list(np.asarray(masks,dtype=float))+temporal+list(uv/[W,H])+[valid,support],np.float32)
    return state, uv, valid*support


def convert_observation(obs, chart, tracker=None, previous=None):
    """The sole live converter, shared by preparation, inventory, and act."""
    for view in ('third','wrist'):
        a = np.asarray(obs[view+'_rgb'])
        if a.shape != (H,W,3) or a.dtype != np.uint8:
            raise ValueError('Expected original uint8 RGB for '+view)
    if not finite(obs.get('t')):
        raise ValueError('Missing robot time')
    third = chart.image(obs['third_rgb'],'third')
    wrist = chart.image(obs['wrist_rgb'],'wrist')
    observed, tracker = update_tracks(detect(third),tracker,float(obs['t']))
    state, tool, tool_conf = state_vector(obs,chart,previous,third)
    return {'third_half': cv2.resize(third,(640,360),interpolation=cv2.INTER_AREA),
            'third': cv2.resize(third,(GW,GH),interpolation=cv2.INTER_AREA),
            'wrist': cv2.resize(wrist,(224,128),interpolation=cv2.INTER_AREA),
            'observed': observed, 'state': state, 'tool':tool,'tool_conf':tool_conf,
            't':float(obs['t'])}, tracker


def contours(mask, budget, type_id, center, tool, tool_conf, goal_dist, confidence):
    curves, hierarchy = cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    if hierarchy is None: return [],[]
    valid = [i for i,c in enumerate(curves) if len(c)>=8 and cv2.arcLength(c,True)>=12]
    valid = sorted(valid,key=lambda i:cv2.arcLength(curves[i],True),reverse=True)[:10]
    nodes, edges = [],[]
    lengths = [cv2.arcLength(curves[i],True) for i in valid]
    total = max(sum(lengths),1.)
    for i, length in zip(valid,lengths):
        room = budget-len(nodes)
        if room < 4: break
        n = min(room,max(4,int(budget*length/total)))
        raw = curves[i][:,0,:].astype(np.float32)
        ds = np.linalg.norm(np.roll(raw,-1,axis=0)-raw,axis=1)
        cum = np.r_[0.,np.cumsum(ds)]
        ss = np.linspace(0,cum[-1],n,endpoint=False)
        loop = np.vstack([raw,raw[0]])
        p = np.stack([np.interp(ss,cum,loop[:,k]) for k in (0,1)],1)
        tangent = np.roll(p,-1,axis=0)-np.roll(p,1,axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),1e-5)
        normal = np.stack([-tangent[:,1],tangent[:,0]],1)
        curv = (np.roll(tangent,-1,axis=0)-np.roll(tangent,1,axis=0))*normal
        hole = float(hierarchy[0,i,3]>=0)
        offset = len(nodes)
        for j,xy in enumerate(p):
            ix,iy = np.clip(np.rint(xy).astype(int),[0,0],[W-1,H-1])
            types = [float(type_id==k) for k in range(5)]
            f = list(xy/[W,H])+list((xy-center)/[W,H])+list(tangent[j])+list(normal[j])+[
                float(curv[j].sum()),hole]+types+list((xy-tool)/[W,H])+[
                float(goal_dist[iy,ix]/1280.),float(confidence*tool_conf if type_id==3 else confidence)]
            nodes.append(f); edges.append([offset+(j-1)%n,offset+(j+1)%n])
    return nodes,edges


def make_graph(selected, others, goal_mask, tool, tool_conf, workspace=None):
    center = selected['center'] if selected is not None else np.zeros(2)
    inside = cv2.distanceTransform(goal_mask.astype(np.uint8),cv2.DIST_L2,3)
    outside = cv2.distanceTransform((goal_mask==0).astype(np.uint8),cv2.DIST_L2,3)
    gd = outside-inside
    nodes,edges=[],[]
    groups=[(full_mask(selected),64,0,selected['confidence'] if selected else 0.),(goal_mask,64,1,1.)]
    # Neighbors get separate loops; no false adjacency between instances or holes.
    nearest=sorted(others,key=lambda p:np.linalg.norm(p['center']-center))[:6]
    for p in nearest: groups.append((full_mask(p),max(4,48//max(len(nearest),1)),2,p['confidence']))
    if workspace is not None:
        groups.append((workspace.astype(np.uint8),12,4,1.))
    for m,b,k,c in groups:
        nn,ee=contours(m,min(b,NODE_COUNT-1-len(nodes)),k,center,tool,tool_conf,gd,c)
        offset=len(nodes); nodes.extend(nn); edges.extend([[x+offset,y+offset] for x,y in ee])
    # Explicit tool node, separate from any hypothesized silhouette/width.
    if len(nodes)<NODE_COUNT:
        f=[0.]*NODE_DIM; f[0:2]=list(tool/[W,H]); f[2:4]=list((tool-center)/[W,H]); f[13]=1.; f[18]=tool_conf
        i=len(nodes); nodes.append(f); edges.append([i,i])
    x=np.zeros((NODE_COUNT,NODE_DIM),np.float32); e=np.zeros((NODE_COUNT,2),np.int64); m=np.zeros(NODE_COUNT,np.float32)
    n=min(len(nodes),NODE_COUNT)
    x[:n]=nodes[:n]; e[:n]=edges[:n]; m[:n]=1.
    return x,e,m


def goal_pack(mask, rgb):
    m=(mask>0).astype(np.uint8)
    small=cv2.resize(m,(GW,GH),interpolation=cv2.INTER_NEAREST)
    colored=cv2.resize(rgb*m[:,:,None],(GW,GH),interpolation=cv2.INTER_AREA)
    return np.concatenate([colored.transpose(2,0,1),small[None]*255],0).astype(np.uint8)


def selected_features(frame, selected, goal_mask, goal_center, workspace=None):
    current=full_mask(selected)
    marker=cv2.resize(current,(GW,GH),interpolation=cv2.INTER_NEAREST)
    c=selected['center'] if selected is not None else np.zeros(2,np.float32)
    # Crop comes only from current target/tool, never the hindsight goal.
    valid=selected is not None
    if valid:
        x,y,w,h=selected['box']; span=max(w,h,128)*1.65
        if frame['tool_conf']:
            span=max(span,min(420.,2*np.max(np.abs(frame['tool']-c))+64.))
        span=float(np.clip(span,160.,480.))
        l=float(np.clip(c[0]-span/2,0,W-span)); t=float(np.clip(c[1]-span/2,0,H-span))
    else: l,t,span=0.,0.,256.
    # Half-resolution storage retains exact crop-to-chart transform.
    affine=np.array([[2*l/128./2,0,0],[0,1,0]],np.float32)  # replaced below by explicit inverse sampling
    xx,yy=np.meshgrid(np.arange(128,dtype=np.float32),np.arange(128,dtype=np.float32))
    mx=(l+(xx+.5)*span/128.)/2.-.5; my=(t+(yy+.5)*span/128.)/2.-.5
    local=cv2.remap(frame['third_half'],mx,my,cv2.INTER_LINEAR)
    cm=cv2.remap(current,(l+(xx+.5)*span/128.).astype(np.float32),(t+(yy+.5)*span/128.).astype(np.float32),cv2.INTER_NEAREST)
    local=np.concatenate([local.transpose(2,0,1),cm[None]*255],0).astype(np.uint8)
    extra=np.r_[c/[W,H],goal_center/[W,H],(goal_center-c)/[W,H],
                float(valid),selected['confidence'] if valid else 0.,l/W,t/H,span/W,span/H].astype(np.float32)
    graph=make_graph(selected,[p for p in frame['observed'].values() if p is not selected],goal_mask,frame['tool'],frame['tool_conf'],workspace)
    return {'state':np.r_[frame['state'],extra].astype(np.float32),'marker':marker[None]*255,'local':local,
            'nodes':graph[0],'edges':graph[1],'node_mask':graph[2], 'center':c.astype(np.float32),
            'valid':bool(valid and selected['confidence']>=.25)}
