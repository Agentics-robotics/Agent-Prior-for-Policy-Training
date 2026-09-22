import numpy as np
import cv2
from geometry import features


def sample_loops(loops,n=64):
    good=[]
    for loop in loops:
        p=np.asarray(loop['xy'],float)
        if p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or not np.isfinite(p).all():
            raise ValueError('Invalid contour')
        length=np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1).sum()
        if length>.002:
            good.append((p,bool(loop.get('hole',False)),length))
    if not good or len(good)>12:
        raise ValueError('Unsupported topology')
    lengths=np.array([x[2] for x in good]);counts=np.maximum(4,lengths/lengths.sum()*n).astype(int)
    while counts.sum()>n:
        counts[np.argmax(counts)]-=1
    while counts.sum()<n:
        counts[np.argmax(lengths/counts)]+=1
    out=[]
    for (p,h,l),cnt in zip(good,counts):
        d=np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1);cum=np.r_[0,np.cumsum(d)]
        ix=np.minimum(np.searchsorted(cum,np.arange(cnt)*cum[-1]/cnt,side='right')-1,len(p)-1)
        u=(np.arange(cnt)*cum[-1]/cnt-cum[ix])/np.maximum(d[ix],1e-8)
        q=p[ix]+u[:,None]*(np.roll(p,-1,axis=0)[ix]-p[ix])
        v=np.roll(q,-1,axis=0)-np.roll(q,1,axis=0)
        area=np.sum(q[:,0]*np.roll(q[:,1],-1)-q[:,1]*np.roll(q[:,0],-1))
        normal=np.c_[v[:,1],-v[:,0]]*(1 if area>=0 else -1)*(-1 if h else 1)
        normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-8)
        out.extend(np.c_[q,normal,np.full(cnt,float(h))])
    return np.asarray(out)


def sdf(loops,p):
    outside=[];inside=[]
    for l in loops:
        value=cv2.pointPolygonTest(np.asarray(l['xy'],np.float32).reshape(-1,1,2),(float(p[0]),float(p[1])),True)
        (inside if l.get('hole',False) else outside).append(value)
    if not outside:
        raise ValueError('Missing exterior')
    occupied=any(v>=0 for v in outside) and not any(v>=0 for v in inside)
    return min(abs(x) for x in outside+inside)*(-1 if occupied else 1)


def initialize_contact(actor,instances,workspace,R,t,radius,uncertainty,max_length):
    shape=sample_loops(actor['contours']);_,_,geom=features(shape,R,t,radius,uncertainty)
    goal=shape[:,:2]@R.T+t-shape[:,:2]
    protected=[o for o in instances if o['instance_id']!=actor['instance_id']]
    ws=np.asarray(workspace,np.float32).reshape(-1,1,2)
    best=None
    for k,(p,d,length) in enumerate(zip(geom['points'],geom['directions'],geom['lengths'])):
        if length>max_length:
            continue
        normal=shape[k//15,2:4];g=goal[k//15]
        if g@d<=.001:
            continue
        center=p+normal*(radius+uncertainty+.003)
        if sdf(actor['contours'],center)<radius+uncertainty:
            continue
        safe=True
        for f in np.linspace(0,1,7):
            q=p+normal*radius+d*length*f
            if cv2.pointPolygonTest(ws,(float(q[0]),float(q[1])),True)<radius+uncertainty or any(sdf(o['contours'],q)<radius+uncertainty for o in protected):
                safe=False;break
        if not safe:
            continue
        # Kinematic contact-access prior only. Not a trained forward dynamics model.
        score=float(g@d/(np.linalg.norm(g)+1e-8)-.1*shape[k//15,4]-.3*abs(length-min(.01,np.linalg.norm(g))))
        if best is None or score>best[0]:
            best=(score,{'contact_point_P':p.tolist(),'outward_normal_P':normal.tolist(),'direction_P':d.tolist(),'stroke_length_m':float(length),'boundary_is_hole':bool(shape[k//15,4])})
    return None if best is None else best[1]


def executor_objective(decision,contract):
    if decision is None:
        return {'status':'no_objective'}
    c=contract
    common=['commissioned','scene_revision','calibration_id','T_base_P','guard_profile_id','tool_geometry_id','max_speed_m_s']
    if any(k not in c for k in common) or not c['commissioned']:
        return {'status':'blocked_uncommissioned'}
    if c['scene_revision']!=decision['scene_revision'] or c['calibration_id']!=decision['calibration_id']:
        return {'status':'stale_scene'}
    B=np.asarray(c['T_base_P'],float)
    if B.shape!=(4,4) or not np.isfinite(B).all() or np.linalg.norm(B[:3,:3].T@B[:3,:3]-np.eye(3))>.001:
        return {'status':'invalid_calibration'}
    # Continuation is an EE displacement: it needs no inferred physical tip label.
    if decision['mode']=='continue':
        start=np.asarray(decision['T_base_ee'],float);end=start.copy()
        end[:3,3]+=B[:3,:3]@np.r_[decision['delta_P_m'],0.]
        path={'start_T_base_ee':start.tolist(),'end_T_base_ee':end.tolist(),'mode':'guarded_short_stroke','preserve_start_orientation':True}
    else:
        if any(k not in c for k in ['R_P_tool','support_point_tool_m','contact_height_P_m']):
            return {'status':'blocked_uncommissioned_contact_support'}
        R=np.asarray(c['R_P_tool'],float);support=np.asarray(c['support_point_tool_m'],float)
        if R.shape!=(3,3) or support.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(support).all():
            return {'status':'invalid_calibration'}
        T=np.eye(4);T[:3,:3]=R;T[:3,3]=np.r_[decision['contact_point_P'],c['contact_height_P_m']]-R@support
        standoff=T.copy();standoff[:3,3]+=np.r_[decision['outward_normal_P'],0.]*.025
        path={'contact_T_base_tool':(B@T).tolist(),'standoff_T_base_tool':(B@standoff).tolist(),'mode':'prepare_then_reobserve','require_fresh_arrival_scene':True}
    return dict(path,status='requires_planner_validation',instance_id=decision['instance_id'],scene_revision=decision['scene_revision'],calibration_id=decision['calibration_id'],allowed_contact_selected_instance_only=True,require_full_tool_robot_envelope_check=True,require_contact_height_and_table_clearance_check=True,guard_profile_id=c['guard_profile_id'],tool_geometry_id=c['tool_geometry_id'],max_speed_m_s=min(.015,max(0.,float(c['max_speed_m_s']))),max_duration_s=1.,completion_semantics='stroke or travel completed, not object/task success')
