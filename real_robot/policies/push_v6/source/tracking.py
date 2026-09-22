import numpy as np
from scipy.spatial import cKDTree
from geometry import rotation, register


def material_area(s):
    total=0.
    for j in np.unique(s['loop']):
        p=s['xy'][s['loop']==j]
        a=abs(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1)))/2
        total+=a if j==0 else -a
    return total


def restore_causal_holes(shapes,references):
    result=[]
    for raw in shapes:
        if np.max(raw['loop'])>0:
            result.append(raw);continue
        area=material_area(raw);best=None
        for ref in references:
            if np.max(ref['loop'])<1:
                continue
            ra=material_area(ref)
            if not .72<area/max(ra,1.e-7)<1.18:
                continue
            for angle in [0,np.pi/2,np.pi,3*np.pi/2]:
                Q=rotation(angle);p=ref['xy']@Q.T
                error,R,t=register(p,raw['xy'])
                RR=R@Q;fitted=ref['xy']@RR.T+t
                dr=cKDTree(fitted).query(raw['xy'])[0];df=cKDTree(raw['xy']).query(fitted)[0]
                # A small slit caused by tool occlusion may be restored only if
                # most of the same complete causal shape is currently visible.
                if error<.0055 and np.mean(dr<.008)>.85 and np.mean(df<.008)>.85:
                    if best is None or error<best[0]:
                        best=(error,ref,RR,t,fitted)
        if best is None:
            result.append(raw)
        else:
            error,ref,R,t,fitted=best;s=dict(raw)
            s['xy']=fitted;s['normal']=ref['normal']@R.T;s['loop']=ref['loop'].copy();s['diameter']=float(np.linalg.norm(np.ptp(fitted,axis=0)));s['metric_area']=material_area(s);s['causal_hole_restoration_error_m']=float(error)
            result.append(s)
    return result
