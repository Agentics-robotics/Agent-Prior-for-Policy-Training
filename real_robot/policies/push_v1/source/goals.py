"""Training-only frozen anchors and executable, class-free goal conversion."""
import cv2
import numpy as np
from perception import W,H,detect,compact,full_mask,mask_iou,finite

# Verbatim ORIGINAL distorted-pixel anchor boxes in public.segments() order.
# These constants are used only to derive training GOALS and selected track labels.
# They are never an online feature, inventory, action selector, or lookup alphabet.
ANCHOR_BOXES = [
 [500,490,695,640],[275,310,465,432],[505,365,680,490],[405,135,545,245],
 [250,245,425,385],[250,462,448,605],[528,150,645,260],[490,263,642,370],
 [500,135,635,217],[490,190,635,284],[315,238,477,343],[290,333,470,452],
 [598,492,805,651],[408,312,571,430],[590,148,713,252],[592,363,762,490],
 [580,258,740,370],[587,123,716,206],[580,186,726,278],[201,387,415,529],
 [368,232,529,333],[352,370,529,493]]


def canonical(piece, size=96):
    m=piece['mask']; h,w=m.shape
    scale=(size-12)/max(h,w)
    a=np.array([[scale,0,(size-scale*w)/2],[0,scale,(size-scale*h)/2]],np.float32)
    return cv2.warpAffine(m,a,(size,size),flags=cv2.INTER_NEAREST)


def registration(reference, target):
    """Training-label or diagnostic registration, never a current-pose oracle.
    Enumerate shape-overlap rotation hypotheses, preserving symmetry ties.
    Similarity residual is an approximation for the oblique camera.
    """
    a=canonical(reference); b=canonical(target)
    scores=[]
    for angle in range(-180,180,5):
        rot=cv2.getRotationMatrix2D((47.5,47.5),-angle,1.)
        aa=cv2.warpAffine(a,rot,(96,96),flags=cv2.INTER_NEAREST)
        inter=np.logical_and(aa,b).sum(); union=np.logical_or(aa,b).sum()
        scores.append(float(inter/max(union,1)))
    best=max(scores)
    angles=[-180+5*i for i,s in enumerate(scores) if s>=best-.025]
    # Deduplicate contiguous near-ties into circular clusters.
    modes=[]
    for angle in sorted(angles,key=lambda x:scores[(x+180)//5],reverse=True):
        if all(abs((angle-m+180)%360-180)>15 for m in modes): modes.append(angle)
    return {'iou':best,'angles_deg':modes,'ambiguous':len(modes)>1,
            'area_ratio':float(target['area']/max(reference['area'],1.))}


def anchor_goal(raw, chart, box):
    l,t,r,b=box
    candidates=[]
    for p in detect(raw):
        m=full_mask(p); inside=int(m[t:b,l:r].sum())
        fraction=inside/max(p['area'],1.)
        if fraction>=.45 and inside>=250:
            candidates.append((inside,p,fraction))
    if not candidates: raise ValueError('anchor_foreground_absent')
    candidates.sort(key=lambda x:x[0],reverse=True)
    if len(candidates)>1 and candidates[1][0]>.65*candidates[0][0]:
        raise ValueError('anchor_instance_ambiguous')
    _,p,fraction=candidates[0]
    # Extract ONLY foreground inside the frozen box; no completing hidden tool pixels.
    mask=full_mask(p); clipped=np.zeros_like(mask); clipped[t:b,l:r]=mask[t:b,l:r]
    mask=chart.image(clipped,'third',mask=True)
    rgb=chart.image(raw,'third')
    goal=compact(mask,rgb)
    if goal is None or goal['area']<250: raise ValueError('anchor_mask_invalid')
    goal['confidence']=p['confidence']*fraction
    return mask,rgb,goal


def polygon_mask(points):
    p=np.asarray(points,np.float64)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or not finite(p):
        raise ValueError('workspace_polygon_invalid')
    if (p<0).any() or (p>np.array([W-1,H-1])).any():
        raise ValueError('workspace_outside_chart')
    m=np.zeros((H,W),np.uint8)
    cv2.fillPoly(m,[np.rint(p).astype(np.int32)],1)
    if m.sum()<1000: raise ValueError('workspace_polygon_degenerate')
    return m


def render_goal(template, spatial_goal, calibration_version, workspace):
    if template.get('calibration_version')!=calibration_version:
        raise ValueError('template_calibration_mismatch')
    mask=np.asarray(template['mask'],np.uint8)
    rgb=np.asarray(template['rgb'],np.uint8)
    if mask.shape!=(H,W) or rgb.shape!=(H,W,3): raise ValueError('template_coordinates_invalid')
    piece=compact(mask,rgb)
    if piece is None or piece['area']<350: raise ValueError('template_foreground_invalid')
    center=np.asarray(spatial_goal['center_uv'],np.float64)
    angle=float(spatial_goal['angle_deg'])
    if not finite(center,(2,)) or not np.isfinite(angle): raise ValueError('goal_nonfinite')
    if (center<0).any() or (center>np.array([W-1,H-1])).any(): raise ValueError('goal_outside_chart')
    # Positive angle is clockwise on the image (u right, v down).
    plane=spatial_goal.get('plane_chart')
    if plane is not None:
        if plane.get('verified') is not True or not plane.get('commissioning_id'):
            raise ValueError('plane_calibration_unverified')
        hh=np.asarray(plane['H_image_to_plane'],np.float64)
        if not finite(hh,(3,3)) or abs(np.linalg.det(hh))<1e-10:
            raise ValueError('plane_homography_invalid')
        def pt(x):
            q=hh@np.r_[x,1.]
            if abs(q[2])<1e-8: raise ValueError('plane_horizon')
            return q[:2]/q[2]
        old,new=pt(piece['center']),pt(center)
        th=angle*np.pi/180.; rot=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
        t=np.eye(3); t[:2,:2]=rot; t[:2,2]=new-rot@old
        transform=np.linalg.inv(hh)@t@hh
        approximate=False
    else:
        if np.linalg.norm(center-piece['center'])>160.:
            raise ValueError('calibration_required_for_large_perspective_change')
        if spatial_goal.get('accept_similarity_approximation') is not True:
            raise ValueError('image_similarity_approximation_not_accepted')
        transform=np.eye(3)
        a=cv2.getRotationMatrix2D(tuple(piece['center'].astype(float)),-angle,1.)
        a[:,2]+=center-piece['center']; transform[:2]=a; approximate=True
    gm=cv2.warpPerspective(mask,transform,(W,H),flags=cv2.INTER_NEAREST)
    gr=cv2.warpPerspective(rgb*mask[:,:,None],transform,(W,H),flags=cv2.INTER_LINEAR)
    if gm.sum()<.7*mask.sum() or gm.sum()>1.45*mask.sum():
        raise ValueError('goal_scale_or_clipping_inconsistent')
    if np.logical_and(gm,workspace==0).sum()>0: raise ValueError('goal_outside_allowed_workspace')
    supplied=spatial_goal.get('foreground')
    if supplied is not None:
        supplied=np.asarray(supplied)>0
        if supplied.shape!=(H,W): raise ValueError('goal_foreground_coordinates_invalid')
        overlap=np.logical_and(supplied,gm).sum()/max(np.logical_or(supplied,gm).sum(),1)
        if overlap<.7: raise ValueError('goal_foreground_not_rigidly_realizable')
    goal=compact(gm,gr)
    return gm,gr,goal,{'similarity_approximation':approximate,'transform':transform.tolist(),
                     'semantic_orientation': 'external; rotation is relative to observed template'}
