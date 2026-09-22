import numpy as np
import torch
from geometry import SAM2ImagePredictor, proposals, shape_from_mask, features, rotation


def convert_scene(model,observation,spec):
    if 'tracks' in observation:
        return observation
    needed=['rgb_third','camera','T_base_cam','top_z_base_m']
    if any(k not in observation for k in needed):
        return None
    model.sam.eval();predictor=SAM2ImagePredictor(model.sam)
    image=np.asarray(observation['rgb_third'],np.uint8)
    roi=np.asarray(observation['workspace_mask'],np.uint8) if 'workspace_mask' in observation else None
    with torch.no_grad():
        masks=proposals(image,predictor,automatic=True,roi=roi)
    tracks=[]
    for m in masks:
        s=shape_from_mask(m,observation['camera'],np.array(observation['T_base_cam']),float(observation['top_z_base_m']))
        if s is not None:
            tracks.append({'instance_id':'proposal_'+str(len(tracks)),'xy':s['xy'].tolist(),'normal':s['normal'].tolist(),'loop':s['loop'].tolist()})
    out=dict(observation);out['tracks']=tracks;out['association_required']=True
    return out


def empty(status):
    return {'memory':{},'decision':None,'status':status}


def act(model,observation,call,memory,spec):
    if call.get('interrupted',False):
        return empty('interrupted')
    if not observation or not call:
        return empty('needs_view')
    if not observation.get('calibration_valid',False):
        return empty('calibration_required')
    if call.get('scene_revision')!=observation.get('revision') or float(observation.get('age_s',1))>.20:
        return empty('stale_scene')
    if float(observation.get('uncertainty_m',1))>.008:
        return empty('needs_view')
    scene=convert_scene(model,observation,spec)
    if scene is None:
        return empty('needs_view')
    if scene.get('association_required',False):
        out=empty('associate_instances');out['scene']={'tracks':scene['tracks'],'revision':scene['revision'],'frame':'base_xy'}
        return out
    instance=call.get('instance_id');selected=[t for t in scene['tracks'] if t['instance_id']==instance]
    if len(selected)!=1 or 'goal_delta' not in call:
        return empty('ambiguous_geometry')
    if any(k not in scene for k in ['workspace','tool_radius_m','contact_height_m']):
        return empty('calibration_required')
    goal=list(call['goal_delta'])
    if len(goal)!=3 or not np.isfinite(goal).all() or not .001<=float(scene['tool_radius_m'])<=.01 or float(call.get('max_stroke_m',.02))<=0:
        return empty('invalid_constraints')
    if float(call.get('position_tolerance_m',.008))<float(scene['uncertainty_m']):
        return empty('unsupported_precision')
    key={'instance_id':instance,'goal_delta':goal,'calibration_id':scene.get('calibration_id'),'reference_id':call.get('reference_id')}
    retries=0
    if memory and memory.get('key')==key:
        retries=int(memory.get('nonprogress',0))
        if call.get('last_result')=='no_progress':
            retries+=1
        elif call.get('last_result')=='progress':
            retries=0
    if retries>=2:
        out=empty('executor_replan');out['memory']={'key':key,'nonprogress':retries};return out
    xy=np.array(selected[0]['xy'],np.float64);normal=np.array(selected[0]['normal'],np.float64);loop=np.array(selected[0]['loop'],int)
    if xy.shape!=(96,2) or normal.shape!=(96,2) or loop.shape!=(96,) or not np.isfinite(xy).all() or not np.isfinite(normal).all():
        return empty('ambiguous_geometry')
    normal=normal/np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1.e-8)
    center=xy.mean(0);R=rotation(float(goal[2]));goal_xy=(xy-center)@R.T+center+np.array(goal[:2])
    poserr=float(np.linalg.norm(goal[:2]));yawerr=abs(float(goal[2]))
    if poserr<float(call.get('position_tolerance_m',.008)) and yawerr<float(call.get('yaw_tolerance_rad',.08)) and observation.get('stable',False):
        return empty('already_at_goal')
    obslist=[np.array(t['xy']) for t in scene['tracks'] if t['instance_id']!=instance]
    obstacles=np.concatenate(obslist) if obslist else np.zeros((0,2))
    point,cand,feasible,action,ids=features(xy,normal,loop,goal_xy,obstacles,float(scene['tool_radius_m']),scene['workspace'],min(.02,float(call.get('max_stroke_m',.02))))
    if not feasible.any():
        return empty('no_safe_candidate')
    device=next(model.parameters()).device
    with torch.no_grad():
        logits=model(torch.as_tensor(point[None],device=device),torch.as_tensor(cand[None],device=device),torch.as_tensor(feasible[None],device=device)).squeeze(0)
        best=int(torch.argmax(logits).item())
    a=action[best];pid=int(ids[best])
    decision={'scene_revision':scene['revision'],'instance_id':instance,'frame':'base_xy','contact_point_m':a[:2].tolist(),'outward_normal':normal[pid].tolist(),'direction_unit':a[2:4].tolist(),'stroke_length_m':float(a[4]),'boundary_loop_id':int(loop[pid]),'contact_height_m':float(scene['contact_height_m']),'tool_radius_m':float(scene['tool_radius_m']),'uncertainty_m':float(scene['uncertainty_m']),'max_duration_s':1.,'requires_arrival_revalidation':True}
    return {'status':'proposed','decision':decision,'memory':{'key':key,'nonprogress':retries,'revision':scene['revision']},'measured_error':{'translation_m':poserr,'yaw_rad':yawerr}}


def executor_request(decision,executor_contract,spec):
    needed=['T_flange_tool','R_base_tool','tool_collision_model_id','guard_profile_id']
    if not executor_contract.get('commissioned',False) or any(not executor_contract.get(k) for k in needed):
        return {'status':'calibration_required','execute':False}
    if decision['scene_revision']!=executor_contract.get('scene_revision'):
        return {'status':'stale_scene','execute':False}
    c=np.array(decision['contact_point_m']);n=np.array(decision['outward_normal']);d=np.array(decision['direction_unit']);r=decision['tool_radius_m'];L=min(.02,decision['stroke_length_m'])
    start=np.r_[c+n*r,decision['contact_height_m']];end=start+np.r_[d*L,0.]
    rotation_base=np.array(executor_contract['R_base_tool']);chain=np.array(executor_contract['T_flange_tool']);goals=[]
    for p in [start+np.r_[n*.015,0.],start,end]:
        T=np.eye(4);T[:3,:3]=rotation_base;T[:3,3]=p;goals.append((T@np.linalg.inv(chain)).tolist())
    return {'status':'objective_only','execute':False,'scene_revision':decision['scene_revision'],'frame':'base','units':'m_rad_s','mode':'prepare_then_revalidate_then_single_push','T_base_flange_standoff':goals[0],'T_base_flange_contact':goals[1],'T_base_flange_end':goals[2],'selected_contact_allowance':{'instance_id':decision['instance_id'],'point_m':decision['contact_point_m'],'loop_id':decision['boundary_loop_id']},'tool_collision_model_id':executor_contract['tool_collision_model_id'],'protected_instances':executor_contract.get('protected_instances',[]),'constraints':{'speed_m_s':min(.02,float(executor_contract.get('speed_m_s',.01))),'duration_s':1.,'must_subdivide_to_duration':True,'guard_profile_id':executor_contract['guard_profile_id'],'stop_on':['stale_scene','unexpected_neighbor_motion','contact_lost','torque_guard','interruption'],'inner_loop_requires_full_envelope_insertion_check':True},'handoff':'Prepare completion is not success. Reobserve/revalidate before one stroke; no stale automatic continuation.'}


def calling_cases(arrays,metadata,spec):
    xy=arrays['xy'][0];g=arrays['goal_xy'][0];x=xy-xy.mean(0);y=g-g.mean(0)
    theta=float(np.arctan2(np.sum(x[:,0]*y[:,1]-x[:,1]*y[:,0]),np.sum(x*y)));shift=(g.mean(0)-xy.mean(0)).tolist()
    observation={'tracks':[{'instance_id':'actor','xy':xy.tolist(),'normal':arrays['normal'][0].tolist(),'loop':arrays['loop'][0].tolist()}],'workspace':[[.1,-.7],[1.,-.7],[1.,.7],[.1,.7]],'tool_radius_m':.003,'contact_height_m':.045,'calibration_valid':True,'calibration_id':'fixture_only','uncertainty_m':.006,'age_s':.02,'revision':11,'stable':False}
    call={'instance_id':'actor','scene_revision':11,'reference_id':'ref','goal_delta':shift+[theta],'max_stroke_m':.01}
    executor={'commissioned':True,'scene_revision':11,'T_flange_tool':np.eye(4).tolist(),'R_base_tool':np.eye(3).tolist(),'tool_collision_model_id':'synthetic_fixture','guard_profile_id':'test_only'}
    def case(name,o,c,status,decision,reset=True):
        return {'name':name,'observation':o,'call':c,'reset':reset,'executor_contract':executor,'expected_status':status,'expect_decision':decision}
    cases=[case('real_contour_synthetic_executor',observation,call,'proposed',True)]
    cases.append(case('missing_scene',{},call,'needs_view',False))
    stale=dict(call);stale['scene_revision']=10;cases.append(case('stale_revision',observation,stale,'stale_scene',False,False))
    interrupted=dict(call);interrupted['interrupted']=True;cases.append(case('interruption_resets',observation,interrupted,'interrupted',False,False))
    bad=dict(observation);bad['calibration_valid']=False;cases.append(case('uncommissioned_geometry',bad,call,'calibration_required',False))
    done=dict(call);done['goal_delta']=[0,0,0];stable=dict(observation);stable['stable']=True;cases.append(case('stable_geometric_goal',stable,done,'already_at_goal',False))
    fine=dict(call);fine['position_tolerance_m']=.001;cases.append(case('unsupported_tolerance',observation,fine,'unsupported_precision',False))
    return cases
