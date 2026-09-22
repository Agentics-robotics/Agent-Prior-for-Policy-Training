import copy
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from real_robot.training_pipeline import public
from data_tools import image_crop, valid_pose, age_valid, base_features, history_features


def prediction(model,batch):
    logits,mu,logsd = model(batch)
    ids = torch.arange(len(mu),device=mu.device)
    selected = mu[ids,logits.argmax(-1)]*model.scales
    prob = torch.softmax(logits,dim=-1)
    center = (prob[:,:,None]*mu).sum(1)
    var = (prob[:,:,None]*(torch.exp(2*logsd)+(mu-center[:,None,:])**2)).sum(1)
    return torch.cat((selected,var.clamp_min(1e-8).sqrt()*model.scales),dim=-1)


def invalid(status,reason):
    return {'memory':None,'decision':None,'status':status,'reason':reason,'completion':'unknown'}


def parse_sample(sample,cfg):
    T = np.asarray(sample['T_base_ee'],dtype=np.float64)
    tau = np.asarray(sample['tau_ext'],dtype=np.float64)
    width, t = float(sample['gripper_width_m']), float(sample['t'])
    ages = np.asarray(sample['image_age_s'],dtype=np.float64)
    if not valid_pose(T) or tau.shape != (7,) or ages.shape != (2,) or not np.isfinite(tau).all() or not np.isfinite([width,t]).all():
        raise ValueError('Invalid state')
    vm = age_valid(ages,cfg)
    imgs = []
    for v,cam in enumerate(['third','wrist']):
        rgb = sample.get(cam+'_rgb')
        if rgb is None:
            vm[v] = False
            im = np.zeros((3,cfg['image_height'],cfg['image_width']),np.uint8)
        else:
            im = image_crop(rgb,cam,cfg)
        if not vm[v]:
            im[:] = 0
        imgs.append(im)
    return {'images':np.stack(imgs).tolist(),'base':base_features(T,tau,width,ages,vm).tolist(),'T':T.tolist(),'t':t,'valid':bool(vm.any()),'width':width}


def encode_history(history,goal):
    history = history[-5:]
    count = len(history)
    images = np.zeros((5,2,3,144,192),np.uint8)
    base = np.zeros((5,21),np.float32)
    poses = np.repeat(np.eye(4)[None],5,axis=0)
    times, mask = np.zeros(5,np.float64), np.zeros(5,np.float32)
    for j,s in enumerate(history):
        k = 5-count+j
        images[k], base[k], poses[k], times[k], mask[k] = np.asarray(s['images'],np.uint8), s['base'], s['T'], s['t'], s['valid']
    return images,history_features(base,poses,times,mask),mask,np.asarray(goal,np.uint8)


def act_impl(model,observation,call,memory,spec):
    # No recording access: everything is invocation-time evidence or causal memory.
    if call.get('interrupt',False) or call.get('feedback','ok') in ['interrupted','infeasible','contact_limit','tool_slip','tracking_error','no_progress']:
        return invalid('interrupted','Cancel old goals; revalidate attachment; preserve initial goal in Agent.')
    if call.get('task') != 'turn_over_in_same_pan' or call.get('tool_template') != 'demonstrated_spatula' or 'landing_point' in call:
        return invalid('unsupported_goal','Only the demonstrated tool and same-pan turn-over goal are supported.')
    if not all(call.get(k,False) for k in ['tool_retained','scene_valid','supported_context','authorized_for_proposal']):
        return invalid('entry_not_verified','Retained tool, observed supported context and authorization required.')
    cfg = spec['preprocessing']
    if call.get('calibration_id') != cfg['calibration_id']:
        return invalid('invalid_observation','Camera/TCP setup identifier mismatch.')
    session, refid = call.get('session_id'), call.get('reference_id')
    if not isinstance(session,str) or not session or not isinstance(refid,str) or not refid:
        return invalid('invalid_observation','Session/reference ID required.')
    reset = call.get('reset',False) or memory is None
    if not reset and (memory.get('session_id') != session or memory.get('reference_id') != refid):
        return invalid('goal_changed','Reset explicitly; do not redefine initial face on retry.')
    try:
        if reset:
            goal = image_crop(call['initial_third_rgb'],'third',cfg).tolist()
            history = []
        else:
            goal, history = memory['goal'], list(memory['history'])
        obsid = observation['observation_id']
        if not isinstance(obsid,str):
            raise ValueError('String observation id required')
        if not reset and obsid == memory['observation_id']:
            return invalid('stale_observation','Observation already used.')
        samples = observation['samples']
        if not 1 <= len(samples) <= 5:
            raise ValueError('One through five recent samples required')
        for s in samples:
            item = parse_sample(s,cfg)
            if history:
                dt = item['t']-history[-1]['t']
                if dt <= 0:
                    return invalid('stale_observation','Non-increasing sample time; never replay.')
                if dt < .05 or dt > .2:
                    history = []
            history.append(item)
            history = history[-5:]
        last = history[-1]
        now = float(observation['now'])
        if not np.isfinite(now) or now < last['t'] or now-last['t'] > .1:
            return invalid('stale_observation','Robot observation expired or clock mismatch.')
        if not last['valid']:
            return invalid('invalid_observation','Both RGB views missing/stale/outside skew tolerance.')
        if not .025 <= last['width'] <= .045:
            return invalid('entry_not_verified','Width outside observed hold envelope; width alone never establishes retention.')
        images,f,mask,g = encode_history(history,goal)
        device = next(model.parameters()).device
        b = {'images':torch.tensor(images[None],dtype=torch.float32,device=device)/255.,'features':torch.tensor(f[None],dtype=torch.float32,device=device),'mask':torch.tensor(mask[None],dtype=torch.float32,device=device),'goal':torch.tensor(g[None],dtype=torch.float32,device=device)/255.}
        with torch.no_grad():
            out = prediction(model,b)[0].cpu().numpy()
        limits = np.asarray(call['max_predictive_spread'],dtype=float)
        if limits.shape != (6,) or not np.isfinite(limits).all() or np.any(limits<=0):
            raise ValueError('Six positive commissioning spread limits required')
        if not np.isfinite(out).all() or np.any(out[6:]>limits):
            return invalid('uncertain','Predictive spread exceeds commissioning limits; not a certified OOD detector.')
        decision = {'kind':'ee_local_motion','observation_id':obsid,'session_id':session,'timestamp':last['t'],'expires_at':last['t']+.1,'horizon_s':.1,'frame':'current_recorded_ee','velocity':out[:6].tolist(),'spread':out[6:].tolist(),'T_base_ee':last['T'],'gripper':'retain_validated_grasp','completion':'unknown'}
        return {'memory':{'session_id':session,'reference_id':refid,'goal':goal,'history':history,'observation_id':obsid},'decision':decision,'status':'active','history_length':len(history),'completion':'unknown','support':'caller_observed_not_force_measured'}
    except (KeyError,ValueError,TypeError,IndexError) as exc:
        return invalid('invalid_observation',str(exc))


def request_impl(decision,executor_contract,spec):
    if decision is None:
        return {'operation':'cancel','reason':'No valid proposal; use installed safety stop semantics.'}
    e = executor_contract
    if not all(e.get(k,False) for k in ['contact_tracking_validated','collision_geometry_validated','workspace_interlock_clear','retention_validated']):
        return {'operation':'reject','reason':'Unverified physical execution prerequisite'}
    try:
        now = float(e['now'])
        if not np.isfinite(now) or now < decision['timestamp'] or now >= decision['expires_at']:
            return {'operation':'reject','reason':'expired_or_clock_mismatch'}
        A, FEE, u = np.asarray(decision['T_base_ee'],float), np.asarray(e['T_flange_ee'],float), np.asarray(decision['velocity'],float)
        if not valid_pose(A) or not valid_pose(FEE) or u.shape != (6,) or not np.isfinite(u).all():
            raise ValueError('Invalid geometry')
        if np.linalg.norm(u[:3])>float(e['max_linear_speed_m_s']) or np.linalg.norm(u[3:])>float(e['max_angular_speed_rad_s']):
            return {'operation':'reject','reason':'speed_limit_no_silent_retiming'}
        goal = A.copy()
        goal[:3,3] += A[:3,:3] @ u[:3] * .1
        goal[:3,:3] = A[:3,:3] @ Rotation.from_rotvec(u[3:]*.1).as_matrix()
        bounds = np.asarray(e['workspace_bounds_m'],float)
        if bounds.shape != (2,3) or not np.isfinite(bounds).all() or np.any(goal[:3,3]<bounds[0]) or np.any(goal[:3,3]>bounds[1]):
            return {'operation':'reject','reason':'workspace'}
        flange = goal @ np.linalg.inv(FEE)
        return {'operation':'track_cartesian_contact_segment','observation_id':decision['observation_id'],'session_id':decision['session_id'],'start_T_base_ee':A.tolist(),'goal_T_base_ee':goal.tolist(),'goal_T_base_flange':flange.tolist(),'duration_s':.1,'expires_at':decision['expires_at'],'interpolation':'straight_origin_translation_and_local_SO3_geodesic','path_timing_policy':'preserve_or_reject_no_reroute_no_material_retime','gripper':'retain_validated_grasp','scene_geometry_id':e['scene_geometry_id'],'safety_profile_id':e['safety_profile_id'],'required_feedback':['tracking_error','actual_pose','infeasible','interrupted','contact_limit','tool_slip'],'completion_semantics':'geometry_only_not_egg_flip'}
    except (KeyError,ValueError,TypeError) as exc:
        return {'operation':'reject','reason':str(exc)}


def source_observation(tid,i):
    r = public.records(tid)
    return {'t':float(r['t'][i]),'T_base_ee':r['T_base_ee'][i].tolist(),'tau_ext':r['tau_ext'][i].tolist(),'gripper_width_m':float(r['gripper_width_m'][i]),'image_age_s':[float(r['img_age_third'][i]),float(r['img_age_wrist'][i])],'third_rgb':public.rgb(tid,i,'third'),'wrist_rgb':public.rgb(tid,i,'wrist')}


def cases_impl(arrays,metadata,spec):
    # Source access only in this OFFLINE fixture builder.
    ix = int(np.flatnonzero(arrays['split']==2)[0])
    ep = int(arrays['example_episode'][ix])
    tid, i = metadata['trajectory_ids'][ep], int(arrays['example_source_index'][ix])
    samples = [source_observation(tid,j) for j in range(i-12,i+1,3)]
    obs = {'samples':samples,'now':samples[-1]['t'],'observation_id':'recorded_contract_fixture'}
    call = {'session_id':'contract_fixture','reference_id':'original_initial_scene','task':'turn_over_in_same_pan','tool_template':'demonstrated_spatula','initial_third_rgb':public.rgb(tid,0,'third'),'reset':True,'calibration_id':spec['preprocessing']['calibration_id'],'tool_retained':True,'scene_valid':True,'supported_context':True,'authorized_for_proposal':True,'max_predictive_spread':[10,10,10,10,10,10]}
    e = {'now':samples[-1]['t'],'contact_tracking_validated':True,'collision_geometry_validated':True,'workspace_interlock_clear':True,'retention_validated':True,'T_flange_ee':[[0,1,0,0],[-1,0,0,0],[0,0,1,.24],[0,0,0,1]],'max_linear_speed_m_s':1.,'max_angular_speed_rad_s':4.,'workspace_bounds_m':[[-2,-2,-2],[2,2,2]],'scene_geometry_id':'synthetic_fixture_not_installed','safety_profile_id':'synthetic_fixture_not_approved'}
    hist = [parse_sample(s,spec['preprocessing']) for s in samples]
    images,f,mask,g = encode_history(hist,image_crop(call['initial_third_rgb'],'third',spec['preprocessing']))
    assert np.array_equal(images,arrays['frames'][arrays['sequence'][ix]])
    assert np.allclose(f,arrays['features'][ix],atol=1e-6)
    assert np.array_equal(mask,arrays['history_mask'][ix])
    assert np.array_equal(g,arrays['goals'][ep])
    probe = {'T_base_ee':samples[-1]['T_base_ee'],'velocity':[.01,0,0,0,.02,0],'timestamp':samples[-1]['t'],'expires_at':samples[-1]['t']+.1,'observation_id':'probe','session_id':'fixture'}
    request = request_impl(probe,e,spec)
    assert request['operation'] == 'track_cartesian_contact_segment'
    P, G = np.asarray(probe['T_base_ee']), np.asarray(request['goal_T_base_ee'])
    assert np.allclose(P[:3,:3].T@(G[:3,3]-P[:3,3]),[.001,0,0])
    assert np.allclose(np.asarray(request['goal_T_base_flange'])@np.asarray(e['T_flange_ee']),G)
    unbound = dict(e,contact_tracking_validated=False)
    assert request_impl(probe,unbound,spec)['operation'] == 'reject'
    def case(name,o,c,status,decision=True,executor=None):
        return {'name':name,'observation':o,'call':c,'reset':True,'executor_contract':e if executor is None else executor,'expected_status':status,'expect_decision':decision}
    bad = copy.deepcopy(obs)
    bad['samples'][-1]['third_rgb'], bad['samples'][-1]['wrist_rgb'] = None, None
    single = copy.deepcopy(obs)
    single['samples'][-1]['wrist_rgb'] = None
    old = copy.deepcopy(obs)
    old['now'] += .5
    noinit = dict(call)
    noinit.pop('initial_third_rgb')
    public.write_report('calling_preprocessing_parity.json',{'source':tid,'index':i,'pixels_equal':True,'features_equal_atol':1e-6,'target_inputs_absent':True,'adapter_roundtrip':True,'scope':'Recorded RGB/state with synthetic authorization/safety fixtures; no physical binding certified.'})
    return [case('real_source_full_history',obs,call,'active'),case('reset_without_history',dict(obs,samples=[samples[-1]],observation_id='one_frame'),call,'active'),case('partial_view_masked',single,call,'active'),case('both_views_missing',bad,call,'invalid_observation',False),case('stale_robot',old,call,'stale_observation',False),case('no_initial_reference',obs,noinit,'invalid_observation',False),case('interruption',obs,dict(call,interrupt=True),'interrupted',False),case('tool_not_retained',obs,dict(call,tool_retained=False),'entry_not_verified',False),case('unsupported_goal',obs,dict(call,landing_point=[0,0,0]),'unsupported_goal',False),case('unbound_executor_rejects',obs,call,'active',True,unbound)]
