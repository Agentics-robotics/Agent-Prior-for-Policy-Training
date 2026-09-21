"""Runnable HLA interface. All actions are candidates; there is no hardware I/O."""
import json
import cv2
import numpy as np
import torch
from perception import (current,features,template,unpacked,template_rgb,calibration,center,
                        registration,W,H)
from geometry import graph,goal_image,render_goal,discrepancy
from networks import distribution_prediction


def chart_id(metadata):
    c=calibration(metadata)
    return json.dumps({'third':c['third'],'wrist':c['wrist'],'tcp':metadata.get('tcp')},sort_keys=True,separators=(',',':'))

def observe_scene(observation,scene_version,scene_memory=None):
    """Automatic proposal inventory; arbitrary track IDs, no character recognition."""
    try:
        if not isinstance(scene_version,str) or not scene_version: raise ValueError('scene_version required')
        if scene_memory is not None and scene_memory['scene_version']!=scene_version: scene_memory=None
        fr=current(observation,None if scene_memory is None else {'tracker':scene_memory['tracker']})
        cid=chart_id(observation['metadata'])
        ts=[]
        for k,p in fr['assigned'].items():
            if p['confidence']>=.3:
                ts.append(template(p,fr['rgb'],scene_version,scene_version+'/p'+str(k),cid,observation['sample_index']))
        mem={'tracker':fr['tracker'],'scene_version':scene_version,'sample_index':int(observation['sample_index'])}
        return {'status':'inventory_available' if ts else 'uncertain','instances':ts,'memory':mem,
                'diagnostics':{'recognition':'none','foreground_model':'brown material components',
                               'chart_id':cid,'metric_geometry':False,'num_proposals':len(fr['proposals'])}}
    except (ValueError,KeyError,TypeError) as exc:
        return {'status':'unavailable','instances':[],'memory':None,'diagnostics':{'reason':str(exc)}}

def polygon_mask(polygons):
    m=np.zeros((H,W),np.uint8)
    for polygon in polygons:
        xy=np.asarray(polygon,np.float64)
        if xy.ndim!=2 or xy.shape[0]<3 or xy.shape[1]!=2 or not np.isfinite(xy).all():
            raise ValueError('invalid original-chart polygon')
        if np.any(xy<0) or np.any(xy>np.array([1280,720])): raise ValueError('polygon outside image chart')
        cv2.fillPoly(m,[np.rint(xy/2).astype(np.int32)],1)
    return m

def reexpress_goal(new_template,old_goal_mask):
    """Preserve an intended foreground across policy switches, or explicitly refuse a bad fit.
    old_goal_mask is a numeric 360x640 foreground returned by preview_goal.
    """
    target=np.asarray(old_goal_mask,np.uint8)
    if target.shape!=(H,W) or not np.isin(target,[0,1]).all(): raise ValueError('invalid fixed goal foreground')
    reg=registration(unpacked(new_template),target)
    if not reg['valid'] or not (.88<=reg['scale']<=1.12):
        return {'status':'calibration_required','reason':'new template cannot realize old silhouette with supported rigid image renderer','registration':reg}
    candidates=[]
    for theta in reg['angles_deg_clockwise']:
        goal={'frame':'third_undistorted_original_pixels','center_uv':(2*center(target)).tolist(),
              'theta_deg_clockwise':theta,'accept_similarity_approximation':True}
        try:
            m,_,_=render_goal(new_template,goal)
        except ValueError:
            continue
        score=discrepancy(m,target)['foreground_iou'];candidates.append((score,goal))
    if not candidates: return {'status':'calibration_required','reason':'goal not renderable','registration':reg}
    candidates.sort(key=lambda z:z[0],reverse=True)
    if candidates[0][0]<.65:
        return {'status':'uncertain','reason':'insufficient foreground agreement; do not silently change placement','registration':reg}
    return {'status':'goal_reexpressed','spatial_goal':candidates[0][1],
            'foreground_iou':candidates[0][0],'orientation_hypotheses':reg['angles_deg_clockwise'],
            'semantic_orientation_verified':False}

def preview_goal(template_ref,spatial_goal,workspace_polygon_uv):
    ws=polygon_mask([workspace_polygon_uv])
    mask,rgb,info=render_goal(template_ref,spatial_goal,ws)
    return {'foreground_half':mask,'rgb_half':rgb,'diagnostics':info}

def call_signature(call):
    # Exact contents prevent silent changes of instance, template, geometry, tolerances or budget.
    return json.dumps({k:v for k,v in call.items() if k!='cancel'},sort_keys=True,separators=(',',':'))

def result(status,memory,reason,diagnostics=None,action=None):
    d={'reason':reason,'hardware_io':False,'request_verified_stop':status!='running',
       'tool_clearance_estimate':None,'contact_state':None,'successor_ready':False}
    if diagnostics: d.update(diagnostics)
    return {'action':action,'memory':memory,'status':status,'diagnostics':d}

def initial_memory(ob,call,fr,policy_id):
    for key in ('call_id','scene_version','instance_id','template','spatial_goal','workspace_polygon_uv','tolerances','max_duration_s'):
        if key not in call: raise ValueError('missing caller argument '+key)
    t=call['template']
    if t['scene_version']!=call['scene_version'] or t['instance_id']!=call['instance_id']:
        raise ValueError('scene/instance template mismatch')
    if t['chart_id']!=chart_id(ob['metadata']): raise ValueError('calibration/chart changed')
    age=int(ob['sample_index'])-int(t['sample_index'])
    if not 0<=age<=10: raise ValueError('refresh automatic instance association: stale invocation template')
    mask=unpacked(t); scores=[]
    for k,p in fr['assigned'].items():
        inter=np.logical_and(mask,p['mask']).sum();union=np.logical_or(mask,p['mask']).sum()
        overlap=inter/max(1,union)
        distance=np.linalg.norm(center(mask)-p['center'])
        scores.append((overlap-distance/200.,k))
    scores.sort(reverse=True)
    if not scores or scores[0][0]<.25 or (len(scores)>1 and scores[0][0]-scores[1][0]<.15):
        raise ValueError('ambiguous current template association')
    selected=scores[0][1]
    workspace=polygon_mask([call['workspace_polygon_uv']])
    goal,grgb,gdiag=render_goal(t,call['spatial_goal'],workspace)
    obstacles=polygon_mask(call.get('obstacle_polygons_uv',[]))
    protected={}
    for tt in call.get('protected_templates',[]):
        if tt['chart_id']!=t['chart_id'] or tt['scene_version']!=call['scene_version']: raise ValueError('invalid protected-instance reference')
        pm=unpacked(tt);obstacles=np.maximum(obstacles,pm)
        choices=[(float(np.logical_and(pm,p['mask']).sum()/max(1,np.logical_or(pm,p['mask']).sum())),k) for k,p in fr['assigned'].items() if k!=selected]
        choices.sort(reverse=True)
        if not choices or choices[0][0]<.3: raise ValueError('refresh protected-instance association')
        protected[choices[0][1]]=fr['assigned'][choices[0][1]]['center'].copy()
    if np.any((goal>0)&(obstacles>0)): raise ValueError('blocked: goal overlaps supplied protected/obstacle foreground')
    tol=call['tolerances']
    for key in ('position_px','angle_deg','foreground_iou','stability_s','neighbor_motion_px'):
        if key not in tol or not np.isfinite(float(tol[key])): raise ValueError('invalid tolerance '+key)
    if not (0<tol['position_px']<=100 and 0<tol['angle_deg']<=180 and .1<=tol['foreground_iou']<=1 and .1<=tol['stability_s']<=5 and tol['neighbor_motion_px']>0):
        raise ValueError('invalid tolerances')
    if not 0<float(call['max_duration_s'])<=600: raise ValueError('invalid call budget')
    return {'signature':call_signature(call),'policy_id':policy_id,'tracker':fr['tracker'],'selected':selected,
            'goal':goal,'goal_image':goal_image(goal,grgb),'goal_diagnostics':gdiag,
            'workspace':workspace,'obstacles':obstacles,'protected':protected,
            'neighbors':{k:p['center'].copy() for k,p in fr['assigned'].items() if k!=selected},
            'started':float(ob['t']),'history':[],'errors':[],'stable_since':None,
            'last_sample':None,'previous_obs':None,'terminal':False}

def online_batch(memory,spec):
    history=memory['history'];latest=history[-1][0];selected=[];valid=[]
    for target in range(latest-3*7,latest+1,3):
        choices=[row for row in history if row[0]<=target and target-row[0]<=2]
        if choices: selected.append(choices[-1][1]);valid.append(1.)
        else: selected.append(history[-1][1]);valid.append(0.)
    def ten(x,dtype=torch.float32):return torch.as_tensor(np.array(x,copy=True),dtype=dtype,device=spec['device'])
    b={'state':ten([f['state'] for f in selected])[None], 'history_mask':ten(valid)[None]}
    if memory['policy_id']=='contour_push':
        b['nodes']=ten([f['nodes'] for f in selected])[None]
        b['edges']=ten([f['edges'] for f in selected],torch.long)[None]
        b['node_mask']=ten([f['node_mask'] for f in selected])[None]
    else:
        for key in ('global','wrist','detail'): b[key]=ten([f[key] for f in selected],torch.uint8).permute(0,3,1,2)[None]
        b['goal']=ten(memory['goal_image'],torch.uint8).permute(2,0,1)[None]
    return b

def act(model,observation,call,memory,spec):
    ob=observation
    try:
        if call.get('cancel',False):
            if memory is not None: memory['terminal']=True
            return result('cancelled',memory,'caller cancelled; use verified stop outside this package')
        for k,n in [('q',7),('dq',7),('tau_ext',7),('T_base_ee',16),('T_base_flange',16)]:
            v=np.asarray(ob[k],np.float64)
            if v.size!=n or not np.isfinite(v).all(): raise ValueError('missing/nonfinite measured state '+k)
        for v in ('third','wrist'):
            age=float(ob['img_age_'+v])
            if not np.isfinite(age) or abs(age)>.5: raise ValueError('image age unavailable or exceeds 0.5 seconds; no clock realignment')
        if not np.isfinite(float(ob['t'])): raise ValueError('invalid robot timestamp')
        if memory is not None:
            if memory['policy_id']!=model.policy_id or memory['signature']!=call_signature(call):
                return result('unavailable',None,'new instance/goal/template/control requires memory=None and a new call_id')
            if memory['terminal']: return result('unavailable',memory,'invocation terminated; start a new call')
            if int(ob['sample_index'])<=memory['last_sample'] or float(ob['t'])<=float(memory['previous_obs']['t']):
                return result('unavailable',memory,'non-increasing original sample/time; not re-paired')
            if float(ob['t'])-float(memory['previous_obs']['t'])>.5:
                memory['terminal']=True
                return result('uncertain',memory,'observation gap >0.5 s; stop and refresh association')
            if call['template']['chart_id']!=chart_id(ob['metadata']): raise ValueError('calibration changed during call')
        fr=current(ob,None if memory is None else {'tracker':memory['tracker']})
        if memory is None: memory=initial_memory(ob,call,fr,model.policy_id)
        memory['tracker']=fr['tracker'];memory['last_sample']=int(ob['sample_index'])
        prev=memory['previous_obs']
        memory['previous_obs']={k:ob.get(k) for k in ('t','img_t_third','img_t_wrist')}
        if float(ob['t'])-memory['started']>float(call['max_duration_s']):
            memory['terminal']=True;return result('timeout',memory,'caller time budget exhausted')
        p=fr['assigned'].get(memory['selected'])
        if p is None or p['confidence']<.3:
            memory['stable_since']=None
            return result('uncertain',memory,'selected identity is lost/ambiguous; no stale-mask candidate',{'track_confidence':0. if p is None else float(p['confidence'])})
        if model.policy_id=='contour_push' and p['confidence']<.65:
            memory['stable_since']=None
            return result('uncertain',memory,'contour/association confidence proxy below 0.65; inspect RGB alternative only if identity remains valid',{'track_confidence':float(p['confidence'])})
        if float((p['mask']*memory['workspace']).sum())<.95*p['area']:
            return result('blocked',memory,'selected foreground outside supplied image workspace')
        shifts={str(k):float(np.linalg.norm(fr['assigned'][k]['center']-c)*2) for k,c in memory['neighbors'].items() if k in fr['assigned']}
        protected_shifts=[float(np.linalg.norm(fr['assigned'][k]['center']-c)*2) for k,c in memory['protected'].items() if k in fr['assigned']]
        if any(k not in fr['assigned'] for k in memory['protected']):
            return result('uncertain',memory,'protected instance no longer observable')
        if protected_shifts and max(protected_shifts)>call['tolerances']['neighbor_motion_px']:
            return result('blocked',memory,'possible protected-neighbor displacement exceeds caller tolerance',{'possible_neighbor_displacement_px':shifts})
        f=features(ob,fr,p,prev)
        nodes,edges,nm=graph(p['mask'],memory['goal'],[pp['mask'] for k,pp in fr['assigned'].items() if k!=memory['selected']],f['tool'],f['tool_valid'],p['confidence'])
        f.update({'nodes':nodes,'edges':edges,'node_mask':nm})
        memory['history'].append((int(ob['sample_index']),f))
        memory['history']=[row for row in memory['history'] if int(ob['sample_index'])-row[0]<=30]
        error=discrepancy(p['mask'],memory['goal']);reg=registration(p['mask'],memory['goal'])
        angle=min([min(a,360-a) for a in reg['angles_deg_clockwise']]+[180])
        now=float(ob['t']);memory['errors'].append((now,error['center_error_px']))
        memory['errors']=[e for e in memory['errors'] if now-e[0]<=2.]
        trend=None
        if len(memory['errors'])>2:
            tt=np.asarray([e[0]-now for e in memory['errors']]);ee=np.asarray([e[1] for e in memory['errors']])
            trend=float(np.sum((tt-tt.mean())*(ee-ee.mean()))/max(1e-8,np.sum((tt-tt.mean())**2)))
        differences=np.diff([e[1] for e in memory['errors']]);sig=np.sign(differences[np.abs(differences)>.5])
        osc=int(np.sum(sig[1:]!=sig[:-1])) if len(sig)>1 else 0
        diag={**error,'orientation_residual_deg_modulo_symmetry':angle,'orientation_hypotheses_deg_clockwise':reg['angles_deg_clockwise'],
              'registration_iou':reg['iou'],'track_confidence':float(p['confidence']),
              'contour_quality_proxy':float(p['confidence']),'projected_tcp_uv':(2*f['tool']).tolist(),
              'tcp_in_image':bool(f['tool_valid']),'tool_projection_visually_verified':False,
              'possible_neighbor_displacement_px':shifts,'error_trend_px_per_s':trend,
              'error_direction_changes':osc,'recontact_events':None,'view_consistency':None,
              'scene_version':call['scene_version'],'goal_renderer':memory['goal_diagnostics'],
              'controller_verified':False,'geometric_agreement_only':True,'history_observations':len(memory['history'])}
        tol=call['tolerances'];agree=(error['center_error_px']<=tol['position_px'] and error['foreground_iou']>=tol['foreground_iou'] and angle<=tol['angle_deg'] and reg['valid'])
        if agree:
            if memory['stable_since'] is None: memory['stable_since']=now
        else: memory['stable_since']=None
        if memory['stable_since'] is not None and now-memory['stable_since']>=tol['stability_s']:
            memory['terminal']=True
            return result('goal_observed',memory,'foreground agreement stable; stop and independently verify tool clearance, identity and layout',diag)
        batch=online_batch(memory,spec)
        with torch.no_grad(): candidate,spread=distribution_prediction(model,batch)
        action=candidate[0].detach().cpu().numpy()
        if not np.isfinite(action).all(): return result('unavailable',memory,'nonfinite learned candidate')
        diag['action_distribution_std_native']=spread[0].detach().cpu().tolist()
        return result('running',memory,'learned non-executable candidate; decoder/controller verification still required',diag,action.tolist())
    except (KeyError,ValueError,TypeError,IndexError) as exc:
        reason=str(exc)
        return result('blocked' if 'blocked:' in reason else 'unavailable',memory,reason)


def decode_action(action,controller_contract,spec):
    """Pure command-format gate, NEVER actuates or claims verification on behalf of a caller."""
    def refuse(reason):return {'enabled':False,'command':None,'reason':reason}
    try:
        c=controller_contract
        if not isinstance(c,dict) or c.get('verified') is not True or not c.get('verification_id'):
            return refuse('recording-controller verification absent; candidate only')
        if c.get('mode')!='joint_velocity' or c.get('units')!='rad/s' or c.get('recorded_dq_units')!='rad/s' or c.get('q_units')!='rad':
            return refuse('only independently verified rad/s recorded-command interpretation is supported')
        order=c['joint_order']
        if len(order)!=7 or len(set(order))!=7 or order!=c['recorded_joint_order']:
            return refuse('joint ordering not verified against recording')
        for key in ('timing_verified','limits_verified','watchdog_verified','stop_verified','workspace_check_passed','singularity_check_passed','tool_geometry_verified'):
            if c.get(key) is not True: return refuse('missing external verification: '+key)
        if not 0<float(c['update_hz'])<=1000 or not 0<float(c['hold_s'])<=1 or not 0<float(c['watchdog_s'])<=1:
            return refuse('invalid verified command timing')
        if not 0<=float(c['measured_latency_s'])<=float(c['max_latency_s'])<=float(c['watchdog_s']):
            return refuse('latency/watchdog violation')
        names=['velocity_limits','acceleration_limits','q_min','q_max','current_q','previous_command']
        vals=[np.asarray(c[k],np.float64) for k in names];a=np.asarray(action,np.float64)
        if a.shape!=(7,) or not np.isfinite(a).all() or any(v.shape!=(7,) or not np.isfinite(v).all() for v in vals):
            return refuse('invalid action or live limit context')
        vmax,amax,qlo,qhi,q,previous=vals;dt=float(c['elapsed_since_previous_s'])
        if not np.all(vmax>0) or not np.all(amax>0) or not np.all(qhi>qlo) or not 0<dt<=c['watchdog_s']:
            return refuse('invalid joint limits or elapsed timing')
        if np.any(np.abs(a)>vmax) or np.any(np.abs(a-previous)>amax*dt):
            return refuse('velocity/acceleration limit violation; not silently clipped')
        # Conservative one-hold limit test, not an output position target or IK controller.
        qhold=q+a*float(c['hold_s'])
        if np.any(q<qlo) or np.any(q>qhi) or np.any(qhold<qlo) or np.any(qhold>qhi):
            return refuse('current or one-hold joint bound violation')
        return {'enabled':True,'command':{'mode':'joint_velocity','dq':a.tolist(),'units':'rad/s',
                'joint_order':order,'update_hz':float(c['update_hz']),'hold_s':float(c['hold_s']),
                'verification_id':c['verification_id'],'hardware_io':False},
                'reason':'format/limit gate passed using supplied verified contract; no actuation performed'}
    except (KeyError,ValueError,TypeError):return refuse('incomplete or malformed verified controller contract')
