"""Executable inventory and call-local semantic interface for the two priors."""
import copy
import numpy as np
import torch
from perception import (Chart,convert_observation,selected_features,full_mask,compact,mask_iou,
                        goal_pack,finite,W,H)
from goals import polygon_mask,render_goal,registration
from data_pipeline import GRAPH_OFFSETS,VISUAL_OFFSETS,RAW_KEYS


def calibration_version(metadata):
    return str(metadata['calibration']['generated'])


def equal(a,b):
    if isinstance(a,dict) and isinstance(b,dict):
        return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple,np.ndarray)) or isinstance(b,(list,tuple,np.ndarray)):
        try: return bool(np.array_equal(np.asarray(a),np.asarray(b)))
        except (ValueError,TypeError): return False
    return a==b


def inventory(observation,scene_version,memory=None):
    """Automatic proposals, not character recognition. HLA chooses one instance.
    Persistent inventory tracking is separate from call-local motor memory.
    """
    try:
        if not isinstance(scene_version,str) or not scene_version: raise ValueError('scene_version_required')
        version=calibration_version(observation['metadata'])
        if memory is None or memory['scene_version']!=scene_version or memory['version']!=version:
            memory={'scene_version':scene_version,'version':version,'chart':Chart(observation['metadata']),
                    'tracker':None,'previous':None}
        frame,tracker=convert_observation(observation,memory['chart'],memory['tracker'],memory['previous'])
        memory['tracker']=tracker; memory['previous']={k:observation.get(k) for k in RAW_KEYS}
        rgb=memory['chart'].image(observation['third_rgb'],'third')
        instances=[]
        for tid,p in frame['observed'].items():
            name=scene_version+'/track_'+str(tid); mask=full_mask(p)
            template={'instance_id':name,'scene_version':scene_version,'template_id':name+'/at_'+str(observation['sample_index']),
                      'calibration_version':version,'reference_t':float(observation['t']),
                      'mask':mask,'rgb':rgb*mask[:,:,None],'confidence':p['confidence']}
            instances.append({'instance_id':name,'center_uv':p['center'].tolist(),'confidence':p['confidence'],
                              'template':template,'semantic_label':None,'orientation_semantics':'uninterpreted observed shape'})
        return {'status':'available' if instances else 'unavailable','instances':instances,'memory':memory,
                'diagnostics':{'reason':'warm-material foreground proposals; no open-alphabet recognizer',
                               'calibration_version':version,'image_chart':'undistorted K unchanged, 1280x720'}}
    except (ValueError,KeyError,TypeError) as exc:
        return {'status':'unavailable','instances':[],'memory':None,'diagnostics':{'reason':str(exc)}}


def history_batch(policy_id,history,goal,device):
    offsets=GRAPH_OFFSETS if policy_id=='contour_push' else VISUAL_OFFSETS
    # Match original paired-row offsets, NOT call count or nearest timestamp.
    # A 20-Hz consumer of a 30-Hz recorder gets explicitly masked skipped rows.
    by_index={f['source_index']:f for f in history}; current=history[-1]['source_index']
    frames=[]; hm=[]
    for lag in offsets:
        wanted=current-lag; exists=wanted in by_index
        frames.append(by_index[wanted] if exists else history[0]); hm.append(float(exists))
    def pack(key,dtype=torch.float32,scale=1.):
        return torch.as_tensor(np.stack([f[key] for f in frames])[None].copy(),device=device,dtype=dtype)/scale
    batch={'state':pack('state'),'history_mask':torch.tensor([hm],device=device,dtype=torch.float32)}
    if policy_id=='contour_push':
        batch.update({'nodes':pack('nodes'),'node_mask':pack('node_mask'),
                      'edges':torch.as_tensor(np.stack([f['edges'] for f in frames])[None].copy(),device=device,dtype=torch.long)})
    else:
        for key in ('third','wrist','local','marker'): batch[key]=pack(key,scale=255.)
        batch['goal']=torch.as_tensor(goal[None].copy(),device=device,dtype=torch.float32)/255.
    return batch


def answer(status,reason,memory,diagnostics=None,action=None):
    d={'reason':reason,'execution_enabled':False,'stop_request':'Use only an independently verified stop interface',
       'tool_clearance_m':None,'contact_state':None,'word_success':None}
    if diagnostics: d.update(diagnostics)
    return {'action':action,'memory':memory,'status':status,'diagnostics':d}


def prepare_call(observation,call,policy_id):
    for key in ('call_id','scene_version','instance_id','template','spatial_goal','workspace_polygon',
                'tolerance_px','tolerance_deg','max_duration_s'):
        if key not in call: raise ValueError('missing_call_argument_'+key)
    if call.get('policy_id',policy_id)!=policy_id: raise ValueError('loaded_policy_and_call_differ')
    for key in ('call_id','scene_version','instance_id'):
        if not isinstance(call[key],str) or not call[key]: raise ValueError('invalid_'+key)
    template=call['template']
    if template['instance_id']!=call['instance_id'] or template['scene_version']!=call['scene_version']:
        raise ValueError('template_instance_or_scene_mismatch')
    if not finite(template.get('reference_t')) or abs(float(observation['t'])-float(template['reference_t']))>1.:
        raise ValueError('new_invocation_needs_a_current_observation_template')
    if float(template.get('confidence',0.))<.4: raise ValueError('template_confidence_insufficient')
    if not 5<=float(call['tolerance_px'])<=100: raise ValueError('pixel_tolerance_invalid_or_below_calibration_uncertainty')
    if not 5<=float(call['tolerance_deg'])<=180: raise ValueError('angular_tolerance_invalid')
    if not .1<=float(call['max_duration_s'])<=300: raise ValueError('duration_invalid')
    md=observation['metadata']; chart=Chart(md); version=calibration_version(md)
    workspace=polygon_mask(call['workspace_polygon'])
    goal_mask,goal_rgb,gp,render=render_goal(template,call['spatial_goal'],version,workspace)
    frame,tracker=convert_observation(observation,chart,None,None)
    reference=compact(np.asarray(template['mask'],np.uint8),np.asarray(template['rgb'],np.uint8))
    matches=[]
    for tid,p in frame['observed'].items():
        overlap=mask_iou(p,reference); distance=np.linalg.norm(p['center']-reference['center'])
        if distance<=60: matches.append((overlap-.002*distance,tid))
    matches.sort(reverse=True)
    if not matches or matches[0][0]<.25: raise ValueError('template_not_associated_to_current_foreground')
    if len(matches)>1 and matches[0][0]-matches[1][0]<.15: raise ValueError('current_instance_ambiguous')
    chosen=matches[0][1]; obstacles=np.zeros((H,W),np.uint8)
    for item in call.get('obstacles',[]):
        if item.get('instance_id')==call['instance_id']: continue
        m=np.asarray(item['mask'],np.uint8)
        if m.shape!=(H,W): raise ValueError('obstacle_chart_invalid')
        if item.get('calibration_version')!=version: raise ValueError('obstacle_calibration_mismatch')
        obstacles=np.maximum(obstacles,m>0)
    if np.logical_and(goal_mask,obstacles).any(): raise ValueError('goal_overlaps_protected_obstacle')
    memory={'call':copy.deepcopy(call),'policy_id':policy_id,'chart':chart,'calibration':copy.deepcopy(md['calibration']),
            'tracker':tracker,'selected_id':chosen,'reference':reference,'goal_mask':goal_mask,
            'goal_piece':gp,'goal_pack':goal_pack(goal_mask,goal_rgb),'workspace':workspace,'obstacles':obstacles,
            'render':render,'started':float(observation['t']),'previous':None,'history':[],
            'monitor':[],'stable_since':None,'neighbors':{k:p['center'].copy() for k,p in frame['observed'].items() if k!=chosen},
            'last_t':None,'last_index':None}
    return memory,frame


def act(model,observation,call,memory,spec):
    policy_id=spec['policy_id']
    try:
        if not isinstance(call,dict): raise ValueError('structured_call_required_not_a_word')
        if not isinstance(observation,dict): raise ValueError('paired_observation_dictionary_required')
        if call.get('cancel',False): return answer('cancelled','Caller cancellation; no synthetic zero command',None)
        for key,shape in [('q',(7,)),('T_base_ee',(4,4)),('T_base_flange',(4,4))]:
            if not finite(observation.get(key),shape): raise ValueError('required_state_missing_'+key)
        now=float(observation['t'])
        if not np.isfinite(now): raise ValueError('robot_timestamp_invalid')
        index=observation['sample_index']
        if not isinstance(index,(int,np.integer)) or index<0: raise ValueError('original_paired_sample_index_required')
        index=int(index)
        for view in ('third','wrist'):
            age=observation.get('img_age_'+view)
            if not finite(age) or abs(float(age))>.5: raise ValueError('camera_age_invalid_'+view)
        changed=memory is None or memory.get('policy_id')!=policy_id or not equal(memory['call'],call)
        if changed:
            memory,frame=prepare_call(observation,call,policy_id)
        else:
            if not equal(observation['metadata']['calibration'],memory['calibration']):
                return answer('unavailable','Calibration changed; obtain a fresh template and start a new call',None)
            if memory['last_t'] is not None and (now<=memory['last_t'] or index<=memory['last_index']):
                return answer('waiting','A strictly newer paired robot record is required',memory)
            frame,tracker=convert_observation(observation,memory['chart'],memory['tracker'],memory['previous'])
            memory['tracker']=tracker
        if now-memory['started']>float(call['max_duration_s']):
            return answer('timeout','Invocation duration exceeded; not success',memory)
        memory['last_t']=now; memory['last_index']=index
        memory['previous']={k:observation.get(k) for k in RAW_KEYS}
        selected=frame['observed'].get(memory['selected_id'])
        feat=selected_features(frame,selected,memory['goal_mask'],memory['goal_piece']['center'])
        feat['third']=frame['third'].transpose(2,0,1); feat['wrist']=frame['wrist'].transpose(2,0,1)
        feat['source_index']=index
        memory['history'].append(feat); memory['history']=[f for f in memory['history'] if index-f['source_index']<=14]
        if not feat['valid']:
            memory['stable_since']=None
            return answer('uncertain','Selected causal track not currently observed; no completed/anchor-derived contour',memory,
                          {'track_valid':False,'request':'Reobserve; cancel/reinvoke if identity cannot be recovered'})
        cur=full_mask(selected)
        if np.logical_and(cur,memory['workspace']==0).sum()>0:
            return answer('blocked','Selected foreground outside caller allowed image workspace',memory)
        if np.logical_and(cur,memory['obstacles']).sum()>0:
            return answer('blocked','Selected foreground intersects a caller protected obstacle mask',memory)
        visibility=selected['area']/max(memory['reference']['area'],1.)
        if visibility<.3 or visibility>3.:
            return answer('uncertain','Shape area changed incompatibly with the selected planar instance',memory)
        if policy_id=='contour_push' and (selected['confidence']<.45 or visibility<.5):
            return answer('uncertain','Contour observability insufficient for boundary_graph; identity may remain usable in RGB',memory)
        neighbors=[]
        threshold=float(call.get('neighbor_motion_threshold_px',20.))
        if not np.isfinite(threshold) or threshold<5: raise ValueError('neighbor_motion_threshold_invalid')
        for tid,initial in memory['neighbors'].items():
            if tid in frame['observed']:
                shift=float(np.linalg.norm(frame['observed'][tid]['center']-initial))
                if shift>threshold: neighbors.append({'track':int(tid),'displacement_px':shift})
        centroid=float(np.linalg.norm(selected['center']-memory['goal_piece']['center']))
        overlap=mask_iou(selected,memory['goal_piece']); reg=registration(selected,memory['goal_piece'])
        angle_error=min([abs(x) for x in reg['angles_deg']] or [180.])
        hist=memory['monitor']; hist.append([now,centroid,overlap]); memory['monitor']=hist[-60:]
        trend=(centroid-hist[0][1])/max(now-hist[0][0],.001)
        signs=np.sign(np.diff([h[1] for h in hist])); oscillations=int(np.sum(signs[1:]*signs[:-1]<0)) if len(signs)>1 else 0
        d={'scene_version':call['scene_version'],'instance_id':call['instance_id'],'track_valid':True,
           'track_confidence':selected['confidence'],'visible_area_ratio':visibility,'centroid_error_px':centroid,
           'foreground_iou':overlap,'angle_error_deg':angle_error,'rotation_hypotheses_deg':reg['angles_deg'],
           'rotation_ambiguous':reg['ambiguous'],'discrepancy_trend_px_per_s':trend,'oscillation_count_proxy':oscillations,
           'possible_neighbor_displacement':neighbors,'tool_projection_uv':frame['tool'].tolist(),
           'tool_projection_edge_support':frame['tool_conf'],'tool_clearance_m':None,
           'rendering':memory['render'],'call_reset':changed,'view_valid':[True,True],
           'recontact_count':None,'note':'Tool edge support is not a calibrated silhouette, clearance, or contact measurement.'}
        if neighbors: return answer('uncertain','Possible unintended neighbor movement; HLA must recheck scene',memory,d)
        close=(centroid<=float(call['tolerance_px']) and angle_error<=float(call['tolerance_deg']) and overlap>=.78)
        if close:
            if memory['stable_since'] is None: memory['stable_since']=now
        else: memory['stable_since']=None
        if memory['stable_since'] is not None and now-memory['stable_since']>=.5:
            d['goal_geometry_observed']=True
            return answer('goal_geometry_observed','Foreground stable for 0.5 s; request verified stop and independent tool-clearance/readability check',memory,d)
        batch=history_batch(policy_id,memory['history'],memory['goal_pack'],spec['device'])
    except (ValueError,KeyError,TypeError) as exc:
        return answer('unavailable',str(exc),memory)
    # Neural computation is outside input-error handling; implementation errors are not concealed.
    with torch.no_grad():
        out=model(batch,augment=False); a=model.decode_mean(out)[0]
        prob=torch.softmax(out['logits'],dim=-1); av=(out['mean']*prob[:,:,None]).sum(1,keepdim=True)
        variance=(prob[:,:,None]*(torch.exp(2*out['log_scale'])+(out['mean']-av).square())).sum(1)
        dispersion=(variance.sqrt()*model.action_std)[0].cpu().numpy().tolist()
    value=a.detach().cpu().numpy()
    if not finite(value,(7,)): return answer('unavailable','Nonfinite model candidate',memory,d)
    d['action_std_native']=dispersion; d['candidate_units']='recorded command dq; hardware interpretation unverified'
    return answer('running','Learned candidate only; verified decoder/controller required',memory,d,value.tolist())
