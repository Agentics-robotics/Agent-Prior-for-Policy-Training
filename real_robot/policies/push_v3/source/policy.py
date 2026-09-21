"""Public hooks and causal, candidate-only agent interface."""
import json
import numpy as np
import cv2
import torch
from geometry import workspace, perceive, init_tracks, update_tracks, scene_labels, center, pix_to_w, w_to_pix, features, HISTORY, finite, base_to_w, rotation_error
from data_pipeline import prepare, batch
from networks import Prior, loss
POLICIES=['tool_waypoint_v1','piece_contact_graph_v1','piece_goal_field_v1']

def prepare_data(spec):
    data=prepare(spec)
    # Extrema floors keep bounded residuals able to represent rare nonzero w
    # and large valid translations. These are native normalizers, not SI claims.
    a=data['arrays']; meta=data['metadata']
    for j,pid in enumerate(spec['policy_ids']):
        u=a['native_action'][a['policy_weight'][:,j]>0]
        scale=meta['normalization'][pid]
        linear=max(scale[0],float(np.max(np.linalg.norm(u[:,:3],axis=1)))/1.8)
        angular=max(scale[3],float(np.max(np.abs(u[:,3:])))/1.8)
        meta['normalization'][pid]=[linear]*3+[angular]*3
    meta['normalization_rule']='Per-policy quantile scales with native magnitude extrema/1.8 floors; rare angular intent retained within residual support.'
    return data

def make_batch(policy_id,arrays,metadata,indices,spec): return batch(policy_id,arrays,metadata,indices,spec)
def build_model(policy_id,spec): return Prior(policy_id,spec['data_metadata']['normalization'][policy_id]).to(spec['device'])
def compute_loss(model,batch,spec): return loss(model,batch)
def predict(model,batch,spec): return model.reconstruct(model(batch)['mean'],batch)

def decode_action(action,controller_contract,spec):
    if action is None or np.asarray(action).shape!=(6,) or not np.isfinite(np.asarray(action,dtype=float)).all():
        return {'enabled':False,'command':None,'reason':'Invalid candidate'}
    return {'enabled':False,'command':None,'reason':'Candidate is recorded v,w encoding only. No audited recording-controller implementation, current kinematic/collision model or verified follower is shipped. Caller assertions cannot enable this decoder; an independently verified adapter implementation is required.'}

def inventory(observation,spec):
    try:
        T,sigma,verified=workspace(observation['metadata']); mask,valid,dis=perceive(observation,T); tracks=init_tracks(mask); result=[]
        for i,tr in enumerate(tracks):
            contours,h=cv2.findContours(tr['mask'],cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE); rings=[]
            for j,c in enumerate(contours): rings.append({'points_W':pix_to_w(c[:,0,:]).tolist(),'hole':bool(h[0,j,3]>=0)})
            result.append({'handle':'scene:%s:%d'%(str(observation['t']),i),'seed_W':pix_to_w(center(tr['mask'])).tolist(),'rings':rings,'semantic_hypotheses':None,'confidence_kind':'unvalidated color component, not identity','area_proxy_m2':float(tr['mask'].sum()/256**2)})
        return {'status':'PERCEPTION_UNCERTAIN' if not result else 'CANDIDATE_INVENTORY','instances':result,'t':float(observation['t']),'T_base_W':T.tolist(),'plane_verified':verified,'plane_sigma_design_margin_m':sigma,'view_disagreement':dis,'semantic_readiness':False,'limitations':'Handles are snapshot-local; HLA must bind a persistent instance_id. No automatic character or upright-orientation recognition.'}
    except (KeyError,ValueError,TypeError) as e: return {'status':'PERCEPTION_UNCERTAIN','reason':str(e),'instances':[]}

def plan_layout(word,inventory_result,semantic_hypotheses,origin_W,reading_direction_W,gap_m=.015):
    """Distinct-instance matching from caller-supplied current semantics."""
    from scipy.optimize import linear_sum_assignment
    items=inventory_result['instances']
    if semantic_hypotheses is None: return {'status':'PERCEPTION_UNCERTAIN','reason':'Live semantic hypotheses are required; none are inferred from shape aliases.'}
    if len(word)>len(items): return {'status':'INFEASIBLE_REQUEST','reason':'Insufficient distinct physical instances'}
    if not word: return {'status':'INFEASIBLE_REQUEST','reason':'Empty request'}
    probs=np.array([[float(semantic_hypotheses.get(x['handle'],{}).get(ch,0.)) for x in items] for ch in word])
    if not np.isfinite(probs).all(): return {'status':'PERCEPTION_UNCERTAIN','reason':'Nonfinite semantic hypotheses'}
    row,col=linear_sum_assignment(-probs)
    if any(probs[i,j]<.8 for i,j in zip(row,col)):
        return {'status':'PERCEPTION_UNCERTAIN','reason':'No unambiguous distinct assignment above caller probability 0.8; distinguish absent inventory from uncertain recognition.'}
    d=np.asarray(reading_direction_W,float); d=d/max(np.linalg.norm(d),1e-9)
    if not np.isfinite(d).all() or np.linalg.norm(d)<.5: return {'status':'INFEASIBLE_REQUEST','reason':'Invalid reading direction'}
    cursor=np.asarray(origin_W,float); goals=[]
    for i,j in zip(row,col):
        obj=items[j]; pts=np.concatenate([np.asarray(r['points_W']) for r in obj['rings']]); extent=float(np.ptp(pts@d)); desired=cursor+d*extent/2
        delta=desired-np.asarray(obj['seed_W']); G=np.eye(3); G[:2,2]=delta
        goals.append({'character':word[i],'snapshot_handle':obj['handle'],'instance_seed_W':obj['seed_W'],'goal_SE2_W':G.tolist(),'semantic_orientation_requires_confirmation':True}); cursor=cursor+d*(extent+max(0.,gap_m))
    return {'status':'PROPOSED_LAYOUT','goals':goals,'requires_agent_checks':['semantic upright orientation','table support','reachability','obstacles and staging','spacing under plane uncertainty'],'not_a_motion_plan':True}

def footprint(call,initial):
    if ('goal_SE2_W' in call)==('goal_footprint_W' in call): raise ValueError('Provide exactly one goal_SE2_W or goal_footprint_W')
    if 'goal_SE2_W' in call:
        G=np.asarray(call['goal_SE2_W'],float).reshape(3,3)
        if not np.isfinite(G).all() or np.linalg.norm(G[2]-[0,0,1])>1e-6 or np.linalg.norm(G[:2,:2].T@G[:2,:2]-np.eye(2))>.01 or np.linalg.det(G[:2,:2])<.99: raise ValueError('goal_SE2_W must be rigid, not scale/reflection')
        P=np.array([[1/256,0,-.2],[0,1/256,-.35],[0,0,1.]]); M=np.linalg.inv(P)@G@P
        out=cv2.warpAffine(initial,M[:2].astype(np.float32),(256,256),flags=cv2.INTER_NEAREST)
        if out.sum()<.85*initial.sum(): raise ValueError('Goal extends outside representation bounds')
        return out
    fp=call['goal_footprint_W']; out=np.zeros((256,256),np.uint8)
    for ring in fp['outer']:
        v=np.asarray(ring,float)
        if not np.isfinite(v).all(): raise ValueError('Nonfinite goal footprint')
        p=np.rint(w_to_pix(v)).astype(np.int32)
        if (p<0).any() or (p>=256).any(): raise ValueError('Goal ring outside workspace map')
        cv2.fillPoly(out,[p],1)
    for ring in fp.get('holes',[]):
        v=np.asarray(ring,float)
        if not np.isfinite(v).all(): raise ValueError('Nonfinite goal hole')
        cv2.fillPoly(out,[np.rint(w_to_pix(v)).astype(np.int32)],0)
    if out.sum()<12: raise ValueError('Goal silhouette empty or too small')
    from geometry import register_masks
    _,fit,_=register_masks(initial,out)
    if fit<.5: raise ValueError('Goal footprint does not match selected observed shape')
    return out

def live_batch(history,spec):
    H=len(history); selected=[history[max(0,H-1-o)] for o in HISTORY]; last=history[-1]; out={}
    for k in ['state','maps','nodes']: out[k]=torch.tensor(np.stack([f[k] for f in selected])[None],dtype=torch.float32,device=spec['device'])
    for k in ['base','basis','score']: out[k]=torch.tensor(last[k][None],dtype=torch.float32,device=spec['device'])
    out['history_valid']=torch.tensor([[float(H-1-o>=0) for o in HISTORY]],dtype=torch.float32,device=spec['device'])
    return out

def unavailable(status,reason,memory=None,diagnostics=None):
    d={'reason':reason,'hardware_enabled':False,'successor_readiness':{'physical_ready':False}}
    if diagnostics is not None: d.update(diagnostics)
    return {'action':None,'memory':memory,'status':status,'diagnostics':d}

def rigid(T):
    a=np.asarray(T,float).reshape(4,4)
    if not np.isfinite(a).all() or np.linalg.norm(a[3]-[0,0,0,1])>1e-5 or np.linalg.norm(a[:3,:3].T@a[:3,:3]-np.eye(3))>.02 or np.linalg.det(a[:3,:3])<.98: raise ValueError('Invalid rigid TCP/flange pose')
    return a

def act(model,observation,call,memory,spec):
    try: return _act(model,observation,call,memory,spec)
    except (KeyError,ValueError,TypeError,IndexError) as e: return unavailable('PERCEPTION_UNCERTAIN','Invalid/missing input: '+str(e),memory)

def _act(model,o,call,mem,spec):
    pid=call['policy_id']
    if pid not in POLICIES or pid!=model.policy_id: return unavailable('ABORTED','Policy/model mismatch',mem)
    if call.get('abort',False): return unavailable('ABORTED','Caller interruption',None)
    if not isinstance(call.get('call_id'),str) or not call['call_id']: raise ValueError('Nonempty call_id required')
    for key in ['max_age_s','timeout_s','speed_cap_m_s','clearance_m','stall_s','tolerance_m','tolerance_rad','neighbor_tolerance_m']:
        if key in call and (not np.isfinite(float(call[key])) or float(call[key])<=0): raise ValueError('Expected positive finite '+key)
    now=float(call['now_s']); t=float(o['t'])
    if not np.isfinite(now) or not np.isfinite(t) or abs(now-t)>float(call.get('max_age_s',.25)): return unavailable('PERCEPTION_UNCERTAIN','Stale/invalid observation clock',mem)
    ages=[abs(float(o.get('img_age_'+c,1.))) for c in ['third','wrist']]
    if not np.isfinite(ages).all() or max(ages)>float(call.get('max_age_s',.25)): return unavailable('PERCEPTION_UNCERTAIN','Stale camera pair',mem)
    E=rigid(o['T_base_ee']); F=rigid(o['T_base_flange']); T,sigma,planeverified=workspace(o['metadata'],call.get('workspace')); tool=pid=='tool_waypoint_v1'
    terminal=np.asarray(call.get('terminal_twist',[0.]*6),float)
    if terminal.shape!=(6,) or not np.isfinite(terminal).all(): raise ValueError('Invalid terminal native twist')
    if tool: rigid(call['goal_tcp_base'])
    if 'exit_tool_base' in call: rigid(call['exit_tool_base'])
    signature=json.dumps({k:call.get(k) for k in ['policy_id','call_id','instance_id','goal_SE2_W','goal_footprint_W','goal_tcp_base','exit_tool_base','workspace','protected_instance_ids','arrival_mode','terminal_twist']},sort_keys=True)
    if mem is not None and mem['signature']!=signature: return unavailable('ABORTED','Goal/instance/frame changed: start new call with memory=None',mem)
    if mem is not None and (t<=mem['last_t'] or int(o['sample_index'])==mem['last_index']): return unavailable('PERCEPTION_UNCERTAIN','Repeated/out-of-order observation: history not advanced',mem)
    if mem is not None and t-mem['last_t']>.5: return unavailable('PERCEPTION_UNCERTAIN','Tracking gap: reacquire identity and reset policy memory',mem)
    external=call.get('live_mask')
    if external is not None and abs(float(call['live_mask_t'])-t)>.05: return unavailable('PERCEPTION_UNCERTAIN','External live mask is stale',mem)
    fg,valid,dis=perceive(o,T,external)
    if mem is None:
        tracks=init_tracks(fg)
        if not tracks: return unavailable('PERCEPTION_UNCERTAIN','No foreground pieces; supply validated live_mask or repair perception')
        centers=np.stack([pix_to_w(center(tr['mask'])) for tr in tracks]); target=0; bindings={}
        if not tool:
            if not call.get('instance_id'): raise ValueError('Persistent instance_id required')
            if not np.isfinite(float(call['inventory_t'])) or abs(float(call['inventory_t'])-t)>.5: raise ValueError('Reacquire a current instance seed')
            seed=np.asarray(call['instance_seed_W'],float)
            if seed.shape!=(2,) or not np.isfinite(seed).all(): raise ValueError('Invalid instance seed')
            dist=np.linalg.norm(centers-seed,axis=1); j=int(dist.argmin())
            if dist[j]>.06 or (len(dist)>1 and np.sort(dist)[1]-dist[j]<.008): raise ValueError('Ambiguous instance seed')
            target=j+1; bindings[call['instance_id']]=j
        for name,seed in call.get('instance_bindings_W',{}).items():
            p=np.asarray(seed,float)
            if p.shape!=(2,) or not np.isfinite(p).all(): raise ValueError('Invalid protected-instance seed')
            d=np.linalg.norm(centers-p,axis=1); j=int(d.argmin())
            if d[j]>.06 or (len(d)>1 and np.sort(d)[1]-d[j]<.008): raise ValueError('Protected identity cannot be acquired: '+name)
            if j in bindings.values() and name not in bindings: raise ValueError('Two identities assigned to same instance')
            bindings[name]=j
        for name in call.get('protected_instance_ids',[]):
            if name not in bindings: raise ValueError('Protected IDs need current instance_bindings_W seeds')
        goalmask=np.zeros((256,256),np.uint8) if tool else footprint(call,tracks[target-1]['mask'])
        mem={'signature':signature,'start_t':t,'last_t':t-1e-4,'last_index':-1,'tracks':tracks,'target':target,'goalmask':goalmask,'history':[],'baseline_centers':centers,'bindings':bindings,'best_error':1e6,'last_progress_t':t,'reached_count':0,'resets':0,'last_phase':-1,'T_base_W':T,'default_exit':E.copy(),'image_times':None,'image_repeat':0}
    else: mem['tracks']=update_tracks(mem['tracks'],fg)
    if np.linalg.norm(T-mem['T_base_W'])>.001: return unavailable('ABORTED','Workspace transform changed during call',mem)
    img_times=(float(o.get('img_t_third',t)),float(o.get('img_t_wrist',t))); mem['image_repeat']=mem['image_repeat']+1 if img_times==mem['image_times'] else 0; mem['image_times']=img_times
    if mem['image_repeat']>2: return unavailable('PERCEPTION_UNCERTAIN','Both camera frames repeated',mem)
    mem['last_t']=t; mem['last_index']=int(o['sample_index'])
    if t-mem['start_t']>float(call.get('timeout_s',30.)): return unavailable('ABORTED','Call timeout',mem)
    target=mem['target']; tracks=mem['tracks']; labels=scene_labels(tracks); conf=float(tracks[target-1]['confidence']) if target else float(np.mean([x['confidence'] for x in tracks]))
    if conf<.25: return unavailable('PERCEPTION_UNCERTAIN','Geometry not reliably visible; identity retained, not reassigned',mem,{'track_confidence':conf})
    goaltool=rigid(call['goal_tcp_base']) if tool else rigid(call.get('exit_tool_base',mem['default_exit']))
    last=np.asarray(call.get('last_executed_action',[0.]*6),float)
    if last.shape!=(6,) or not np.isfinite(last).all(): raise ValueError('Invalid actual previous native command')
    f=features(o,labels,target,mem['goalmask'],bool(target),conf,T,goaltool,terminal,last,sigma); mem['history'].append(f); mem['history']=mem['history'][-12:]
    changed=[]; unknown_neighbors=[]
    for j,tr in enumerate(tracks):
        if j+1==target: continue
        names=[name for name,v in mem['bindings'].items() if v==j]; name=names[0] if names else 'unbound_neighbor_%d'%j
        if tr['confidence']<.25: unknown_neighbors.append(name)
        elif np.linalg.norm(pix_to_w(center(tr['mask']))-mem['baseline_centers'][j])>float(call.get('neighbor_tolerance_m',.015))+2*sigma: changed.append(name)
    if changed: return unavailable('BLOCKED','Protected/nonselected instance moved; inspect',mem,{'changed_neighbors':changed,'uncertain_neighbors':unknown_neighbors})
    q=np.asarray(o['q'],float); dq=np.asarray(o['dq'],float); tau=np.asarray(o['tau_ext'],float)
    if q.shape!=(7,) or dq.shape!=(7,) or not np.isfinite(q).all() or not np.isfinite(dq).all(): return unavailable('PERCEPTION_UNCERTAIN','Invalid measured robot state',mem)
    guards=call.get('guards',{})
    if 'q_min' in guards and (np.any(q<np.asarray(guards['q_min'])+.02) or np.any(q>np.asarray(guards['q_max'])-.02)): return unavailable('BLOCKED','Joint-limit margin violated',mem)
    if 'tau_abs_max' in guards and (not np.isfinite(tau).all() or np.any(np.abs(tau)>np.asarray(guards['tau_abs_max']))): return unavailable('BLOCKED','Joint residual threshold exceeded; not contact force',mem)
    b=live_batch(mem['history'],spec)
    with torch.no_grad():
        out=model(b); a=model.reconstruct(out['mean'],b)[0].detach().cpu().numpy(); phase=int(out['phase_logits'][0].argmax().item()); uncertainty=(torch.exp(.5*out['logvar'][0])*model.scale).detach().cpu().numpy()
    if not np.isfinite(a).all(): return unavailable('ABORTED','Nonfinite neural proposal',mem)
    if phase==2 and mem['last_phase']!=2: mem['resets']+=1
    mem['last_phase']=phase
    toolerror=float(np.linalg.norm(goaltool[:3,3]-E[:3,3])); angleerror=float(np.linalg.norm(rotation_error(E[:3,:3],goaltool[:3,:3]))); error=toolerror if tool else f['contour_error']
    if error<mem['best_error']-.002: mem['best_error']=error; mem['last_progress_t']=t
    diag={'candidate_encoding':'recorded action.v then action.w; not enabled physical twist','geometric_error_m':error,'tool_waypoint_error_m':toolerror,'tool_angle_error_rad':angleerror,'contour_error_proxy_m':None if tool else f['contour_error'],'track_confidence':conf,'native_action_std_unvalidated':uncertainty.tolist(),'plane_sigma_design_margin_m':sigma,'plane_verified':planeverified,'view_disagreement':dis,'changed_neighbors':changed,'uncertain_neighbors':unknown_neighbors,'phase':['approach','contact','reset','exit','hold'][phase],'contact_proximity_proxy_m':f['proximity'],'pose_symmetry_ambiguous':bool(f['ambiguous']),'high_measured_joint_motion':bool(np.max(np.abs(dq))>.6),'joint_limits_checked':'q_min' in guards,'singularity_checked':False,'rod_geometry_verified':bool(call.get('rod_geometry',{}).get('verified',False)),'hardware_enabled':False,'requested_speed_cap_m_s':call.get('speed_cap_m_s',.03),'physical_speed_cap_enforced':False,'clearance_m_requested':call.get('clearance_m',.02),'clearance_verified':False,'tool_pose_base':E.tolist(),'retry_count':mem['resets'],'last_command_available':'last_executed_action' in call,'successor_readiness':{'physical_ready':False,'fresh_observation':True,'track_available':conf>=.25,'goal_preserved':True,'controller_verified':False,'safe_to_handoff':False}}
    if model.graph:
        weights=out['candidate_weights'][0].detach().cpu().numpy(); selected=int(np.argmax(weights)); diag['contact_candidate_index']=selected; diag['contact_mixture_weights']=weights.tolist(); diag['selected_boundary_point_W']=(f['nodes'][selected,:2]*.15+f['center_W']).tolist()
    diag['predicted_motion_proxy_unvalidated']=out['response'][0].detach().cpu().numpy().tolist()
    if t-mem['last_progress_t']>float(call.get('stall_s',8.)) or mem['resets']>int(call.get('retry_budget',8)): return unavailable('NEEDS_REPOSITION','Progress stalled or reset budget exceeded; preserve goal/identity',mem,diag)
    cap=np.asarray(call.get('native_abs_cap',model.scale.detach().cpu().numpy()*3.),float)
    if cap.shape!=(6,) or not np.isfinite(cap).all() or np.any(cap<=0): raise ValueError('native_abs_cap must be finite positive length six')
    a=np.clip(a,-cap,cap); projection=0.; mapping=call.get('intent_mapping')
    if mapping is not None and mapping.get('verified',False):
        M=np.asarray(mapping['native_to_base_twist'],float).reshape(6,6)
        if not np.isfinite(M).all() or not mapping.get('audit_reference') or np.linalg.cond(M)>1e6: raise ValueError('Intent mapping needs audit reference and finite invertible matrix')
        twist=M@a; speed=np.linalg.norm(twist[:3]); factor=min(1.,float(call.get('speed_cap_m_s',.03))/max(speed,1e-8)); twist[:3]*=factor
        p0=w_to_pix(base_to_w(E[:3,3],T)[:2]); p1=w_to_pix(base_to_w(E[:3,3]+twist[:3]*.15,T)[:2]); line=np.linspace(p0,p1,16).astype(int)
        if (line<0).any() or (line>=256).any(): return unavailable('BLOCKED','Planar proposal outside represented workspace',mem,diag)
        obstacles=((labels>0)&(labels!=target)).astype(np.uint8); radius=float(call.get('rod_geometry',{}).get('radius_m',.006))+sigma
        if not np.isfinite(radius) or radius<=0: raise ValueError('Invalid rod radius/margin')
        rad=max(1,min(20,int(np.ceil(radius*256)))); inflated=cv2.dilate(obstacles,np.ones((rad*2+1,rad*2+1),np.uint8))
        if np.any(inflated[line[:,1],line[:,0]]): return unavailable('BLOCKED','Planar protected-footprint sweep blocked; no certified lift path',mem,diag)
        safe=np.linalg.solve(M,twist); projection=float(np.linalg.norm(safe-a)); a=safe; diag['physical_speed_cap_enforced']=True
    diag['collision_projection_native_norm']=projection
    tol=max(float(call.get('tolerance_m',.01)),2*sigma); geom_ok=error<=tol and (not tool or angleerror<=float(call.get('tolerance_rad',.08))); semantic=call.get('semantic_orientation'); semantic_ok=True
    if semantic is not None:
        semantic_ok=False
        if 'observed_axis_W' in semantic and abs(float(semantic.get('observed_t',-1))-t)<.1:
            u=np.asarray(semantic['observed_axis_W'],float); v=np.asarray(semantic['goal_axis_W'],float)
            if not np.isfinite(u).all() or not np.isfinite(v).all() or np.linalg.norm(u)*np.linalg.norm(v)<1e-8: raise ValueError('Invalid semantic axis')
            se=float(np.arccos(np.clip(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)),-1,1))); diag['semantic_angle_error_rad']=se; semantic_ok=se<float(call.get('tolerance_rad',.08))
    diag['semantic_orientation_checked']=semantic is not None and semantic_ok; arrival=call.get('arrival_mode','stop')
    if arrival not in ['stop','pass_through']: raise ValueError('arrival_mode must be stop or pass_through')
    exit_ok=toolerror<=tol and angleerror<=float(call.get('tolerance_rad',.08)) if tool or 'exit_tool_base' in call else True
    stop_ok=arrival=='pass_through' or (call.get('last_command_verified',False) and 'last_executed_action' in call and np.linalg.norm(last)<1e-6 and np.max(np.abs(dq))<.03)
    mem['reached_count']=mem['reached_count']+1 if geom_ok and semantic_ok and exit_ok and stop_ok and not unknown_neighbors else 0
    diag['goal_satisfied_proxy']=bool(geom_ok); diag['exit_waypoint_satisfied']=bool(exit_ok)
    if not exit_ok and geom_ok and not tool: diag['phase']='exit'
    status='CONTROLLER_UNVERIFIED'
    if mem['reached_count']>=3 and planeverified:
        status='REACHED_LOCAL_GOAL'
        if arrival=='stop': a=np.zeros(6)
    if np.max(np.abs(dq))>.6: diag['agent_judgment_required']='Large measured reconfiguration: inspect robot margins; not evidence of corrupt label.'
    diag['successor_readiness']['achieved_local_geometry']=bool(mem['reached_count']>=3)
    return {'action':finite(a,(6,)).tolist(),'memory':mem,'status':status,'diagnostics':diag}
