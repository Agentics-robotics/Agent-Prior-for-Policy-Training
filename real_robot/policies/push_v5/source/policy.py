import math
import numpy as np
import torch
from torch import nn
import cv2
from real_robot.training_pipeline import public
from flow_preparation import prepare
from flow_geometry import encode, support_candidates, track_pair
from geometry import camera, to_plane
from adapter import initialize_contact, executor_objective


class Continuation(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(4,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(75,64),nn.SiLU(),nn.Linear(64,48),nn.SiLU(),nn.Linear(48,9))
        self.best=1e30
        self.best_step=0

    def forward(self,actor,state):
        h=self.encoder(actor)
        out=self.head(torch.cat([h.mean(1),h.max(1).values,state],1)).reshape(-1,3,3)
        means=torch.tanh(out[:,:,1:])*.9
        return out[:,:,0],means


def prepare_data(spec):
    result=prepare(spec)
    # A real-source callable fixture, with hindsight goal on the call side only.
    arrays=result['arrays'];meta=result['metadata']
    if len(arrays['example_source_index']):
        k=0;ep=int(arrays['example_episode'][k]);tid=spec['trajectory_ids'][ep]
        a=int(arrays['example_source_index'][k]);e=int(arrays['target_stop'][k])-1
        rec=public.records(tid);cal=public.metadata(tid)['calibration'];cam=camera(cal,'third')
        rgb0=public.rgb(tid,a,'third');rgb1=public.rgb(tid,e,'third')
        choices=[]
        for c in support_candidates(rgb0,cam,np.array([0.,0.,.068])):
            prox=float(np.min(np.linalg.norm(c['occupied']-rec['T_base_ee'][a,:2,3],axis=1)))
            f=track_pair(rgb0,rgb1,c['uv'],cam,np.array([0.,0.,.068]))
            if f is not None and f['motion']>.004 and prox<.04:
                choices.append((prox,c,f))
        if choices:
            choices.sort(key=lambda x:x[0]);_,c,f=choices[0]
            meta['calling_fixture']={'support':c['occupied'].tolist(),'T_base_ee':rec['T_base_ee'][a].tolist(),'R':f['R'].tolist(),'t':f['t'].tolist(),'previous_delta':(rec['T_base_ee'][a,:2,3]-rec['T_base_ee'][a-6,:2,3]).tolist(),'source':{'episode':ep,'a':a,'e':e}}
    return result


def build_model(policy_id,spec):
    return Continuation()


def make_batch(policy_id,arrays,metadata,indices,spec):
    return {k:torch.as_tensor(arrays[k][indices],device=spec['device'],dtype=torch.float32) for k in ['actor','state','target']}


def distribution(model,batch):
    return model(batch['actor'],batch['state'])


def chosen_delta(logits,means):
    ix=logits.argmax(1)
    return means[torch.arange(len(ix),device=ix.device),ix]


def compute_loss(model,batch,spec):
    actor=batch['actor'];state=batch['state']
    if model.training:
        actor=actor+torch.randn_like(actor)*.012
        state=state.clone()
        keep=(torch.rand((len(state),1),device=state.device)>.25).float()
        state[:,6:9]=state[:,6:9]*keep
    logits,means=model(actor,state)
    err=((batch['target'][:,None,:]-means)/.15).square().sum(-1)
    # Heavy-tailed mixture likelihood handles multimodal continuation, not
    # unchosen-action failure labels. Scale .15 = 3mm measured EE displacement.
    loss=-torch.logsumexp(torch.log_softmax(logits,1)-3*torch.log1p(err/4),1).mean()
    pred=chosen_delta(logits,means)
    error_mm=torch.linalg.vector_norm(pred-batch['target'],dim=1).mean()*20
    baseline=batch['actor'][:,:,2:4].mean(1)*2.5
    baseline=baseline/torch.linalg.vector_norm(baseline,dim=1,keepdim=True).clamp_min(.9)*.9
    return {'loss':loss,'EE_endpoint_error_mm':error_mm,'goal_translation_baseline_mm':torch.linalg.vector_norm(baseline-batch['target'],dim=1).mean()*20,'zero_baseline_mm':torch.linalg.vector_norm(batch['target'],dim=1).mean()*20}


def predict(model,batch,spec):
    logits,means=distribution(model,batch)
    return torch.cat([torch.softmax(logits,1)[:,:,None],means],2).reshape(-1,9)


def configure_optimizer(model,spec):
    return torch.optim.AdamW(model.parameters(),lr=.0007,weight_decay=.005)


def sample_indices(policy_id,arrays,metadata,rng,step,spec):
    eligible=np.where((arrays['example_episode']==0)&(arrays['policy_weight'][:,0]>0))[0]
    if len(eligible)==0:
        raise ValueError('No valid A supervision')
    groups=np.unique(arrays['group'][eligible]);out=[]
    for j in range(spec['policy_config']['batch_size']):
        g=rng.choice(groups);pool=eligible[arrays['group'][eligible]==g]
        out.append(rng.choice(pool))
    return np.asarray(out,np.int64)


def before_update(optimizer,step,spec):
    lr=.0007*min(1.,step/30.)*(.15+.85*.5*(1+math.cos(math.pi*min(1.,step/spec['policy_config']['updates']))))
    for group in optimizer.param_groups:
        group['lr']=lr


def after_update(model,ema,arrays,metadata,step,metrics,spec):
    if step!=1 and step%50!=0 and step!=spec['policy_config']['updates']:
        return {'stop':False,'select':None,'metrics':{}}
    report={};scores=[]
    for ep,label in [(0,'train_A'),(1,'development_B')]:
        ids=np.where(arrays['example_episode']==ep)[0];group_scores=[]
        for g in np.unique(arrays['group'][ids]):
            take=ids[arrays['group'][ids]==g];totals={};count=0
            for j in range(0,len(take),64):
                batch=make_batch(spec['policy_id'],arrays,metadata,take[j:j+64],spec)
                m=compute_loss(model,batch,spec);n=len(take[j:j+64]);count+=n
                for key,value in m.items():
                    totals[key]=totals.get(key,0.)+float(value.item())*n
            vals={key:v/count for key,v in totals.items()};report[label+'_group_'+str(int(g))]=vals
            group_scores.append(vals['EE_endpoint_error_mm'])
        score=float(np.mean(group_scores)) if group_scores else 1e6
        report[label+'_macro_EE_error_mm']=score;scores.append(score)
    score=scores[1] if (arrays['example_episode']==1).any() else scores[0]
    select=None
    if score<model.best-.01:
        model.best=score;model.best_step=step;select='model'
    report.update({'best_step':model.best_step,'selection':'macro B measured EE endpoint error; NOT physical success','untouched_test':False})
    return {'stop':step>=400 and step-model.best_step>=300,'select':select,'metrics':report}


def inventory(observation,spec):
    # Class-agnostic online converter. Never uses recorded colors or templates.
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    if any(k not in observation for k in ['rgb_third','camera','top_plane','scene_revision']):
        return {'status':'missing_input','instances':[]}
    net=build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',public.asset_path('sam2.1_hiera_small.pt'),device=spec['device'],apply_postprocessing=False)
    gen=SAM2AutomaticMaskGenerator(net,points_per_side=16,pred_iou_thresh=.85,stability_score_thresh=.92)
    rgb=np.asarray(observation['rgb_third'],np.uint8);proposals=gen.generate(rgb)
    c=observation['camera'];cam=(np.asarray(c['K']),np.asarray(c['dist']),np.asarray(c['T_base_cam']))
    plane=np.asarray(observation['top_plane']);out=[]
    for m in proposals:
        mask=m['segmentation'].astype(np.uint8)
        if not 250<mask.sum()<85000:
            continue
        yy,xx=np.where(mask);take=np.linspace(0,len(xx)-1,min(256,len(xx))).astype(int)
        support=to_plane(np.c_[xx[take],yy[take]],cam,plane)
        cs,hi=cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE);loops=[]
        if hi is not None:
            for j,p in enumerate(cs):
                if cv2.contourArea(p)>80:
                    loops.append({'xy':to_plane(p[:,0],cam,plane).tolist(),'hole':bool(hi[0,j,3]>=0)})
        out.append({'instance_id':str(observation['scene_revision'])+':'+str(len(out)),'support_points_P':support.tolist(),'contours':loops,'geometry_complete':False,'identity':'unknown','bbox_rgb':m['bbox']})
    return {'status':'proposals_require_association_and_geometry_validation','instances':out,'frame':'P: base-parallel XY, z=top_plane','scene_revision':observation['scene_revision'],'semantic_owner':'High-level visual Agent','not_a_contact_certificate':True}


def act(model,observation,call,memory,spec):
    mem={} if memory is None else dict(memory)
    def result(status,decision=None):
        return {'status':status,'decision':decision,'memory':mem}
    if call.get('interrupted',False):
        mem={};return result('interrupted')
    required=['scene_revision','time_s','calibration_id','instances','T_base_ee','T_base_P','uncertainty_m']
    if any(k not in observation for k in required) or any(k not in call for k in ['instance_id','goal_R','goal_t','scene_revision','now_s']):
        mem={};return result('missing_input')
    if call['scene_revision']!=observation['scene_revision'] or not 0<=call['now_s']-observation['time_s']<=.25:
        mem={};return result('stale_scene')
    key=[call['instance_id'],call['goal_R'],call['goal_t'],observation['calibration_id']]
    if mem.get('key')!=key:
        mem={'key':key,'failures':0}
    if call.get('previous_outcome') in ['guard_stop','unexpected_neighbor_motion','contact_lost']:
        mem={};return result('executor_replan')
    if call.get('previous_outcome')=='no_progress' and mem.get('last_failure_ticket')!=call.get('previous_ticket'):
        mem['failures']=mem.get('failures',0)+1;mem['last_failure_ticket']=call.get('previous_ticket')
    if mem.get('failures',0)>=2:
        return result('executor_replan')
    tracks=[x for x in observation['instances'] if x['instance_id']==call['instance_id']]
    if len(tracks)!=1:
        mem={};return result('ambiguous_instance')
    actor=tracks[0]
    try:
        R=np.asarray(call['goal_R'],float);t=np.asarray(call['goal_t'],float)
        if R.shape!=(2,2) or t.shape!=(2,) or not np.isfinite(R).all() or not np.isfinite(t).all() or np.linalg.norm(R.T@R-np.eye(2))>.001 or np.linalg.det(R)<.999:
            return result('invalid_goal')
        unc=float(observation['uncertainty_m'])
        if not np.isfinite(unc) or not 0<=unc<=.012 or not observation.get('chart_validated',False):
            return result('needs_calibration_or_view')
        p=np.asarray(actor['support_points_P'],float)
        if p.ndim!=2 or p.shape[1]!=2 or len(p)<5 or not np.isfinite(p).all():
            return result('needs_view')
        residual=np.max(np.linalg.norm(p@R.T+t-p,axis=1))
        if residual<call.get('position_tolerance_m',.003):
            return result('at_goal_observation_not_task_success')
        if residual>.08:
            return result('needs_local_goal')
        common={'instance_id':actor['instance_id'],'scene_revision':observation['scene_revision'],'calibration_id':observation['calibration_id'],'frame':'P','units':'m','uncertainty_m':unc,'max_duration_s':1.}
        max_stroke=min(.015,float(call.get('max_stroke_m',.015)))
        if max_stroke<=0:
            return result('invalid_limits')
        if not call.get('contact_confirmed',False):
            if not actor.get('geometry_complete',False) or not observation.get('geometry_validated',False) or any(k not in observation for k in ['tool_radius_m','workspace']):
                return result('needs_full_geometry_for_initialization')
            init=initialize_contact(actor,observation['instances'],observation['workspace'],R,t,float(observation['tool_radius_m']),unc,max_stroke)
            if init is None:
                return result('no_safe_candidate')
            return result('proposed_prepare',dict(common,mode='prepare',**init))
        B=np.asarray(observation['T_base_P'],float);E=np.asarray(observation['T_base_ee'],float)
        if B.shape!=(4,4) or E.shape!=(4,4) or not np.isfinite(B).all() or not np.isfinite(E).all() or np.linalg.norm(B[:3,:3].T@B[:3,:3]-np.eye(3))>.001:
            return result('invalid_geometry')
        ee=B[:3,:3].T@(E[:3,3]-B[:3,3]);ee[2]+=.068
        axis=B[:3,:3].T@E[:3,2]
        prev=observation.get('previous_delta_P_m') if .15<=observation.get('history_dt_s',0.)<=.25 else None
        actor_tensor,state,F=encode(p,R,t,ee,axis,prev,unc)
        with torch.no_grad():
            logits,means=model(torch.tensor(actor_tensor[None],device=spec['device']),torch.tensor(state[None],device=spec['device']))
            delta=chosen_delta(logits,means)[0].cpu().numpy()@F.T*.02
        delta*=min(1.,max_stroke/max(float(np.linalg.norm(delta)),1e-8))
        return result('proposed_continuation',dict(common,mode='continue',delta_P_m=delta.tolist(),T_base_ee=E.tolist(),requires_current_contact_revalidation=True))
    except (KeyError,ValueError,TypeError,IndexError):
        return result('invalid_geometry')


def executor_request(decision,executor_contract,spec):
    return executor_objective(decision,executor_contract)


def calling_cases(arrays,metadata,spec):
    p=np.array([[x,y] for x in np.linspace(.40,.48,8) for y in np.linspace(-.04,.04,8)])
    B=np.eye(4);B[2,3]=.068;E=np.eye(4);E[:3,3]=[.395,0.,.058];E[:3,:3]=np.diag([1.,-1.,-1.])
    actor={'instance_id':'x','support_points_P':p.tolist(),'geometry_complete':True,'contours':[{'xy':[[.4,-.04],[.48,-.04],[.48,.04],[.4,.04]],'hole':False}]}
    obs={'scene_revision':7,'time_s':10.,'calibration_id':'synthetic','instances':[actor],'T_base_ee':E.tolist(),'T_base_P':B.tolist(),'uncertainty_m':.002,'chart_validated':True,'geometry_validated':True,'tool_radius_m':.004,'workspace':[[.2,-.3],[.8,-.3],[.8,.3],[.2,.3]]}
    call={'instance_id':'x','goal_R':np.eye(2).tolist(),'goal_t':[.012,0.],'scene_revision':7,'now_s':10.01,'contact_confirmed':True}
    ex={'commissioned':True,'scene_revision':7,'calibration_id':'synthetic','T_base_P':B.tolist(),'guard_profile_id':'synthetic_only','tool_geometry_id':'synthetic','max_speed_m_s':.01,'R_P_tool':np.eye(3).tolist(),'support_point_tool_m':[0.,0.,0.],'contact_height_P_m':-.005}
    cases=[{'name':'synthetic_continuation','observation':obs,'call':call,'reset':True,'executor_contract':ex,'expected_status':'proposed_continuation','expect_decision':True},{'name':'deterministic_contact_initializer','observation':obs,'call':dict(call,contact_confirmed=False),'reset':True,'executor_contract':ex,'expected_status':'proposed_prepare','expect_decision':True},{'name':'missing_observation','observation':{},'call':call,'reset':True,'executor_contract':{},'expected_status':'missing_input','expect_decision':False},{'name':'stale_scene','observation':obs,'call':dict(call,scene_revision=6),'reset':False,'executor_contract':ex,'expected_status':'stale_scene','expect_decision':False},{'name':'interruption_reset','observation':obs,'call':dict(call,interrupted=True),'reset':False,'executor_contract':ex,'expected_status':'interrupted','expect_decision':False},{'name':'calibration_required','observation':dict(obs,chart_validated=False),'call':call,'reset':True,'executor_contract':{},'expected_status':'needs_calibration_or_view','expect_decision':False},{'name':'observed_goal_not_success','observation':obs,'call':dict(call,goal_t=[0.,0.]),'reset':True,'executor_contract':ex,'expected_status':'at_goal_observation_not_task_success','expect_decision':False},{'name':'guard_feedback','observation':obs,'call':dict(call,previous_outcome='guard_stop'),'reset':False,'executor_contract':ex,'expected_status':'executor_replan','expect_decision':False}]
    if 'calling_fixture' in metadata:
        f=metadata['calling_fixture'];real=dict(obs,T_base_ee=f['T_base_ee'],instances=[dict(actor,support_points_P=f['support'],geometry_complete=False)],previous_delta_P_m=f['previous_delta'],history_dt_s=.2,uncertainty_m=.008)
        cases.append({'name':'real_source_geometry_hindsight_goal_'+str(f['source']['a']),'observation':real,'call':dict(call,goal_R=f['R'],goal_t=f['t']),'reset':True,'executor_contract':dict(ex,commissioned=False),'expected_status':'proposed_continuation','expect_decision':True})
    return cases
