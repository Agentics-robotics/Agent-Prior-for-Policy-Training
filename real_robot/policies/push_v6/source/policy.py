import math
import numpy as np
import torch
from torch import nn
from real_robot.training_pipeline import public
from geometry import make_sam
from preparation import prepare
from tracking import restore_causal_holes
from quality import filter_prepared,qualify_actor
import online


class ShapePush(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.point_encoder=nn.Sequential(nn.Linear(8,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
        self.scorer=nn.Sequential(nn.Linear(148,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
        self.sam=make_sam(device)

    def forward(self,points,candidates,feasible):
        h=self.point_encoder(points);context=torch.cat([h.mean(1),h.amax(1)],dim=-1);ctx=context[:,None,:].expand(-1,candidates.shape[1],-1)
        return self.scorer(torch.cat([ctx,candidates],dim=-1)).squeeze(-1).masked_fill(feasible<.5,-1.e4)


def prepare_data(spec):
    return filter_prepared(prepare(spec),spec)


def build_model(policy_id,spec):
    return ShapePush(spec['device'])


def make_batch(policy_id,arrays,metadata,indices,spec):
    keys=['point','candidate','feasible','action','target_contact','target_direction','target_length','target_validity']
    return {k:torch.as_tensor(arrays[k][indices],dtype=torch.float32,device=spec['device']) for k in keys}


def compute_loss(model,batch,spec):
    logits=model(batch['point'],batch['candidate'],batch['feasible']);action=batch['action'];masks=batch['target_validity']
    cp=((action[:,:,:2]-batch['target_contact'][:,None,:])**2).sum(-1)/(2*.007**2);dot=(action[:,:,2:4]*batch['target_direction'][:,None,:]).sum(-1).clamp(-1,1)
    dr=(1-dot)/.18;ln=((action[:,:,4]-batch['target_length'][:,None])/.006)**2/2;logp=torch.log_softmax(logits,dim=-1)
    target=torch.softmax((-(cp+dr+ln)).masked_fill(batch['feasible']<.5,-1.e4),dim=-1);full=-(target*logp).sum(-1)
    lp=logp.reshape(-1,96,3,3);valid=batch['feasible'].reshape(-1,96,3,3)>.5
    contact_target=torch.softmax((-cp.reshape(-1,96,3,3)[:,:,0,0]).masked_fill(~valid.any(-1).any(-1),-1.e4),dim=-1)
    contact_loss=-(contact_target*torch.logsumexp(lp.reshape(-1,96,9),dim=-1)).sum(-1);cd_cost=(cp+dr).reshape(-1,96,3,3)[:,:,:,0]
    cd_target=torch.softmax((-cd_cost).masked_fill(~valid.any(-1),-1.e4).reshape(-1,288),dim=-1);cd_loss=-(cd_target*torch.logsumexp(lp,dim=-1).reshape(-1,288)).sum(-1)
    loss=torch.where(masks[:,1]<.5,contact_loss,torch.where(masks[:,2]<.5,cd_loss,full)).mean();ix=logits.argmax(-1);chosen=action[torch.arange(len(ix),device=ix.device),ix]
    contact=torch.linalg.vector_norm(chosen[:,:2]-batch['target_contact'],dim=-1).mean();direction=((1-(chosen[:,2:4]*batch['target_direction']).sum(-1))*masks[:,1]).sum()/masks[:,1].sum().clamp_min(1)
    length=((chosen[:,4]-batch['target_length']).abs()*masks[:,2]).sum()/masks[:,2].sum().clamp_min(1)
    return {'loss':loss,'contact_error_m':contact,'direction_cosine_error':direction,'length_error_m':length}


def predict(model,batch,spec):
    ix=model(batch['point'],batch['candidate'],batch['feasible']).argmax(-1)
    return batch['action'][torch.arange(len(ix),device=ix.device),ix]


def configure_optimizer(model,spec):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0005,weight_decay=.002)


def sample_indices(policy_id,arrays,metadata,rng,step,spec):
    K=len(arrays['point']);a=(step-1)*16
    if a<K:
        return np.arange(a,min(a+16,K),dtype=np.int64)
    w=arrays['policy_weight'][:,0].astype(np.float64);w/=w.sum()
    return rng.choice(K,size=min(16,K),replace=True,p=w).astype(np.int64)


def before_update(optimizer,step,spec):
    maximum=spec['policy_config']['updates'];lr=.0005*min(1.,step/150)*(.15+.85*.5*(1+math.cos(math.pi*step/maximum)))
    for group in optimizer.param_groups:
        group['lr']=lr


def after_update(model,ema,arrays,metadata,step,metrics,spec):
    if step!=spec['policy_config']['updates']:
        return {'stop':False,'select':None,'metrics':{}}
    sums={};n=0
    for a in range(0,len(arrays['point']),16):
        ids=np.arange(a,min(a+16,len(arrays['point'])));met=compute_loss(model,make_batch('shape_push_v1',arrays,metadata,ids,spec),spec)
        for k,v in met.items():
            sums[k]=sums.get(k,0.)+float(v.item())*len(ids)
        n+=len(ids)
    result={k:v/max(1,n) for k,v in sums.items()};result['num_final_gradient_examples']=n;result['scope']='Final all-valid-data fitting diagnostics; NOT held-out or robot performance.'
    public.write_report('final_fit_diagnostics.json',result)
    return {'stop':True,'select':'model','metrics':result}


def array_shape(t):
    s=dict(t);s['xy']=np.array(t['xy']);s['normal']=np.array(t['normal']);s['loop']=np.array(t['loop']);s['diameter']=float(np.linalg.norm(np.ptp(s['xy'],axis=0)));return s


def act(model,observation,call,memory,spec):
    if not observation or not call or call.get('interrupted') or not observation.get('calibration_valid') or call.get('scene_revision')!=observation.get('revision'):
        return online.act(model,observation,call,memory,spec)
    if 'tracks' in observation:
        if not observation.get('geometry_complete_validated',False):
            return {'memory':{},'decision':None,'status':'needs_view'}
        if observation.get('causal_shape_references'):
            o=dict(observation);raw=[array_shape(t) for t in o['tracks']];refs=[array_shape(t) for t in o['causal_shape_references']];restored=restore_causal_holes(raw,refs)
            selected=[s for s in restored if s['instance_id']==call.get('instance_id')]
            if selected and not qualify_actor(selected[0],refs)[0]:
                return {'memory':{},'decision':None,'status':'needs_view'}
            o['tracks']=[{'instance_id':s['instance_id'],'xy':s['xy'].tolist(),'normal':s['normal'].tolist(),'loop':s['loop'].tolist()} for s in restored]
            return online.act(model,o,call,memory,spec)
    return online.act(model,observation,call,memory,spec)


def executor_request(decision,executor_contract,spec):
    return online.executor_request(decision,executor_contract,spec)


def calling_cases(arrays,metadata,spec):
    cases=online.calling_cases(arrays,metadata,spec)
    for c in cases:
        if 'tracks' in c['observation']:
            c['observation']['geometry_complete_validated']=True
    unvalidated=dict(cases[0]['observation']);unvalidated['geometry_complete_validated']=False
    cases.append({'name':'incomplete_scene_refused','observation':unvalidated,'call':cases[0]['call'],'reset':True,'executor_contract':cases[0]['executor_contract'],'expected_status':'needs_view','expect_decision':False})
    tid=spec['trajectory_ids'][0];cam=public.metadata(tid)['calibration']['third']
    observation={'rgb_third':public.rgb(tid,0,'third').tolist(),'camera':cam,'T_base_cam':cam['T_base_cam'],'top_z_base_m':metadata['label_provenance']['calibration']['top_z_base_m'],'calibration_valid':True,'calibration_id':'fixture_only_not_commissioned','uncertainty_m':.006,'age_s':.02,'revision':1}
    cases.append({'name':'real_rgb_generic_grid_proposals_fixture_calibration','observation':observation,'call':{'scene_revision':1},'reset':True,'executor_contract':{'commissioned':False},'expected_status':'associate_instances','expect_decision':False})
    return cases
