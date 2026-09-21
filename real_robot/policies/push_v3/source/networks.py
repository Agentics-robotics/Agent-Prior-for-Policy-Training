"""Independent constrained neural priors; no optimizer or hardware runner."""
import torch
from torch import nn
import torch.nn.functional as F

class Prior(nn.Module):
    def __init__(self, policy_id, scales):
        super().__init__()
        self.policy_id=policy_id
        self.graph=policy_id=='piece_contact_graph_v1'
        self.field=policy_id=='piece_goal_field_v1'
        self.register_buffer('scale',torch.tensor(scales,dtype=torch.float32))
        self.state_encoder=nn.Sequential(nn.Linear(112,192),nn.SiLU(),nn.Linear(192,192),nn.SiLU())
        self.map_encoder=nn.Sequential(nn.Conv2d(8,32,5,2,2),nn.GroupNorm(4,32),nn.SiLU(),nn.Conv2d(32,64,3,2,1),nn.GroupNorm(8,64),nn.SiLU(),nn.Conv2d(64,96,3,2,1),nn.GroupNorm(8,96),nn.SiLU(),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(1536,192),nn.SiLU())
        self.memory_cell=nn.GRUCell(384,192)
        self.core=nn.Sequential(nn.Linear(192,512),nn.SiLU(),nn.Linear(512,1024),nn.SiLU(),nn.Linear(1024,512),nn.SiLU())
        if self.graph:
            self.node_encoder=nn.Sequential(nn.Linear(16,96),nn.SiLU(),nn.Linear(96,96),nn.SiLU())
            self.message=nn.Sequential(nn.Linear(192,128),nn.SiLU(),nn.Linear(128,96),nn.SiLU())
            self.contact_score=nn.Sequential(nn.Linear(608,128),nn.SiLU(),nn.Linear(128,1))
            self.response_head=nn.Sequential(nn.Linear(614,128),nn.SiLU(),nn.Linear(128,6))
        else:
            self.response_head=nn.Sequential(nn.Linear(518,128),nn.SiLU(),nn.Linear(128,6))
        width=608 if self.graph else 512
        self.phase_head=nn.Linear(width,5)
        self.residual_head=nn.Linear(width,30)
        self.speed_head=nn.Linear(width,1)
        self.uncertainty_head=nn.Linear(width,6)
        nn.init.zeros_(self.residual_head.bias)

    def encode(self,b):
        B,H=b['state'].shape[:2]
        v=self.map_encoder(b['maps'].reshape(B*H,8,64,64)).reshape(B,H,192)
        s=self.state_encoder(b['state'])
        h=torch.zeros(B,192,device=s.device,dtype=s.dtype)
        for j in range(H):
            nh=self.memory_cell(torch.cat([s[:,j],v[:,j]],-1),h)
            mask=b['history_valid'][:,j:j+1]
            h=mask*nh+(1-mask)*h
        return self.core(h)

    def forward(self,b,supervised_action=None):
        context=self.encode(b); B=context.shape[0]; base=b['base']; info={}
        if self.graph:
            raw=b['nodes'][:,-1]; nodes=self.node_encoder(raw)
            dist=torch.cdist(raw[:,:,:2],raw[:,:,:2]); idx=dist.topk(4,largest=False).indices
            batch=torch.arange(B,device=nodes.device)[:,None,None]
            neighbor=nodes[batch,idx].mean(2)
            nodes=nodes+self.message(torch.cat([nodes,neighbor],-1))
            ctx=context[:,None,:].expand(-1,32,-1); joined=torch.cat([ctx,nodes],-1)
            det=b['score']; top=det.argmax(1)
            ref=raw[torch.arange(B,device=raw.device),top,2:4]
            ref=F.normalize(ref,dim=-1,eps=1e-5)
            near=raw[:,:,6:8]*.2; normal=raw[:,:,2:4]; flow=raw[:,:,4:6]
            tangent=torch.stack([-normal[:,:,1],normal[:,:,0]],-1)
            drive=normal*.65+tangent*(flow*tangent).sum(-1,keepdim=True).clamp(-.5,.5)
            drive=torch.where((near.norm(dim=-1,keepdim=True)>.018),-near/.06,drive)
            drive=drive/drive.norm(dim=-1,keepdim=True).clamp(min=1.)
            vx=(drive*ref[:,None,:]).sum(-1); vy=drive[:,:,1]*ref[:,None,0]-drive[:,:,0]*ref[:,None,1]
            prop=torch.zeros(B,32,6,device=raw.device); prop[:,:,0]=vx; prop[:,:,1]=vy
            prop[:,:,2]=base[:,None,2]
            planned=self.response_head(torch.cat([joined,prop],-1))
            lever=raw[:,:,:2]; torque=lever[:,:,0]*drive[:,:,1]-lever[:,:,1]*drive[:,:,0]
            mechanics=torch.stack([vx*.08,vy*.08,torque*.03],-1)
            response=mechanics+.25*torch.tanh(planned[:,:,:3])
            goalx=(flow*ref[:,None,:]).sum(-1); goaly=flow[:,:,1]*ref[:,None,0]-flow[:,:,0]*ref[:,None,1]
            improvement=response[:,:,0]*goalx+response[:,:,1]*goaly
            scores=det+self.contact_score(joined).squeeze(-1)+.2*torch.tanh(improvement)-.02*F.softplus(planned[:,:,3:]).mean(-1)
            scores=scores-8*(1-raw[:,:,10]); weights=torch.softmax(scores/0.5,-1)
            selected=(nodes*weights[:,:,None]).sum(1); joined_context=torch.cat([context,selected],-1)
            newbase=(prop*weights[:,:,None]).sum(1)
            exitflag=b['state'][:,-1,90:91]
            base=torch.where(((base[:,2:3].abs()>.5)|(exitflag>.5)),base,newbase)
            info['contact_logits']=scores; info['candidate_weights']=weights
            action_for_response=supervised_action if supervised_action is not None else newbase
            conditioned=self.response_head(torch.cat([joined,action_for_response[:,None,:].expand(-1,32,-1)],-1))
            ix=b.get('contact_index',scores.argmax(-1)); k=torch.arange(B,device=raw.device)
            info['response']=conditioned[k,ix,:3]; info['response_logvar']=conditioned[k,ix,3:].clamp(-5,3)
        else:
            joined_context=context
            action_for_response=supervised_action if supervised_action is not None else base
            motion=self.response_head(torch.cat([context,action_for_response],-1))
            info['response']=motion[:,:3]; info['response_logvar']=motion[:,3:].clamp(-5,3)
        phase=self.phase_head(joined_context); mixing=torch.softmax(phase,-1)
        residuals=2.5*torch.tanh(self.residual_head(joined_context).reshape(B,5,6))
        residuals=residuals*torch.tensor([1.,.8,1.,1.,.12],device=context.device)[None,:,None]
        residual=(residuals*mixing[:,:,None]).sum(1)
        speed=torch.sigmoid(self.speed_head(joined_context))*1.5
        mean=base*speed*(1-mixing[:,4:5])+residual
        info.update({'mean':mean,'logvar':self.uncertainty_head(joined_context).clamp(-5,2),'phase_logits':phase,'residual':residual})
        return info

    def local_target(self,b):
        u=b['native_action']; v=torch.bmm(b['basis'].transpose(1,2),u[:,:3,None]).squeeze(-1)
        return torch.cat([v,u[:,3:]],-1)/self.scale

    def reconstruct(self,local,b):
        scaled=local*self.scale; v=torch.bmm(b['basis'],scaled[:,:3,None]).squeeze(-1)
        return torch.cat([v,scaled[:,3:]],-1)

def loss(model,b):
    target=model.local_target(b); out=model(b,target); err=out['mean']-target
    bc=F.smooth_l1_loss(out['mean'],target,beta=.25)
    nll=(.5*err.square()*torch.exp(-out['logvar'])+.5*out['logvar']).mean()
    mode=F.cross_entropy(out['phase_logits'],b['phase'])
    response_err=out['response']-b['response']
    motion=(F.smooth_l1_loss(out['response'],b['response'],reduction='none')+.03*(response_err.square()*torch.exp(-out['response_logvar'])+out['response_logvar'])).mean(-1)
    mask=b['response_valid']; response_loss=(motion*mask).sum()/mask.sum().clamp(min=1)
    total=bc+.03*nll+.12*mode+.15*response_loss+.002*out['residual'].square().mean()
    if model.graph:
        contact=F.cross_entropy(out['contact_logits'],b['contact_index'],reduction='none')
        contact=(contact*b['geometry_valid']).sum()/b['geometry_valid'].sum().clamp(min=1)
        total=total+.07*contact
    else: contact=total*0
    if model.policy_id=='tool_waypoint_v1':
        aux_target=target[:,:3]-b['base'][:,:3]
        aux_err=out['response']-aux_target
        # Train both auxiliary mean and variance even without object successors.
        aux=F.smooth_l1_loss(out['response'],aux_target)+.03*(aux_err.square()*torch.exp(-out['response_logvar'])+out['response_logvar']).mean()
        total=total+.02*aux
    return {'loss':total,'bc':bc,'mode':mode,'response':response_loss,'contact':contact,'nll':nll}
