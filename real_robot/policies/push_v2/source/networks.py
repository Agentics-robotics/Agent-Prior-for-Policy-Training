"""Two independently initialized neural action distributions; no shared learned weights."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self,ni,nh,no):
        super().__init__();self.net=nn.Sequential(nn.Linear(ni,nh),nn.SiLU(),nn.Linear(nh,no))
    def forward(self,x): return self.net(x)

class MaskedMemory(nn.Module):
    def __init__(self,ni,width=384):
        super().__init__();self.first=nn.GRUCell(ni,width);self.second=nn.GRUCell(width,width);self.width=width
    def forward(self,x,mask):
        h=x.new_zeros((x.shape[0],self.width));g=torch.zeros_like(h)
        for t in range(x.shape[1]):
            m=mask[:,t,None];nh=self.first(x[:,t],h);ng=self.second(nh,g)
            h=nh*m+h*(1-m);g=ng*m+g*(1-m)
        return g

class GraphEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.project=MLP(20,192,192)
        self.messages=nn.ModuleList([MLP(384,256,192) for _ in range(3)])
        self.norms=nn.ModuleList([nn.LayerNorm(192) for _ in range(3)])
        self.affinity=MLP(192,96,1)
        self.readout=MLP(192*6,512,384)
    def forward(self,nodes,edges,mask):
        b,t,n,d=nodes.shape;raw=nodes.reshape(b*t,n,d);e=edges.reshape(b*t,n,2);m=mask.reshape(b*t,n)
        x=self.project(raw)
        batch=torch.arange(b*t,device=x.device)[:,None,None]
        for f,norm in zip(self.messages,self.norms):
            adjacent=x[batch,e].mean(2)
            x=norm(x+f(torch.cat([x,adjacent],-1)))*m[:,:,None]
        attention=self.affinity(x).squeeze(-1).masked_fill(m<.5,-1e4).softmax(-1)
        pools=[(x*attention[:,:,None]).sum(1)]
        # Explicit selected/goal/neighbor/workspace/tool readouts preserve identity roles.
        for i in range(5):
            w=raw[:,:,9+i]*m
            pools.append((x*w[:,:,None]).sum(1)/w.sum(1,keepdim=True).clamp_min(1))
        return self.readout(torch.cat(pools,-1)).reshape(b,t,384)

class ImageEncoder(nn.Module):
    def __init__(self,channels,out):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(channels,24,5,stride=2,padding=2),nn.GroupNorm(6,24),nn.SiLU(),
            nn.Conv2d(24,48,3,stride=2,padding=1),nn.GroupNorm(8,48),nn.SiLU(),
            nn.Conv2d(48,96,3,stride=2,padding=1),nn.GroupNorm(8,96),nn.SiLU(),
            nn.Conv2d(96,128,3,stride=2,padding=1),nn.GroupNorm(8,128),nn.SiLU(),
            nn.AdaptiveAvgPool2d((3,4)),nn.Flatten(),nn.Linear(128*12,out),nn.SiLU())
    def forward(self,x):return self.net(x)

class Policy(nn.Module):
    def __init__(self,policy_id,metadata):
        super().__init__();self.policy_id=policy_id
        self.register_buffer('state_mean',torch.tensor(metadata.get('state_mean',[0.]*142),dtype=torch.float32))
        self.register_buffer('state_std',torch.tensor(metadata.get('state_std',[1.]*142),dtype=torch.float32))
        self.register_buffer('action_mean',torch.tensor(metadata.get('action_mean',[0.]*7),dtype=torch.float32))
        self.register_buffer('action_std',torch.tensor(metadata.get('action_std',[1.]*7),dtype=torch.float32))
        self.state_encoder=MLP(142,256,128)
        if policy_id=='contour_push':
            self.graph_encoder=GraphEncoder();self.fusion=MLP(512,512,384)
        elif policy_id=='visual_push':
            self.global_encoder=ImageEncoder(4,192);self.detail_encoder=ImageEncoder(4,128)
            self.wrist_encoder=ImageEncoder(3,192);self.goal_encoder=ImageEncoder(4,128)
            self.fusion=MLP(768,512,384)
        else: raise ValueError('unknown policy')
        self.memory=MaskedMemory(384,384)
        self.distribution=MLP(384,384,5*(1+7+7))
        self.response=MLP(384,192,2)
    def encode(self,batch):
        state=(batch['state']-self.state_mean)/self.state_std.clamp_min(.001)
        s=self.state_encoder(state.clamp(-20,20));b,t,_=s.shape
        if self.policy_id=='contour_push':
            geometry=self.graph_encoder(batch['nodes'],batch['edges'],batch['node_mask'])
            x=self.fusion(torch.cat([geometry,s],-1))
        else:
            streams=[]
            for key,encoder in [('global',self.global_encoder),('detail',self.detail_encoder),('wrist',self.wrist_encoder)]:
                img=batch[key].float()/255.
                z=encoder(img.reshape(b*t,*img.shape[2:])).reshape(b,t,-1)
                streams.append(z)
            g=self.goal_encoder(batch['goal'].float()/255.)[:,None,:].expand(-1,t,-1)
            x=self.fusion(torch.cat(streams+[g,s],-1))
        return self.memory(x,batch['history_mask'])
    def forward(self,batch):
        z=self.encode(batch);p=self.distribution(z).reshape(-1,5,15)
        return p[:,:,0],p[:,:,1:8],p[:,:,8:15].clamp(-4.,1.5),self.response(z)

def build_model(policy_id,spec):
    return Policy(policy_id,spec.get('data_metadata',{}))

def augment(batch,policy_id):
    b=dict(batch)
    if policy_id=='visual_push':
        n=batch['state'].shape[0];device=batch['state'].device
        gain=.85+.3*torch.rand((n,1,3,1,1),device=device)
        bias=(torch.rand((n,1,3,1,1),device=device)-.5)*16
        for key in ('global','detail','wrist'):
            image=batch[key].float().clone()
            gg=gain if key!='wrist' else .85+.3*torch.rand_like(gain)
            image[:,:,:3]=(image[:,:,:3]*gg+bias).clamp(0,255)
            b[key]=image
        goal=batch['goal'].float().clone();goal[:,:3]=(goal[:,:3]*gain[:,0]+bias[:,0]).clamp(0,255)
        goal[:,:3]*=(goal[:,3:4]/255.);b['goal']=goal
    # Bounded causal history dropout: NEVER remove the latest frame or label a blind command.
    mask=batch['history_mask'].clone()
    if mask.shape[1]>1: mask[:,:-1]*=(torch.rand_like(mask[:,:-1])>.08)
    b['history_mask']=mask
    return b

def compute_loss(model,batch,spec):
    batch_aug=augment(batch,model.policy_id)
    logits,mu,ls,aux=model(batch_aug)
    y=(batch['action']-model.action_mean)/model.action_std
    log_density=-.5*(((y[:,None]-mu)/ls.exp())**2+2*ls+math.log(2*math.pi)).sum(-1)
    nll=-torch.logsumexp(logits.log_softmax(-1)+log_density,-1).mean()
    err=F.smooth_l1_loss(aux,batch['future']*100.,reduction='none').mean(-1)
    aux_loss=(err*batch['future_valid']).sum()/batch['future_valid'].sum().clamp_min(1.)
    loss=nll+.15*aux_loss
    return {'loss':loss,'nll':nll,'response_loss':aux_loss,'mean_component_scale':ls.exp().mean()}

def distribution_prediction(model,batch):
    logits,mu,ls,_=model(batch)
    # Choose a learned mode rather than averaging incompatible re-contact modes.
    index=logits.argmax(-1);row=torch.arange(mu.shape[0],device=mu.device)
    native=mu[row,index]*model.action_std+model.action_mean
    probability=logits.softmax(-1)
    mean=(probability[:,:,None]*mu).sum(1)
    var=(probability[:,:,None]*(ls.mul(2).exp()+(mu-mean[:,None])**2)).sum(1)
    std=var.sqrt()*model.action_std
    return native,std

def predict(model,batch,spec):
    return distribution_prediction(model,batch)[0]
