"""Independent neural conditional distributions; no scripted motor paths."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def mlp(a,b,c):
    return nn.Sequential(nn.Linear(a,b),nn.SiLU(),nn.Linear(b,c))


class MixturePolicy(nn.Module):
    def __init__(self, metadata):
        super().__init__()
        self.register_buffer('state_mean',torch.tensor(metadata['state_mean'],dtype=torch.float32))
        self.register_buffer('state_std',torch.tensor(metadata['state_std'],dtype=torch.float32))
        self.register_buffer('action_mean',torch.tensor(metadata['action_mean'],dtype=torch.float32))
        self.register_buffer('action_std',torch.tensor(metadata['action_std'],dtype=torch.float32))
        self.head=nn.Sequential(nn.Linear(512,512),nn.SiLU(),nn.Linear(512,5*15))
        self.aux_head=mlp(512,256,2)

    def normalize_state(self,x):
        return ((x-self.state_mean)/self.state_std).clamp(-12,12)

    def distribution(self,z):
        y=self.head(z).reshape(-1,5,15)
        return {'logits':y[:,:,0],'mean':y[:,:,1:8],
                'log_scale':y[:,:,8:15].clamp(-4.,2.),'aux':self.aux_head(z)}

    def decode_mean(self,out):
        k=out['logits'].argmax(-1)
        mu=out['mean'][torch.arange(len(k),device=k.device),k]
        return mu*self.action_std+self.action_mean


class BoundaryLayer(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.message=mlp(3*width,2*width,width)
        self.norm=nn.LayerNorm(width)
    def forward(self,h,edges,mask):
        b,n,c=h.shape
        i=edges[:,:,0:1].expand(b,n,c); j=edges[:,:,1:2].expand(b,n,c)
        prev=torch.gather(h,1,i); nxt=torch.gather(h,1,j)
        out=self.norm(h+self.message(torch.cat([h,prev-h,nxt-h],-1)))
        return F.silu(out)*mask[:,:,None]


class ContourPolicy(MixturePolicy):
    def __init__(self,metadata):
        super().__init__(metadata)
        self.input=mlp(19,256,192)
        self.layers=nn.ModuleList([BoundaryLayer(192) for _ in range(3)])
        self.affinity=mlp(192,128,1)
        self.global_state=mlp(metadata['state_dim'],256,256)
        self.fuse=mlp(192*2+256,512,512)
        self.temporal=nn.GRUCell(512,512)

    def forward(self,batch,augment=False):
        x=batch['nodes']; mask=batch['node_mask']; edges=batch['edges']
        b,t,n,c=x.shape
        if augment:
            # Bounded chart-coordinate extraction noise, not a rotated physical demonstration.
            noise=torch.randn((b,t,n,2),device=x.device)*.0008
            x=x.clone(); x[:,:,:,:2]+=noise; x[:,:,:,2:4]+=noise; x[:,:,:,15:17]+=noise
        xx=x.reshape(b*t,n,c); mm=mask.reshape(b*t,n); ee=edges.reshape(b*t,n,2)
        h=self.input(xx)*mm[:,:,None]
        for layer in self.layers: h=layer(h,ee,mm)
        # Learned soft contact-neighborhood affinity; no fabricated contact labels.
        score=self.affinity(h).squeeze(-1)
        dist=xx[:,:,15:17].square().sum(-1).sqrt()
        current=xx[:,:,10]
        score=score-2.*dist+current
        score=score.masked_fill(mm<.5,-1e4)
        weight=torch.softmax(score,-1)*mm
        weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-6)
        local=(h*weight[:,:,None]).sum(1)
        global_h=(h*mm[:,:,None]).sum(1)/mm.sum(1,keepdim=True).clamp_min(1.)
        state=self.global_state(self.normalize_state(batch['state'])).reshape(b*t,-1)
        tokens=self.fuse(torch.cat([local,global_h,state],-1)).reshape(b,t,512)
        hidden=torch.zeros((b,512),device=x.device,dtype=x.dtype)
        hm=batch['history_mask'].clone()
        if augment and t>1:
            hm[:,:-1]*=(torch.rand_like(hm[:,:-1])>.08).float()
        for j in range(t):
            update=self.temporal(tokens[:,j],hidden)
            hidden=torch.where(hm[:,j,None]>.5,update,hidden)
        return self.distribution(hidden)


class ImageEncoder(nn.Module):
    def __init__(self,channels):
        super().__init__()
        widths=[32,64,128,192]
        layers=[]; prev=channels
        for width in widths:
            layers.extend([nn.Conv2d(prev,width,3,stride=2,padding=1),nn.GroupNorm(8,width),nn.SiLU(),
                           nn.Conv2d(width,width,3,padding=1),nn.GroupNorm(8,width),nn.SiLU()])
            prev=width
        self.conv=nn.Sequential(*layers)
        self.readout=nn.Sequential(nn.Flatten(),nn.Linear(192*3*4,384),nn.SiLU())
    def forward(self,x):
        return self.readout(F.adaptive_avg_pool2d(self.conv(x),(3,4)))


class VisualPolicy(MixturePolicy):
    def __init__(self,metadata):
        super().__init__(metadata)
        # Weights shared across third-view scales, NOT across policies.
        self.third_encoder=ImageEncoder(4)
        self.wrist_encoder=ImageEncoder(3)
        self.goal_encoder=ImageEncoder(4)
        self.state_encoder=mlp(metadata['state_dim'],256,256)
        self.fuse=nn.Sequential(nn.Linear(384*4+256+2,768),nn.SiLU(),nn.Linear(768,512),nn.SiLU())
        self.temporal=nn.GRUCell(512,512)

    def forward(self,batch,augment=False):
        third=batch['third']; marker=batch['marker']; local=batch['local']; wrist=batch['wrist']; goal=batch['goal']
        b,t=third.shape[:2]
        viewmask=torch.ones((b,t,2),device=third.device)
        if augment:
            # Consistent third/reference photometry, separate wrist exposure. No geometric warps.
            gain=.85+.3*torch.rand((b,1,3,1,1),device=third.device)
            bias=(torch.rand((b,1,3,1,1),device=third.device)-.5)*.08
            third=(third*gain+bias).clamp(0,1)
            local=torch.cat([(local[:,:,:3]*gain+bias).clamp(0,1),local[:,:,3:]],2)
            goal=torch.cat([(goal[:,:3]*gain[:,0]+bias[:,0]*goal[:,3:]).clamp(0,1),goal[:,3:]],1)
            wrist=(wrist*(.85+.3*torch.rand((b,1,3,1,1),device=wrist.device))).clamp(0,1)
            # Only past observations can be dropped; current identity/views always remain present.
            if t>1:
                drop=(torch.rand((b,t-1),device=wrist.device)<.08)
                viewmask[:,:-1,1]=(~drop).float()
                wrist=wrist*viewmask[:,:,1,None,None,None]
        th=torch.cat([third,marker],2).reshape(b*t,4,third.shape[-2],third.shape[-1])
        lo=local.reshape(b*t,4,local.shape[-2],local.shape[-1])
        wi=wrist.reshape(b*t,3,wrist.shape[-2],wrist.shape[-1])
        eg=self.goal_encoder(goal)[:,None].expand(-1,t,-1)
        et=self.third_encoder(th).reshape(b,t,-1)
        el=self.third_encoder(lo).reshape(b,t,-1)
        ew=self.wrist_encoder(wi).reshape(b,t,-1)
        state=self.state_encoder(self.normalize_state(batch['state']))
        tokens=self.fuse(torch.cat([et,el,ew,eg,state,viewmask],-1))
        hidden=torch.zeros((b,512),device=third.device)
        hm=batch['history_mask'].clone()
        if augment and t>1: hm[:,:-1]*=(torch.rand_like(hm[:,:-1])>.06).float()
        for j in range(t):
            candidate=self.temporal(tokens[:,j],hidden)
            hidden=torch.where(hm[:,j,None]>.5,candidate,hidden)
        return self.distribution(hidden)


def loss(model,batch):
    out=model(batch,augment=True)
    target=(batch['action']-model.action_mean)/model.action_std
    error=(target[:,None]-out['mean'])*torch.exp(-out['log_scale'])
    log_prob=-.5*(error.square()+math.log(2*math.pi)).sum(-1)-out['log_scale'].sum(-1)
    nll=-torch.logsumexp(F.log_softmax(out['logits'],-1)+log_prob,-1).mean()
    prob=F.softmax(out['logits'],-1)
    avg=(out['mean']*prob[:,:,None]).sum(1)
    huber=F.smooth_l1_loss(avg,target)
    # Auxiliary is displacement in chart dimensions, rescaled to useful gradient magnitude.
    aux_each=F.smooth_l1_loss(out['aux'],batch['aux']*100.,reduction='none').mean(-1)
    aux=(aux_each*batch['aux_valid']).sum()/batch['aux_valid'].sum().clamp_min(1.)
    total=nll+.05*huber+.2*aux
    native=model.decode_mean(out)
    return {'loss':total,'nll':nll,'huber':huber,'displacement_loss':aux,
            'native_mae':(native-batch['action']).abs().mean(),
            'mixture_entropy':(-(prob*F.log_softmax(out['logits'],-1)).sum(-1)).mean()}
