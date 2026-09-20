import math
import torch
from torch import nn
from appl.public import DiffusionBackbone, epsilon_loss

F = torch.nn.functional


def rotation_matrix(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3,3))


def rot6(q):
    r = rotation_matrix(q)
    return torch.cat((r[..., :, 0], r[..., :, 1]), -1)


def drawer_center(s):
    return torch.stack((0.19-s[...,39], torch.zeros_like(s[...,39]),
                        torch.full_like(s[...,39], 0.035)), -1)


class VectorDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('frequencies', torch.exp(-math.log(10000)*torch.arange(32).float()/31))
        self.input = nn.Linear(35+256+11+65, 256)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(256), nn.Linear(256,256),
                                  nn.SiLU(), nn.Linear(256,256)) for _ in range(3)])
        self.output = nn.Sequential(nn.LayerNorm(256), nn.SiLU(), nn.Linear(256,35))
        nn.init.normal_(self.output[-1].weight, std=0.01)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, x, alpha, context, semantics):
        a = alpha.reshape(-1,1).expand(x.shape[0],1).clamp(1e-5,1-1e-5)
        logsnr = (a.log()-(1-a).log()).clamp(-12,12)
        angle = logsnr*self.frequencies
        time = torch.cat((angle.sin(),angle.cos(),logsnr/12), -1)
        h = F.silu(self.input(torch.cat((x,context,semantics,time), -1)))
        for block in self.blocks:
            h = h + block(h)/math.sqrt(2)
        return self.output(h)


class WaypointPolicy(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(n['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(n['std'], dtype=torch.float32))
        scales = torch.tensor(n['std'], dtype=torch.float32)
        self.register_buffer('xyz_scale', torch.stack((scales[18:21],scales[25:28],scales[32:35])).amax(0))
        self.encoder = nn.Sequential(nn.Linear(213,256),nn.LayerNorm(256),nn.SiLU(),
                                     nn.Linear(256,256),nn.LayerNorm(256),nn.SiLU())
        self.event_head = nn.Linear(256,7)
        self.actor_head = nn.Linear(256,3)
        self.censor_head = nn.Linear(256,1)
        self.waypoint_diffusion = VectorDenoiser()
        self.local_state = nn.Sequential(nn.Linear(256,128),nn.SiLU())
        self.intent_embed = nn.Sequential(nn.Linear(46,256),nn.LayerNorm(256),nn.SiLU(),nn.Linear(256,256),nn.SiLU())
        self.action_diffusion = DiffusionBackbone(384, spec['training'])
        self.register_buffer('waypoint_weights', torch.tensor([1.5]*6+[0.25]*12+[0.5]*7+[1.0]*3+[0.5]*7))
        # These are auxiliary waypoint noise levels, NOT the fixed action DDPM schedule.
        self.plan_alphas = (0.05,0.20,0.50,0.80,0.95,1.0)

    def frame(self, raw):
        s = (raw-self.obs_mean)/self.obs_scale
        rel = torch.cat((raw[...,18:21]-raw[...,25:28], raw[...,18:21]-raw[...,32:35],
                         raw[...,25:28]-raw[...,41:44], raw[...,32:35]-raw[...,44:47],
                         raw[...,18:21]-raw[...,41:44], raw[...,18:21]-drawer_center(raw)), -1)
        rel = rel / self.xyz_scale.repeat(6)
        return torch.cat((s[...,:21],rot6(raw[...,21:25]),s[...,25:28],rot6(raw[...,28:32]),
                          s[...,32:35],rot6(raw[...,35:39]),s[...,39:47],rel),-1)

    def encode(self, history):
        f = self.frame(history)
        return self.encoder(torch.cat((f[:,0],f[:,1],4*(f[:,1]-f[:,0])), -1))

    def semantic_predictions(self, context):
        event = self.event_head(context)
        actor = self.actor_head(context)
        censor = self.censor_head(context)
        prob = torch.cat((event.softmax(-1), actor.softmax(-1), censor.sigmoid()), -1)
        return event, actor, censor, prob

    def central_waypoint(self, context, semantics):
        # A deterministic central DDIM path gives an identical spatial intent at
        # every action-denoising step for the same causal history. No private clock.
        x = context.new_zeros((context.shape[0],35))
        clean = x
        for i in range(len(self.plan_alphas)-1):
            a, an = self.plan_alphas[i], self.plan_alphas[i+1]
            eps = self.waypoint_diffusion(x, context.new_full((context.shape[0],),a),context,semantics)
            clean = ((x-math.sqrt(1-a)*eps)/math.sqrt(a)).clamp(-5,5)
            x = math.sqrt(an)*clean + math.sqrt(1-an)*eps
        return clean

    def plan(self, history):
        c = self.encode(history)
        event, actor, censor, semantics = self.semantic_predictions(c)
        w = self.central_waypoint(c, semantics)
        return c, w, semantics, event, actor, censor

    def decode(self, noisy, timestep, context, waypoint, semantics):
        cond = torch.cat((self.local_state(context), self.intent_embed(torch.cat((waypoint,semantics), -1))), -1)
        return self.action_diffusion(noisy,timestep,cond)

    def forward(self, noisy_action, timestep, raw_history):
        c,w,p,_,_,_ = self.plan(raw_history)
        return self.decode(noisy_action,timestep,c,w,p)

    def label_phase(self, s):
        # Used ONLY to construct training labels; this function is never called
        # from forward or by the deployment sampler. No action is scripted here.
        tcp, red, blue = s[...,18:21],s[...,25:28],s[...,32:35]
        width = s[...,7]+s[...,8]
        extent = 0.02*rotation_matrix(s[...,28:32]).abs().sum(-1)[...,:2]
        red_done = (((red[...,:2]-s[...,41:43]).abs()+extent <= 0.06).all(-1)
                    & (red[...,2]>0.014) & (red[...,2]<0.031))
        near_pad = (tcp[...,:2]-s[...,41:43]).norm(dim=-1)<0.10
        retreat_red = red_done & near_pad & (tcp[...,2]<0.285)
        at_drawer = (~red_done) & (tcp[...,0]<-0.32) & (tcp[...,1].abs()<0.06) & ((tcp-red).norm(dim=-1)>0.10)
        actor = torch.where(red_done & ~retreat_red, torch.full_like(width,2,dtype=torch.long),torch.ones_like(width,dtype=torch.long))
        actor = torch.where(at_drawer,torch.zeros_like(actor),actor)
        event = torch.full_like(actor,5)  # target-high / high approach
        red_held = (width<0.06) & ((tcp-red).norm(dim=-1)<0.045)
        blue_held = (width<0.06) & ((tcp-blue).norm(dim=-1)<0.05)
        red_at_target = (red[...,:2]-s[...,41:43]).norm(dim=-1)<0.055
        red_pickup = ((tcp[...,:2]-red[...,:2]).norm(dim=-1)<0.06) & ((tcp[...,2]-red[...,2])<0.29)
        event = torch.where((actor==1) & ~red_held & red_pickup,3,event)
        event = torch.where((actor==1) & red_held & ~red_at_target & (red[...,2]<0.27),4,event)
        event = torch.where((actor==1) & red_held & red_at_target,6,event)
        event = torch.where((actor==1) & red_held & red_at_target & (red[...,2]<0.05),1,event)
        event = torch.where((actor==1) & red_done,2,event)
        event = torch.where((actor==1) & red_done & (width<0.065),1,event)
        event = torch.where((actor==2) & ~blue_held & ((tcp[...,:2]-blue[...,:2]).norm(dim=-1)<0.07),3,event)
        event = torch.where((actor==2) & blue_held,4,event)
        event = torch.where((actor==0) & (tcp[...,2]<0.34),2,event)
        event = torch.where((actor==0) & (tcp[...,2]<0.15),1,event)
        event = torch.where((actor==0) & (s[...,39]<0.28) & (width<0.05),0,event)
        return event,actor

    def select_geometry(self, s, actor):
        xyz = torch.where((actor==0).unsqueeze(-1),drawer_center(s),
                          torch.where((actor==1).unsqueeze(-1),s[...,25:28],s[...,32:35]))
        identity = s.new_tensor([1,0,0,0,1,0]).expand(s.shape[:-1]+(6,))
        rotation = torch.where((actor==0).unsqueeze(-1),identity,
                               torch.where((actor==1).unsqueeze(-1),rot6(s[...,28:32]),rot6(s[...,35:39])))
        return xyz,rotation

    @torch.no_grad()
    def targets(self, batch):
        future = batch['future_obs']
        valid = batch['future_mask'][...,0]>0
        now = batch['raw_obs'][:,-1]
        b,h,_ = future.shape
        ef, af = self.label_phase(future)
        ec, ac = self.label_phase(now)
        slots = torch.arange(h,device=future.device).unsqueeze(0).expand(b,h)
        usable = valid & (slots>0)
        changed = usable & ((ef!=ec[:,None]) | (af!=ac[:,None]))
        first = torch.where(changed,slots,torch.full_like(slots,h)).amin(-1)
        last = torch.where(usable,slots,torch.zeros_like(slots)).amax(-1)
        censored = first==h
        index = torch.where(censored,last,first)
        rows = torch.arange(b,device=future.device)
        target = future[rows,index]
        previous = future[rows,(index-1).clamp_min(0)]
        e,a = ef[rows,index],af[rows,index]
        origin,_ = self.select_geometry(now,a)
        pos,rotation = self.select_geometry(target,a)
        prevpos,_ = self.select_geometry(previous,a)
        goal = torch.where((a==0)[:,None], now.new_tensor([-0.11,0,0.035]).expand(b,3),
                           torch.where((a==1)[:,None],now[:,41:44],now[:,44:47]))
        normtarget = (target-self.obs_mean)/self.obs_scale
        waypoint = torch.cat(((target[:,18:21]-origin)/self.xyz_scale,
                              (pos-goal)/self.xyz_scale, rot6(target[:,21:25]),rotation,
                              normtarget[:,:9],normtarget[:,39:40],
                              8*(target[:,18:21]-previous[:,18:21])/self.xyz_scale,
                              8*(pos-prevpos)/self.xyz_scale,index[:,None].to(future.dtype)/max(h-1,1)), -1)
        labelsem = torch.cat((F.one_hot(e,7),F.one_hot(a,3),censored[:,None]), -1).to(future.dtype)
        return waypoint,e,a,censored.to(future.dtype),usable.any(-1).to(future.dtype),labelsem


def build_model(spec):
    return WaypointPolicy(spec)


def masked_mean(value, valid):
    return (value*valid).sum()/valid.sum().clamp_min(1)


def compute_loss(model,batch,spec):
    target,event,actor,censored,valid,labelsem = model.targets(batch)
    context,wp,sem,event_logits,actor_logits,censor_logits = model.plan(batch['raw_obs'])
    # Half the examples use noisy teacher intents. All others use precisely the
    # differentiable central intent used by forward; no future input at deployment.
    teach = (torch.rand((wp.shape[0],1),device=wp.device)<0.5) & (valid[:,None]>0)
    used_wp = torch.where(teach,target+0.02*torch.randn_like(target),wp)
    used_sem = torch.where(teach,labelsem,sem)
    prediction = model.decode(batch['noisy_action'],batch['timesteps'],context,used_wp,used_sem)
    diffusion = epsilon_loss(prediction,batch['noise'],batch['mask'])
    # Use the framework's actual alpha_bar to train the auxiliary VP denoiser.
    alpha = batch['alpha_bar'].reshape(-1,1).clamp(1e-5,1-1e-5)
    noise = torch.randn_like(target)
    noisy_wp = alpha.sqrt()*target+(1-alpha).sqrt()*noise
    prednoise = model.waypoint_diffusion(noisy_wp,alpha,context,sem)
    wp_diffusion = masked_mean((prednoise-noise).square().mean(-1),valid)
    spatial = F.smooth_l1_loss(wp,target,reduction='none')
    spatial = masked_mean((spatial*model.waypoint_weights).sum(-1)/model.waypoint_weights.sum(),valid)
    events = masked_mean(F.cross_entropy(event_logits,event,reduction='none'),valid)
    actors = masked_mean(F.cross_entropy(actor_logits,actor,reduction='none'),valid)
    censor = masked_mean(F.binary_cross_entropy_with_logits(censor_logits[:,0],censored,reduction='none'),valid)
    prior = 0.4*wp_diffusion+0.6*spatial+0.15*events+0.1*actors+0.05*censor
    return {'loss':diffusion+prior,'diffusion_loss':diffusion,'prior_loss':prior,
            'waypoint_denoising_loss':wp_diffusion,'waypoint_reconstruction_loss':spatial,
            'event_loss':events,'actor_loss':actors,'censor_loss':censor}
