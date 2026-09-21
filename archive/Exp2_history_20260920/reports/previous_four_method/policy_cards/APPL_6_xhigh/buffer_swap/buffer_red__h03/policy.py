import math
import torch
from torch import nn
from appl.public import DiffusionBackbone, epsilon_loss


def rotation6(q):
    q = q / q.square().sum(-1, keepdim=True).clamp_min(1e-8).sqrt()
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y),
                        2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)), -1)


def mse_mask(x, y, mask):
    return ((x-y).square()*mask).sum() / (mask.sum()*x.shape[-1]).clamp_min(1)


def weighted_mse(x, y, mask, reliability):
    return ((x-y).square()*mask*reliability).sum() / (mask.sum()*x.shape[-1]).clamp_min(1)


class RouteDiffuser(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('freq', torch.exp(torch.arange(32).float()*(-math.log(10000)/31)))
        self.input = nn.Linear(44+256+64, 256)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 512),
                                                 nn.SiLU(), nn.Linear(512, 256)) for _ in range(3)])
        self.output = nn.Sequential(nn.LayerNorm(256), nn.SiLU(), nn.Linear(256, 44))

    def forward(self, y, t, context):
        tt = torch.as_tensor(t, device=y.device, dtype=y.dtype).reshape(-1).expand(y.shape[0])
        angles = tt[:, None]*self.freq[None]
        h = self.input(torch.cat((y.flatten(1), context, angles.sin(), angles.cos()), -1))
        for block in self.blocks:
            h = h + block(h)
        return self.output(h).reshape(-1, 4, 11)


class TrajectoryPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.position = nn.Parameter(torch.randn(1, 16, 32)*0.02)
        self.input = nn.Conv1d(8+256+32, 128, 1)
        self.blocks = nn.ModuleList([nn.Sequential(nn.GroupNorm(8, 128), nn.SiLU(),
                                                  nn.Conv1d(128, 128, 3, padding=1), nn.SiLU(),
                                                  nn.Conv1d(128, 128, 3, padding=1)) for _ in range(3)])
        self.output = nn.Conv1d(128, 9, 1)

    def forward(self, actions, context, baseline):
        b, h, _ = actions.shape
        x = torch.cat((actions, context[:, None].expand(-1, h, -1),
                       self.position[:, :h].expand(b, -1, -1)), -1).transpose(1, 2)
        x = self.input(x)
        for block in self.blocks:
            x = x + block(x)
        return baseline[:, None] + self.output(x).transpose(1, 2)


class HierarchicalPolicy(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(n['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(n['std'], dtype=torch.float32))
        self.register_buffer('xyz_scale', 2*torch.tensor(n['std'][18:21], dtype=torch.float32))
        self.encoder = nn.Sequential(nn.Linear(171, 256), nn.SiLU(), nn.Linear(256, 256),
                                     nn.LayerNorm(256), nn.SiLU())
        self.reference_head = nn.Linear(256, 2)
        self.coarse = RouteDiffuser()
        self.phase_head = nn.Sequential(nn.Linear(256+44, 128), nn.SiLU(), nn.Linear(128, 20))
        self.fine_condition = nn.Sequential(nn.Linear(256+44+20+2, 256), nn.SiLU(),
                                           nn.Linear(256, 256))
        self.fine = DiffusionBackbone(256, spec['training'])
        self.rollout = TrajectoryPredictor()
        # Internal route diffusion only. The motor DDPM is supplied by the framework.
        s = torch.arange(101, dtype=torch.float32)/100
        c = torch.cos((s+0.008)/1.008*math.pi/2).square()
        self.register_buffer('route_abar', (c[1:]/c[0]).clamp_min(1e-4))
        # One fixed Gaussian latent gives a repeatable, observation-revised route.
        self.register_buffer('route_seed', torch.randn(1, 4, 11))
        self.route_times = (99, 79, 59, 39, 19, 0)

    def encode(self, raw):
        obs = (raw-self.obs_mean)/self.obs_scale
        tcp, red, blue = raw[..., 18:21], raw[..., 25:28], raw[..., 32:35]
        rel = torch.cat((tcp-red, tcp-blue, blue-red,
                         red-raw[..., 41:44], blue-raw[..., 44:47]), -1)
        rel = rel / self.xyz_scale.repeat(5)
        x = torch.cat((obs.flatten(1), obs[:, 1]-obs[:, 0], rel.flatten(1)), -1)
        context = self.encoder(x)
        logits = self.reference_head(context)
        weights = logits.softmax(-1)
        anchor = weights[:, :1]*red[:, -1]+weights[:, 1:]*blue[:, -1]
        return context, logits, weights, anchor

    def sample_route(self, context):
        y = self.route_seed.expand(context.shape[0], -1, -1)
        clean = y
        for i, t in enumerate(self.route_times):
            a = self.route_abar[t]
            eps = self.coarse(y, t, context)
            clean = ((y-(1-a).sqrt()*eps)/a.sqrt()).clamp(-2, 2)
            clean = torch.cat((clean[..., :3], clean[..., 3:].clamp(-1, 1)), -1)
            if i+1 < len(self.route_times):
                an = self.route_abar[self.route_times[i+1]]
                # Deterministic DDIM for the auxiliary route, not a new motor sampler.
                y = an.sqrt()*clean+(1-an).sqrt()*eps
        return clean

    def predict(self, noisy_action, timestep, raw_history):
        context, logits, weights, anchor = self.encode(raw_history)
        route = self.sample_route(context)
        phase_logits = self.phase_head(torch.cat((context, route.flatten(1)), -1)).reshape(-1, 4, 5)
        cond = self.fine_condition(torch.cat((context, route.flatten(1),
                                             phase_logits.sigmoid().flatten(1), weights), -1))
        eps = self.fine(noisy_action, timestep, cond)
        return eps, (context, logits, anchor, route, phase_logits)

    def forward(self, noisy_action, timestep, raw_history):
        return self.predict(noisy_action, timestep, raw_history)[0]

    def tcp_representation(self, states):
        return torch.cat(((states[..., 18:21]-self.obs_mean[18:21])/self.obs_scale[18:21],
                          rotation6(states[..., 21:25])), -1)

    def route_pose(self, route, anchor):
        pos = (route[..., :3]*self.xyz_scale+anchor[:, None]-self.obs_mean[18:21])/self.obs_scale[18:21]
        return torch.cat((pos, route[..., 3:9]), -1)


def build_model(spec):
    return HierarchicalPolicy(spec)


def keyframe_targets(model, batch, anchor):
    f = batch['future_obs']
    valid = batch['future_mask']*batch['mask']
    action = batch['encoded_action']
    prev = torch.cat((batch['raw_obs'][:, :1], f[:, :-1]), 1)
    delta = f[..., 18:21]-prev[..., 18:21]
    lateral = delta[..., :2].square().sum(-1).sqrt()
    dz = delta[..., 2]
    grip_delta = torch.cat((torch.zeros_like(action[:, :1, 7]), action[:, 1:, 7]-action[:, :-1, 7]), 1)
    finger_delta = f[..., 7:9].sum(-1)-prev[..., 7:9].sum(-1)
    phases = torch.stack(((dz > 0.001).float(), (dz < -0.001).float(),
                          ((lateral > 0.001) & (f[..., 20] > 0.18)).float(),
                          ((grip_delta > 0.3) | (finger_delta > 0.0003)).float(),
                          ((grip_delta < -0.3) | (finger_delta < -0.0003)).float()), -1)
    band = ((f[..., 20] > 0.26) & (prev[..., 20] <= 0.26)).float()
    previous_dz = torch.cat((dz[:, :1], dz[:, :-1]), 1)
    turn = ((dz*previous_dz < 0) & ((dz-previous_dz).abs() > 0.0005)).float()
    # Temporal order is retained by taking an event-weighted representative in each quartile.
    event_weight = 1+4*(phases[..., 3]+phases[..., 4])+2*band+2*turn
    weight = event_weight[..., None]*valid
    b = f.shape[0]
    w = weight.reshape(b, 4, 4, 1)
    denom = w.sum(2).clamp_min(1e-6)
    pos = (f[..., 18:21]-anchor.detach()[:, None])/model.xyz_scale
    pose = torch.cat((pos, rotation6(f[..., 21:25]), action[..., 7:8]), -1)
    pooled = (pose.reshape(b, 4, 4, 10)*w).sum(2)/denom
    local_time = torch.linspace(-1, 1, 4, device=f.device, dtype=f.dtype).reshape(1, 1, 4, 1)
    tau = (local_time*w).sum(2)/denom
    labels = torch.cat((pooled, tau), -1)
    phase_labels = (phases.reshape(b, 4, 4, 5)*w).sum(2)/denom
    waypoint_mask = (valid.reshape(b, 4, 4, 1).sum(2)>0).to(f.dtype)
    return labels, phase_labels, waypoint_mask


def sample_at_waypoints(values, route):
    b, _, d = values.shape
    starts = torch.arange(4, device=route.device, dtype=route.dtype)*4
    times = starts[None]+1.5*(route[..., 10].clamp(-1, 1)+1)
    low = times.floor().long().clamp(0, 15)
    high = (low+1).clamp(max=15)
    fraction = (times-low.to(times.dtype))[..., None]
    v0 = values.gather(1, low[..., None].expand(b, 4, d))
    v1 = values.gather(1, high[..., None].expand(b, 4, d))
    return v0*(1-fraction)+v1*fraction


def compute_loss(model, batch, spec):
    eps, info = model.predict(batch['noisy_action'], batch['timesteps'], batch['raw_obs'])
    context, reference_logits, anchor, route, phase_logits = info
    diffusion_loss = epsilon_loss(eps, batch['noise'], batch['mask'])
    target_route, target_phase, waypoint_mask = keyframe_targets(model, batch, anchor)
    b = eps.shape[0]
    coarse_t = torch.randint(0, 100, (b,), device=eps.device)
    a = model.route_abar[coarse_t].reshape(b, 1, 1)
    route_noise = torch.randn_like(target_route)
    noisy_route = a.sqrt()*target_route+(1-a).sqrt()*route_noise
    route_eps = model.coarse(noisy_route, coarse_t, context)
    coarse_diffusion = mse_mask(route_eps, route_noise, waypoint_mask)
    route_supervision = mse_mask(route, target_route, waypoint_mask)
    phase_element = torch.nn.functional.binary_cross_entropy_with_logits(phase_logits, target_phase, reduction='none')
    phase_loss = (phase_element*waypoint_mask).sum()/(5*waypoint_mask.sum()).clamp_min(1)

    f = batch['future_obs']
    valid = batch['future_mask']*batch['mask']
    d_red = (f[..., 18:21]-f[..., 25:28]).square().sum(-1)
    d_blue = (f[..., 18:21]-f[..., 32:35]).square().sum(-1)
    distances = torch.stack((d_red, d_blue), -1)
    distances = (distances*valid).sum(1)/valid.sum(1).clamp_min(1)
    reference_target = (-distances/0.01).softmax(-1)
    reference_loss = -(reference_target*reference_logits.log_softmax(-1)).sum(-1).mean()

    abar = batch['alpha_bar'].reshape(b, 1, 1).clamp_min(1e-5)
    decoded = ((batch['noisy_action']-(1-abar).sqrt()*eps)/abar.sqrt()).clamp(-1.5, 1.5)
    baseline = model.tcp_representation(batch['raw_obs'][:, -1])
    rollout_true = model.rollout(batch['encoded_action'], context, baseline)
    rollout_decoded = model.rollout(decoded, context, baseline)
    future_tcp = model.tcp_representation(f)
    rollout_loss = mse_mask(rollout_true, future_tcp, valid)
    rollout_loss = rollout_loss+weighted_mse(rollout_decoded, future_tcp, valid, abar)

    # Agreement depends on both learned route and action-conditioned learned trajectory.
    predicted_keyframes = sample_at_waypoints(rollout_decoded, route)
    key_valid = sample_at_waypoints(valid, route).detach()*waypoint_mask
    consistency = weighted_mse(predicted_keyframes, model.route_pose(route, anchor), key_valid, abar)
    grip_at_waypoints = sample_at_waypoints(decoded[..., 7:8], route)
    consistency = consistency+0.25*weighted_mse(grip_at_waypoints, route[..., 9:10], key_valid, abar)

    # Exempt demonstrated low aligned approach and supported placement. This gate is a
    # training label, not a controller or proof of contact. The penalized motion is predicted.
    native_tcp = f[..., 18:21]
    rxy = (native_tcp[..., :2]-f[..., 25:27]).square().sum(-1).sqrt()
    bxy = (native_tcp[..., :2]-f[..., 32:34]).square().sum(-1).sqrt()
    supported_near = ((rxy < 0.04) & (f[..., 27] < 0.04)) | ((bxy < 0.04) & (f[..., 34] < 0.04))
    allowed = supported_near & ((batch['encoded_action'][..., 7] > 0) | (native_tcp[..., 2] < 0.065))
    pred_xyz = rollout_decoded[..., :3]*model.obs_scale[18:21]+model.obs_mean[18:21]
    xy_step = (pred_xyz[:, 1:, :2]-pred_xyz[:, :-1, :2]).square().sum(-1).clamp_min(1e-12).sqrt()
    height_gate = torch.sigmoid((0.12-0.5*(pred_xyz[:, 1:, 2]+pred_xyz[:, :-1, 2]))/0.025)
    excess = torch.relu(xy_step-0.0015).square()/0.01**2
    pair_mask = valid[:, 1:, 0]*valid[:, :-1, 0]*(~allowed[:, 1:]).to(eps.dtype)
    sweep_loss = (excess*height_gate*pair_mask*abar[:, 0]).sum()/pair_mask.sum().clamp_min(1)

    prior_loss = (0.4*coarse_diffusion+0.25*route_supervision+0.15*rollout_loss+
                  0.10*consistency+0.03*phase_loss+0.01*reference_loss+0.02*sweep_loss)
    return {'loss': diffusion_loss+prior_loss, 'diffusion_loss': diffusion_loss,
            'prior_loss': prior_loss, 'coarse_diffusion_loss': coarse_diffusion,
            'route_supervision_loss': route_supervision, 'rollout_loss': rollout_loss,
            'consistency_loss': consistency, 'sweep_loss': sweep_loss}
