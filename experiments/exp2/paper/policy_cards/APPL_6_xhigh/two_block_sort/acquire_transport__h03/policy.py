import math
import torch
from torch import nn
from torch.nn import functional as F
from appl.public import DiffusionBackbone, epsilon_loss


def mlp(din, hidden, dout):
    return nn.Sequential(nn.Linear(din, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, dout))


def masked_mean(value, mask):
    return (value * mask).sum() / (mask.sum() * value.shape[-1]).clamp_min(1.0)


class WaypointPolicy(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(n['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(n['std'], dtype=torch.float32))
        self.register_buffer('space_scale', torch.tensor(n['std'][18:21], dtype=torch.float32))
        self.register_buffer('pose_scale', torch.tensor(n['std'][18:21] + [1.0]*4 + n['std'][18:21] + [1.0]*4, dtype=torch.float32))
        self.encoder = mlp(120, 256, 256)
        self.phase_head = mlp(256, 128, 8)
        self.plan_net = mlp(60 + 264 + 32 + 14, 256, 60)
        nn.init.normal_(self.plan_net[-1].weight, std=0.005)
        nn.init.zeros_(self.plan_net[-1].bias)
        self.reference_encoder = mlp(16*14 + 4, 192, 128)
        self.tracker = DiffusionBackbone(392, spec['training'])
        # A separate, small eight-step geometric diffusion, not the action scheduler.
        s = torch.arange(9, dtype=torch.float32) / 8
        abar = torch.cos((s + 0.008) / 1.008 * math.pi / 2).square()
        abar = abar / abar[0]
        beta = (1 - abar[1:] / abar[:-1]).clamp(0.0001, 0.999)
        alpha = 1 - beta
        ab = torch.cumprod(alpha, 0)
        prev = torch.cat([torch.ones(1), ab[:-1]])
        self.register_buffer('plan_ab', ab)
        self.register_buffer('plan_c0', beta * prev.sqrt() / (1-ab))
        self.register_buffer('plan_cx', (1-prev) * alpha.sqrt() / (1-ab))
        self.register_buffer('time_freq', torch.exp(torch.arange(16, dtype=torch.float32) * (-math.log(10000)/15)))
        self.register_buffer('query_times', torch.arange(16, dtype=torch.float32))

    def roles(self, raw):
        # Role association only: no action or waypoint is selected by this predicate.
        # The demonstrated red-first order is identifiable before predecessor release.
        last = raw[:, -1]
        blue = ((last[:, 25:27] - last[:, 41:43]).square().sum(-1) < 0.075**2)
        requested = torch.where(blue[:, None, None], raw[:, :, 32:39], raw[:, :, 25:32])
        goal = torch.where(blue[:, None, None], raw[:, :, 44:47], raw[:, :, 41:44])
        return blue, requested, goal

    def context(self, raw):
        blue, req, goal = self.roles(raw)
        tcp = raw[:, :, 18:21]
        rel = torch.cat([(tcp-req[:, :, :3])/self.space_scale,
                         (goal-req[:, :, :3])/self.space_scale,
                         (tcp-raw[:, :, 41:44])/self.space_scale,
                         (tcp-raw[:, :, 25:28])/self.space_scale,
                         blue[:, None, None].expand(-1, 2, 1).to(raw.dtype)], -1)
        norm = (raw-self.obs_mean)/self.obs_scale
        hidden = self.encoder(torch.cat([norm.flatten(1), rel.flatten(1)], -1))
        logits = self.phase_head(hidden)
        phase = logits.softmax(-1)
        # Soft progress selector determines the coordinate reference, not an action.
        bridge = phase[:, :3].sum(-1, keepdim=True) * blue[:, None].to(raw.dtype)
        anchor = bridge*raw[:, -1, 41:43] + (1-bridge)*req[:, -1, :2]
        return hidden, logits, phase, anchor, blue, req

    def encode_poses(self, tcp, obj, anchor):
        # XY is role-relative; Z remains world height with the shared affine scale.
        zmean = self.obs_mean[20].expand(anchor.shape[0], 1)
        pmean = torch.cat([anchor, zmean], -1)
        while pmean.ndim < tcp.ndim:
            pmean = pmean.unsqueeze(1)
        return torch.cat([(tcp[..., :3]-pmean)/self.space_scale, tcp[..., 3:7],
                          (obj[..., :3]-pmean)/self.space_scale, obj[..., 3:7]], -1)

    def decode_positions(self, code, anchor):
        pmean = torch.cat([anchor, self.obs_mean[20].expand(anchor.shape[0], 1)], -1)
        while pmean.ndim < code.ndim:
            pmean = pmean.unsqueeze(1)
        return code[..., :3]*self.space_scale+pmean, code[..., 7:10]*self.space_scale+pmean

    def unit_pose(self, pose):
        tq = F.normalize(pose[..., 3:7], dim=-1, eps=1e-6)
        oq = F.normalize(pose[..., 10:14], dim=-1, eps=1e-6)
        return torch.cat([pose[..., :3], tq, pose[..., 7:10], oq], -1)

    def denoise_plan(self, noisy, t, context, current):
        if not torch.is_tensor(t):
            t = torch.full((noisy.shape[0],), float(t), device=noisy.device, dtype=noisy.dtype)
        angles = (t.to(noisy.dtype).reshape(-1, 1) / 7.0 * 100.0) * self.time_freq[None]
        temb = torch.cat([angles.sin(), angles.cos()], -1)
        out = self.plan_net(torch.cat([noisy.flatten(1), context, temb, current], -1)).reshape(-1, 4, 15)
        d = out[..., :14].tanh()
        residual = torch.cat([2*d[..., :3], 0.15*d[..., 3:7], 2*d[..., 7:10], 0.15*d[..., 10:14]], -1)
        pose = self.unit_pose(current[:, None, :] + residual)
        # Positive durations measured in demonstration sample intervals.
        duration_log = 2.0 * out[..., 14:15].tanh()
        return torch.cat([pose, duration_log], -1)

    def geometric_plan(self, context, current):
        # Deterministic posterior-mean decoding from the zero mean of N(0,I).
        # No sampled future state, action, phase label or persistent progress is used.
        x = current.new_zeros((current.shape[0], 4, 15))
        for k in range(7, -1, -1):
            clean = self.denoise_plan(x, k, context, current)
            x = self.plan_c0[k]*clean + self.plan_cx[k]*x
        return clean

    def interpolate(self, plan, current):
        durations = 4.0 * plan[..., 14].clamp(-2, 2).exp()
        ends = durations.cumsum(-1)
        starts = torch.cat([ends.new_zeros((ends.shape[0], 1)), ends[:, :-1]], -1)
        u = ((self.query_times[None, :, None]-starts[:, None, :])/durations[:, None, :]).clamp(0, 1)
        smooth = u.square()*(3-2*u)
        previous = torch.cat([current[:, None], plan[:, :-1, :14]], 1)
        increments = plan[..., :14]-previous
        # Disjoint interval increments yield a convex interpolation between knots.
        ref = current[:, None, :] + (smooth[..., None]*increments[:, None, :, :]).sum(2)
        return self.unit_pose(ref)

    def condition(self, hidden, phase, reference, duration_log):
        r = self.reference_encoder(torch.cat([reference.flatten(1), duration_log], -1))
        return torch.cat([hidden, phase, r], -1)

    def forward(self, noisy_action, timestep, raw_history, return_aux=False):
        hidden, logits, phase, anchor, blue, req = self.context(raw_history)
        current = self.encode_poses(raw_history[:, -1, 18:25], req[:, -1], anchor)
        context = torch.cat([hidden, phase], -1)
        plan = self.geometric_plan(context, current)
        reference = self.interpolate(plan, current)
        cond = self.condition(hidden, phase, reference, plan[..., 14])
        eps = self.tracker(noisy_action, timestep, cond)
        if return_aux:
            return eps, (hidden, logits, phase, anchor, blue, req, current, context, plan, reference)
        return eps

    def phase_labels(self, obs, blue):
        # Weak geometric supervision, not runtime controller phases or contact truth.
        req = torch.where(blue[:, None, None], obs[:, :, 32:39], obs[:, :, 25:32])
        tcp = obs[:, :, 18:21]
        distance = (tcp[..., :2]-req[..., :2]).norm(dim=-1)
        pred_distance = (tcp[..., :2]-obs[:, :, 25:27]).norm(dim=-1)
        pad_distance = (tcp[..., :2]-obs[:, :, 41:43]).norm(dim=-1)
        width = obs[:, :, 7:9].mean(-1)
        loaded = (width < 0.030) & (distance < 0.05) & ((tcp[..., 2]-req[..., 2]).abs() < 0.04)
        labels = torch.full_like(width, 3, dtype=torch.long)
        labels = torch.where((distance < 0.04) & (tcp[..., 2] < 0.265), 4, labels)
        labels = torch.where((distance < 0.04) & (tcp[..., 2] < 0.055) & (req[..., 2] < 0.04), 5, labels)
        labels = torch.where(loaded & (req[..., 2] >= 0.028) & (req[..., 2] < 0.265), 6, labels)
        labels = torch.where(loaded & (req[..., 2] >= 0.265), 7, labels)
        bridge = blue[:, None] & (pad_distance < 0.075)
        labels = torch.where(bridge & (obs[:, :, 27] < 0.04) & (tcp[..., 2] < 0.275), 2, labels)
        labels = torch.where(bridge & (obs[:, :, 27] < 0.04) & (tcp[..., 2] < 0.055), 1, labels)
        labels = torch.where(blue[:, None] & (pred_distance < 0.05) & (obs[:, :, 27] >= 0.04) & (width < 0.03), 0, labels)
        return labels

    def make_targets(self, batch, blue, anchor, current):
        future = batch['future_obs']
        fm = batch['future_mask'][..., 0]
        obj = torch.where(blue[:, None, None], future[:, :, 32:39], future[:, :, 25:32])
        poses = self.encode_poses(future[:, :, 18:25], obj, anchor.detach())
        # Align quaternion signs to the causal pose; no rotations are discarded.
        tq = poses[..., 3:7]
        oq = poses[..., 10:14]
        tq = tq*torch.where((tq*current[:, None, 3:7].detach()).sum(-1, keepdim=True) < 0, -1.0, 1.0)
        oq = oq*torch.where((oq*current[:, None, 10:14].detach()).sum(-1, keepdim=True) < 0, -1.0, 1.0)
        poses = torch.cat([poses[..., :3], tq, poses[..., 7:10], oq], -1)
        # Four local arc-time knots. Constant time mass keeps stationary closure visible.
        delta = poses[:, 1:, :3]-poses[:, :-1, :3]
        odelta = poses[:, 1:, 7:10]-poses[:, :-1, 7:10]
        motion = delta.norm(dim=-1) + 0.5*odelta.norm(dim=-1) + 0.03
        valid_pair = fm[:, 1:]*fm[:, :-1]
        progress = torch.cat([fm.new_zeros((fm.shape[0], 1)), (motion*valid_pair).cumsum(-1)], -1)
        wanted = progress[:, -1:] * poses.new_tensor([0.25, 0.5, 0.75, 1.0])[None]
        indices = torch.searchsorted(progress.contiguous(), wanted.contiguous()).clamp(0, 15)
        last_valid = (fm*self.query_times[None]).long().amax(-1)
        previous = torch.zeros_like(last_valid)
        selected, masks, logs = [], [], []
        for k in range(4):
            ix = torch.minimum(torch.maximum(indices[:, k], previous+1), last_valid)
            ok = (ix > previous) & (fm.gather(1, ix[:, None])[:, 0] > 0)
            dur = (ix-previous).clamp_min(1).to(poses.dtype)
            selected.append(poses.gather(1, ix[:, None, None].expand(-1, 1, 14))[:, 0])
            masks.append(ok.to(poses.dtype))
            logs.append((dur/4.0).log())
            previous = ix
        target = torch.cat([torch.stack(selected, 1), torch.stack(logs, 1)[..., None]], -1).detach()
        mask = torch.stack(masks, 1)[..., None]
        return target, mask, poses.detach(), obj


def build_model(spec):
    return WaypointPolicy(spec)


def compute_loss(model, batch, spec):
    predicted, aux = model(batch['noisy_action'], batch['timesteps'], batch['raw_obs'], return_aux=True)
    hidden, logits, phase, anchor, blue, req, current, context, plan, reference = aux
    target, knot_mask, pose_target, object_target = model.make_targets(batch, blue, anchor, current)
    fm = batch['future_mask']
    main_diffusion = epsilon_loss(predicted, batch['noise'], batch['mask'])

    # The actual causal planner is used in the main forward. A second tracker view
    # learns from noisy future-labelled references; those labels never enter forward.
    teacher_ref = torch.where(fm.bool(), pose_target + 0.02*torch.randn_like(pose_target), reference.detach())
    teacher_dur = torch.where(knot_mask[..., 0].bool(), target[..., 14], plan[..., 14].detach())
    teacher_dur = teacher_dur + 0.04*torch.randn_like(teacher_dur)
    teacher_cond = model.condition(hidden, phase, teacher_ref, teacher_dur)
    teacher_eps = model.tracker(batch['noisy_action'], batch['timesteps'], teacher_cond)
    teacher_diffusion = epsilon_loss(teacher_eps, batch['noise'], batch['mask'])
    diffusion_loss = (main_diffusion + 0.15*teacher_diffusion)/1.15

    # Conditional x0 diffusion regression is a weighted score-matching objective.
    pt = torch.randint(0, 8, (target.shape[0],), device=target.device)
    pa = model.plan_ab[pt][:, None, None]
    pnoise = torch.randn_like(target)
    noisy_plan = pa.sqrt()*target + (1-pa).sqrt()*pnoise
    predicted_plan = model.denoise_plan(noisy_plan, pt, context, current)
    plan_diffusion = masked_mean((predicted_plan-target).square(), knot_mask)
    plan_rollout = masked_mean((plan-target).square(), knot_mask)
    trajectory_loss = masked_mean((reference-pose_target).square(), fm)
    phase_labels = model.phase_labels(batch['raw_obs'], blue)[:, -1]
    phase_loss = F.cross_entropy(logits, phase_labels)

    # Soft geometric penalties act on generated poses, not solely observed inputs.
    tcp, obj = model.decode_positions(reference, anchor)
    labels = model.phase_labels(batch['future_obs'], blue)
    approach = ((labels == 4) | (labels == 5)).to(tcp.dtype)[..., None]*fm
    lateral = (tcp[..., :2]-object_target[..., :2]).norm(dim=-1, keepdim=True)
    approach_loss = masked_mean((F.relu(lateral-0.015)/model.space_scale[0]).square(), approach)
    pair_mask = fm[:, 1:]*fm[:, :-1]
    step_xy = ((tcp[:, 1:, :2]-tcp[:, :-1, :2])/model.space_scale[:2]).square().sum(-1, keepdim=True)
    empty = ((labels[:, 1:] == 2) | (labels[:, 1:] == 3)).to(tcp.dtype)[..., None]*pair_mask
    low_clearance = F.relu((0.25-torch.minimum(tcp[:, 1:, 2:3], tcp[:, :-1, 2:3]))/model.space_scale[2])
    separation_loss = masked_mean(step_xy*low_clearance.square(), empty)
    loaded = ((labels[:, 1:] == 6) | (labels[:, 1:] == 7)).to(tcp.dtype)[..., None]*pair_mask
    low_lift = F.relu((0.25-torch.minimum(obj[:, 1:, 2:3], obj[:, :-1, 2:3]))/model.space_scale[2])
    lift_loss = masked_mean(step_xy*low_lift.square(), loaded)
    table_loss = masked_mean((F.relu((0.005-tcp[..., 2:3])/model.space_scale[2])).square(), fm)
    geometry_loss = approach_loss + separation_loss + lift_loss + table_loss

    # Match demonstrated actuator increments, not a fictitious IK trajectory.
    ab = batch['alpha_bar'].reshape(-1, 1, 1).clamp_min(1e-6)
    x0 = (batch['noisy_action']-(1-ab).sqrt()*predicted)/ab.sqrt()
    delta_error = (x0[:, 1:]-x0[:, :-1]) - (batch['encoded_action'][:, 1:]-batch['encoded_action'][:, :-1])
    smooth_mask = batch['mask'][:, 1:]*batch['mask'][:, :-1]
    smooth_loss = masked_mean(F.smooth_l1_loss(delta_error, torch.zeros_like(delta_error), reduction='none')*ab, smooth_mask)

    prior_loss = (0.15*plan_diffusion + 0.15*plan_rollout + 0.30*trajectory_loss +
                  0.05*phase_loss + 0.10*geometry_loss + 0.02*smooth_loss)
    return {'loss': diffusion_loss+prior_loss, 'diffusion_loss': diffusion_loss,
            'prior_loss': prior_loss, 'plan_diffusion_loss': plan_diffusion,
            'plan_rollout_loss': plan_rollout, 'trajectory_loss': trajectory_loss,
            'phase_loss': phase_loss, 'geometry_loss': geometry_loss,
            'action_increment_loss': smooth_loss}
