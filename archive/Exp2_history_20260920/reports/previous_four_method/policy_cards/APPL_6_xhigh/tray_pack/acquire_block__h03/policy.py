import math
import torch
from torch import nn
from torch.nn import functional as F
from appl.public import DiffusionBackbone, normalize_observation, epsilon_loss


def unit(q):
    return F.normalize(q, dim=-1, eps=1e-8)


def conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def multiply(a, b):
    av, bv = torch.broadcast_tensors(a[..., 1:], b[..., 1:])
    w = a[..., :1] * b[..., :1] - (av * bv).sum(-1, keepdim=True)
    v = a[..., :1] * bv + b[..., :1] * av + torch.cross(av, bv, dim=-1)
    return torch.cat((w, v), dim=-1)


def rotate(q, v):
    qv, v = torch.broadcast_tensors(q[..., 1:], v)
    uv = torch.cross(qv, v, dim=-1)
    return v + 2.0 * (q[..., :1] * uv + torch.cross(qv, uv, dim=-1))


def canonical(q):
    largest = q.gather(-1, q.abs().argmax(-1, keepdim=True))
    return q * torch.where(largest < 0, -torch.ones_like(largest), torch.ones_like(largest))


def poses(s):
    p = torch.stack((s[..., 18:21], s[..., 25:28], s[..., 32:35]), dim=-2)
    q = unit(torch.stack((s[..., 21:25], s[..., 28:32], s[..., 35:39]), dim=-2))
    return p, q


def relative(p, q):
    inverse = conjugate(q[..., :1, :])
    return rotate(inverse, p[..., 1:, :] - p[..., :1, :]), unit(multiply(inverse, q[..., 1:, :]))


def angle_error(a, b):
    return 1.0 - (unit(a) * unit(b)).sum(-1).square().clamp(0.0, 1.0)


def huber(x, beta=0.05):
    return F.smooth_l1_loss(x, torch.zeros_like(x), reduction='none', beta=beta)


def average(x, mask):
    mask = mask.to(x.dtype)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(x)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation))

    def forward(self, x):
        return x + self.net(x)


class TwoRegimeForecast(nn.Module):
    def __init__(self, horizon, position_scale):
        super().__init__()
        self.register_buffer('position_scale', position_scale.clone())
        self.register_buffer('frequencies', 2.0 ** torch.arange(8, dtype=torch.float32))
        self.slots = nn.Parameter(torch.randn(horizon, 16) * 0.02)
        self.input = nn.Linear(256 + 3 + 8 + 16 + 16, 192)
        self.temporal = nn.Sequential(*[TemporalBlock(192, d) for d in (1, 2, 4)])
        self.output = nn.Linear(192, 41)
        nn.init.normal_(self.output.weight, std=0.001)
        nn.init.zeros_(self.output.bias)

    def forward(self, action, level, context, belief_logits, history):
        b, h, _ = action.shape
        phase = level.reshape(b, 1) * self.frequencies[None] * math.pi
        time = torch.cat((phase.sin(), phase.cos()), -1)
        inp = torch.cat((context[:, None].expand(-1, h, -1),
                         belief_logits[:, None].expand(-1, h, -1), action,
                         time[:, None].expand(-1, h, -1),
                         self.slots[None, :h].expand(b, -1, -1)), -1)
        features = self.temporal(F.silu(self.input(inp)).transpose(1, 2)).transpose(1, 2)
        out = self.output(features)
        p0, q0 = poses(history[:, -1])
        cp0, cq0 = relative(p0, q0)
        scale = self.position_scale
        tcp_p = p0[:, None, 0] + out[..., :3] * scale
        tcp_q = unit(q0[:, None, 0] + 0.1 * out[..., 3:7])
        independent_delta = out[..., 7:21].reshape(b, h, 2, 7)
        independent_p = p0[:, None, 1:] + independent_delta[..., :3] * scale
        independent_q = unit(q0[:, None, 1:] + 0.1 * independent_delta[..., 3:])
        c_delta = out[..., 21:35].reshape(b, h, 2, 7)
        cp = cp0[:, None] + c_delta[..., :3] * scale.max()
        cq = unit(cq0[:, None] + 0.1 * c_delta[..., 3:])
        attached_p = tcp_p[..., None, :] + rotate(tcp_q[..., None, :], cp)
        attached_q = unit(multiply(tcp_q[..., None, :], cq))
        logits = out[..., 35:38] + belief_logits[:, None]
        probabilities = logits.softmax(-1)
        attached_weight = probabilities[..., 1:, None]
        obj_p = attached_weight * attached_p + (1.0 - attached_weight) * independent_p
        sign = torch.where((attached_q * independent_q).sum(-1, keepdim=True) < 0, -1.0, 1.0)
        obj_q = unit(attached_weight * attached_q * sign + (1.0 - attached_weight) * independent_q)
        p = torch.cat((tcp_p[..., None, :], obj_p), -2)
        q = torch.cat((tcp_q[..., None, :], obj_q), -2)
        sigma = 0.005 + 0.25 * torch.sigmoid(out[..., 38:41])
        summary = torch.cat((((p - p0[:, None]) / scale).flatten(-2), probabilities, sigma), -1)
        return {'p': p, 'q': q, 'independent_p': independent_p, 'independent_q': independent_q,
                'attached_p': attached_p, 'attached_q': attached_q, 'cp': cp, 'cq': cq,
                'logits': logits, 'probabilities': probabilities, 'sigma': sigma, 'summary': summary}


class AttachmentDiffusionPolicy(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.train_steps = spec['training']['denoising_train_steps']
        scale = torch.tensor(spec['normalizer']['std'][18:21], dtype=torch.float32)
        self.register_buffer('position_scale', scale)
        self.encoder = nn.Sequential(nn.Linear(167, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.LayerNorm(256))
        self.belief = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 3))
        horizon = spec['training']['horizon']
        self.forecast = TwoRegimeForecast(horizon, scale)
        self.forecast_encoder = nn.Sequential(nn.Linear(horizon * 15, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        self.diffusion = DiffusionBackbone(256 + 3 + 128, spec['training'])

    def encode(self, raw):
        p, q = poses(raw)
        cp, cq = relative(p, q)
        goals = torch.stack((raw[..., 41:44], raw[..., 44:47]), -2)
        geometry = torch.cat(((cp / self.position_scale.max()).flatten(-2), canonical(cq).flatten(-2),
                              ((goals - p[..., 1:, :]) / self.position_scale).flatten(-2),
                              ((goals - p[..., :1, :]) / self.position_scale).flatten(-2),
                              (p[..., 2, :] - p[..., 1, :]) / self.position_scale), -1)
        motion = torch.cat((((p[:, 1] - p[:, 0]) / self.position_scale).flatten(1),
                            ((cp[:, 1] - cp[:, 0]) / self.position_scale.max()).flatten(1)), -1)
        inputs = torch.cat((normalize_observation(raw, self.spec).flatten(1), geometry.flatten(1), motion), -1)
        context = self.encoder(inputs)
        return context, self.belief(context)

    def level(self, timestep, batch, device):
        t = torch.as_tensor(timestep, device=device).reshape(-1).expand(batch).float()
        return (t + 1.0) / float(self.train_steps)

    def evaluate(self, noisy_action, timestep, raw_history):
        context, belief = self.encode(raw_history)
        forecast = self.forecast(noisy_action, self.level(timestep, noisy_action.shape[0], noisy_action.device),
                                 context, belief, raw_history)
        fc = self.forecast_encoder(forecast['summary'].flatten(1))
        condition = torch.cat((context, belief.softmax(-1), fc), -1)
        eps = self.diffusion(noisy_action, timestep, condition)
        return eps, context, belief, forecast

    def forward(self, noisy_action, timestep, raw_history):
        return self.evaluate(noisy_action, timestep, raw_history)[0]


def build_model(spec):
    return AttachmentDiffusionPolicy(spec)


@torch.no_grad()
def regime_labels(previous, current, command=None):
    # Conservative kinematic proxies. Contact/closure ambiguity is masked, not labeled as a miss.
    p, q = poses(current)
    pp, pq = poses(previous)
    cp, cq = relative(p, q)
    cpp, cqp = relative(pp, pq)
    goals = torch.stack((current[..., 41:44], current[..., 44:47]), -2)
    goal_xy = (p[..., 1:, :2] - goals[..., :2]).norm(dim=-1)
    support = torch.where(goal_xy < 0.08, goals[..., 2], torch.full_like(goal_xy, 0.02))
    separation = (p[..., 1:, :] - p[..., :1, :]).norm(dim=-1)
    object_motion = p[..., 1:, :] - pp[..., 1:, :]
    tcp_motion = p[..., :1, :] - pp[..., :1, :]
    moving = object_motion.norm(dim=-1) > 0.00025
    elevated = p[..., 1:, 2] > support + 0.01
    away_from_contact = (p[..., 1:, 2] > support + 0.003) & (pp[..., 1:, 2] > support + 0.001)
    rigid = ((cp - cpp).norm(dim=-1) < 0.0025) & (angle_error(cq, cqp) < 0.001)
    matches_motion = (object_motion - tcp_motion).norm(dim=-1) < 0.0025
    width = current[..., 7:9].sum(-1)
    closed = (width > 0.025) & (width < 0.050)
    opened = width > 0.058
    if command is not None:
        closed = closed & (command < -0.25)
        opened = opened | ((command > 0.25) & (width > 0.046))
    attached = closed[..., None] & (separation < 0.035) & rigid & matches_motion & away_from_contact & (moving | elevated)
    count = attached.sum(-1)
    any_attached = count == 1
    all_far = (separation > 0.065).all(-1)
    label = torch.where(any_attached, attached.long().argmax(-1) + 1, torch.zeros_like(count))
    valid = any_attached | ((opened | all_far) & (count == 0))
    return label, valid


def pose_loss(pred, target_p, target_q, mask, scale, reliability=None):
    error = huber((pred['p'] - target_p) / scale).mean(-1) + 0.1 * angle_error(pred['q'], target_q)
    if reliability is not None:
        error = error * reliability
    return average(error, mask)


def regime_loss(logits, labels, valid):
    error = F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1), reduction='none').reshape_as(labels)
    return average(error, valid)


def branch_loss(pred, tp, tq, labels, valid, scale):
    attached = F.one_hot(labels, 3)[..., 1:].to(tp.dtype)
    ap = huber((pred['attached_p'] - tp[..., 1:, :]) / scale).mean(-1)
    aq = angle_error(pred['attached_q'], tq[..., 1:, :])
    ip = huber((pred['independent_p'] - tp[..., 1:, :]) / scale).mean(-1)
    iq = angle_error(pred['independent_q'], tq[..., 1:, :])
    return average(attached * (ap + 0.1 * aq) + (1.0 - attached) * (ip + 0.1 * iq), valid)


def rigidity_loss(pred, labels, valid, scale):
    cp, cq = relative(pred['p'], pred['q'])
    attached = F.one_hot(labels, 3)[..., 1:].bool()
    pairs = attached[:, 1:] & attached[:, :-1] & valid[:, 1:, None].bool() & valid[:, :-1, None].bool()
    dp = huber((cp[:, 1:] - cp[:, :-1]) / scale.max()).mean(-1)
    dq = angle_error(cq[:, 1:], cq[:, :-1])
    # The branch transform is also learned; neither term is a constant pose penalty.
    dc = huber((pred['cp'][:, 1:] - pred['cp'][:, :-1]) / scale.max()).mean(-1)
    dr = angle_error(pred['cq'][:, 1:], pred['cq'][:, :-1])
    return average(dp + 0.1 * dq + dc + 0.1 * dr, pairs)


def stationary_loss(pred, tp, labels, valid, scale):
    independent = ~F.one_hot(labels, 3)[..., 1:].bool()
    still = (tp[:, 1:, 1:] - tp[:, :-1, 1:]).norm(dim=-1) < 0.0003
    pairs = independent[:, 1:] & independent[:, :-1] & still & valid[:, 1:, None].bool() & valid[:, :-1, None].bool()
    change = (pred['p'][:, 1:, 1:] - pred['p'][:, :-1, 1:]) / scale
    return average(huber(change).mean(-1), pairs)


def uncertainty_loss(pred, tp, mask, scale):
    # Residual-scale calibration, not a claim of epistemic or failure uncertainty.
    target = ((pred['p'].detach() - tp) / scale).square().mean(-1).add(1e-8).sqrt().clamp(0.005, 0.255)
    return average(huber(pred['sigma'] - target, beta=0.02), mask)


def compute_loss(model, batch, spec):
    raw = batch['raw_obs']
    eps, context, belief, noisy_forecast = model.evaluate(batch['noisy_action'], batch['timesteps'], raw)
    diffusion_loss = epsilon_loss(eps, batch['noise'], batch['mask'])
    mask = (batch['mask'] * batch['future_mask']).squeeze(-1)
    future = batch['future_obs']
    previous = torch.cat((raw[:, :1], future[:, :-1]), dim=1)
    labels, valid = regime_labels(previous, future, batch['native_action'][..., 7])
    previous_valid = torch.cat((torch.ones_like(mask[:, :1]), mask[:, :-1]), dim=1)
    valid = valid.to(mask.dtype) * mask * previous_valid
    current_label, current_valid = regime_labels(raw[:, 0], raw[:, 1])
    tp, tq = poses(future)
    scale = model.position_scale
    zero_level = torch.zeros_like(batch['alpha_bar'])
    clean_forecast = model.forecast(batch['encoded_action'], zero_level, context, belief, raw)
    alpha = batch['alpha_bar'].reshape(-1)
    forecast_weight = 0.2 + 0.8 * alpha[:, None, None]
    clean_pose = pose_loss(clean_forecast, tp, tq, mask, scale)
    noisy_pose = pose_loss(noisy_forecast, tp, tq, mask, scale, forecast_weight)
    branches = branch_loss(clean_forecast, tp, tq, labels, valid, scale)
    regimes = regime_loss(clean_forecast['logits'], labels, valid)
    regimes = regimes + 0.5 * regime_loss(noisy_forecast['logits'], labels, valid)
    regimes = regimes + regime_loss(belief, current_label, current_valid)
    rigidity = rigidity_loss(clean_forecast, labels, valid, scale) + 0.5 * rigidity_loss(noisy_forecast, labels, valid, scale)
    stationary = stationary_loss(clean_forecast, tp, labels, valid, scale)
    uncertainty = uncertainty_loss(clean_forecast, tp, mask, scale) + uncertainty_loss(noisy_forecast, tp, mask, scale)

    # Distill forecast consequences into denoising, without changing the fixed sampler.
    a = alpha[:, None, None]
    x0 = ((batch['noisy_action'] - (1.0 - a).sqrt() * eps) / a.sqrt().clamp_min(1e-4)).clamp(-1.25, 1.25)
    frozen = {name: value.detach() for name, value in model.forecast.named_parameters()}
    frozen.update({name: value.detach() for name, value in model.forecast.named_buffers()})
    estimated = torch.func.functional_call(model.forecast, frozen, (x0, zero_level, context.detach(), belief.detach(), raw))
    reliability = (0.03 / (clean_forecast['sigma'].detach() + 0.01)).clamp(0.15, 1.0)
    reliability = reliability * (alpha.square() * (alpha > 0.2))[:, None, None]
    consequence = pose_loss(estimated, tp, tq, mask, scale, reliability)
    prior_loss = (0.60 * clean_pose + 0.20 * noisy_pose + 0.20 * branches + 0.08 * regimes
                  + 0.15 * rigidity + 0.05 * stationary + 0.05 * uncertainty + 0.15 * consequence)
    return {'loss': diffusion_loss + prior_loss, 'diffusion_loss': diffusion_loss, 'prior_loss': prior_loss}
