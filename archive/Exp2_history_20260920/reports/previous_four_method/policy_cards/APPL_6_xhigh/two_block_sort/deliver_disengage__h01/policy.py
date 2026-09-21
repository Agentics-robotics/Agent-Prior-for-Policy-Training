import math
import torch
from torch import nn
from appl.public import DiffusionBackbone, epsilon_loss


def unit(q):
    return q / q.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()


def conj(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def mul(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.cross(av, bv, dim=-1)), -1)


def rotate(q, v):
    # Explicit broadcast is needed by torch.cross.
    qv = q[..., 1:] + torch.zeros_like(v)
    v = v + torch.zeros_like(qv)
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


def align(q, reference):
    sign = torch.where((q * reference).sum(-1, keepdim=True) < 0,
                       -torch.ones_like(q[..., :1]), torch.ones_like(q[..., :1]))
    return q * sign


def poses(raw):
    tcp_p, tcp_q = raw[..., 18:21], unit(raw[..., 21:25])
    obj_p = torch.stack((raw[..., 25:28], raw[..., 32:35]), -2)
    obj_q = unit(torch.stack((raw[..., 28:32], raw[..., 35:39]), -2))
    goals = torch.stack((raw[..., 41:44], raw[..., 44:47]), -2)
    return tcp_p, tcp_q, obj_p, obj_q, goals


def relative(raw):
    tp, tq, op, oq, _ = poses(raw)
    qi = conj(tq).unsqueeze(-2).expand_as(oq)
    return rotate(qi, op - tp.unsqueeze(-2)), unit(mul(qi, oq))


def qerror(q, target):
    return (1.0 - (unit(q) * unit(target)).sum(-1).square()).clamp_min(0.0)


def masked_mean(value, weight):
    weight = weight + torch.zeros_like(value)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


class TemporalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(),
                                 nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                                 nn.GroupNorm(8, width), nn.SiLU(),
                                 nn.Conv1d(width, width, 1))

    def forward(self, x):
        return x + self.net(x)


class GraspFrameDiffusion(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(n['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(n['std'], dtype=torch.float32))
        # Derived features use the shared whole-demonstration TCP half ranges.
        self.register_buffer('position_scale', torch.tensor(n['std'][18:21], dtype=torch.float32))
        self.register_buffer('frequencies', torch.exp(-math.log(10000.0) * torch.arange(32).float() / 31.0))
        self.horizon = spec['training']['horizon']
        self.encoder = nn.Sequential(nn.Linear(181, 256), nn.SiLU(),
                                     nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU())
        self.current_mode = nn.Linear(256, 3)
        self.filter_weights = nn.Linear(256, 4)
        self.action_embedding = nn.Linear(8, 128)
        self.state_embedding = nn.Linear(256, 128)
        self.time_embedding = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128))
        self.slot_embedding = nn.Parameter(torch.randn(1, self.horizon, 128) * 0.01)
        self.temporal = nn.Sequential(*[TemporalBlock(128, d) for d in (1, 2, 4, 1)])
        # Two object pose increments, a free TCP increment, and three mode logits.
        self.reference_head = nn.Conv1d(128, 21, 1)
        nn.init.normal_(self.reference_head.weight, std=0.001)
        nn.init.zeros_(self.reference_head.bias)
        nn.init.zeros_(self.filter_weights.weight)
        nn.init.zeros_(self.filter_weights.bias)
        self.reference_encoder = nn.Sequential(nn.Linear(self.horizon * 37, 256), nn.SiLU(),
                                               nn.Linear(256, 128), nn.SiLU())
        self.condition_encoder = nn.Sequential(nn.Linear(401, 384), nn.SiLU())
        self.denoiser = DiffusionBackbone(384, spec['training'])

    def encode_history(self, raw):
        normalized = (raw - self.obs_mean) / self.obs_scale
        bp, bq = relative(raw)
        _, _, op, _, goals = poses(raw)
        geometry = torch.cat((bp / self.position_scale, bq,
                              (goals - op) / self.position_scale), -1)
        features = torch.cat((normalized.flatten(1), geometry.flatten(1),
                              normalized[:, -1] - normalized[:, -2]), -1)
        context = self.encoder(features)
        weights = self.filter_weights(context).reshape(-1, 2, 2).softmax(-1)
        bp = bp.transpose(1, 2)
        bq = bq.transpose(1, 2)
        bq = align(bq, bq[:, :, -1:])
        filtered_p = (weights.unsqueeze(-1) * bp).sum(2)
        filtered_q = unit((weights.unsqueeze(-1) * bq).sum(2))
        mode = self.current_mode(context)
        return context, filtered_p, filtered_q, mode

    def predict_references(self, noisy_action, timestep, raw):
        context, bp, bq, current_logits = self.encode_history(raw)
        batch, horizon, _ = noisy_action.shape
        t = torch.as_tensor(timestep, device=noisy_action.device).reshape(-1).expand(batch)
        phase = t.to(noisy_action.dtype).unsqueeze(-1) * self.frequencies
        time = self.time_embedding(torch.cat((phase.sin(), phase.cos()), -1))
        h = self.action_embedding(noisy_action) + self.slot_embedding[:, :horizon]
        h = h + (self.state_embedding(context) + time).unsqueeze(1)
        ref = self.reference_head(self.temporal(h.transpose(1, 2))).transpose(1, 2)
        tp, tq, op, oq, goals = poses(raw[:, -1])
        obj_delta = ref[..., :12].reshape(batch, horizon, 2, 6)
        obj_p = op.unsqueeze(1) + obj_delta[..., :3] * self.position_scale
        obj_rot = unit(torch.cat((torch.ones_like(obj_delta[..., :1]), obj_delta[..., 3:]), -1))
        obj_q = unit(mul(oq.unsqueeze(1).expand_as(obj_rot), obj_rot))
        free_p = tp.unsqueeze(1) + ref[..., 12:15] * self.position_scale
        free_rot = unit(torch.cat((torch.ones_like(ref[..., :1]), ref[..., 15:18]), -1))
        free_q = unit(mul(tq.unsqueeze(1).expand_as(free_rot), free_rot))
        mode_logits = current_logits.unsqueeze(1) + ref[..., 18:21]
        mode = mode_logits.softmax(-1)
        # Trust an observed grasp transform only if attachment is inferred now
        # AND predicted at this horizon slot. New closure within a chunk uses
        # the free TCP branch until the next observation refreshes the offset.
        coupled_weight = mode[..., 1:] * current_logits.softmax(-1)[:, None, 1:]
        blend = torch.cat((1.0 - coupled_weight.sum(-1, keepdim=True), coupled_weight), -1)
        # T_tcp_reference = T_object_reference * inverse(B).
        coupled_q = unit(mul(obj_q, conj(bq).unsqueeze(1).expand_as(obj_q)))
        coupled_p = obj_p - rotate(coupled_q, bp.unsqueeze(1))
        all_p = torch.cat((free_p.unsqueeze(2), coupled_p), 2)
        all_q = torch.cat((free_q.unsqueeze(2), coupled_q), 2)
        all_q = align(all_q, tq[:, None, None, :])
        tcp_p = (blend.unsqueeze(-1) * all_p).sum(2)
        tcp_q = unit((blend.unsqueeze(-1) * all_q).sum(2))
        # Keep both object streams, goal residuals and free/coupled references.
        obj_features = torch.cat(((obj_p - goals.unsqueeze(1)) / self.position_scale,
                                  (obj_p - op.unsqueeze(1)) / self.position_scale, obj_q), -1)
        ref_features = torch.cat((obj_features.flatten(2),
                                  (tcp_p - tp.unsqueeze(1)) / self.position_scale, tcp_q,
                                  (free_p - tp.unsqueeze(1)) / self.position_scale, free_q, mode), -1)
        ref_context = self.reference_encoder(ref_features.flatten(1))
        filter_context = torch.cat((bp / self.position_scale, bq), -1).flatten(1)
        condition = self.condition_encoder(torch.cat((context, ref_context, filter_context,
                                                       current_logits.softmax(-1)), -1))
        return condition, {'object_p': obj_p, 'object_q': obj_q, 'tcp_p': tcp_p,
                           'tcp_q': tcp_q, 'free_p': free_p, 'free_q': free_q,
                           'mode_logits': mode_logits, 'current_logits': current_logits,
                           'offset_p': bp, 'offset_q': bq}

    def evaluate(self, noisy_action, timestep, raw):
        condition, refs = self.predict_references(noisy_action, timestep, raw)
        epsilon = self.denoiser(noisy_action, timestep, condition)
        return epsilon, refs

    def forward(self, noisy_action, timestep, raw_history):
        return self.evaluate(noisy_action, timestep, raw_history)[0]


def build_model(spec):
    return GraspFrameDiffusion(spec)


@torch.no_grad()
def attachment_labels(raw, previous, lift_confirmation):
    tp, _, op, _, _ = poses(raw)
    bp, bq = relative(raw)
    pp, pq = relative(previous)
    width = raw[..., 7:9].mean(-1, keepdim=True)
    closed = torch.sigmoid((0.027 - width) / 0.0015)
    distance = (op - tp.unsqueeze(-2)).square().sum(-1).sqrt()
    near = torch.sigmoid((0.045 - distance) / 0.005)
    drift = (bp - pp).square().sum(-1).sqrt()
    stable = torch.sigmoid((0.012 - drift) / 0.002) * torch.exp(-qerror(bq, pq) / 0.01)
    # A later observed lift may confirm a just-closing contact, never an open hand.
    confirmed = stable + (1.0 - stable) * lift_confirmation
    scores = closed * near * confirmed
    scores = scores / scores.sum(-1, keepdim=True).clamp_min(1.0)
    free = (1.0 - scores.sum(-1, keepdim=True)).clamp_min(0.0)
    probabilities = torch.cat((free, scores), -1)
    return (probabilities + 1e-4) / (1.0 + 3e-4)


def compute_loss(model, batch, spec):
    epsilon, r = model.evaluate(batch['noisy_action'], batch['timesteps'], batch['raw_obs'])
    diffusion = epsilon_loss(epsilon, batch['noise'], batch['mask'])
    future = batch['future_obs']
    valid = batch['future_mask'][..., 0].to(epsilon.dtype)
    tp, tq, op, oq, _ = poses(future)
    history = batch['raw_obs']
    current = history[:, -1]
    scale = model.position_scale
    with torch.no_grad():
        height = op[..., 2]
        available_height = torch.where(valid.unsqueeze(-1) > 0, height, torch.full_like(height, -1.0))
        later_height = available_height.flip(1).cummax(1).values.flip(1)
        lift = ((later_height > height + 0.006) & (later_height > 0.035)).to(epsilon.dtype)
        previous = torch.cat((history[:, -2:-1], future[:, :-1]), 1)
        labels = attachment_labels(future, previous, lift)
        cur_height = poses(current)[2][..., 2]
        cur_lift = ((later_height[:, 0] > cur_height + 0.006) & (later_height[:, 0] > 0.035)).to(epsilon.dtype)
        current_labels = attachment_labels(current, history[:, -2], cur_lift)
        # Apply rigidity only to confidently held pairs across this replan.
        held = ((labels[..., 1:] > 0.8) & (current_labels[:, None, 1:] > 0.8)).to(epsilon.dtype)
        held = held * valid.unsqueeze(-1)
    obj_position = masked_mean(((r['object_p'] - op) / scale).square().mean(-1), valid.unsqueeze(-1))
    tcp_position = masked_mean(((r['tcp_p'] - tp) / scale).square().mean(-1), valid)
    free_position = masked_mean(((r['free_p'] - tp) / scale).square().mean(-1), valid)
    orientation = masked_mean(qerror(r['object_q'], oq), valid.unsqueeze(-1))
    orientation = orientation + masked_mean(qerror(r['tcp_q'], tq), valid)
    orientation = orientation + 0.3 * masked_mean(qerror(r['free_q'], tq), valid)
    reference = obj_position + tcp_position + 0.3 * free_position + 0.05 * orientation
    mode_ce = -(labels * r['mode_logits'].log_softmax(-1)).sum(-1)
    current_ce = -(current_labels * r['current_logits'].log_softmax(-1)).sum(-1)
    attachment = masked_mean(mode_ce, valid) + current_ce.mean()
    # This is deliberately evaluated on the independently predicted free TCP,
    # not on the algebraically coupled TCP where equality would be tautological.
    free_inverse = conj(r['free_q']).unsqueeze(2).expand_as(r['object_q'])
    predicted_bp = rotate(free_inverse, r['object_p'] - r['free_p'].unsqueeze(2))
    predicted_bq = unit(mul(free_inverse, r['object_q']))
    rigidity = masked_mean(((predicted_bp - r['offset_p'].unsqueeze(1)) / scale).square().mean(-1), held)
    rigidity = rigidity + 0.1 * masked_mean(qerror(predicted_bq, r['offset_q'].unsqueeze(1)), held)
    target_bp, target_bq = relative(future)
    offset = masked_mean(((r['offset_p'].unsqueeze(1) - target_bp) / scale).square().mean(-1), held)
    offset = offset + 0.1 * masked_mean(qerror(r['offset_q'].unsqueeze(1), target_bq), held)
    pair_valid = valid[:, 1:] * valid[:, :-1]
    continuity = masked_mean(qerror(r['object_q'][:, 1:], r['object_q'][:, :-1]), pair_valid.unsqueeze(-1))
    continuity = continuity + masked_mean(qerror(r['tcp_q'][:, 1:], r['tcp_q'][:, :-1]), pair_valid)
    # Direct x0-style geometric supervision at every ACTION diffusion noise level.
    # No x0 division by sqrt(alpha_bar), so the auxiliaries remain well conditioned.
    prior = 0.4 * reference + 0.1 * attachment + 0.06 * rigidity + 0.03 * offset + 0.005 * continuity
    return {'loss': diffusion + prior, 'diffusion_loss': diffusion, 'prior_loss': prior,
            'reference_loss': reference, 'attachment_loss': attachment,
            'rigidity_loss': rigidity, 'offset_loss': offset, 'orientation_continuity_loss': continuity}
