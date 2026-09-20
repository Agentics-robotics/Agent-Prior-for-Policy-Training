"""Attachment-consistent learned action diffusion; no executable action controller."""
import math
import torch
from torch import nn
from appl.public import DiffusionBackbone, epsilon_loss


def unit(q):
    return q / q.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()


def canonical(q):
    q = unit(q)
    pivot = q.gather(-1, q.abs().argmax(-1, keepdim=True))
    return q * torch.where(pivot < 0, -torch.ones_like(pivot), torch.ones_like(pivot))


def conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def multiply(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.cross(av, bv, dim=-1)), -1)


def rotate(q, v):
    q = unit(q)
    u = q[..., 1:]
    c = 2 * torch.cross(u, v, dim=-1)
    return v + q[..., :1] * c + torch.cross(u, c, dim=-1)


def angle_square(a, b):
    d = multiply(conjugate(unit(a)), unit(b))
    vn = (d[..., 1:].square().sum(-1) + 1e-12).sqrt()
    return (2 * torch.atan2(vn, d[..., 0].abs().clamp_min(1e-6))).square()


def pack(raw):
    return torch.cat((raw[..., 18:39], raw[..., 39:40], raw[..., 7:9]), -1)


def normalize_poses(s):
    return torch.cat((s[..., :3], canonical(s[..., 3:7]),
                      s[..., 7:10], canonical(s[..., 10:14]),
                      s[..., 14:17], canonical(s[..., 17:21]), s[..., 21:]), -1)


def relative(s):
    """Packed TCP/red/blue poses -> two tool-frame translations and wxyz rotations."""
    objects = torch.stack((s[..., 7:14], s[..., 14:21]), -2)
    iq = conjugate(unit(s[..., 3:7])).unsqueeze(-2).expand_as(objects[..., 3:7])
    p = rotate(iq, objects[..., :3] - s[..., None, :3])
    q = canonical(multiply(iq, unit(objects[..., 3:7])))
    return torch.cat((p, q), -1)


def handle_coordinates(s):
    return torch.stack((s[..., 0] + s[..., 21], s[..., 1], s[..., 2]), -1)


def mean_masked(value, mask, sample_weight=None):
    if sample_weight is not None:
        weight = sample_weight.reshape((-1,) + (1,) * (value.ndim - 1))
        value = value * weight
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def evidence(previous, current):
    """Uncertain training-only co-motion labels, NOT contact sensing or control."""
    old, new = pack(previous), pack(current)
    ro, rn = relative(old), relative(new)
    width_old, width_new = old[..., 22:24].mean(-1), new[..., 22:24].mean(-1)
    closed_old = torch.sigmoid((0.024 - width_old) / 0.0015) * torch.sigmoid((width_old - 0.012) / 0.0015)
    closed_new = torch.sigmoid((0.024 - width_new) / 0.0015) * torch.sigmoid((width_new - 0.012) / 0.0015)
    proximity = torch.exp(-(rn[..., :3].square().sum(-1).sqrt() / 0.045).pow(4))
    proximity = proximity * torch.exp(-(ro[..., :3].square().sum(-1).sqrt() / 0.045).pow(4))
    stable = torch.exp(-(rn[..., :3] - ro[..., :3]).square().sum(-1) / (0.004 ** 2)
                       - angle_square(rn[..., 3:7], ro[..., 3:7]) / (0.12 ** 2))
    tcp_motion = (new[..., :3] - old[..., :3]).square().sum(-1).sqrt()
    object_motion = torch.stack(((new[..., 7:10] - old[..., 7:10]).square().sum(-1).sqrt(),
                                 (new[..., 14:17] - old[..., 14:17]).square().sum(-1).sqrt()), -1)
    motion = (torch.minimum(tcp_motion.unsqueeze(-1), object_motion) / 0.0015).clamp(0, 1)
    blocks = (closed_old * closed_new).unsqueeze(-1) * proximity * stable * (0.55 + 0.45 * motion)
    ho, hn = handle_coordinates(old), handle_coordinates(new)
    center = hn.new_tensor([-0.14, 0.0, 0.128])
    tolerance = hn.new_tensor([0.05, 0.05, 0.04])
    near_handle = torch.exp(-((hn - center) / tolerance).square().sum(-1))
    fingers_handle = torch.sigmoid((0.012 - width_old) / 0.001) * torch.sigmoid((0.012 - width_new) / 0.001)
    x_stable = torch.exp(-(hn[..., 0] - ho[..., 0]).square() / (0.004 ** 2))
    drawer_motion = (new[..., 21] - old[..., 21]).abs()
    handle = near_handle * fingers_handle * x_stable * (0.55 + 0.45 * (drawer_motion / 0.0015).clamp(0, 1))
    scores = torch.cat((handle.unsqueeze(-1), blocks), -1)
    none = (1 - scores.max(-1, keepdim=True).values).clamp_min(0)
    probabilities = torch.cat((none, scores), -1)
    probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-6)
    block_mask = ((blocks > 0.80) & (motion > 0.25)).to(new.dtype)
    handle_mask = ((handle > 0.75) & (drawer_motion > 0.0005)).to(new.dtype)
    return probabilities, block_mask, handle_mask


class AttachmentDiffusion(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(n['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(n['std'], dtype=torch.float32))
        ids = list(range(18, 40)) + [7, 8]
        self.register_buffer('state_mean', self.obs_mean[ids].clone())
        self.register_buffer('state_scale', self.obs_scale[ids].clone())
        position_ids = [18, 19, 20, 25, 26, 27, 32, 33, 34]
        self.register_buffer('relative_scale', self.obs_scale[position_ids].max().clone())
        hs = self.obs_scale[18:21].clone()
        hs[0] = (hs[0].square() + self.obs_scale[39].square()).sqrt()
        self.register_buffer('handle_scale', hs)
        self.encoder = nn.Sequential(nn.Linear(145, 256), nn.SiLU(), nn.Linear(256, 192), nn.SiLU())
        self.identity = nn.Linear(192, 4)
        self.transform = nn.Sequential(nn.Linear(192, 128), nn.SiLU(), nn.Linear(128, 14))
        nn.init.normal_(self.transform[-1].weight, std=0.0005)
        nn.init.zeros_(self.transform[-1].bias)
        self.conditioner = nn.Sequential(nn.Linear(210, 256), nn.SiLU(), nn.Linear(256, 256))
        self.denoiser = DiffusionBackbone(256, spec['training'])
        self.rollout_init = nn.Linear(256, 192)
        self.rollout_cell = nn.GRUCell(33, 192)
        self.state_head = nn.Sequential(nn.Linear(192, 128), nn.SiLU(), nn.Linear(128, 24))
        nn.init.normal_(self.state_head[-1].weight, std=0.0005)
        nn.init.zeros_(self.state_head[-1].bias)
        self.future_identity = nn.Linear(192, 4)

    def scaled_relative(self, r):
        return torch.cat((r[..., :3] / self.relative_scale, r[..., 3:7]), -1)

    def encode(self, history):
        normalized = (history - self.obs_mean) / self.obs_scale
        states = pack(history)
        rel = relative(states)
        scaled_rel = self.scaled_relative(rel)
        hc = handle_coordinates(states) / self.handle_scale
        features = torch.cat((normalized.flatten(1), scaled_rel.flatten(1),
                              (scaled_rel[:, 1] - scaled_rel[:, 0]).flatten(1),
                              hc.flatten(1), hc[:, 1] - hc[:, 0]), -1)
        z = self.encoder(features)
        logits = self.identity(z)
        posterior = logits.softmax(-1)
        correction = self.transform(z).reshape(-1, 2, 7)
        held = torch.cat((rel[:, -1, :, :3] + correction[..., :3] * self.relative_scale,
                          canonical(rel[:, -1, :, 3:7] + correction[..., 3:7])), -1)
        weighted_held = self.scaled_relative(held) * posterior[:, 2:4, None]
        condition = self.conditioner(torch.cat((z, posterior, weighted_held.flatten(1)), -1))
        return condition, logits, held

    def forward(self, noisy_action, timestep, raw_history):
        condition, _, _ = self.encode(raw_history)
        return self.denoiser(noisy_action, timestep, condition)

    def rollout(self, condition, history, actions):
        """Slot zero ends at current state; slots 1..15 roll forward without teacher states."""
        state = normalize_poses(pack(history[:, -1]))
        hidden = torch.tanh(self.rollout_init(condition))
        states = [state]
        logits = [self.future_identity(hidden)]
        horizon = actions.shape[1]
        for j in range(1, horizon):
            time = actions.new_full((actions.shape[0], 1), j / max(1, horizon - 1))
            inputs = torch.cat((actions[:, j], (state - self.state_mean) / self.state_scale, time), -1)
            hidden = self.rollout_cell(inputs, hidden)
            state = normalize_poses(state + self.state_head(hidden) * self.state_scale)
            states.append(state)
            logits.append(self.future_identity(hidden))
        return torch.stack(states, 1), torch.stack(logits, 1)


def state_loss(model, prediction, target, valid, weight=None):
    ids = [0, 1, 2, 7, 8, 9, 14, 15, 16, 21, 22, 23]
    p, t = prediction[:, 1:], target[:, 1:]
    error = torch.nn.functional.smooth_l1_loss(p[..., ids] / model.state_scale[ids],
                                              t[..., ids] / model.state_scale[ids], reduction='none').mean(-1)
    orientation = torch.stack([angle_square(p[..., k:k+4], t[..., k:k+4]) for k in (3, 10, 17)], -1).mean(-1)
    return mean_masked(error + orientation / (math.pi ** 2), valid[:, 1:], weight)


def consistency_loss(model, prediction, block_mask, handle_mask, pair_valid, closed_command, weight=None):
    r = relative(prediction)
    translation = ((r[:, 1:, :, :3] - r[:, :-1, :, :3]) / model.relative_scale).square().mean(-1)
    rotation = angle_square(r[:, 1:, :, 3:7], r[:, :-1, :, 3:7]) / (math.pi ** 2)
    bm = block_mask[:, 1:] * pair_valid.unsqueeze(-1) * closed_command[:, 1:, None]
    block_loss = mean_masked(translation + rotation, bm, weight)
    # The drawer has a prismatic X joint: no rigid handle orientation or YZ target.
    h = handle_coordinates(prediction)[..., 0]
    x_error = ((h[:, 1:] - h[:, :-1]) / model.handle_scale[0]).square()
    hm = handle_mask[:, 1:] * pair_valid * closed_command[:, 1:]
    return block_loss + mean_masked(x_error, hm, weight)


def soft_classification(logits, labels, valid, weight=None):
    error = -(labels * logits.log_softmax(-1)).sum(-1)
    return mean_masked(error, valid, weight)


def build_model(spec):
    return AttachmentDiffusion(spec)


def compute_loss(model, batch, spec):
    history = batch['raw_obs']
    condition, identity_logits, held = model.encode(history)
    epsilon = model.denoiser(batch['noisy_action'], batch['timesteps'], condition)
    diffusion = epsilon_loss(epsilon, batch['noise'], batch['mask'])
    alpha = batch['alpha_bar'].reshape(-1, 1, 1).clamp_min(1e-6)
    x0 = (batch['noisy_action'] - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()
    # This clamp is only for a training surrogate, never a sampler replacement.
    predicted_state, predicted_logits = model.rollout(condition, history, x0.clamp(-1.5, 1.5))
    teacher_state, teacher_logits = model.rollout(condition, history, batch['encoded_action'])
    weight = batch['alpha_bar'].reshape(-1).square()
    with torch.no_grad():
        future = batch['future_obs']
        previous = torch.cat((history[:, :1], future[:, :-1]), 1)
        labels, block_mask, handle_mask = evidence(previous, future)
        current_labels, _, _ = evidence(history[:, 0], history[:, 1])
        target = normalize_poses(pack(future))
        valid = (batch['future_mask'] * batch['mask']).squeeze(-1)
        pair_valid = valid[:, 1:] * valid[:, :-1]
        closed_command = (batch['native_action'][..., 7] < 0).to(valid.dtype)
        measured_held = relative(pack(history[:, -1]))
    supervised_gt = state_loss(model, teacher_state, target, valid)
    supervised_prediction = state_loss(model, predicted_state, target, valid, weight)
    identity_loss = soft_classification(identity_logits, current_labels, torch.ones_like(valid[:, 0]))
    identity_loss = identity_loss + 0.5 * soft_classification(teacher_logits[:, 1:], labels[:, 1:], valid[:, 1:])
    identity_loss = identity_loss + 0.5 * soft_classification(predicted_logits[:, 1:], labels[:, 1:], valid[:, 1:], weight)
    transform_error = ((held[..., :3] - measured_held[..., :3]) / model.relative_scale).square().mean(-1)
    transform_error = transform_error + angle_square(held[..., 3:7], measured_held[..., 3:7]) / (math.pi ** 2)
    transform_loss = mean_masked(transform_error, current_labels[:, 2:4])
    # Train the causal transform refinement to estimate the near-future persistent
    # attachment, only when the same block remains a plausible attachment.
    future_rel = relative(target[:, 1:5])
    continuation = (labels[:, :5, 2:4] > 0.5).to(valid.dtype).cumprod(1)[:, 1:5]
    continuation = continuation * current_labels[:, None, 2:4] * valid[:, 1:5, None]
    hp = held[:, None].expand_as(future_rel)
    transform_future = ((hp[..., :3] - future_rel[..., :3]) / model.relative_scale).square().mean(-1)
    transform_future = transform_future + angle_square(hp[..., 3:7], future_rel[..., 3:7]) / (math.pi ** 2)
    transform_loss = transform_loss + mean_masked(transform_future, continuation)
    rigid_gt = consistency_loss(model, teacher_state, block_mask, handle_mask, pair_valid, closed_command)
    rigid_prediction = consistency_loss(model, predicted_state, block_mask, handle_mask, pair_valid, closed_command, weight)
    prior = (0.20 * supervised_gt + 0.10 * supervised_prediction + 0.05 * identity_loss
             + 0.02 * transform_loss + 0.05 * rigid_gt + 0.10 * rigid_prediction)
    return {'loss': diffusion + prior, 'diffusion_loss': diffusion, 'prior_loss': prior}
