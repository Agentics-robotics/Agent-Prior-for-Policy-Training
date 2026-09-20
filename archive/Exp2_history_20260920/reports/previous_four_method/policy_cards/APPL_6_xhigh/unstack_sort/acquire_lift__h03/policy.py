import math
import torch
from torch import nn
from torch.nn import functional as F
from appl.public import DiffusionBackbone, epsilon_loss


def linear(layer, x, frozen=False):
    return F.linear(x, layer.weight.detach() if frozen else layer.weight,
                    layer.bias.detach() if frozen else layer.bias)


class Response(nn.Module):
    """Action-conditioned, temporally causal response model, not robot kinematics."""
    def __init__(self, feature_dim, horizon, width=128):
        super().__init__()
        self.context1 = nn.Linear(feature_dim, width)
        self.context2 = nn.Linear(width, width)
        self.action = nn.Linear(8, width)
        self.slot = nn.Parameter(torch.randn(horizon, width) * 0.01)
        self.convs = nn.ModuleList([nn.Conv1d(width, width, 3, dilation=d) for d in [1, 2, 4, 8]])
        self.head = nn.Linear(width, 20)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, features, action, frozen=False):
        context = linear(self.context2, F.silu(linear(self.context1, features, frozen)), frozen)
        slot = self.slot.detach() if frozen else self.slot
        x = context[:, None, :] + linear(self.action, action, frozen) + slot[None, :action.shape[1]]
        x = F.silu(x).transpose(1, 2)
        for layer in self.convs:
            w = layer.weight.detach() if frozen else layer.weight
            b = layer.bias.detach() if frozen else layer.bias
            y = F.conv1d(F.pad(x, (2 * layer.dilation[0], 0)), w, b, dilation=layer.dilation[0])
            x = (x + F.silu(y)) / math.sqrt(2.0)
        result = linear(self.head, x.transpose(1, 2), frozen)
        return result[..., :18], result[..., 18:]


class ClearanceDiffusion(nn.Module):
    def __init__(self, spec):
        super().__init__()
        config = spec.get('candidate_config', {})
        self.cfg = config
        self.guidance_strength = float(config.get('guidance_strength', 0.02))
        self.other_weight = float(config.get('other_stability_weight', 0.25))
        self.energy_weight = float(config.get('energy_weight', 0.02))
        self.clearance_margin = float(config.get('clearance_margin_m', 0.025))
        n = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(n['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(n['std'], dtype=torch.float32))
        scales = list(n['std'][18:21]) + list(n['std'][25:32]) + list(n['std'][32:39])
        scales += [n['std'][7] + n['std'][8]]
        self.register_buffer('response_scale', torch.tensor(scales, dtype=torch.float32))
        # Calibrate the supplied fixed schedule, without assuming its beta family.
        # The ratio remains correct whether EMA copies or averages these buffers.
        steps = spec['training']['denoising_train_steps']
        self.register_buffer('alpha_sum', torch.zeros(steps))
        self.register_buffer('alpha_count', torch.zeros(steps))
        self.register_buffer('response_updates', torch.zeros(()))
        self.encoder = nn.Sequential(nn.Linear(124, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU())
        self.backbone = DiffusionBackbone(256, spec['training'])
        self.responses = nn.ModuleList([Response(124, spec['training']['horizon'], w) for w in [128, 160]])

    def roles(self, history):
        now, before = history[:, -1], history[:, -2]
        tcp, red, blue = now[:, 18:21], now[:, 25:28], now[:, 32:35]
        width = now[:, 7:9].sum(-1)
        goal_dist = torch.linalg.vector_norm(red[:, :2] - now[:, 41:43], dim=-1)
        selected_blue = torch.sigmoid((0.12 - goal_dist) / 0.018)
        previous = selected_blue * torch.exp(-((tcp - red) / 0.055).square().sum(-1))
        closed = torch.sigmoid((0.053 - width) / 0.004)
        dtcp = tcp - before[:, 18:21]
        attaches = []
        for p, old in [(red, before[:, 25:28]), (blue, before[:, 32:35])]:
            near = torch.sigmoid((0.033 - torch.linalg.vector_norm(tcp - p, dim=-1)) / 0.006)
            consistent = torch.exp(-((dtcp - (p - old)) / 0.012).square().sum(-1))
            attaches.append(closed * near * consistent)
        released = selected_blue * torch.sigmoid((0.035 - red[:, 2]) / 0.006)
        released = released * torch.sigmoid((width - 0.060) / 0.005)
        released = released * torch.exp(-((tcp - red) / 0.20).square().sum(-1))
        return torch.stack([1 - selected_blue, selected_blue, previous, released, attaches[0], attaches[1]], -1)

    def features(self, history):
        norm = (history - self.obs_mean) / self.obs_scale
        now, before = history[:, -1], history[:, -2]
        tcp, red, blue = now[:, 18:21], now[:, 25:28], now[:, 32:35]
        rel = torch.cat([(tcp - red) / self.obs_scale[18:21],
                         (tcp - blue) / self.obs_scale[18:21],
                         (red - blue) / torch.maximum(self.obs_scale[25:28], self.obs_scale[32:35]),
                         (red - now[:, 41:44]) / self.obs_scale[25:28],
                         (blue - now[:, 44:47]) / self.obs_scale[32:35]], -1)
        motion = torch.cat([(now[:, i:i+3] - before[:, i:i+3]) / self.obs_scale[i:i+3]
                            for i in [18, 25, 32]], -1)
        return torch.cat([norm.flatten(1), rel, motion, self.roles(history)], -1)

    def response_base(self, history):
        s = history[:, -1]
        return torch.cat([s[:, 18:21], s[:, 25:32], s[:, 32:39], s[:, 7:9].sum(-1, keepdim=True)], -1)

    def rollout(self, history, action, frozen=False):
        feature = self.features(history)
        output = [m(feature, action, frozen) for m in self.responses]
        delta = torch.stack([r[0] for r in output], 0)
        logits = torch.stack([r[1] for r in output], 0)
        native = self.response_base(history)[None, :, None, :] + delta * self.response_scale
        return native, delta, logits

    def uncertainty(self, responses):
        diff = responses[0] - responses[1]
        pos = torch.cat([diff[..., :3], diff[..., 3:6], diff[..., 10:13]], -1)
        return pos.square().mean((1, 2)).sqrt()

    @staticmethod
    def capped_motion(displacement, tolerance=0.03):
        return 4 * torch.tanh((displacement / tolerance).square().sum(-1) / 4)

    def energy(self, history, action, responses, logits, mask=None):
        s = history[:, -1]
        role = self.roles(history).detach()
        pred = responses.mean(0)
        tcp, red, blue, width = pred[..., :3], pred[..., 3:6], pred[..., 10:13], pred[..., 17]
        dtcp = tcp - s[:, None, 18:21]
        dr = red - s[:, None, 25:28]
        db = blue - s[:, None, 32:35]
        # The attachment classifier is trained by labels, not allowed to turn off
        # a costly energy by reducing its own gate through an energy gradient.
        learned_attach = logits.sigmoid().mean(0)[:, 0].detach()
        gate_red = role[:, 0] * role[:, 4] * (0.25 + 0.75 * learned_attach[:, 0])
        gate_blue = role[:, 1] * (1 - role[:, 2]) * role[:, 5] * (0.25 + 0.75 * learned_attach[:, 1])
        source_red = torch.exp(-((s[:, 25:27] - s[:, 32:34]) / 0.060).square().sum(-1))
        source_red = source_red * torch.sigmoid((0.080 - (s[:, 27] - s[:, 34] - 0.040)) / 0.018)
        source_red = source_red * torch.sigmoid((torch.linalg.vector_norm(s[:, 25:27] - s[:, 41:43], dim=-1) - 0.12) / 0.025)
        source_blue = torch.sigmoid((0.080 - (s[:, 34] - 0.020)) / 0.018)
        source_blue = source_blue * torch.sigmoid((torch.linalg.vector_norm(s[:, 32:34] - s[:, 44:46], dim=-1) - 0.12) / 0.025)
        clearance_r = red[..., 2] - 0.020 - (s[:, None, 34] + 0.020)
        clearance_b = blue[..., 2] - 0.020
        deficit_r = 8 * torch.tanh(F.softplus((self.clearance_margin - clearance_r) / 0.010) / 8)
        deficit_b = 8 * torch.tanh(F.softplus((self.clearance_margin - clearance_b) / 0.010) / 8)
        low_r, low_b = gate_red * source_red, gate_blue * source_blue
        lateral = low_r[:, None] * deficit_r * self.capped_motion(dr[..., :2])
        lateral = lateral + low_b[:, None] * deficit_b * self.capped_motion(db[..., :2])
        stability = low_r[:, None] * self.capped_motion(db)
        stability = stability + low_b[:, None] * self.capped_motion(dr)
        # Stable pinch, not a forced absolute vertical waypoint.
        attach = gate_red[:, None] * self.capped_motion(dr - dtcp)
        attach = attach + gate_blue[:, None] * self.capped_motion(db - dtcp)
        grip = (gate_red + gate_blue)[:, None] * ((action[..., 7] + 1).square()
                    + 0.1 * ((width - s[:, None, 7:9].sum(-1)) / 0.020).square().clamp(max=4))
        # Prior-red completion: only released-object stability and empty-hand
        # retreat consistency. Never interpret red following retreat as blue lift.
        retreat_low = 8 * torch.tanh(F.softplus((0.065 - tcp[..., 2]) / 0.010) / 8)
        release = role[:, 3, None] * (0.20 * self.capped_motion(dr, 0.020)
                    + 0.05 * retreat_low * self.capped_motion(dtcp[..., :2])
                    + 0.05 * (action[..., 7] - 1).square())
        e = lateral + self.other_weight * stability + 0.10 * attach + 0.05 * grip + release
        if mask is None:
            mask = torch.ones_like(e)
        else:
            mask = mask.squeeze(-1)
        # Slot zero is the action preceding the current observation.
        temporal = (torch.arange(action.shape[1], device=action.device) > 0).to(e.dtype)
        mask = mask * temporal[None]
        return (e * mask).sum(-1) / mask.sum(-1).clamp_min(1)

    def timesteps(self, timestep, batch, device):
        t = torch.as_tensor(timestep, device=device, dtype=torch.long).reshape(-1)
        return t.expand(batch)

    def forward(self, noisy_action, timestep, raw_history):
        eps = self.backbone(noisy_action, timestep, self.encoder(self.features(raw_history)))
        if self.training or self.guidance_strength <= 0:
            return eps
        t = self.timesteps(timestep, noisy_action.shape[0], noisy_action.device)
        valid = (self.alpha_count[t] > 0) & (t <= 24)
        if not bool(valid.any()):
            return eps
        alpha = (self.alpha_sum[t] / self.alpha_count[t].clamp_min(1e-12)).clamp(1e-6, 1 - 1e-6)
        ab = alpha[:, None, None]
        original = (noisy_action - (1 - ab).sqrt() * eps) / ab.sqrt()
        # enable_grad alone is insufficient if the outer sampler uses inference_mode.
        with torch.inference_mode(False), torch.enable_grad():
            history = raw_history.detach().clone()
            a = original.detach().clone().clamp(-1, 1).requires_grad_(True)
            pred, _, logits = self.rollout(history, a, frozen=True)
            energy = self.energy(history, a, pred, logits)
            grad = torch.autograd.grad(energy.sum(), a, create_graph=False)[0]
            disagreement = self.uncertainty(pred).detach()
            confidence = torch.exp(-(disagreement / 0.015).square())
            warm = (self.response_updates.detach().clone() / 2000).clamp(0, 1)
            correction = -self.guidance_strength * warm * confidence[:, None, None] * grad
            correction = torch.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)
            rms = correction.square().mean((1, 2), keepdim=True).sqrt()
            correction = correction * (0.005 / rms.clamp_min(0.005))
            correction = correction.clamp(-0.015, 0.015)
            correction[:, 0] = 0
            proposed = (a.detach() + correction.detach()).clamp(-1, 1)
            with torch.no_grad():
                p2, _, l2 = self.rollout(history, proposed, frozen=True)
                e2 = self.energy(history, proposed, p2, l2)
                u2 = self.uncertainty(p2)
                saturation = (original.detach().clone().abs() > 1.15).float().mean((1, 2))
                accept = valid.detach().clone() & (disagreement < 0.030) & (u2 < 0.030)
                accept = accept & (e2 <= energy.detach() + 1e-7) & (saturation < 0.20)
                accept = accept & torch.isfinite(e2) & torch.isfinite(grad).all(dim=(1, 2))
                delta = (proposed - a.detach()) * accept[:, None, None]
        # Equivalent to a small gradient step on the denoised estimate, while
        # leaving the repository's DDPM posterior and action decoder unchanged.
        return eps - (ab / (1 - ab)).sqrt() * delta.detach()


def build_model(spec):
    return ClearanceDiffusion(spec)


def future_targets(model, history, future):
    base = model.response_base(history)
    # Resolve quaternion double-cover against the causal quaternion, labels only.
    rq = future[..., 28:32]
    bq = future[..., 35:39]
    rq = rq * torch.where((rq * history[:, -1, None, 28:32]).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    bq = bq * torch.where((bq * history[:, -1, None, 35:39]).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    target = torch.cat([future[..., 18:21], future[..., 25:28], rq,
                        future[..., 32:35], bq, future[..., 7:9].sum(-1, keepdim=True)], -1)
    delta = (target - base[:, None]) / model.response_scale
    width = future[..., 7:9].sum(-1)
    closed = torch.sigmoid((0.053 - width) / 0.004)
    attach = []
    for i in [25, 32]:
        rel = future[..., 18:21] - future[..., i:i+3]
        prev_rel = history[:, -1, 18:21] - history[:, -1, i:i+3]
        rel_change = rel - torch.cat([prev_rel[:, None], rel[:, :-1]], 1)
        near = torch.sigmoid((0.033 - torch.linalg.vector_norm(rel, dim=-1)) / 0.006)
        consistent = torch.exp(-(rel_change / 0.012).square().sum(-1))
        attach.append(closed * near * consistent)
    return delta, torch.stack(attach, -1)


def masked_huber(pred, target, mask):
    loss = F.smooth_l1_loss(pred, target, reduction='none')
    return (loss * mask).sum() / (mask.sum() * pred.shape[-1]).clamp_min(1)


def compute_loss(model, batch, spec):
    history = batch['raw_obs']
    action = batch['encoded_action']
    mask = batch['future_mask']
    # These buffers contain ONLY the fixed noising coefficients, never states,
    # actions, identities or segment indices.
    with torch.no_grad():
        ts = batch['timesteps'].long().reshape(-1)
        alphas = batch['alpha_bar'].reshape(-1).to(model.alpha_sum)
        model.alpha_sum.index_add_(0, ts, alphas)
        model.alpha_count.index_add_(0, ts, torch.ones_like(alphas))
        model.response_updates.add_(1)
    predicted = model(batch['noisy_action'], batch['timesteps'], history)
    diffusion = epsilon_loss(predicted, batch['noise'], batch['mask'])
    target, attachment_target = future_targets(model, history, batch['future_obs'])
    _, deltas, logits = model.rollout(history, action)
    supervised = diffusion.new_zeros(())
    for k in range(len(model.responses)):
        supervised = supervised + masked_huber(deltas[k], target, mask)
        bce = F.binary_cross_entropy_with_logits(logits[k], attachment_target, reduction='none')
        supervised = supervised + 0.10 * (bce * mask).sum() / (2 * mask.sum()).clamp_min(1)
    supervised = supervised / len(model.responses)
    ab = batch['alpha_bar'].reshape(-1, 1, 1)
    clean = ((batch['noisy_action'] - (1 - ab).sqrt() * predicted) / ab.sqrt().clamp_min(1e-6)).clamp(-1, 1)
    response, clean_delta, clean_logits = model.rollout(history, clean, frozen=True)
    reliability = ab.reshape(-1).square()
    warm = ((model.response_updates - 500) / 1500).clamp(0, 1)
    uncertainty = model.uncertainty(response).detach()
    confidence = torch.exp(-(uncertainty / 0.015).square())
    energy = model.energy(history, clean, response, clean_logits, mask)
    energy_loss = (energy * reliability * confidence).mean()
    # This frozen-model response matching teaches the diffusion model to preserve
    # demonstrated upward motion instead of minimizing energy by staying still.
    cmask = mask * reliability[:, None, None]
    # Divide by the unweighted valid count: high-noise rows must truly be downweighted.
    matching = F.smooth_l1_loss(clean_delta.mean(0), target, reduction='none')
    matching = (matching * cmask).sum() / (mask.sum() * target.shape[-1]).clamp_min(1)
    prior = 0.50 * supervised + warm * (0.05 * matching + model.energy_weight * energy_loss)
    return {'loss': diffusion + prior, 'diffusion_loss': diffusion, 'prior_loss': prior,
            'response_loss': supervised, 'clearance_energy': energy_loss,
            'response_matching': matching}
