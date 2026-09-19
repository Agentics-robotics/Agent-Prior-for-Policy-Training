import torch
from torch import nn
from torch.nn import functional as F
from appl.public import DiffusionBackbone, epsilon_loss


def unit_quaternion(q):
    identity = torch.zeros_like(q)
    identity[..., 0] = 1.0
    norm = q.square().sum(-1, keepdim=True).sqrt()
    return torch.where(norm > 1e-6, q / norm.clamp_min(1e-6), identity)


def projected_extents(q):
    q = unit_quaternion(q)
    w, x, y, z = q.unbind(-1)
    rx = torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)), -1)
    ry = torch.stack((2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)), -1)
    return 0.02 * torch.stack((rx.abs().sum(-1), ry.abs().sum(-1)), -1)


def goal_margins(position, quaternion, goal):
    extent = projected_extents(quaternion)
    xy = 0.06 - (position[..., :2] - goal[..., :2]).abs() - extent
    z = 0.011 - (position[..., 2:3] - goal[..., 2:3]).abs()
    return torch.cat((xy, z), -1)


def object_state(raw):
    position = torch.stack((raw[..., 25:28], raw[..., 32:35]), -2)
    quaternion = torch.stack((raw[..., 28:32], raw[..., 35:39]), -2)
    goal = torch.stack((raw[..., 41:44], raw[..., 44:47]), -2)
    return position, unit_quaternion(quaternion), goal


def observed_contract(raw):
    """Exact observed geometry only; returns red, blue, and conjunction booleans."""
    p, q, g = object_state(raw)
    achieved = (goal_margins(p, q, g) >= 0).all(-1)
    return {'red_at_goal': achieved[..., 0], 'blue_at_goal': achieved[..., 1],
            'both_at_goal': achieved.all(-1)}


def weighted_mean(value, weight):
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


class OutcomeHead(nn.Module):
    """Local action-conditioned outcome surrogate, not kinematics or a controller."""
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon
        self.a1 = nn.Linear(horizon * 9, 192)
        self.a2 = nn.Linear(192, 128)
        self.f1 = nn.Linear(384, 256)
        self.f2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, horizon * 16)
        nn.init.normal_(self.output.weight, std=0.001)
        nn.init.zeros_(self.output.bias)

    def linear(self, layer, x, frozen):
        if frozen:
            return F.linear(x, layer.weight.detach(), layer.bias.detach())
        return layer(x)

    def forward(self, action, context, mask, frozen=False):
        a = torch.cat((action * mask, mask), -1).flatten(1)
        a = F.silu(self.linear(self.a1, a, frozen))
        a = F.silu(self.linear(self.a2, a, frozen))
        h = F.silu(self.linear(self.f1, torch.cat((context, a), -1), frozen))
        h = F.silu(self.linear(self.f2, h, frozen))
        return self.linear(self.output, h, frozen).reshape(action.shape[0], self.horizon, 2, 8)


class MarginDiffusionPolicy(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = spec['normalizer']
        self.horizon = spec['training']['horizon']
        self.register_buffer('obs_mean', torch.tensor(n['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(n['std'], dtype=torch.float32))
        self.register_buffer('position_scale', torch.tensor([n['std'][25:28], n['std'][32:35]], dtype=torch.float32))
        relative_scale = torch.stack((self.obs_scale[18:21], self.obs_scale[25:28], self.obs_scale[32:35])).amax(0)
        self.register_buffer('relative_scale', relative_scale.clone())
        self.register_buffer('margin_scale', torch.tensor([0.06, 0.06, 0.011], dtype=torch.float32))
        self.register_buffer('loss_updates', torch.zeros((), dtype=torch.long))
        # 47 shared-normalized channels + 22 exact geometric channels per state.
        # Two causal states and their difference: 69 * 3.
        self.encoder = nn.Sequential(nn.Linear(207, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU())
        self.placement_mode = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 2))
        self.denoiser = DiffusionBackbone(258, spec['training'])
        self.outcome = OutcomeHead(self.horizon)

    def features(self, history):
        norm = (history - self.obs_mean) / self.obs_scale
        p, q, g = object_state(history)
        displacement = ((p - g) / self.relative_scale).flatten(-2)
        tcp_relative = ((history[..., 18:21].unsqueeze(-2) - p) / self.relative_scale).flatten(-2)
        extents = (projected_extents(q) / 0.06).flatten(-2)
        margins = (goal_margins(p, q, g) / self.margin_scale).flatten(-2)
        f = torch.cat((norm, displacement, tcp_relative, extents, margins), -1)
        return torch.cat((f.flatten(1), f[:, -1] - f[:, -2]), -1)

    def context(self, history):
        h = self.encoder(self.features(history))
        mode = self.placement_mode(h)
        return h, mode

    def predict_noise(self, noisy_action, timestep, h, mode):
        return self.denoiser(noisy_action, timestep, torch.cat((h, mode.sigmoid()), -1))

    def forward(self, noisy_action, timestep, raw_history):
        h, mode = self.context(raw_history)
        return self.predict_noise(noisy_action, timestep, h, mode)

    def predict_outcome(self, action, context, history, mask, frozen=False):
        out = self.outcome(action, context, mask, frozen=frozen)
        p, q, _ = object_state(history[:, -1])
        position = p[:, None] + out[..., :3] * self.position_scale
        quaternion = unit_quaternion(q[:, None] + out[..., 3:7])
        return position, quaternion, out[..., 7]

    def observed_contract(self, raw_history):
        return observed_contract(raw_history[:, -1])


def build_model(spec):
    return MarginDiffusionPolicy(spec)


def pose_support_losses(prediction, target_p, target_q, support_target, mask, model, sample_weight):
    p, q, support_logit = prediction
    pos_error = F.smooth_l1_loss(p / model.position_scale, target_p / model.position_scale,
                               reduction='none').mean(-1)
    # q and -q encode the same rotation.
    orient_error = torch.minimum((q - target_q).square().sum(-1),
                                 (q + target_q).square().sum(-1))
    support_error = F.binary_cross_entropy_with_logits(support_logit, support_target, reduction='none')
    weight = mask.expand_as(pos_error)
    # Do not renormalize away the diffusion-SNR reliability weight.
    pose_loss = weighted_mean((pos_error + 0.25 * orient_error) * sample_weight, weight)
    support_loss = weighted_mean(support_error * sample_weight, weight)
    return pose_loss, support_loss


def compute_loss(model, batch, spec):
    history = batch['raw_obs']
    context, mode = model.context(history)
    predicted_noise = model.predict_noise(batch['noisy_action'], batch['timesteps'], context, mode)
    diffusion_loss = epsilon_loss(predicted_noise, batch['noise'], batch['mask'])
    alpha = batch['alpha_bar'].reshape(-1, 1, 1).clamp(1e-6, 1.0)
    clean_estimate = (batch['noisy_action'] - (1 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
    # Auxiliary trust region only. The external sampler keeps its own fixed clipping.
    clean_estimate = clean_estimate.clamp(-1.25, 1.25)
    reliability = alpha.square()

    target_p, target_q, target_g = object_state(batch['future_obs'])
    target_margin = goal_margins(target_p, target_q, target_g)
    support = ((target_p[..., 2] - target_g[..., 2]).abs() <= 0.011).to(target_p.dtype)
    valid = batch['future_mask'] * batch['mask']
    # Slot zero is the state after action t-1, already in the causal history.
    valid = valid.clone()
    valid[:, 0] = 0
    near_place = ((target_margin[..., :2] >= 0).all(-1) &
                  ((target_p[..., 2] - target_g[..., 2]).abs() <= 0.05)).to(target_p.dtype)
    mode_target = (near_place * valid).sum(1) / valid.sum(1).clamp_min(1.0)
    mode_error = F.binary_cross_entropy_with_logits(mode, mode_target, reduction='none')
    mode_loss = weighted_mean(mode_error, (valid.sum(1) > 0).to(mode.dtype).expand_as(mode_error))

    clean_prediction = model.predict_outcome(batch['encoded_action'], context, history, batch['mask'])
    pose_clean, support_clean = pose_support_losses(clean_prediction, target_p, target_q, support,
                                                     valid, model, torch.ones_like(reliability))
    noisy_prediction = model.predict_outcome(clean_estimate.detach(), context, history, batch['mask'])
    pose_noisy, support_noisy = pose_support_losses(noisy_prediction, target_p, target_q, support,
                                                     valid, model, reliability)

    # Freeze surrogate parameters AND observation context for the guidance branch.
    # The only trainable path here is x0 -> epsilon -> denoiser/causal encoder.
    guided_p, guided_q, _ = model.predict_outcome(clean_estimate, context.detach(), history,
                                                 batch['mask'], frozen=True)
    predicted_margin = goal_margins(guided_p, guided_q, target_g)
    negative = F.relu(-predicted_margin / model.margin_scale)
    hinge = F.smooth_l1_loss(negative, torch.zeros_like(negative), reduction='none').mean(-1)
    # Exact future contract labels select place/support windows. There is no pressure
    # to put an airborne transport state at support height early.
    placement_window = (target_margin >= 0).all(-1).to(hinge.dtype)
    margin_weight = valid * placement_window
    margin_loss = weighted_mean(hinge * reliability, margin_weight)

    current_p, current_q, current_goal = object_state(history[:, -1])
    current_margin = goal_margins(current_p, current_q, current_goal)
    blue_already = (current_margin[:, 1] >= 0).all(-1).to(hinge.dtype)
    preserve_weight = valid[..., 0] * placement_window[..., 1] * blue_already[:, None]
    blue_drift = F.smooth_l1_loss((guided_p[:, :, 1] - current_p[:, None, 1]) / model.position_scale[1],
                                torch.zeros_like(guided_p[:, :, 1]), reduction='none').mean(-1)
    preservation_loss = weighted_mean((hinge[..., 1] + blue_drift) * reliability[..., 0], preserve_weight)

    # Warm up the supervised surrogate before using its action Jacobian.
    ramp = ((model.loss_updates.to(diffusion_loss.dtype) - 500.0) / 1500.0).clamp(0.0, 1.0)
    prior_loss = (0.10 * pose_clean + 0.025 * pose_noisy +
                  0.02 * support_clean + 0.005 * support_noisy + 0.02 * mode_loss +
                  ramp * (0.05 * margin_loss + 0.025 * preservation_loss))
    if model.training:
        with torch.no_grad():
            model.loss_updates.add_(1)
    return {'loss': diffusion_loss + prior_loss, 'diffusion_loss': diffusion_loss,
            'prior_loss': prior_loss, 'pose_loss': pose_clean,
            'support_loss': support_clean, 'mode_loss': mode_loss,
            'margin_loss': margin_loss, 'preservation_loss': preservation_loss}
