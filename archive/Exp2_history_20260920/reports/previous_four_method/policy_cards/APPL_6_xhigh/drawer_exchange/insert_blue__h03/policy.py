import torch
from torch import nn
from appl.public import DiffusionBackbone, epsilon_loss


def mlp(din, hidden, dout):
    return nn.Sequential(nn.Linear(din, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, dout))


def unit_quaternion(q):
    return q / q.square().sum(-1, keepdim=True).clamp_min(1.0e-12).sqrt()


def extent_xy(q):
    w, x, y, z = unit_quaternion(q).unbind(-1)
    r0 = torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)), -1)
    r1 = torch.stack((2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)), -1)
    return 0.02 * torch.stack((r0.abs().sum(-1), r1.abs().sum(-1)), -1)


def margins(s):
    # Non-strict XY and strict height/drawer comparisons are in achieved().
    red_xy = s.new_tensor([0.06, 0.06]) - (s[..., 25:27] - s.new_tensor([-0.18, -0.3])).abs() - extent_xy(s[..., 28:32])
    center = torch.stack((0.19 - s[..., 39], torch.zeros_like(s[..., 39])), -1)
    blue_xy = s.new_tensor([0.172, 0.182]) - (s[..., 32:34] - center).abs() - extent_xy(s[..., 35:39])
    return torch.cat((s[..., 39:40] - 0.26, red_xy, s[..., 27:28] - 0.014, 0.031 - s[..., 27:28], blue_xy, s[..., 34:35] - 0.053, 0.074 - s[..., 34:35]), -1)


def achieved(s):
    m = margins(s)
    red = (m[..., 1:3] >= 0).all(-1) & (m[..., 3:5] > 0).all(-1)
    blue = (m[..., 5:7] >= 0).all(-1) & (m[..., 7:9] > 0).all(-1)
    return m[..., 0] > 0, red, blue


def weighted_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def weighted_mean_per_sample(x, mask):
    return (x * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)


class TransitionMember(nn.Module):
    def __init__(self, obs_mean, obs_scale):
        super().__init__()
        self.register_buffer('obs_mean', obs_mean[:41].clone())
        self.register_buffer('obs_scale', obs_scale[:41].clone())
        self.initial = mlp(153, 192, 192)
        self.cell = nn.GRUCell(49, 192)
        self.delta = mlp(192, 192, 41)
        nn.init.normal_(self.delta[-1].weight, std=0.001)
        nn.init.zeros_(self.delta[-1].bias)

    def normalize_poses(self, s):
        return torch.cat((s[..., :21], unit_quaternion(s[..., 21:25]), s[..., 25:28], unit_quaternion(s[..., 28:32]), s[..., 32:35], unit_quaternion(s[..., 35:39]), s[..., 39:41]), -1)

    def forward(self, features, current, actions):
        # Slot 0 is past action t-1; its successor s_t is already observed.
        # Only slots 1..15 affect predictions; no future state is an input.
        h = self.initial(features)
        raw = current[..., :41]
        states = [raw]
        z = (raw - self.obs_mean) / self.obs_scale
        for j in range(1, actions.shape[1]):
            h = self.cell(torch.cat((z, actions[:, j]), -1), h)
            z = z + 0.1 * self.delta(h)
            raw = self.normalize_poses(z * self.obs_scale + self.obs_mean)
            z = (raw - self.obs_mean) / self.obs_scale
            states.append(raw)
        return torch.stack(states, 1)


class LookaheadDiffusion(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(n['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(n['std'], dtype=torch.float32))
        self.register_buffer('action_min', torch.tensor(n['action_min'], dtype=torch.float32))
        self.register_buffer('action_scale', torch.tensor(n['action_scale'], dtype=torch.float32))
        self.register_buffer('training_updates', torch.zeros((), dtype=torch.long))
        self.horizon = spec['training']['horizon']
        self.encoder = nn.Sequential(nn.Linear(153, 256), nn.LayerNorm(256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU())
        # Short-horizon future-work forecast, not a demonstration clock.
        self.progress = mlp(153, 192, self.horizon * 6)
        self.support = mlp(153, 256, self.horizon * 16)
        self.members = nn.ModuleList([TransitionMember(self.obs_mean, self.obs_scale) for _ in range(3)])
        self.diffusion = DiffusionBackbone(256 + self.horizon * 6, spec['training'])
        idx = list(range(21)) + [25, 26, 27, 32, 33, 34, 39, 40]
        self.register_buffer('linear_indices', torch.tensor(idx, dtype=torch.long))
        self.register_buffer('margin_scale', torch.stack((self.obs_scale[39], self.obs_scale[25], self.obs_scale[26], self.obs_scale[27], self.obs_scale[27], self.obs_scale[32], self.obs_scale[33], self.obs_scale[34], self.obs_scale[34])))

    def features(self, history):
        norm = (history - self.obs_mean) / self.obs_scale
        s = history[:, -1]
        sr = (self.obs_scale[18:21].square() + self.obs_scale[25:28].square()).sqrt()
        sb = (self.obs_scale[18:21].square() + self.obs_scale[32:35].square()).sqrt()
        rel = torch.cat(((s[:, 18:21] - s[:, 25:28]) / sr,
                         (s[:, 18:21] - s[:, 32:35]) / sb,
                         (s[:, 25:28] - s[:, 41:44]) / self.obs_scale[25:28],
                         (s[:, 32:35] - s[:, 44:47]) / self.obs_scale[32:35]), -1)
        return torch.cat((norm.flatten(1), norm[:, 1] - norm[:, 0], rel), -1)

    def forecast(self, features):
        z = self.progress(features).reshape(-1, self.horizon, 6)
        return torch.cat((torch.nn.functional.softplus(z[..., :4]), z[..., 4:].sigmoid()), -1), z

    def behavior(self, features):
        z = self.support(features).reshape(-1, self.horizon, 16)
        return z[..., :8].tanh(), z[..., 8:].clamp(-3.0, 0.0)

    def forward(self, noisy_action, timestep, raw_history):
        f = self.features(raw_history)
        p, _ = self.forecast(f)
        # Stop the diffusion loss from changing the forecast's supervised meaning.
        condition = torch.cat((self.encoder(f), p.detach().flatten(1)), -1)
        return self.diffusion(noisy_action, timestep, condition)

    def work(self, states):
        m = margins(states) / self.margin_scale
        red_work = torch.relu(-m[..., 1:5]).sum(-1)
        tcp_scale = (self.obs_scale[18:21].square() + self.obs_scale[32:35].square()).sqrt()
        approach_work = ((states[..., 18:21] - states[..., 32:35]) / tcp_scale).square().sum(-1).clamp_min(1.0e-10).sqrt()
        # Demonstrated interior aim, distinct from the actual cavity center.
        aim = torch.stack((0.125 - states[..., 39], torch.zeros_like(states[..., 39]) + 0.075, torch.zeros_like(states[..., 39]) + 0.063), -1)
        blue_work = ((states[..., 32:35] - aim) / self.obs_scale[32:35]).square().sum(-1).clamp_min(1.0e-10).sqrt()
        blue_deficit = torch.relu(-m[..., 5:9]).sum(-1)
        return torch.stack((red_work, approach_work, blue_work, blue_deficit), -1)

    def future_labels(self, states):
        _, red, _ = achieved(states)
        # Relevance label only, not a completion or handoff threshold.
        near = ((states[..., 32] - (0.125 - states[..., 39])).abs() < 0.10) & ((states[..., 33] - 0.075).abs() < 0.10) & (states[..., 34] > 0.04) & (states[..., 34] < 0.18) & red
        return self.work(states), torch.stack((red, near), -1).to(states.dtype)

    def transition_error(self, predicted, targets, mask):
        idx = self.linear_indices
        scale = self.obs_scale[idx]
        state_error = ((predicted[..., idx] - targets[..., idx]) / scale).square().mean(-1)
        pose_idx = [18, 19, 20, 25, 26, 27, 32, 33, 34, 39, 40]
        pose_error = ((predicted[..., pose_idx] - targets[..., pose_idx]) / self.obs_scale[pose_idx]).square().mean(-1)
        rotation_error = torch.zeros_like(state_error)
        for start in (21, 28, 35):
            qp = unit_quaternion(predicted[..., start:start+4])
            qt = unit_quaternion(targets[..., start:start+4])
            # sin^2(theta/2), a sign-invariant smooth geodesic surrogate.
            rotation_error = rotation_error + (1.0 - (qp * qt).sum(-1).square()).clamp_min(0)
        margin_error = ((margins(predicted) - margins(targets)) / self.margin_scale).square().mean(-1)
        fit = weighted_mean(state_error + 3.0 * pose_error + 0.2 * rotation_error + margin_error, mask)
        p_delta = predicted[:, 1:, idx] - predicted[:, :-1, idx]
        t_delta = targets[:, 1:, idx] - targets[:, :-1, idx]
        inc = ((p_delta - t_delta) / scale).square().mean(-1)
        pair = mask[:, 1:] * mask[:, :-1]
        # The current state at slot 0 is observed even if its past action is padded.
        pair = torch.cat((mask[:, 1:2], pair[:, 1:]), 1)
        return fit + 0.5 * weighted_mean(inc, pair)

    def score_candidates(self, candidates, history, features, forecast, mean, logstd, valid):
        # No-gradient training teacher only; the deployment forward does not call it.
        b, k, h, a = candidates.shape
        feats = features[:, None].expand(-1, k, -1).reshape(b*k, -1)
        current = history[:, -1, None].expand(-1, k, -1).reshape(b*k, -1)
        flat = candidates.reshape(b*k, h, a)
        ensemble = torch.stack([member(feats, current, flat).reshape(b, k, h, 41) for member in self.members], 0)
        m = margins(ensemble) / self.margin_scale
        work = self.work(ensemble)
        drawer, red, blue = achieved(history[:, -1])
        success = drawer & red & blue
        now = margins(history[:, -1]) / self.margin_scale
        # Preserve existing margins up to modest interior buffers; do not demand
        # a deeper final pose, release, dwell, or exact terminal joint posture.
        buffer = self.margin_scale.new_tensor([0.01, 0.005, 0.005, 0.001, 0.001, 0.005, 0.005, 0.001, 0.001]) / self.margin_scale
        retain_to = torch.minimum(now.clamp_min(0), buffer)
        retention = torch.relu(retain_to[None, :, None, None, :] - m).square()
        keep = torch.cat((drawer[:, None], red[:, None].expand(-1, 4), blue[:, None].expand(-1, 4)), -1).to(m.dtype)
        retention = (retention * keep[None, :, None, None, :]).sum(-1)
        reference = forecast[:, None, :, :4]
        phase_error = torch.nn.functional.smooth_l1_loss(work, reference[None].expand_as(work), reduction='none').mean(-1)
        phase_error = phase_error * (~success)[None, :, None, None]
        near = forecast[:, None, :, 5]
        placement = torch.relu(-m[..., 5:9]).square().sum(-1) * near[None] * (~success)[None, :, None, None]
        per_step = 12.0 * retention + phase_error + 0.25 * placement
        v = valid[:, None, :]
        risk_each = (per_step * v[None]).sum(-1) / v.sum(-1).clamp_min(1)[None]
        risk = 0.5 * risk_each.mean(0) + 0.5 * risk_each.max(0).values
        spatial_idx = [18, 19, 20, 25, 26, 27, 32, 33, 34, 39]
        spread = (ensemble[..., spatial_idx] / self.obs_scale[spatial_idx]).var(0, unbiased=False).mean(-1)
        uncertainty = (spread * v).sum(-1) / v.sum(-1).clamp_min(1)
        deviation = ((candidates - mean[:, None]) / logstd.exp()[:, None]).square().mean(-1)
        support_cost = (deviation * v).sum(-1) / v.sum(-1).clamp_min(1)
        return risk + 2.0 * uncertainty + 0.03 * support_cost, uncertainty


def build_model(spec):
    return LookaheadDiffusion(spec)


def compute_loss(model, batch, spec):
    history = batch['raw_obs']
    predicted_noise = model(batch['noisy_action'], batch['timesteps'], history)
    diffusion_loss = epsilon_loss(predicted_noise, batch['noise'], batch['mask'])
    features = model.features(history)
    forecast, forecast_logits = model.forecast(features)
    mean, logstd = model.behavior(features)
    action_mask = batch['mask'].squeeze(-1)
    future_mask = batch['future_mask'].squeeze(-1) * action_mask
    # Slot 0 is already observed, not a consequence of a new proposed action.
    valid = torch.cat((torch.zeros_like(future_mask[:, :1]), future_mask[:, 1:]), 1)
    future = batch['future_obs'][..., :41]
    targets = torch.cat((history[:, -1:, :41], future[:, 1:]), 1)
    target_work, target_gates = model.future_labels(targets)
    progress_fit = torch.nn.functional.smooth_l1_loss(forecast[..., :4], target_work, reduction='none').mean(-1)
    gate_fit = torch.nn.functional.binary_cross_entropy_with_logits(forecast_logits[..., 4:], target_gates, reduction='none').mean(-1)
    progress_loss = weighted_mean(progress_fit + 0.25 * gate_fit, valid)
    behavior_error = (0.5 * ((batch['encoded_action'] - mean) / logstd.exp()).square() + logstd + 3.0).mean(-1)
    support_loss = weighted_mean(behavior_error, action_mask)
    dynamics_loss = predicted_noise.sum() * 0.0
    for member in model.members:
        predicted_states = member(features, history[:, -1], batch['encoded_action'])
        bootstrap = (torch.rand((history.shape[0], 1), device=history.device) < 0.8).to(valid.dtype)
        dynamics_loss = dynamics_loss + model.transition_error(predicted_states, targets, valid * bootstrap) / len(model.members)

    alpha = batch['alpha_bar'].reshape(-1, 1, 1).clamp_min(1.0e-6)
    clean = (batch['noisy_action'] - (1.0 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
    # Warm up only the teacher's influence, not its computation. Thus interface
    # checks also exercise candidate scoring, masking and selection numerics.
    ramp = ((model.training_updates.to(clean.dtype) - 2000.0) / 4000.0).clamp(0, 1)
    with torch.no_grad():
        base = clean.detach().clamp(-1, 1)
        noise = torch.randn_like(base).transpose(1, 2)
        noise = torch.nn.functional.avg_pool1d(noise, kernel_size=5, stride=1, padding=2).transpose(1, 2)
        perturb = 0.025 * noise.clamp(-1, 1)
        step = (mean.detach() - base).clamp(-0.025, 0.025)
        candidates = torch.stack((base, base + perturb, base - perturb, base + step, base - step), 1).clamp(-1, 1)
        scores, uncertainty = model.score_candidates(candidates, history, features.detach(), forecast.detach(), mean.detach(), logstd.detach(), valid)
        best = scores.argmin(1)
        row = torch.arange(base.shape[0], device=base.device)
        target = candidates[row, best]
        improvement = scores[:, 0] - scores[row, best]
        proximity = weighted_mean_per_sample((base - batch['encoded_action']).square().mean(-1), valid)
        trust = (alpha[:, 0, 0] >= 0.5) & (proximity < 0.0144) & (uncertainty[row, best] < 0.0025) & (improvement > 1.0e-5)
        sample_weight = trust.to(clean.dtype) * alpha[:, 0, 0] * (improvement / 0.01).clamp(0, 1)
    # Detached selection -> differentiable x0 -> U-Net/causal encoder. The
    # critic cannot improve this loss by inventing favorable counterfactual states.
    fit = (clean - target).square().mean(-1)
    ranking_loss = weighted_mean(fit * sample_weight[:, None], valid)
    prior_loss = 0.3 * dynamics_loss + 0.1 * progress_loss + 0.02 * support_loss + 0.5 * ramp * ranking_loss
    loss = diffusion_loss + prior_loss
    if model.training:
        model.training_updates.add_(1)
    return {'loss': loss, 'diffusion_loss': diffusion_loss, 'prior_loss': prior_loss,
            'dynamics_loss': dynamics_loss, 'progress_loss': progress_loss,
            'support_loss': support_loss, 'ranking_loss': ranking_loss}
