import torch
from torch import nn
from appl.public import DiffusionBackbone, epsilon_loss


class ContactEventDiffusion(nn.Module):
    """Two-observation, duration-augmented phase filter and action denoiser."""
    def __init__(self, spec):
        super().__init__()
        self.horizon = int(spec['training']['horizon'])
        self.phases = 8
        self.ages = 16
        norm = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(norm['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(norm['std'], dtype=torch.float32))
        # Relative positions retain the shared full-demonstration coordinate scales.
        scales = torch.tensor(norm['std'], dtype=torch.float32)
        rel_scale = torch.cat((torch.sqrt(scales[18:21].square() + scales[25:28].square()),
                               torch.sqrt(scales[18:21].square() + scales[32:35].square()),
                               torch.sqrt(scales[18:21].square() + scales[39].square())))
        self.register_buffer('relative_scale', rel_scale)
        self.register_buffer('drawer_origin', torch.tensor([0.19, 0.0, 0.035]))
        self.register_buffer('age_values', torch.linspace(0.0, 1.0, self.ages))
        terminal = torch.ones(1, self.phases, 1)
        terminal[:, -1, :] = 0.0
        self.register_buffer('nonterminal', terminal)
        # 47 normalized states, 47 causal differences, 9 relative coordinates,
        # and one flag indicating whether this is a distinct second observation.
        self.token = nn.Sequential(nn.Linear(104, 128), nn.Mish(), nn.Linear(128, 128), nn.LayerNorm(128))
        self.recurrent = nn.GRUCell(128, 128)
        self.emission = nn.Sequential(nn.Linear(128, 64), nn.Mish(), nn.Linear(64, self.phases))
        self.initial_age = nn.Linear(128, self.phases * self.ages)
        self.hazard = nn.Linear(128, self.phases)
        nn.init.zeros_(self.hazard.weight)
        nn.init.constant_(self.hazard.bias, -3.0)
        self.age_bias = nn.Parameter(torch.linspace(-1.0, 1.0, self.ages).repeat(self.phases, 1))
        # Denoising has a geometric/proprioceptive path as well as a contact path.
        self.context = nn.Sequential(nn.Linear(128 + 128 + 47 + 47 + 9, 256), nn.Mish(),
                                     nn.Linear(256, 256), nn.LayerNorm(256))
        self.adapters = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 32), nn.Mish(), nn.Linear(32, 64))
            for _ in range(self.phases)
        ])
        self.backbone = DiffusionBackbone(256 + 64 + 8 + 8, spec['training'])
        # Ordered future phase prediction is an auxiliary training task, not an
        # inference input and not a replacement for action diffusion.
        self.future_hazard = nn.Sequential(nn.Linear(128 + 8 + 8, 128), nn.Mish(),
                                          nn.Linear(128, (self.horizon - 1) * self.phases))
        nn.init.normal_(self.future_hazard[-1].weight, std=0.005)
        nn.init.zeros_(self.future_hazard[-1].bias)

    def relative(self, raw):
        tcp = raw[..., 18:21]
        d = raw[..., 39:40]
        drawer = self.drawer_origin + torch.cat((-d, torch.zeros_like(d), torch.zeros_like(d)), dim=-1)
        return torch.cat((raw[..., 25:28] - tcp, raw[..., 32:35] - tcp, drawer - tcp), dim=-1) / self.relative_scale

    def transition(self, joint, base_hazard):
        """Stay/age or advance exactly one phase/reset age; last age is censored."""
        hazard = torch.sigmoid(base_hazard.unsqueeze(-1) + self.age_bias) * self.nonterminal
        stay = joint * (1.0 - hazard)
        # Ages 0..14 advance; the final bin represents 15 or more observations.
        aged = torch.cat((torch.zeros_like(stay[:, :, :1]), stay[:, :, :-2],
                          stay[:, :, -2:].sum(dim=-1, keepdim=True)), dim=-1)
        advance = (joint * hazard).sum(dim=-1)
        incoming = torch.cat((torch.zeros_like(advance[:, :1]), advance[:, :-1]), dim=-1)
        reset = torch.cat((incoming.unsqueeze(-1), torch.zeros_like(joint[:, :, 1:])), dim=-1)
        return aged + reset

    def encode(self, raw):
        z = (raw - self.obs_mean) / self.obs_scale
        rel = self.relative(raw)
        delta = z[:, 1] - z[:, 0]
        # The interface has no explicit history mask. Exact duplicated padding
        # is treated as an uninformative second observation, also during pauses.
        valid = ((raw[:, 1] - raw[:, 0]).abs().amax(dim=-1, keepdim=True) > 1e-8).to(raw.dtype)
        zero_delta = torch.zeros_like(delta)
        t0 = self.token(torch.cat((z[:, 0], zero_delta, rel[:, 0], torch.zeros_like(valid)), dim=-1))
        h0 = self.recurrent(t0, torch.zeros(raw.shape[0], 128, device=raw.device, dtype=raw.dtype))
        t1 = self.token(torch.cat((z[:, 1], delta, rel[:, 1], valid), dim=-1))
        h1_update = self.recurrent(t1, h0)
        h1 = valid * h1_update + (1.0 - valid) * h0
        p0 = torch.softmax(self.emission(h0), dim=-1)
        age0 = torch.softmax(self.initial_age(h0).reshape(-1, self.phases, self.ages), dim=-1)
        joint0 = p0.unsqueeze(-1) * age0
        base = self.hazard(h1)
        proposed = self.transition(joint0, base)
        likelihood = torch.softmax(self.emission(h1), dim=-1)
        joint1 = proposed * likelihood.unsqueeze(-1)
        joint1 = joint1 / joint1.sum(dim=(1, 2), keepdim=True).clamp_min(1e-8)
        joint1 = valid.unsqueeze(-1) * joint1 + (1.0 - valid.unsqueeze(-1)) * joint0
        p1 = joint1.sum(dim=-1)
        dwell = (joint1 * self.age_values).sum(dim=-1)
        features = self.context(torch.cat((h1, h1 - h0, z[:, 1], delta, rel[:, 1]), dim=-1))
        experts = torch.stack([adapter(h1) for adapter in self.adapters], dim=1)
        adapted = (experts * p1.unsqueeze(-1)).sum(dim=1)
        condition = torch.cat((features, adapted, p1, dwell), dim=-1)
        return condition, {'h': h1, 'p0': p0, 'p1': p1, 'joint': joint1,
                           'dwell': dwell, 'proposed': proposed.sum(dim=-1),
                           'base': base, 'valid': valid.squeeze(-1)}

    def predict(self, noisy_action, timestep, raw_history):
        condition, latent = self.encode(raw_history)
        return self.backbone(noisy_action, timestep, condition), latent

    def forward(self, noisy_action, timestep, raw_history):
        return self.predict(noisy_action, timestep, raw_history)[0]

    def forecast_phases(self, latent):
        offsets = self.future_hazard(torch.cat((latent['h'], latent['p1'], latent['dwell']), dim=-1))
        offsets = offsets.reshape(-1, self.horizon - 1, self.phases)
        joint = latent['joint']
        probabilities = [joint.sum(dim=-1)]
        for j in range(self.horizon - 1):
            joint = self.transition(joint, latent['base'] + offsets[:, j])
            probabilities.append(joint.sum(dim=-1))
        return torch.stack(probabilities, dim=1)


def build_model(spec):
    return ContactEventDiffusion(spec)


@torch.no_grad()
def phase_targets(state, next_state, next_valid):
    """Soft, overlapping geometric/contact-motion proxies, never forward inputs.

    Labels distinguish observed finger aperture from the commanded aperture.
    Metre/second thresholds below are physical proxy-label tolerances, not
    fitted observation/action normalization scales or execution constraints.
    """
    s = torch.sigmoid
    tcp = state[..., 18:21]
    red = state[..., 25:28]
    d = state[..., 39]
    width = state[..., 7:9].mean(dim=-1)
    next_width = next_state[..., 7:9].mean(dim=-1)
    v_tcp = (next_state[..., 18:21] - tcp) / 0.05
    v_red = (next_state[..., 25:28] - red) / 0.05
    f_velocity = torch.where(next_valid > 0.5, (next_width - width) / 0.05,
                             state[..., 16:18].mean(dim=-1))
    opened = s((width - 0.028) / 0.003)
    opening = s((f_velocity - 0.025) / 0.008)
    open_evidence = 1.0 - (1.0 - opened) * (1.0 - opening)
    late = s((d - 0.275) / 0.008)
    # Demonstrated handle-contact TCP locus, NOT an instrumented handle pose.
    handle_distance = ((tcp[..., 0] + 0.140 + d) / 0.055).square()
    handle_distance = handle_distance + (tcp[..., 1] / 0.050).square()
    handle_distance = handle_distance + ((tcp[..., 2] - 0.128) / 0.040).square()
    near_handle = torch.exp(-handle_distance)
    post_handle = late * (1.0 - near_handle * (1.0 - open_evidence))
    handle_closed = s((0.014 - width) / 0.002)
    pulling = 1.0 - (1.0 - s((d - 0.022) / 0.006)) * (1.0 - s((state[..., 40] - 0.009) / 0.003))
    pull = handle_closed * pulling
    closure = near_handle * (1.0 - pull)
    pre = 1.0 - post_handle
    release = post_handle * near_handle
    travel = post_handle * (1.0 - near_handle)
    red_side = s((tcp[..., 0] - red[..., 0] + 0.180) / 0.025)
    distance = torch.linalg.vector_norm(tcp - red, dim=-1)
    red_close = s((0.045 - distance) / 0.007)
    red_grip = torch.exp(-((width - 0.01825) / 0.006).square())
    motion_match = torch.exp(-((v_red - v_tcp) / 0.040).square().sum(dim=-1))
    rising = s((torch.minimum(v_red[..., 2], v_tcp[..., 2]) - 0.020) / 0.004)
    co_lift = rising * motion_match * next_valid
    height = s((red[..., 2] - 0.076) / 0.003)
    lift = red_grip * (1.0 - (1.0 - height) * (1.0 - co_lift))
    targets = torch.stack((pre * (1.0 - pull) * (1.0 - near_handle),
                           pre * closure,
                           pre * pull,
                           release,
                           travel * (1.0 - red_side),
                           travel * red_side * (1.0 - red_close),
                           travel * red_side * red_close * (1.0 - lift),
                           travel * red_side * red_close * lift), dim=-1)
    # Small label smoothing: these are successful-data proxies, not contact truth.
    targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return 0.99 * targets + 0.01 / 8.0


def cross_entropy(probability, target):
    return -(target * probability.clamp_min(1e-7).log()).sum(dim=-1)


def weighted_mean(value, weight):
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def compute_loss(model, batch, spec):
    raw = batch['raw_obs']
    # Train also on truncated takeover histories; no persistent/private latent.
    if model.training:
        truncate = torch.rand(raw.shape[0], 1, 1, device=raw.device) < 0.15
        raw = torch.cat((torch.where(truncate, raw[:, 1:2], raw[:, 0:1]), raw[:, 1:2]), dim=1)
    predicted_noise, latent = model.predict(batch['noisy_action'], batch['timesteps'], raw)
    diffusion_loss = epsilon_loss(predicted_noise, batch['noise'], batch['mask'])
    future = batch['future_obs']
    future_valid = batch['future_mask'].squeeze(-1)
    with torch.no_grad():
        # At t, slot 0 is state t (after action t-1); slot 1 is state t+1.
        q0 = phase_targets(raw[:, 0], raw[:, 1], latent['valid'])
        q1 = phase_targets(raw[:, 1], future[:, 1], future_valid[:, 1])
        following = torch.cat((future[:, 1:], future[:, -1:]), dim=1)
        following_valid = torch.cat((future_valid[:, 1:], torch.zeros_like(future_valid[:, :1])), dim=1)
        following_valid = following_valid * future_valid
        q_future = phase_targets(future, following, following_valid)
        # Suppress anti-chatter pressure where proxy labels indicate an event.
        stable = (1.0 - 0.5 * (q0 - q1).abs().sum(dim=-1)).clamp(0.0, 1.0).square()
    phase_loss = (0.25 * cross_entropy(latent['p0'], q0) +
                  0.75 * cross_entropy(latent['p1'], q1)).mean()
    forecast = model.forecast_phases(latent)
    forecast_loss = weighted_mean(cross_entropy(forecast, q_future), future_valid)
    p0 = latent['p0'].clamp_min(1e-7)
    p1 = latent['p1'].clamp_min(1e-7)
    middle = 0.5 * (p0 + p1)
    js = 0.5 * ((p0 * (p0.log() - middle.log())).sum(dim=-1) +
                (p1 * (p1.log() - middle.log())).sum(dim=-1))
    dwell_loss = weighted_mean(js, stable * latent['valid'])
    transition_kl = (p1 * (p1.log() - latent['proposed'].clamp_min(1e-7).log())).sum(dim=-1)
    transition_loss = weighted_mean(transition_kl, latent['valid'])
    prior_loss = 0.12 * phase_loss + 0.05 * forecast_loss + 0.02 * dwell_loss + 0.01 * transition_loss
    loss = diffusion_loss + prior_loss
    return {'loss': loss, 'diffusion_loss': diffusion_loss, 'prior_loss': prior_loss,
            'phase_loss': phase_loss, 'forecast_loss': forecast_loss,
            'dwell_loss': dwell_loss, 'transition_loss': transition_loss}
