import math
import torch
from torch import nn
from torch.nn import functional as F
from appl.public import DiffusionBackbone


N_EVENTS = 9
LOOKAHEAD = 14


def event_labels(state, command):
    """Weak TRAINING labels, never called by the deployed forward path.

    Commands label intention at the observation before that command. Geometry
    separates same-command events; thresholds are annotation conventions only.
    """
    tcp = state[..., 18:21]
    red = state[..., 25:28]
    blue = state[..., 32:35]
    bg = state[..., 44:47]
    rg = state[..., 41:44]
    dr = (tcp - red).square().sum(-1)
    db = (tcp - blue).square().sum(-1)
    near_goal = (blue[..., :2] - bg[..., :2]).square().sum(-1) < 0.045 ** 2
    supported = (blue[..., 2] - bg[..., 2]).abs() < 0.02
    red_held_side = dr < db
    open_command = command > 0
    blue_aligned = (tcp[..., :2] - blue[..., :2]).square().sum(-1) < 0.06 ** 2
    red_aligned = (tcp[..., :2] - red[..., :2]).square().sum(-1) < 0.06 ** 2
    blue_transit = ((blue[..., 1] < rg[..., 1] - 0.045) |
                    (blue[..., 2] > bg[..., 2] + 0.22))
    # Closed blue: acquisition/lift, elevated transport, or goal descent/setdown.
    closed_blue = torch.where(near_goal, 5, torch.where(blue_transit, 4, 3))
    # Red can be held at entry or reacquired at exit, distinguished by blue pose.
    closed_red = torch.where(near_goal & supported, 8, 0)
    closed_label = torch.where(red_held_side, closed_red, closed_blue)
    open_incoming = torch.where(blue_aligned, 2, 1)
    open_outgoing = torch.where(red_aligned, 7, 6)
    open_label = torch.where(near_goal & supported, open_outgoing, open_incoming)
    return torch.where(open_command, open_label, closed_label).long()


class EventDiffusionPolicy(nn.Module):
    def __init__(self, spec):
        super().__init__()
        norm = spec['normalizer']
        mean = torch.tensor(norm['mean'], dtype=torch.float32)
        scale = torch.tensor(norm['std'], dtype=torch.float32)
        self.register_buffer('obs_mean', mean)
        self.register_buffer('obs_scale', scale)
        # Derived exclusively from the common full-demonstration normalizer.
        geom_scale = torch.stack((scale[18:21], scale[25:28], scale[32:35])).amax(0)
        self.register_buffer('geom_scale', geom_scale)
        self.frame_encoder = nn.Sequential(nn.Linear(77, 192), nn.Mish(),
                                           nn.Linear(192, 128), nn.LayerNorm(128), nn.Mish())
        self.recurrent_update = nn.GRUCell(128, 128)
        self.emission = nn.Linear(128, N_EVENTS)
        self.hazard = nn.Sequential(nn.Linear(128, 128), nn.Mish(),
                                    nn.Linear(128, N_EVENTS * LOOKAHEAD))
        nn.init.constant_(self.hazard[-1].bias, -3.0)
        self.readiness = nn.Linear(128, N_EVENTS)
        self.event_embedding = nn.Parameter(torch.randn(N_EVENTS, 32) * 0.05)
        self.conditioner = nn.Sequential(nn.Linear(269, 256), nn.Mish(),
                                         nn.Linear(256, 256))
        self.backbone = DiffusionBackbone(256, spec['training'])
        order = torch.arange(N_EVENTS)
        forbidden = ((order[None, :] < order[:, None]) |
                     (order[None, :] > order[:, None] + 1)).float()
        self.register_buffer('forbidden_transition', forbidden)
        active = torch.ones(N_EVENTS)
        active[-1] = 0.0
        self.register_buffer('nonterminal', active)

    def observation_features(self, raw_history):
        z = (raw_history - self.obs_mean) / self.obs_scale
        tcp = raw_history[..., 18:21]
        red = raw_history[..., 25:28]
        blue = raw_history[..., 32:35]
        rel = torch.cat(((red - tcp) / self.geom_scale,
                         (blue - tcp) / self.geom_scale,
                         (raw_history[..., 41:44] - red) / self.geom_scale,
                         (raw_history[..., 44:47] - blue) / self.geom_scale), dim=-1)
        positions = torch.cat((z[..., :9], tcp / self.geom_scale,
                               red / self.geom_scale, blue / self.geom_scale), dim=-1)
        # Actual observations are 0.05 seconds apart. Saturation applies only to
        # the engineered derivative feature, never to measured qvel or actions.
        velocity = torch.tanh((positions[:, 1] - positions[:, 0]) / 0.05)
        derivatives = torch.stack((torch.zeros_like(velocity), velocity), dim=1)
        return z, torch.cat((z, rel, derivatives), dim=-1)

    def encode(self, raw_history):
        z, features = self.observation_features(raw_history)
        frames = self.frame_encoder(features)
        h0 = frames[:, 0]
        h1 = self.recurrent_update(frames[:, 1], h0)
        p0 = self.emission(h0).softmax(-1)
        hazards0 = self.hazard(h0).view(-1, N_EVENTS, LOOKAHEAD).sigmoid()
        flow = p0 * hazards0[:, :, 0] * self.nonterminal
        ordered_prior = p0 - flow + F.pad(flow[:, :-1], (1, 0))
        # A reset floor and tempered evidence preserve posterior reinitialization
        # and permit nonmonotone correction. This is not a hard event automaton.
        prior = 0.90 * ordered_prior + 0.10 / N_EVENTS
        logits = self.emission(h1)
        logp = F.log_softmax(logits + 0.5 * prior.clamp_min(1e-6).log(), dim=-1)
        p = logp.exp()
        hazard_logits = self.hazard(h1).view(-1, N_EVENTS, LOOKAHEAD)
        hazards = hazard_logits.sigmoid()
        cdf = 1.0 - (1.0 - hazards).cumprod(-1)
        expected_cdf = (p[:, :, None] * cdf).sum(1)
        readiness_by_event = self.readiness(h1).sigmoid()
        readiness = (p * readiness_by_event).sum(-1, keepdim=True)
        entropy = -(p * logp).sum(-1, keepdim=True) / math.log(N_EVENTS)
        summary = expected_cdf[:, [0, 3, 7, 13]]
        context = torch.cat((z.flatten(1), h1, p @ self.event_embedding,
                             p, summary, readiness, entropy), dim=-1)
        condition = self.conditioner(context)
        return {'condition': condition, 'probability': p, 'log_probability': logp,
                'previous_probability': p0, 'hazard_logits': hazard_logits,
                'readiness_by_event': readiness_by_event,
                'readiness': readiness, 'entropy': entropy,
                'advance_cdf': expected_cdf}

    def event_belief(self, raw_history):
        """Optional causal diagnostic; not a required framework output."""
        out = self.encode(raw_history)
        return {'probability': out['probability'], 'entropy': out['entropy'],
                'advance_cdf': out['advance_cdf'], 'readiness': out['readiness']}

    def forward(self, noisy_action, timestep, raw_history):
        condition = self.encode(raw_history)['condition']
        return self.backbone(noisy_action, timestep, condition)


def build_model(spec):
    return EventDiffusionPolicy(spec)


def weighted_mean(value, weight):
    return (value * weight).sum() / weight.sum().clamp_min(1e-6)


def mode_cross_entropy(log_probability, label):
    target = 0.97 * F.one_hot(label, N_EVENTS).to(log_probability.dtype) + 0.03 / N_EVENTS
    return -(target * log_probability).sum(-1)


def compute_loss(model, batch, spec):
    history = batch['raw_obs']
    native = batch['native_action']
    mask = batch['mask'][..., 0]
    future = batch['future_obs']
    future_mask = batch['future_mask'][..., 0]
    encoded = model.encode(history)
    prediction = model.backbone(batch['noisy_action'], batch['timesteps'], encoded['condition'])
    with torch.no_grad():
        # Slot 0 is the action at t-1; slot 1 is the action at current state t.
        previous_label = event_labels(history[:, 0], native[:, 0, 7])
        label = event_labels(history[:, 1], native[:, 1, 7])
        next_label = event_labels(future[:, 1], native[:, 2, 7])
        # future_obs[j] is state t+j, paired with action[j+1], not action[j].
        future_labels = event_labels(future[:, :15], native[:, 1:16, 7])
        valid_future = future_mask[:, :15] * mask[:, 1:16]
        valid_current = mask[:, 1]
        counts = (F.one_hot(label, N_EVENTS).float() * valid_current[:, None]).sum(0)
        balance = (valid_current.sum() / (N_EVENTS * counts.clamp_min(1.0))).clamp(0.33, 3.0)
        sample_weight = balance[label]

    per_action = (prediction - batch['noise']).square().mean(-1)
    # Importance weighting changes event emphasis, not action timing or scales.
    diffusion_loss = weighted_mean(per_action, mask * sample_weight[:, None])

    next_history = torch.stack((history[:, 1], future[:, 1]), dim=1)
    next_encoded = model.encode(next_history)
    w = sample_weight * valid_current
    wp = sample_weight * mask[:, 0]
    wn = sample_weight * future_mask[:, 1] * mask[:, 2]
    current_ce = weighted_mean(mode_cross_entropy(encoded['log_probability'], label), w)
    previous_ce = weighted_mean(mode_cross_entropy(encoded['previous_probability'].clamp_min(1e-8).log(), previous_label), wp)
    next_ce = weighted_mean(mode_cross_entropy(next_encoded['log_probability'], next_label), wn)
    mode_loss = 0.5 * current_ce + 0.25 * previous_ce + 0.25 * next_ce

    with torch.no_grad():
        advance = (future_labels[:, 1:] > label[:, None]).float()
        at_risk = torch.cumprod(torch.cat((torch.ones_like(advance[:, :1]),
                                          1.0 - advance[:, :-1]), dim=1), dim=1)
        at_risk = at_risk * valid_future[:, 1:]
        lag = torch.arange(1, LOOKAHEAD + 1, device=history.device, dtype=history.dtype)
        proximity_target = (advance * valid_future[:, 1:] * (1.0 - lag / 15.0)).amax(-1)
        # Absence of a boundary is informative only for complete lookahead;
        # a visible boundary remains informative near right-censored slice ends.
        readiness_valid = ((valid_future[:, 1:].amin(-1) > 0) |
                           (proximity_target > 0)).float()
    index = label[:, None, None].expand(-1, 1, LOOKAHEAD)
    selected_hazard = encoded['hazard_logits'].gather(1, index).squeeze(1)
    survival_ce = F.binary_cross_entropy_with_logits(selected_hazard, advance, reduction='none')
    hazard_loss = weighted_mean(survival_ce, at_risk * w[:, None])
    selected_readiness = encoded['readiness_by_event'].gather(1, label[:, None]).squeeze(1)
    progress_loss = weighted_mean((selected_readiness - proximity_target).square(), w * readiness_valid)

    def illegal_mass(left, right):
        return ((left @ model.forbidden_transition) * right).sum(-1)

    order_loss = 0.5 * (weighted_mean(illegal_mass(encoded['previous_probability'],
                                                  encoded['probability']), wp * valid_current) +
                        weighted_mean(illegal_mass(encoded['probability'],
                                                  next_encoded['probability']), wn * valid_current))
    prior_loss = 0.15 * mode_loss + 0.08 * hazard_loss + 0.05 * progress_loss + 0.02 * order_loss
    return {'loss': diffusion_loss + prior_loss, 'diffusion_loss': diffusion_loss,
            'prior_loss': prior_loss, 'mode_loss': mode_loss, 'hazard_loss': hazard_loss,
            'progress_loss': progress_loss, 'order_loss': order_loss}
