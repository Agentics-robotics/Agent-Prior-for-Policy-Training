import torch
from appl.public import DiffusionBackbone, epsilon_loss

nn = torch.nn
F = torch.nn.functional


def cube_goal_margins(raw):
    """World-axis containment margins for a rotated 0.04 m cube, in metres."""
    result = []
    for pose_start, goal_start in ((25, 41), (32, 44)):
        p = raw[..., pose_start:pose_start + 3]
        q = raw[..., pose_start + 3:pose_start + 7]
        q = q / q.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-8)
        w, x, y, z = q.unbind(-1)
        r00 = 1 - 2 * (y * y + z * z)
        r01 = 2 * (x * y - w * z)
        r02 = 2 * (x * z + w * y)
        r10 = 2 * (x * y + w * z)
        r11 = 1 - 2 * (x * x + z * z)
        r12 = 2 * (y * z - w * x)
        ex = 0.02 * (r00.abs() + r01.abs() + r02.abs())
        ey = 0.02 * (r10.abs() + r11.abs() + r12.abs())
        err = (p - raw[..., goal_start:goal_start + 3]).abs()
        result.append(torch.stack((0.06 - ex - err[..., 0],
                                   0.06 - ey - err[..., 1],
                                   0.011 - err[..., 2]), -1))
    return torch.stack(result, -2)


def future_any(event, valid, lookahead=6):
    """Target-only short lookahead, never used in the deployment encoder."""
    out = event & valid
    for offset in range(1, lookahead + 1):
        shifted = torch.cat((event[:, offset:] & valid[:, offset:],
                             torch.zeros_like(event[:, :offset])), 1)
        out = out | shifted
    return out


@torch.no_grad()
def semantic_targets(states, valid):
    """Weak contact/progress labels, not a motor program or a contact sensor."""
    margin = cube_goal_margins(states)
    goal = (margin >= 0).all(-1)
    red = states[..., 25:28]
    blue = states[..., 32:35]
    tcp = states[..., 18:21]
    aperture = states[..., 7:9].mean(-1)
    open_hand = aperture > 0.030
    closed_hand = aperture < 0.028
    dr = (tcp[..., :2] - red[..., :2]).square().sum(-1).sqrt()
    db = (tcp[..., :2] - blue[..., :2]).square().sum(-1).sqrt()
    near_red = (tcp - red).square().sum(-1).sqrt() < 0.050
    supported_blue = goal[..., 1] & (blue[..., 2] < 0.027)
    departed_blue = supported_blue & open_hand & ((tcp - blue).square().sum(-1) > 0.035 ** 2)
    future_departure = future_any(departed_blue, valid)
    blue_release_tail = supported_blue & open_hand & (departed_blue | future_departure)
    label = torch.zeros_like(aperture, dtype=torch.long)
    label = torch.where(blue_release_tail, torch.ones_like(label), label)
    red_approach = supported_blue & (dr + 0.015 < db)
    label = torch.where(red_approach, torch.full_like(label, 2), label)
    lifted_red = closed_hand & near_red & (red[..., 2] > 0.040)
    future_lift = future_any(lifted_red, valid)
    attached_red = closed_hand & near_red & ((red[..., 2] > 0.040) |
                                             ((red[..., 2] > 0.027) & future_lift))
    label = torch.where(attached_red, torch.full_like(label, 3), label)
    goal_xy = (red[..., :2] - states[..., 41:43]).square().sum(-1).sqrt() < 0.045
    prev_z = torch.cat((red[:, :1, 2], red[:, :-1, 2]), 1)
    lowering = red[..., 2] - prev_z < -0.0003
    place_red = goal_xy & ((red[..., 2] < 0.250) | lowering) & (near_red | goal[..., 0])
    label = torch.where(place_red, torch.full_like(label, 4), label)
    retreat = goal.all(-1) & open_hand & (tcp[..., 2] - red[..., 2] > 0.030)
    label = torch.where(retreat, torch.full_like(label, 5), label)
    return label, margin, goal


class ProgressMemoryDP(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(n['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(n['std'], dtype=torch.float32))
        scale = self.obs_scale
        xyz_scale = torch.stack((scale[18:21], scale[25:28], scale[32:35])).amax(0)
        self.register_buffer('xyz_scale', xyz_scale.clone())
        self.frame_encoder = nn.Sequential(nn.Linear(77, 192), nn.LayerNorm(192),
                                           nn.SiLU(), nn.Linear(192, 128), nn.SiLU())
        self.initializer = nn.Sequential(nn.Linear(128, 128), nn.SiLU(),
                                         nn.Linear(128, 128), nn.Tanh())
        self.memory = nn.GRUCell(128, 128)
        self.mode_head = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 6))
        self.goal_head = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 8))
        # A training-only action-conditioned forecast of progress, not kinematics.
        self.action_encoder = nn.Sequential(nn.Linear(8, 128), nn.SiLU(), nn.Linear(128, 128))
        self.progress_transition = nn.GRUCell(128, 128)
        # Hidden 128 + normalized causal states 94 + belief 6 + predicted
        # margins 6 + goal probabilities 2 + observation disagreement 2.
        self.backbone = DiffusionBackbone(238, spec['training'])

    def norm_obs(self, raw):
        return (raw - self.obs_mean) / self.obs_scale

    def features(self, raw, previous):
        relatives = torch.cat(((raw[..., 25:28] - raw[..., 18:21]) / self.xyz_scale,
                               (raw[..., 32:35] - raw[..., 18:21]) / self.xyz_scale,
                               (raw[..., 25:28] - raw[..., 41:44]) / self.xyz_scale,
                               (raw[..., 32:35] - raw[..., 44:47]) / self.xyz_scale), -1)
        deltas = torch.cat(((raw[..., :9] - previous[..., :9]) / self.obs_scale[:9],
                            (raw[..., 18:21] - previous[..., 18:21]) / self.xyz_scale,
                            (raw[..., 25:28] - previous[..., 25:28]) / self.xyz_scale,
                            (raw[..., 32:35] - previous[..., 32:35]) / self.xyz_scale), -1)
        return torch.cat((self.norm_obs(raw), relatives, deltas), -1)

    def history_hidden(self, raw_history):
        previous = torch.cat((raw_history[:, :1], raw_history[:, :-1]), 1)
        z = self.frame_encoder(self.features(raw_history, previous))
        h = self.initializer(z[:, 0])
        hs = []
        for j in range(raw_history.shape[1]):
            h = self.memory(z[:, j], h)
            hs.append(h)
        return torch.stack(hs, 1)

    def forecast(self, h, encoded_actions):
        z = self.action_encoder(encoded_actions)
        hs = []
        for j in range(encoded_actions.shape[1]):
            h = self.progress_transition(z[:, j], h)
            hs.append(h)
        return torch.stack(hs, 1)

    def condition(self, h, history):
        gp = self.goal_head(h)
        mode = self.mode_head(h).softmax(-1)
        predicted_goals = gp[..., 6:].sigmoid()
        measured_goals = (cube_goal_margins(history[:, -1]) >= 0).all(-1).to(h.dtype)
        disagreement = predicted_goals - measured_goals
        return torch.cat((h, self.norm_obs(history).flatten(1), mode, gp[..., :6],
                          predicted_goals, disagreement), -1)

    def forward(self, noisy_action, timestep, raw_history):
        # No hidden cache: repeated DDPM calls cannot advance task memory.
        h = self.history_hidden(raw_history)[:, -1]
        return self.backbone(noisy_action, timestep, self.condition(h, raw_history))

    def progress_diagnostics(self, raw_history):
        h = self.history_hidden(raw_history)[:, -1]
        gp = self.goal_head(h)
        margin = cube_goal_margins(raw_history[:, -1])
        measured = (margin >= 0).all(-1).to(h.dtype)
        return {'mode_probability': self.mode_head(h).softmax(-1),
                'predicted_goal_probability': gp[..., 6:].sigmoid(),
                'measured_goal_flags': measured,
                'measured_margins_m': margin,
                'goal_disagreement': gp[..., 6:].sigmoid() - measured}


def build_model(spec):
    return ProgressMemoryDP(spec)


def weighted_mean(value, weight):
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def head_errors(model, hidden, labels, margins, goals):
    logits = model.mode_head(hidden)
    gp = model.goal_head(hidden)
    ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten(),
                         reduction='none').reshape_as(labels)
    ml = F.smooth_l1_loss(gp[..., :6], margins, reduction='none').mean(-1)
    gl = F.binary_cross_entropy_with_logits(gp[..., 6:], goals.to(gp.dtype),
                                            reduction='none').mean(-1)
    return logits, ce, ml, gl


def compute_loss(model, batch, spec):
    history = batch['raw_obs']
    h_history = model.history_hidden(history)
    current = h_history[:, -1]
    predicted = model.backbone(batch['noisy_action'], batch['timesteps'],
                               model.condition(current, history))
    # All future observations are targets only. Action slot zero ends at the
    # current state, so forecast from slot one onward, never replay slot zero.
    future_valid = batch['future_mask'][:, 1:, 0] > 0
    states = torch.cat((history, batch['future_obs'][:, 1:]), 1)
    valid = torch.cat((torch.ones_like(history[..., 0], dtype=torch.bool), future_valid), 1)
    labels, margins_m, goals = semantic_targets(states, valid)
    margins = (margins_m / model.xyz_scale).flatten(-2)
    changes = torch.cat((torch.zeros_like(labels[:, :1], dtype=torch.bool),
                         labels[:, 1:] != labels[:, :-1]), 1)
    g = batch['native_action'][..., 7]
    grip_change = torch.cat((torch.zeros_like(g[:, :1]),
                             ((g[:, 1:] - g[:, :-1]).abs() > 0.5).to(g.dtype)), 1)
    target_valid = batch['future_mask'][..., 0].to(g.dtype)
    action_weights = (1.0 + 0.75 * grip_change
                      + target_valid * (0.5 * changes[:, 1:].to(g.dtype)
                                        + 0.35 * (labels[:, 1:] == 5).to(g.dtype)))
    diffusion_loss = epsilon_loss(predicted, batch['noise'],
                                  batch['mask'] * action_weights.unsqueeze(-1))

    # The learned forecast sees demonstrated normalized actions, NOT measured
    # future states. Its mode/goal predictions train the causal initial memory.
    forecast = model.forecast(current, batch['encoded_action'][:, 1:])
    hseq = torch.cat((h_history, forecast), 1)
    weight = valid.to(current.dtype) * (1.0 + 0.5 * changes.to(current.dtype)
                                        + 0.35 * (labels == 5).to(current.dtype))
    logits, ce, ml, gl = head_errors(model, hseq, labels, margins, goals)
    mode_loss = weighted_mean(ce, weight)
    margin_loss = weighted_mean(ml, weight)
    goal_loss = weighted_mean(gl, weight)
    prob = logits.softmax(-1)
    pair_valid = valid[:, 1:] & valid[:, :-1]
    # Penalize decreases in ordered CDF tails, never prohibit a transition.
    tails = 1.0 - prob.cumsum(-1)[..., :-1]
    loss_of_goal = (goals[:, :-1] & ~goals[:, 1:]).any(-1)
    ordered_mask = pair_valid & (labels[:, 1:] >= labels[:, :-1]) & ~loss_of_goal
    backward = F.relu(tails[:, :-1] - tails[:, 1:] - 0.02).square().mean(-1)
    order_loss = weighted_mean(backward, ordered_mask.to(current.dtype))
    same_mode = pair_valid & (labels[:, 1:] == labels[:, :-1]) & ~loss_of_goal
    persistence_loss = weighted_mean((prob[:, 1:] - prob[:, :-1]).square().mean(-1),
                                      same_mode.to(current.dtype))

    # Reinitialize at the current causal state with a duplicated history. This
    # teaches the state initializer to approximate the two-frame belief at
    # random dataset anchors without requiring a predecessor hidden vector.
    reset_history = history[:, -1:].expand(-1, 2, -1)
    reset_h = model.history_hidden(reset_history)[:, -1:]
    reset_logits, rce, rml, rgl = head_errors(model, reset_h, labels[:, 1:2],
                                             margins[:, 1:2], goals[:, 1:2])
    teacher = prob[:, 1:2].detach()
    kl = (teacher * (teacher.clamp_min(1e-8).log() - reset_logits.log_softmax(-1))).sum(-1)
    state_agreement = (F.layer_norm(reset_h, (128,)) -
                       F.layer_norm(current[:, None].detach(), (128,))).square().mean(-1)
    initialization_loss = (rce + rml + 0.5 * rgl + kl + 0.1 * state_agreement).mean()

    # Action/progress coupling is differentiable through the DDPM clean estimate.
    # Clamp only this auxiliary input, not the action representation or sampler.
    alpha = batch['alpha_bar'].reshape(-1, 1, 1).clamp_min(1e-6)
    x0 = (batch['noisy_action'] - (1.0 - alpha).sqrt() * predicted) / alpha.sqrt()
    predicted_forecast = model.forecast(current, x0[:, 1:].clamp(-1.25, 1.25))
    _, pce, pml, pgl = head_errors(model, predicted_forecast, labels[:, 2:],
                                  margins[:, 2:], goals[:, 2:])
    # Keep alpha in the numerator; do not renormalize away high-noise damping.
    forecast_weight = weight[:, 2:] * batch['mask'][:, 1:, 0]
    coupling_loss = weighted_mean((pce + pml + 0.5 * pgl) * alpha[:, 0, 0:1],
                                  forecast_weight)
    prior_loss = (0.12 * mode_loss + 0.05 * margin_loss + 0.06 * goal_loss
                  + 0.025 * order_loss + 0.01 * persistence_loss
                  + 0.04 * initialization_loss + 0.03 * coupling_loss)
    return {'loss': diffusion_loss + prior_loss,
            'diffusion_loss': diffusion_loss, 'prior_loss': prior_loss}
