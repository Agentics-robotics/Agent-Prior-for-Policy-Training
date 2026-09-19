import torch
from torch import nn
from torch.nn import functional as F
from appl.public import DiffusionBackbone, epsilon_loss


# Quaternion convention is wxyz. These features do not transform joint targets.
def qconj(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def qmul(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.cross(av, bv, dim=-1)), dim=-1)


def qrotate(q, v):
    u = q[..., 1:]
    uv = torch.cross(u, v, dim=-1)
    return v + 2.0 * (q[..., :1] * uv + torch.cross(u, uv, dim=-1))


def unitq(q):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class ContactDiffusion(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(n['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(n['std'], dtype=torch.float32))
        scales = torch.tensor(n['std'], dtype=torch.float32)
        # A scalar inherited from the full-demonstration positional ranges, not
        # a fitted skill-local scale. Scalar scaling also works after rotation.
        self.register_buffer('pos_scale', torch.cat((scales[18:21], scales[25:28],
                                                       scales[32:35])).max())
        self.horizon = spec['training']['horizon']
        self.dt = 0.05
        self.feature_net = nn.Sequential(nn.Linear(91, 128), nn.SiLU(),
                                         nn.Linear(128, 128), nn.LayerNorm(128))
        self.initial_hidden = nn.Sequential(nn.Linear(91, 128), nn.Tanh())
        self.filter_cell = nn.GRUCell(128, 128)
        self.mode_evidence = nn.Linear(128, 6)
        self.transition_delta = nn.Linear(128, 36)
        nn.init.zeros_(self.transition_delta.weight)
        nn.init.zeros_(self.transition_delta.bias)
        self.transition_logits = nn.Parameter(2.5 * torch.eye(6))
        self.mode_embedding = nn.Sequential(nn.Linear(6, 32), nn.SiLU())
        # Base condition: both normalized states, recurrent features, posterior,
        # and a learned posterior embedding. No command or role metadata input.
        self.base_dim = 94 + 128 + 6 + 32
        self.step_embedding = nn.Embedding(self.horizon, 16)
        self.grip_initial_hidden = nn.Sequential(nn.Linear(self.base_dim, 96), nn.Tanh())
        self.grip_initial_probability = nn.Linear(self.base_dim, 1)
        self.grip_cell = nn.GRUCell(17, 96)
        self.grip_hazards = nn.Linear(96, 2)
        nn.init.constant_(self.grip_hazards.bias, -3.0)
        self.grip_embedding = nn.Sequential(nn.Linear(self.horizon * 3, 64),
                                           nn.SiLU(), nn.Linear(64, 32))
        self.condition_dim = self.base_dim + 32
        self.backbone = DiffusionBackbone(self.condition_dim, spec['training'])
        self.dynamics_initial = nn.Sequential(nn.Linear(self.condition_dim, 128), nn.Tanh())
        self.dynamics_cell = nn.GRUCell(8 + 16, 128)
        self.dynamics_readout = nn.Sequential(nn.Linear(128, 128), nn.SiLU(),
                                             nn.Linear(128, 18))
        self.register_buffer('physical_weights', torch.tensor(
            [1., 1.] + [5., 5., 5., 1., 1., 1., 1.] * 2 + [2., 2.]))

    def normalize_obs(self, raw):
        return (raw - self.obs_mean) / self.obs_scale

    def relative_pose(self, raw):
        tcp = raw[..., 18:21]
        inv = qconj(unitq(raw[..., 21:25]))
        result = []
        for start in (25, 32):
            translation = qrotate(inv, raw[..., start:start+3] - tcp) / self.pos_scale
            rotation = unitq(qmul(inv, unitq(raw[..., start+3:start+7])))
            # The relative orientation is near 180 degrees here. Canonicalizing
            # by w would jump at the demonstrated pose; use the dominant axis.
            dominant = rotation.gather(-1, rotation.abs().argmax(-1, keepdim=True))
            rotation = rotation * torch.where(dominant < 0,
                                               -torch.ones_like(dominant),
                                               torch.ones_like(dominant))
            result.extend((translation, rotation))
        return torch.cat(result, dim=-1)

    def features(self, raw):
        norm = self.normalize_obs(raw)
        rel = self.relative_pose(raw)
        goal = torch.cat((raw[..., 25:28] - raw[..., 41:44],
                          raw[..., 32:35] - raw[..., 44:47]), dim=-1) / self.pos_scale
        positions = torch.cat((raw[..., 18:21], raw[..., 25:28], raw[..., 32:35]), dim=-1)
        velocity = (positions[:, 1:2] - positions[:, 0:1]) / (self.dt * self.pos_scale)
        relative_rate = (rel[:, 1:2] - rel[:, 0:1]) / self.dt
        velocity = torch.cat((torch.zeros_like(velocity), velocity), dim=1)
        relative_rate = torch.cat((torch.zeros_like(relative_rate), relative_rate), dim=1)
        # Repeated padding or snapshot truncation supplies no measured motion.
        valid = ((raw[:, 1:2] - raw[:, 0:1]).abs().amax(-1, keepdim=True) > 1e-8).to(raw.dtype)
        valid = torch.cat((torch.zeros_like(valid), valid), dim=1)
        return torch.cat((norm, rel, goal, velocity, relative_rate, valid), dim=-1), norm

    def filter_history(self, raw):
        features, norm = self.features(raw)
        hidden = self.initial_hidden(features[:, 0])
        encoded = self.feature_net(features)
        hidden = self.filter_cell(encoded[:, 0], hidden)
        posterior = self.mode_evidence(hidden).softmax(-1)
        hidden = self.filter_cell(encoded[:, 1], hidden)
        transitions = (self.transition_logits[None] +
                       self.transition_delta(hidden).reshape(-1, 6, 6)).softmax(-1)
        # Elementwise contraction is equivalent to p @ T and avoids a backend
        # small-bmm backward path that needs prohibited runtime compilation.
        predicted_prior = (posterior[:, :, None] * transitions).sum(dim=1)
        posterior = (predicted_prior.clamp_min(1e-6).log() + self.mode_evidence(hidden)).softmax(-1)
        base = torch.cat((norm.flatten(1), hidden, posterior,
                          self.mode_embedding(posterior)), dim=-1)
        return base, posterior

    def gripper_distribution(self, base):
        # State is P(open). Hazards are close->open and open->close. The model
        # evolves a distribution, not the previous ground-truth/current action.
        hidden = self.grip_initial_hidden(base)
        probability = self.grip_initial_probability(base).sigmoid()
        probabilities = [probability]
        hazards = [torch.zeros((base.shape[0], 2), device=base.device, dtype=base.dtype)]
        for j in range(1, self.horizon):
            step = self.step_embedding.weight[j][None].expand(base.shape[0], -1)
            hidden = self.grip_cell(torch.cat((step, probability), dim=-1), hidden)
            hazard = self.grip_hazards(hidden).sigmoid()
            probability = (1.0 - probability) * hazard[:, :1] + probability * (1.0 - hazard[:, 1:])
            probabilities.append(probability)
            hazards.append(hazard)
        probabilities = torch.stack(probabilities, dim=1)
        hazards = torch.stack(hazards, dim=1)
        return probabilities, hazards

    def details(self, noisy_action, timestep, raw_history):
        base, posterior = self.filter_history(raw_history)
        probability, hazards = self.gripper_distribution(base)
        grip_features = torch.cat((2.0 * probability - 1.0, hazards), dim=-1).flatten(1)
        condition = torch.cat((base, self.grip_embedding(grip_features)), dim=-1)
        epsilon = self.backbone(noisy_action, timestep, condition)
        return epsilon, condition, posterior, probability, hazards

    def forward(self, noisy_action, timestep, raw_history):
        # Stateless across calls/DDPM iterations: only the two supplied causal
        # observations are filtered; there is no undisclosed history cache.
        return self.details(noisy_action, timestep, raw_history)[0]

    def predict_physics(self, encoded_action, condition):
        hidden = self.dynamics_initial(condition)
        predictions = []
        for j in range(self.horizon):
            step = self.step_embedding.weight[j][None].expand(encoded_action.shape[0], -1)
            hidden = self.dynamics_cell(torch.cat((encoded_action[:, j], step), dim=-1), hidden)
            predictions.append(self.dynamics_readout(hidden))
        return torch.stack(predictions, dim=1)

    def physical_state(self, raw):
        fingers = (raw[..., 7:9] - self.obs_mean[7:9]) / self.obs_scale[7:9]
        heights = torch.stack((raw[..., 27], raw[..., 34]), dim=-1) / self.pos_scale
        return torch.cat((fingers, self.relative_pose(raw), heights), dim=-1)


# These are weak, training-only explanatory targets, not a deployed mode switch,
# contact sensor, success predicate or action controller.
def weak_mode_target(batch):
    history = batch['raw_obs']
    now, old = history[:, -1], history[:, 0]
    tcp = now[:, 18:21]
    objects = torch.stack((now[:, 25:28], now[:, 32:35]), dim=1)
    old_objects = torch.stack((old[:, 25:28], old[:, 32:35]), dim=1)
    near = torch.exp(-((objects - tcp[:, None]).square().sum(-1)) / (2.0 * 0.05 ** 2))
    width = now[:, 7:9].mean(-1)
    closed = torch.sigmoid((0.029 - width) / 0.003)
    lifted = torch.sigmoid((objects[..., 2] - 0.040) / 0.012)
    residual = objects - old_objects - (tcp - old[:, 18:21])[:, None]
    comotion = torch.exp(-residual.square().sum(-1) / (2.0 * 0.008 ** 2))
    future = batch['future_obs']
    future_z = torch.stack((future[..., 27], future[..., 34]), dim=-1)
    future_valid = (batch['future_mask'] * batch['mask']).squeeze(-1)
    future_valid = future_valid.clone()
    future_valid[:, 0] = 0.0
    confirmed_z = torch.where(future_valid[..., None] > 0, future_z,
                              objects[:, None, :, 2]).amax(dim=1)
    confirmation = torch.sigmoid((confirmed_z - 0.05) / 0.012)
    # Slot 1 is action[t]. It is a LABEL only, never an encoder input.
    known_command = batch['mask'][:, 1, 0]
    close_label = (batch['native_action'][:, 1, 7] < 0).to(now.dtype)
    close_label = close_label * known_command + closed * (1.0 - known_command)
    closing = near * (1.0 - lifted) * (0.1 + 0.9 * close_label[:, None])
    closing = closing * (0.35 + 0.65 * closed[:, None]) * (0.7 + 0.3 * confirmation)
    loaded = near * lifted * closed[:, None] * (0.4 + 0.6 * comotion)
    red_goal = torch.exp(-(now[:, 25:27] - now[:, 41:43]).square().sum(-1) / (2.0 * 0.07 ** 2))
    release_progress = torch.sigmoid((0.037 - width) / 0.002)
    predecessor = red_goal * near[:, 0] * (closed + (1.0 - closed) * release_progress)
    other = torch.stack((closing[:, 0] * (1.0 - red_goal),
                         loaded[:, 0] * (1.0 - red_goal),
                         closing[:, 1], loaded[:, 1], predecessor), dim=-1)
    empty = (1.0 - other.amax(-1, keepdim=True)).clamp_min(0.0)
    target = torch.cat((empty, other), dim=-1) + 0.015
    return target / target.sum(-1, keepdim=True)


def build_model(spec):
    return ContactDiffusion(spec)


def compute_loss(model, batch, spec):
    history = batch['raw_obs']
    if model.training:
        # The fixed loader samples the entire expanded slices. This augmentation
        # simulates overlap entry with only a snapshot; no slice is discarded.
        truncate = (torch.rand((history.shape[0], 1, 1), device=history.device) < 0.15)
        first = torch.where(truncate, history[:, 1:2], history[:, 0:1])
        history = torch.cat((first, history[:, 1:2]), dim=1)
    epsilon, condition, posterior, probability, hazards = model.details(
        batch['noisy_action'], batch['timesteps'], history)
    diffusion = epsilon_loss(epsilon, batch['noise'], batch['mask'])
    with torch.no_grad():
        mode_target = weak_mode_target(batch)
        # Difference from state t: future slot 0 is state t, not state t+1.
        physics_target = model.physical_state(batch['future_obs']) - model.physical_state(batch['raw_obs'][:, -1:])
    mode_loss = -(mode_target * posterior.clamp_min(1e-6).log()).sum(-1).mean()

    action_mask = batch['mask'].squeeze(-1)
    open_label = (batch['native_action'][..., 7] > 0).to(epsilon.dtype)
    p = probability.squeeze(-1).clamp(1e-5, 1.0 - 1e-5)
    categorical = masked_mean(F.binary_cross_entropy(p, open_label, reduction='none'), action_mask)
    hz = hazards[:, 1:].clamp(1e-5, 1.0 - 1e-5)
    previous = open_label[:, :-1]
    conditional_open = (1.0 - previous) * hz[..., 0] + previous * (1.0 - hz[..., 1])
    pair_mask = action_mask[:, 1:] * action_mask[:, :-1]
    # Observed switches are allowed and explicitly supervised, rather than
    # penalized by a universal action-smoothing constraint.
    switch_weight = 1.0 + 4.0 * (open_label[:, 1:] - previous).abs()
    transition_loss = masked_mean(
        F.binary_cross_entropy(conditional_open, open_label[:, 1:], reduction='none') * switch_weight,
        pair_mask)

    physics_mask = (batch['future_mask'] * batch['mask']).squeeze(-1).clone()
    physics_mask[:, 0] = 0.0
    teacher_prediction = model.predict_physics(batch['encoded_action'], condition)
    weights = model.physical_weights / model.physical_weights.sum()
    teacher_error = (F.smooth_l1_loss(teacher_prediction, physics_target, reduction='none') * weights).sum(-1)
    dynamics_loss = masked_mean(teacher_error, physics_mask)

    alpha = batch['alpha_bar'].reshape(-1, 1, 1).clamp_min(1e-6)
    clean = (batch['noisy_action'] - torch.sqrt(1.0 - alpha) * epsilon) / torch.sqrt(alpha)
    # Suppress high-noise amplification, and bound inputs to the approximate
    # learned response model. This is an auxiliary, not an output action clamp.
    denoised_prediction = model.predict_physics(clean.clamp(-2.0, 2.0), condition)
    denoised_error = (F.smooth_l1_loss(denoised_prediction, physics_target, reduction='none') * weights).sum(-1)
    coupled_loss = masked_mean(denoised_error * alpha[:, 0, 0, None], physics_mask)

    prior = (0.08 * mode_loss + 0.15 * categorical + 0.10 * transition_loss +
             0.20 * dynamics_loss + 0.05 * coupled_loss)
    return {'loss': diffusion + prior, 'diffusion_loss': diffusion, 'prior_loss': prior,
            'mode_loss': mode_loss, 'gripper_loss': categorical,
            'transition_loss': transition_loss, 'dynamics_loss': dynamics_loss,
            'coupled_loss': coupled_loss}
