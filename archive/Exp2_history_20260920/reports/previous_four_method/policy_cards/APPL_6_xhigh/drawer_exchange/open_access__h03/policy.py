import math
import torch
from appl.public import DiffusionBackbone, epsilon_loss

nn = torch.nn
F = torch.nn.functional


def masked_per_sample(value, mask):
    return (value * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) * value.shape[-1]).clamp_min(1.0)


def unit_poses(state):
    # Quaternions have unit-component normalization in the shared normalizer.
    return torch.cat((state[..., :21], F.normalize(state[..., 21:25], dim=-1, eps=1e-6),
                      state[..., 25:28], F.normalize(state[..., 28:32], dim=-1, eps=1e-6),
                      state[..., 32:35], F.normalize(state[..., 35:39], dim=-1, eps=1e-6),
                      state[..., 39:41]), dim=-1)


def readiness(raw):
    d = raw[..., 39]
    width = raw[..., 7] + raw[..., 8]
    distance = torch.linalg.vector_norm(raw[..., 18:21] - raw[..., 25:28], dim=-1)
    opened = torch.sigmoid((d - 0.26) / 0.015)
    retreat = opened * torch.sigmoid((width - 0.065) / 0.006) * torch.sigmoid((raw[..., 20] - 0.15) / 0.03)
    lifted = opened * torch.sigmoid((0.05 - width) / 0.008) * torch.sigmoid((0.04 - distance) / 0.012) * torch.sigmoid((raw[..., 27] - 0.075) / 0.012)
    return torch.stack((opened, retreat, lifted), dim=-1)


class RecurrentDynamics(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.initial = nn.Sequential(nn.Linear(94, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.Tanh())
        self.cell = nn.GRUCell(41 + 8, hidden)
        self.increment = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 41))
        self.progress = nn.Sequential(nn.Linear(hidden + 41, 64), nn.SiLU(), nn.Linear(64, 3))
        nn.init.normal_(self.increment[-1].weight, std=0.001)
        nn.init.zeros_(self.increment[-1].bias)

    def forward(self, history, actions):
        hidden = self.initial(history.flatten(1))
        state = history[:, -1, :41]
        states, logits = [], []
        for j in range(actions.shape[1]):
            hidden = self.cell(torch.cat((state, actions[:, j]), dim=-1), hidden)
            state = unit_poses(state + self.increment(hidden))
            states.append(state)
            logits.append(self.progress(torch.cat((hidden, state), dim=-1)))
        return torch.stack(states, dim=1), torch.stack(logits, dim=1)


class SuccessorDiffusion(nn.Module):
    def __init__(self, spec):
        super().__init__()
        normalizer = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(normalizer['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(normalizer['std'], dtype=torch.float32))
        self.register_buffer('linear_indices', torch.tensor(list(range(21)) + [25, 26, 27, 32, 33, 34, 39, 40], dtype=torch.long))
        self.register_buffer('uncertainty_indices', torch.tensor([7, 8, 18, 19, 20, 25, 26, 27, 39, 40], dtype=torch.long))
        self.horizon = int(spec['training']['horizon'])
        self.steps = int(spec['training']['denoising_train_steps'])
        self.prefix = 8
        self.candidates = 3
        self.encoder = nn.Sequential(nn.Linear(141, 256), nn.SiLU(), nn.Linear(256, 128), nn.LayerNorm(128), nn.SiLU())
        self.backbone = DiffusionBackbone(128, spec['training'])
        self.time_net = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 64))
        self.register_buffer('time_frequencies', torch.exp(-math.log(10000.0) * torch.arange(32).float() / 31))
        position = torch.arange(self.horizon).float().unsqueeze(-1)
        freq = torch.exp(-math.log(1000.0) * torch.arange(8).float() / 7)
        self.register_buffer('positions', torch.cat((torch.sin(position * freq), torch.cos(position * freq)), dim=-1))
        self.heads = nn.ModuleList([nn.Sequential(nn.Conv1d(224, 128, 3, padding=1), nn.SiLU(), nn.Conv1d(128, 16, 1)) for _ in range(self.candidates)])
        for head in self.heads:
            nn.init.normal_(head[-1].weight, std=0.005)
            nn.init.zeros_(head[-1].bias)
        self.anchor = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, self.horizon * 8), nn.Tanh())
        self.dynamics = nn.ModuleList([RecurrentDynamics(128) for _ in range(3)])

    def normalize(self, raw):
        return (raw - self.obs_mean) / self.obs_scale

    def decode_state(self, encoded):
        return encoded * self.obs_scale[:41] + self.obs_mean[:41]

    def stage_gates(self, raw):
        width = raw[..., 7] + raw[..., 8]
        distance = torch.linalg.vector_norm(raw[..., 18:21] - raw[..., 25:28], dim=-1)
        closed = torch.sigmoid((0.045 - width) / 0.006)
        pull = closed * torch.sigmoid((0.04 - (raw[..., 20] - 0.128).abs()) / 0.012) * torch.sigmoid((0.03 - raw[..., 19].abs()) / 0.01) * torch.sigmoid((distance - 0.12) / 0.02)
        pickup = closed * torch.sigmoid((0.045 - distance) / 0.012) * torch.sigmoid((raw[..., 27] - 0.067) / 0.01)
        near = torch.sigmoid((0.08 - distance) / 0.02)
        return pull, pickup, near

    def rollout(self, history, actions, frozen=False):
        states, logits = [], []
        for member in self.dynamics:
            if frozen:
                parameters = {name: value.detach() for name, value in member.named_parameters()}
                s, l = torch.func.functional_call(member, parameters, (history, actions))
            else:
                s, l = member(history, actions)
            states.append(s)
            logits.append(l)
        return torch.stack(states, dim=0), torch.stack(logits, dim=0)

    def state_error(self, pred, target, mask):
        linear = (pred.index_select(-1, self.linear_indices) - target.index_select(-1, self.linear_indices)).square()
        linear = masked_per_sample(linear, mask)
        rotations = []
        for start in (21, 28, 35):
            p = F.normalize(pred[..., start:start+4], dim=-1, eps=1e-6)
            q = F.normalize(target[..., start:start+4], dim=-1, eps=1e-6)
            sign = torch.where((p * q).sum(-1, keepdim=True) >= 0, 1.0, -1.0)
            q = q * sign
            # Sign-invariant squared SO(3) geodesic angle, stable at identity.
            minus = torch.sqrt((p - q).square().sum(-1) + 1e-12)
            plus = torch.sqrt((p + q).square().sum(-1) + 1e-12)
            angle = 4.0 * torch.atan2(minus, plus)
            rotations.append((angle / math.pi).square())
        rotation = masked_per_sample(torch.stack(rotations, dim=-1), mask)
        return 5.0 * linear + 0.2 * rotation

    def energy(self, history, actions, anchor, states, logits, mask):
        # All quantities here are predicted outcomes or candidate actions, not
        # an observation-only training penalty. Normalizers are shared ranges.
        raw0 = history[:, -1]
        raw = self.decode_state(states.mean(dim=0))
        predicted_readiness = torch.sigmoid(logits).mean(dim=0)
        pull0, pickup0, near0 = self.stage_gates(raw0)
        pull_future, _, _ = self.stage_gates(raw)
        full = torch.sigmoid((raw0[:, 39] - 0.275) / 0.01)
        preserve = torch.sigmoid((raw0[:, 39] - 0.18) / 0.02)
        d0 = raw0[:, None, 39]
        opening_loss = preserve[:, None] * (F.relu(d0 - raw[..., 39]) / self.obs_scale[39]).square()
        overpull = (F.relu(raw[..., 39] - 0.300) / self.obs_scale[39]).square()
        displacement = raw[..., 25:28] - raw0[:, None, 25:28]
        drawer_displacement = raw[..., 39] - d0
        pull_displacement = torch.stack((displacement[..., 0] + drawer_displacement, displacement[..., 1], displacement[..., 2]), dim=-1)
        pull_coupling = (pull_displacement / self.obs_scale[25:28]).square().mean(dim=-1)
        # Deactivate handle coupling as the predicted fingers release.
        pull_coupling = pull_coupling * pull0[:, None] * pull_future
        lateral = ((raw[..., 19] - raw0[:, None, 19]) / self.obs_scale[19]).square() * pull0[:, None] * pull_future
        relative = raw[..., 25:28] - raw[..., 18:21]
        relative0 = raw0[:, 25:28] - raw0[:, 18:21]
        relative_scale = torch.maximum(self.obs_scale[25:28], self.obs_scale[18:21])
        pickup_coupling = ((relative - relative0[:, None]) / relative_scale).square().mean(dim=-1) * pickup0[:, None]
        progress_cost = pull0[:, None] * (1.0 - predicted_readiness[..., 0])
        progress_cost = progress_cost + (full * (1.0 - near0))[:, None] * (1.0 - predicted_readiness[..., 1])
        progress_cost = progress_cost + (full * near0)[:, None] * (1.0 - predicted_readiness[..., 2])
        physical = 2.0 * opening_loss + 2.0 * overpull + 4.0 * pull_coupling + 2.0 * lateral + 4.0 * pickup_coupling + 0.05 * progress_cost
        physical = masked_per_sample(physical.unsqueeze(-1), mask)
        uncertainty = states.index_select(-1, self.uncertainty_indices).var(dim=0, unbiased=False)
        uncertainty = masked_per_sample(uncertainty, mask)
        # Unit-variance normalized-action Gaussian proxy, not exact DDPM density.
        imitation = masked_per_sample((actions - anchor).square(), mask)
        return 2.0 * imitation + 2.0 * uncertainty + physical, uncertainty

    def details(self, noisy_action, timestep, raw_history):
        b, h, _ = noisy_action.shape
        normalized = self.normalize(raw_history)
        condition = self.encoder(torch.cat((normalized.flatten(1), normalized[:, -1] - normalized[:, -2]), dim=-1))
        t = torch.as_tensor(timestep, device=noisy_action.device).reshape(-1).expand(b)
        phase = t.to(noisy_action.dtype).unsqueeze(-1) * self.time_frequencies
        time = self.time_net(torch.cat((phase.sin(), phase.cos()), dim=-1))
        base = self.backbone(noisy_action, t, condition)
        head_input = torch.cat((noisy_action, base, condition[:, None].expand(-1, h, -1), time[:, None].expand(-1, h, -1), self.positions[None, :h].expand(b, -1, -1)), dim=-1).transpose(1, 2)
        experts, companions = [], []
        for head in self.heads:
            pair = head(head_input).transpose(1, 2)
            experts.append(base + 0.25 * pair[..., :8].tanh())
            companions.append(pair[..., 8:].tanh())
        experts = torch.stack(experts, dim=1)
        companions = torch.stack(companions, dim=1)
        anchor = self.anchor(condition).reshape(b, self.horizon, 8)
        # Interface adaptation: rank learned, paired denoising refinements, not
        # separate completed DDPM chains. Future labels are never read here.
        with torch.no_grad():
            candidate_prefix = companions[:, :, 1:1+self.prefix].reshape(b * self.candidates, self.prefix, 8)
            repeated_history = raw_history[:, None].expand(-1, self.candidates, -1, -1).reshape(b * self.candidates, 2, 47)
            normalized_history = self.normalize(repeated_history)
            reference = anchor[:, None, 1:1+self.prefix].expand(-1, self.candidates, -1, -1).reshape(b * self.candidates, self.prefix, 8)
            states, logits = self.rollout(normalized_history, candidate_prefix)
            prefix_mask = noisy_action.new_ones((b * self.candidates, self.prefix, 1))
            cost, uncertainty = self.energy(repeated_history, candidate_prefix, reference, states, logits, prefix_mask)
            costs = cost.reshape(b, self.candidates)
            preference = torch.softmax(-costs / 0.15, dim=1)
            # Weak, uncertainty-tempered late-denoising selection. Low variance
            # is not a certificate; independent model biases may be shared.
            confidence = 1.0 / (1.0 + 20.0 * uncertainty.reshape(b, self.candidates).mean(dim=1, keepdim=True))
            strength = 0.5 * (1.0 - t.float().unsqueeze(-1) / max(self.steps - 1, 1)).clamp(0, 1).square() * confidence
            weights = (1.0 - strength) / self.candidates + strength * preference
        predicted = (experts * weights[:, :, None, None]).sum(dim=1)
        return predicted, experts, companions, anchor

    def forward(self, noisy_action, timestep, raw_history):
        return self.details(noisy_action, timestep, raw_history)[0]


def build_model(spec):
    return SuccessorDiffusion(spec)


def compute_loss(model, batch, spec):
    raw = batch['raw_obs']
    pred, experts, companions, anchor = model.details(batch['noisy_action'], batch['timesteps'], raw)
    action_mask = batch['mask']
    diffusion_loss = epsilon_loss(pred, batch['noise'], action_mask)
    expert_loss = sum(epsilon_loss(experts[:, k], batch['noise'], action_mask) for k in range(model.candidates)) / model.candidates
    companion_loss = sum(masked_per_sample((companions[:, k] - batch['encoded_action']).square(), action_mask).mean() for k in range(model.candidates)) / model.candidates
    alpha = batch['alpha_bar'].reshape(-1, 1, 1).clamp(1e-6, 1.0)
    consistency = pred.new_zeros(())
    for k in range(model.candidates):
        reconstruction = alpha.sqrt() * companions[:, k] + (1.0 - alpha).sqrt() * experts[:, k]
        consistency = consistency + masked_per_sample((reconstruction - batch['noisy_action']).square(), action_mask).mean() / model.candidates
    anchor_loss = masked_per_sample((anchor - batch['encoded_action']).square(), action_mask).mean()

    # Slot zero was executed from t-1 and its successor is current observation
    # t. Start consequence rollouts from t with slots 1..8, never slot zero.
    sl = slice(1, 1 + model.prefix)
    prefix_mask = batch['future_mask'][:, sl] * action_mask[:, sl]
    true_future = batch['future_obs'][:, sl]
    target_state = model.normalize(true_future)[..., :41]
    target_readiness = readiness(true_future)
    normalized_history = model.normalize(raw)
    states, logits = model.rollout(normalized_history, batch['encoded_action'][:, sl])
    dynamics_loss = pred.new_zeros(())
    readiness_loss = pred.new_zeros(())
    for member in range(len(model.dynamics)):
        state_error = model.state_error(states[member], target_state, prefix_mask)
        ready_error = masked_per_sample(F.binary_cross_entropy_with_logits(logits[member], target_readiness, reduction='none'), prefix_mask)
        bootstrap = (torch.rand_like(state_error) < 0.8).to(state_error.dtype)
        denominator = bootstrap.sum().clamp_min(1.0)
        dynamics_loss = dynamics_loss + (state_error * bootstrap).sum() / denominator / len(model.dynamics)
        readiness_loss = readiness_loss + (ready_error * bootstrap).sum() / denominator / len(model.dynamics)

    # Exact training scheduler alpha is supplied in the batch. No assumed
    # deployment noise schedule or sampler replacement is needed by the model.
    clean = ((batch['noisy_action'] - (1.0 - alpha).sqrt() * pred) / alpha.sqrt()).clamp(-1.0, 1.0)
    # Dynamics parameters are detached here, NOT candidate actions: consequence
    # errors backpropagate into the epsilon denoiser through predicted x0.
    generated_states, generated_logits = model.rollout(normalized_history, clean[:, sl], frozen=True)
    consequence_error = pred.new_zeros((raw.shape[0],))
    for member in range(len(model.dynamics)):
        state_error = model.state_error(generated_states[member], target_state, prefix_mask)
        ready_error = masked_per_sample(F.binary_cross_entropy_with_logits(generated_logits[member], target_readiness, reduction='none'), prefix_mask)
        consequence_error = consequence_error + (state_error + 0.25 * ready_error) / len(model.dynamics)
    energy, _ = model.energy(raw, clean[:, sl], anchor[:, sl].detach(), generated_states, generated_logits, prefix_mask)
    reliability = batch['alpha_bar'].reshape(-1).square()
    consequence_loss = (reliability * consequence_error).mean()
    selection_loss = (reliability * energy).mean()
    prior_loss = (0.10 * expert_loss + 0.20 * companion_loss + 0.20 * consistency + 0.10 * anchor_loss
                  + dynamics_loss + 0.25 * readiness_loss + 0.10 * consequence_loss + 0.02 * selection_loss)
    return {'loss': diffusion_loss + prior_loss, 'diffusion_loss': diffusion_loss, 'prior_loss': prior_loss,
            'dynamics_loss': dynamics_loss, 'readiness_loss': readiness_loss,
            'consequence_loss': consequence_loss, 'selection_loss': selection_loss,
            'companion_loss': companion_loss, 'consistency_loss': consistency}
