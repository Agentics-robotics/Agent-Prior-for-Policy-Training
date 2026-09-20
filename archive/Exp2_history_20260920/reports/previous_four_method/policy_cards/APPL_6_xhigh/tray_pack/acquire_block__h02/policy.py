import torch
from torch import nn
from torch.nn import functional as F
from appl.public import DiffusionBackbone, epsilon_loss


class PhaseDiffusion(nn.Module):
    """Short-history, progress-conditioned hidden semi-Markov diffusion policy."""
    def __init__(self, spec):
        super().__init__()
        n = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(n['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(n['std'], dtype=torch.float32))
        s = self.obs_scale
        # Relative features use shared original-demonstration scales, not slice fits.
        rel_scale = torch.cat((torch.sqrt(s[18:21] ** 2 + s[25:28] ** 2),
                               torch.sqrt(s[18:21] ** 2 + s[32:35] ** 2),
                               s[25:28], s[32:35]))
        self.register_buffer('rel_scale', rel_scale)
        self.modes = 6
        self.ages = 8
        self.token = nn.Sequential(nn.Linear(107, 192), nn.SiLU(),
                                   nn.Linear(192, 128), nn.LayerNorm(128), nn.SiLU())
        self.history_cell = nn.GRUCell(128, 128)
        self.emission = nn.Linear(128, 6)
        self.age_initializer = nn.Linear(128, 48)
        self.role_head = nn.Linear(128, 2)
        self.readiness_head = nn.Linear(128, 1)
        self.hazard_head = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 6))
        self.age_hazard = nn.Parameter(torch.zeros(6, 8))
        self.jump_head = nn.Linear(128, 36)
        self.roll_cell = nn.GRUCell(136, 128)
        self.mode_embedding = nn.Parameter(torch.randn(6, 32) * 0.04)
        self.condition = nn.Sequential(nn.Linear(407, 256), nn.SiLU(),
                                       nn.Linear(256, 256), nn.LayerNorm(256))
        self.denoiser = DiffusionBackbone(256, spec['training'])
        # A finite preference, never an irreversible controller or forbidden edge.
        cost = torch.ones(6, 6)
        for i in range(6):
            cost[i, i] = 0.0
            cost[i, (i + 1) % 6] = 0.0
        self.register_buffer('skip_cost', cost)
        self.register_buffer('identity', torch.eye(6))
        nn.init.zeros_(self.jump_head.weight)
        nn.init.zeros_(self.jump_head.bias)
        nn.init.constant_(self.hazard_head[-1].bias, -2.8)

    def features(self, raw, previous, valid):
        z = (raw - self.obs_mean) / self.obs_scale
        dz = ((raw - previous) / self.obs_scale) * (20.0 * valid)
        rel = torch.cat((raw[:, 18:21] - raw[:, 25:28],
                         raw[:, 18:21] - raw[:, 32:35],
                         raw[:, 25:28] - raw[:, 41:44],
                         raw[:, 32:35] - raw[:, 44:47]), dim=-1) / self.rel_scale
        return torch.cat((z, dz, rel, valid), dim=-1)

    def initial_joint(self, hidden):
        phase = self.emission(hidden).softmax(-1)
        age = self.age_initializer(hidden).reshape(-1, 6, 8).softmax(-1)
        return phase.unsqueeze(-1) * age

    def transition(self, hidden):
        hazard = torch.sigmoid(self.hazard_head(hidden).unsqueeze(-1) + self.age_hazard)
        logits = self.jump_head(hidden).reshape(-1, 6, 6)
        logits = logits - 2.5 * self.skip_cost - 10000.0 * self.identity
        return hazard, logits.softmax(-1)

    def advance(self, joint, hazard, jump):
        stay = joint * (1.0 - hazard)
        # Ages 0..6 and a right-censored 7+ bin; arbitrary dwell remains possible.
        shifted = torch.cat((torch.zeros_like(stay[:, :, :1]), stay[:, :, :-1]), dim=-1)
        shifted = torch.cat((shifted[:, :, :-1], shifted[:, :, -1:] + stay[:, :, -1:]), dim=-1)
        exits = (joint * hazard).sum(-1)
        # Equivalent to vector-matrix bmm, without its JIT outer-product backward.
        arrivals = (exits.unsqueeze(-1) * jump).sum(1)
        return shifted + torch.cat((arrivals.unsqueeze(-1), torch.zeros_like(stay[:, :, 1:])), dim=-1)

    def encode(self, raw_history):
        a, b = raw_history[:, 0], raw_history[:, 1]
        valid = ((a - b).abs().sum(-1, keepdim=True) > 1e-9).to(a.dtype)
        zero = torch.zeros_like(valid)
        h0 = self.history_cell(self.token(self.features(a, a, zero)))
        h1 = self.history_cell(self.token(self.features(b, a, valid)), h0 * valid)
        j0 = self.initial_joint(h0)
        hz, jp = self.transition(h1)
        predicted = self.advance(j0, hz, jp)
        emission = self.emission(h1).softmax(-1)
        filtered = predicted * emission.unsqueeze(-1)
        filtered = filtered / filtered.sum((1, 2), keepdim=True).clamp_min(1e-8)
        fresh = self.initial_joint(h1)
        joint = valid.unsqueeze(-1) * filtered + (1.0 - valid.unsqueeze(-1)) * fresh
        role = self.role_head(h1).softmax(-1)
        ready_logit = self.readiness_head(h1)
        # Causal predictive phase rollout, with no future states or source indices.
        hidden = h1
        joints = [joint]
        transitions = []
        for _ in range(14):
            p = joint.sum(-1)
            hidden = self.roll_cell(torch.cat((h1, p, role), dim=-1), hidden)
            hz, jp = self.transition(hidden)
            transitions.append((hz, jp))
            joint = self.advance(joint, hz, jp)
            joints.append(joint)
        p = joints[0].sum(-1)
        prognosis = torch.cat([joints[k].sum(-1) for k in (2, 4, 8, 14)], dim=-1)
        # Direct causal state pathway retains the absolute-joint chart and both identities.
        f0 = self.features(a, a, zero)
        f1 = self.features(b, a, valid)
        c = torch.cat((f0, f1, h1, p, p @ self.mode_embedding,
                       prognosis, role, torch.sigmoid(ready_logit)), dim=-1)
        condition = self.condition(c)
        aux = {'initial': j0, 'joints': joints, 'transitions': transitions,
               'phase': p, 'role': role, 'ready_logit': ready_logit,
               'history_valid': valid}
        return condition, aux

    def forward(self, noisy_action, timestep, raw_history):
        condition, _ = self.encode(raw_history)
        return self.denoiser(noisy_action, timestep, condition)

    def belief(self, raw_history):
        # Optional introspection only; the fixed forward contract returns epsilon.
        _, aux = self.encode(raw_history)
        return {'phase': aux['phase'], 'pending_role': aux['role'],
                'readiness': torch.sigmoid(aux['ready_logit'])}


def build_model(spec):
    return PhaseDiffusion(spec)


@torch.no_grad()
def soft_phase_targets(batch):
    """Training-only weak labels; all lookahead is confined to masked batch futures."""
    raw = batch['raw_obs']
    future = batch['future_obs']
    # Action slots are t-1..t+14; future slot j is AFTER action j.
    pre = torch.cat((raw[:, :1], raw[:, 1:2], future[:, 1:-1]), dim=1)
    valid = batch['mask'].squeeze(-1).to(pre.dtype)
    fm = batch['future_mask'].squeeze(-1).to(pre.dtype)
    state_valid = torch.cat((torch.ones_like(valid[:, :2]), fm[:, 1:-1]), dim=1)
    valid = valid * state_valid
    tcp = pre[:, :, 18:21]
    red = pre[:, :, 25:28]
    blue = pre[:, :, 32:35]
    rg = pre[:, :, 41:44]
    sig = torch.sigmoid
    red_goal_xy = torch.linalg.vector_norm(red[:, :, :2] - rg[:, :, :2], dim=-1)
    pending_blue = sig((0.075 - red_goal_xy) / 0.012)
    role = torch.stack((1.0 - pending_blue, pending_blue), dim=-1)
    obj = red * (1.0 - pending_blue.unsqueeze(-1)) + blue * pending_blue.unsqueeze(-1)
    width = pre[:, :, 7:9].sum(-1)
    closed_fingers = sig((0.050 - width) / 0.005)
    closed_command = ((1.0 - batch['native_action'][:, :, 7]) * 0.5).clamp(0.0, 1.0)
    distance = torch.linalg.vector_norm(tcp - obj, dim=-1)
    xy_distance = torch.linalg.vector_norm(tcp[:, :, :2] - obj[:, :, :2], dim=-1)
    aligned = sig((0.025 - xy_distance) / 0.006) * sig((0.014 - (tcp[:, :, 2] - obj[:, :, 2]).abs()) / 0.004)
    # Incoming red lowering/release is distinct from grasping the pending blue block.
    red_near_tcp = sig((0.035 - torch.linalg.vector_norm(tcp - red, dim=-1)) / 0.006)
    red_above_goal = sig((red[:, :, 2] - rg[:, :, 2] - 0.009) / 0.003)
    incoming = pending_blue * sig((0.065 - torch.linalg.vector_norm(tcp[:, :, :2] - red[:, :, :2], dim=-1)) / 0.010)
    incoming = incoming * sig((0.220 - tcp[:, :, 2]) / 0.020)
    incoming = incoming * (1.0 - (1.0 - closed_fingers) * (1.0 - red_near_tcp) * (1.0 - red_above_goal))
    # Up to four post-action states from each pre-state, never beyond the slice.
    end = (torch.arange(pre.shape[1], device=pre.device) + 3).clamp_max(pre.shape[1] - 1)
    post = future.index_select(1, end)
    look_valid = fm * fm.index_select(1, end)
    post_obj = post[:, :, 25:28] * (1.0 - pending_blue.unsqueeze(-1)) + post[:, :, 32:35] * pending_blue.unsqueeze(-1)
    do = post_obj - obj
    dt = post[:, :, 18:21] - tcp
    match = torch.exp(-0.5 * ((do - dt) / 0.008).square().sum(-1))
    upward_comotion = match * sig((do[:, :, 2] - 0.003) / 0.001)
    upward_comotion = upward_comotion * sig((torch.linalg.vector_norm(do, dim=-1) - 0.003) / 0.001) * look_valid
    elevated = sig((obj[:, :, 2] - 0.032) / 0.003)
    attached = closed_command * closed_fingers * sig((0.035 - distance) / 0.007)
    attached = attached * (elevated + (1.0 - elevated) * upward_comotion)
    backward = torch.cat((torch.zeros_like(obj[:, :1]), obj[:, 1:] - obj[:, :-1]), dim=1)
    motion = look_valid.unsqueeze(-1) * do + (1.0 - look_valid.unsqueeze(-1)) * backward
    translating = sig((obj[:, :, 2] - 0.255) / 0.012) * sig((torch.linalg.vector_norm(motion[:, :, :2], dim=-1) - 0.003) / 0.001)
    free = 1.0 - incoming
    unattached = free * (1.0 - attached)
    phase = torch.stack((incoming,
                         unattached * (1.0 - closed_command) * (1.0 - aligned),
                         unattached * (1.0 - closed_command) * aligned,
                         unattached * closed_command,
                         free * attached * (1.0 - translating),
                         free * attached * translating), dim=-1)
    confidence = ((phase.max(-1).values - 0.38) / 0.35).clamp(0.0, 1.0) * valid
    phase = 0.99 * phase + 0.01 / 6.0
    return phase, confidence, role, valid


def weighted_mean(value, weight):
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def compute_loss(model, batch, spec):
    history = batch['raw_obs']
    # Explicitly train two-frame reset rather than pretending an eight-frame buffer exists.
    reset = (torch.rand(history.shape[0], 1, 1, device=history.device) < 0.15)
    first = torch.where(reset, history[:, 1:2], history[:, :1])
    history_used = torch.cat((first, history[:, 1:2]), dim=1)
    condition, aux = model.encode(history_used)
    eps = model.denoiser(batch['noisy_action'], batch['timesteps'], condition)
    diffusion = epsilon_loss(eps, batch['noise'], batch['mask'])
    labels, confidence, roles, valid = soft_phase_targets(batch)
    p0 = aux['initial'].sum(-1)
    p1 = aux['phase']
    ce0 = -(labels[:, 0] * p0.clamp_min(1e-7).log()).sum(-1)
    ce1 = -(labels[:, 1] * p1.clamp_min(1e-7).log()).sum(-1)
    c0 = confidence[:, 0] * (~reset[:, 0, 0]).to(confidence.dtype)
    phase_loss = (ce0.mul(c0).sum() + ce1.mul(confidence[:, 1]).sum()) / (c0.sum() + confidence[:, 1].sum()).clamp_min(1.0)
    role_ce = -(roles[:, 1] * aux['role'].clamp_min(1e-7).log()).sum(-1)
    role_loss = weighted_mean(role_ce, valid[:, 1])
    ready_target = labels[:, 1, 4:6].sum(-1)
    ready_ce = F.binary_cross_entropy_with_logits(aux['ready_logit'].squeeze(-1), ready_target, reduction='none')
    readiness_loss = weighted_mean(ready_ce, confidence[:, 1])
    # Marginal duration/transition likelihood through the latent run-age filter.
    # Label conditioning is exclusively in this training loss, never the denoiser.
    q = aux['joints'][0]
    start_evidence = (1.0 - confidence[:, 1:2]) + confidence[:, 1:2] * labels[:, 1]
    q = q * start_evidence.unsqueeze(-1)
    q = q / q.sum((1, 2), keepdim=True).clamp_min(1e-8)
    duration_num = diffusion * 0.0
    duration_den = confidence.new_zeros(())
    skip_num = diffusion * 0.0
    skip_den = confidence.new_zeros(())
    for k, (hazard, jump) in enumerate(aux['transitions']):
        j = k + 2
        q = model.advance(q, hazard, jump)
        w = confidence[:, j]
        # A fully masked/ambiguous observation is unit evidence and contributes no loss.
        evidence = (1.0 - w.unsqueeze(-1)) + w.unsqueeze(-1) * labels[:, j]
        likelihood = (q.sum(-1) * evidence).sum(-1).clamp_min(1e-8)
        duration_num = duration_num - likelihood.log().sum()
        duration_den = duration_den + w.sum()
        q = q * evidence.unsqueeze(-1) / likelihood[:, None, None]
        forecast = aux['joints'][k]
        exit_mass = (forecast * hazard).sum(-1)
        skip = (exit_mass * (jump * model.skip_cost).sum(-1)).sum(-1)
        skip_num = skip_num + (skip * valid[:, j]).sum()
        skip_den = skip_den + valid[:, j].sum()
    duration_loss = duration_num / duration_den.clamp_min(1.0)
    skip_loss = skip_num / skip_den.clamp_min(1.0)
    prior = 0.15 * phase_loss + 0.08 * duration_loss + 0.03 * role_loss + 0.03 * readiness_loss + 0.01 * skip_loss
    total = diffusion + prior
    return {'loss': total, 'diffusion_loss': diffusion, 'prior_loss': prior,
            'phase_loss': phase_loss, 'duration_loss': duration_loss,
            'role_loss': role_loss, 'readiness_loss': readiness_loss, 'skip_loss': skip_loss}
