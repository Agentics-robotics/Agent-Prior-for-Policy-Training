import math
import torch
from torch import nn
from appl.public import DiffusionBackbone, epsilon_loss


class ContactDiffusion(nn.Module):
    """Causal two-frame contact filter with event-dependent diffusion conditioning."""
    def __init__(self, spec):
        super().__init__()
        n = spec['normalizer']
        self.horizon = int(spec['training']['horizon'])
        self.register_buffer('obs_mean', torch.tensor(n['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(n['std'], dtype=torch.float32))
        scales = torch.tensor(n['std'], dtype=torch.float32)
        xyz = torch.stack([scales[18:21], scales[25:28], scales[32:35]]).amax(0)
        self.register_buffer('xyz_scale', xyz)
        self.register_buffer('slot_ids', torch.arange(self.horizon))
        self.register_buffer('time_freq', torch.exp(-math.log(10000.0) * torch.arange(16).float() / 15.0))
        # 47 normalized state + 18 relative positions + 18 motion increments
        # + availability and motion-valid flags.
        self.frame = nn.Sequential(nn.Linear(85, 160), nn.SiLU(), nn.Linear(160, 128), nn.SiLU())
        self.filter = nn.GRUCell(128, 128)
        self.contact = nn.Linear(128, 8)
        self.continuous = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 32))
        self.relative_readout = nn.Linear(32, 18)
        self.future_contact = nn.Sequential(nn.Linear(160, 192), nn.SiLU(), nn.Linear(192, self.horizon * 8))
        self.hazard = nn.Sequential(nn.Linear(160, 128), nn.SiLU(), nn.Linear(128, self.horizon))
        nn.init.constant_(self.hazard[-1].bias, -3.0)
        self.backbone = DiffusionBackbone(128 + 32 + 8 + 1 + self.horizon, spec['training'])
        self.slot = nn.Embedding(self.horizon, 16)
        self.mode_embedding = nn.Linear(8, 32, bias=False)
        self.local_context = nn.Linear(160, 64)
        self.time_embedding = nn.Sequential(nn.Linear(32, 32), nn.SiLU())
        self.event_adapter = nn.Sequential(
            nn.Conv1d(152, 128, 3, padding=1), nn.SiLU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.SiLU(), nn.Conv1d(128, 8, 1))
        nn.init.zeros_(self.event_adapter[-1].weight)
        nn.init.zeros_(self.event_adapter[-1].bias)
        # A learned auxiliary dynamics model, not kinematics or an action controller.
        self.dyn_context = nn.Linear(160, 64)
        self.dyn_initial = nn.Linear(128, 128)
        self.dyn_filter = nn.GRUCell(88, 128)
        self.dyn_output = nn.Linear(128, 18)

    def relative(self, raw):
        tcp, red, blue = raw[..., 18:21], raw[..., 25:28], raw[..., 32:35]
        drawer_center = torch.stack([0.19 - raw[..., 39], torch.zeros_like(raw[..., 39]),
                                     torch.full_like(raw[..., 39], 0.035)], dim=-1)
        return torch.cat([(red - tcp) / self.xyz_scale,
                          (blue - tcp) / self.xyz_scale,
                          (red - raw[..., 41:44]) / self.xyz_scale,
                          (blue - raw[..., 44:47]) / self.xyz_scale,
                          (tcp - raw[..., 44:47]) / self.xyz_scale,
                          (blue - drawer_center) / self.xyz_scale], dim=-1)

    def motion_state(self, raw):
        tcp, red, blue = raw[..., 18:21], raw[..., 25:28], raw[..., 32:35]
        return torch.cat([tcp / self.xyz_scale, red / self.xyz_scale, blue / self.xyz_scale,
                          (red - tcp) / self.xyz_scale, (blue - tcp) / self.xyz_scale,
                          raw[..., 7:9] / self.obs_scale[7:9],
                          raw[..., 39:40] / self.obs_scale[39:40]], dim=-1)

    def encode(self, raw):
        b = raw.shape[0]
        # The API provides no history-valid flag. Identical left padding is treated
        # as one-frame input. This also conservatively masks truly identical pairs.
        available = ((raw[:, 1] - raw[:, 0]).abs().amax(-1, keepdim=True) > 1e-7).to(raw.dtype)
        norm = (raw - self.obs_mean) / self.obs_scale
        rel = self.relative(raw)
        motion = self.motion_state(raw)
        delta = (motion[:, 1] - motion[:, 0]) * available
        zero = torch.zeros_like(available)
        first = torch.cat([norm[:, 0], rel[:, 0], torch.zeros_like(delta), available, zero], dim=-1)
        second = torch.cat([norm[:, 1], rel[:, 1], delta, torch.ones_like(available), available], dim=-1)
        h0 = self.filter(self.frame(first), raw.new_zeros(b, 128)) * available
        h = self.filter(self.frame(second), h0)
        z = self.continuous(h)
        logp = self.contact(h).log_softmax(-1)
        p = logp.exp()
        entropy = -(p * logp).sum(-1, keepdim=True) / math.log(8.0)
        hz = torch.cat([h, z], dim=-1)
        future_logits = self.future_contact(hz).reshape(b, self.horizon, 8)
        hazards = self.hazard(hz)
        # Slot zero represents the already-observed state; no future hazard there.
        event_prob = torch.sigmoid(hazards)
        event_prob = torch.cat([torch.zeros_like(event_prob[:, :1]), event_prob[:, 1:]], dim=1)
        survival = (1.0 - event_prob).cumprod(dim=1)
        slot_fraction = self.slot_ids.to(raw.dtype)[None, :] / max(1, self.horizon - 1)
        retention = survival * torch.exp(-2.0 * entropy * slot_fraction)
        return {'h': h, 'z': z, 'p': p, 'logp': logp, 'entropy': entropy,
                'future_logits': future_logits, 'hazards': hazards,
                'event_prob': event_prob, 'retention': retention,
                'available': available, 'h0': h0}

    def decode(self, noisy_action, timestep, belief):
        b, hlen, _ = noisy_action.shape
        hz = torch.cat([belief['h'], belief['z']], dim=-1)
        condition = torch.cat([hz, belief['p'], belief['entropy'], belief['event_prob']], dim=-1)
        epsilon = self.backbone(noisy_action, timestep, condition)
        t = torch.as_tensor(timestep, dtype=noisy_action.dtype, device=noisy_action.device).reshape(-1)
        if t.numel() == 1:
            t = t.expand(b)
        angle = t[:, None] * self.time_freq[None, :]
        temb = self.time_embedding(torch.cat([angle.sin(), angle.cos()], dim=-1))
        gate = belief['retention'][:, :hlen, None]
        current_mode = self.mode_embedding(belief['p'])[:, None, :]
        future_mode = self.mode_embedding(belief['future_logits'][:, :hlen].softmax(-1))
        # Soft commitment only: attenuate the persistence of today's contact mode
        # across a likely transition. No execution scheduler is changed here.
        mode = gate * current_mode + (1.0 - gate) * future_mode
        local = torch.cat([noisy_action, mode,
                           self.local_context(hz)[:, None, :].expand(-1, hlen, -1),
                           self.slot(self.slot_ids[:hlen])[None].expand(b, -1, -1),
                           temb[:, None, :].expand(-1, hlen, -1)], dim=-1)
        return epsilon + self.event_adapter(local.transpose(1, 2)).transpose(1, 2)

    def forward(self, noisy_action, timestep, raw_history):
        return self.decode(noisy_action, timestep, self.encode(raw_history))

    def predict_motion(self, actions, belief):
        hz = torch.cat([belief['h'], belief['z']], dim=-1)
        context = self.dyn_context(hz)
        state = self.dyn_initial(belief['h'])
        slots = self.slot(self.slot_ids)
        outputs = []
        for j in range(actions.shape[1]):
            inp = torch.cat([actions[:, j], context, slots[j][None].expand(actions.shape[0], -1)], dim=-1)
            state = self.dyn_filter(inp, state)
            outputs.append(self.dyn_output(state))
        return torch.stack(outputs, dim=1)


def build_model(spec):
    return ContactDiffusion(spec)


def _comotion(first, last, valid):
    tcp_motion = last[..., 18:21] - first[..., 18:21]
    amount2 = tcp_motion.square().sum(-1)
    excitation = torch.sigmoid((amount2.sqrt() - 0.0015) / 0.001)
    scores = []
    for start in (25, 32):
        mismatch = last[..., start:start+3] - first[..., start:start+3] - tcp_motion
        agreement = torch.exp(-mismatch.square().sum(-1) / (0.003 ** 2 + 0.25 * amount2))
        score = excitation * agreement
        scores.append(valid * score + (1.0 - valid) * 0.5)
    return scores


def _contact_targets(raw, red_comotion, blue_comotion):
    """Weak soft labels, never used to generate actions or enforce success."""
    tcp, red, blue = raw[..., 18:21], raw[..., 25:28], raw[..., 32:35]
    width = raw[..., 7] + raw[..., 8]
    closed = torch.sigmoid((0.058 - width) / 0.005)
    red_near = torch.exp(-(red - tcp).square().sum(-1) / 0.05 ** 2)
    blue_near = torch.exp(-(blue - tcp).square().sum(-1) / 0.05 ** 2)
    red_low = torch.sigmoid((0.045 - red[..., 2]) / 0.006)
    red_pad = torch.exp(-(red[..., :2] - raw[..., 41:43]).square().sum(-1) / 0.07 ** 2)
    elevated = torch.sigmoid((blue[..., 2] - 0.038) / 0.006)
    goal_xy = torch.exp(-(blue[..., :2] - raw[..., 44:46]).square().sum(-1) / 0.07 ** 2)
    lowered = goal_xy * torch.sigmoid((0.09 - blue[..., 2]) / 0.008)
    placed = goal_xy * torch.exp(-((blue[..., 2] - raw[..., 46]) / 0.015).square())
    clearance = torch.sigmoid((tcp[..., 2] - blue[..., 2] - 0.13) / 0.025)
    rheld = closed * red_near * (0.65 + 0.35 * red_comotion)
    rrelease = (1.0 - closed) * red_near * red_low * red_pad
    bactive = closed * blue_near * (0.65 + 0.35 * blue_comotion)
    bsource = bactive * (1.0 - lowered) * (1.0 - elevated)
    btransport = bactive * (1.0 - lowered) * elevated
    blowered = bactive * lowered
    brelease = (1.0 - closed) * placed * (1.0 - clearance)
    terminal = (1.0 - closed) * placed * clearance
    occupied = rheld + rrelease + bsource + btransport + blowered + brelease + terminal
    free = (1.0 - occupied).clamp_min(0.02)
    weights = torch.stack([rheld, rrelease, free, bsource, btransport, blowered, brelease, terminal], dim=-1)
    p = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
    return 0.98 * p + 0.02 / 8.0


def _masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def compute_loss(model, batch, spec):
    raw = batch['raw_obs']
    b, hlen, _ = batch['noisy_action'].shape
    # Train one-frame takeovers as well as the complete two-frame history.
    drop = (torch.rand(b, 1, device=raw.device) < 0.30)
    first = torch.where(drop, raw[:, 1], raw[:, 0])
    history = torch.stack([first, raw[:, 1]], dim=1)
    predicted = model(batch['noisy_action'], batch['timesteps'], history)
    diffusion = epsilon_loss(predicted, batch['noise'], batch['mask'])
    belief = model.encode(history)
    with torch.no_grad():
        future = batch['future_obs']
        valid = (batch['future_mask'] * batch['mask']).squeeze(-1)
        # Future slot zero is the current state; slots j>0 are labels only.
        look_ids = (torch.arange(hlen, device=raw.device) + 3).clamp_max(hlen - 1)
        later = future.index_select(1, look_ids)
        look_valid = valid * valid.index_select(1, look_ids)
        # A shortened lookahead at the segment end is masked, not fabricated.
        look_valid = look_valid * ((look_ids - torch.arange(hlen, device=raw.device)) == 3).to(valid.dtype)[None]
        co_r, co_b = _comotion(future, later, look_valid)
        target_modes = _contact_targets(future, co_r, co_b)
        cur_co_r, cur_co_b = _comotion(raw[:, 1], future[:, min(3, hlen - 1)], valid[:, min(3, hlen - 1)])
        current_target = _contact_targets(raw[:, 1], cur_co_r, cur_co_b)
        old_co_r, old_co_b = _comotion(history[:, 0], raw[:, 1], belief['available'].squeeze(-1))
        old_target = _contact_targets(history[:, 0], old_co_r, old_co_b)
        modes = target_modes.argmax(-1)
        event_target = torch.zeros_like(valid)
        event_target[:, 1:] = (modes[:, 1:] != modes[:, :-1]).to(valid.dtype)
        event_valid = valid.clone()
        event_valid[:, 0] = 0.0
        event_valid[:, 1:] = valid[:, 1:] * valid[:, :-1]
        # Prefix masking affects auxiliaries only. Every valid action remains in
        # the DDPM objective, including the overlap and retained continuation.
        choices = torch.tensor([4, 8, 16], device=raw.device)
        prefix = choices[torch.randint(3, (b,), device=raw.device)]
        prefix_mask = (torch.arange(hlen, device=raw.device)[None] < prefix[:, None]).to(valid.dtype)
        aux_valid = valid * prefix_mask
        event_valid = event_valid * prefix_mask
        target_motion = model.motion_state(future) - model.motion_state(raw[:, 1])[:, None]
        target_relative = model.relative(raw[:, 1])

    current_ce = -(current_target * belief['logp']).sum(-1).mean()
    previous_logp = model.contact(belief['h0']).log_softmax(-1)
    old_ce = _masked_mean(-(old_target * previous_logp).sum(-1), belief['available'].squeeze(-1))
    future_ce = _masked_mean(-(target_modes * belief['future_logits'].log_softmax(-1)).sum(-1), aux_valid)
    hazard_ce = nn.functional.binary_cross_entropy_with_logits(belief['hazards'], event_target, reduction='none')
    # Moderate positive weighting for sparse contact changes; not calibration.
    hazard_loss = _masked_mean(hazard_ce * (1.0 + 3.0 * event_target), event_valid)
    relative_loss = nn.functional.smooth_l1_loss(model.relative_readout(belief['z']), target_relative, beta=0.1)

    alpha = batch['alpha_bar'].reshape(b, 1, 1).clamp_min(1e-6)
    clean = (batch['noisy_action'] - (1.0 - alpha).sqrt() * predicted) / alpha.sqrt()
    # This clipping is only for the auxiliary learned predictor, not DDPM output.
    clean = clean.clamp(-2.0, 2.0)
    teacher_motion = model.predict_motion(batch['encoded_action'], belief)
    denoised_motion = model.predict_motion(clean, belief)
    teacher_error = nn.functional.smooth_l1_loss(teacher_motion, target_motion, reduction='none', beta=0.1).mean(-1)
    denoised_error = nn.functional.smooth_l1_loss(denoised_motion, target_motion, reduction='none', beta=0.1).mean(-1)
    teacher_loss = _masked_mean(teacher_error, aux_valid)
    # Multiplication (rather than renormalizing by alpha) suppresses unstable
    # high-noise clean-estimate gradients while retaining a denoiser gradient.
    coupled_loss = _masked_mean(denoised_error * alpha[:, 0, 0, None], aux_valid)
    prior = (0.10 * current_ce + 0.025 * old_ce + 0.05 * future_ce +
             0.05 * hazard_loss + 0.05 * relative_loss +
             0.15 * teacher_loss + 0.05 * coupled_loss)
    return {'loss': diffusion + prior, 'diffusion_loss': diffusion, 'prior_loss': prior,
            'contact_loss': current_ce, 'future_contact_loss': future_ce,
            'event_loss': hazard_loss, 'relative_loss': relative_loss,
            'motion_loss': teacher_loss, 'coupled_motion_loss': coupled_loss}
