import math
import torch
import appl.public as public


PHASE_COUNT = 5


def normalizer_from_spec(spec):
    if 'normalizer' in spec:
        return spec['normalizer']
    if 'shared_normalizer' in spec:
        return spec['shared_normalizer']
    raise KeyError('spec must contain normalizer')


def sinusoidal_embedding(timestep, dim, device, dtype):
    if not torch.is_tensor(timestep):
        timestep = torch.tensor(timestep, device=device)
    timestep = timestep.to(device=device)
    if timestep.dim() == 0:
        timestep = timestep[None]
    timestep = timestep.to(dtype=dtype)
    half = dim // 2
    if half <= 1:
        return timestep[:, None]
    freqs = torch.exp(
        torch.arange(half, device=device, dtype=dtype) * (-(math.log(10000.0) / (half - 1)))
    )
    args = timestep[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=device, dtype=dtype)], dim=-1)
    return emb


def select_active_pose(obs, active_is_blue):
    red = obs[..., 25:32]
    blue = obs[..., 32:39]
    if obs.dim() == 2:
        gate = active_is_blue.to(dtype=obs.dtype, device=obs.device).view(-1, 1)
    else:
        gate = active_is_blue.to(dtype=obs.dtype, device=obs.device).view(-1, 1, 1)
    return red * (1.0 - gate) + blue * gate


def hard_active_is_blue_from_current(current_obs):
    red = current_obs[:, 25:28]
    red_goal = current_obs[:, 41:44]
    red_goal_xy = torch.linalg.norm(red[:, :2] - red_goal[:, :2], dim=-1)
    red_low = red[:, 2] < 0.08
    red_in_region = red_goal_xy < 0.09
    return (red_low & red_in_region).to(dtype=current_obs.dtype)


def phase_labels_from_obs(obs, active_is_blue):
    active = select_active_pose(obs, active_is_blue)[..., :3]
    tcp = obs[..., 18:21]
    finger_width = obs[..., 7] + obs[..., 8]
    finger_vel = obs[..., 16] + obs[..., 17]
    xy_dist = torch.linalg.norm(active[..., :2] - tcp[..., :2], dim=-1)
    active_z = active[..., 2]
    tcp_z = tcp[..., 2]

    labels = torch.zeros_like(active_z, dtype=torch.long)
    align = (xy_dist < 0.060) & (tcp_z < 0.105) & (active_z < 0.060) & (finger_width > 0.055)
    close = ((finger_width < 0.070) | (finger_vel < -0.002)) & (active_z < 0.070)
    lift = (active_z >= 0.070) & (active_z < 0.120) & (finger_width < 0.065)
    carry = (active_z >= 0.120) & (finger_width < 0.065)
    labels = torch.where(align, torch.ones_like(labels), labels)
    labels = torch.where(close, torch.full_like(labels, 2), labels)
    labels = torch.where(lift, torch.full_like(labels, 3), labels)
    labels = torch.where(carry, torch.full_like(labels, 4), labels)
    return labels


def make_mlp(sizes, final_activation=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or final_activation:
            layers.append(torch.nn.SiLU())
            if i < len(sizes) - 2:
                layers.append(torch.nn.LayerNorm(sizes[i + 1]))
    return torch.nn.Sequential(*layers)


class ContactPhaseBottleneckPolicy(torch.nn.Module):
    """Learned DDPM epsilon model with an auxiliary contact-phase bottleneck."""

    def __init__(self, spec):
        super().__init__()
        normalizer = normalizer_from_spec(spec)
        self.horizon = int(spec['training']['horizon'])
        self.condition_dim = 384
        self.feature_dim = 135
        self.time_residual_dim = 32
        self.step_dim = 16

        self.register_buffer('obs_mean', torch.tensor(normalizer['mean'], dtype=torch.float32), persistent=False)
        self.register_buffer('obs_std', torch.tensor(normalizer['std'], dtype=torch.float32), persistent=False)

        self.encoder = make_mlp([self.feature_dim, 256, 256, 256])
        self.current_phase_head = torch.nn.Linear(256, PHASE_COUNT)
        self.horizon_phase_head = torch.nn.Linear(256, self.horizon * PHASE_COUNT)
        self.aux_head = torch.nn.Linear(256, self.horizon * 2)
        self.condition_projector = make_mlp([256 + PHASE_COUNT + self.horizon * PHASE_COUNT, 384, self.condition_dim])
        self.backbone = public.DiffusionBackbone(self.condition_dim, spec['training'])

        self.hidden_step = torch.nn.Linear(256, 64)
        self.step_embedding = torch.nn.Parameter(torch.zeros(self.horizon, self.step_dim))
        torch.nn.init.normal_(self.step_embedding, std=0.02)
        residual_in = 8 + PHASE_COUNT + self.time_residual_dim + 64 + self.step_dim
        self.residual_decoder = make_mlp([residual_in, 128, 128, 8])
        self.residual_scale = torch.nn.Parameter(torch.tensor(0.05))

    def features(self, raw_history):
        dtype = raw_history.dtype
        device = raw_history.device
        mean = self.obs_mean.to(device=device, dtype=dtype)
        std = self.obs_std.to(device=device, dtype=dtype).clamp_min(1.0e-6)
        norm_hist = (raw_history - mean.view(1, 1, -1)) / std.view(1, 1, -1)
        flat = norm_hist.reshape(raw_history.shape[0], -1)

        prev = raw_history[:, 0]
        cur = raw_history[:, -1]
        tcp = cur[:, 18:21]
        red = cur[:, 25:28]
        blue = cur[:, 32:35]
        red_goal = cur[:, 41:44]
        blue_goal = cur[:, 44:47]

        tcp_vel = (cur[:, 18:21] - prev[:, 18:21]) / 0.25
        red_vel = (cur[:, 25:28] - prev[:, 25:28]) / 0.25
        blue_vel = (cur[:, 32:35] - prev[:, 32:35]) / 0.25
        motion = torch.cat([tcp_vel, red_vel, blue_vel], dim=-1)

        finger_width = cur[:, 7:8] + cur[:, 8:9]
        prev_width = prev[:, 7:8] + prev[:, 8:9]
        finger = torch.cat([finger_width / 0.08, (finger_width - prev_width) / 0.04], dim=-1)

        red_goal_xy = torch.linalg.norm(red[:, :2] - red_goal[:, :2], dim=-1, keepdim=True)
        red_done_soft = torch.sigmoid((0.08 - red_goal_xy) / 0.025) * torch.sigmoid((0.08 - red[:, 2:3]) / 0.020)
        active_blue = red_done_soft.clamp(0.0, 1.0)
        active = red * (1.0 - active_blue) + blue * active_blue
        active_goal = red_goal * (1.0 - active_blue) + blue_goal * active_blue

        pos_scale = torch.tensor([0.35, 0.35, 0.30], device=device, dtype=dtype).view(1, 3)
        rel_features = torch.cat([
            (red - tcp) / pos_scale,
            (blue - tcp) / pos_scale,
            (active - tcp) / pos_scale,
            (red - red_goal) / pos_scale,
            (blue - blue_goal) / pos_scale,
            (active - active_goal) / pos_scale,
        ], dim=-1)

        red_tcp_xy = torch.linalg.norm(red[:, :2] - tcp[:, :2], dim=-1, keepdim=True) / 0.35
        blue_tcp_xy = torch.linalg.norm(blue[:, :2] - tcp[:, :2], dim=-1, keepdim=True) / 0.35
        active_tcp_xy = torch.linalg.norm(active[:, :2] - tcp[:, :2], dim=-1, keepdim=True) / 0.35
        blue_goal_xy = torch.linalg.norm(blue[:, :2] - blue_goal[:, :2], dim=-1, keepdim=True)
        scalars = torch.cat([
            red_tcp_xy,
            blue_tcp_xy,
            active_tcp_xy,
            red_goal_xy / 0.35,
            blue_goal_xy / 0.35,
            red[:, 2:3] / 0.30,
            blue[:, 2:3] / 0.30,
            active[:, 2:3] / 0.30,
            red_done_soft,
            active_blue,
            (red[:, 2:3] - 0.02) / 0.30,
            (blue[:, 2:3] - 0.02) / 0.30,
        ], dim=-1)
        return torch.cat([flat, motion, finger, rel_features, scalars], dim=-1)

    def condition_parts(self, raw_history):
        h = self.encoder(self.features(raw_history))
        current_logits = self.current_phase_head(h)
        horizon_logits = self.horizon_phase_head(h).view(raw_history.shape[0], self.horizon, PHASE_COUNT)
        current_probs = torch.softmax(current_logits, dim=-1)
        horizon_probs = torch.softmax(horizon_logits, dim=-1)
        cond_in = torch.cat([h, current_probs, horizon_probs.reshape(raw_history.shape[0], -1)], dim=-1)
        condition = self.condition_projector(cond_in)
        aux = self.aux_head(h).view(raw_history.shape[0], self.horizon, 2)
        return condition, h, current_logits, horizon_logits, horizon_probs, aux

    def auxiliary(self, raw_history):
        _, _, current_logits, horizon_logits, _, aux = self.condition_parts(raw_history)
        return current_logits, horizon_logits, aux

    def forward(self, noisy_action, timestep, raw_history):
        condition, h, _, _, horizon_probs, _ = self.condition_parts(raw_history)
        base = self.backbone(noisy_action, timestep, condition)

        b, horizon, _ = noisy_action.shape
        device = noisy_action.device
        dtype = noisy_action.dtype
        if horizon != self.horizon:
            phase_seq = horizon_probs[:, :horizon]
            step_emb = self.step_embedding[:horizon]
        else:
            phase_seq = horizon_probs
            step_emb = self.step_embedding
        t_emb = sinusoidal_embedding(timestep, self.time_residual_dim, device, dtype)
        if t_emb.shape[0] == 1 and b != 1:
            t_emb = t_emb.expand(b, -1)
        h_step = self.hidden_step(h).to(dtype=dtype)
        step = step_emb.to(device=device, dtype=dtype).view(1, horizon, self.step_dim).expand(b, -1, -1)
        residual_input = torch.cat([
            noisy_action,
            phase_seq.to(dtype=dtype),
            t_emb[:, None, :].expand(-1, horizon, -1),
            h_step[:, None, :].expand(-1, horizon, -1),
            step,
        ], dim=-1)
        residual = self.residual_decoder(residual_input.reshape(b * horizon, -1)).view(b, horizon, 8)
        return base + self.residual_scale.to(dtype=dtype) * residual


def build_model(spec):
    return ContactPhaseBottleneckPolicy(spec)


def compute_loss(model, batch, spec):
    predicted = model(batch['noisy_action'], batch['timesteps'], batch['raw_obs'])
    mask = batch['mask']
    diffusion_loss = public.epsilon_loss(predicted, batch['noise'], mask)

    raw_obs = batch['raw_obs']
    current_obs = raw_obs[:, -1]
    active_is_blue = hard_active_is_blue_from_current(current_obs)
    current_target = phase_labels_from_obs(current_obs, active_is_blue)
    future_target = phase_labels_from_obs(batch['future_obs'], active_is_blue)

    current_logits, horizon_logits, aux = model.auxiliary(raw_obs)
    ce_current = torch.nn.functional.cross_entropy(current_logits, current_target, reduction='mean')
    ce_future_all = torch.nn.functional.cross_entropy(
        horizon_logits.reshape(-1, PHASE_COUNT), future_target.reshape(-1), reduction='none'
    ).view_as(future_target).to(dtype=mask.dtype)
    m = batch['future_mask'].squeeze(-1).to(dtype=ce_future_all.dtype)
    ce_future = (ce_future_all * m).sum() / m.sum().clamp_min(1.0)

    active_future = select_active_pose(batch['future_obs'], active_is_blue)[..., :3]
    z_target = (active_future[..., 2] - 0.16) / 0.16
    width_target = ((batch['future_obs'][..., 7] + batch['future_obs'][..., 8]) - 0.058) / 0.04
    aux_target = torch.stack([z_target, width_target], dim=-1).to(dtype=aux.dtype)
    aux_mse = (((aux - aux_target) ** 2) * m[..., None]).sum() / (m.sum().clamp_min(1.0) * 2.0)

    alpha_bar = batch['alpha_bar'].to(device=predicted.device, dtype=predicted.dtype).view(-1, 1, 1).clamp(1.0e-5, 1.0)
    x0 = (batch['noisy_action'] - torch.sqrt(1.0 - alpha_bar) * predicted) / torch.sqrt(alpha_bar)
    native_x0 = public.denormalize_action(x0, spec)
    alpha_weight = (alpha_bar.view(-1, 1) > 0.20).to(dtype=native_x0.dtype)
    gm = mask.squeeze(-1).to(dtype=native_x0.dtype) * alpha_weight
    gripper_mse = (((native_x0[..., 7] - batch['native_action'][..., 7]) ** 2) * gm).sum() / gm.sum().clamp_min(1.0)

    prior_loss = 0.05 * ce_current + 0.05 * ce_future + 0.05 * aux_mse + 0.02 * gripper_mse
    total = diffusion_loss + prior_loss
    return {'loss': total, 'diffusion_loss': diffusion_loss, 'prior_loss': prior_loss}
