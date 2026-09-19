import torch
import appl.public as public


class RedRelativeLiftDiffusion(torch.nn.Module):
    """Diffusion policy with a red-object-relative state encoder.

    The action diffusion model remains the standard epsilon-predicting learned
    policy. The prior enters through causal red-relative conditioning features
    and through a learned auxiliary future-descriptor head used only for a
    differentiable training loss.
    """

    def __init__(self, spec):
        super().__init__()
        cfg = spec['candidate_config'] if 'candidate_config' in spec else {}
        self.horizon = int(spec['training']['horizon'] if 'horizon' in spec['training'] else 16)
        self.action_dim = 8
        self.cond_dim = int(cfg['condition_dim'] if 'condition_dim' in cfg else 256)
        self.phase_dim = int(cfg['phase_dim'] if 'phase_dim' in cfg else 5)
        self.aux_dim = 13
        self.prior_weight = float(cfg['prior_loss_weight'] if 'prior_loss_weight' in cfg else 0.05)

        n = spec['normalizer']
        self.register_buffer('obs_mean', torch.as_tensor(n['mean'], dtype=torch.float32))
        self.register_buffer('obs_std', torch.as_tensor(n['std'], dtype=torch.float32))
        self.register_buffer('action_min', torch.as_tensor(n['action_min'], dtype=torch.float32))
        self.register_buffer('action_scale', torch.as_tensor(n['action_scale'], dtype=torch.float32))
        # Fixed metric scales, deliberately not fit on the tiny skill slice.
        self.register_buffer('rel_pos_scale', torch.tensor([0.25, 0.25, 0.25], dtype=torch.float32))
        self.register_buffer('goal_pos_scale', torch.tensor([0.40, 0.40, 0.25], dtype=torch.float32))
        self.register_buffer('small_motion_scale', torch.tensor([0.06, 0.06, 0.08], dtype=torch.float32))
        self.register_buffer('arm_delta_scale', torch.tensor([0.35, 0.60, 0.30, 0.45, 0.20, 0.60, 0.50, 0.02, 0.02], dtype=torch.float32))
        self.register_buffer('aux_scale', torch.tensor([
            0.12, 0.12, 0.12,
            0.35, 0.35, 0.35,
            0.45, 0.45, 0.35,
            0.06, 0.06, 0.04,
            0.08
        ], dtype=torch.float32))

        # 2*47 normalized raw history + 2*18 relative features + 18 deltas + 6 scalar cues.
        feature_dim = 154
        self.phase_head = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, 64),
            torch.nn.Mish(),
            torch.nn.Linear(64, self.phase_dim)
        )
        self.condition_encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_dim + self.phase_dim, 256),
            torch.nn.Mish(),
            torch.nn.Linear(256, self.cond_dim),
            torch.nn.Mish(),
            torch.nn.Linear(self.cond_dim, self.cond_dim)
        )
        self.backbone = public.DiffusionBackbone(self.cond_dim, spec['training'])
        self.aux_head = torch.nn.Sequential(
            torch.nn.Linear(self.cond_dim + self.horizon * self.action_dim, 384),
            torch.nn.Mish(),
            torch.nn.Linear(384, 256),
            torch.nn.Mish(),
            torch.nn.Linear(256, self.horizon * self.aux_dim)
        )

    def normalize_raw_obs_local(self, raw_history):
        mean = self.obs_mean.to(device=raw_history.device, dtype=raw_history.dtype)
        std = self.obs_std.to(device=raw_history.device, dtype=raw_history.dtype)
        return (raw_history - mean) / std

    def relative_features_one_step(self, obs):
        tcp = obs[:, 18:21]
        red = obs[:, 25:28]
        blue = obs[:, 32:35]
        red_goal = obs[:, 41:44]
        blue_goal = obs[:, 44:47]
        rscale = self.rel_pos_scale.to(device=obs.device, dtype=obs.dtype)
        gscale = self.goal_pos_scale.to(device=obs.device, dtype=obs.dtype)
        return torch.cat([
            (tcp - red) / rscale,
            (tcp - blue) / rscale,
            (red - blue) / rscale,
            (red_goal - red) / gscale,
            (blue_goal - blue) / gscale,
            (red_goal - tcp) / gscale
        ], dim=-1)

    def encode_condition(self, raw_history):
        # raw_history is [B, 2, 47] and is the only observation input used at deployment.
        raw = raw_history
        dtype = raw.dtype
        norm_flat = self.normalize_raw_obs_local(raw).reshape(raw.shape[0], -1)
        prev = raw[:, 0]
        cur = raw[:, 1]

        rel_prev = self.relative_features_one_step(prev)
        rel_cur = self.relative_features_one_step(cur)

        tcp_delta = (cur[:, 18:21] - prev[:, 18:21]) / self.small_motion_scale.to(raw.device, dtype=dtype)
        red_delta = (cur[:, 25:28] - prev[:, 25:28]) / self.small_motion_scale.to(raw.device, dtype=dtype)
        blue_delta = (cur[:, 32:35] - prev[:, 32:35]) / self.small_motion_scale.to(raw.device, dtype=dtype)
        qpos_delta = (cur[:, 0:9] - prev[:, 0:9]) / self.arm_delta_scale.to(raw.device, dtype=dtype)

        tcp = cur[:, 18:21]
        red = cur[:, 25:28]
        blue = cur[:, 32:35]
        red_goal = cur[:, 41:44]
        finger_sum = cur[:, 7:8] + cur[:, 8:9]
        tcp_above_red = (tcp[:, 2:3] - red[:, 2:3]) / 0.25
        red_clearance = (red[:, 2:3] - blue[:, 2:3] - 0.04) / 0.30
        tcp_red_xy = torch.sqrt(((tcp[:, 0:2] - red[:, 0:2]) * (tcp[:, 0:2] - red[:, 0:2])).sum(dim=-1, keepdim=True) + 1.0e-9) / 0.20
        red_goal_xy = torch.sqrt(((red_goal[:, 0:2] - red[:, 0:2]) * (red_goal[:, 0:2] - red[:, 0:2])).sum(dim=-1, keepdim=True) + 1.0e-9) / 0.50
        blue_motion_xy = torch.sqrt(((cur[:, 32:34] - prev[:, 32:34]) * (cur[:, 32:34] - prev[:, 32:34])).sum(dim=-1, keepdim=True) + 1.0e-9) / 0.04
        scalar = torch.cat([finger_sum / 0.08, tcp_above_red, red_clearance, tcp_red_xy, red_goal_xy, blue_motion_xy], dim=-1)

        features = torch.cat([
            norm_flat, rel_prev, rel_cur, tcp_delta, red_delta, blue_delta, qpos_delta, scalar
        ], dim=-1)
        phase = torch.softmax(self.phase_head(features), dim=-1)
        cond = self.condition_encoder(torch.cat([features, phase], dim=-1))
        return cond

    def forward(self, noisy_action, timestep, raw_history):
        cond = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, cond)

    def clean_action_estimate(self, noisy_action, predicted_epsilon, alpha_bar):
        ab = alpha_bar.reshape(alpha_bar.shape[0], 1, 1).to(device=noisy_action.device, dtype=noisy_action.dtype)
        ab = ab.clamp(1.0e-4, 0.9999)
        return (noisy_action - torch.sqrt(1.0 - ab) * predicted_epsilon) / torch.sqrt(ab)

    def predict_future_descriptors(self, cond, clean_encoded_action):
        # Bound the high-noise DDPM x0 estimate before the auxiliary predictor.
        bounded = 1.5 * torch.tanh(clean_encoded_action / 1.5)
        x = torch.cat([cond, bounded.reshape(bounded.shape[0], -1)], dim=-1)
        return self.aux_head(x).reshape(bounded.shape[0], self.horizon, self.aux_dim)


def build_model(spec):
    return RedRelativeLiftDiffusion(spec)


def future_descriptor_targets(model, raw_history, future_obs):
    cur = raw_history[:, 1]
    tcp_f = future_obs[:, :, 18:21]
    red_f = future_obs[:, :, 25:28]
    blue_f = future_obs[:, :, 32:35]
    red_goal_f = future_obs[:, :, 41:44]
    current_blue = cur[:, None, 32:35]
    finger_sum_f = future_obs[:, :, 7:8] + future_obs[:, :, 8:9]
    target = torch.cat([
        tcp_f - red_f,
        red_f - blue_f,
        red_goal_f - red_f,
        blue_f - current_blue,
        finger_sum_f
    ], dim=-1)
    scale = model.aux_scale.to(device=future_obs.device, dtype=future_obs.dtype).reshape(1, 1, -1)
    return target / scale


def compute_loss(model, batch, spec):
    noisy = batch['noisy_action']
    pred = model(noisy, batch['timesteps'], batch['raw_obs'])
    mask = batch['mask']
    diffusion_loss = public.epsilon_loss(pred, batch['noise'], mask)

    cond = model.encode_condition(batch['raw_obs'])
    clean_est = model.clean_action_estimate(noisy, pred, batch['alpha_bar'])
    aux_pred = model.predict_future_descriptors(cond, clean_est)
    aux_target = future_descriptor_targets(model, batch['raw_obs'], batch['future_obs'])
    if 'future_mask' in batch:
        fmask = batch['future_mask'].to(device=aux_pred.device, dtype=aux_pred.dtype)
    else:
        fmask = mask.to(device=aux_pred.device, dtype=aux_pred.dtype)
    prior_loss = ((aux_pred - aux_target).square() * fmask).sum() / (fmask.sum() * aux_pred.shape[-1]).clamp_min(1.0)

    total = diffusion_loss + model.prior_weight * prior_loss
    return {'loss': total, 'diffusion_loss': diffusion_loss, 'prior_loss': prior_loss}
