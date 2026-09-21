import torch
from appl.public import DiffusionBackbone, normalize_observation, epsilon_loss


RED_XYZ = slice(25, 28)
BLUE_XYZ = slice(32, 35)
TCP_XYZ = slice(18, 21)
RED_GOAL = slice(41, 44)
BLUE_GOAL = slice(44, 47)


class TraySceneEncoder(torch.nn.Module):
    """Causal scene encoder with explicit two-object/tray geometric features."""

    def __init__(self, spec, output_dim=256):
        super().__init__()
        self.spec = spec
        engineered_per_step = 40
        input_dim = 2 * 47 + 2 * engineered_per_step
        hidden = 256
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.Mish(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Mish(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, output_dim),
            torch.nn.Mish(),
        )

    def signed_distances(self, xy):
        x = xy[..., 0]
        y = xy[..., 1]
        xmin = -0.23
        xmax = -0.01
        ymin = -0.15
        ymax = 0.15
        return torch.stack((x - xmin, xmax - x, y - ymin, ymax - y), dim=-1) / 0.15

    def target_box_distances(self, xyz, goal):
        d = xyz[..., :2] - goal[..., :2]
        half = 0.06
        sx = half - torch.abs(d[..., 0])
        sy = half - torch.abs(d[..., 1])
        zerr = xyz[..., 2] - goal[..., 2]
        return torch.stack((sx / half, sy / half, zerr / 0.08), dim=-1)

    def engineer_one(self, obs):
        red = obs[..., RED_XYZ]
        blue = obs[..., BLUE_XYZ]
        tcp = obs[..., TCP_XYZ]
        rg = obs[..., RED_GOAL]
        bg = obs[..., BLUE_GOAL]
        qpos = obs[..., 0:9]
        qvel = obs[..., 9:18]
        grip = 0.5 * (qpos[..., 7:8] + qpos[..., 8:9])
        grip_vel = 0.5 * (qvel[..., 7:8] + qvel[..., 8:9])

        metric_scale = 0.25
        rel = [
            (red - rg) / metric_scale,
            (blue - bg) / metric_scale,
            (tcp - red) / metric_scale,
            (tcp - blue) / metric_scale,
            (red - blue) / metric_scale,
            (red - bg) / metric_scale,
            (blue - rg) / metric_scale,
        ]
        zfeat = torch.cat(((red[..., 2:3] - 0.036) / 0.15,
                           (blue[..., 2:3] - 0.036) / 0.15,
                           (tcp[..., 2:3] - 0.12) / 0.25), dim=-1)
        tray = torch.cat((self.signed_distances(red[..., :2]),
                          self.signed_distances(blue[..., :2])), dim=-1)
        goal_box = torch.cat((self.target_box_distances(red, rg),
                              self.target_box_distances(blue, bg)), dim=-1)
        grip_feat = torch.cat(((grip - 0.0275) / 0.0125, grip_vel / 0.05), dim=-1)
        return torch.cat(rel + [zfeat, tray, goal_box, grip_feat], dim=-1)

    def forward(self, raw_history):
        norm = normalize_observation(raw_history, self.spec).reshape(raw_history.shape[0], -1)
        eng0 = self.engineer_one(raw_history[:, 0])
        eng1 = self.engineer_one(raw_history[:, 1])
        return self.net(torch.cat((norm, eng0, eng1), dim=-1))


class TrayNoninterferenceDiffusion(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        cond_dim = int(spec.get('candidate_config', {}).get('condition_dim', 256))
        self.encoder = TraySceneEncoder(spec, cond_dim)
        self.backbone = DiffusionBackbone(cond_dim, spec['training'])
        horizon = spec['training']['horizon']

        self.action_encoder = torch.nn.Sequential(
            torch.nn.Linear(horizon * 8, 256),
            torch.nn.Mish(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 128),
            torch.nn.Mish(),
        )
        self.future_head = torch.nn.Sequential(
            torch.nn.Linear(cond_dim + 128, 256),
            torch.nn.Mish(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 256),
            torch.nn.Mish(),
            torch.nn.Linear(256, horizon * 2 * 3),
        )
        self.role_head = torch.nn.Sequential(
            torch.nn.Linear(cond_dim, 128),
            torch.nn.Mish(),
            torch.nn.Linear(128, 3),
        )

    def encode_condition(self, raw_history):
        return self.encoder(raw_history)

    def forward(self, noisy_action, timestep, raw_history):
        cond = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, cond)

    def auxiliary_predictions(self, raw_history, clean_action_estimate):
        cond = self.encode_condition(raw_history)
        safe_action = 2.0 * torch.tanh(clean_action_estimate / 2.0)
        act = self.action_encoder(safe_action.reshape(safe_action.shape[0], -1))
        out = self.future_head(torch.cat((cond, act), dim=-1))
        h = self.spec['training']['horizon']
        object_delta = out.reshape(raw_history.shape[0], h, 2, 3)
        role_logits = self.role_head(cond)
        return object_delta, role_logits


def build_model(spec):
    return TrayNoninterferenceDiffusion(spec)


def masked_mean(value, mask, feature_count):
    while mask.dim() < value.dim():
        mask = mask.unsqueeze(-1)
    denom = (mask.sum() * feature_count).clamp_min(1.0)
    return (value * mask).sum() / denom


def role_targets(raw_obs, future_obs, future_mask):
    cur = raw_obs[:, 1]
    red0 = cur[:, RED_XYZ]
    blue0 = cur[:, BLUE_XYZ]
    tcp = cur[:, TCP_XYZ]
    qpos = cur[:, 0:9]
    grip = 0.5 * (qpos[:, 7] + qpos[:, 8])

    red_f = future_obs[..., RED_XYZ]
    blue_f = future_obs[..., BLUE_XYZ]
    m = future_mask.squeeze(-1)
    denom = m.sum(dim=1).clamp_min(1.0)
    red_motion = (torch.linalg.norm(red_f - red0[:, None, :], dim=-1) * m).sum(dim=1) / denom
    blue_motion = (torch.linalg.norm(blue_f - blue0[:, None, :], dim=-1) * m).sum(dim=1) / denom

    red_close = torch.linalg.norm(tcp - red0, dim=-1) < 0.065
    blue_close = torch.linalg.norm(tcp - blue0, dim=-1) < 0.065
    closed = grip < 0.030
    red_held = (red0[:, 2] > 0.060) & red_close & closed
    blue_held = (blue0[:, 2] > 0.060) & blue_close & closed

    red_active = red_held | ((red_motion > 0.006) & (red_motion > blue_motion + 0.003))
    blue_active = blue_held | ((blue_motion > 0.006) & (blue_motion > red_motion + 0.003))
    target = torch.zeros(raw_obs.shape[0], device=raw_obs.device, dtype=torch.long)
    target = torch.where(red_active, torch.ones_like(target), target)
    target = torch.where(blue_active & (~red_active), torch.full_like(target, 2), target)
    return target


def xy_bounds_violation(pos):
    xmin = -0.23
    xmax = -0.01
    ymin = -0.15
    ymax = 0.15
    vx = torch.relu(xmin - pos[..., 0]) + torch.relu(pos[..., 0] - xmax)
    vy = torch.relu(ymin - pos[..., 1]) + torch.relu(pos[..., 1] - ymax)
    return vx.square() + vy.square()


def compute_loss(model, batch, spec):
    raw_obs = batch['raw_obs']
    noisy_action = batch['noisy_action']
    noise = batch['noise']
    timesteps = batch['timesteps']
    mask = batch['mask']
    future_obs = batch['future_obs']
    future_mask = batch['future_mask']
    alpha_bar = batch['alpha_bar']

    pred_noise = model(noisy_action, timesteps, raw_obs)
    diffusion_loss = epsilon_loss(pred_noise, noise, mask)

    ab = alpha_bar.reshape(-1, 1, 1).to(dtype=noisy_action.dtype, device=noisy_action.device)
    ab = ab.clamp_min(0.02)
    clean_est = (noisy_action - torch.sqrt(1.0 - ab) * pred_noise) / torch.sqrt(ab)

    pred_delta_scaled, role_logits = model.auxiliary_predictions(raw_obs, clean_est)
    cur = raw_obs[:, 1]
    red0 = cur[:, RED_XYZ]
    blue0 = cur[:, BLUE_XYZ]
    rg = cur[:, RED_GOAL]
    bg = cur[:, BLUE_GOAL]

    target_red_delta = (future_obs[..., RED_XYZ] - red0[:, None, :]) / 0.25
    target_blue_delta = (future_obs[..., BLUE_XYZ] - blue0[:, None, :]) / 0.25
    target_delta = torch.stack((target_red_delta, target_blue_delta), dim=2)
    dyn_err = torch.nn.functional.smooth_l1_loss(pred_delta_scaled, target_delta, reduction='none')
    dynamics_loss = masked_mean(dyn_err, future_mask, 2 * 3)

    role_target = role_targets(raw_obs, future_obs, future_mask)
    role_loss = torch.nn.functional.cross_entropy(role_logits, role_target)
    red_active = (role_target == 1).to(dtype=noisy_action.dtype).reshape(-1, 1, 1)
    blue_active = (role_target == 2).to(dtype=noisy_action.dtype).reshape(-1, 1, 1)
    none_active = (role_target == 0).to(dtype=noisy_action.dtype).reshape(-1, 1, 1)

    pred_red_delta = pred_delta_scaled[:, :, 0, :]
    pred_blue_delta = pred_delta_scaled[:, :, 1, :]
    inactive_err = (red_active * pred_blue_delta.square()
                    + blue_active * pred_red_delta.square()
                    + none_active * (pred_red_delta.square() + pred_blue_delta.square()))
    noninterference_loss = masked_mean(inactive_err, future_mask, 3)

    pred_red_pos = red0[:, None, :] + 0.25 * pred_red_delta
    pred_blue_pos = blue0[:, None, :] + 0.25 * pred_blue_delta
    red_goal_xy = torch.linalg.norm(future_obs[..., RED_XYZ][..., :2] - rg[:, None, :2], dim=-1, keepdim=True)
    blue_goal_xy = torch.linalg.norm(future_obs[..., BLUE_XYZ][..., :2] - bg[:, None, :2], dim=-1, keepdim=True)
    red_goal_gate = red_active * (red_goal_xy < 0.09).to(dtype=noisy_action.dtype) * future_mask
    blue_goal_gate = blue_active * (blue_goal_xy < 0.09).to(dtype=noisy_action.dtype) * future_mask
    red_goal_err = ((pred_red_pos[..., :2] - rg[:, None, :2]) / 0.06).square().sum(dim=-1, keepdim=True)
    blue_goal_err = ((pred_blue_pos[..., :2] - bg[:, None, :2]) / 0.06).square().sum(dim=-1, keepdim=True)
    z_gate_red = red_goal_gate * (future_obs[..., RED_XYZ][..., 2:3] < 0.08).to(dtype=noisy_action.dtype)
    z_gate_blue = blue_goal_gate * (future_obs[..., BLUE_XYZ][..., 2:3] < 0.08).to(dtype=noisy_action.dtype)
    red_z_err = ((pred_red_pos[..., 2:3] - rg[:, None, 2:3]) / 0.05).square()
    blue_z_err = ((pred_blue_pos[..., 2:3] - bg[:, None, 2:3]) / 0.05).square()
    goal_num = (red_goal_err * red_goal_gate + blue_goal_err * blue_goal_gate
                + red_z_err * z_gate_red + blue_z_err * z_gate_blue).sum()
    goal_den = (red_goal_gate.sum() + blue_goal_gate.sum() + z_gate_red.sum() + z_gate_blue.sum()).clamp_min(1.0)
    goal_loss = goal_num / goal_den

    contain_num = (xy_bounds_violation(pred_red_pos).unsqueeze(-1) * red_goal_gate
                   + xy_bounds_violation(pred_blue_pos).unsqueeze(-1) * blue_goal_gate).sum()
    contain_den = (red_goal_gate.sum() + blue_goal_gate.sum()).clamp_min(1.0)
    containment_loss = contain_num / contain_den / (0.05 * 0.05)

    red_low_gate = red_goal_gate * (future_obs[..., RED_XYZ][..., 2:3] < 0.10).to(dtype=noisy_action.dtype)
    blue_low_gate = blue_goal_gate * (future_obs[..., BLUE_XYZ][..., 2:3] < 0.10).to(dtype=noisy_action.dtype)
    margin = 0.055
    red_to_blue0 = torch.linalg.norm(pred_red_pos[..., :2] - blue0[:, None, :2], dim=-1, keepdim=True)
    blue_to_red0 = torch.linalg.norm(pred_blue_pos[..., :2] - red0[:, None, :2], dim=-1, keepdim=True)
    keepout = (torch.relu(margin - red_to_blue0).square() * red_low_gate
               + torch.relu(margin - blue_to_red0).square() * blue_low_gate)
    keepout_loss = keepout.sum() / (red_low_gate.sum() + blue_low_gate.sum()).clamp_min(1.0) / (margin * margin)

    prior_loss = (0.05 * dynamics_loss
                  + 0.02 * role_loss
                  + 0.05 * noninterference_loss
                  + 0.02 * goal_loss
                  + 0.01 * containment_loss
                  + 0.01 * keepout_loss)
    loss = diffusion_loss + prior_loss
    return {
        'loss': loss,
        'diffusion_loss': diffusion_loss,
        'prior_loss': prior_loss,
    }
