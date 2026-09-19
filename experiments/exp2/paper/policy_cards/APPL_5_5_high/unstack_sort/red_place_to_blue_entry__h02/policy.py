import torch
from appl.public import DiffusionBackbone, epsilon_loss


class ReleaseResetDiffusionPolicy(torch.nn.Module):
    """Learned diffusion policy with a causal release/reset phase latent."""

    def __init__(self, spec):
        super().__init__()
        self.horizon = int(spec["training"]["horizon"])
        self.action_dim = 8
        n = spec["normalizer"]
        self.register_buffer("obs_mean", torch.tensor(n["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.tensor(n["std"], dtype=torch.float32))
        self.register_buffer("action_min", torch.tensor(n["action_min"], dtype=torch.float32))
        self.register_buffer("action_scale", torch.tensor(n["action_scale"], dtype=torch.float32))

        feature_dim = 2 * 47 + 47 + 41
        hidden = 256
        phase_dim = 32
        cond_dim = 256

        self.feature_net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden),
            torch.nn.Mish(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Mish(),
        )
        self.stage_head = torch.nn.Linear(hidden, 4)
        self.phase_embeddings = torch.nn.Parameter(torch.randn(4, phase_dim) * 0.02)
        self.cond_net = torch.nn.Sequential(
            torch.nn.Linear(hidden + 4 + phase_dim, cond_dim),
            torch.nn.Mish(),
            torch.nn.LayerNorm(cond_dim),
            torch.nn.Linear(cond_dim, cond_dim),
            torch.nn.Mish(),
        )
        self.backbone = DiffusionBackbone(cond_dim, spec["training"])
        self.gripper_intent_head = torch.nn.Sequential(
            torch.nn.Linear(hidden + 4, hidden),
            torch.nn.Mish(),
            torch.nn.Linear(hidden, self.horizon),
        )
        self.red_future_head = torch.nn.Sequential(
            torch.nn.Linear(cond_dim + self.horizon * self.action_dim, hidden),
            torch.nn.Mish(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Mish(),
            torch.nn.Linear(hidden, self.horizon * 3),
        )

    def normalize_obs(self, raw_history):
        return (raw_history - self.obs_mean.to(raw_history.dtype)) / self.obs_std.to(raw_history.dtype)

    def vector_norm(self, x, keepdim=True):
        return torch.sqrt(torch.sum(x * x, dim=-1, keepdim=keepdim) + 1.0e-8)

    def make_features(self, raw_history):
        raw = raw_history
        normed = self.normalize_obs(raw)
        cur = raw[:, -1, :]
        prev = raw[:, 0, :]
        ncur = normed[:, -1, :]
        nprev = normed[:, 0, :]

        norm_flat = normed.reshape(raw.shape[0], -1)
        norm_delta = ncur - nprev

        tcp = cur[:, 18:21]
        tcp_prev = prev[:, 18:21]
        red = cur[:, 25:28]
        red_prev = prev[:, 25:28]
        blue = cur[:, 32:35]
        blue_prev = prev[:, 32:35]
        red_goal = cur[:, 41:44]
        blue_goal = cur[:, 44:47]

        tcp_scale = self.obs_std[18:21].to(device=raw.device, dtype=raw.dtype).clamp_min(1.0e-3)
        red_scale = self.obs_std[25:28].to(device=raw.device, dtype=raw.dtype).clamp_min(1.0e-3)
        blue_scale = self.obs_std[32:35].to(device=raw.device, dtype=raw.dtype).clamp_min(1.0e-3)

        width_cur = cur[:, 7:8] + cur[:, 8:9]
        width_prev = prev[:, 7:8] + prev[:, 8:9]
        width_delta = width_cur - width_prev
        finger_vel_sum = (cur[:, 16:17] + cur[:, 17:18]) / 0.10

        red_goal_rel = (red - red_goal) / red_scale
        blue_goal_rel = (blue - blue_goal) / blue_scale
        tcp_red_rel = (tcp - red) / tcp_scale
        tcp_blue_rel = (tcp - blue) / tcp_scale
        red_blue_rel = (red - blue) / red_scale
        red_vel_rel = (red - red_prev) / red_scale
        blue_vel_rel = (blue - blue_prev) / blue_scale
        tcp_vel_rel = (tcp - tcp_prev) / tcp_scale

        red_goal_xy = self.vector_norm((red[:, :2] - red_goal[:, :2]) / red_scale[:2], True)
        blue_goal_xy = self.vector_norm((blue[:, :2] - blue_goal[:, :2]) / blue_scale[:2], True)
        tcp_red_dist = self.vector_norm((tcp - red) / tcp_scale, True)
        tcp_blue_dist = self.vector_norm((tcp - blue) / tcp_scale, True)
        red_z_above = (red[:, 2:3] - red_goal[:, 2:3]) / red_scale[2].clamp_min(1.0e-3)
        blue_z_above = (blue[:, 2:3] - blue_goal[:, 2:3]) / blue_scale[2].clamp_min(1.0e-3)

        closed_soft = torch.sigmoid((0.055 - width_cur) / 0.006)
        open_soft = torch.sigmoid((width_cur - 0.060) / 0.006)
        red_near_soft = torch.sigmoid((0.050 - self.vector_norm(red[:, :2] - red_goal[:, :2], True)) / 0.015)
        red_table_soft = torch.sigmoid((0.040 - (red[:, 2:3] - red_goal[:, 2:3])) / 0.015)
        tcp_high_soft = torch.sigmoid(((tcp[:, 2:3] - red[:, 2:3]) - 0.080) / 0.030)
        blue_lift_soft = torch.sigmoid(((blue[:, 2:3] - blue_goal[:, 2:3]) - 0.060) / 0.030)
        near_blue_soft = torch.sigmoid((0.060 - self.vector_norm(tcp[:, :2] - blue[:, :2], True)) / 0.020)

        engineered = torch.cat([
            width_cur, width_prev, width_delta, finger_vel_sum,
            red_goal_rel, blue_goal_rel, tcp_red_rel, tcp_blue_rel, red_blue_rel,
            red_vel_rel, blue_vel_rel, tcp_vel_rel,
            red_goal_xy, blue_goal_xy, tcp_red_dist, tcp_blue_dist,
            red_z_above, blue_z_above,
            closed_soft, open_soft, red_near_soft, red_table_soft, tcp_high_soft,
            blue_lift_soft, near_blue_soft,
        ], dim=-1)
        return torch.cat([norm_flat, norm_delta, engineered], dim=-1)

    def encode(self, raw_history):
        features = self.make_features(raw_history)
        base = self.feature_net(features)
        stage_logits = self.stage_head(base)
        stage_probs = torch.softmax(stage_logits, dim=-1)
        phase_mix = stage_probs.matmul(self.phase_embeddings.to(dtype=raw_history.dtype))
        cond = self.cond_net(torch.cat([base, stage_probs, phase_mix], dim=-1))
        return cond, base, stage_logits, stage_probs

    def forward(self, noisy_action, timestep, raw_history):
        cond, base, stage_logits, stage_probs = self.encode(raw_history)
        return self.backbone(noisy_action, timestep, cond)

    def auxiliary(self, raw_history, clean_action_encoded):
        cond, base, stage_logits, stage_probs = self.encode(raw_history)
        grip_logits = self.gripper_intent_head(torch.cat([base, stage_probs], dim=-1))
        action_flat = clean_action_encoded.reshape(clean_action_encoded.shape[0], -1)
        red_delta = self.red_future_head(torch.cat([cond, action_flat], dim=-1)).reshape(
            clean_action_encoded.shape[0], self.horizon, 3)
        return {
            "stage_logits": stage_logits,
            "stage_probs": stage_probs,
            "gripper_logits": grip_logits,
            "red_delta": red_delta,
        }


def build_model(spec):
    return ReleaseResetDiffusionPolicy(spec)


def stage_labels_from_history(raw_history):
    cur = raw_history[:, -1, :]
    prev = raw_history[:, 0, :]
    width = cur[:, 7] + cur[:, 8]
    red = cur[:, 25:28]
    red_prev = prev[:, 25:28]
    blue = cur[:, 32:35]
    tcp = cur[:, 18:21]
    red_goal = cur[:, 41:44]
    blue_goal = cur[:, 44:47]

    red_goal_xy = torch.sqrt(torch.sum((red[:, :2] - red_goal[:, :2]) ** 2, dim=-1) + 1.0e-8)
    red_z_above = red[:, 2] - red_goal[:, 2]
    red_dz = red[:, 2] - red_prev[:, 2]
    tcp_red_xy = torch.sqrt(torch.sum((tcp[:, :2] - red[:, :2]) ** 2, dim=-1) + 1.0e-8)
    tcp_red_z = tcp[:, 2] - red[:, 2]
    tcp_blue_xy = torch.sqrt(torch.sum((tcp[:, :2] - blue[:, :2]) ** 2, dim=-1) + 1.0e-8)
    blue_lift = (blue[:, 2] - blue_goal[:, 2]) > 0.060

    red_near = red_goal_xy < 0.050
    red_low = red_z_above < 0.050
    closed = width < 0.055
    opening_or_open = (width > 0.055) | ((cur[:, 16] + cur[:, 17]) > 0.010)
    retreat_geometry = (tcp_red_z > 0.080) | (tcp_red_xy > 0.060) | (tcp_blue_xy < 0.080) | blue_lift

    labels = torch.zeros(raw_history.shape[0], dtype=torch.long, device=raw_history.device)
    descend = red_near & closed & ((red_z_above < 0.140) | (red_dz < -0.010))
    open_release = red_near & red_low & (opening_or_open | (torch.abs(red_dz) < 0.004)) & (~retreat_geometry)
    retreat_switch = red_near & red_low & ((width > 0.060) | blue_lift) & retreat_geometry
    labels = torch.where(descend, torch.ones_like(labels), labels)
    labels = torch.where(open_release, torch.full_like(labels, 2), labels)
    labels = torch.where(retreat_switch, torch.full_like(labels, 3), labels)
    return labels


def compute_loss(model, batch, spec):
    pred_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = epsilon_loss(pred_noise, batch["noise"], batch["mask"])

    alpha_bar = batch["alpha_bar"].reshape(-1, 1, 1).to(dtype=pred_noise.dtype)
    sqrt_alpha = torch.sqrt(alpha_bar.clamp_min(1.0e-6))
    sqrt_one_minus = torch.sqrt((1.0 - alpha_bar).clamp_min(0.0))
    x0_est = (batch["noisy_action"] - sqrt_one_minus * pred_noise) / sqrt_alpha
    x0_aux = x0_est.clamp(-1.5, 1.5)

    aux = model.auxiliary(batch["raw_obs"], x0_aux)
    stage_labels = stage_labels_from_history(batch["raw_obs"])
    stage_loss = torch.nn.functional.cross_entropy(aux["stage_logits"], stage_labels)

    horizon_mask = batch["mask"].squeeze(-1).to(dtype=pred_noise.dtype)
    denom = horizon_mask.sum().clamp_min(1.0)
    target_open = (batch["native_action"][..., 7] > 0.0).to(dtype=pred_noise.dtype)
    grip_intent_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        aux["gripper_logits"], target_open, reduction="none")
    grip_intent_loss = (grip_intent_bce * horizon_mask).sum() / denom

    alpha_w = alpha_bar.reshape(-1, 1).detach().clamp(0.02, 1.0)
    grip_x0_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        3.0 * x0_aux[..., 7], target_open, reduction="none")
    grip_x0_loss = (grip_x0_bce * horizon_mask * alpha_w).sum() / (horizon_mask * alpha_w).sum().clamp_min(1.0)

    cur_red = batch["raw_obs"][:, -1, 25:28]
    future_red = batch["future_obs"][..., 25:28]
    red_goal = batch["raw_obs"][:, -1, 41:44]
    red_scale = model.obs_std[25:28].to(device=future_red.device, dtype=future_red.dtype).clamp_min(1.0e-3)
    red_target = (future_red - cur_red[:, None, :]) / red_scale.view(1, 1, 3)
    red_xy_near = torch.sqrt(torch.sum((future_red[..., :2] - red_goal[:, None, :2]) ** 2, dim=-1) + 1.0e-8) < 0.055
    red_on_table = future_red[..., 2] < (red_goal[:, None, 2] + 0.050)
    release_weight = 0.25 + 0.75 * (red_xy_near & red_on_table).to(dtype=pred_noise.dtype)
    red_weight = batch["future_mask"].squeeze(-1).to(dtype=pred_noise.dtype) * release_weight * alpha_w
    red_loss = (((aux["red_delta"] - red_target) ** 2).mean(dim=-1) * red_weight).sum() / red_weight.sum().clamp_min(1.0)

    prior_loss = 0.05 * stage_loss + 0.03 * grip_intent_loss + 0.03 * grip_x0_loss + 0.05 * red_loss
    loss = diffusion_loss + prior_loss
    return {
        "loss": loss,
        "diffusion_loss": diffusion_loss,
        "prior_loss": prior_loss,
    }
