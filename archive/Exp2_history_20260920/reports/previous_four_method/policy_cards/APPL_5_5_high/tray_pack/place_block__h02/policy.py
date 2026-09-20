import torch
from appl.public import DiffusionBackbone, epsilon_loss


class SupportGatedReleasePolicy(torch.nn.Module):
    """Learned diffusion policy with a trainable support/release phase posterior."""

    def __init__(self, spec):
        super().__init__()
        training = spec["training"]
        normalizer = spec.get("normalizer", spec.get("shared_normalizer", {}))
        self.horizon = int(training.get("horizon", 16))
        self.condition_dim = 256
        self.phase_count = 5

        self.register_buffer("obs_mean", torch.as_tensor(normalizer["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.as_tensor(normalizer["std"], dtype=torch.float32))
        self.register_buffer("act_min", torch.as_tensor(normalizer["action_min"], dtype=torch.float32))
        self.register_buffer("act_scale", torch.as_tensor(normalizer["action_scale"], dtype=torch.float32))

        feature_dim = 142  # two normalized observations plus causal geometric features
        hidden = 256
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden),
            torch.nn.Mish(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Mish(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Mish(),
        )
        self.phase_head = torch.nn.Linear(hidden, self.phase_count)
        self.support_head = torch.nn.Linear(hidden, 1)
        self.height_head = torch.nn.Linear(hidden, self.horizon)
        self.cond_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden + self.phase_count + 1 + self.horizon, self.condition_dim),
            torch.nn.Mish(),
            torch.nn.LayerNorm(self.condition_dim),
        )
        self.backbone = DiffusionBackbone(self.condition_dim, training)

    def normalize_obs(self, raw_history):
        return (raw_history - self.obs_mean.to(raw_history.device, raw_history.dtype)) / self.obs_std.to(raw_history.device, raw_history.dtype)

    @staticmethod
    def xy_norm(v):
        return torch.sqrt((v[..., 0] * v[..., 0]) + (v[..., 1] * v[..., 1]) + 1.0e-8)

    @staticmethod
    def vec_norm(v):
        return torch.sqrt((v * v).sum(dim=-1) + 1.0e-8)

    def active_blue_bool(self, obs):
        qpos = obs[:, 0:9]
        tcp = obs[:, 18:21]
        red = obs[:, 25:28]
        blue = obs[:, 32:35]
        red_goal = obs[:, 41:44]
        blue_goal = obs[:, 44:47]
        width = qpos[:, 7] + qpos[:, 8]
        red_dxy = self.xy_norm(red[:, 0:2] - red_goal[:, 0:2])
        blue_dxy = self.xy_norm(blue[:, 0:2] - blue_goal[:, 0:2])
        red_at_goal = (red_dxy < 0.08) & (torch.abs(red[:, 2] - red_goal[:, 2]) < 0.03)
        blue_elevated_or_goal = (blue[:, 2] > 0.055) | ((blue_dxy < 0.10) & red_at_goal)
        blue_grasped_closed = (self.vec_norm(tcp - blue) < 0.055) & (width < 0.055) & red_at_goal
        return blue_elevated_or_goal | blue_grasped_closed

    def engineered_features(self, raw_history):
        prev = raw_history[:, 0]
        cur = raw_history[:, 1]
        qpos = cur[:, 0:9]
        qvel = cur[:, 9:18]
        tcp = cur[:, 18:21]
        red = cur[:, 25:28]
        blue = cur[:, 32:35]
        red_goal = cur[:, 41:44]
        blue_goal = cur[:, 44:47]

        prev_tcp = prev[:, 18:21]
        prev_red = prev[:, 25:28]
        prev_blue = prev[:, 32:35]

        width = qpos[:, 7:8] + qpos[:, 8:9]
        finger_balance = qpos[:, 7:8] - qpos[:, 8:9]
        finger_vsum = qvel[:, 7:8] + qvel[:, 8:9]
        finger_vbalance = qvel[:, 7:8] - qvel[:, 8:9]

        tcp_delta = (tcp - prev_tcp) / 0.10
        red_delta = (red - prev_red) / 0.10
        blue_delta = (blue - prev_blue) / 0.10

        red_rel_goal = (red - red_goal) / 0.25
        blue_rel_goal = (blue - blue_goal) / 0.25
        tcp_rel_red = (tcp - red) / 0.25
        tcp_rel_blue = (tcp - blue) / 0.25
        red_dxy = self.xy_norm(red[:, 0:2] - red_goal[:, 0:2]).unsqueeze(-1) / 0.25
        blue_dxy = self.xy_norm(blue[:, 0:2] - blue_goal[:, 0:2]).unsqueeze(-1) / 0.25
        tcp_red_dist = self.vec_norm(tcp - red).unsqueeze(-1) / 0.25
        tcp_blue_dist = self.vec_norm(tcp - blue).unsqueeze(-1) / 0.25
        red_zerr = (red[:, 2:3] - red_goal[:, 2:3]) / 0.25
        blue_zerr = (blue[:, 2:3] - blue_goal[:, 2:3]) / 0.25
        tcp_red_z = (tcp[:, 2:3] - red[:, 2:3]) / 0.25
        tcp_blue_z = (tcp[:, 2:3] - blue[:, 2:3]) / 0.25

        active_blue = self.active_blue_bool(cur).to(cur.dtype).unsqueeze(-1)
        active_obj = torch.where(active_blue > 0.5, blue, red)
        active_goal = torch.where(active_blue > 0.5, blue_goal, red_goal)
        tcp_rel_active = (tcp - active_obj) / 0.25
        active_rel_goal = (active_obj - active_goal) / 0.25
        active_zerr = (active_obj[:, 2:3] - active_goal[:, 2:3]) / 0.25
        active_goal_dxy = self.xy_norm(active_obj[:, 0:2] - active_goal[:, 0:2]).unsqueeze(-1) / 0.25
        active_tcp_dist = self.vec_norm(tcp - active_obj).unsqueeze(-1) / 0.25

        red_goal_like = torch.sigmoid((0.06 - red_dxy * 0.25) * 40.0) * torch.sigmoid((0.025 - torch.abs(red[:, 2:3] - red_goal[:, 2:3])) * 80.0)
        blue_goal_like = torch.sigmoid((0.06 - blue_dxy * 0.25) * 40.0) * torch.sigmoid((0.025 - torch.abs(blue[:, 2:3] - blue_goal[:, 2:3])) * 80.0)

        return torch.cat([
            width / 0.08,
            finger_balance / 0.02,
            finger_vsum / 0.20,
            finger_vbalance / 0.20,
            tcp_delta,
            red_delta,
            blue_delta,
            red_rel_goal,
            tcp_rel_red,
            red_dxy,
            tcp_red_dist,
            red_zerr,
            tcp_red_z,
            blue_rel_goal,
            tcp_rel_blue,
            blue_dxy,
            tcp_blue_dist,
            blue_zerr,
            tcp_blue_z,
            (red - blue) / 0.35,
            active_blue,
            red_goal_like,
            blue_goal_like,
            tcp_rel_active,
            active_rel_goal,
            active_zerr,
            active_goal_dxy,
            active_tcp_dist,
        ], dim=-1)

    def encode_context(self, raw_history):
        norm_hist = self.normalize_obs(raw_history).reshape(raw_history.shape[0], -1)
        eng = self.engineered_features(raw_history)
        features = torch.cat([norm_hist, eng], dim=-1)
        hidden = self.encoder(features)
        phase_logits = self.phase_head(hidden)
        phase_prob = torch.softmax(phase_logits, dim=-1)
        support_logit = self.support_head(hidden)
        support_prob = torch.sigmoid(support_logit)
        height_pred = self.height_head(hidden)
        cond = self.cond_proj(torch.cat([hidden, phase_prob, support_prob, height_pred], dim=-1))
        aux = {
            "phase_logits": phase_logits,
            "support_logit": support_logit,
            "height_pred": height_pred,
            "phase_prob": phase_prob,
            "support_prob": support_prob,
        }
        return cond, aux

    def predict_with_aux(self, noisy_action, timestep, raw_history):
        cond, aux = self.encode_context(raw_history)
        eps = self.backbone(noisy_action, timestep, cond)
        return eps, aux

    def forward(self, noisy_action, timestep, raw_history):
        eps, unused_aux = self.predict_with_aux(noisy_action, timestep, raw_history)
        return eps

    def current_labels(self, raw_history):
        cur = raw_history[:, 1]
        qpos = cur[:, 0:9]
        qvel = cur[:, 9:18]
        tcp = cur[:, 18:21]
        red = cur[:, 25:28]
        blue = cur[:, 32:35]
        red_goal = cur[:, 41:44]
        blue_goal = cur[:, 44:47]
        active_blue = self.active_blue_bool(cur)
        active_obj = torch.where(active_blue.unsqueeze(-1), blue, red)
        active_goal = torch.where(active_blue.unsqueeze(-1), blue_goal, red_goal)

        width = qpos[:, 7] + qpos[:, 8]
        finger_v = qvel[:, 7] + qvel[:, 8]
        dxy = self.xy_norm(active_obj[:, 0:2] - active_goal[:, 0:2])
        zerr = active_obj[:, 2] - active_goal[:, 2]
        tcp_obj_dist = self.vec_norm(tcp - active_obj)
        supported = (torch.abs(zerr) < 0.018) & (dxy < 0.085)
        open_now = width > 0.065
        opening_now = (width > 0.050) | (finger_v > 0.025)
        retreat = supported & open_now & ((tcp[:, 2] - active_obj[:, 2] > 0.08) | (tcp_obj_dist > 0.10))
        open_release = supported & (open_now | opening_now) & (~retreat)
        settle = supported & (~open_release) & (~retreat)
        descend = (~supported) & ((dxy < 0.11) | (active_obj[:, 2] < 0.16))

        phase = torch.zeros(raw_history.shape[0], dtype=torch.long, device=raw_history.device)
        phase = torch.where(descend, torch.ones_like(phase), phase)
        phase = torch.where(settle, torch.full_like(phase, 2), phase)
        phase = torch.where(open_release, torch.full_like(phase, 3), phase)
        phase = torch.where(retreat, torch.full_like(phase, 4), phase)
        return phase, supported.to(raw_history.dtype), active_blue, active_goal

    def future_active_targets(self, raw_history, future_obs):
        cur = raw_history[:, 1]
        active_blue = self.active_blue_bool(cur)
        red_future = future_obs[:, :, 25:28]
        blue_future = future_obs[:, :, 32:35]
        red_goal = cur[:, 41:44]
        blue_goal = cur[:, 44:47]
        active_future = torch.where(active_blue[:, None, None], blue_future, red_future)
        active_goal = torch.where(active_blue[:, None], blue_goal, red_goal)
        rel_height = (active_future[:, :, 2] - active_goal[:, None, 2]) / 0.25
        dxy = self.xy_norm(active_future[:, :, 0:2] - active_goal[:, None, 0:2])
        support = (torch.abs(active_future[:, :, 2] - active_goal[:, None, 2]) < 0.018) & (dxy < 0.085)
        width = future_obs[:, :, 7] + future_obs[:, :, 8]
        open_future = width > 0.065
        return rel_height, support, open_future


def build_model(spec):
    return SupportGatedReleasePolicy(spec)


def safe_masked_mean(value, mask):
    denom = mask.sum().clamp_min(1.0)
    return (value * mask).sum() / denom


def compute_loss(model, batch, spec):
    pred_noise, aux = model.predict_with_aux(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = epsilon_loss(pred_noise, batch["noise"], batch["mask"])

    phase_target, support_target, unused_active, unused_goal = model.current_labels(batch["raw_obs"])
    phase_loss = torch.nn.functional.cross_entropy(aux["phase_logits"], phase_target)
    support_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        aux["support_logit"].squeeze(-1), support_target
    )

    rel_height_target, future_support, future_open = model.future_active_targets(batch["raw_obs"], batch["future_obs"])
    future_mask = batch["future_mask"].squeeze(-1)
    height_loss = safe_masked_mean((aux["height_pred"] - rel_height_target).square(), future_mask)

    alpha_bar = batch["alpha_bar"].reshape(-1, 1, 1).to(pred_noise.device, pred_noise.dtype)
    sqrt_ab = torch.sqrt(alpha_bar.clamp_min(1.0e-8))
    sqrt_omab = torch.sqrt((1.0 - alpha_bar).clamp_min(0.0))
    x0_pred = (batch["noisy_action"] - sqrt_omab * pred_noise) / sqrt_ab

    pred_grip = x0_pred[:, :, 7]
    gate_target = torch.where(future_open, torch.ones_like(pred_grip), -torch.ones_like(pred_grip))
    gate_mask = ((~future_support) | future_open).to(pred_grip.dtype) * future_mask
    snr_weight = alpha_bar.reshape(-1, 1).clamp(0.05, 1.0)
    gripper_gate_loss = safe_masked_mean((pred_grip - gate_target).square(), gate_mask * snr_weight)

    prior_loss = 0.10 * phase_loss + 0.05 * support_loss + 0.05 * height_loss + 0.05 * gripper_gate_loss
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
