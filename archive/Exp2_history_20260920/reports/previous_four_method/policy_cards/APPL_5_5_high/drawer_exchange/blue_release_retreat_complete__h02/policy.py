import math
import torch
from appl.public import DiffusionBackbone, epsilon_loss


class SupportBeforeRetreatPolicy(torch.nn.Module):
    """Learned diffusion policy with a trainable support/readiness latent."""

    def __init__(self, spec):
        super().__init__()
        training = spec["training"]
        normalizer = spec["normalizer"]
        self.horizon = int(training.get("horizon", 16))
        self.action_dim = 8
        self.feature_dim = 175
        self.core_dim = 224
        self.support_embed_dim = 32
        self.condition_dim = self.core_dim + self.support_embed_dim

        self.register_buffer("obs_mean", torch.tensor(normalizer["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.tensor(normalizer["std"], dtype=torch.float32))
        self.register_buffer("action_min", torch.tensor(normalizer["action_min"], dtype=torch.float32))
        self.register_buffer("action_scale", torch.tensor(normalizer["action_scale"], dtype=torch.float32))

        self.feature_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, 256),
            torch.nn.Mish(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, self.core_dim),
            torch.nn.Mish(),
            torch.nn.LayerNorm(self.core_dim),
        )
        self.support_head = torch.nn.Linear(self.core_dim, 4)
        self.support_embed = torch.nn.Sequential(
            torch.nn.Linear(8, 64),
            torch.nn.Mish(),
            torch.nn.Linear(64, self.support_embed_dim),
            torch.nn.Mish(),
        )
        self.backbone = DiffusionBackbone(self.condition_dim, training)

        aux_in = self.condition_dim + self.horizon * self.action_dim
        self.future_head = torch.nn.Sequential(
            torch.nn.Linear(aux_in, 256),
            torch.nn.Mish(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 256),
            torch.nn.Mish(),
            torch.nn.Linear(256, self.horizon * 6),
        )

    def normalize_obs_tensor(self, raw):
        return (raw - self.obs_mean.to(device=raw.device, dtype=raw.dtype)) / self.obs_std.to(device=raw.device, dtype=raw.dtype)

    def denormalize_action(self, encoded):
        amin = self.action_min.to(device=encoded.device, dtype=encoded.dtype)
        scale = self.action_scale.to(device=encoded.device, dtype=encoded.dtype)
        return (encoded + 1.0) * scale / 2.0 + amin

    def make_features(self, raw_history):
        if raw_history.dtype != self.obs_mean.dtype:
            raw_history = raw_history.to(dtype=self.obs_mean.dtype)
        norm = self.normalize_obs_tensor(raw_history)
        norm_flat = norm.reshape(norm.shape[0], -1)
        norm_delta = norm[:, -1, :] - norm[:, 0, :]

        last = raw_history[:, -1, :]
        prev = raw_history[:, 0, :]
        qpos = last[:, 0:9]
        tcp = last[:, 18:25]
        red = last[:, 25:32]
        blue = last[:, 32:39]
        drawer = last[:, 39:40]
        drawer_vel = last[:, 40:41]
        red_goal = last[:, 41:44]
        blue_goal = last[:, 44:47]

        prev_qpos = prev[:, 0:9]
        prev_tcp = prev[:, 18:25]
        prev_blue = prev[:, 32:39]

        tcp_xyz = tcp[:, 0:3]
        blue_xyz = blue[:, 0:3]
        red_xyz = red[:, 0:3]
        zero = torch.zeros_like(drawer)
        drawer_center = torch.cat([0.19 - drawer, zero, torch.full_like(drawer, 0.063)], dim=-1)

        rel_tcp_blue = (tcp_xyz - blue_xyz) / 0.15
        rel_blue_goal = (blue_xyz - blue_goal) / 0.15
        rel_blue_cavity = torch.cat([
            (blue_xyz[:, 0:1] - drawer_center[:, 0:1]) / 0.172,
            (blue_xyz[:, 1:2] - drawer_center[:, 1:2]) / 0.182,
            (blue_xyz[:, 2:3] - drawer_center[:, 2:3]) / 0.15,
        ], dim=-1)
        rel_tcp_goal = (tcp_xyz - blue_goal) / 0.20
        rel_red_goal = (red_xyz - red_goal) / 0.15

        finger_width = qpos[:, 7:8] + qpos[:, 8:9]
        finger_balance = (qpos[:, 7:8] - qpos[:, 8:9]) / 0.04
        prev_width = prev_qpos[:, 7:8] + prev_qpos[:, 8:9]
        finger_delta = (finger_width - prev_width) / 0.02
        drawer_open_margin = (drawer - 0.26) / 0.15
        drawer_velocity_scaled = drawer_vel / 0.05
        blue_z_lower = (blue_xyz[:, 2:3] - 0.053) / 0.05
        blue_z_upper = (0.074 - blue_xyz[:, 2:3]) / 0.05
        tcp_minus_blue_z = (tcp_xyz[:, 2:3] - blue_xyz[:, 2:3]) / 0.15
        tcp_z_scaled = tcp_xyz[:, 2:3] / 0.35
        blue_z_scaled = blue_xyz[:, 2:3] / 0.35
        tcp_delta_scaled = (tcp_xyz - prev_tcp[:, 0:3]) / 0.05
        blue_delta_scaled = (blue_xyz - prev_blue[:, 0:3]) / 0.05
        drawer_center_x_scaled = drawer_center[:, 0:1] / 0.30
        blue_xy_goal_dist = torch.linalg.norm((blue_xyz[:, 0:2] - blue_goal[:, 0:2]) / 0.15, dim=-1, keepdim=True)
        red_xy_goal_dist = torch.linalg.norm((red_xyz[:, 0:2] - red_goal[:, 0:2]) / 0.15, dim=-1, keepdim=True)

        extra = torch.cat([
            rel_tcp_blue,
            rel_blue_goal,
            rel_blue_cavity,
            rel_tcp_goal,
            rel_red_goal,
            finger_width / 0.08,
            finger_balance,
            finger_delta,
            drawer_open_margin,
            drawer_velocity_scaled,
            blue_z_lower,
            blue_z_upper,
            tcp_minus_blue_z,
            tcp_z_scaled,
            blue_z_scaled,
            tcp_delta_scaled,
            blue_delta_scaled,
            drawer_center_x_scaled,
            blue_xy_goal_dist,
            red_xy_goal_dist,
        ], dim=-1)
        return torch.cat([norm_flat, norm_delta, extra], dim=-1)

    def encode_history(self, raw_history):
        features = self.make_features(raw_history)
        core = self.feature_mlp(features)
        support_logits = self.support_head(core)
        support_prob = torch.sigmoid(support_logits)
        support_embedding = self.support_embed(torch.cat([support_logits, support_prob], dim=-1))
        condition = torch.cat([core, support_embedding], dim=-1)
        return condition, support_logits

    def forward(self, noisy_action, timestep, raw_history):
        condition, unused_logits = self.encode_history(raw_history)
        return self.backbone(noisy_action, timestep, condition)

    def auxiliary_predictions(self, raw_history, clean_action_encoded):
        condition, support_logits = self.encode_history(raw_history)
        action_flat = clean_action_encoded.reshape(clean_action_encoded.shape[0], -1)
        future = self.future_head(torch.cat([condition, action_flat], dim=-1))
        future = future.reshape(clean_action_encoded.shape[0], self.horizon, 6)
        return support_logits, future


def build_model(spec):
    return SupportBeforeRetreatPolicy(spec)


def masked_mean_square(error, mask, channels):
    return (error.square() * mask).sum() / (mask.sum() * channels).clamp_min(1.0)


def support_targets_from_future(raw_obs, future_obs, future_mask):
    last = raw_obs[:, -1, :]
    cur_blue = last[:, 32:35]
    cur_tcp_z = last[:, 20:21]
    blue_goal = last[:, 44:47]

    future_blue = future_obs[:, :, 32:35]
    future_tcp_z = future_obs[:, :, 20:21]
    future_width = future_obs[:, :, 7:8] + future_obs[:, :, 8:9]

    z_ok = (future_blue[:, :, 2:3] > 0.053) & (future_blue[:, :, 2:3] < 0.074)
    xy_goal_ok = (torch.abs(future_blue[:, :, 0:2] - blue_goal[:, None, 0:2]) < 0.045).all(dim=-1, keepdim=True)
    current_z_ok = (cur_blue[:, 2:3] > 0.053) & (cur_blue[:, 2:3] < 0.074)
    current_xy_ok = (torch.abs(cur_blue[:, 0:2] - blue_goal[:, 0:2]) < 0.045).all(dim=-1, keepdim=True)
    current_place = (current_z_ok & current_xy_ok).float()

    open_seq = (future_width > 0.070) & (future_mask > 0.5)
    support_seq_bool = z_ok & xy_goal_ok & (future_mask > 0.5)
    support_seq = support_seq_bool.float()
    open_target = (open_seq.float().sum(dim=1) > 0.0).float()
    support_future = (support_seq.sum(dim=1) > 0.0).float()
    support_or_current = torch.clamp(support_future + current_place, 0.0, 1.0)
    retreat_seq = support_seq_bool & open_seq & ((future_tcp_z > 0.10) | (future_tcp_z > (cur_tcp_z[:, None, :] + 0.03)))
    retreat_target = (retreat_seq.float().sum(dim=1) > 0.0).float()
    targets = torch.cat([support_or_current, open_target, retreat_target, current_place], dim=-1)
    return targets, support_seq


def compute_loss(model, batch, spec):
    predicted_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = epsilon_loss(predicted_noise, batch["noise"], batch["mask"])

    alpha_bar = batch["alpha_bar"].reshape(-1, 1, 1).to(device=predicted_noise.device, dtype=predicted_noise.dtype)
    noisy = batch["noisy_action"]
    clean_estimate = (noisy - torch.sqrt((1.0 - alpha_bar).clamp_min(0.0)) * predicted_noise) / torch.sqrt(alpha_bar.clamp_min(1.0e-6))
    clean_aux = clean_estimate.clamp(-1.25, 1.25)

    support_logits, future_pred = model.auxiliary_predictions(batch["raw_obs"], clean_aux)
    support_target, support_seq_target = support_targets_from_future(batch["raw_obs"], batch["future_obs"], batch["future_mask"])
    support_loss = torch.nn.functional.binary_cross_entropy_with_logits(support_logits, support_target)

    last = batch["raw_obs"][:, -1, :]
    cur_blue = last[:, 32:35]
    cur_tcp_z = last[:, 20:21]
    future_blue = batch["future_obs"][:, :, 32:35]
    future_tcp_z = batch["future_obs"][:, :, 20:21]
    future_width = batch["future_obs"][:, :, 7:8] + batch["future_obs"][:, :, 8:9]
    cont_target = torch.cat([
        (future_blue - cur_blue[:, None, :]) / 0.15,
        (future_tcp_z - cur_tcp_z[:, None, :]) / 0.15,
        future_width / 0.08,
    ], dim=-1)
    future_mask = batch["future_mask"]
    low_noise_weight = alpha_bar.detach().clamp(0.05, 1.0)
    cont_mask = future_mask * low_noise_weight
    future_cont_loss = masked_mean_square(future_pred[:, :, 0:5] - cont_target, cont_mask, 5)
    future_support_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        future_pred[:, :, 5:6], support_seq_target, reduction="none"
    )
    future_support_loss = (future_support_loss * future_mask).sum() / future_mask.sum().clamp_min(1.0)

    native_pred = model.denormalize_action(clean_aux)
    gripper_error = native_pred[:, :, 7:8] - batch["native_action"][:, :, 7:8]
    gripper_loss = (gripper_error.square() * batch["mask"] * low_noise_weight).sum() / batch["mask"].sum().clamp_min(1.0)

    prior_loss = 0.05 * support_loss + 0.02 * future_cont_loss + 0.02 * future_support_loss + 0.01 * gripper_loss
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
