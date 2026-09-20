import math
import torch
from appl.public import DiffusionBackbone, epsilon_loss


class BlueRelativeDiffusionPolicy(torch.nn.Module):
    """Learned DDPM epsilon model with blue-relative conditioning."""

    def __init__(self, spec):
        super().__init__()
        cfg = spec.get("candidate_config", {})
        self.cond_dim = int(cfg.get("condition_dim", 256))
        self.hidden_dim = int(cfg.get("encoder_hidden_dim", 256))
        self.aux_weight = float(cfg.get("aux_weight", 0.05))
        self.aux_state_weight = float(cfg.get("aux_state_weight", 1.0))
        self.aux_binary_weight = float(cfg.get("aux_binary_weight", 0.25))

        normalizer = spec["normalizer"]
        self.register_buffer("obs_mean", torch.tensor(normalizer["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.tensor(normalizer["std"], dtype=torch.float32))
        self.register_buffer("rel_xyz_scale", torch.tensor([0.25, 0.25, 0.35], dtype=torch.float32))
        self.register_buffer("goal_xyz_scale", torch.tensor([0.50, 0.50, 0.30], dtype=torch.float32))
        self.register_buffer("tcp_blue_aux_scale", torch.tensor([0.15, 0.15, 0.30], dtype=torch.float32))

        feature_dim = 94 + 34 + 9 + 8
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Linear(self.hidden_dim, self.cond_dim),
            torch.nn.SiLU(),
        )
        self.backbone = DiffusionBackbone(self.cond_dim, spec["training"])
        self.aux_head = torch.nn.Sequential(
            torch.nn.Linear(self.cond_dim + 8 + 1, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_dim, 7),
        )

    def normalize_obs(self, raw_history):
        mean = self.obs_mean.to(device=raw_history.device, dtype=raw_history.dtype)
        std = self.obs_std.to(device=raw_history.device, dtype=raw_history.dtype).clamp_min(1.0e-6)
        return (raw_history - mean) / std

    def make_features(self, raw_history):
        raw = raw_history
        norm = self.normalize_obs(raw).reshape(raw.shape[0], -1)
        rel_scale = self.rel_xyz_scale.to(device=raw.device, dtype=raw.dtype)
        goal_scale = self.goal_xyz_scale.to(device=raw.device, dtype=raw.dtype)

        per_step = []
        for k in range(raw.shape[1]):
            obs = raw[:, k]
            tcp = obs[:, 18:21]
            red = obs[:, 25:28]
            blue = obs[:, 32:35]
            drawer = obs[:, 39:40]
            red_goal = obs[:, 41:44]
            blue_goal = obs[:, 44:47]
            qpos = obs[:, 0:9]
            qvel = obs[:, 9:18]
            finger_sum = qpos[:, 7:8] + qpos[:, 8:9]
            finger_speed_sum = qvel[:, 7:8] + qvel[:, 8:9]
            red_goal_xy = (red[:, 0:2] - red_goal[:, 0:2]) / 0.12
            drawer_margin = (drawer - 0.26) / 0.15
            feat = torch.cat(
                [
                    (tcp - blue) / rel_scale,
                    (tcp - red) / rel_scale,
                    (blue - blue_goal) / goal_scale,
                    (red - red_goal) / goal_scale,
                    finger_sum / 0.08,
                    finger_speed_sum / 0.20,
                    red_goal_xy,
                    drawer_margin,
                ],
                dim=-1,
            )
            per_step.append(feat)
        per_step = torch.cat(per_step, dim=-1)

        prev = raw[:, 0]
        cur = raw[:, -1]
        tcp_delta = (cur[:, 18:21] - prev[:, 18:21]) / rel_scale
        blue_delta = (cur[:, 32:35] - prev[:, 32:35]) / rel_scale
        qpos_delta = cur[:, 0:3] - prev[:, 0:3]
        deltas = torch.cat([tcp_delta, blue_delta, qpos_delta], dim=-1)

        tcp = cur[:, 18:21]
        blue = cur[:, 32:35]
        red = cur[:, 25:28]
        qpos = cur[:, 0:9]
        qvel = cur[:, 9:18]
        finger_sum = qpos[:, 7:8] + qpos[:, 8:9]
        tcp_blue = tcp - blue
        xy_dist = torch.sqrt((tcp_blue[:, 0:1] ** 2 + tcp_blue[:, 1:2] ** 2).clamp_min(1.0e-8))
        blue_height = (blue[:, 2:3] - 0.02) / 0.30
        red_pad_dist = torch.sqrt(((red[:, 0:2] - cur[:, 41:43]) ** 2).sum(dim=-1, keepdim=True).clamp_min(1.0e-8))
        phase = torch.cat(
            [
                xy_dist / 0.25,
                tcp_blue[:, 2:3] / 0.35,
                finger_sum / 0.08,
                (qvel[:, 7:8] + qvel[:, 8:9]) / 0.20,
                blue_height,
                red_pad_dist / 0.12,
                (cur[:, 39:40] - 0.26) / 0.15,
                torch.ones_like(finger_sum),
            ],
            dim=-1,
        )
        return torch.cat([norm, per_step, deltas, phase], dim=-1)

    def encode_condition(self, raw_history):
        return self.encoder(self.make_features(raw_history))

    def forward(self, noisy_action, timestep, raw_history):
        cond = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, cond)

    def aux_predict(self, clean_action_estimate, raw_history):
        cond = self.encode_condition(raw_history)
        b, h, value_dim = clean_action_estimate.shape
        if h == 1:
            step = torch.zeros((b, h, 1), device=clean_action_estimate.device, dtype=clean_action_estimate.dtype)
        else:
            step_values = torch.linspace(-1.0, 1.0, h, device=clean_action_estimate.device, dtype=clean_action_estimate.dtype)
            step = step_values.view(1, h, 1).expand(b, h, 1)
        cond_seq = cond.unsqueeze(1).expand(b, h, cond.shape[-1])
        return self.aux_head(torch.cat([clean_action_estimate, step, cond_seq], dim=-1))


def build_model(spec):
    return BlueRelativeDiffusionPolicy(spec)


def compute_loss(model, batch, spec):
    pred_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    mask = batch["mask"]
    diffusion_loss = epsilon_loss(pred_noise, batch["noise"], mask)

    alpha_bar = batch["alpha_bar"].to(device=pred_noise.device, dtype=pred_noise.dtype).view(-1, 1, 1)
    alpha_bar = alpha_bar.clamp_min(1.0e-5)
    clean_est = (batch["noisy_action"] - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
    clean_est = clean_est.clamp(-1.5, 1.5)

    aux_pred = model.aux_predict(clean_est, batch["raw_obs"])
    future = batch["future_obs"]
    current = batch["raw_obs"][:, -1]
    future_mask = batch["future_mask"] * mask
    weight = future_mask * alpha_bar.clamp(0.05, 1.0)

    scale = model.tcp_blue_aux_scale.to(device=future.device, dtype=future.dtype)
    blue_z_rel = ((future[:, :, 34:35] - current[:, None, 34:35]) / 0.30)
    tcp_blue = (future[:, :, 18:21] - future[:, :, 32:35]) / scale
    finger_sum = (future[:, :, 7:8] + future[:, :, 8:9]) / 0.08
    state_target = torch.cat([blue_z_rel, tcp_blue, finger_sum], dim=-1)
    state_loss = ((aux_pred[:, :, 0:5] - state_target).square() * weight).sum() / (weight.sum() * 5.0).clamp_min(1.0)

    tcp_blue_native = future[:, :, 18:21] - future[:, :, 32:35]
    xy_dist = torch.sqrt((tcp_blue_native[:, :, 0:1] ** 2 + tcp_blue_native[:, :, 1:2] ** 2).clamp_min(1.0e-8))
    finger_sum_native = future[:, :, 7:8] + future[:, :, 8:9]
    contact_target = ((xy_dist < 0.035) & (torch.abs(tcp_blue_native[:, :, 2:3]) < 0.040) & (finger_sum_native < 0.060)).to(aux_pred.dtype)
    lift_target = ((future[:, :, 34:35] - current[:, None, 34:35]) > 0.040).to(aux_pred.dtype)
    logits = aux_pred[:, :, 5:7]
    binary_target = torch.cat([contact_target, lift_target], dim=-1)
    binary_raw = torch.nn.functional.binary_cross_entropy_with_logits(logits, binary_target, reduction="none")
    binary_loss = (binary_raw * weight).sum() / (weight.sum() * 2.0).clamp_min(1.0)

    prior_loss = model.aux_weight * (model.aux_state_weight * state_loss + model.aux_binary_weight * binary_loss)
    total_loss = diffusion_loss + prior_loss
    return {"loss": total_loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
