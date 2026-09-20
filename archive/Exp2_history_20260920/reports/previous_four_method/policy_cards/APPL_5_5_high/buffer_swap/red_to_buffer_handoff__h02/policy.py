import torch
import math
from appl.public import DiffusionBackbone, epsilon_loss


class BufferTopologyDiffusionPolicy(torch.nn.Module):
    """Learned diffusion policy with an explicit red-to-buffer topology prior.

    The model remains an epsilon-predicting action diffusion model. The prior is
    implemented as learned conditioning: a causal observation encoder predicts a
    temporary red placement subgoal near the demonstrated central buffer and a
    red-buffered predicate. Those trainable predictions are fed to the DDPM
    backbone and are also supervised by differentiable auxiliary losses in
    compute_loss.
    """

    def __init__(self, spec):
        super().__init__()
        n = spec["normalizer"]
        self.register_buffer("obs_mean", torch.tensor(n["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.tensor(n["std"], dtype=torch.float32))
        self.register_buffer("rel_scale", torch.tensor([0.25, 0.25, 0.15], dtype=torch.float32))
        self.register_buffer("subgoal_residual_scale", torch.tensor([0.08, 0.08, 0.035], dtype=torch.float32))

        self.normalized_history_dim = 2 * 47
        self.prior_feature_dim = 39
        encoder_in = self.normalized_history_dim + self.prior_feature_dim
        latent_dim = 160
        condition_dim = 192

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(encoder_in, 256),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, latent_dim),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(latent_dim),
        )
        self.subgoal_head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 3),
        )
        self.buffered_head = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 1),
        )
        cond_in = latent_dim + 3 + 1 + self.prior_feature_dim
        self.condition_projector = torch.nn.Sequential(
            torch.nn.Linear(cond_in, 256),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, condition_dim),
        )
        self.backbone = DiffusionBackbone(condition_dim, spec["training"])

    def normalize_obs(self, raw_history):
        mean = self.obs_mean.to(device=raw_history.device, dtype=raw_history.dtype)
        std = self.obs_std.to(device=raw_history.device, dtype=raw_history.dtype)
        return (raw_history - mean) / std

    def buffer_center(self, latest):
        red_goal = latest[..., 41:44]
        blue_goal = latest[..., 44:47]
        x = torch.full_like(red_goal[..., 0:1], -0.181)
        y = 0.5 * (red_goal[..., 1:2] + blue_goal[..., 1:2])
        z = 0.5 * (red_goal[..., 2:3] + blue_goal[..., 2:3])
        return torch.cat([x, y, z], dim=-1)

    def soft_region_score(self, xyz, center, xy_radius=0.065, z_radius=0.035):
        xy_dist = torch.linalg.norm(xyz[..., :2] - center[..., :2], dim=-1, keepdim=True)
        z_dist = torch.abs(xyz[..., 2:3] - center[..., 2:3])
        xy_score = torch.sigmoid((xy_radius - xy_dist) / 0.020)
        z_score = torch.sigmoid((z_radius - z_dist) / 0.012)
        return xy_score * z_score

    def hard_buffer_label(self, latest):
        red = latest[..., 25:28]
        buffer = self.buffer_center(latest)
        xy_dist = torch.linalg.norm(red[..., :2] - buffer[..., :2], dim=-1, keepdim=True)
        z_dist = torch.abs(red[..., 2:3] - buffer[..., 2:3])
        return ((xy_dist < 0.075) & (z_dist < 0.035)).to(dtype=latest.dtype)

    def prior_features(self, raw_history):
        latest = raw_history[:, -1]
        qpos = latest[:, 0:9]
        tcp = latest[:, 18:21]
        red = latest[:, 25:28]
        blue = latest[:, 32:35]
        red_goal = latest[:, 41:44]
        blue_goal = latest[:, 44:47]
        buffer = self.buffer_center(latest)
        rel_scale = self.rel_scale.to(device=raw_history.device, dtype=raw_history.dtype)

        rel_vecs = [
            (tcp - red) / rel_scale,
            (tcp - blue) / rel_scale,
            (tcp - buffer) / rel_scale,
            (red - buffer) / rel_scale,
            (red - blue_goal) / rel_scale,
            (blue - red_goal) / rel_scale,
            (red - blue) / rel_scale,
        ]

        def xy_distance(a, b):
            return torch.linalg.norm(a[:, :2] - b[:, :2], dim=-1, keepdim=True) / 0.30

        distances = [
            xy_distance(tcp, red),
            xy_distance(tcp, blue),
            xy_distance(tcp, buffer),
            xy_distance(red, buffer),
            xy_distance(red, blue_goal),
            xy_distance(blue, red_goal),
            xy_distance(red, blue),
        ]

        red_in_initial_blue_goal = self.soft_region_score(red, blue_goal)
        blue_in_red_goal = self.soft_region_score(blue, red_goal)
        red_buffered = self.soft_region_score(red, buffer, xy_radius=0.075, z_radius=0.035)
        symbolic = [red_in_initial_blue_goal, blue_in_red_goal, red_buffered]

        heights = [
            (red[:, 2:3] - buffer[:, 2:3]) / 0.15,
            (blue[:, 2:3] - buffer[:, 2:3]) / 0.15,
            (tcp[:, 2:3] - buffer[:, 2:3]) / 0.35,
        ]
        finger_width = qpos[:, 7:8] + qpos[:, 8:9]
        finger_features = [
            (finger_width - 0.04) / 0.04,
            torch.sigmoid((finger_width - 0.06) / 0.010),
        ]
        goal_feature = [(red_goal - blue_goal) / rel_scale]
        return torch.cat(rel_vecs + distances + symbolic + heights + finger_features + goal_feature, dim=-1)

    def condition(self, raw_history):
        norm_hist = self.normalize_obs(raw_history).reshape(raw_history.shape[0], -1)
        prior_features = self.prior_features(raw_history)
        latent = self.encoder(torch.cat([norm_hist, prior_features], dim=-1))
        latest = raw_history[:, -1]
        buffer = self.buffer_center(latest)
        residual = torch.tanh(self.subgoal_head(latent)) * self.subgoal_residual_scale.to(
            device=raw_history.device, dtype=raw_history.dtype
        )
        subgoal = buffer + residual
        buffered_logit = self.buffered_head(latent)
        pos_mean = self.obs_mean[25:28].to(device=raw_history.device, dtype=raw_history.dtype)
        pos_std = self.obs_std[25:28].to(device=raw_history.device, dtype=raw_history.dtype)
        subgoal_norm = (subgoal - pos_mean) / pos_std
        buffered_prob = torch.sigmoid(buffered_logit)
        cond = self.condition_projector(torch.cat([latent, subgoal_norm, buffered_prob, prior_features], dim=-1))
        return cond, {"subgoal": subgoal, "buffered_logit": buffered_logit, "prior_features": prior_features}

    def forward(self, noisy_action, timestep, raw_history):
        cond, _ = self.condition(raw_history)
        return self.backbone(noisy_action, timestep, cond)

    def auxiliary_loss(self, batch):
        raw_history = batch["raw_obs"]
        _, aux = self.condition(raw_history)
        latest = raw_history[:, -1]
        current_red = latest[:, 25:28]
        buffer = self.buffer_center(latest)
        current_buffered = self.hard_buffer_label(latest)
        target = buffer

        if "future_obs" in batch and batch["future_obs"] is not None:
            future = batch["future_obs"]
            future_red = future[..., 25:28]
            fbuffer = buffer.unsqueeze(1)
            xy = torch.linalg.norm(future_red[..., :2] - fbuffer[..., :2], dim=-1)
            z = torch.abs(future_red[..., 2] - fbuffer[..., 2])
            candidate = (xy < 0.075) & (z < 0.035)
            if "future_mask" in batch and batch["future_mask"] is not None:
                candidate = candidate & (batch["future_mask"][..., 0] > 0.5)
            horizon = future_red.shape[1]
            idx_values = torch.arange(horizon, device=future_red.device).view(1, horizon).expand(future_red.shape[0], horizon)
            first_idx = torch.where(candidate, idx_values, torch.full_like(idx_values, horizon)).min(dim=1).values
            has_candidate = first_idx < horizon
            safe_idx = first_idx.clamp(max=horizon - 1)
            batch_idx = torch.arange(future_red.shape[0], device=future_red.device)
            future_target = future_red[batch_idx, safe_idx]
            target = torch.where(has_candidate.unsqueeze(-1), future_target, target)

        target = torch.where(current_buffered.bool(), current_red, target)
        pos_std = self.obs_std[25:28].to(device=raw_history.device, dtype=raw_history.dtype)
        subgoal_loss = (((aux["subgoal"] - target) / pos_std) ** 2).mean()
        buffer_label = current_buffered
        bce = torch.nn.functional.binary_cross_entropy_with_logits(aux["buffered_logit"], buffer_label)
        return subgoal_loss, bce


def build_model(spec):
    return BufferTopologyDiffusionPolicy(spec)


def compute_loss(model, batch, spec):
    predicted_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = epsilon_loss(predicted_noise, batch["noise"], batch["mask"])
    subgoal_loss, buffered_loss = model.auxiliary_loss(batch)
    prior_loss = 0.20 * subgoal_loss + 0.05 * buffered_loss
    total = diffusion_loss + prior_loss
    return {
        "loss": total,
        "diffusion_loss": diffusion_loss,
        "prior_loss": prior_loss,
    }
