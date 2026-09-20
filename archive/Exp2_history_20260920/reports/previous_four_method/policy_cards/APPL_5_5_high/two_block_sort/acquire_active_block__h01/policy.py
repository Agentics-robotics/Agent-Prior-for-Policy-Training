import torch
import appl.public as public


class ActiveObjectRelativeDiffusion(torch.nn.Module):
    """Diffusion denoiser with an active-object-relative state encoder."""

    def __init__(self, spec):
        super().__init__()
        cfg = spec.get("candidate_config", {})
        self.obs_steps = int(spec["training"].get("observation_steps", 2))
        self.cond_dim = int(cfg.get("condition_dim", 192))
        self.done_xy = float(cfg.get("red_done_xy_m", 0.075))
        self.done_z = float(cfg.get("red_done_z_m", 0.055))
        self.geom_dim = 8

        n = spec["normalizer"]
        mean = torch.tensor(n["mean"], dtype=torch.float32)
        std = torch.tensor(n["std"], dtype=torch.float32).clamp_min(1.0e-6)
        self.register_buffer("obs_mean", mean)
        self.register_buffer("obs_std", std)
        rel_scale = torch.maximum(std[18:21], torch.maximum(std[25:28], std[32:35])).clamp_min(0.05)
        self.register_buffer("rel_scale", rel_scale)
        self.register_buffer("wide_xyz_scale", torch.tensor([0.50, 0.50, 0.30], dtype=torch.float32))
        self.register_buffer("geom_weights", torch.tensor([1.0, 1.0, 1.0, 0.5, 0.8, 0.4, 0.4, 0.4], dtype=torch.float32))

        self.feature_dim = 59
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.obs_steps * self.feature_dim, 256),
            torch.nn.Mish(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, self.cond_dim),
            torch.nn.Mish(),
            torch.nn.LayerNorm(self.cond_dim),
        )
        self.backbone = public.DiffusionBackbone(self.cond_dim, spec["training"])
        self.geom_head = torch.nn.Sequential(
            torch.nn.Linear(self.cond_dim + 8 + 1, 160),
            torch.nn.Mish(),
            torch.nn.Linear(160, 160),
            torch.nn.Mish(),
            torch.nn.Linear(160, self.geom_dim),
        )

    def active_blue_mask(self, raw_history):
        last = raw_history[:, -1]
        red_xyz = last[:, 25:28]
        red_goal = last[:, 41:44]
        xy_dist = torch.linalg.norm(red_xyz[:, 0:2] - red_goal[:, 0:2], dim=-1)
        z_dist = torch.abs(red_xyz[:, 2] - red_goal[:, 2])
        red_done = (xy_dist < self.done_xy) & (z_dist < self.done_z)
        return red_done.to(dtype=raw_history.dtype).view(-1, 1, 1)

    def select_active(self, raw, active_blue):
        red_pose = raw[..., 25:32]
        blue_pose = raw[..., 32:39]
        red_goal = raw[..., 41:44]
        blue_goal = raw[..., 44:47]
        active_pose = torch.where(active_blue > 0.5, blue_pose, red_pose)
        inactive_pose = torch.where(active_blue > 0.5, red_pose, blue_pose)
        active_goal = torch.where(active_blue > 0.5, blue_goal, red_goal)
        inactive_goal = torch.where(active_blue > 0.5, red_goal, blue_goal)
        return active_pose, inactive_pose, active_goal, inactive_goal

    def normalize_full_obs(self, raw):
        return (raw - self.obs_mean.to(device=raw.device, dtype=raw.dtype)) / self.obs_std.to(device=raw.device, dtype=raw.dtype)

    def object_relative_features(self, raw_history):
        raw = raw_history
        dtype = raw.dtype
        obs_norm = self.normalize_full_obs(raw)
        active_blue = self.active_blue_mask(raw_history)
        active_pose, inactive_pose, active_goal, inactive_goal = self.select_active(raw, active_blue)

        tcp_xyz = raw[..., 18:21]
        tcp_quat = raw[..., 21:25]
        qpos = raw[..., 0:9]
        active_xyz = active_pose[..., 0:3]
        inactive_xyz = inactive_pose[..., 0:3]

        rel_scale = self.rel_scale.to(device=raw.device, dtype=dtype)
        wide_scale = self.wide_xyz_scale.to(device=raw.device, dtype=dtype)
        red_abs = (raw[..., 25:28] - self.obs_mean[25:28].to(raw.device, dtype)) / self.obs_std[25:28].to(raw.device, dtype)
        blue_abs = (raw[..., 32:35] - self.obs_mean[32:35].to(raw.device, dtype)) / self.obs_std[32:35].to(raw.device, dtype)
        active_abs = torch.where(active_blue > 0.5, blue_abs, red_abs)
        inactive_abs = torch.where(active_blue > 0.5, red_abs, blue_abs)

        active_lift = (active_xyz[..., 2:3] - active_goal[..., 2:3]) / rel_scale[2]
        gripper_sum = (qpos[..., 7:8] + qpos[..., 8:9] - 0.058) / 0.025
        goal_abs = active_goal / wide_scale
        inactive_goal_rel = (inactive_xyz - inactive_goal) / rel_scale

        pieces = [
            obs_norm[..., 0:9],
            obs_norm[..., 9:18],
            obs_norm[..., 18:21],
            tcp_quat,
            (tcp_xyz - active_xyz) / rel_scale,
            (tcp_xyz - active_goal) / rel_scale,
            (active_xyz - active_goal) / rel_scale,
            (inactive_xyz - active_xyz) / rel_scale,
            active_abs,
            inactive_abs,
            active_pose[..., 3:7],
            inactive_pose[..., 3:7],
            active_lift,
            gripper_sum,
            goal_abs,
            inactive_goal_rel,
        ]
        return torch.cat(pieces, dim=-1)

    def encode_condition(self, raw_history):
        feat = self.object_relative_features(raw_history)
        if feat.shape[1] < self.obs_steps:
            pad = feat[:, :1].expand(-1, self.obs_steps - feat.shape[1], -1)
            feat = torch.cat([pad, feat], dim=1)
        elif feat.shape[1] > self.obs_steps:
            feat = feat[:, -self.obs_steps:]
        flat = feat.reshape(feat.shape[0], -1)
        return self.encoder(flat)

    def forward(self, noisy_action, timestep, raw_history):
        cond = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, cond)

    def predict_geometry(self, encoded_action_x0, raw_history):
        cond = self.encode_condition(raw_history)
        b, h, _ = encoded_action_x0.shape
        if h > 1:
            tau = torch.linspace(0.0, 1.0, h, device=encoded_action_x0.device, dtype=encoded_action_x0.dtype)
        else:
            tau = torch.zeros(h, device=encoded_action_x0.device, dtype=encoded_action_x0.dtype)
        tau = tau.view(1, h, 1).expand(b, h, 1)
        cond_h = cond.unsqueeze(1).expand(b, h, cond.shape[-1])
        x = torch.cat([encoded_action_x0, tau, cond_h], dim=-1)
        return self.geom_head(x)

    def future_geometry_target(self, raw_history, future_obs):
        dtype = future_obs.dtype
        active_blue = self.active_blue_mask(raw_history)
        active_pose, inactive_pose, active_goal, inactive_goal = self.select_active(future_obs, active_blue)
        tcp_xyz = future_obs[..., 18:21]
        active_xyz = active_pose[..., 0:3]
        qpos = future_obs[..., 0:9]
        rel_scale = self.rel_scale.to(device=future_obs.device, dtype=dtype)
        gripper_sum = (qpos[..., 7:8] + qpos[..., 8:9] - 0.058) / 0.025
        active_lift = (active_xyz[..., 2:3] - active_goal[..., 2:3]) / rel_scale[2]
        return torch.cat([
            (tcp_xyz - active_xyz) / rel_scale,
            gripper_sum,
            active_lift,
            (active_xyz - active_goal) / rel_scale,
        ], dim=-1)


def build_model(spec):
    return ActiveObjectRelativeDiffusion(spec)


def compute_loss(model, batch, spec):
    pred_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    mask = batch["mask"]
    diffusion_loss = public.epsilon_loss(pred_noise, batch["noise"], mask)

    alpha_bar = batch["alpha_bar"].to(device=pred_noise.device, dtype=pred_noise.dtype).view(-1, 1, 1)
    sqrt_ab = torch.sqrt(alpha_bar.clamp_min(1.0e-6))
    sqrt_om = torch.sqrt((1.0 - alpha_bar).clamp_min(0.0))
    x0 = (batch["noisy_action"] - sqrt_om * pred_noise) / sqrt_ab
    x0_safe = x0.clamp(-2.0, 2.0)
    geom_pred = model.predict_geometry(x0_safe, batch["raw_obs"])
    geom_tgt = model.future_geometry_target(batch["raw_obs"], batch["future_obs"])
    if "future_mask" in batch:
        valid_mask = batch["future_mask"] * mask
    else:
        valid_mask = mask
    geom_mask = valid_mask * alpha_bar.clamp(0.05, 1.0)
    weights = model.geom_weights.to(device=pred_noise.device, dtype=pred_noise.dtype).view(1, 1, -1)
    geom_err = (geom_pred - geom_tgt).square() * weights
    prior_loss = (geom_err * geom_mask).sum() / (geom_mask.sum() * geom_pred.shape[-1]).clamp_min(1.0)

    cfg = spec.get("candidate_config", {})
    prior_w = float(cfg.get("prior_loss_weight", 0.02))
    loss = diffusion_loss + prior_w * prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
