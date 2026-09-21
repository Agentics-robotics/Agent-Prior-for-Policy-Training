import torch
from appl.public import DiffusionBackbone, epsilon_loss


class StageAwareHighArcPolicy(torch.nn.Module):
    """Learned diffusion policy with a causal high-arc/translate/insert stage prior."""

    def __init__(self, spec):
        super().__init__()
        normalizer = spec["normalizer"]
        obs_mean = torch.tensor(normalizer["mean"], dtype=torch.float32)
        obs_std = torch.tensor(normalizer["std"], dtype=torch.float32)
        action_min = torch.tensor(normalizer["action_min"], dtype=torch.float32)
        action_scale = torch.tensor(normalizer["action_scale"], dtype=torch.float32)
        self.register_buffer("obs_mean", obs_mean)
        self.register_buffer("obs_std", obs_std.clamp_min(1.0e-6))
        self.register_buffer("action_min", action_min)
        self.register_buffer("action_scale", action_scale.clamp_min(1.0e-6))
        self.register_buffer("pos_scale", obs_std[32:35].clamp_min(0.05))
        self.horizon = int(spec["training"]["horizon"])
        self.condition_dim = 256
        self.feature_dim = 94 + 41

        self.feature_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, 256),
            torch.nn.Mish(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 256),
            torch.nn.Mish(),
            torch.nn.LayerNorm(256),
        )
        self.current_stage_head = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, 96),
            torch.nn.Mish(),
            torch.nn.Linear(96, 3),
        )
        self.stage_to_condition = torch.nn.Sequential(
            torch.nn.Linear(3, 64),
            torch.nn.Mish(),
            torch.nn.Linear(64, self.condition_dim),
        )
        self.backbone = DiffusionBackbone(self.condition_dim, spec["training"])

        aux_in = self.condition_dim + self.horizon * 8
        aux_out = self.horizon * 8  # rel_xyz, stage3, inside, open
        self.aux_head = torch.nn.Sequential(
            torch.nn.Linear(aux_in, 384),
            torch.nn.Mish(),
            torch.nn.LayerNorm(384),
            torch.nn.Linear(384, 384),
            torch.nn.Mish(),
            torch.nn.Linear(384, aux_out),
        )

    def norm_obs(self, raw_history):
        return (raw_history - self.obs_mean.to(raw_history.device, raw_history.dtype)) / self.obs_std.to(raw_history.device, raw_history.dtype)

    def stage_targets_from_obs(self, obs):
        blue = obs[..., 32:35]
        goal = obs[..., 44:47]
        xy = blue[..., 0:2] - goal[..., 0:2]
        xy_err = torch.linalg.norm(xy, dim=-1)
        high = blue[..., 2] > 0.245
        aligned = xy_err < 0.060
        # 0: lift to clearance, 1: high lateral transport, 2: aligned descent/insertion/release.
        target = torch.zeros_like(xy_err, dtype=torch.long)
        target = torch.where(high & (~aligned), torch.ones_like(target), target)
        target = torch.where(aligned, torch.full_like(target, 2), target)
        return target

    def make_features(self, raw_history):
        dtype = raw_history.dtype
        device = raw_history.device
        norm_flat = self.norm_obs(raw_history).reshape(raw_history.shape[0], -1)
        prev = raw_history[:, 0]
        cur = raw_history[:, -1]

        tcp = cur[:, 18:21]
        prev_tcp = prev[:, 18:21]
        red = cur[:, 25:28]
        blue = cur[:, 32:35]
        prev_blue = prev[:, 32:35]
        goal = cur[:, 44:47]
        drawer = cur[:, 39:40]
        drawer_vel = cur[:, 40:41]
        fingers = cur[:, 7:9]
        qvel = cur[:, 9:18]

        pos_scale = self.pos_scale.to(device=device, dtype=dtype)
        rel_blue_goal = (blue - goal) / pos_scale
        rel_tcp_blue = (tcp - blue) / pos_scale
        rel_tcp_goal = (tcp - goal) / pos_scale
        blue_delta = (blue - prev_blue) / pos_scale
        tcp_delta = (tcp - prev_tcp) / pos_scale
        drawer_center = torch.cat([
            torch.full_like(drawer, 0.19) - drawer,
            torch.zeros_like(drawer),
            torch.full_like(drawer, 0.035),
        ], dim=-1)
        rel_blue_drawer = (blue - drawer_center) / pos_scale
        red_pad = torch.tensor([-0.18, -0.30, 0.02], device=device, dtype=dtype).view(1, 3)
        rel_red_pad = (red - red_pad) / pos_scale

        xy_err = torch.linalg.norm(blue[:, 0:2] - goal[:, 0:2], dim=-1, keepdim=True)
        high_soft = torch.sigmoid((blue[:, 2:3] - 0.245) / 0.025)
        aligned_soft = torch.sigmoid((0.060 - xy_err) / 0.020)
        descend_soft = aligned_soft
        translate_soft = high_soft * (1.0 - aligned_soft)
        lift_soft = (1.0 - high_soft) * (1.0 - aligned_soft)
        near_insert_z = torch.sigmoid((0.095 - torch.abs(blue[:, 2:3] - 0.063)) / 0.030)
        open_soft = torch.sigmoid(((fingers.mean(dim=-1, keepdim=True)) - 0.030) / 0.005)
        closed_soft = torch.sigmoid((0.025 - (fingers.mean(dim=-1, keepdim=True))) / 0.004)
        drawer_open_margin = (drawer - 0.260) / 0.150

        engineered = torch.cat([
            rel_blue_goal,
            rel_tcp_blue,
            rel_tcp_goal,
            rel_blue_drawer,
            rel_red_pad,
            blue_delta,
            tcp_delta,
            drawer_open_margin,
            drawer_vel / 0.15,
            fingers,
            qvel[:, 0:7] / 2.0,
            xy_err / 0.30,
            high_soft,
            aligned_soft,
            lift_soft,
            translate_soft,
            descend_soft,
            near_insert_z,
            open_soft,
            closed_soft,
        ], dim=-1)
        return torch.cat([norm_flat, engineered], dim=-1)

    def encode_condition(self, raw_history):
        features = self.make_features(raw_history)
        base = self.feature_encoder(features)
        stage_logits = self.current_stage_head(features)
        stage_probs = torch.softmax(stage_logits, dim=-1)
        return base + self.stage_to_condition(stage_probs)

    def forward(self, noisy_action, timestep, raw_history):
        condition = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, condition)

    def clean_action_estimate(self, batch, predicted_noise):
        alpha_bar = batch["alpha_bar"].to(device=predicted_noise.device, dtype=predicted_noise.dtype).view(-1, 1, 1)
        alpha_bar = alpha_bar.clamp(1.0e-4, 0.9999)
        noisy = batch["noisy_action"].to(dtype=predicted_noise.dtype)
        x0 = (noisy - torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
        return torch.clamp(x0, -1.5, 1.5)

    def aux_predictions(self, raw_history, clean_encoded_action):
        condition = self.encode_condition(raw_history)
        aux_in = torch.cat([condition, clean_encoded_action.reshape(clean_encoded_action.shape[0], -1)], dim=-1)
        return self.aux_head(aux_in).view(clean_encoded_action.shape[0], self.horizon, 8)

    def prior_loss(self, batch, predicted_noise):
        raw_history = batch["raw_obs"]
        future = batch["future_obs"]
        mask = batch["future_mask"].to(device=predicted_noise.device, dtype=predicted_noise.dtype)
        alpha_bar = batch["alpha_bar"].to(device=predicted_noise.device, dtype=predicted_noise.dtype).view(-1, 1, 1)
        confidence = alpha_bar.clamp(0.05, 1.0)
        weighted_mask = mask * confidence
        denom = weighted_mask.sum().clamp_min(1.0)

        x0 = self.clean_action_estimate(batch, predicted_noise)
        aux = self.aux_predictions(raw_history, x0)

        blue = future[..., 32:35]
        goal = future[..., 44:47]
        rel_target = (blue - goal) / self.pos_scale.to(device=blue.device, dtype=blue.dtype)
        rel_pred = aux[..., 0:3]
        rel_loss = ((rel_pred - rel_target).square() * weighted_mask).sum() / (denom * 3.0)

        stage_logits = aux[..., 3:6].reshape(-1, 3)
        stage_target = self.stage_targets_from_obs(future).reshape(-1)
        stage_ce = torch.nn.functional.cross_entropy(stage_logits, stage_target, reduction="none").view(future.shape[0], future.shape[1], 1)
        stage_loss = (stage_ce * weighted_mask).sum() / denom

        drawer = future[..., 39:40]
        drawer_center_x = 0.19 - drawer[..., 0]
        cavity_dx = torch.abs(blue[..., 0] - drawer_center_x)
        cavity_dy = torch.abs(blue[..., 1])
        z_ok = (blue[..., 2] > 0.053) & (blue[..., 2] < 0.074)
        xy_ok = (cavity_dx + 0.020 <= 0.172) & (cavity_dy + 0.020 <= 0.182)
        inside_target = (z_ok & xy_ok).to(dtype=predicted_noise.dtype).unsqueeze(-1)
        inside_loss = torch.nn.functional.binary_cross_entropy_with_logits(aux[..., 6:7], inside_target, reduction="none")
        inside_loss = (inside_loss * weighted_mask).sum() / denom

        future_finger = future[..., 7:9].mean(dim=-1, keepdim=True)
        open_target = (future_finger > 0.030).to(dtype=predicted_noise.dtype)
        open_loss = torch.nn.functional.binary_cross_entropy_with_logits(aux[..., 7:8], open_target, reduction="none")
        open_loss = (open_loss * weighted_mask).sum() / denom

        # Directly bias the denoised gripper command toward opening only when the
        # observed future has entered the release/insertion part of the overlap.
        desired_gripper = open_target * 2.0 - 1.0
        grip_action_loss = ((x0[..., 7:8] - desired_gripper).square() * weighted_mask).sum() / denom

        features = self.make_features(raw_history)
        current_logits = self.current_stage_head(features)
        current_target = self.stage_targets_from_obs(raw_history[:, -1])
        current_stage_loss = torch.nn.functional.cross_entropy(current_logits, current_target, reduction="mean")

        return (0.06 * rel_loss +
                0.025 * stage_loss +
                0.020 * inside_loss +
                0.015 * open_loss +
                0.010 * grip_action_loss +
                0.010 * current_stage_loss)


def build_model(spec):
    return StageAwareHighArcPolicy(spec)


def compute_loss(model, batch, spec):
    predicted_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = epsilon_loss(predicted_noise, batch["noise"], batch["mask"])
    prior_loss = model.prior_loss(batch, predicted_noise)
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
