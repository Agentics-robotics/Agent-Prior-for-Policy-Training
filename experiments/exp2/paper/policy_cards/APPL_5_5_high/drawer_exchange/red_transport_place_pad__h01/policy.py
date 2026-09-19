import math

import torch
from appl.public import DiffusionBackbone, epsilon_loss


class PadFrameConditionEncoder(torch.nn.Module):
    """Causal state encoder for red-pad placement.

    It keeps the shared observation normalization for the original 47-D state and
    augments it with world-frame relative quantities that express the assigned
    heuristic: red-to-pad error, TCP-to-red offset, and transition cues toward
    blue after release.  All quantities are computed from the two causal
    observations only.
    """

    def __init__(self, spec, condition_dim):
        super().__init__()
        normalizer = spec["normalizer"]
        self.register_buffer("obs_mean", torch.tensor(normalizer["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.tensor(normalizer["std"], dtype=torch.float32))
        input_dim = 160
        hidden = 256
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.Mish(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Mish(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, condition_dim),
            torch.nn.Mish(),
        )

    def per_step_features(self, obs):
        tcp = obs[:, 18:21]
        red = obs[:, 25:28]
        blue = obs[:, 32:35]
        drawer = obs[:, 39:40]
        drawer_vel = obs[:, 40:41]
        red_goal = obs[:, 41:44]
        blue_goal = obs[:, 44:47]
        finger_width = obs[:, 7:8] + obs[:, 8:9]
        red_to_goal = red_goal - red
        tcp_to_red = tcp - red
        tcp_to_goal = red_goal - tcp
        tcp_to_blue = blue - tcp
        blue_to_goal = blue_goal - blue
        height_err = red[:, 2:3] - red_goal[:, 2:3]
        abs_xy_err = torch.abs(red_to_goal[:, 0:2])
        return torch.cat(
            [
                red_to_goal / 0.15,
                tcp_to_red / 0.15,
                tcp_to_goal / 0.15,
                tcp_to_blue / 0.30,
                blue_to_goal / 0.15,
                (drawer - 0.26) / 0.15,
                drawer_vel / 0.15,
                finger_width / 0.08,
                height_err / 0.15,
                abs_xy_err / 0.06,
            ],
            dim=-1,
        )

    def forward(self, raw_history):
        obs_norm = (raw_history - self.obs_mean.to(raw_history.dtype)) / self.obs_std.to(raw_history.dtype)
        base = obs_norm.reshape(raw_history.shape[0], -1)

        prev = raw_history[:, 0, :]
        cur = raw_history[:, 1, :]
        per_prev = self.per_step_features(prev)
        per_cur = self.per_step_features(cur)

        tcp_delta = (cur[:, 18:21] - prev[:, 18:21]) / 0.05
        red_delta = (cur[:, 25:28] - prev[:, 25:28]) / 0.05
        qpos_delta = (cur[:, 0:7] - prev[:, 0:7]) / 0.20
        finger_delta = ((cur[:, 7:8] + cur[:, 8:9]) - (prev[:, 7:8] + prev[:, 8:9])) / 0.02
        offset_cur = cur[:, 18:21] - cur[:, 25:28]
        offset_prev = prev[:, 18:21] - prev[:, 25:28]
        offset_delta = (offset_cur - offset_prev) / 0.05
        drawer_delta = (cur[:, 39:40] - prev[:, 39:40]) / 0.05
        deltas = torch.cat([tcp_delta, red_delta, qpos_delta, finger_delta, offset_delta, drawer_delta], dim=-1)

        red_goal_err = cur[:, 25:28] - cur[:, 41:44]
        dx = torch.abs(red_goal_err[:, 0:1])
        dy = torch.abs(red_goal_err[:, 1:2])
        z = cur[:, 27:28]
        finger_width = cur[:, 7:8] + cur[:, 8:9]
        predicate_margins = torch.cat(
            [
                (0.04 - dx) / 0.04,
                (0.04 - dy) / 0.04,
                (z - 0.014) / 0.02,
                (0.031 - z) / 0.02,
                (cur[:, 39:40] - 0.26) / 0.05,
                (finger_width - 0.06) / 0.03,
            ],
            dim=-1,
        )
        features = torch.cat([base, per_prev, per_cur, deltas, predicate_margins], dim=-1)
        return self.net(features)


class PadFrameRedPlacementPolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        config = spec.get("candidate_config", {})
        self.horizon = int(spec["training"]["horizon"])
        self.condition_dim = int(config.get("condition_dim", 256))
        self.encoder = PadFrameConditionEncoder(spec, self.condition_dim)
        self.backbone = DiffusionBackbone(self.condition_dim, spec["training"])
        aux_hidden = int(config.get("aux_hidden", 256))
        self.aux_head = torch.nn.Sequential(
            torch.nn.Linear(self.condition_dim + self.horizon * 8 + 1, aux_hidden),
            torch.nn.Mish(),
            torch.nn.LayerNorm(aux_hidden),
            torch.nn.Linear(aux_hidden, aux_hidden),
            torch.nn.Mish(),
            torch.nn.Linear(aux_hidden, self.horizon * 5),
        )

    def forward(self, noisy_action, timestep, raw_history):
        condition = self.encoder(raw_history)
        return self.backbone(noisy_action, timestep, condition)

    def auxiliary_predictions(self, x0_action, raw_history, timestep):
        b = x0_action.shape[0]
        condition = self.encoder(raw_history)
        x0_limited = torch.clamp(x0_action, -2.0, 2.0).reshape(b, -1)
        if torch.is_tensor(timestep):
            t = timestep.to(device=x0_action.device, dtype=x0_action.dtype)
            if t.dim() == 0:
                t = t.expand(b)
        else:
            t = torch.full((b,), float(timestep), device=x0_action.device, dtype=x0_action.dtype)
        t = (t.reshape(b, 1) / 99.0).clamp(0.0, 1.5)
        out = self.aux_head(torch.cat([condition, x0_limited, t], dim=-1))
        return out.reshape(b, self.horizon, 5)


def build_model(spec):
    return PadFrameRedPlacementPolicy(spec)


def masked_mean(value, mask, channels=1):
    denom = (mask.sum() * channels).clamp_min(1.0)
    return (value * mask).sum() / denom


def red_on_pad_target(future_obs):
    red_pos = future_obs[:, :, 25:28]
    red_goal = future_obs[:, :, 41:44]
    quat = future_obs[:, :, 28:32]
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    w = quat[:, :, 0]
    x = quat[:, :, 1]
    y = quat[:, :, 2]
    zq = quat[:, :, 3]
    r00 = 1.0 - 2.0 * (y * y + zq * zq)
    r01 = 2.0 * (x * y - zq * w)
    r02 = 2.0 * (x * zq + y * w)
    r10 = 2.0 * (x * y + zq * w)
    r11 = 1.0 - 2.0 * (x * x + zq * zq)
    r12 = 2.0 * (y * zq - x * w)
    extent_x = 0.02 * (torch.abs(r00) + torch.abs(r01) + torch.abs(r02))
    extent_y = 0.02 * (torch.abs(r10) + torch.abs(r11) + torch.abs(r12))
    xy_err = torch.abs(red_pos[:, :, 0:2] - red_goal[:, :, 0:2])
    inside_x = xy_err[:, :, 0] + extent_x <= 0.06
    inside_y = xy_err[:, :, 1] + extent_y <= 0.06
    inside_z = (red_pos[:, :, 2] > 0.014) & (red_pos[:, :, 2] < 0.031)
    return (inside_x & inside_y & inside_z).to(future_obs.dtype).unsqueeze(-1)


def compute_loss(model, batch, spec):
    predicted_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = epsilon_loss(predicted_noise, batch["noise"], batch["mask"])

    alpha_bar = batch["alpha_bar"].to(batch["noisy_action"].dtype).reshape(-1, 1, 1)
    sqrt_alpha = torch.sqrt(alpha_bar.clamp_min(1.0e-5))
    sqrt_one_minus = torch.sqrt((1.0 - alpha_bar).clamp_min(0.0))
    x0_est = (batch["noisy_action"] - sqrt_one_minus * predicted_noise) / sqrt_alpha

    aux = model.auxiliary_predictions(x0_est, batch["raw_obs"], batch["timesteps"])
    pred_red_err = aux[:, :, 0:3]
    pred_onpad_logit = aux[:, :, 3:4]
    pred_open_logit = aux[:, :, 4:5]

    future_obs = batch["future_obs"]
    future_mask = batch["future_mask"].to(predicted_noise.dtype)
    low_noise_weight = (0.20 + 0.80 * alpha_bar).detach()
    weighted_mask = future_mask * low_noise_weight

    target_red_err = future_obs[:, :, 25:28] - future_obs[:, :, 41:44]
    scale = torch.tensor([0.15, 0.15, 0.15], device=target_red_err.device, dtype=target_red_err.dtype)
    target_red_err = torch.clamp(target_red_err / scale, -3.0, 3.0)
    red_err_loss = masked_mean((pred_red_err - target_red_err).square(), weighted_mask, channels=3)

    onpad_target = red_on_pad_target(future_obs)
    onpad_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        pred_onpad_logit, onpad_target, reduction="none"
    )
    onpad_loss = masked_mean(onpad_bce, weighted_mask, channels=1)

    finger_width = future_obs[:, :, 7:8] + future_obs[:, :, 8:9]
    open_target = (finger_width > 0.06).to(future_obs.dtype)
    open_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        pred_open_logit, open_target, reduction="none"
    )
    open_loss = masked_mean(open_bce, weighted_mask, channels=1)

    prior_weight = float(spec.get("candidate_config", {}).get("prior_weight", 0.05))
    prior_loss = prior_weight * (red_err_loss + 0.25 * onpad_loss + 0.10 * open_loss)
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
