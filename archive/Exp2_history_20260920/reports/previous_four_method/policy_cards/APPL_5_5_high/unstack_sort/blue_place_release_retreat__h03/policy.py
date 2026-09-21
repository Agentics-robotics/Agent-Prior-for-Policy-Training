import torch
from appl.public import DiffusionBackbone, epsilon_loss


class TerminalRetreatDiffusionPolicy(torch.nn.Module):
    """Action diffusion model with a terminal retreat / stability prior."""

    def __init__(self, spec):
        super().__init__()
        training = spec["training"]
        cfg = spec.get("candidate_config", {}).get("config", {})

        self.horizon = int(training.get("horizon", 16))
        self.action_dim = 8
        self.obs_dim = int(spec.get("observation_dimension", 47))
        self.condition_dim = int(cfg.get("condition_dim", 256))
        hidden = int(cfg.get("encoder_hidden_dim", 384))
        aux_hidden = int(cfg.get("aux_hidden_dim", 192))

        n = spec["normalizer"]
        self.register_buffer("obs_mean", torch.tensor(n["mean"], dtype=torch.float32), persistent=False)
        self.register_buffer("obs_std", torch.tensor(n["std"], dtype=torch.float32), persistent=False)

        feature_dim = self.obs_dim * 2 + 56
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, self.condition_dim),
            torch.nn.LayerNorm(self.condition_dim),
            torch.nn.GELU(),
        )

        self.backbone = DiffusionBackbone(self.condition_dim, training)

        self.phase_head = torch.nn.Sequential(
            torch.nn.Linear(self.condition_dim, aux_hidden),
            torch.nn.GELU(),
            torch.nn.Linear(aux_hidden, 4),
        )
        action_summary_dim = self.action_dim * 3
        self.terminal_head = torch.nn.Sequential(
            torch.nn.Linear(self.condition_dim + action_summary_dim, aux_hidden),
            torch.nn.GELU(),
            torch.nn.Linear(aux_hidden, aux_hidden),
            torch.nn.GELU(),
            torch.nn.Linear(aux_hidden, 2),
        )

    def norm_obs(self, raw_history):
        return (raw_history - self.obs_mean.to(raw_history.dtype)) / self.obs_std.to(raw_history.dtype)

    def relative_features(self, raw_history):
        prev = raw_history[:, 0]
        cur = raw_history[:, -1]

        qpos = cur[:, 0:9]
        qvel = cur[:, 9:18]
        tcp = cur[:, 18:25]
        red = cur[:, 25:32]
        blue = cur[:, 32:39]
        red_goal = cur[:, 41:44]
        blue_goal = cur[:, 44:47]

        tcp_prev = prev[:, 18:25]
        red_prev = prev[:, 25:32]
        blue_prev = prev[:, 32:39]
        qpos_prev = prev[:, 0:9]

        blue_to_goal = (blue[:, 0:3] - blue_goal) / 0.25
        red_to_goal = (red[:, 0:3] - red_goal) / 0.25
        tcp_to_blue = (tcp[:, 0:3] - blue[:, 0:3]) / 0.10
        tcp_to_red = (tcp[:, 0:3] - red[:, 0:3]) / 0.25
        tcp_to_blue_goal = (tcp[:, 0:3] - blue_goal) / 0.25
        tcp_delta = (tcp[:, 0:3] - tcp_prev[:, 0:3]) * 10.0
        blue_delta = (blue[:, 0:3] - blue_prev[:, 0:3]) * 10.0
        red_delta = (red[:, 0:3] - red_prev[:, 0:3]) * 10.0
        qpos_delta = qpos - qpos_prev

        finger_left = qpos[:, 7:8]
        finger_right = qpos[:, 8:9]
        finger_width = finger_left + finger_right
        finger_balance = finger_left - finger_right
        finger_vel_mean = (qvel[:, 7:8] + qvel[:, 8:9]) * 0.5
        finger_features = torch.cat(
            [finger_left / 0.04, finger_right / 0.04, finger_width / 0.08,
             finger_balance / 0.02, finger_vel_mean / 0.05], dim=-1)

        local_scalars = torch.cat([
            (tcp[:, 2:3] - blue[:, 2:3]) / 0.25,
            (blue[:, 2:3] - 0.02) / 0.25,
            (red[:, 2:3] - 0.02) / 0.25,
            torch.linalg.norm(blue[:, 0:2] - blue_goal[:, 0:2], dim=-1, keepdim=True) / 0.25,
            torch.linalg.norm(red[:, 0:2] - red_goal[:, 0:2], dim=-1, keepdim=True) / 0.25,
            torch.linalg.norm(tcp[:, 0:3] - blue[:, 0:3], dim=-1, keepdim=True) / 0.25,
        ], dim=-1)

        quat_features = torch.cat([tcp[:, 3:7], red[:, 3:7], blue[:, 3:7]], dim=-1)

        return torch.cat([
            blue_to_goal, red_to_goal, tcp_to_blue, tcp_to_red, tcp_to_blue_goal,
            tcp_delta, blue_delta, red_delta, qpos_delta,
            finger_features, local_scalars, quat_features,
        ], dim=-1)

    def encode_condition(self, raw_history):
        norm_hist = self.norm_obs(raw_history).reshape(raw_history.shape[0], -1)
        rel = self.relative_features(raw_history)
        features = torch.cat([norm_hist, rel], dim=-1)
        return self.encoder(features)

    def forward(self, noisy_action, timestep, raw_history):
        cond = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, cond)

    def action_summary(self, encoded_x0, mask):
        m = mask.to(dtype=encoded_x0.dtype)
        denom = m.sum(dim=1).clamp_min(1.0)
        mean_action = (encoded_x0 * m).sum(dim=1) / denom
        first_action = encoded_x0[:, 0]
        valid_counts = m[:, :, 0].sum(dim=1).long().clamp_min(1)
        last_indices = (valid_counts - 1).view(-1, 1, 1).expand(-1, 1, encoded_x0.shape[-1])
        last_action = encoded_x0.gather(1, last_indices).squeeze(1)
        return torch.cat([mean_action, first_action, last_action], dim=-1)

    def auxiliary_predictions(self, raw_history, encoded_x0, mask):
        cond = self.encode_condition(raw_history)
        phase_logits = self.phase_head(cond)
        action_feat = self.action_summary(encoded_x0, mask)
        terminal = self.terminal_head(torch.cat([cond, action_feat], dim=-1))
        return terminal[:, 0], terminal[:, 1], phase_logits


def last_valid_future(future_obs, future_mask):
    counts = future_mask[:, :, 0].sum(dim=1).long().clamp_min(1)
    idx = (counts - 1).view(-1, 1, 1).expand(-1, 1, future_obs.shape[-1])
    return future_obs.gather(1, idx).squeeze(1)


def success_target(obs):
    red = obs[:, 25:28]
    blue = obs[:, 32:35]
    red_goal = obs[:, 41:44]
    blue_goal = obs[:, 44:47]
    red_xy = (red[:, 0:2] - red_goal[:, 0:2]).abs().amax(dim=-1)
    blue_xy = (blue[:, 0:2] - blue_goal[:, 0:2]).abs().amax(dim=-1)
    red_z = (red[:, 2] - red_goal[:, 2]).abs()
    blue_z = (blue[:, 2] - blue_goal[:, 2]).abs()
    ok = (red_xy <= 0.04) & (blue_xy <= 0.04) & (red_z <= 0.011) & (blue_z <= 0.011)
    return ok.to(dtype=obs.dtype)


def phase_target(cur):
    blue = cur[:, 32:35]
    blue_goal = cur[:, 44:47]
    tcp = cur[:, 18:21]
    finger_width = cur[:, 7] + cur[:, 8]
    blue_goal_xy = torch.linalg.norm(blue[:, 0:2] - blue_goal[:, 0:2], dim=-1)
    blue_at_table = blue[:, 2] <= 0.035
    open_fingers = finger_width >= 0.065
    high_tcp = tcp[:, 2] >= 0.10

    phase = torch.zeros(cur.shape[0], device=cur.device, dtype=torch.long)
    phase = torch.where((blue[:, 2] < 0.16) & (~blue_at_table), torch.ones_like(phase), phase)
    phase = torch.where(blue_at_table & (blue_goal_xy < 0.05), torch.full_like(phase, 2), phase)
    phase = torch.where(blue_at_table & open_fingers & high_tcp, torch.full_like(phase, 3), phase)
    return phase


def safe_mean(weighted_values, weights):
    return weighted_values.sum() / weights.sum().clamp_min(1.0)


def build_model(spec):
    return TerminalRetreatDiffusionPolicy(spec)


def compute_loss(model, batch, spec):
    pred_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = epsilon_loss(pred_noise, batch["noise"], batch["mask"])

    alpha_bar = batch["alpha_bar"].view(-1, 1, 1).to(dtype=pred_noise.dtype)
    sqrt_alpha = torch.sqrt(alpha_bar.clamp_min(1.0e-6))
    sqrt_one_minus = torch.sqrt((1.0 - alpha_bar).clamp_min(0.0))
    encoded_x0 = (batch["noisy_action"] - sqrt_one_minus * pred_noise) / sqrt_alpha
    encoded_x0 = encoded_x0.clamp(-1.25, 1.25)

    success_logit, disturb_logit, phase_logits = model.auxiliary_predictions(
        batch["raw_obs"], encoded_x0, batch["mask"])

    cur = batch["raw_obs"][:, -1]
    last_future = last_valid_future(batch["future_obs"], batch["future_mask"])
    success = success_target(last_future)

    low_noise_w = batch["alpha_bar"].to(dtype=pred_noise.dtype).clamp(0.05, 1.0)
    bce_success = torch.nn.functional.binary_cross_entropy_with_logits(
        success_logit, success, reduction="none")
    success_loss = (bce_success * low_noise_w).mean()

    red_disp = torch.linalg.norm(last_future[:, 25:28] - cur[:, 25:28], dim=-1)
    blue_disp = torch.linalg.norm(last_future[:, 32:35] - cur[:, 32:35], dim=-1)
    disturbance = ((red_disp + blue_disp) / 0.025).clamp(0.0, 1.0)
    bce_disturb = torch.nn.functional.binary_cross_entropy_with_logits(
        disturb_logit, disturbance, reduction="none")
    disturb_loss = (bce_disturb * low_noise_w).mean()

    phase_loss = torch.nn.functional.cross_entropy(phase_logits, phase_target(cur))

    blue = cur[:, 32:35]
    blue_goal = cur[:, 44:47]
    blue_goal_xy = torch.linalg.norm(blue[:, 0:2] - blue_goal[:, 0:2], dim=-1)
    finger_width = cur[:, 7] + cur[:, 8]
    post_release = ((blue[:, 2] <= 0.035) & (blue_goal_xy < 0.05) & (finger_width >= 0.065)).to(pred_noise.dtype)
    gripper = encoded_x0[:, :, 7]
    action_mask = batch["mask"][:, :, 0].to(dtype=pred_noise.dtype)
    open_weights = action_mask * post_release.view(-1, 1) * low_noise_w.view(-1, 1)
    open_loss = safe_mean((gripper - 1.0).square() * open_weights, open_weights)

    cfg = spec.get("candidate_config", {}).get("config", {})
    prior_loss = (float(cfg.get("success_loss_weight", 0.04)) * success_loss +
                  float(cfg.get("disturbance_loss_weight", 0.04)) * disturb_loss +
                  float(cfg.get("phase_loss_weight", 0.015)) * phase_loss +
                  float(cfg.get("open_gripper_loss_weight", 0.025)) * open_loss)
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
