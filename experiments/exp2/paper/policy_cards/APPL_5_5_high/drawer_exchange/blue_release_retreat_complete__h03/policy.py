import math

import torch
import appl.public as public


class BlueReleaseRetreatModel(torch.nn.Module):
    """Near-goal residual stabilizer implemented as a conditional diffusion model.

    The model does not execute a scripted stabilizing controller. It learns a
    nominal encoded action trajectory from causal state features and conditions a
    standard DDPM noise predictor on that nominal trajectory. Auxiliary losses
    keep the learned denoised trajectory close to this local nominal and to a
    learned future-state descriptor for the release/retreat phase.
    """

    def __init__(self, spec):
        super().__init__()
        normalizer = spec.get("normalizer", spec.get("shared_normalizer"))
        training = spec["training"]
        self.horizon = int(training["horizon"])
        self.action_dim = 8
        self.desc_dim = 10
        self.feature_dim = 187
        self.latent_dim = int(spec.get("candidate_config", {}).get("latent_dim", 256))
        self.condition_dim = int(spec.get("candidate_config", {}).get("condition_dim", 256))

        self.register_buffer("obs_mean", torch.tensor(normalizer["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.tensor(normalizer["std"], dtype=torch.float32))
        self.register_buffer("action_min", torch.tensor(normalizer["action_min"], dtype=torch.float32))
        self.register_buffer("action_scale", torch.tensor(normalizer["action_scale"], dtype=torch.float32))
        self.register_buffer("pos_scale", torch.tensor([0.15, 0.15, 0.15], dtype=torch.float32))
        self.register_buffer("drawer_origin", torch.tensor([0.19, 0.0, 0.035], dtype=torch.float32))

        self.feature_net = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, self.latent_dim),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(self.latent_dim),
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.SiLU(),
        )
        self.nominal_head = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.latent_dim, self.horizon * self.action_dim),
        )
        self.condition_net = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim + self.horizon * self.action_dim, self.condition_dim),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(self.condition_dim),
            torch.nn.Linear(self.condition_dim, self.condition_dim),
        )
        self.outcome_head = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim + self.horizon * self.action_dim, self.latent_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.latent_dim, self.horizon * self.desc_dim),
        )
        self.backbone = public.DiffusionBackbone(self.condition_dim, training)

    def normalize_obs_history(self, raw_history):
        return (raw_history - self.obs_mean.to(raw_history.dtype)) / self.obs_std.to(raw_history.dtype)

    def features_for_one_observation(self, obs):
        dtype = obs.dtype
        pos_scale = self.pos_scale.to(dtype)
        blue = obs[:, 32:35]
        blue_q = obs[:, 35:39]
        tcp = obs[:, 18:21]
        tcp_q = obs[:, 21:25]
        red = obs[:, 25:28]
        red_q = obs[:, 28:32]
        drawer_pos = obs[:, 39:40]
        drawer_vel = obs[:, 40:41]
        red_goal = obs[:, 41:44]
        blue_goal = obs[:, 44:47]
        drawer_center = torch.cat(
            [0.19 - drawer_pos, torch.zeros_like(drawer_pos), torch.full_like(drawer_pos, 0.035)],
            dim=-1,
        )
        finger_width = obs[:, 7:8] + obs[:, 8:9]
        finger_cmd = (0.5 * finger_width - 0.015) / 0.025
        blue_err = (blue - blue_goal) / pos_scale
        tcp_to_blue = (tcp - blue) / pos_scale
        tcp_to_goal = (tcp - blue_goal) / pos_scale
        red_err = (red - red_goal) / pos_scale
        blue_drawer = (blue - drawer_center) / pos_scale
        tcp_drawer = (tcp - drawer_center) / pos_scale
        dist_blue_goal = torch.linalg.norm(blue_err, dim=-1, keepdim=True)
        dist_tcp_blue = torch.linalg.norm(tcp_to_blue, dim=-1, keepdim=True)
        z_blue_goal = (blue[:, 2:3] - blue_goal[:, 2:3]) / 0.15
        z_tcp_blue = (tcp[:, 2:3] - blue[:, 2:3]) / 0.15
        return torch.cat(
            [
                blue_err,
                tcp_to_blue,
                tcp_to_goal,
                red_err,
                blue_drawer,
                tcp_drawer,
                blue_q,
                tcp_q,
                red_q,
                drawer_pos / 0.30,
                drawer_vel / 0.10,
                finger_width / 0.08,
                finger_cmd,
                dist_blue_goal,
                dist_tcp_blue,
                z_blue_goal,
                z_tcp_blue,
            ],
            dim=-1,
        )

    def make_features(self, raw_history):
        raw_norm = self.normalize_obs_history(raw_history).reshape(raw_history.shape[0], -1)
        prev = raw_history[:, 0]
        cur = raw_history[:, 1]
        per_step = torch.cat([self.features_for_one_observation(prev), self.features_for_one_observation(cur)], dim=-1)
        delta_blue = (cur[:, 32:35] - prev[:, 32:35]) / 0.05
        delta_tcp = (cur[:, 18:21] - prev[:, 18:21]) / 0.05
        delta_qpos = (cur[:, 0:9] - prev[:, 0:9]) / 0.10
        delta_finger = ((cur[:, 7:8] + cur[:, 8:9]) - (prev[:, 7:8] + prev[:, 8:9])) / 0.02
        delta_drawer = (cur[:, 39:40] - prev[:, 39:40]) / 0.01
        delta = torch.cat([delta_blue, delta_tcp, delta_qpos, delta_finger, delta_drawer], dim=-1)
        return torch.cat([raw_norm, per_step, delta], dim=-1)

    def condition_parts(self, raw_history):
        features = self.make_features(raw_history)
        latent = self.feature_net(features)
        nominal = self.nominal_head(latent).reshape(-1, self.horizon, self.action_dim)
        condition = self.condition_net(torch.cat([latent, nominal.reshape(nominal.shape[0], -1)], dim=-1))
        return latent, nominal, condition

    def forward(self, noisy_action, timestep, raw_history):
        parts = self.condition_parts(raw_history)
        condition = parts[2]
        return self.backbone(noisy_action, timestep, condition)

    def predict_outcome_descriptor(self, raw_history, clean_action_estimate):
        latent = self.feature_net(self.make_features(raw_history))
        clean_flat = clean_action_estimate.clamp(-2.5, 2.5).reshape(clean_action_estimate.shape[0], -1)
        pred = self.outcome_head(torch.cat([latent, clean_flat], dim=-1))
        return pred.reshape(-1, self.horizon, self.desc_dim)

    def future_descriptor_target(self, future_obs):
        blue = future_obs[:, :, 32:35]
        tcp = future_obs[:, :, 18:21]
        blue_goal = future_obs[:, :, 44:47]
        drawer_pos = future_obs[:, :, 39:40]
        finger_width = future_obs[:, :, 7:8] + future_obs[:, :, 8:9]
        qvel = future_obs[:, :, 9:16]
        blue_err = (blue - blue_goal) / self.pos_scale.to(future_obs.dtype)
        tcp_to_blue = (tcp - blue) / self.pos_scale.to(future_obs.dtype)
        qvel_norm = torch.linalg.norm(qvel, dim=-1, keepdim=True) / 0.50
        tcp_z = tcp[:, :, 2:3] / 0.35
        return torch.cat(
            [blue_err, tcp_to_blue, finger_width / 0.08, qvel_norm, drawer_pos / 0.30, tcp_z],
            dim=-1,
        )


def build_model(spec):
    return BlueReleaseRetreatModel(spec)


def masked_mse(value, target, mask, weights=None):
    err = (value - target).square()
    if weights is not None:
        adjusted = weights
        while adjusted.dim() < err.dim():
            adjusted = adjusted.unsqueeze(0)
        err = err * adjusted.to(device=err.device, dtype=err.dtype)
    err = err * mask
    denom = (mask.sum() * err.shape[-1]).clamp_min(1.0)
    return err.sum() / denom


def compute_loss(model, batch, spec):
    pred_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    mask = batch["mask"]
    diffusion_loss = public.epsilon_loss(pred_noise, batch["noise"], mask)

    parts = model.condition_parts(batch["raw_obs"])
    nominal = parts[1]
    action_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5], device=nominal.device, dtype=nominal.dtype)
    nominal_loss = masked_mse(nominal, batch["encoded_action"], mask, action_weights)

    alpha = batch["alpha_bar"].to(pred_noise.dtype).reshape(-1, 1, 1)
    clean_estimate = (batch["noisy_action"] - torch.sqrt((1.0 - alpha).clamp_min(0.0)) * pred_noise) / torch.sqrt(alpha.clamp_min(1.0e-6))
    low_noise_weight = ((alpha - 0.25) / 0.75).clamp(0.0, 1.0)
    residual = clean_estimate - nominal
    residual_loss = ((residual.square() * mask * low_noise_weight).sum() / ((mask * low_noise_weight).sum() * residual.shape[-1]).clamp_min(1.0))

    outcome_pred = model.predict_outcome_descriptor(batch["raw_obs"], clean_estimate)
    outcome_target = model.future_descriptor_target(batch["future_obs"])
    fmask = batch["future_mask"]
    outcome_loss = ((outcome_pred - outcome_target).square() * fmask * low_noise_weight).sum() / ((fmask * low_noise_weight).sum() * outcome_pred.shape[-1]).clamp_min(1.0)

    prior_loss = 0.10 * nominal_loss + 0.005 * residual_loss + 0.01 * outcome_loss
    total_loss = diffusion_loss + prior_loss
    return {"loss": total_loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
