import torch
import math
import appl.public as public


def candidate_config(spec):
    cfg = spec.get("candidate_config", {})
    if isinstance(cfg, dict) and "config" in cfg and isinstance(cfg["config"], dict):
        return cfg["config"]
    if isinstance(cfg, dict):
        return cfg
    return {}


class RedFrameFinalTransferPolicy(torch.nn.Module):
    """Learned DDPM epsilon model with active-red geometric conditioning."""

    def __init__(self, spec):
        super().__init__()
        training = spec["training"]
        cfg = candidate_config(spec)
        self.horizon = int(training.get("horizon", 16))
        self.cond_dim = int(cfg.get("condition_dimension", 256))
        self.pos_scale_value = float(cfg.get("relative_position_scale_m", 0.25))
        self.vel_scale_value = float(cfg.get("relative_delta_scale_m", 0.08))
        self.action_aux_scale = float(cfg.get("action_aux_timestep_scale", 99.0))

        normalizer = spec["normalizer"]
        self.register_buffer("obs_mean", torch.tensor(normalizer["mean"], dtype=torch.float32), persistent=False)
        self.register_buffer("obs_std", torch.tensor(normalizer["std"], dtype=torch.float32), persistent=False)
        self.register_buffer("pos_scale", torch.tensor([self.pos_scale_value, self.pos_scale_value, self.pos_scale_value], dtype=torch.float32), persistent=False)
        self.register_buffer("vel_scale", torch.tensor([self.vel_scale_value, self.vel_scale_value, self.vel_scale_value], dtype=torch.float32), persistent=False)

        with torch.no_grad():
            feature_dim = self.make_features(torch.zeros(1, 2, 47, dtype=torch.float32)).shape[-1]
        self.feature_dim = int(feature_dim)

        hidden = int(cfg.get("encoder_hidden", 256))
        self.condition_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, self.cond_dim),
            torch.nn.SiLU(),
        )
        self.backbone = public.DiffusionBackbone(self.cond_dim, training)

        aux_hidden = int(cfg.get("aux_hidden", 256))
        self.future_head = torch.nn.Sequential(
            torch.nn.Linear(self.cond_dim, aux_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(aux_hidden, aux_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(aux_hidden, self.horizon * 4),
        )
        self.final_error_head = torch.nn.Sequential(
            torch.nn.Linear(self.cond_dim, aux_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(aux_hidden, 3),
        )
        self.action_effect_head = torch.nn.Sequential(
            torch.nn.Linear(self.cond_dim + self.horizon * 8 + 1, aux_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(aux_hidden, aux_hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(aux_hidden, self.horizon * 4),
        )

    def safe_normalize_obs(self, raw_history):
        mean = self.obs_mean.to(device=raw_history.device, dtype=raw_history.dtype)
        std = self.obs_std.to(device=raw_history.device, dtype=raw_history.dtype).clamp_min(1.0e-6)
        return (raw_history - mean) / std

    def xyz(self, obs, start):
        return obs[..., start:start + 3]

    def scaled_rel(self, a, b):
        scale = self.pos_scale.to(device=a.device, dtype=a.dtype)
        return (a - b) / scale

    def scaled_delta(self, a, b):
        scale = self.vel_scale.to(device=a.device, dtype=a.dtype)
        return (a - b) / scale

    def make_features(self, raw_history):
        raw = raw_history
        norm_flat = self.safe_normalize_obs(raw).reshape(raw.shape[0], -1)
        prev = raw[:, 0]
        cur = raw[:, -1]

        pieces = [norm_flat]
        for obs in [prev, cur]:
            tcp = self.xyz(obs, 18)
            red = self.xyz(obs, 25)
            blue = self.xyz(obs, 32)
            red_goal = self.xyz(obs, 41)
            blue_goal = self.xyz(obs, 44)
            qpos = obs[..., 0:9]
            qvel = obs[..., 9:18]
            finger_left = qpos[..., 7:8]
            finger_right = qpos[..., 8:9]
            aperture = finger_left + finger_right

            pieces.extend([
                self.scaled_rel(tcp, red),
                self.scaled_rel(red_goal, red),
                self.scaled_rel(blue, red),
                self.scaled_rel(red, red_goal),
                self.scaled_rel(blue, blue_goal),
                self.scaled_rel(tcp, red_goal),
                self.scaled_rel(tcp, blue),
                self.scaled_rel(blue_goal, blue),
                self.scaled_rel(red_goal, blue_goal),
                tcp[..., 2:3] / self.pos_scale_value,
                red[..., 2:3] / self.pos_scale_value,
                blue[..., 2:3] / self.pos_scale_value,
                finger_left / 0.04,
                finger_right / 0.04,
                aperture / 0.08,
                qvel[..., 0:7],
                qvel[..., 7:9] / 0.05,
            ])

        prev_tcp = self.xyz(prev, 18)
        cur_tcp = self.xyz(cur, 18)
        prev_red = self.xyz(prev, 25)
        cur_red = self.xyz(cur, 25)
        prev_blue = self.xyz(prev, 32)
        cur_blue = self.xyz(cur, 32)
        cur_red_goal = self.xyz(cur, 41)
        cur_blue_goal = self.xyz(cur, 44)
        qpos_prev = prev[..., 0:9]
        qpos_cur = cur[..., 0:9]
        pieces.extend([
            self.scaled_delta(cur_tcp, prev_tcp),
            self.scaled_delta(cur_red, prev_red),
            self.scaled_delta(cur_blue, prev_blue),
            self.scaled_rel(cur_tcp, cur_red),
            self.scaled_rel(cur_red, cur_red_goal),
            self.scaled_rel(cur_blue, cur_blue_goal),
            ((qpos_cur[..., 7:8] + qpos_cur[..., 8:9]) - (qpos_prev[..., 7:8] + qpos_prev[..., 8:9])) / 0.04,
        ])
        tcp_red = cur_tcp - cur_red
        red_goal_err = cur_red - cur_red_goal
        blue_goal_err = cur_blue - cur_blue_goal
        pieces.extend([
            torch.linalg.norm(tcp_red[..., 0:2], dim=-1, keepdim=True) / self.pos_scale_value,
            torch.linalg.norm(tcp_red, dim=-1, keepdim=True) / self.pos_scale_value,
            torch.linalg.norm(red_goal_err[..., 0:2], dim=-1, keepdim=True) / self.pos_scale_value,
            torch.linalg.norm(blue_goal_err[..., 0:2], dim=-1, keepdim=True) / self.pos_scale_value,
            (cur_red[..., 2:3] - cur_red_goal[..., 2:3]) / self.pos_scale_value,
            (cur_tcp[..., 2:3] - cur_red[..., 2:3]) / self.pos_scale_value,
        ])
        return torch.cat(pieces, dim=-1)

    def encode_condition(self, raw_history):
        return self.condition_encoder(self.make_features(raw_history))

    def forward(self, noisy_action, timestep, raw_history):
        cond = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, cond)

    def predict_condition_future(self, raw_history):
        cond = self.encode_condition(raw_history)
        out = self.future_head(cond).reshape(raw_history.shape[0], self.horizon, 4)
        final = self.final_error_head(cond)
        return out, final, cond

    def predict_action_effect(self, cond, clean_action_estimate, timestep):
        B = clean_action_estimate.shape[0]
        if torch.is_tensor(timestep):
            t = timestep.to(device=clean_action_estimate.device, dtype=clean_action_estimate.dtype).reshape(B, 1)
        else:
            t = torch.full((B, 1), float(timestep), device=clean_action_estimate.device, dtype=clean_action_estimate.dtype)
        t = t / self.action_aux_scale
        action_flat = clean_action_estimate.clamp(-2.0, 2.0).reshape(B, self.horizon * 8)
        out = self.action_effect_head(torch.cat([cond, action_flat, t], dim=-1)).reshape(B, self.horizon, 4)
        return out


def build_model(spec):
    return RedFrameFinalTransferPolicy(spec)


def masked_mse(error, mask):
    denom = (mask.sum() * error.shape[-1]).clamp_min(1.0)
    return (error.square() * mask).sum() / denom


def weighted_masked_mse(error, mask, weight):
    wmask = mask * weight
    denom = (wmask.sum() * error.shape[-1]).clamp_min(1.0)
    return (error.square() * wmask).sum() / denom


def last_valid_target(target, mask):
    lengths = mask.squeeze(-1).sum(dim=1).long().clamp_min(1)
    idx = (lengths - 1).clamp(max=target.shape[1] - 1)
    batch_idx = torch.arange(target.shape[0], device=target.device)
    return target[batch_idx, idx]


def compute_loss(model, batch, spec):
    pred_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = public.epsilon_loss(pred_noise, batch["noise"], batch["mask"])

    cfg = candidate_config(spec)
    future_weight = float(cfg.get("future_geometry_loss_weight", 0.025))
    final_weight = float(cfg.get("final_goal_loss_weight", 0.02))
    action_effect_weight = float(cfg.get("action_effect_loss_weight", 0.005))

    future = batch["future_obs"]
    fmask = batch["future_mask"]
    dtype = future.dtype
    device = future.device
    scale = torch.tensor([model.pos_scale_value, model.pos_scale_value, model.pos_scale_value], device=device, dtype=dtype)

    red_xyz = future[..., 25:28]
    red_goal = future[..., 41:44]
    target_red_rel = (red_xyz - red_goal) / scale
    target_lift = (red_xyz[..., 2:3] - red_goal[..., 2:3]) / model.pos_scale_value
    target_traj = torch.cat([target_red_rel, target_lift], dim=-1)

    pred_traj, pred_final, cond = model.predict_condition_future(batch["raw_obs"])
    condition_future_loss = masked_mse(pred_traj - target_traj.to(pred_traj.dtype), fmask.to(pred_traj.dtype))
    final_target = last_valid_target(target_red_rel, fmask).to(pred_final.dtype)
    final_loss = (pred_final - final_target).square().mean()

    alpha_bar = batch["alpha_bar"].to(device=batch["noisy_action"].device, dtype=batch["noisy_action"].dtype).reshape(-1, 1, 1)
    sqrt_ab = torch.sqrt(alpha_bar.clamp_min(1.0e-8))
    sqrt_one_minus = torch.sqrt((1.0 - alpha_bar).clamp_min(0.0))
    clean_est = (batch["noisy_action"] - sqrt_one_minus * pred_noise) / sqrt_ab
    pred_effect = model.predict_action_effect(cond, clean_est, batch["timesteps"])
    action_effect_loss = weighted_masked_mse(pred_effect - target_traj.to(pred_effect.dtype), fmask.to(pred_effect.dtype), alpha_bar.detach().clamp_min(0.05))

    prior_loss = future_weight * condition_future_loss + final_weight * final_loss + action_effect_weight * action_effect_loss
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
