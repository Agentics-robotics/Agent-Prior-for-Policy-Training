import math
import torch
from appl.public import DiffusionBackbone, epsilon_loss


class ActiveGoalFunnelPolicy(torch.nn.Module):
    """Learned DDPM epsilon model with goal-relative active-object conditioning."""

    def __init__(self, spec):
        super().__init__()
        self.spec_training = spec["training"]
        n = spec["normalizer"]
        self.register_buffer("obs_mean", torch.tensor(n["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.tensor(n["std"], dtype=torch.float32))
        self.register_buffer("pos_scale", torch.tensor([0.25, 0.25, 0.30], dtype=torch.float32))
        self.register_buffer("dist_scale", torch.tensor([0.40], dtype=torch.float32))
        config = spec["candidate_config"] if "candidate_config" in spec else {}
        self.condition_dim = int(config["condition_dim"] if "condition_dim" in config else 256)
        self.hidden_dim = int(config["encoder_hidden_dim"] if "encoder_hidden_dim" in config else 384)

        with torch.no_grad():
            feature_dim = int(self.make_features(torch.zeros(1, 2, 47, dtype=torch.float32)).shape[-1])
        self.feature_dim = feature_dim

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, self.hidden_dim),
            torch.nn.Mish(),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.Mish(),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Linear(self.hidden_dim, self.condition_dim),
        )
        self.backbone = DiffusionBackbone(self.condition_dim, self.spec_training)
        self.aux_head = torch.nn.Sequential(
            torch.nn.Linear(self.condition_dim + 8 + 1, 192),
            torch.nn.Mish(),
            torch.nn.Linear(192, 192),
            torch.nn.Mish(),
            torch.nn.Linear(192, 7),
        )

    def split_state(self, x):
        return (
            x[..., 0:9],
            x[..., 9:18],
            x[..., 18:25],
            x[..., 25:32],
            x[..., 32:39],
            x[..., 41:44],
            x[..., 44:47],
        )

    def role_weights_from_current(self, cur):
        tcp = cur[..., 18:21]
        red = cur[..., 25:28]
        blue = cur[..., 32:35]
        red_goal = cur[..., 41:44]
        blue_goal = cur[..., 44:47]
        red_tcp = torch.linalg.norm(tcp - red, dim=-1)
        blue_tcp = torch.linalg.norm(tcp - blue, dim=-1)
        red_xy = torch.linalg.norm(red[..., :2] - red_goal[..., :2], dim=-1)
        blue_xy = torch.linalg.norm(blue[..., :2] - blue_goal[..., :2], dim=-1)
        red_z = red[..., 2] - red_goal[..., 2]
        blue_z = blue[..., 2] - blue_goal[..., 2]
        red_score = -5.0 * red_tcp + 8.0 * red_z + 2.5 * red_xy
        blue_score = -5.0 * blue_tcp + 8.0 * blue_z + 2.5 * blue_xy
        scores = torch.stack([red_score, blue_score], dim=-1)
        return torch.softmax(scores, dim=-1)

    def make_features(self, raw_history):
        raw = raw_history
        b = raw.shape[0]
        normed = (raw - self.obs_mean.to(raw.device, raw.dtype)) / self.obs_std.to(raw.device, raw.dtype).clamp_min(1.0e-6)
        flat_normed = normed.reshape(b, -1)
        prev = raw[:, 0]
        cur = raw[:, -1]
        qpos, qvel, tcp_pose, red_pose, blue_pose, red_goal, blue_goal = self.split_state(cur)
        _, _, prev_tcp_pose, prev_red_pose, prev_blue_pose, _, _ = self.split_state(prev)
        tcp = tcp_pose[..., 0:3]
        red = red_pose[..., 0:3]
        blue = blue_pose[..., 0:3]
        scale = self.pos_scale.to(raw.device, raw.dtype)
        red_rel = (red - red_goal) / scale
        blue_rel = (blue - blue_goal) / scale
        tcp_red = (tcp - red) / scale
        tcp_blue = (tcp - blue) / scale
        red_blue = (red - blue) / scale
        goal_delta = (red_goal - blue_goal) / scale
        red_tcp_dist = torch.linalg.norm(tcp - red, dim=-1, keepdim=True) / self.dist_scale.to(raw.device, raw.dtype)
        blue_tcp_dist = torch.linalg.norm(tcp - blue, dim=-1, keepdim=True) / self.dist_scale.to(raw.device, raw.dtype)
        finger_mean = qpos[..., 7:9].mean(dim=-1, keepdim=True)
        finger_diff = qpos[..., 7:8] - qpos[..., 8:9]
        finger_feats = torch.cat([finger_mean / 0.04, finger_diff / 0.02, qpos[..., 7:9] / 0.04], dim=-1)
        red_vel = (red - prev_red_pose[..., 0:3]) / scale
        blue_vel = (blue - prev_blue_pose[..., 0:3]) / scale
        tcp_vel = (tcp - prev_tcp_pose[..., 0:3]) / scale
        prev_red_rel = (prev_red_pose[..., 0:3] - red_goal) / scale
        prev_blue_rel = (prev_blue_pose[..., 0:3] - blue_goal) / scale
        role = self.role_weights_from_current(cur)
        pr = role[..., 0:1]
        pb = role[..., 1:2]
        active_rel = pr * red_rel + pb * blue_rel
        active_tcp = pr * tcp_red + pb * tcp_blue
        active_quat = pr * red_pose[..., 3:7] + pb * blue_pose[..., 3:7]
        inactive_rel = pr * blue_rel + pb * red_rel
        active_goal = (pr * red_goal + pb * blue_goal) / scale
        active_height = active_rel[..., 2:3]
        active_xy_err = torch.linalg.norm(active_rel[..., 0:2], dim=-1, keepdim=True)
        active_tcp_norm = torch.linalg.norm(active_tcp, dim=-1, keepdim=True)
        inactive_xy_err = torch.linalg.norm(inactive_rel[..., 0:2], dim=-1, keepdim=True)
        release_cues = torch.cat([active_height, active_xy_err, active_tcp_norm, inactive_xy_err], dim=-1)
        feats = torch.cat([
            flat_normed,
            red_rel,
            blue_rel,
            tcp_red,
            tcp_blue,
            red_tcp_dist,
            blue_tcp_dist,
            red_blue,
            goal_delta,
            tcp_pose[..., 3:7],
            red_pose[..., 3:7],
            blue_pose[..., 3:7],
            finger_feats,
            red_vel,
            blue_vel,
            tcp_vel,
            prev_red_rel,
            prev_blue_rel,
            role,
            active_rel,
            active_tcp,
            active_quat,
            inactive_rel,
            pr - pb,
            active_goal,
            release_cues,
        ], dim=-1)
        return feats

    def encode_condition(self, raw_history):
        return self.encoder(self.make_features(raw_history))

    def forward(self, noisy_action, timestep, raw_history):
        condition = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, condition)

    def auxiliary_prediction(self, raw_history, noisy_action, predicted_noise, alpha_bar):
        condition = self.encode_condition(raw_history)
        ab = alpha_bar.reshape(-1, 1, 1).to(noisy_action.device, noisy_action.dtype).clamp(1.0e-4, 0.9999)
        x0 = (noisy_action - torch.sqrt(1.0 - ab) * predicted_noise) / torch.sqrt(ab)
        h = noisy_action.shape[1]
        if h > 1:
            steps = torch.linspace(-1.0, 1.0, h, device=noisy_action.device, dtype=noisy_action.dtype)
        else:
            steps = torch.zeros(h, device=noisy_action.device, dtype=noisy_action.dtype)
        steps = steps.view(1, h, 1).expand(noisy_action.shape[0], h, 1)
        cond_h = condition.unsqueeze(1).expand(-1, h, -1)
        return self.aux_head(torch.cat([cond_h, x0, steps], dim=-1))


def build_model(spec):
    return ActiveGoalFunnelPolicy(spec)


def auxiliary_targets(model, batch):
    raw = batch["raw_obs"]
    future = batch["future_obs"]
    cur = raw[:, -1]
    role = model.role_weights_from_current(cur).unsqueeze(1)
    pr = role[..., 0:1]
    pb = role[..., 1:2]
    red_goal = cur[:, 41:44].unsqueeze(1)
    blue_goal = cur[:, 44:47].unsqueeze(1)
    red_f = future[..., 25:28]
    blue_f = future[..., 32:35]
    tcp_f = future[..., 18:21]
    active_rel = pr * (red_f - red_goal) + pb * (blue_f - blue_goal)
    active_obj = pr * red_f + pb * blue_f
    tcp_active = tcp_f - active_obj
    scale = model.pos_scale.to(future.device, future.dtype).view(1, 1, 3)
    rel_t = active_rel / scale
    tcp_t = tcp_active / scale
    finger = future[..., 7:9].mean(dim=-1, keepdim=True)
    open_t = ((finger - 0.029) / 0.011).clamp(-1.0, 1.0)
    target = torch.cat([rel_t, tcp_t, open_t], dim=-1)
    xy = torch.linalg.norm(active_rel[..., 0:2], dim=-1, keepdim=True)
    z = active_rel[..., 2:3].abs()
    funnel = torch.exp(-xy / 0.050) * torch.exp(-z / 0.080)
    open_amount = ((finger - 0.020) / 0.020).clamp(0.0, 1.0)
    weight = 1.0 + 2.0 * funnel + 0.75 * open_amount
    return target, weight


def compute_loss(model, batch, spec):
    predicted_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = epsilon_loss(predicted_noise, batch["noise"], batch["mask"])
    aux_pred = model.auxiliary_prediction(batch["raw_obs"], batch["noisy_action"], predicted_noise, batch["alpha_bar"])
    aux_target, aux_weight = auxiliary_targets(model, batch)
    mask = batch["future_mask"]
    component_weight = torch.tensor([1.0, 1.0, 1.25, 0.35, 0.35, 0.45, 0.25], device=aux_pred.device, dtype=aux_pred.dtype).view(1, 1, 7)
    sample_weight = (0.25 + 0.75 * batch["alpha_bar"].to(aux_pred.device, aux_pred.dtype).view(-1, 1, 1))
    aux_sq = (aux_pred - aux_target).square() * component_weight * aux_weight * sample_weight * mask
    aux_loss = aux_sq.sum() / (mask.sum() * aux_pred.shape[-1]).clamp_min(1.0)
    config = spec["candidate_config"] if "candidate_config" in spec else {}
    lam = float(config["prior_loss_weight"] if "prior_loss_weight" in config else 0.05)
    prior_loss = aux_loss * lam
    return {"loss": diffusion_loss + prior_loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
