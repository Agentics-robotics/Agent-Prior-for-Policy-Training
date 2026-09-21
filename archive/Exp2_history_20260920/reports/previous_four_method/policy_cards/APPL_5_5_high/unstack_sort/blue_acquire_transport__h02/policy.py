import torch
from appl.public import DiffusionBackbone, epsilon_loss


class HeightStagedBlueTransportPolicy(torch.nn.Module):
    """Learned diffusion policy with a height-staged blue-acquire/transport prior."""

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        cfg = spec.get("candidate_config", {}).get("config", {})
        self.safe_height = float(cfg.get("safe_height_m", 0.28))
        self.goal_xy_threshold = float(cfg.get("goal_xy_threshold_m", 0.04))
        self.closed_finger_width = float(cfg.get("closed_finger_width_m", 0.032))
        self.feature_dim = 142
        self.latent_dim = int(cfg.get("latent_dim", 192))
        self.condition_dim = int(cfg.get("condition_dim", 256))

        normalizer = spec["normalizer"]
        self.register_buffer("obs_mean", torch.as_tensor(normalizer["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.as_tensor(normalizer["std"], dtype=torch.float32))

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, 256),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 256),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, self.latent_dim),
            torch.nn.SiLU(),
        )
        self.stage_head = torch.nn.Linear(self.latent_dim, 6)
        self.waypoint_head = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 3),
        )
        self.future_blue_head = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 9),
        )
        self.grip_head = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Tanh(),
        )
        self.condition_head = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim + 6 + 3 + 9 + 1, self.condition_dim),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(self.condition_dim),
            torch.nn.Linear(self.condition_dim, self.condition_dim),
            torch.nn.SiLU(),
        )
        self.backbone = DiffusionBackbone(self.condition_dim, spec["training"])

    def phase_labels_from_last(self, last):
        tcp = last[:, 18:21]
        blue = last[:, 32:35]
        goal = last[:, 44:47]
        fingers = last[:, 7:9]
        finger_avg = fingers.mean(dim=-1)
        tcp_blue_xy = torch.linalg.norm((tcp - blue)[:, :2], dim=-1)
        blue_goal_xy = torch.linalg.norm((goal - blue)[:, :2], dim=-1)
        blue_z = blue[:, 2]
        closed = finger_avg < self.closed_finger_width

        labels = torch.zeros(last.shape[0], device=last.device, dtype=torch.long)
        aligned_open = (tcp_blue_xy < 0.08) & (~closed) & (blue_z < 0.06)
        grasp_low = (finger_avg < 0.036) & (blue_z < 0.06) & (tcp_blue_xy < 0.05)
        lifting = closed & (blue_z >= 0.06) & (blue_z < 0.245)
        translating = closed & (blue_z >= 0.245) & (blue_goal_xy >= self.goal_xy_threshold)
        goal_near = closed & (blue_z >= 0.18) & (blue_goal_xy < self.goal_xy_threshold)

        labels = torch.where(aligned_open, torch.ones_like(labels), labels)
        labels = torch.where(grasp_low, torch.full_like(labels, 2), labels)
        labels = torch.where(lifting, torch.full_like(labels, 3), labels)
        labels = torch.where(translating, torch.full_like(labels, 4), labels)
        labels = torch.where(goal_near, torch.full_like(labels, 5), labels)
        return labels

    def one_hot_phase(self, raw_history):
        labels = self.phase_labels_from_last(raw_history[:, -1])
        return torch.nn.functional.one_hot(labels, num_classes=6).to(dtype=raw_history.dtype)

    def features(self, raw_history):
        raw = raw_history.to(dtype=self.obs_mean.dtype)
        mean = self.obs_mean.to(device=raw.device, dtype=raw.dtype)
        std = self.obs_std.to(device=raw.device, dtype=raw.dtype).clamp_min(1.0e-6)
        obs_norm = ((raw - mean) / std).reshape(raw.shape[0], -1)

        last = raw[:, -1]
        prev = raw[:, 0]
        tcp = last[:, 18:21]
        red = last[:, 25:28]
        blue = last[:, 32:35]
        red_goal = last[:, 41:44]
        blue_goal = last[:, 44:47]
        prev_tcp = prev[:, 18:21]
        prev_blue = prev[:, 32:35]
        fingers = last[:, 7:9]
        qvel_fingers = last[:, 16:18]

        pos_scale = torch.as_tensor([0.5, 0.6, 0.35], device=raw.device, dtype=raw.dtype)
        local_scale = torch.as_tensor([0.4, 0.4, 0.35], device=raw.device, dtype=raw.dtype)
        e_blue_goal = (blue_goal - blue) / pos_scale
        e_red_goal = (red_goal - red) / pos_scale
        e_tcp_blue = (tcp - blue) / local_scale
        e_tcp_red = (tcp - red) / local_scale
        e_blue_red = (blue - red) / local_scale
        d_tcp = (tcp - prev_tcp) / 0.15
        d_blue = (blue - prev_blue) / 0.15

        finger_left = (fingers[:, 0:1] - 0.0275) / 0.025
        finger_right = (fingers[:, 1:2] - 0.0275) / 0.025
        finger_avg = ((fingers.mean(dim=-1, keepdim=True)) - 0.0275) / 0.025
        finger_diff = (fingers[:, 0:1] - fingers[:, 1:2]) / 0.025
        finger_features = torch.cat([finger_left, finger_right, finger_avg, finger_diff], dim=-1)

        blue_goal_xy = torch.linalg.norm((blue_goal - blue)[:, :2], dim=-1, keepdim=True) / 0.5
        tcp_blue_xy = torch.linalg.norm((tcp - blue)[:, :2], dim=-1, keepdim=True) / 0.4
        red_goal_xy = torch.linalg.norm((red_goal - red)[:, :2], dim=-1, keepdim=True) / 0.5
        blue_speed_xy = torch.linalg.norm((blue - prev_blue)[:, :2], dim=-1, keepdim=True) / 0.1
        tcp_speed_xy = torch.linalg.norm((tcp - prev_tcp)[:, :2], dim=-1, keepdim=True) / 0.1
        closed_score = (0.032 - fingers.mean(dim=-1, keepdim=True)) / 0.02
        scalar_features = torch.cat([
            blue[:, 2:3] / 0.35,
            tcp[:, 2:3] / 0.35,
            red[:, 2:3] / 0.35,
            blue_goal_xy,
            tcp_blue_xy,
            red_goal_xy,
            blue_speed_xy,
            tcp_speed_xy,
            closed_score,
        ], dim=-1)

        goal_features = torch.cat([blue_goal / pos_scale, red_goal / pos_scale], dim=-1)
        qvel_finger_features = qvel_fingers / 0.2
        phase_one_hot = self.one_hot_phase(raw)

        geom = torch.cat([
            e_blue_goal,
            e_red_goal,
            e_tcp_blue,
            e_tcp_red,
            e_blue_red,
            d_tcp,
            d_blue,
            finger_features,
            scalar_features,
            goal_features,
            qvel_finger_features,
            phase_one_hot,
        ], dim=-1)
        return torch.cat([obs_norm, geom], dim=-1)

    def auxiliary(self, raw_history):
        feats = self.features(raw_history)
        latent = self.encoder(feats)
        stage_logits = self.stage_head(latent)
        waypoint = self.waypoint_head(latent)
        future_blue = self.future_blue_head(latent).reshape(raw_history.shape[0], 3, 3)
        grip = self.grip_head(latent)
        stage_probs = torch.nn.functional.softmax(stage_logits, dim=-1)
        cond = self.condition_head(torch.cat([
            latent,
            stage_probs,
            waypoint,
            future_blue.reshape(raw_history.shape[0], 9),
            grip,
        ], dim=-1))
        return {
            "condition": cond,
            "stage_logits": stage_logits,
            "stage_probs": stage_probs,
            "waypoint": waypoint,
            "future_blue": future_blue,
            "grip": grip,
        }

    def forward(self, noisy_action, timestep, raw_history):
        aux = self.auxiliary(raw_history)
        return self.backbone(noisy_action, timestep, aux["condition"])


def waypoint_targets(model, raw_history, labels):
    last = raw_history[:, -1]
    blue = last[:, 32:35]
    blue_goal = last[:, 44:47]
    safe_z = torch.full_like(blue[:, 2:3], model.safe_height)
    grasp_z = torch.full_like(blue[:, 2:3], 0.025)

    above_blue = torch.cat([blue[:, :2], safe_z], dim=-1)
    grasp_blue = torch.cat([blue[:, :2], grasp_z], dim=-1)
    over_goal = torch.cat([blue_goal[:, :2], safe_z], dim=-1)

    target = above_blue
    target = torch.where((labels == 1).unsqueeze(-1), grasp_blue, target)
    target = torch.where((labels == 4).unsqueeze(-1), over_goal, target)
    target = torch.where((labels == 5).unsqueeze(-1), over_goal, target)
    scale = torch.as_tensor([0.5, 0.6, 0.35], device=raw_history.device, dtype=raw_history.dtype)
    return target / scale


def build_model(spec):
    return HeightStagedBlueTransportPolicy(spec)


def compute_loss(model, batch, spec):
    predicted_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = epsilon_loss(predicted_noise, batch["noise"], batch["mask"])

    aux = model.auxiliary(batch["raw_obs"])
    labels = model.phase_labels_from_last(batch["raw_obs"][:, -1])
    stage_loss = torch.nn.functional.cross_entropy(aux["stage_logits"], labels)

    waypoint_target = waypoint_targets(model, batch["raw_obs"], labels)
    waypoint_loss = torch.nn.functional.mse_loss(aux["waypoint"], waypoint_target)

    slots = torch.as_tensor([3, 7, 15], device=batch["future_obs"].device, dtype=torch.long)
    current_blue = batch["raw_obs"][:, -1, 32:35]
    future_blue = batch["future_obs"].index_select(1, slots)[:, :, 32:35]
    future_mask = batch["future_mask"].index_select(1, slots).squeeze(-1)
    delta_scale = torch.as_tensor([0.4, 0.4, 0.35], device=future_blue.device, dtype=future_blue.dtype)
    future_target = (future_blue - current_blue.unsqueeze(1)) / delta_scale
    future_sq = (aux["future_blue"] - future_target).square().sum(dim=-1)
    future_loss = (future_sq * future_mask).sum() / (future_mask.sum() * 3.0).clamp_min(1.0)

    desired_grip = torch.where(labels < 2, torch.ones_like(aux["grip"].squeeze(-1)), -torch.ones_like(aux["grip"].squeeze(-1))).unsqueeze(-1)
    grip_loss = torch.nn.functional.mse_loss(aux["grip"], desired_grip)

    alpha_bar = batch["alpha_bar"].to(dtype=predicted_noise.dtype).reshape(-1, 1, 1).clamp(1.0e-4, 0.9999)
    clean_estimate = (batch["noisy_action"] - torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
    encoded_grip_target = desired_grip.reshape(-1, 1).unsqueeze(1)
    valid_phase = ((labels == 0) | (labels >= 3)).to(dtype=predicted_noise.dtype).reshape(-1, 1)
    action_mask = batch["mask"].squeeze(-1) * valid_phase
    action_weight = alpha_bar.squeeze(-1).squeeze(-1).reshape(-1, 1)
    grip_action_sq = (clean_estimate[:, :, 7] - encoded_grip_target.squeeze(1)).square()
    grip_action_loss = (grip_action_sq * action_mask * action_weight).sum() / (action_mask.sum()).clamp_min(1.0)

    prior_loss = (
        0.05 * stage_loss
        + 0.10 * waypoint_loss
        + 0.10 * future_loss
        + 0.02 * grip_loss
        + 0.02 * grip_action_loss
    )
    total_loss = diffusion_loss + prior_loss
    return {
        "loss": total_loss,
        "diffusion_loss": diffusion_loss,
        "prior_loss": prior_loss,
    }
