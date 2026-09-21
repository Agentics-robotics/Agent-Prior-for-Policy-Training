import torch
import appl.public as public


class WaypointChainDiffusionPolicy(torch.nn.Module):
    """Learned diffusion policy with a trainable waypoint/progress bottleneck."""

    def __init__(self, spec):
        super().__init__()
        normalizer = spec["normalizer"]
        self.register_buffer("obs_mean", torch.tensor(normalizer["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.tensor(normalizer["std"], dtype=torch.float32))
        self.register_buffer("action_min", torch.tensor(normalizer["action_min"], dtype=torch.float32))
        self.register_buffer("action_scale", torch.tensor(normalizer["action_scale"], dtype=torch.float32))
        self.register_buffer("pos_scale", torch.tensor([0.30, 0.30, 0.30], dtype=torch.float32))
        self.register_buffer("retreat_offset", torch.tensor([-0.008, -0.005, 0.0], dtype=torch.float32))

        self.history_steps = 2
        self.obs_dim = 47
        self.feature_dim = 30
        self.waypoint_count = 5
        self.condition_dim = 256

        in_dim = self.history_steps * self.obs_dim + self.feature_dim
        self.obs_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 256),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 256),
            torch.nn.SiLU(),
        )
        self.aux_head = torch.nn.Linear(256, 1 + 5 + self.waypoint_count * 3)
        self.condition_proj = torch.nn.Sequential(
            torch.nn.Linear(256 + 1 + 5 + self.waypoint_count * 3, 256),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, self.condition_dim),
        )
        self.backbone = public.DiffusionBackbone(self.condition_dim, spec["training"])

    def normalize_obs(self, raw_history):
        return (raw_history - self.obs_mean.to(raw_history.dtype)) / self.obs_std.to(raw_history.dtype)

    def state_features(self, raw_history):
        last = raw_history[:, -1, :]
        prev = raw_history[:, 0, :]
        scale = self.pos_scale.to(device=raw_history.device, dtype=raw_history.dtype)

        tcp = last[:, 18:21]
        red = last[:, 25:28]
        blue = last[:, 32:35]
        red_goal = last[:, 41:44]
        blue_goal = last[:, 44:47]
        tcp_prev = prev[:, 18:21]
        red_prev = prev[:, 25:28]
        qpos = last[:, 0:9]
        qpos_prev = prev[:, 0:9]

        red_to_goal = (red_goal - red) / scale
        tcp_to_red = (red - tcp) / scale
        tcp_to_goal = (red_goal - tcp) / scale
        blue_to_goal = (blue_goal - blue) / scale
        red_delta = (red - red_prev) / scale
        tcp_delta = (tcp - tcp_prev) / scale

        red_goal_xy = (red_goal[:, :2] - red[:, :2]).norm(dim=-1, keepdim=True) / 0.30
        tcp_red_xy = (red[:, :2] - tcp[:, :2]).norm(dim=-1, keepdim=True) / 0.30
        tcp_goal_xy = (red_goal[:, :2] - tcp[:, :2]).norm(dim=-1, keepdim=True) / 0.30
        blue_goal_xy = (blue_goal[:, :2] - blue[:, :2]).norm(dim=-1, keepdim=True) / 0.30
        table_z = red_goal[:, 2:3]
        red_z_rel = (red[:, 2:3] - table_z) / 0.30
        tcp_z_rel = (tcp[:, 2:3] - table_z) / 0.30
        finger_width = (qpos[:, 7:8] + qpos[:, 8:9]) / 0.08
        finger_delta = ((qpos[:, 7:8] + qpos[:, 8:9]) - (qpos_prev[:, 7:8] + qpos_prev[:, 8:9])) / 0.08

        red_lifted = (red[:, 2:3] > table_z + 0.055).to(raw_history.dtype)
        at_goal = (red_goal_xy < (0.05 / 0.30)).to(raw_history.dtype)
        gripper_open = ((qpos[:, 7:8] + qpos[:, 8:9]) > 0.065).to(raw_history.dtype)
        gripper_closed = ((qpos[:, 7:8] + qpos[:, 8:9]) < 0.055).to(raw_history.dtype)

        return torch.cat([
            red_to_goal, tcp_to_red, tcp_to_goal, blue_to_goal, red_delta, tcp_delta,
            red_goal_xy, tcp_red_xy, tcp_goal_xy, blue_goal_xy,
            red_z_rel, tcp_z_rel, finger_width, finger_delta,
            red_lifted, at_goal, gripper_open, gripper_closed,
        ], dim=-1)

    def base_waypoints(self, raw_history):
        last = raw_history[:, -1, :]
        tcp = last[:, 18:21]
        red = last[:, 25:28]
        red_goal = last[:, 41:44]
        table_z = red_goal[:, 2:3]
        safe_z = table_z + 0.280
        lift_z = torch.minimum(torch.maximum(red[:, 2:3], table_z + 0.080), safe_z)

        w_lift = torch.cat([red[:, 0:2], lift_z], dim=-1)
        mid_xy = 0.5 * (red[:, 0:2] + red_goal[:, 0:2])
        w_mid = torch.cat([mid_xy, safe_z], dim=-1)
        w_above_goal = torch.cat([red_goal[:, 0:2], safe_z], dim=-1)
        w_release = torch.cat([red_goal[:, 0:2], table_z + 0.006], dim=-1)
        w_retreat = torch.cat([red_goal[:, 0:2], safe_z], dim=-1) + self.retreat_offset.to(device=raw_history.device, dtype=raw_history.dtype)
        waypoints = torch.stack([w_lift, w_mid, w_above_goal, w_release, w_retreat], dim=1)
        rel = (waypoints - tcp[:, None, :]) / self.pos_scale.to(device=raw_history.device, dtype=raw_history.dtype)[None, None, :]
        return rel.reshape(raw_history.shape[0], self.waypoint_count * 3)

    def encode_condition(self, raw_history):
        norm = self.normalize_obs(raw_history).reshape(raw_history.shape[0], -1)
        features = self.state_features(raw_history)
        hidden = self.obs_mlp(torch.cat([norm, features], dim=-1))
        aux_raw = self.aux_head(hidden)
        progress = torch.sigmoid(aux_raw[:, 0:1])
        phase_logits = aux_raw[:, 1:6]
        waypoint_residual = 0.25 * torch.tanh(aux_raw[:, 6:])
        waypoint_prediction = self.base_waypoints(raw_history) + waypoint_residual
        phase_prob = torch.softmax(phase_logits, dim=-1)
        condition = self.condition_proj(torch.cat([hidden, progress, phase_prob, waypoint_prediction], dim=-1))
        return condition, {
            "progress": progress,
            "phase_logits": phase_logits,
            "waypoints": waypoint_prediction,
            "residual": waypoint_residual,
        }

    def forward(self, noisy_action, timestep, raw_history):
        condition, aux = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, condition)

    def phase_targets(self, raw_history):
        last = raw_history[:, -1, :]
        qpos = last[:, 0:9]
        red = last[:, 25:28]
        red_goal = last[:, 41:44]
        table_z = red_goal[:, 2]
        finger_width = qpos[:, 7] + qpos[:, 8]
        red_goal_xy = (red_goal[:, :2] - red[:, :2]).norm(dim=-1)
        red_lifted = red[:, 2] > table_z + 0.055
        red_low = red[:, 2] < table_z + 0.040
        near_goal = red_goal_xy < 0.050
        gripper_open = finger_width > 0.065
        gripper_closed = finger_width < 0.055

        phase = torch.zeros(raw_history.shape[0], device=raw_history.device, dtype=torch.long)
        phase = torch.where(gripper_closed & (~red_lifted), torch.ones_like(phase), phase)
        phase = torch.where(red_lifted & (~near_goal), torch.full_like(phase, 2), phase)
        phase = torch.where(near_goal & (red_lifted | (~red_low) | gripper_closed), torch.full_like(phase, 3), phase)
        phase = torch.where(near_goal & red_low & gripper_open, torch.full_like(phase, 4), phase)
        progress = phase.to(raw_history.dtype).unsqueeze(-1) / 4.0
        return phase, progress


def build_model(spec):
    return WaypointChainDiffusionPolicy(spec)


def x0_reconstruction_loss(predicted_noise, batch):
    alpha_bar = batch["alpha_bar"].reshape(-1, 1, 1).to(predicted_noise.dtype).clamp_min(1.0e-4)
    noisy = batch["noisy_action"]
    clean_target = batch["encoded_action"]
    x0 = (noisy - torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
    x0 = torch.clamp(x0, -1.5, 1.5)
    mask = batch["mask"]
    weighted = (x0 - clean_target).square() * mask * alpha_bar
    return weighted.sum() / (mask.sum() * predicted_noise.shape[-1]).clamp_min(1.0)


def compute_loss(model, batch, spec):
    predicted_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = public.epsilon_loss(predicted_noise, batch["noise"], batch["mask"])

    condition, aux = model.encode_condition(batch["raw_obs"])
    phase_target, progress_target = model.phase_targets(batch["raw_obs"])
    waypoint_target = model.base_waypoints(batch["raw_obs"])

    phase_loss = torch.nn.functional.cross_entropy(aux["phase_logits"], phase_target)
    progress_loss = (aux["progress"] - progress_target).square().mean()
    waypoint_loss = (aux["waypoints"] - waypoint_target).square().mean()
    clean_action_loss = x0_reconstruction_loss(predicted_noise, batch)

    prior_loss = 0.02 * phase_loss + 0.05 * progress_loss + 0.10 * waypoint_loss + 0.02 * clean_action_loss
    loss = diffusion_loss + prior_loss
    return {
        "loss": loss,
        "diffusion_loss": diffusion_loss,
        "prior_loss": prior_loss,
    }
