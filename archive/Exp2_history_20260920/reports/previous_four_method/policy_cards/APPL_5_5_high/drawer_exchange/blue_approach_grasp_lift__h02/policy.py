import torch
from appl.public import DiffusionBackbone, epsilon_loss


class WaypointConditionedDiffusion(torch.nn.Module):
    """Diffusion policy with a learned causal waypoint/stage conditioning prior."""

    def __init__(self, spec):
        super().__init__()
        normalizer = spec["normalizer"]
        self.register_buffer("obs_mean", torch.tensor(normalizer["mean"], dtype=torch.float32), persistent=False)
        self.register_buffer("obs_std", torch.tensor(normalizer["std"], dtype=torch.float32), persistent=False)
        self.obs_dim = int(spec["observation_dimension"])
        self.cond_dim = int(spec["candidate_config"]["condition_dimension"])
        self.geom_dim = 24
        token_dim = self.obs_dim + self.geom_dim
        self.stage_count = 4

        self.history_gru = torch.nn.GRU(token_dim, 128, num_layers=1, batch_first=True)
        self.flat_encoder = torch.nn.Sequential(
            torch.nn.Linear(token_dim * 2, 256),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 128),
            torch.nn.SiLU(),
        )
        self.fuse = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 256),
            torch.nn.SiLU(),
        )
        self.stage_head = torch.nn.Linear(256, self.stage_count)
        self.stage_embedding = torch.nn.Parameter(torch.randn(self.stage_count, 32) * 0.02)
        self.waypoint_head = torch.nn.Linear(256, 3)
        self.clearance_head = torch.nn.Linear(256, 2)
        self.condition_mlp = torch.nn.Sequential(
            torch.nn.Linear(293, self.cond_dim),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(self.cond_dim),
            torch.nn.Linear(self.cond_dim, self.cond_dim),
        )
        self.backbone = DiffusionBackbone(self.cond_dim, spec["training"])

    def geom_features(self, raw):
        tcp = raw[..., 18:21]
        red = raw[..., 25:28]
        blue = raw[..., 32:35]
        drawer = raw[..., 39:40]
        drawer_vel = raw[..., 40:41]
        red_goal = raw[..., 41:44]
        blue_goal = raw[..., 44:47]
        finger_width = raw[..., 7:8] + raw[..., 8:9]
        tcp_blue = (tcp - blue) / 0.30
        tcp_red = (tcp - red) / 0.30
        red_goal_err = (red - red_goal) / 0.30
        blue_goal_err = (blue - blue_goal) / 0.30
        tcp_goal_blue = (tcp - blue_goal) / 0.35
        z_feats = torch.cat([tcp[..., 2:3], red[..., 2:3], blue[..., 2:3]], dim=-1) / 0.30
        gripper = finger_width / 0.08
        drawer_feats = torch.cat([drawer / 0.30, drawer_vel / 0.30], dim=-1)
        dist_blue = torch.linalg.norm(tcp[..., 0:2] - blue[..., 0:2], dim=-1, keepdim=True) / 0.50
        dist_red = torch.linalg.norm(tcp[..., 0:2] - red[..., 0:2], dim=-1, keepdim=True) / 0.50
        height_over_blue = (tcp[..., 2:3] - blue[..., 2:3]) / 0.30
        return torch.cat([
            tcp_blue, tcp_red, red_goal_err, blue_goal_err, tcp_goal_blue,
            z_feats, gripper, drawer_feats, dist_blue, dist_red, height_over_blue
        ], dim=-1)

    def tokens(self, raw_history):
        mean = self.obs_mean.to(device=raw_history.device, dtype=raw_history.dtype)
        std = self.obs_std.to(device=raw_history.device, dtype=raw_history.dtype)
        norm = (raw_history - mean) / std
        geom = self.geom_features(raw_history)
        return torch.cat([norm, geom], dim=-1)

    def encode_condition(self, raw_history):
        tokens = self.tokens(raw_history)
        gru_out, h_last = self.history_gru(tokens)
        h_gru = h_last[-1]
        h_flat = self.flat_encoder(tokens.reshape(tokens.shape[0], -1))
        h = self.fuse(torch.cat([h_gru, h_flat], dim=-1))
        stage_logits = self.stage_head(h)
        stage_prob = torch.softmax(stage_logits, dim=-1)
        stage_context = stage_prob @ self.stage_embedding
        waypoint = self.waypoint_head(h)
        clearance = self.clearance_head(h)
        condition = self.condition_mlp(torch.cat([h, stage_context, waypoint, clearance], dim=-1))
        aux = {"stage_logits": stage_logits, "waypoint": waypoint, "clearance": clearance}
        return condition, aux

    def forward(self, noisy_action, timestep, raw_history):
        condition, aux = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, condition)

    def forward_with_aux(self, noisy_action, timestep, raw_history):
        condition, aux = self.encode_condition(raw_history)
        pred = self.backbone(noisy_action, timestep, condition)
        return pred, aux


def build_model(spec):
    return WaypointConditionedDiffusion(spec)


def stage_targets(raw_obs):
    current = raw_obs[:, -1]
    tcp = current[:, 18:21]
    red = current[:, 25:28]
    blue = current[:, 32:35]
    finger_width = current[:, 7] + current[:, 8]
    dist_blue_xy = torch.linalg.norm(tcp[:, 0:2] - blue[:, 0:2], dim=-1)
    dist_red_xy = torch.linalg.norm(tcp[:, 0:2] - red[:, 0:2], dim=-1)
    high = tcp[:, 2] > 0.235
    near_blue = dist_blue_xy < 0.070
    closed_or_lifted = (finger_width < 0.055) | (blue[:, 2] > 0.035)
    target = torch.zeros(raw_obs.shape[0], device=raw_obs.device, dtype=torch.long)
    target = torch.where((~near_blue) & high, torch.ones_like(target), target)
    target = torch.where((dist_red_xy < 0.10) & (~high) & (~near_blue), torch.zeros_like(target), target)
    target = torch.where(near_blue & (~closed_or_lifted), torch.full_like(target, 2), target)
    target = torch.where(closed_or_lifted, torch.full_like(target, 3), target)
    return target


def masked_terminal_future(future_obs, future_mask):
    mask = future_mask[..., 0]
    lengths = mask.sum(dim=1).long().clamp_min(1)
    gather_index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, future_obs.shape[-1])
    return future_obs.gather(1, gather_index).squeeze(1)


def smooth_l1_mean(pred, target):
    return torch.nn.functional.smooth_l1_loss(pred, target)


def compute_loss(model, batch, spec):
    pred_noise, aux = model.forward_with_aux(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = epsilon_loss(pred_noise, batch["noise"], batch["mask"])
    stage_target = stage_targets(batch["raw_obs"])
    stage_loss = torch.nn.functional.cross_entropy(aux["stage_logits"], stage_target)
    terminal = masked_terminal_future(batch["future_obs"], batch["future_mask"])
    terminal_tcp = terminal[:, 18:21]
    waypoint_target = torch.stack([terminal_tcp[:, 0] / 0.50, terminal_tcp[:, 1] / 0.50, terminal_tcp[:, 2] / 0.30], dim=-1)
    waypoint_loss = smooth_l1_mean(aux["waypoint"], waypoint_target)
    mask = batch["future_mask"][..., 0]
    tcp_z = batch["future_obs"][:, :, 20]
    masked_z = torch.where(mask > 0.5, tcp_z, torch.full_like(tcp_z, -1.0e6))
    max_z = masked_z.max(dim=1).values.clamp_min(0.0)
    terminal_z = terminal_tcp[:, 2]
    clearance_target = torch.stack([max_z / 0.30, terminal_z / 0.30], dim=-1)
    clearance_loss = smooth_l1_mean(aux["clearance"], clearance_target)
    cfg = spec["candidate_config"]
    prior_loss = float(cfg["stage_loss_weight"]) * stage_loss + float(cfg["waypoint_loss_weight"]) * waypoint_loss + float(cfg["clearance_loss_weight"]) * clearance_loss
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
