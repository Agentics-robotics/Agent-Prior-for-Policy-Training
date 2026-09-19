import torch
import torch.nn.functional as F
from appl.public import DiffusionBackbone, epsilon_loss


class PredicateTerminationDiffusionPolicy(torch.nn.Module):
    """Learned diffusion policy with explicit task-predicate representations.

    The model remains an epsilon-predicting action diffusion model. The prior is
    implemented by sharing a causal state encoder between the diffusion condition
    and auxiliary red_at_goal / blue_at_goal heads, plus a small action-conditioned
    future predicate head used only for training.
    """

    def __init__(self, spec):
        super().__init__()
        self.obs_dim = int(spec.get("observation_dimension", 47))
        self.horizon = int(spec["training"].get("horizon", 16))
        self.condition_dim = int(spec.get("candidate_config", {}).get("condition_dim", 256))
        geom_dim = 39
        encoder_in = 2 * self.obs_dim + geom_dim

        self.register_buffer("obs_mean", torch.tensor(spec["normalizer"]["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.tensor(spec["normalizer"]["std"], dtype=torch.float32))
        self.register_buffer("action_min", torch.tensor(spec["normalizer"]["action_min"], dtype=torch.float32))
        self.register_buffer("action_scale", torch.tensor(spec["normalizer"]["action_scale"], dtype=torch.float32))

        hidden = 256
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(encoder_in, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, self.condition_dim),
            torch.nn.LayerNorm(self.condition_dim),
            torch.nn.SiLU(),
        )
        self.backbone = DiffusionBackbone(self.condition_dim, spec["training"])

        self.current_predicate_head = torch.nn.Sequential(
            torch.nn.Linear(self.condition_dim, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 2),
        )
        self.action_embed = torch.nn.Sequential(
            torch.nn.Linear(8, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 64),
            torch.nn.SiLU(),
        )
        self.step_embed = torch.nn.Embedding(self.horizon, 16)
        self.future_predicate_head = torch.nn.Sequential(
            torch.nn.Linear(self.condition_dim + 64 + 16, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 2),
        )

    def normalize_obs(self, raw_history):
        return (raw_history - self.obs_mean.to(raw_history.dtype)) / self.obs_std.to(raw_history.dtype).clamp_min(1e-6)

    def denormalize_action(self, encoded_action):
        return (encoded_action + 1.0) * self.action_scale.to(encoded_action.dtype) / 2.0 + self.action_min.to(encoded_action.dtype)

    def geometric_features(self, raw_history):
        """Causal metric features in world frame using fixed physical scales."""
        cur = raw_history[:, -1]
        prev = raw_history[:, 0]
        qpos = cur[:, 0:9]
        qvel = cur[:, 9:18]
        tcp = cur[:, 18:25]
        red = cur[:, 25:32]
        blue = cur[:, 32:39]
        red_goal = cur[:, 41:44]
        blue_goal = cur[:, 44:47]

        tcp_pos = tcp[:, 0:3]
        red_pos = red[:, 0:3]
        blue_pos = blue[:, 0:3]
        prev_tcp_pos = prev[:, 18:21]
        prev_red_pos = prev[:, 25:28]
        prev_blue_pos = prev[:, 32:35]

        red_goal_delta = red_pos - red_goal
        blue_goal_delta = blue_pos - blue_goal
        tcp_red_delta = tcp_pos - red_pos
        tcp_red_goal_delta = tcp_pos - red_goal
        red_motion = red_pos - prev_red_pos
        blue_motion = blue_pos - prev_blue_pos
        tcp_motion = tcp_pos - prev_tcp_pos

        red_xy_abs = red_goal_delta[:, 0:2].abs()
        blue_xy_abs = blue_goal_delta[:, 0:2].abs()
        red_xy_dist = torch.linalg.vector_norm(red_goal_delta[:, 0:2], dim=-1, keepdim=True)
        blue_xy_dist = torch.linalg.vector_norm(blue_goal_delta[:, 0:2], dim=-1, keepdim=True)
        red_z_err = red_goal_delta[:, 2:3]
        blue_z_err = blue_goal_delta[:, 2:3]

        finger_width = qpos[:, 7:8] + qpos[:, 8:9]
        qvel_summary = torch.stack(
            [torch.linalg.vector_norm(qvel[:, 0:7], dim=-1), torch.linalg.vector_norm(qvel[:, 7:9], dim=-1)], dim=-1
        )

        red_margin_xy = 0.04 - red_xy_abs
        blue_margin_xy = 0.04 - blue_xy_abs
        red_z_margin = 0.011 - red_z_err.abs()
        blue_z_margin = 0.011 - blue_z_err.abs()

        feats = [
            red_goal_delta / 0.20,
            blue_goal_delta / 0.20,
            tcp_red_delta / 0.20,
            tcp_red_goal_delta / 0.30,
            red_motion / 0.05,
            blue_motion / 0.05,
            tcp_motion / 0.05,
            red_xy_abs / 0.06,
            blue_xy_abs / 0.06,
            red_xy_dist / 0.10,
            blue_xy_dist / 0.10,
            red_z_err / 0.15,
            blue_z_err / 0.15,
            (finger_width - 0.04) / 0.04,
            qvel_summary / 0.50,
            red_margin_xy / 0.06,
            blue_margin_xy / 0.06,
            red_z_margin / 0.02,
            blue_z_margin / 0.02,
            (tcp_pos[:, 2:3] - red_pos[:, 2:3]) / 0.30,
        ]
        return torch.cat(feats, dim=-1)

    def encode(self, raw_history):
        norm = self.normalize_obs(raw_history).reshape(raw_history.shape[0], -1)
        geom = self.geometric_features(raw_history).to(dtype=norm.dtype)
        return self.encoder(torch.cat([norm, geom], dim=-1))

    def forward(self, noisy_action, timestep, raw_history):
        cond = self.encode(raw_history)
        return self.backbone(noisy_action, timestep, cond)

    def current_predicate_logits(self, raw_history):
        return self.current_predicate_head(self.encode(raw_history))

    def future_predicate_logits(self, cond, encoded_action_estimate):
        b, h, dims = encoded_action_estimate.shape
        act = self.action_embed(encoded_action_estimate)
        steps = torch.arange(h, device=encoded_action_estimate.device).clamp_max(self.horizon - 1)
        step = self.step_embed(steps)[None, :, :].expand(b, h, -1)
        c = cond[:, None, :].expand(b, h, -1)
        return self.future_predicate_head(torch.cat([c, act, step], dim=-1))


def build_model(spec):
    return PredicateTerminationDiffusionPolicy(spec)


def predicate_labels_from_obs(obs):
    """Return red_at_goal and blue_at_goal labels from raw observations."""
    red = obs[..., 25:32]
    blue = obs[..., 32:39]
    red_goal = obs[..., 41:44]
    blue_goal = obs[..., 44:47]

    def one_object(pose, goal):
        pos = pose[..., 0:3]
        q = pose[..., 3:7]
        w, x, y, z = q.unbind(dim=-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        extent = 0.02 * (torch.cos(yaw).abs() + torch.sin(yaw).abs())
        dxy = (pos[..., 0:2] - goal[..., 0:2]).abs()
        xy_ok = (dxy[..., 0] + extent <= 0.06) & (dxy[..., 1] + extent <= 0.06)
        z_ok = (pos[..., 2] - goal[..., 2]).abs() <= 0.011
        return (xy_ok & z_ok).to(dtype=obs.dtype)

    return torch.stack([one_object(red, red_goal), one_object(blue, blue_goal)], dim=-1)


def masked_mean(value, mask, denom_last_dim=1):
    return (value * mask).sum() / (mask.sum() * denom_last_dim).clamp_min(1.0)


def compute_loss(model, batch, spec):
    raw_obs = batch["raw_obs"]
    noisy_action = batch["noisy_action"]
    timesteps = batch["timesteps"]
    noise = batch["noise"]
    mask = batch["mask"]

    cond = model.encode(raw_obs)
    pred_noise = model.backbone(noisy_action, timesteps, cond)
    diffusion_loss = epsilon_loss(pred_noise, noise, mask)

    current_logits = model.current_predicate_head(cond)
    current_labels = predicate_labels_from_obs(raw_obs[:, -1]).to(current_logits.dtype)
    current_bce = F.binary_cross_entropy_with_logits(current_logits, current_labels)

    alpha_bar = batch["alpha_bar"].to(noisy_action.dtype).reshape(-1, 1, 1).clamp(1e-5, 1.0)
    x0 = (noisy_action - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
    x0_aux = x0.clamp(-1.5, 1.5)

    future_logits = model.future_predicate_logits(cond, x0_aux)
    future_labels = predicate_labels_from_obs(batch["future_obs"]).to(future_logits.dtype)
    future_mask = batch["future_mask"].to(future_logits.dtype)
    noise_quality = torch.sqrt(alpha_bar).detach().clamp(0.05, 1.0)
    future_bce_raw = F.binary_cross_entropy_with_logits(future_logits, future_labels, reduction="none")
    future_bce = (future_bce_raw * future_mask * noise_quality).sum() / (future_mask.sum() * 2.0).clamp_min(1.0)

    native_x0 = model.denormalize_action(x0_aux)
    success = (future_labels[..., 0:1] * future_labels[..., 1:2]) * future_mask
    open_err = (native_x0[..., 7:8] - 1.0).square()
    open_loss = masked_mean(open_err, success * noise_quality, 1)

    fut = batch["future_obs"]
    fingers_open = ((fut[..., 7:8] + fut[..., 8:9]) > 0.07).to(fut.dtype)
    tcp_high = (fut[..., 20:21] > 0.25).to(fut.dtype)
    qvel_low = (torch.linalg.vector_norm(fut[..., 9:16], dim=-1, keepdim=True) < 0.03).to(fut.dtype)
    settled = success * fingers_open * tcp_high * qvel_low
    joint_scale = model.action_scale[:7].to(native_x0.dtype).reshape(1, 1, 7).clamp_min(0.05)
    hold_err = ((native_x0[..., 0:7] - fut[..., 0:7]) / joint_scale).square().mean(dim=-1, keepdim=True)
    hold_loss = masked_mean(hold_err, settled * noise_quality, 1)

    prior_loss = 0.10 * current_bce + 0.05 * future_bce + 0.02 * open_loss + 0.01 * hold_loss
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
