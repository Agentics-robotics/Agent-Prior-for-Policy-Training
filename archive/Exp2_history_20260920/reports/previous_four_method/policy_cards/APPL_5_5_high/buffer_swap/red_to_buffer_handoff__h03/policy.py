import torch
import appl.public as public


class OverlapReadinessDiffusionPolicy(torch.nn.Module):
    """State-conditioned DDPM epsilon model with an auxiliary handoff-readiness head."""

    def __init__(self, spec):
        super().__init__()
        normalizer = spec.get("normalizer", spec.get("shared_normalizer"))
        self.register_buffer("obs_mean", torch.as_tensor(normalizer["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.as_tensor(normalizer["std"], dtype=torch.float32))
        # Demonstrated red temporary buffer, estimated from the assigned slice evidence.
        # It is used only as a normalized geometric feature and for soft auxiliary labels,
        # not as a controller target.
        self.register_buffer("buffer_xyz", torch.tensor([-0.181, -0.001, 0.020], dtype=torch.float32))
        self.feature_dim = 85
        self.step_hidden = 128
        self.condition_dim = 256

        self.step_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, 192),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(192),
            torch.nn.Linear(192, self.step_hidden),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(self.step_hidden),
        )
        self.readiness_head = torch.nn.Sequential(
            torch.nn.Linear(self.step_hidden, 96),
            torch.nn.SiLU(),
            torch.nn.Linear(96, 1),
        )
        cond_in = 2 * self.step_hidden + 3 * self.feature_dim + 4
        self.condition_encoder = torch.nn.Sequential(
            torch.nn.Linear(cond_in, 384),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(384),
            torch.nn.Linear(384, self.condition_dim),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(self.condition_dim),
        )
        self.backbone = public.DiffusionBackbone(self.condition_dim, spec["training"])

    def normalize_obs(self, raw):
        return (raw - self.obs_mean.to(device=raw.device, dtype=raw.dtype)) / self.obs_std.to(device=raw.device, dtype=raw.dtype)

    def features(self, raw_history):
        """Return per-observation causal features [B, 2, 85]."""
        raw = raw_history
        norm = self.normalize_obs(raw)
        qpos = raw[..., 0:9]
        tcp = raw[..., 18:21]
        red = raw[..., 25:28]
        blue = raw[..., 32:35]
        red_goal = raw[..., 41:44]
        blue_goal = raw[..., 44:47]
        buffer_xyz = self.buffer_xyz.to(device=raw.device, dtype=raw.dtype).view(1, 1, 3)

        pos_scale = 0.50
        z_scale = 0.30
        rels = [
            (tcp - red) / pos_scale,
            (tcp - blue) / pos_scale,
            (tcp - buffer_xyz) / pos_scale,
            (red - buffer_xyz) / pos_scale,
            (blue - blue_goal) / pos_scale,
            (red - red_goal) / pos_scale,
            (blue - red_goal) / pos_scale,
            (red - blue) / pos_scale,
        ]

        finger_width = qpos[..., 7:8] + qpos[..., 8:9]
        finger_balance = qpos[..., 7:8] - qpos[..., 8:9]
        red_buffer = red - buffer_xyz
        scalars = [
            torch.linalg.norm((tcp - blue)[..., 0:2], dim=-1, keepdim=True) / pos_scale,
            torch.linalg.norm((tcp - red)[..., 0:2], dim=-1, keepdim=True) / pos_scale,
            torch.linalg.norm(red_buffer[..., 0:2], dim=-1, keepdim=True) / pos_scale,
            torch.linalg.norm((blue - blue_goal)[..., 0:2], dim=-1, keepdim=True) / pos_scale,
            torch.linalg.norm((red - red_goal)[..., 0:2], dim=-1, keepdim=True) / pos_scale,
            red[..., 2:3] / z_scale,
            blue[..., 2:3] / z_scale,
            tcp[..., 2:3] / 0.50,
            finger_width / 0.08,
            finger_balance / 0.04,
            (finger_width - 0.04) / 0.04,
        ]
        goal_span = (red_goal - blue_goal) / pos_scale
        feat = torch.cat([norm] + rels + scalars + [goal_span], dim=-1)
        return feat

    def encode_condition(self, raw_history):
        feat = self.features(raw_history)
        b, t, f = feat.shape
        step_h = self.step_encoder(feat.reshape(b * t, f)).reshape(b, t, self.step_hidden)
        logits = self.readiness_head(step_h).squeeze(-1)
        probs = torch.sigmoid(logits)
        flat_h = step_h.reshape(b, t * self.step_hidden)
        flat_feat = feat.reshape(b, t * f)
        delta_feat = feat[:, -1, :] - feat[:, 0, :]
        cond_input = torch.cat([flat_h, flat_feat, delta_feat, logits, probs], dim=-1)
        cond = self.condition_encoder(cond_input)
        return cond, logits

    def forward(self, noisy_action, timestep, raw_history):
        condition, extra_logits = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, condition)

    def readiness_logits(self, raw_history):
        condition, logits = self.encode_condition(raw_history)
        return logits

    def handoff_readiness(self, raw_history):
        """Causal readiness probability for the latest observation in the two-step history."""
        return torch.sigmoid(self.readiness_logits(raw_history)[:, -1])

    def readiness_targets(self, raw_history):
        """Soft labels inferred from observed slice geometry; no future state is used at inference."""
        raw = raw_history
        qpos = raw[..., 0:9]
        tcp = raw[..., 18:21]
        red = raw[..., 25:28]
        blue = raw[..., 32:35]
        buffer_xyz = self.buffer_xyz.to(device=raw.device, dtype=raw.dtype).view(1, 1, 3)
        finger_width = qpos[..., 7] + qpos[..., 8]

        d_red_buffer = torch.linalg.norm(red[..., 0:2] - buffer_xyz[..., 0:2], dim=-1)
        d_tcp_blue = torch.linalg.norm(tcp[..., 0:2] - blue[..., 0:2], dim=-1)
        red_table = torch.sigmoid((0.040 - red[..., 2]) / 0.008)
        red_buffered = torch.sigmoid((0.085 - d_red_buffer) / 0.020) * red_table

        open_gripper = torch.sigmoid((finger_width - 0.065) / 0.008)
        high_retreat = torch.sigmoid((tcp[..., 2] - 0.220) / 0.035)
        approach_xy = torch.sigmoid((0.180 - d_tcp_blue) / 0.060)
        high_corridor = 0.35 * high_retreat + 0.65 * approach_xy

        low_contact = torch.sigmoid((0.060 - d_tcp_blue) / 0.020) * torch.sigmoid((0.075 - tcp[..., 2]) / 0.020)
        corridor_ready = torch.maximum(open_gripper * high_corridor, low_contact)
        target = torch.clamp(red_buffered * corridor_ready, 0.0, 1.0)
        return target


def build_model(spec):
    return OverlapReadinessDiffusionPolicy(spec)


def compute_loss(model, batch, spec):
    predicted_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = public.epsilon_loss(predicted_noise, batch["noise"], batch["mask"])

    logits = model.readiness_logits(batch["raw_obs"])
    targets = model.readiness_targets(batch["raw_obs"]).detach()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    pred_delta = torch.sigmoid(logits[:, -1]) - torch.sigmoid(logits[:, 0])
    target_delta = torch.clamp(targets[:, -1] - targets[:, 0], -0.25, 0.25)
    smooth = torch.nn.functional.mse_loss(pred_delta, target_delta)

    cfg = spec.get("candidate_config", {}).get("config", {})
    readiness_weight = float(cfg.get("readiness_loss_weight", 0.05))
    smooth_weight = float(cfg.get("readiness_smoothness_weight", 0.01))
    prior_loss = readiness_weight * bce + smooth_weight * smooth
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
