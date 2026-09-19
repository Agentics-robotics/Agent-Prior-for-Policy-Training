import math
import torch
import appl.public as public


class ProgressCoordinatePolicy(torch.nn.Module):
    """Progress-conditioned learned DDPM policy for side pulling the drawer open."""

    def __init__(self, spec):
        super().__init__()
        normalizer = spec["normalizer"]
        self.register_buffer("obs_mean", torch.tensor(normalizer["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.tensor(normalizer["std"], dtype=torch.float32))
        self.register_buffer("drawer_origin", torch.tensor([0.19, 0.0, 0.035], dtype=torch.float32))
        self.register_buffer("coord_scale", torch.tensor(0.30, dtype=torch.float32))
        self.register_buffer("lateral_scale", torch.tensor(0.15, dtype=torch.float32))

        self.feature_dim = 168
        cfg = spec["candidate_config"] if "candidate_config" in spec else {}
        self.cond_dim = int(cfg["condition_dim"]) if "condition_dim" in cfg else 256
        if self.cond_dim < 1 or self.cond_dim > 512:
            self.cond_dim = 256

        self.feature_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, 256),
            torch.nn.Mish(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, self.cond_dim),
            torch.nn.Mish(),
            torch.nn.LayerNorm(self.cond_dim),
        )
        self.backbone = public.DiffusionBackbone(self.cond_dim, spec["training"])

        self.action_effect = torch.nn.Sequential(
            torch.nn.Conv1d(8, 64, kernel_size=3, padding=1),
            torch.nn.Mish(),
            torch.nn.Conv1d(64, 64, kernel_size=3, padding=1),
            torch.nn.Mish(),
        )
        self.effect_cond = torch.nn.Sequential(
            torch.nn.Linear(self.cond_dim, 96),
            torch.nn.Mish(),
        )
        self.effect_head = torch.nn.Sequential(
            torch.nn.Linear(64 + 96 + 4, 128),
            torch.nn.Mish(),
            torch.nn.Linear(128, 5),
        )
        self.phase_head = torch.nn.Sequential(
            torch.nn.Linear(self.cond_dim, 64),
            torch.nn.Mish(),
            torch.nn.Linear(64, 3),
        )

    def normalize_obs_internal(self, raw_history):
        return (raw_history - self.obs_mean.to(raw_history.device, raw_history.dtype)) / self.obs_std.to(raw_history.device, raw_history.dtype)

    def make_features(self, raw_history):
        raw = raw_history
        dtype = raw.dtype
        device = raw.device
        norm = self.normalize_obs_internal(raw)
        last = raw[:, -1, :]
        prev = raw[:, 0, :]
        delta_norm = (last - prev) / self.obs_std.to(device, dtype)

        tcp = last[:, 18:21]
        red = last[:, 25:28]
        blue = last[:, 32:35]
        drawer_p = last[:, 39:40]
        drawer_v = last[:, 40:41]
        red_goal = last[:, 41:44]
        blue_goal = last[:, 44:47]
        prev_drawer_p = prev[:, 39:40]
        prev_drawer_v = prev[:, 40:41]

        drawer_center = self.drawer_origin.to(device, dtype).view(1, 3).repeat(last.shape[0], 1)
        drawer_center[:, 0:1] = drawer_center[:, 0:1] - drawer_p
        coord_scale = self.coord_scale.to(device, dtype)
        lateral_scale = self.lateral_scale.to(device, dtype)

        rels = torch.cat([
            (tcp - red) / coord_scale,
            (tcp - drawer_center) / coord_scale,
            (red - drawer_center) / coord_scale,
            (red - red_goal) / coord_scale,
            (blue - blue_goal) / coord_scale,
        ], dim=-1)

        c = drawer_p / coord_scale
        c_prev = prev_drawer_p / coord_scale
        progress_terms = torch.cat([
            c,
            c_prev,
            c - c_prev,
            drawer_v / coord_scale,
            prev_drawer_v / coord_scale,
            (torch.as_tensor(0.26, device=device, dtype=dtype) - drawer_p) / coord_scale,
            (coord_scale - drawer_p) / coord_scale,
            (drawer_p - torch.as_tensor(0.26, device=device, dtype=dtype)) / torch.as_tensor(0.04, device=device, dtype=dtype),
        ], dim=-1)

        fingers = torch.cat([
            last[:, 7:9] / torch.as_tensor(0.04, device=device, dtype=dtype),
            last[:, 16:18] / lateral_scale,
        ], dim=-1)

        return torch.cat([
            norm.reshape(norm.shape[0], -1),
            delta_norm,
            rels,
            progress_terms,
            fingers,
        ], dim=-1)

    def encode_condition(self, raw_history):
        return self.feature_encoder(self.make_features(raw_history))

    def forward(self, noisy_action, timestep, raw_history):
        cond = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, cond)

    def predict_effects(self, raw_history, encoded_action):
        cond = self.encode_condition(raw_history)
        action_feat = self.action_effect(encoded_action.transpose(1, 2)).transpose(1, 2)
        cond_feat = self.effect_cond(cond).unsqueeze(1).expand(-1, encoded_action.shape[1], -1)

        h = encoded_action.shape[1]
        step = torch.linspace(0.0, 1.0, h, device=encoded_action.device, dtype=encoded_action.dtype).view(1, h, 1)
        step_feat = torch.cat([
            step.expand(encoded_action.shape[0], -1, -1),
            (step * step).expand(encoded_action.shape[0], -1, -1),
            torch.sin(math.pi * step).expand(encoded_action.shape[0], -1, -1),
            torch.cos(math.pi * step).expand(encoded_action.shape[0], -1, -1),
        ], dim=-1)

        raw_out = self.effect_head(torch.cat([action_feat, cond_feat, step_feat], dim=-1))
        progress = 1.05 * torch.sigmoid(raw_out[..., 0]) - 0.025
        velocity = torch.tanh(raw_out[..., 1])
        open_logit = raw_out[..., 2]
        red_yz = torch.tanh(raw_out[..., 3:5])
        phase = self.phase_head(cond)
        return {
            "progress": progress,
            "velocity": velocity,
            "open_logit": open_logit,
            "red_yz": red_yz,
            "phase": phase,
        }


def build_model(spec):
    return ProgressCoordinatePolicy(spec)


def compute_loss(model, batch, spec):
    pred_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = public.epsilon_loss(pred_noise, batch["noise"], batch["mask"])

    alpha = batch["alpha_bar"].to(pred_noise.device, pred_noise.dtype).view(-1, 1, 1)
    x0 = (batch["noisy_action"] - torch.sqrt((1.0 - alpha).clamp_min(0.0)) * pred_noise) / torch.sqrt(alpha.clamp_min(1.0e-6))
    x0_for_effect = x0.clamp(-1.5, 1.5)
    effects = model.predict_effects(batch["raw_obs"], x0_for_effect)

    future = batch["future_obs"]
    mask = batch["mask"] * batch["future_mask"]
    mask1 = mask.squeeze(-1)
    snr_weight = alpha.view(-1, 1).detach().clamp(0.05, 1.0)
    w1 = mask1 * snr_weight
    w = w1.unsqueeze(-1)

    progress_label = (future[:, :, 39] / 0.30).clamp(-0.05, 1.10)
    progress_loss = ((effects["progress"] - progress_label).square() * w1).sum() / w1.sum().clamp_min(1.0)

    velocity_label = (future[:, :, 40] / 0.20).clamp(-1.0, 1.0)
    velocity_loss = ((effects["velocity"] - velocity_label).square() * w1).sum() / w1.sum().clamp_min(1.0)

    open_label = (future[:, :, 39] > 0.26).to(pred_noise.dtype)
    open_bce = torch.nn.functional.binary_cross_entropy_with_logits(effects["open_logit"], open_label, reduction="none")
    open_loss = (open_bce * w1).sum() / w1.sum().clamp_min(1.0)

    current_red_yz = batch["raw_obs"][:, -1, 26:28].unsqueeze(1)
    red_yz_label = ((future[:, :, 26:28] - current_red_yz) / 0.15).clamp(-1.0, 1.0)
    lateral_loss = ((effects["red_yz"] - red_yz_label).square() * w).sum() / (w.sum() * 2.0).clamp_min(1.0)

    if effects["progress"].shape[1] > 1:
        dp = effects["progress"][:, 1:] - effects["progress"][:, :-1]
        mono_mask = (w1[:, 1:] * w1[:, :-1]).sqrt()
        monotonic_loss = (torch.relu(-dp).square() * mono_mask).sum() / mono_mask.sum().clamp_min(1.0)
    else:
        monotonic_loss = effects["progress"].sum() * 0.0

    current_p = batch["raw_obs"][:, -1, 39]
    current_finger = batch["raw_obs"][:, -1, 7:9].mean(dim=-1)
    phase_label = torch.zeros_like(current_p, dtype=torch.long)
    phase_label = torch.where((current_p > 0.03) & (current_p <= 0.26) & (current_finger < 0.02), torch.ones_like(phase_label), phase_label)
    phase_label = torch.where(current_p > 0.26, torch.full_like(phase_label, 2), phase_label)
    phase_loss = torch.nn.functional.cross_entropy(effects["phase"], phase_label)

    prior_loss = (
        0.20 * progress_loss
        + 0.05 * velocity_loss
        + 0.05 * open_loss
        + 0.03 * lateral_loss
        + 0.02 * monotonic_loss
        + 0.02 * phase_loss
    )
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
