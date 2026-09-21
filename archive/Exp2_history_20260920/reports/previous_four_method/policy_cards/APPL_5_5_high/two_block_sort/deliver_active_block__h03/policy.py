import torch
from appl.public import DiffusionBackbone, normalize_observation, epsilon_loss


class PredicateReadinessDiffusionModel(torch.nn.Module):
    """Action diffusion model conditioned on learned predicate/readiness forecasts."""

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        cfg = spec.get("candidate_config", {})
        self.aux_weight = float(cfg.get("auxiliary_bce_weight", 0.20))
        self.terminal_action_weight = float(cfg.get("terminal_action_weight", 0.05))
        hidden = int(cfg.get("encoder_hidden", 256))
        cond_dim = int(cfg.get("condition_dimension", 256))

        feature_dim = 2 * int(spec.get("observation_dimension", 47)) + 48
        self.obs_encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden),
        )
        self.aux_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden // 2, 12),
        )
        self.condition_head = torch.nn.Sequential(
            torch.nn.Linear(hidden + 12, cond_dim),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(cond_dim),
            torch.nn.Linear(cond_dim, cond_dim),
            torch.nn.SiLU(),
        )
        self.backbone = DiffusionBackbone(cond_dim, spec["training"])

    def engineered_features(self, raw_history):
        # Features are scaled by task/workspace-scale constants, not by a per-skill
        # fit. They only describe causal geometry available in raw_history.
        pos_scale = 0.25
        z_scale = 0.30
        finger_scale = 0.04
        step_feats = []
        for k in range(raw_history.shape[1]):
            obs = raw_history[:, k]
            tcp = obs[:, 18:21]
            red = obs[:, 25:28]
            blue = obs[:, 32:35]
            red_goal = obs[:, 41:44]
            blue_goal = obs[:, 44:47]
            fingers = obs[:, 7:9]
            step_feats.append(torch.cat([
                (red - red_goal) / pos_scale,
                (blue - blue_goal) / pos_scale,
                (tcp - red) / pos_scale,
                (tcp - blue) / pos_scale,
                torch.cat([tcp[:, 2:3], red[:, 2:3], blue[:, 2:3]], dim=-1) / z_scale,
                fingers / finger_scale - 1.0,
            ], dim=-1))

        prev = raw_history[:, 0]
        cur = raw_history[:, -1]
        red_step = (cur[:, 25:28] - prev[:, 25:28]) / pos_scale
        blue_step = (cur[:, 32:35] - prev[:, 32:35]) / pos_scale
        tcp_step = (cur[:, 18:21] - prev[:, 18:21]) / pos_scale
        finger_step = (cur[:, 7:9] - prev[:, 7:9]) / finger_scale
        red_linf = torch.amax(torch.abs(cur[:, 25:27] - cur[:, 41:43]) / 0.04, dim=-1, keepdim=True)
        blue_linf = torch.amax(torch.abs(cur[:, 32:34] - cur[:, 44:46]) / 0.04, dim=-1, keepdim=True)
        finger_width = (cur[:, 7:8] + cur[:, 8:9]) / 0.08
        return torch.cat(step_feats + [red_step, blue_step, tcp_step, finger_step, red_linf, blue_linf, finger_width], dim=-1)

    def feature_vector(self, raw_history):
        norm = normalize_observation(raw_history, self.spec).reshape(raw_history.shape[0], -1)
        rel = self.engineered_features(raw_history)
        return torch.cat([norm, rel], dim=-1)

    def condition_and_logits(self, raw_history):
        encoded_obs = self.obs_encoder(self.feature_vector(raw_history))
        logits = self.aux_head(encoded_obs)
        # The denoiser is conditioned on the model's own learned causal forecasts,
        # not on future labels. Gradients from diffusion and BCE both reach aux_head.
        probs = torch.sigmoid(logits)
        condition = self.condition_head(torch.cat([encoded_obs, probs], dim=-1))
        return condition, logits

    def forward(self, noisy_action, timestep, raw_history):
        condition, aux_logits = self.condition_and_logits(raw_history)
        return self.backbone(noisy_action, timestep, condition)


def build_model(spec):
    return PredicateReadinessDiffusionModel(spec)


def predicate_labels(obs):
    """Return [...,4] labels: red_at, blue_at, red_ready, blue_ready."""
    red = obs[..., 25:28]
    blue = obs[..., 32:35]
    tcp = obs[..., 18:21]
    red_goal = obs[..., 41:44]
    blue_goal = obs[..., 44:47]
    qpos = obs[..., 0:9]

    red_xy = (torch.abs(red[..., 0] - red_goal[..., 0]) <= 0.04) & (torch.abs(red[..., 1] - red_goal[..., 1]) <= 0.04)
    blue_xy = (torch.abs(blue[..., 0] - blue_goal[..., 0]) <= 0.04) & (torch.abs(blue[..., 1] - blue_goal[..., 1]) <= 0.04)
    red_z = torch.abs(red[..., 2] - red_goal[..., 2]) <= 0.011
    blue_z = torch.abs(blue[..., 2] - blue_goal[..., 2]) <= 0.011
    red_at = red_xy & red_z
    blue_at = blue_xy & blue_z
    open_gripper = (qpos[..., 7] > 0.035) & (qpos[..., 8] > 0.035)
    tcp_clear = tcp[..., 2] > 0.15
    red_ready = red_at & open_gripper & tcp_clear
    blue_ready = blue_at & open_gripper & tcp_clear
    return torch.stack([red_at, blue_at, red_ready, blue_ready], dim=-1).to(dtype=obs.dtype)


def masked_bce_with_logits(logits, labels, weights):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (bce * weights).sum() / weights.sum().clamp_min(1.0)


def compute_loss(model, batch, spec):
    predicted_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = epsilon_loss(predicted_noise, batch["noise"], batch["mask"])

    # Causal predicate/readiness forecasts trained from current and future labels.
    condition, aux_logits = model.condition_and_logits(batch["raw_obs"])
    current_labels = predicate_labels(batch["raw_obs"][:, -1])
    mid_index = min(7, batch["future_obs"].shape[1] - 1)
    end_index = batch["future_obs"].shape[1] - 1
    mid_labels = predicate_labels(batch["future_obs"][:, mid_index])
    end_labels = predicate_labels(batch["future_obs"][:, end_index])
    aux_labels = torch.cat([current_labels, mid_labels, end_labels], dim=-1)

    current_w = torch.ones_like(current_labels)
    mid_w = batch["future_mask"][:, mid_index, :].expand_as(mid_labels)
    end_w = batch["future_mask"][:, end_index, :].expand_as(end_labels)
    readiness_weight = torch.tensor([1.0, 1.0, 1.5, 1.5], device=aux_logits.device, dtype=aux_logits.dtype)
    aux_weights = torch.cat([current_w, mid_w, end_w], dim=-1) * readiness_weight.repeat(3)
    aux_loss = masked_bce_with_logits(aux_logits, aux_labels, aux_weights)

    # Terminal/release/retreat action consistency on the denoiser's clean-action estimate.
    alpha_bar = batch["alpha_bar"].reshape(-1, 1, 1).to(dtype=predicted_noise.dtype)
    sqrt_alpha = torch.sqrt(alpha_bar.clamp_min(1.0e-6))
    sqrt_one_minus = torch.sqrt((1.0 - alpha_bar).clamp_min(0.0))
    x0_pred = (batch["noisy_action"] - sqrt_one_minus * predicted_noise) / sqrt_alpha

    future_labels = predicate_labels(batch["future_obs"])
    at_any_goal = torch.maximum(future_labels[..., 0], future_labels[..., 1])
    ready_any = torch.maximum(future_labels[..., 2], future_labels[..., 3])
    terminal_weight = torch.clamp(at_any_goal + ready_any, 0.0, 1.0).unsqueeze(-1) * batch["future_mask"]
    terminal_weight = terminal_weight * alpha_bar.clamp(0.05, 1.0)
    terminal_action_loss = (((x0_pred - batch["encoded_action"]).square()) * terminal_weight).sum() / (terminal_weight.sum() * x0_pred.shape[-1]).clamp_min(1.0)

    prior_loss = model.aux_weight * aux_loss + model.terminal_action_weight * terminal_action_loss
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
