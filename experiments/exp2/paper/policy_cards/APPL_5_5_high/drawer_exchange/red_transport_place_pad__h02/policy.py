import math
import torch
from appl.public import DiffusionBackbone, epsilon_loss


class HighCarrySoftPlacePolicy(torch.nn.Module):
    """Learned DDPM epsilon model with a high-carry/soft-place stage bottleneck."""

    def __init__(self, spec):
        super().__init__()
        normalizer = spec["normalizer"]
        self.register_buffer("obs_mean", torch.tensor(normalizer["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.tensor(normalizer["std"], dtype=torch.float32))
        self.register_buffer("action_min", torch.tensor(normalizer["action_min"], dtype=torch.float32))
        self.register_buffer("action_scale", torch.tensor(normalizer["action_scale"], dtype=torch.float32))
        self.horizon = int(spec["training"].get("horizon", 16))
        self.condition_dim = 256
        self.engineered_dim = 39
        self.flat_obs_dim = 2 * int(spec.get("observation_dimension", 47))

        self.obs_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.flat_obs_dim, 192),
            torch.nn.Mish(),
            torch.nn.LayerNorm(192),
            torch.nn.Linear(192, 192),
            torch.nn.Mish(),
        )
        self.feature_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.engineered_dim, 96),
            torch.nn.Mish(),
            torch.nn.LayerNorm(96),
            torch.nn.Linear(96, 96),
            torch.nn.Mish(),
        )
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(288, 256),
            torch.nn.Mish(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 224),
            torch.nn.Mish(),
        )
        self.stage_head = torch.nn.Linear(224, 4)
        self.stage_embedding = torch.nn.Parameter(torch.randn(4, 28) * 0.02)
        self.condition_projector = torch.nn.Sequential(
            torch.nn.Linear(224 + 4 + 28, 256),
            torch.nn.Mish(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, self.condition_dim),
        )
        self.backbone = DiffusionBackbone(self.condition_dim, spec["training"])

        self.future_head = torch.nn.Sequential(
            torch.nn.Linear(self.condition_dim + self.horizon * 8, 256),
            torch.nn.Mish(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 256),
            torch.nn.Mish(),
            torch.nn.Linear(256, self.horizon * 7),
        )

    def normalize_obs(self, raw_history):
        mean = self.obs_mean.to(device=raw_history.device, dtype=raw_history.dtype)
        std = self.obs_std.to(device=raw_history.device, dtype=raw_history.dtype)
        return (raw_history - mean) / std

    def engineered_features(self, raw_history):
        cur = raw_history[:, -1, :]
        prev = raw_history[:, 0, :]
        tcp = cur[:, 18:21]
        red = cur[:, 25:28]
        blue = cur[:, 32:35]
        drawer_pos = cur[:, 39:40]
        drawer_vel = cur[:, 40:41]
        red_goal = cur[:, 41:44]
        blue_goal = cur[:, 44:47]
        prev_tcp = prev[:, 18:21]
        prev_red = prev[:, 25:28]
        finger_width = cur[:, 7:8] + cur[:, 8:9]
        prev_finger_width = prev[:, 7:8] + prev[:, 8:9]
        qvel = cur[:, 9:18]

        red_to_pad = (red - red_goal) / 0.30
        tcp_to_red = (tcp - red) / 0.30
        tcp_to_pad = (tcp - red_goal) / 0.40
        tcp_to_blue = (tcp - blue) / 0.60
        blue_to_goal = (blue - blue_goal) / 0.40
        red_delta = (red - prev_red) / 0.08
        tcp_delta = (tcp - prev_tcp) / 0.08
        finger_scaled = (finger_width - 0.05) / 0.04
        finger_delta = (finger_width - prev_finger_width) / 0.04
        drawer_scaled = torch.cat(((drawer_pos - 0.26) / 0.15, drawer_vel / 0.20), dim=-1)
        red_xy_dist = torch.linalg.norm(red_to_pad[:, 0:2], dim=-1, keepdim=True)
        tcp_red_dist = torch.linalg.norm(tcp_to_red, dim=-1, keepdim=True)
        tcp_blue_xy_dist = torch.linalg.norm(tcp_to_blue[:, 0:2], dim=-1, keepdim=True)
        heights = torch.cat(((red[:, 2:3] - 0.02) / 0.30, (tcp[:, 2:3] - 0.02) / 0.35), dim=-1)
        qvel_summary = torch.cat((qvel[:, 0:7] / 1.0, qvel[:, 7:9] / 0.25), dim=-1)

        return torch.cat((
            red_to_pad,
            tcp_to_red,
            tcp_to_pad,
            tcp_to_blue,
            blue_to_goal,
            red_delta,
            tcp_delta,
            finger_scaled,
            finger_delta,
            drawer_scaled,
            red_xy_dist,
            tcp_red_dist,
            tcp_blue_xy_dist,
            heights,
            qvel_summary,
        ), dim=-1)

    def encode_condition(self, raw_history):
        norm = self.normalize_obs(raw_history).reshape(raw_history.shape[0], -1)
        engineered = self.engineered_features(raw_history)
        obs_latent = self.obs_encoder(norm)
        feat_latent = self.feature_encoder(engineered)
        hidden = self.trunk(torch.cat((obs_latent, feat_latent), dim=-1))
        stage_logits = self.stage_head(hidden)
        stage_prob = torch.softmax(stage_logits, dim=-1)
        stage_context = stage_prob @ self.stage_embedding.to(device=stage_prob.device, dtype=stage_prob.dtype)
        condition = self.condition_projector(torch.cat((hidden, stage_prob, stage_context), dim=-1))
        return condition, stage_logits

    def forward(self, noisy_action, timestep, raw_history):
        condition, unused_stage_logits = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, condition)

    def predict_with_condition(self, noisy_action, timestep, raw_history):
        condition, stage_logits = self.encode_condition(raw_history)
        predicted_noise = self.backbone(noisy_action, timestep, condition)
        return predicted_noise, condition, stage_logits

    def denormalize_action(self, encoded_action):
        amin = self.action_min.to(device=encoded_action.device, dtype=encoded_action.dtype)
        scale = self.action_scale.to(device=encoded_action.device, dtype=encoded_action.dtype)
        return (encoded_action + 1.0) * scale / 2.0 + amin

    def clean_action_estimate(self, noisy_action, predicted_noise, alpha_bar):
        while alpha_bar.dim() < noisy_action.dim():
            alpha_bar = alpha_bar.unsqueeze(-1)
        alpha_bar = alpha_bar.to(device=noisy_action.device, dtype=noisy_action.dtype).clamp_min(1.0e-4)
        return (noisy_action - torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)

    def predict_future_features(self, condition, clean_action_encoded):
        h = min(clean_action_encoded.shape[1], self.horizon)
        if h < self.horizon:
            pad = torch.zeros(clean_action_encoded.shape[0], self.horizon - h, clean_action_encoded.shape[2],
                              device=clean_action_encoded.device, dtype=clean_action_encoded.dtype)
            action_flat = torch.cat((clean_action_encoded[:, :h], pad), dim=1).reshape(clean_action_encoded.shape[0], -1)
        else:
            action_flat = clean_action_encoded[:, :self.horizon].reshape(clean_action_encoded.shape[0], -1)
        pred = self.future_head(torch.cat((condition, action_flat), dim=-1))
        return pred.reshape(clean_action_encoded.shape[0], self.horizon, 7)


def stage_labels(raw_history):
    cur = raw_history[:, -1, :]
    red = cur[:, 25:28]
    tcp = cur[:, 18:21]
    goal = cur[:, 41:44]
    finger_width = cur[:, 7] + cur[:, 8]
    red_xy_dist = torch.linalg.norm(red[:, 0:2] - goal[:, 0:2], dim=-1)
    tcp_red_xy = torch.linalg.norm(tcp[:, 0:2] - red[:, 0:2], dim=-1)
    tcp_above_red = tcp[:, 2] - red[:, 2]
    placed = (red_xy_dist < 0.055) & (red[:, 2] < 0.040)
    released = (finger_width > 0.060) | (tcp_above_red > 0.070) | (tcp_red_xy > 0.060)
    near_pad_or_descending = (red_xy_dist < 0.090) | ((red[:, 1] < -0.22) & (red[:, 2] < 0.16))
    labels = torch.zeros(raw_history.shape[0], device=raw_history.device, dtype=torch.long)
    labels = torch.where(near_pad_or_descending, torch.ones_like(labels), labels)
    labels = torch.where(placed & (~released), torch.full_like(labels, 2), labels)
    labels = torch.where(placed & released, torch.full_like(labels, 3), labels)
    return labels


def future_feature_targets(raw_history, future_obs):
    cur = raw_history[:, -1, :]
    cur_red = cur[:, 25:28].unsqueeze(1)
    cur_tcp = cur[:, 18:21].unsqueeze(1)
    cur_finger = (cur[:, 7:8] + cur[:, 8:9]).unsqueeze(1)
    future_red = future_obs[:, :, 25:28]
    future_tcp = future_obs[:, :, 18:21]
    future_finger = (future_obs[:, :, 7:8] + future_obs[:, :, 8:9])
    red_delta = (future_red - cur_red) / 0.30
    tcp_delta = (future_tcp - cur_tcp) / 0.30
    finger_delta = (future_finger - cur_finger) / 0.08
    return torch.cat((red_delta, tcp_delta, finger_delta), dim=-1)


def build_model(spec):
    return HighCarrySoftPlacePolicy(spec)


def compute_loss(model, batch, spec):
    predicted_noise, condition, stage_logits = model.predict_with_condition(
        batch["noisy_action"], batch["timesteps"], batch["raw_obs"]
    )
    mask = batch["mask"]
    diffusion_loss = epsilon_loss(predicted_noise, batch["noise"], mask)

    labels = stage_labels(batch["raw_obs"])
    stage_loss = torch.nn.functional.cross_entropy(stage_logits, labels)

    clean = model.clean_action_estimate(batch["noisy_action"], predicted_noise, batch["alpha_bar"])
    clean_for_aux = clean.clamp(-1.5, 1.5)
    future_pred = model.predict_future_features(condition, clean_for_aux)
    target = future_feature_targets(batch["raw_obs"], batch["future_obs"])
    h = min(future_pred.shape[1], target.shape[1])
    future_mask = batch["future_mask"][:, :h]
    future_err = (future_pred[:, :h] - target[:, :h]).square() * future_mask
    alpha = batch["alpha_bar"].to(device=future_err.device, dtype=future_err.dtype).view(-1, 1, 1).detach().clamp(0.05, 1.0)
    future_loss = (future_err * alpha).sum() / (future_mask.sum() * future_pred.shape[-1]).clamp_min(1.0)

    native_clean = model.denormalize_action(clean_for_aux)
    grip = native_clean[:, :h, 7:8]
    stage = labels.view(-1, 1, 1)
    early = ((stage == 0) | (stage == 1)).to(dtype=grip.dtype)
    retreat = (stage == 3).to(dtype=grip.dtype)
    low_and_placed = ((batch["raw_obs"][:, -1, 27] < 0.026) & (labels >= 2)).to(dtype=grip.dtype).view(-1, 1, 1)
    closed_penalty = torch.relu(grip + 0.25).square() * early
    open_penalty = torch.relu(0.25 - grip).square() * torch.maximum(retreat, low_and_placed)
    grip_loss = ((closed_penalty + open_penalty) * mask[:, :h] * alpha).sum() / mask[:, :h].sum().clamp_min(1.0)

    prior_loss = 0.15 * stage_loss + 0.10 * future_loss + 0.02 * grip_loss
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
