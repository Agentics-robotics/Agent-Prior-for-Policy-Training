import torch
import appl.public as public


class ContactFunnelDiffusionPolicy(torch.nn.Module):
    """Object-relative contact-funnel diffusion model for the red-to-buffer handoff."""

    def __init__(self, spec):
        super().__init__()
        cfg = spec["candidate_config"] if "candidate_config" in spec else {}
        self.condition_dim = int(cfg["condition_dim"] if "condition_dim" in cfg else 256)
        self.hidden_dim = int(cfg["hidden_dim"] if "hidden_dim" in cfg else 256)
        self.phase_count = int(cfg["phase_count"] if "phase_count" in cfg else 5)
        self.phase_emb_dim = int(cfg["phase_embedding_dim"] if "phase_embedding_dim" in cfg else 64)
        self.xyz_scale = float(cfg["xyz_scale_m"] if "xyz_scale_m" in cfg else 0.25)
        offsets = cfg["future_offsets"] if "future_offsets" in cfg else [3, 7, 11, 15]
        self.future_offsets = tuple(int(x) for x in offsets)

        normalizer = spec["normalizer"]
        self.register_buffer("obs_mean", torch.as_tensor(normalizer["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.as_tensor(normalizer["std"], dtype=torch.float32))
        self.register_buffer("buffer_xyz", torch.as_tensor([-0.181, 0.0, 0.02], dtype=torch.float32))

        feature_dim = 2 * (47 + 24 + 6 + 4 + 9) + (3 + 3 + 3 + 9 + 1)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, self.hidden_dim),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.SiLU(),
        )
        self.phase_head = torch.nn.Linear(self.hidden_dim, self.phase_count)
        self.phase_embedding = torch.nn.Embedding(self.phase_count, self.phase_emb_dim)
        self.condition_head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim + self.phase_emb_dim + self.phase_count, self.condition_dim),
            torch.nn.LayerNorm(self.condition_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.condition_dim, self.condition_dim),
            torch.nn.SiLU(),
        )
        self.object_delta_head = torch.nn.Linear(self.condition_dim, len(self.future_offsets) * 6)
        self.backbone = public.DiffusionBackbone(self.condition_dim, spec["training"])

    def normalized_observation(self, raw_history):
        mean = self.obs_mean.to(device=raw_history.device, dtype=raw_history.dtype)
        std = self.obs_std.to(device=raw_history.device, dtype=raw_history.dtype)
        return (raw_history - mean) / std.clamp_min(1.0e-6)

    def step_features(self, raw_step, norm_step):
        qpos = raw_step[:, 0:9]
        qvel_norm = norm_step[:, 9:18]
        tcp_xyz = raw_step[:, 18:21]
        red_xyz = raw_step[:, 25:28]
        blue_xyz = raw_step[:, 32:35]
        red_goal = raw_step[:, 41:44]
        blue_goal = raw_step[:, 44:47]
        buf = self.buffer_xyz.to(device=raw_step.device, dtype=raw_step.dtype).view(1, 3)
        s = self.xyz_scale

        rel_cat = torch.cat([
            (tcp_xyz - red_xyz) / s,
            (tcp_xyz - blue_xyz) / s,
            (blue_xyz - red_xyz) / s,
            (red_goal - red_xyz) / s,
            (blue_goal - red_xyz) / s,
            (red_goal - blue_xyz) / s,
            (blue_goal - blue_xyz) / s,
            (buf - red_xyz) / s,
        ], dim=-1)

        scalar_geom = torch.cat([
            torch.linalg.vector_norm(tcp_xyz[:, 0:2] - red_xyz[:, 0:2], dim=-1, keepdim=True) / s,
            torch.linalg.vector_norm(tcp_xyz[:, 0:2] - blue_xyz[:, 0:2], dim=-1, keepdim=True) / s,
            torch.linalg.vector_norm(red_xyz[:, 0:2] - buf[:, 0:2], dim=-1, keepdim=True) / s,
            torch.linalg.vector_norm(blue_xyz[:, 0:2] - red_xyz[:, 0:2], dim=-1, keepdim=True) / s,
            (red_xyz[:, 2:3] - 0.02) / s,
            (tcp_xyz[:, 2:3] - 0.02) / s,
        ], dim=-1)

        fingers = qpos[:, 7:9]
        finger_width = fingers.sum(dim=-1, keepdim=True)
        finger_features = torch.cat([fingers / 0.04, finger_width / 0.08, (0.08 - finger_width) / 0.08], dim=-1)
        return torch.cat([norm_step, rel_cat, scalar_geom, finger_features, qvel_norm], dim=-1)

    def causal_features(self, raw_history):
        norm_history = self.normalized_observation(raw_history)
        step0 = self.step_features(raw_history[:, 0, :], norm_history[:, 0, :])
        step1 = self.step_features(raw_history[:, 1, :], norm_history[:, 1, :])
        qpos_delta = norm_history[:, 1, 0:9] - norm_history[:, 0, 0:9]
        tcp_delta = (raw_history[:, 1, 18:21] - raw_history[:, 0, 18:21]) / self.xyz_scale
        red_delta = (raw_history[:, 1, 25:28] - raw_history[:, 0, 25:28]) / self.xyz_scale
        blue_delta = (raw_history[:, 1, 32:35] - raw_history[:, 0, 32:35]) / self.xyz_scale
        width0 = raw_history[:, 0, 7:9].sum(dim=-1, keepdim=True)
        width1 = raw_history[:, 1, 7:9].sum(dim=-1, keepdim=True)
        width_delta = (width1 - width0) / 0.08
        hist_delta = torch.cat([tcp_delta, red_delta, blue_delta, qpos_delta, width_delta], dim=-1)
        return torch.cat([step0, step1, hist_delta], dim=-1)

    def encode_condition(self, raw_history):
        features = self.causal_features(raw_history)
        hidden = self.encoder(features)
        phase_logits = self.phase_head(hidden)
        phase_prob = torch.softmax(phase_logits, dim=-1)
        phase_embed = phase_prob @ self.phase_embedding.weight
        condition = self.condition_head(torch.cat([hidden, phase_embed, phase_prob], dim=-1))
        return condition, phase_logits

    def aux_predictions(self, raw_history):
        condition, phase_logits = self.encode_condition(raw_history)
        object_delta = self.object_delta_head(condition).view(raw_history.shape[0], len(self.future_offsets), 6)
        return phase_logits, object_delta

    def forward(self, noisy_action, timestep, raw_history):
        condition, unused_phase_logits = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, condition)


def build_model(spec):
    return ContactFunnelDiffusionPolicy(spec)


def phase_labels(raw_history, model):
    cur = raw_history[:, -1, :]
    tcp_xyz = cur[:, 18:21]
    red_xyz = cur[:, 25:28]
    blue_xyz = cur[:, 32:35]
    buf = model.buffer_xyz.to(device=cur.device, dtype=cur.dtype).view(1, 3)
    finger_width = cur[:, 7:9].sum(dim=-1)
    red_buffer_xy = torch.linalg.vector_norm(red_xyz[:, 0:2] - buf[:, 0:2], dim=-1)
    tcp_red_xy = torch.linalg.vector_norm(tcp_xyz[:, 0:2] - red_xyz[:, 0:2], dim=-1)
    tcp_blue_xy = torch.linalg.vector_norm(tcp_xyz[:, 0:2] - blue_xyz[:, 0:2], dim=-1)
    red_lifted = red_xyz[:, 2] > 0.075
    red_at_buffer = (red_buffer_xy < 0.045) & (red_xyz[:, 2] < 0.045)
    near_red_low = (tcp_red_xy < 0.045) & (tcp_xyz[:, 2] < 0.085)
    closing_or_closed = finger_width < 0.060
    near_blue = (tcp_blue_xy < 0.055) & red_at_buffer
    labels = torch.zeros(cur.shape[0], device=cur.device, dtype=torch.long)
    labels = torch.where(near_red_low | closing_or_closed, torch.ones_like(labels), labels)
    labels = torch.where(red_lifted & (red_buffer_xy >= 0.055), torch.full_like(labels, 2), labels)
    labels = torch.where((red_buffer_xy < 0.075) & (~red_at_buffer) & (closing_or_closed | red_lifted), torch.full_like(labels, 3), labels)
    labels = torch.where(red_at_buffer | near_blue, torch.full_like(labels, 4), labels)
    return labels


def object_delta_loss(model, raw_history, future_obs, future_mask, pred_delta):
    cur = raw_history[:, -1, :]
    cur_red = cur[:, 25:28]
    cur_blue = cur[:, 32:35]
    targets = []
    masks = []
    horizon = future_obs.shape[1]
    for offset in model.future_offsets:
        idx = min(max(int(offset), 0), horizon - 1)
        red_delta = (future_obs[:, idx, 25:28] - cur_red) / model.xyz_scale
        blue_delta = (future_obs[:, idx, 32:35] - cur_blue) / model.xyz_scale
        targets.append(torch.cat([red_delta, blue_delta], dim=-1))
        masks.append(future_mask[:, idx, :])
    target = torch.stack(targets, dim=1)
    mask = torch.stack(masks, dim=1)
    return ((pred_delta - target).square() * mask).sum() / (mask.sum() * pred_delta.shape[-1]).clamp_min(1.0)


def compute_loss(model, batch, spec):
    predicted_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = public.epsilon_loss(predicted_noise, batch["noise"], batch["mask"])
    phase_logits, pred_delta = model.aux_predictions(batch["raw_obs"])
    labels = phase_labels(batch["raw_obs"], model)
    phase_loss = torch.nn.functional.cross_entropy(phase_logits, labels)
    delta_loss = object_delta_loss(model, batch["raw_obs"], batch["future_obs"], batch["future_mask"], pred_delta)
    cfg = spec["candidate_config"] if "candidate_config" in spec else {}
    phase_weight = float(cfg["phase_loss_weight"] if "phase_loss_weight" in cfg else 0.05)
    delta_weight = float(cfg["object_delta_loss_weight"] if "object_delta_loss_weight" in cfg else 0.50)
    prior_loss = phase_weight * phase_loss + delta_weight * delta_loss
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss, "phase_loss": phase_loss.detach(), "object_delta_loss": delta_loss.detach()}
