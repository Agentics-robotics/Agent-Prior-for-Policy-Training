import torch

from appl.public import DiffusionBackbone, epsilon_loss


# Monotone acquisition phase prior for acquire_active_block__h02.
# The model remains an epsilon-predicting action diffusion policy.  The prior is
# implemented as a learned causal phase encoder whose ordered latent posterior is
# provided to the diffusion denoiser and trained with masked heuristic phase
# labels plus an ordered-progress consistency loss.


class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, activation=torch.nn.SiLU):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers.append(torch.nn.Linear(last, h))
            layers.append(activation())
            last = h
        layers.append(torch.nn.Linear(last, out_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MonotoneAcquisitionPolicy(torch.nn.Module):
    """Phase-conditioned diffusion policy for block acquisition.

    Phases are ordered as:
      0 open/free-space approach
      1 aligned descent above a table block
      2 low contact/closure
      3 attached lift
      4 early carry / successor-ready lifted state
    The posterior over these phases is inferred from two causal observations.
    """

    def __init__(self, spec):
        super().__init__()
        self.obs_dim = int(spec.get("observation_dimension", 47))
        self.num_phases = 5
        self.condition_dim = int(spec.get("candidate_config", {}).get("condition_dim", 256))
        self.phase_embed_dim = int(spec.get("candidate_config", {}).get("phase_embed_dim", 24))
        n = spec["normalizer"]
        self.register_buffer("obs_mean", torch.as_tensor(n["mean"], dtype=torch.float32).view(1, 1, -1))
        self.register_buffer("obs_std", torch.as_tensor(n["std"], dtype=torch.float32).view(1, 1, -1))

        self.step_feature_dim = self.obs_dim + 31
        self.step_encoder = MLP(self.step_feature_dim, [128, 128], 128)
        self.temporal_encoder = torch.nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.phase_head = MLP(128, [96], self.num_phases)
        self.phase_embedding = torch.nn.Parameter(torch.randn(self.num_phases, self.phase_embed_dim) * 0.02)
        self.condition_head = MLP(128 + self.num_phases + self.phase_embed_dim + 16, [256, 256], self.condition_dim)
        self.backbone = DiffusionBackbone(self.condition_dim, spec["training"])

        with torch.no_grad():
            last = self.phase_head.net[-1]
            if isinstance(last, torch.nn.Linear):
                last.bias.zero_()

    def norm_obs(self, raw_history):
        return (raw_history - self.obs_mean.to(raw_history.dtype)) / self.obs_std.to(raw_history.dtype)

    def active_weight_blue(self, obs):
        """Softly choose blue once red is placed on its goal and down on the table."""
        red_xyz = obs[..., 25:28]
        red_goal = obs[..., 41:44]
        red_goal_xy_dist = torch.linalg.vector_norm(red_xyz[..., 0:2] - red_goal[..., 0:2], dim=-1)
        red_z_err = torch.abs(red_xyz[..., 2] - red_goal[..., 2])
        score = (0.065 - red_goal_xy_dist) * 45.0 + (0.045 - red_z_err) * 55.0
        return torch.sigmoid(score).unsqueeze(-1)

    def relative_step_features(self, obs):
        tcp = obs[..., 18:21]
        red = obs[..., 25:28]
        blue = obs[..., 32:35]
        red_goal = obs[..., 41:44]
        blue_goal = obs[..., 44:47]
        w_blue = self.active_weight_blue(obs)
        active = red * (1.0 - w_blue) + blue * w_blue
        active_goal = red_goal * (1.0 - w_blue) + blue_goal * w_blue

        rel_tcp = tcp - active
        rel_goal = active_goal - active
        red_rel_tcp = tcp - red
        blue_rel_tcp = tcp - blue
        red_goal_rel = red_goal - red
        blue_goal_rel = blue_goal - blue
        xyz_scale = torch.tensor([0.40, 0.50, 0.35], device=obs.device, dtype=obs.dtype)
        small_xyz_scale = torch.tensor([0.16, 0.16, 0.25], device=obs.device, dtype=obs.dtype)

        horiz = torch.linalg.vector_norm(rel_tcp[..., 0:2], dim=-1, keepdim=True) / 0.25
        z_rel = rel_tcp[..., 2:3] / 0.30
        active_z = active[..., 2:3] / 0.30
        tcp_z = tcp[..., 2:3] / 0.45
        red_z = red[..., 2:3] / 0.30
        blue_z = blue[..., 2:3] / 0.30
        finger_width = (obs[..., 7:8] + obs[..., 8:9]) / 0.08
        finger_vel = (obs[..., 16:17] + obs[..., 17:18]) / 0.10
        active_is_blue = w_blue
        features = torch.cat([
            rel_tcp / small_xyz_scale, rel_goal / xyz_scale,
            red_rel_tcp / xyz_scale, blue_rel_tcp / xyz_scale,
            red_goal_rel / xyz_scale, blue_goal_rel / xyz_scale,
            horiz, z_rel, active_z, tcp_z, red_z, blue_z,
            finger_width, finger_vel, active_is_blue,
            obs[..., 7:9] / 0.04, obs[..., 16:18] / 0.10,
        ], dim=-1)
        return features

    def step_features(self, raw_history):
        norm = self.norm_obs(raw_history)
        rel = self.relative_step_features(raw_history)
        return torch.cat([norm, rel], dim=-1)

    def encode_condition(self, raw_history):
        step_feat = self.step_features(raw_history)
        B, T, D = step_feat.shape
        step_emb = self.step_encoder(step_feat.reshape(B * T, D)).reshape(B, T, -1)
        out, h_n = self.temporal_encoder(step_emb)
        h = h_n[-1]
        logits = self.phase_head(h)
        probs = torch.softmax(logits, dim=-1)
        phase_emb = probs @ self.phase_embedding

        cur = raw_history[:, -1, :]
        prev = raw_history[:, 0, :]
        w_blue_cur = self.active_weight_blue(cur)
        active_cur = cur[:, 25:28] * (1.0 - w_blue_cur) + cur[:, 32:35] * w_blue_cur
        active_prev = prev[:, 25:28] * (1.0 - w_blue_cur) + prev[:, 32:35] * w_blue_cur
        tcp_cur = cur[:, 18:21]
        tcp_prev = prev[:, 18:21]
        tcp_rel = tcp_cur - active_cur
        horiz = torch.linalg.vector_norm(tcp_rel[:, 0:2], dim=-1, keepdim=True) / 0.25
        dyn = torch.cat([
            tcp_rel / torch.tensor([0.16, 0.16, 0.25], device=cur.device, dtype=cur.dtype),
            (active_cur - active_prev) / torch.tensor([0.05, 0.05, 0.12], device=cur.device, dtype=cur.dtype),
            (tcp_cur - tcp_prev) / torch.tensor([0.05, 0.05, 0.12], device=cur.device, dtype=cur.dtype),
            (cur[:, 7:8] + cur[:, 8:9]) / 0.08,
            (cur[:, 16:17] + cur[:, 17:18]) / 0.10,
            w_blue_cur,
            probs[:, 3:5].sum(dim=-1, keepdim=True),
            (probs * torch.arange(self.num_phases, device=cur.device, dtype=cur.dtype).view(1, -1)).sum(dim=-1, keepdim=True) / 4.0,
            active_cur[:, 2:3] / 0.30,
            horiz,
        ], dim=-1)
        cond = self.condition_head(torch.cat([h, probs, phase_emb, dyn], dim=-1))
        return cond, logits

    def phase_logits(self, raw_history):
        return self.encode_condition(raw_history)[1]

    def forward(self, noisy_action, timestep, raw_history):
        cond, unused_logits = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, cond)


def active_hard_blue(obs):
    red = obs[..., 25:28]
    red_goal = obs[..., 41:44]
    dist_xy = torch.linalg.vector_norm(red[..., 0:2] - red_goal[..., 0:2], dim=-1)
    z_err = torch.abs(red[..., 2] - red_goal[..., 2])
    return (dist_xy < 0.075) & (z_err < 0.055)


def phase_labels_from_obs(obs):
    """Current-state heuristic labels for the ordered acquisition phase."""
    blue_active = active_hard_blue(obs).unsqueeze(-1)
    red = obs[..., 25:28]
    blue = obs[..., 32:35]
    active = torch.where(blue_active, blue, red)
    tcp = obs[..., 18:21]
    horiz = torch.linalg.vector_norm(tcp[..., 0:2] - active[..., 0:2], dim=-1)
    z_rel = tcp[..., 2] - active[..., 2]
    obj_z = active[..., 2]
    finger_width = obs[..., 7] + obs[..., 8]
    finger_vel = obs[..., 16] + obs[..., 17]

    label = torch.zeros(obs.shape[:-1], device=obs.device, dtype=torch.long)
    descend = (horiz < 0.075) & (obj_z < 0.055) & (z_rel > 0.035)
    label = torch.where(descend, torch.ones_like(label), label)
    contact = (horiz < 0.055) & (obj_z < 0.060) & (z_rel < 0.060) & ((finger_width < 0.070) | (finger_vel < -0.002))
    label = torch.where(contact, torch.full_like(label, 2), label)
    lift = obj_z > 0.055
    carry = obj_z > 0.175
    label = torch.where(lift, torch.full_like(label, 3), label)
    label = torch.where(carry, torch.full_like(label, 4), label)
    return label


def masked_cross_entropy(logits, labels, mask=None):
    K = logits.shape[-1]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, K), labels.reshape(-1), reduction="none")
    if mask is None:
        return loss.mean()
    m = mask.reshape(-1).to(loss.dtype)
    return (loss * m).sum() / m.sum().clamp_min(1.0)


def build_model(spec):
    return MonotoneAcquisitionPolicy(spec)


def compute_loss(model, batch, spec):
    predicted = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = epsilon_loss(predicted, batch["noise"], batch["mask"])

    current_logits = model.phase_logits(batch["raw_obs"])
    current_labels = phase_labels_from_obs(batch["raw_obs"][:, -1, :])
    ce_current = masked_cross_entropy(current_logits, current_labels, None)

    future_obs = batch["future_obs"]
    future_mask = batch["future_mask"].squeeze(-1)
    B, H, D = future_obs.shape
    prev_for_future = torch.cat([batch["raw_obs"][:, 0:1, :], future_obs[:, :-1, :]], dim=1)
    future_hist = torch.stack([prev_for_future, future_obs], dim=2).reshape(B * H, 2, D)
    future_logits = model.phase_logits(future_hist).reshape(B, H, model.num_phases)
    future_labels = phase_labels_from_obs(future_obs)
    ce_future = masked_cross_entropy(future_logits, future_labels, future_mask)

    probs = torch.softmax(future_logits, dim=-1)
    phase_values = torch.arange(model.num_phases, device=probs.device, dtype=probs.dtype).view(1, 1, -1)
    expected_phase = (probs * phase_values).sum(dim=-1)
    valid_pairs = future_mask[:, 1:] * future_mask[:, :-1]
    monotone_loss = (torch.relu(expected_phase[:, :-1] - expected_phase[:, 1:]) * valid_pairs).sum() / valid_pairs.sum().clamp_min(1.0)

    prior_loss = 0.035 * ce_current + 0.025 * ce_future + 0.010 * monotone_loss
    total = diffusion_loss + prior_loss
    return {"loss": total, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
