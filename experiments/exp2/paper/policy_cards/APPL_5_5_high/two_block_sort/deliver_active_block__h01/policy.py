import torch
from appl.public import DiffusionBackbone, epsilon_loss


class GoalRelativeDeliveryPolicy(torch.nn.Module):
    """Learned DDPM policy with goal-relative red/blue candidate conditioning."""

    def __init__(self, spec):
        super().__init__()
        normalizer = spec.get("normalizer", spec.get("shared_normalizer"))
        if normalizer is None:
            raise ValueError("spec must contain a shared normalizer")

        mean = torch.tensor(normalizer["mean"], dtype=torch.float32)
        std = torch.tensor(normalizer["std"], dtype=torch.float32)
        self.register_buffer("obs_mean", mean.view(1, 1, -1))
        self.register_buffer("obs_std", std.view(1, 1, -1))
        self.register_buffer("pos_scale", torch.tensor([0.30, 0.30, 0.30], dtype=torch.float32))
        self.register_buffer("goal_abs_scale", torch.tensor([0.50, 0.50, 0.30], dtype=torch.float32))

        self.history_steps = int(spec.get("training", {}).get("observation_steps", 2))
        self.candidate_dim = 33
        self.candidate_embed_dim = 64
        self.condition_dim = 256

        self.candidate_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.candidate_dim, 96),
            torch.nn.SiLU(),
            torch.nn.Linear(96, self.candidate_embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.candidate_embed_dim, self.candidate_embed_dim),
            torch.nn.SiLU(),
        )

        raw_dim = self.history_steps * 47
        gate_in_dim = raw_dim + self.history_steps * 2 * self.candidate_dim
        self.gate_net = torch.nn.Sequential(
            torch.nn.Linear(gate_in_dim, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 2),
        )

        self.raw_encoder = torch.nn.Sequential(
            torch.nn.Linear(raw_dim, 160),
            torch.nn.SiLU(),
            torch.nn.Linear(160, 128),
            torch.nn.SiLU(),
        )

        cond_in_dim = 128 + self.history_steps * self.candidate_embed_dim + self.history_steps * 2 * self.candidate_embed_dim + self.history_steps * self.candidate_dim + 2
        self.condition_net = torch.nn.Sequential(
            torch.nn.Linear(cond_in_dim, 384),
            torch.nn.SiLU(),
            torch.nn.Linear(384, self.condition_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.condition_dim, self.condition_dim),
        )

        self.backbone = DiffusionBackbone(self.condition_dim, spec["training"])
        self.terminal_head = torch.nn.Sequential(
            torch.nn.Linear(self.condition_dim, 96),
            torch.nn.SiLU(),
            torch.nn.Linear(96, 4),
        )

    def normalize_obs(self, raw_history):
        mean = self.obs_mean.to(device=raw_history.device, dtype=raw_history.dtype)
        std = self.obs_std.to(device=raw_history.device, dtype=raw_history.dtype)
        return (raw_history - mean) / std

    def candidate_features(self, obs, use_red):
        tcp_xyz = obs[..., 18:21]
        tcp_q = obs[..., 21:25]
        red_xyz = obs[..., 25:28]
        red_q = obs[..., 28:32]
        blue_xyz = obs[..., 32:35]
        blue_q = obs[..., 35:39]
        red_goal = obs[..., 41:44]
        blue_goal = obs[..., 44:47]
        qpos = obs[..., 0:9]

        if use_red:
            obj_xyz, obj_q, goal = red_xyz, red_q, red_goal
            other_xyz, other_goal = blue_xyz, blue_goal
        else:
            obj_xyz, obj_q, goal = blue_xyz, blue_q, blue_goal
            other_xyz, other_goal = red_xyz, red_goal

        pos_scale = self.pos_scale.to(device=obs.device, dtype=obs.dtype)
        goal_scale = self.goal_abs_scale.to(device=obs.device, dtype=obs.dtype)

        obj_goal = (obj_xyz - goal) / pos_scale
        tcp_obj = (tcp_xyz - obj_xyz) / pos_scale
        tcp_goal = (tcp_xyz - goal) / pos_scale
        other_obj = (other_xyz - obj_xyz) / pos_scale
        other_goal_err = (other_xyz - other_goal) / pos_scale
        goal_abs = goal / goal_scale

        obj_goal_xy_norm = torch.linalg.vector_norm(obj_xyz[..., 0:2] - goal[..., 0:2], dim=-1, keepdim=True) / 0.30
        tcp_obj_norm = torch.linalg.vector_norm(tcp_xyz - obj_xyz, dim=-1, keepdim=True) / 0.30
        tcp_goal_xy_norm = torch.linalg.vector_norm(tcp_xyz[..., 0:2] - goal[..., 0:2], dim=-1, keepdim=True) / 0.30
        grip_width = (qpos[..., 7:8] + qpos[..., 8:9]) / 0.08
        obj_z = obj_xyz[..., 2:3] / 0.30
        goal_z = goal[..., 2:3] / 0.30
        tcp_minus_obj_z = (tcp_xyz[..., 2:3] - obj_xyz[..., 2:3]) / 0.30

        return torch.cat([
            obj_goal,
            tcp_obj,
            tcp_goal,
            obj_q,
            tcp_q,
            obj_z,
            goal_z,
            obj_goal_xy_norm,
            tcp_obj_norm,
            tcp_goal_xy_norm,
            grip_width,
            goal_abs,
            other_obj,
            other_goal_err,
            tcp_minus_obj_z,
        ], dim=-1)

    def encode(self, raw_history):
        B, T, D = raw_history.shape
        if T != self.history_steps or D != 47:
            raise ValueError("raw_history must have shape [B,2,47]")

        raw_norm = self.normalize_obs(raw_history)
        raw_flat = raw_norm.reshape(B, -1)
        red_feat = self.candidate_features(raw_history, True)
        blue_feat = self.candidate_features(raw_history, False)
        candidate_feat = torch.stack([red_feat, blue_feat], dim=2)

        gate_in = torch.cat([raw_flat, candidate_feat.reshape(B, -1)], dim=-1)
        gate_logits = self.gate_net(gate_in)
        active_prob = torch.softmax(gate_logits, dim=-1)

        emb = self.candidate_encoder(candidate_feat.reshape(B * T * 2, self.candidate_dim))
        emb = emb.reshape(B, T, 2, self.candidate_embed_dim)
        weights = active_prob.view(B, 1, 2, 1)
        active_emb = (weights * emb).sum(dim=2).reshape(B, -1)
        active_feat = (weights * candidate_feat).sum(dim=2).reshape(B, -1)

        raw_emb = self.raw_encoder(raw_flat)
        cond_input = torch.cat([raw_emb, active_emb, emb.reshape(B, -1), active_feat, active_prob], dim=-1)
        condition = self.condition_net(cond_input)
        return condition, gate_logits, active_prob

    def forward(self, noisy_action, timestep, raw_history):
        condition, gate_logits, active_prob = self.encode(raw_history)
        return self.backbone(noisy_action, timestep, condition)

    def predict_terminal_goal_state(self, raw_history):
        condition, gate_logits, active_prob = self.encode(raw_history)
        return self.terminal_head(condition), gate_logits, active_prob


def active_labels_from_current(raw_obs):
    current = raw_obs[:, -1]
    tcp = current[:, 18:21]
    red = current[:, 25:28]
    blue = current[:, 32:35]
    red_goal = current[:, 41:44]
    blue_goal = current[:, 44:47]

    red_xy = torch.linalg.vector_norm(red[:, 0:2] - red_goal[:, 0:2], dim=-1)
    blue_xy = torch.linalg.vector_norm(blue[:, 0:2] - blue_goal[:, 0:2], dim=-1)
    red_z = torch.abs(red[:, 2] - red_goal[:, 2])
    blue_z = torch.abs(blue[:, 2] - blue_goal[:, 2])
    red_done = (red_xy < 0.060) & (red_z < 0.025)
    blue_done = (blue_xy < 0.060) & (blue_z < 0.025)

    red_tcp = torch.linalg.vector_norm(tcp - red, dim=-1)
    blue_tcp = torch.linalg.vector_norm(tcp - blue, dim=-1)
    closer_blue = blue_tcp < red_tcp

    labels = closer_blue.long()
    labels = torch.where(red_done & (~blue_done), torch.ones_like(labels), labels)
    labels = torch.where(blue_done & (~red_done), torch.zeros_like(labels), labels)
    return labels


def final_valid_future(future_obs, future_mask):
    B = future_obs.shape[0]
    valid_count = future_mask.squeeze(-1).sum(dim=1).long().clamp(min=1)
    gather_idx = valid_count - 1
    batch_idx = torch.arange(B, device=future_obs.device)
    valid = (future_mask.squeeze(-1).sum(dim=1) > 0).to(future_obs.dtype)
    return future_obs[batch_idx, gather_idx], valid


def terminal_targets(raw_obs, future_obs, future_mask, labels):
    final_obs, valid = final_valid_future(future_obs, future_mask)
    red_final = final_obs[:, 25:28]
    blue_final = final_obs[:, 32:35]
    red_goal = final_obs[:, 41:44]
    blue_goal = final_obs[:, 44:47]
    is_blue = labels.to(torch.bool).view(-1, 1)
    active_xyz = torch.where(is_blue, blue_final, red_final)
    active_goal = torch.where(is_blue, blue_goal, red_goal)
    scale = torch.tensor([0.30, 0.30, 0.30], device=future_obs.device, dtype=future_obs.dtype)
    rel = (active_xyz - active_goal) / scale
    final_width = (final_obs[:, 7:8] + final_obs[:, 8:9]) / 0.08
    target = torch.cat([rel, final_width], dim=-1)
    return target, valid.view(-1, 1)


def build_model(spec):
    return GoalRelativeDeliveryPolicy(spec)


def compute_loss(model, batch, spec):
    predicted_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = epsilon_loss(predicted_noise, batch["noise"], batch["mask"])

    terminal_pred, gate_logits, active_prob = model.predict_terminal_goal_state(batch["raw_obs"])
    labels = active_labels_from_current(batch["raw_obs"])
    gate_loss = torch.nn.functional.cross_entropy(gate_logits, labels)

    terminal_target, terminal_valid = terminal_targets(batch["raw_obs"], batch["future_obs"], batch["future_mask"], labels)
    denom = (terminal_valid.sum() * terminal_pred.shape[-1]).clamp_min(1.0)
    terminal_loss = ((terminal_pred - terminal_target).square() * terminal_valid).sum() / denom

    prior_loss = 0.05 * gate_loss + 0.05 * terminal_loss
    total_loss = diffusion_loss + prior_loss
    return {"loss": total_loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
