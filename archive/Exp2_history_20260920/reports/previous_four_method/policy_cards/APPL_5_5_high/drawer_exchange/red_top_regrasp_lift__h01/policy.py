import torch

from appl.public import DiffusionBackbone, normalize_observation, epsilon_loss


class RedTopRegraspLiftPolicy(torch.nn.Module):
    """Learned DDPM epsilon model with a red-centered top-grasp representation."""

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        cfg = spec.get("candidate_config", {})
        if isinstance(cfg, dict) and "config" in cfg and "condition_dim" not in cfg:
            cfg = cfg.get("config", {})
        self.config = cfg if isinstance(cfg, dict) else {}
        self.condition_dim = int(self.config.get("condition_dim", 256))
        self.history_steps = int(spec.get("training", {}).get("observation_steps", 2))
        self.horizon = int(spec.get("training", {}).get("horizon", 16))

        raw_dim = self.history_steps * int(spec.get("observation_dimension", 47))
        engineered_dim = 52
        hidden = int(self.config.get("encoder_hidden_dim", 128))

        self.raw_encoder = torch.nn.Sequential(
            torch.nn.Linear(raw_dim, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.Mish(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Mish(),
        )
        self.affordance_encoder = torch.nn.Sequential(
            torch.nn.Linear(engineered_dim, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.Mish(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Mish(),
        )
        self.condition_fusion = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden, 2 * hidden),
            torch.nn.Mish(),
            torch.nn.Linear(2 * hidden, self.condition_dim),
            torch.nn.LayerNorm(self.condition_dim),
        )

        self.backbone = DiffusionBackbone(self.condition_dim, spec["training"])

        aux_hidden = int(self.config.get("aux_hidden_dim", 128))
        action_embed_dim = int(self.config.get("action_embed_dim", 64))
        context_dim = int(self.config.get("aux_context_dim", 64))
        pos_dim = int(self.config.get("step_embed_dim", 16))
        self.step_embedding = torch.nn.Parameter(torch.zeros(1, self.horizon, pos_dim))
        torch.nn.init.normal_(self.step_embedding, mean=0.0, std=0.02)
        self.action_encoder = torch.nn.Sequential(
            torch.nn.Linear(8, action_embed_dim),
            torch.nn.Mish(),
            torch.nn.Linear(action_embed_dim, action_embed_dim),
            torch.nn.Mish(),
        )
        self.aux_context = torch.nn.Sequential(
            torch.nn.Linear(self.condition_dim, context_dim),
            torch.nn.Mish(),
        )
        self.aux_head = torch.nn.Sequential(
            torch.nn.Linear(action_embed_dim + context_dim + pos_dim, aux_hidden),
            torch.nn.Mish(),
            torch.nn.Linear(aux_hidden, aux_hidden // 2),
            torch.nn.Mish(),
            torch.nn.Linear(aux_hidden // 2, 8),
        )

    def forward(self, noisy_action, timestep, raw_history):
        condition = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, condition)

    def encode_condition(self, raw_history):
        norm = normalize_observation(raw_history, self.spec)
        raw_flat = norm.reshape(norm.shape[0], -1)
        raw_code = self.raw_encoder(raw_flat)
        affordance = self.engineered_affordance(raw_history)
        aff_code = self.affordance_encoder(affordance)
        return self.condition_fusion(torch.cat([raw_code, aff_code], dim=-1))

    @staticmethod
    def quat_conj(q):
        return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

    @staticmethod
    def quat_mul(a, b):
        aw, ax, ay, az = a.unbind(dim=-1)
        bw, bx, by, bz = b.unbind(dim=-1)
        return torch.stack([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ], dim=-1)

    def engineered_affordance(self, raw_history):
        prev = raw_history[:, 0]
        cur = raw_history[:, -1]
        tcp = cur[:, 18:25]
        tcp_prev = prev[:, 18:25]
        red = cur[:, 25:32]
        red_prev = prev[:, 25:32]
        qpos = cur[:, 0:9]
        qpos_prev = prev[:, 0:9]
        qvel = cur[:, 9:18]

        tcp_xyz = tcp[:, :3]
        red_xyz = red[:, :3]
        tcp_prev_xyz = tcp_prev[:, :3]
        red_prev_xyz = red_prev[:, :3]
        rel = (tcp_xyz - red_xyz) / 0.15
        rel_prev = (tcp_prev_xyz - red_prev_xyz) / 0.15
        rel_change = rel - rel_prev
        red_motion = (red_xyz - red_prev_xyz) / 0.15
        tcp_motion = (tcp_xyz - tcp_prev_xyz) / 0.15

        red_goal = cur[:, 41:44]
        red_to_goal = (red_goal - red_xyz) / 0.30
        drawer_pos = cur[:, 39:40]
        drawer_vel = cur[:, 40:41]
        drawer_center = torch.cat([
            0.19 - drawer_pos,
            torch.zeros_like(drawer_pos),
            torch.full_like(drawer_pos, 0.035),
        ], dim=-1)
        red_to_drawer = (red_xyz - drawer_center) / 0.30
        drawer_open_margin = (drawer_pos - 0.26) / 0.10
        drawer_feats = torch.cat([drawer_pos / 0.30, drawer_vel / 0.10, drawer_open_margin], dim=-1)

        fingers = qpos[:, 7:9]
        fingers_prev = qpos_prev[:, 7:9]
        gap = fingers.sum(dim=-1, keepdim=True) / 0.08
        gap_prev = fingers_prev.sum(dim=-1, keepdim=True) / 0.08
        finger_diff = (fingers[:, :1] - fingers[:, 1:2]) / 0.02
        finger_diff_prev = (fingers_prev[:, :1] - fingers_prev[:, 1:2]) / 0.02
        finger_vel = qvel[:, 7:9] / 0.10
        finger_feats = torch.cat([gap, fingers / 0.04, finger_diff, gap_prev, finger_diff_prev, finger_vel], dim=-1)

        tcp_q = tcp[:, 3:7]
        red_q = red[:, 3:7]
        tcp_q_prev = tcp_prev[:, 3:7]
        red_q_prev = red_prev[:, 3:7]
        rel_q = self.quat_mul(tcp_q, self.quat_conj(red_q))
        rel_q_prev = self.quat_mul(tcp_q_prev, self.quat_conj(red_q_prev))
        qdot = (tcp_q * red_q).sum(dim=-1, keepdim=True)
        xy_dist = torch.linalg.norm(tcp_xyz[:, :2] - red_xyz[:, :2], dim=-1, keepdim=True) / 0.15
        rel_z = (tcp_xyz[:, 2:3] - red_xyz[:, 2:3]) / 0.15
        top_axis_score = tcp_q[:, 2:3].abs()
        scalars = torch.cat([xy_dist, rel_z, top_axis_score, qdot], dim=-1)

        return torch.cat([
            rel,
            rel_prev,
            rel_change,
            red_motion,
            tcp_motion,
            red_to_goal,
            red_to_drawer,
            drawer_feats,
            finger_feats,
            tcp_q,
            red_q,
            rel_q,
            rel_q_prev,
            scalars,
        ], dim=-1)

    def auxiliary_predictions(self, raw_history, encoded_action):
        condition = self.encode_condition(raw_history)
        b, h, other_dim = encoded_action.shape
        action_code = self.action_encoder(encoded_action)
        context = self.aux_context(condition).unsqueeze(1).expand(-1, h, -1)
        if h <= self.step_embedding.shape[1]:
            pos = self.step_embedding[:, :h, :].expand(b, -1, -1)
        else:
            extra = self.step_embedding[:, -1:, :].expand(b, h - self.step_embedding.shape[1], -1)
            pos = torch.cat([self.step_embedding.expand(b, -1, -1), extra], dim=1)
        return self.aux_head(torch.cat([action_code, context, pos], dim=-1))


def auxiliary_targets(raw_obs, future_obs):
    cur = raw_obs[:, -1]
    cur_red = cur[:, 25:28].unsqueeze(1)
    f_tcp = future_obs[:, :, 18:21]
    f_red = future_obs[:, :, 25:28]
    f_qpos = future_obs[:, :, 0:9]
    f_gap_native = f_qpos[:, :, 7:8] + f_qpos[:, :, 8:9]

    rel_future = (f_tcp - f_red) / 0.15
    red_delta = (f_red - cur_red) / 0.15
    gap = f_gap_native / 0.08
    geom = torch.cat([rel_future, red_delta, gap], dim=-1)

    tcp_red_dist = torch.linalg.norm(f_tcp - f_red, dim=-1, keepdim=True)
    lifted = f_red[:, :, 2:3] > 0.12
    closed = f_gap_native < 0.055
    close_to_tcp = tcp_red_dist < 0.05
    attach = (lifted & closed & close_to_tcp).to(future_obs.dtype)
    return geom, attach


def build_model(spec):
    return RedTopRegraspLiftPolicy(spec)


def compute_loss(model, batch, spec):
    predicted_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    mask = batch["mask"].to(predicted_noise.dtype)
    diffusion_loss = epsilon_loss(predicted_noise, batch["noise"], mask)

    alpha_bar = batch["alpha_bar"].to(predicted_noise.dtype).view(-1, 1, 1)
    sqrt_ab = torch.sqrt(alpha_bar).clamp_min(0.05)
    sqrt_om = torch.sqrt((1.0 - alpha_bar).clamp_min(0.0))
    x0_pred = (batch["noisy_action"] - sqrt_om * predicted_noise) / sqrt_ab
    x0_pred = x0_pred.clamp(-2.0, 2.0)

    aux = model.auxiliary_predictions(batch["raw_obs"], x0_pred)
    geom_pred = aux[:, :, :7]
    attach_logit = aux[:, :, 7:8]
    geom_target, attach_target = auxiliary_targets(batch["raw_obs"], batch["future_obs"])

    future_mask = batch["future_mask"].to(predicted_noise.dtype)
    noise_weight = alpha_bar.detach().clamp(0.05, 1.0)
    weighted_mask = future_mask * noise_weight
    geom_denom = (weighted_mask.sum() * geom_pred.shape[-1]).clamp_min(1.0)
    geom_loss = (((geom_pred - geom_target).square()) * weighted_mask).sum() / geom_denom

    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        attach_logit, attach_target, reduction="none")
    attach_loss = (bce * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)

    geom_w = float(model.config.get("aux_geom_weight", 0.03))
    attach_w = float(model.config.get("aux_attach_weight", 0.01))
    prior_loss = geom_w * geom_loss + attach_w * attach_loss
    loss = diffusion_loss + prior_loss
    return {
        "loss": loss,
        "diffusion_loss": diffusion_loss,
        "prior_loss": prior_loss,
    }
