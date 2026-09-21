import math
import numpy as np
import torch
import appl.public as public


class RigidCarryDiffusionPolicy(torch.nn.Module):
    """Diffusion policy with a learned rigid blue-carry dynamics prior.

    The action model is a standard conditional 1-D U-Net.  The conditioning vector
    augments the shared normalized state history with causal engineered features
    describing the current TCP/blue offset, object-goal deltas, drawer geometry,
    finger width, and short finite-difference motion.  An auxiliary learned
    dynamics head maps the model's denoised action estimate to future TCP/blue
    positions.  Its blue prediction contains an explicit rigid-carry branch:
    predicted TCP position plus the current TCP-to-blue world offset, gated against
    a learned free-object branch for pre-grasp and release periods.
    """

    def __init__(self, spec, config=None):
        super().__init__()
        if config is None:
            config = {}
        self.spec = spec
        normalizer = spec["normalizer"]
        self.register_buffer("obs_mean", torch.tensor(normalizer["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.tensor(normalizer["std"], dtype=torch.float32))
        self.register_buffer("act_min", torch.tensor(normalizer["action_min"], dtype=torch.float32))
        self.register_buffer("act_scale", torch.tensor(normalizer["action_scale"], dtype=torch.float32))

        self.condition_dim = int(config.get("condition_dim", 256))
        self.aux_hidden_dim = int(config.get("aux_hidden_dim", 128))
        self.feature_dim = 148

        self.condition_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, 256),
            torch.nn.Mish(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 256),
            torch.nn.Mish(),
            torch.nn.Linear(256, self.condition_dim),
            torch.nn.Mish(),
        )
        self.backbone = public.DiffusionBackbone(self.condition_dim, spec["training"])

        self.aux_condition = torch.nn.Sequential(
            torch.nn.Linear(self.condition_dim, self.aux_hidden_dim),
            torch.nn.Mish(),
            torch.nn.Linear(self.aux_hidden_dim, self.aux_hidden_dim),
            torch.nn.Mish(),
        )
        self.aux_h0 = torch.nn.Linear(self.condition_dim, self.aux_hidden_dim)
        self.action_encoder = torch.nn.Sequential(
            torch.nn.Linear(16, 64),
            torch.nn.Mish(),
            torch.nn.Linear(64, 64),
            torch.nn.Mish(),
        )
        self.aux_gru = torch.nn.GRU(
            input_size=64 + self.aux_hidden_dim,
            hidden_size=self.aux_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.tcp_delta_head = torch.nn.Linear(self.aux_hidden_dim, 3)
        self.free_blue_delta_head = torch.nn.Linear(self.aux_hidden_dim, 3)
        self.attach_logit_head = torch.nn.Linear(self.aux_hidden_dim, 1)

        # Broad metric scales, not fit on this skill slice.
        self.tcp_delta_limit_m = float(config.get("tcp_delta_limit_m", 0.50))
        self.free_blue_delta_limit_m = float(config.get("free_blue_delta_limit_m", 0.50))

    def normalize_obs(self, raw):
        return (raw - self.obs_mean.to(device=raw.device, dtype=raw.dtype)) / self.obs_std.to(device=raw.device, dtype=raw.dtype)

    def denormalize_action(self, encoded):
        return (encoded + 1.0) * self.act_scale.to(device=encoded.device, dtype=encoded.dtype) / 2.0 + self.act_min.to(device=encoded.device, dtype=encoded.dtype)

    def quat_normalize(self, q):
        return q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)

    def quat_conjugate(self, q):
        return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

    def quat_rotate(self, q, v):
        q = self.quat_normalize(q)
        qw = q[..., :1]
        qv = q[..., 1:]
        uv = torch.cross(qv, v, dim=-1)
        uuv = torch.cross(qv, uv, dim=-1)
        return v + 2.0 * (qw * uv + uuv)

    def make_features(self, raw_history):
        raw = raw_history
        norm_flat = self.normalize_obs(raw).reshape(raw.shape[0], -1)
        prev = raw[:, 0, :]
        cur = raw[:, 1, :]

        tcp_prev = prev[:, 18:21]
        tcp = cur[:, 18:21]
        tcp_quat = cur[:, 21:25]
        blue_prev = prev[:, 32:35]
        blue = cur[:, 32:35]
        blue_quat = cur[:, 35:39]
        red = cur[:, 25:28]
        red_goal = cur[:, 41:44]
        blue_goal = cur[:, 44:47]
        drawer_pos = cur[:, 39:40]
        drawer_vel = cur[:, 40:41]
        zeros = torch.zeros_like(drawer_pos)
        drawer_center = torch.cat([0.19 - drawer_pos, zeros, zeros + 0.035], dim=-1)

        rel_cur = blue - tcp
        rel_prev = blue_prev - tcp_prev
        local_offset = self.quat_rotate(self.quat_conjugate(tcp_quat), rel_cur)
        finger_cur = cur[:, 7:9]
        finger_prev = prev[:, 7:9]
        finger_mean_cur = finger_cur.mean(dim=-1, keepdim=True)
        finger_mean_prev = finger_prev.mean(dim=-1, keepdim=True)

        dist_bt = rel_cur.norm(dim=-1, keepdim=True)
        dist_bg = (blue_goal - blue).norm(dim=-1, keepdim=True)
        dist_tg = (blue_goal - tcp).norm(dim=-1, keepdim=True)
        dist_rg = (red_goal - red).norm(dim=-1, keepdim=True)

        eng = torch.cat([
            rel_cur / 0.25,
            rel_prev / 0.25,
            (rel_cur - rel_prev) / 0.05,
            (blue_goal - blue) / 0.35,
            (blue_goal - tcp) / 0.35,
            (drawer_center - blue) / 0.35,
            (red_goal - red) / 0.35,
            (blue - blue_prev) / 0.08,
            (tcp - tcp_prev) / 0.08,
            finger_cur / 0.04,
            finger_prev / 0.04,
            finger_mean_cur / 0.04,
            finger_mean_prev / 0.04,
            (finger_mean_cur - finger_mean_prev) / 0.02,
            dist_bt / 0.35,
            dist_bg / 0.35,
            dist_tg / 0.35,
            dist_rg / 0.35,
            local_offset / 0.25,
            tcp_quat,
            blue_quat,
            drawer_pos / 0.30,
            drawer_vel / 0.20,
            drawer_center / 0.35,
        ], dim=-1)
        return torch.cat([norm_flat, eng], dim=-1)

    def encode_condition(self, raw_history):
        return self.condition_encoder(self.make_features(raw_history))

    def forward(self, noisy_action, timestep, raw_history):
        condition = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, condition)

    def predict_dynamics(self, raw_history, encoded_action_sequence):
        """Predict future positions from causal state and a clean encoded action sequence.

        Returns a dictionary with differentiable predictions.  The method is used
        only during training as an auxiliary prior loss; deployment sampling calls
        forward() exactly as a diffusion policy.
        """
        condition = self.encode_condition(raw_history)
        aux_c = self.aux_condition(condition)
        first_delta = torch.zeros_like(encoded_action_sequence[:, :1, :])
        action_delta = torch.cat([first_delta, encoded_action_sequence[:, 1:, :] - encoded_action_sequence[:, :-1, :]], dim=1)
        action_feat = torch.cat([encoded_action_sequence, action_delta], dim=-1)
        action_emb = self.action_encoder(action_feat)
        repeated_c = aux_c.unsqueeze(1).expand(-1, encoded_action_sequence.shape[1], -1)
        rnn_in = torch.cat([action_emb, repeated_c], dim=-1)
        h0 = torch.tanh(self.aux_h0(condition)).unsqueeze(0)
        rnn_out, nothing = self.aux_gru(rnn_in, h0)

        cur = raw_history[:, 1, :]
        tcp0 = cur[:, 18:21]
        blue0 = cur[:, 32:35]
        offset0 = blue0 - tcp0

        tcp_delta = self.tcp_delta_limit_m * torch.tanh(self.tcp_delta_head(rnn_out))
        free_delta = self.free_blue_delta_limit_m * torch.tanh(self.free_blue_delta_head(rnn_out))
        tcp_hat = tcp0.unsqueeze(1) + tcp_delta
        blue_free_hat = blue0.unsqueeze(1) + free_delta
        attach_logit = self.attach_logit_head(rnn_out)
        attach_prob = torch.sigmoid(attach_logit)
        blue_rigid_hat = tcp_hat + offset0.unsqueeze(1)
        blue_hat = attach_prob * blue_rigid_hat + (1.0 - attach_prob) * blue_free_hat
        return {
            "tcp_pos": tcp_hat,
            "blue_pos": blue_hat,
            "blue_rigid_pos": blue_rigid_hat,
            "blue_free_pos": blue_free_hat,
            "attach_logit": attach_logit,
            "attach_prob": attach_prob,
            "offset0": offset0,
        }


def bce_with_logits(logits, targets):
    return torch.clamp(logits, min=0.0) - logits * targets + torch.log1p(torch.exp(-torch.abs(logits)))


def dynamics_prior_loss(model, raw_history, encoded_actions, future_obs, future_mask, sample_weight=None):
    pred = model.predict_dynamics(raw_history, encoded_actions)
    dtype = encoded_actions.dtype
    device = encoded_actions.device
    mask = future_mask.to(device=device, dtype=dtype)
    if sample_weight is None:
        sw = torch.ones_like(mask)
    else:
        sw = sample_weight.to(device=device, dtype=dtype).reshape(-1, 1, 1)
    w = mask * sw
    denom = w.sum().clamp_min(1.0)

    target_tcp = future_obs[:, :, 18:21].to(device=device, dtype=dtype)
    target_blue = future_obs[:, :, 32:35].to(device=device, dtype=dtype)
    future_finger = future_obs[:, :, 7:9].to(device=device, dtype=dtype).mean(dim=-1, keepdim=True)
    future_blue_z = future_obs[:, :, 34:35].to(device=device, dtype=dtype)
    attach_target = ((future_finger < 0.028) & (future_blue_z > 0.040)).to(dtype=dtype)

    cur = raw_history[:, 1, :].to(device=device, dtype=dtype)
    current_finger = cur[:, 7:9].mean(dim=-1, keepdim=True)
    current_blue_z = cur[:, 34:35]
    current_attached = ((current_finger < 0.028) & (current_blue_z > 0.040)).to(dtype=dtype)
    carry_w = w * attach_target * current_attached.unsqueeze(1)
    carry_denom = carry_w.sum().clamp_min(1.0)

    blue_loss = ((((pred["blue_pos"] - target_blue) / 0.12) ** 2) * w).sum() / (denom * 3.0)
    tcp_loss = ((((pred["tcp_pos"] - target_tcp) / 0.12) ** 2) * w).sum() / (denom * 3.0)
    gate_loss = (bce_with_logits(pred["attach_logit"], attach_target) * w).sum() / denom
    pred_offset = pred["blue_pos"] - pred["tcp_pos"]
    offset_loss = ((((pred_offset - pred["offset0"].unsqueeze(1)) / 0.04) ** 2) * carry_w).sum() / (carry_denom * 3.0)
    return blue_loss + 0.35 * tcp_loss + 0.03 * gate_loss + 0.15 * offset_loss


def clean_action_from_epsilon(noisy_action, predicted_epsilon, alpha_bar):
    dtype = noisy_action.dtype
    device = noisy_action.device
    ab = alpha_bar.to(device=device, dtype=dtype).reshape(-1, 1, 1).clamp(1.0e-5, 0.99999)
    return (noisy_action - torch.sqrt(1.0 - ab) * predicted_epsilon) / torch.sqrt(ab)


def build_model(spec):
    config = {}
    candidate = spec.get("candidate_config", None)
    if isinstance(candidate, dict):
        inner = candidate.get("config", {})
        if isinstance(inner, dict):
            config = inner
    return RigidCarryDiffusionPolicy(spec, config)


def compute_loss(model, batch, spec):
    predicted_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = public.epsilon_loss(predicted_noise, batch["noise"], batch["mask"])

    x0_pred = clean_action_from_epsilon(batch["noisy_action"], predicted_noise, batch["alpha_bar"])
    # Smooth bounding limits high-noise auxiliary amplification while keeping a
    # gradient from the prior loss into the epsilon predictor.
    x0_pred = 1.5 * torch.tanh(x0_pred / 1.5)
    sample_weight = batch["alpha_bar"].detach().clamp(0.05, 1.0)

    pred_dyn_loss = dynamics_prior_loss(
        model,
        batch["raw_obs"],
        x0_pred,
        batch["future_obs"],
        batch["future_mask"],
        sample_weight=sample_weight,
    )
    teacher_dyn_loss = dynamics_prior_loss(
        model,
        batch["raw_obs"],
        batch["encoded_action"].detach(),
        batch["future_obs"],
        batch["future_mask"],
        sample_weight=None,
    )
    prior_weight = 0.08
    candidate = spec.get("candidate_config", None)
    if isinstance(candidate, dict):
        cfg = candidate.get("config", {})
        if isinstance(cfg, dict) and "prior_loss_weight" in cfg:
            prior_weight = float(cfg["prior_loss_weight"])
    prior_loss = prior_weight * (pred_dyn_loss + 0.35 * teacher_dyn_loss)
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
