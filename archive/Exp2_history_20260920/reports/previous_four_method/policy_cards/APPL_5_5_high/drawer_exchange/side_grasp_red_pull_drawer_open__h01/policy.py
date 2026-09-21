import math
import torch
import appl.public as public


class SideGraspRedPullDiffusionPolicy(torch.nn.Module):
    """Learned DDPM policy with a red-relative side-grasp conditioning prior."""

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        cfg = spec.get("candidate_config", {})
        if isinstance(cfg, dict) and "config" in cfg and isinstance(cfg["config"], dict):
            cfg = cfg["config"]
        self.condition_dim = int(cfg.get("condition_dim", 256))
        self.hidden_dim = int(cfg.get("encoder_hidden_dim", 256))
        self.phase_count = int(cfg.get("phase_count", 5))
        self.future_target_dim = int(cfg.get("future_target_dim", 25))
        self.horizon = int(spec.get("training", {}).get("horizon", 16))

        normalizer = spec.get("normalizer", spec.get("shared_normalizer", None))
        if normalizer is None:
            raise ValueError("spec must provide the shared M1_v2 normalizer")
        self.register_buffer("obs_mean", torch.tensor(normalizer["mean"], dtype=torch.float32), persistent=False)
        self.register_buffer("obs_std", torch.tensor(normalizer["std"], dtype=torch.float32), persistent=False)

        dummy = torch.zeros(1, 2, 47, dtype=torch.float32)
        feature_dim = int(self.make_features(dummy).shape[-1])

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, self.hidden_dim),
            torch.nn.Mish(),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.Mish(),
            torch.nn.LayerNorm(self.hidden_dim),
        )
        self.phase_head = torch.nn.Linear(self.hidden_dim, self.phase_count)
        self.phase_table = torch.nn.Parameter(torch.randn(self.phase_count, 64) * 0.02)
        self.condition_head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim + 64, self.condition_dim),
            torch.nn.Mish(),
            torch.nn.LayerNorm(self.condition_dim),
            torch.nn.Linear(self.condition_dim, self.condition_dim),
        )
        self.future_head = torch.nn.Sequential(
            torch.nn.Linear(self.condition_dim, self.hidden_dim),
            torch.nn.Mish(),
            torch.nn.Linear(self.hidden_dim, self.horizon * self.future_target_dim),
        )
        self.backbone = public.DiffusionBackbone(self.condition_dim, spec["training"])

    def normalize_obs_local(self, raw):
        mean = self.obs_mean.to(device=raw.device, dtype=raw.dtype)
        std = self.obs_std.to(device=raw.device, dtype=raw.dtype)
        return (raw - mean) / std

    @staticmethod
    def tensor_constant(device, dtype, values):
        return torch.tensor(values, device=device, dtype=dtype)

    def make_features(self, raw_history):
        raw = raw_history
        dtype = raw.dtype
        device = raw.device
        norm_flat = self.normalize_obs_local(raw).reshape(raw.shape[0], -1)

        qpos = raw[:, :, 0:9]
        tcp = raw[:, :, 18:21]
        red = raw[:, :, 25:28]
        blue = raw[:, :, 32:35]
        drawer = raw[:, :, 39:40]
        dvel = raw[:, :, 40:41]
        red_goal = raw[:, :, 41:44]
        blue_goal = raw[:, :, 44:47]

        zeros = torch.zeros_like(drawer)
        drawer_center = torch.cat((0.19 - drawer, zeros, 0.035 + zeros), dim=-1)
        finger_width = (qpos[:, :, 7:8] + qpos[:, :, 8:9]).clamp_min(0.0)
        finger_balance = qpos[:, :, 7:8] - qpos[:, :, 8:9]
        red_x_plus_drawer = red[:, :, 0:1] + drawer

        grasp_target = torch.cat((red[:, :, 0:1] - 0.245, zeros, red[:, :, 2:3] + 0.065), dim=-1)
        pull_line_target = torch.cat((red[:, :, 0:1] - 0.245, zeros, 0.128 + zeros), dim=-1)

        tcp_scale = self.tensor_constant(device, dtype, [0.5, 0.5, 0.3]).view(1, 1, 3)
        red_scale = self.tensor_constant(device, dtype, [0.4, 0.4, 0.2]).view(1, 1, 3)
        tcp_center = self.tensor_constant(device, dtype, [0.0, 0.0, 0.20]).view(1, 1, 3)

        per_step = torch.cat(
            (
                (tcp - red) / 0.30,
                (tcp - blue) / 0.60,
                (red - red_goal) / 0.40,
                (blue - blue_goal) / 0.60,
                (red - drawer_center) / 0.40,
                (blue - drawer_center) / 0.60,
                (tcp - tcp_center) / tcp_scale,
                red / red_scale,
                finger_width / 0.08,
                finger_balance / 0.08,
                drawer / 0.30,
                dvel / 0.30,
                red_x_plus_drawer / 0.40,
                (tcp - grasp_target) / 0.30,
                (tcp - pull_line_target) / 0.30,
            ),
            dim=-1,
        ).reshape(raw.shape[0], -1)

        tcp_delta = tcp[:, 1] - tcp[:, 0]
        red_delta = red[:, 1] - red[:, 0]
        blue_delta = blue[:, 1] - blue[:, 0]
        rel_delta = (tcp[:, 1] - red[:, 1]) - (tcp[:, 0] - red[:, 0])
        drawer_delta = drawer[:, 1] - drawer[:, 0]
        finger_delta = finger_width[:, 1] - finger_width[:, 0]
        qpos_delta = qpos[:, 1] - qpos[:, 0]
        delta_features = torch.cat(
            (
                tcp_delta / 0.05,
                red_delta / 0.05,
                blue_delta / 0.05,
                drawer_delta / 0.05,
                finger_delta / 0.02,
                rel_delta / 0.05,
                qpos_delta / 0.50,
            ),
            dim=-1,
        )

        cur_drawer = drawer[:, 1]
        cur_finger = finger_width[:, 1]
        cur_tcp_z = tcp[:, 1, 2:3]
        cur_red = red[:, 1]
        smooth_flags = torch.cat(
            (
                cur_drawer / 0.30,
                (0.30 - cur_drawer) / 0.30,
                cur_finger / 0.08,
                dvel[:, 1] / 0.30,
                cur_tcp_z / 0.40,
                cur_red[:, 1:2] / 0.20,
                (cur_red[:, 0:1] + cur_drawer) / 0.40,
                cur_red[:, 0:1] / 0.40,
                torch.tanh(20.0 * (cur_drawer - 0.02)),
                torch.tanh(20.0 * (cur_drawer - 0.26)),
                torch.tanh(60.0 * (cur_finger - 0.04)),
                torch.tanh(20.0 * (cur_tcp_z - 0.22)),
            ),
            dim=-1,
        )

        return torch.cat((norm_flat, per_step, delta_features, smooth_flags), dim=-1)

    def condition(self, raw_history):
        features = self.make_features(raw_history)
        hidden = self.encoder(features)
        phase_logits = self.phase_head(hidden)
        phase_weights = torch.softmax(phase_logits, dim=-1)
        phase_embedding = phase_weights @ self.phase_table.to(dtype=hidden.dtype)
        condition = self.condition_head(torch.cat((hidden, phase_embedding), dim=-1))
        return condition, phase_logits

    def auxiliary(self, raw_history):
        condition, phase_logits = self.condition(raw_history)
        future = self.future_head(condition).reshape(raw_history.shape[0], self.horizon, self.future_target_dim)
        return {"condition": condition, "phase_logits": phase_logits, "future": future}

    def forward(self, noisy_action, timestep, raw_history):
        condition, unused_phase_logits = self.condition(raw_history)
        return self.backbone(noisy_action, timestep, condition)

    def future_targets(self, future_obs):
        norm = self.normalize_obs_local(future_obs)
        qpos_norm = norm[:, :, 0:9]
        tcp_norm = norm[:, :, 18:21]
        red_norm = norm[:, :, 25:28]
        drawer_norm = norm[:, :, 39:41]

        tcp = future_obs[:, :, 18:21]
        red = future_obs[:, :, 25:28]
        drawer = future_obs[:, :, 39:40]
        qpos = future_obs[:, :, 0:9]
        zeros = torch.zeros_like(drawer)
        finger_width = (qpos[:, :, 7:8] + qpos[:, :, 8:9]).clamp_min(0.0)
        grasp_target = torch.cat((red[:, :, 0:1] - 0.245, zeros, red[:, :, 2:3] + 0.065), dim=-1)
        rel = (tcp - red) / 0.30
        red_x_plus_drawer = (red[:, :, 0:1] + drawer) / 0.40
        grasp_error = (tcp - grasp_target) / 0.30
        return torch.cat(
            (qpos_norm, tcp_norm, red_norm, drawer_norm, rel, red_x_plus_drawer, finger_width / 0.08, grasp_error),
            dim=-1,
        )

    def phase_labels(self, raw_history):
        cur = raw_history[:, -1]
        drawer = cur[:, 39]
        finger_width = cur[:, 7] + cur[:, 8]
        tcp_z = cur[:, 20]
        labels = torch.zeros(raw_history.shape[0], dtype=torch.long, device=raw_history.device)
        labels = torch.where((drawer < 0.02) & (finger_width < 0.04), torch.ones_like(labels), labels)
        labels = torch.where((drawer >= 0.02) & (drawer < 0.26), torch.full_like(labels, 2), labels)
        labels = torch.where((drawer >= 0.26) & (finger_width < 0.06), torch.full_like(labels, 3), labels)
        labels = torch.where((drawer >= 0.26) & (finger_width >= 0.06) & (tcp_z < 0.22), torch.full_like(labels, 3), labels)
        labels = torch.where((drawer >= 0.26) & (finger_width >= 0.06) & (tcp_z >= 0.22), torch.full_like(labels, 4), labels)
        return labels


def build_model(spec):
    return SideGraspRedPullDiffusionPolicy(spec)


def compute_loss(model, batch, spec):
    predicted_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = public.epsilon_loss(predicted_noise, batch["noise"], batch["mask"])

    aux = model.auxiliary(batch["raw_obs"])
    target = model.future_targets(batch["future_obs"])
    future_mask = batch["future_mask"]
    future_error = (aux["future"] - target).square() * future_mask
    future_loss = future_error.sum() / (future_mask.sum() * target.shape[-1]).clamp_min(1.0)

    phase_labels = model.phase_labels(batch["raw_obs"])
    phase_loss = torch.nn.functional.cross_entropy(aux["phase_logits"], phase_labels)

    prior_loss = 0.05 * (future_loss + 0.10 * phase_loss)
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
