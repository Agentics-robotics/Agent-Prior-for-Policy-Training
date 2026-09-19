import torch
import appl.public as public


PHASE_NAMES = ["approach", "close", "pull", "release", "retreat"]


class ContactPhaseDiffusionPolicy(torch.nn.Module):
    """Learned diffusion policy with a contact-phase latent conditioning prior."""

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        cfg = spec.get("candidate_config", {})
        self.condition_dim = int(cfg.get("condition_dim", 192))
        self.hidden_dim = int(cfg.get("hidden_dim", 256))
        self.phase_dim = int(cfg.get("phase_embedding_dim", 32))
        self.feature_dim = 31
        self.obs_dim = int(spec.get("observation_dimension", 47))

        normalizer = spec["normalizer"]
        self.register_buffer("obs_mean", torch.tensor(normalizer["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.tensor(normalizer["std"], dtype=torch.float32))

        encoder_in = 2 * self.obs_dim + self.feature_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(encoder_in, self.hidden_dim),
            torch.nn.Mish(),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.Mish(),
            torch.nn.LayerNorm(self.hidden_dim),
        )
        self.phase_head = torch.nn.Linear(self.hidden_dim, len(PHASE_NAMES))
        self.phase_embedding = torch.nn.Embedding(len(PHASE_NAMES), self.phase_dim)
        self.progress_head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, 96),
            torch.nn.Mish(),
            torch.nn.Linear(96, 3),
        )
        self.gripper_head = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, 96),
            torch.nn.Mish(),
            torch.nn.Linear(96, 4),
        )
        cond_in = self.hidden_dim + self.phase_dim + 3 + 4
        self.condition_projector = torch.nn.Sequential(
            torch.nn.Linear(cond_in, self.condition_dim),
            torch.nn.Mish(),
            torch.nn.LayerNorm(self.condition_dim),
        )
        self.backbone = public.DiffusionBackbone(self.condition_dim, spec["training"])

    def normalized_history(self, raw_history):
        mean = self.obs_mean.to(device=raw_history.device, dtype=raw_history.dtype)
        std = self.obs_std.to(device=raw_history.device, dtype=raw_history.dtype)
        return (raw_history - mean) / std

    def make_features(self, raw_history):
        prev = raw_history[:, 0]
        cur = raw_history[:, -1]
        delta = cur - prev

        qpos = cur[:, 0:9]
        qvel = cur[:, 9:18]
        tcp = cur[:, 18:25]
        red = cur[:, 25:32]
        blue = cur[:, 32:39]
        drawer = cur[:, 39:40]
        drawer_v = cur[:, 40:41]
        red_goal = cur[:, 41:44]
        blue_goal = cur[:, 44:47]

        tcp_pos = tcp[:, 0:3]
        red_pos = red[:, 0:3]
        blue_pos = blue[:, 0:3]
        tcp_dpos = delta[:, 18:21]
        red_dpos = delta[:, 25:28]

        finger_l = qpos[:, 7:8]
        finger_r = qpos[:, 8:9]
        finger_width = finger_l + finger_r
        finger_asym = finger_l - finger_r
        finger_vel_sum = qvel[:, 7:8] + qvel[:, 8:9]

        drawer_center = torch.cat([
            0.19 - drawer,
            torch.zeros_like(drawer),
            torch.full_like(drawer, 0.035),
        ], dim=-1)

        tcp_minus_red = tcp_pos - red_pos
        tcp_red_dist = torch.linalg.norm(tcp_minus_red, dim=-1, keepdim=True)
        tcp_minus_red_delta = tcp_dpos - red_dpos
        red_minus_drawer = red_pos - drawer_center

        feats = [
            (finger_width - 0.04) / 0.04,
            finger_asym / 0.04,
            finger_vel_sum / 0.4,
            (drawer - 0.15) / 0.15,
            drawer_v / 0.20,
            tcp_minus_red / 0.30,
            tcp_red_dist / 0.30,
            tcp_dpos / 0.10,
            red_dpos / 0.10,
            tcp_minus_red_delta / 0.10,
            red_minus_drawer / 0.25,
            (red_goal - red_pos) / 0.40,
            (blue_goal - blue_pos) / 0.40,
            (tcp_pos[:, 2:3] - 0.20) / 0.25,
            red_pos[:, 1:2] / 0.30,
            red_pos[:, 0:1] / 0.40,
            (0.26 - drawer) / 0.26,
        ]
        return torch.cat(feats, dim=-1)

    def encode_condition(self, raw_history):
        norm_hist = self.normalized_history(raw_history).reshape(raw_history.shape[0], -1)
        features = self.make_features(raw_history)
        hidden = self.encoder(torch.cat([norm_hist, features], dim=-1))
        phase_logits = self.phase_head(hidden)
        phase_prob = torch.softmax(phase_logits, dim=-1)
        phase_context = phase_prob @ self.phase_embedding.weight
        progress = self.progress_head(hidden)
        gripper = self.gripper_head(hidden)
        condition = self.condition_projector(torch.cat([hidden, phase_context, progress, gripper], dim=-1))
        aux = {
            "phase_logits": phase_logits,
            "progress": progress,
            "gripper": gripper,
        }
        return condition, aux

    def auxiliary(self, raw_history):
        return self.encode_condition(raw_history)[1]

    def predict_with_aux(self, noisy_action, timestep, raw_history):
        condition, aux = self.encode_condition(raw_history)
        eps = self.backbone(noisy_action, timestep, condition)
        return eps, aux

    def forward(self, noisy_action, timestep, raw_history):
        eps, aux = self.predict_with_aux(noisy_action, timestep, raw_history)
        return eps


def phase_labels(raw_obs, future_obs, future_mask):
    cur = raw_obs[:, -1]
    drawer = cur[:, 39]
    drawer_v = cur[:, 40]
    finger_width = cur[:, 7] + cur[:, 8]

    valid = future_mask[:, :, 0] > 0.5
    future_finger = future_obs[:, :, 7] + future_obs[:, :, 8]
    large = torch.full_like(future_finger, 10.0)
    small = torch.full_like(future_finger, -10.0)
    min_future_finger = torch.where(valid, future_finger, large).min(dim=1).values
    max_future_finger = torch.where(valid, future_finger, small).max(dim=1).values
    has_future = valid.any(dim=1)
    min_future_finger = torch.where(has_future, min_future_finger, finger_width)
    max_future_finger = torch.where(has_future, max_future_finger, finger_width)

    labels = torch.zeros(cur.shape[0], device=cur.device, dtype=torch.long)

    close = (drawer < 0.035) & (finger_width > 0.025) & (
        (min_future_finger < 0.035) | ((finger_width - min_future_finger) > 0.015)
    )
    pull = ((finger_width <= 0.030) | (drawer_v > 0.01)) & (drawer < 0.255)
    release = (drawer >= 0.255) & ((finger_width < 0.065) | (max_future_finger > 0.070))
    retreat = (drawer >= 0.255) & (finger_width >= 0.065)

    labels = torch.where(close, torch.ones_like(labels), labels)
    labels = torch.where(pull, torch.full_like(labels, 2), labels)
    labels = torch.where(release, torch.full_like(labels, 3), labels)
    labels = torch.where(retreat, torch.full_like(labels, 4), labels)
    return labels


def progress_loss_value(progress_pred, future_obs, future_mask):
    indices = [3, 7, 15]
    target = future_obs[:, indices, 39]
    mask = future_mask[:, indices, 0]
    target = (target - 0.15) / 0.15
    sq = (progress_pred - target).square() * mask
    return sq.sum() / mask.sum().clamp_min(1.0)


def gripper_intention_loss_value(gripper_pred, native_action, action_mask):
    indices = [0, 3, 7, 15]
    target = native_action[:, indices, 7]
    mask = action_mask[:, indices, 0]
    sq = (gripper_pred - target).square() * mask
    return sq.sum() / mask.sum().clamp_min(1.0)


def build_model(spec):
    return ContactPhaseDiffusionPolicy(spec)


def compute_loss(model, batch, spec):
    pred_noise, aux = model.predict_with_aux(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = public.epsilon_loss(pred_noise, batch["noise"], batch["mask"])

    labels = phase_labels(batch["raw_obs"], batch["future_obs"], batch["future_mask"])
    phase_loss = torch.nn.functional.cross_entropy(aux["phase_logits"], labels)
    drawer_loss = progress_loss_value(aux["progress"], batch["future_obs"], batch["future_mask"])
    grip_loss = gripper_intention_loss_value(aux["gripper"], batch["native_action"], batch["mask"])

    cfg = spec.get("candidate_config", {})
    phase_w = float(cfg.get("phase_loss_weight", 0.05))
    progress_w = float(cfg.get("progress_loss_weight", 0.10))
    gripper_w = float(cfg.get("gripper_loss_weight", 0.05))
    prior_loss = phase_w * phase_loss + progress_w * drawer_loss + gripper_w * grip_loss
    loss = diffusion_loss + prior_loss
    return {
        "loss": loss,
        "diffusion_loss": diffusion_loss,
        "prior_loss": prior_loss,
    }
