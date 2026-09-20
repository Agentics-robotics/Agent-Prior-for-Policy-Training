import torch
from appl.public import DiffusionBackbone, epsilon_loss


class BlueAttachmentLatentDiffusionPolicy(torch.nn.Module):
    """Diffusion policy with a learned causal blue-attachment latent.

    The latent is not a controller. It is predicted from the two-observation
    history and is used as part of the global condition of a standard action
    denoising U-Net. During training it receives auxiliary supervision derived
    from future blue lift/contact evidence inside the sampled training slice.
    """

    def __init__(self, spec):
        super().__init__()
        self.observation_dim = int(spec.get("observation_dimension", 47))
        self.action_dim = 8

        candidate = spec.get("candidate_config", {})
        if isinstance(candidate, dict) and "config" in candidate and "condition_dim" not in candidate:
            candidate = candidate.get("config", {})
        if not isinstance(candidate, dict):
            candidate = {}
        self.condition_dim = int(candidate.get("condition_dim", 256))
        self.attachment_loss_weight = float(candidate.get("attachment_loss_weight", 0.05))
        self.gripper_phase_loss_weight = float(candidate.get("gripper_phase_loss_weight", 0.015))

        normalizer = spec.get("normalizer", spec.get("shared_normalizer", None))
        if normalizer is None:
            raise ValueError("A shared M1_v2 normalizer must be present in spec['normalizer'].")
        self.register_buffer("obs_mean", torch.as_tensor(normalizer["mean"], dtype=torch.float32), persistent=False)
        self.register_buffer("obs_std", torch.as_tensor(normalizer["std"], dtype=torch.float32), persistent=False)

        feature_dim = 2 * self.observation_dim + 46
        hidden = int(candidate.get("encoder_hidden_dim", 256))
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden),
            torch.nn.Mish(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, self.condition_dim),
            torch.nn.Mish(),
            torch.nn.LayerNorm(self.condition_dim),
        )
        self.attachment_head = torch.nn.Linear(self.condition_dim, 1)
        self.free_token = torch.nn.Parameter(torch.zeros(self.condition_dim))
        self.attached_token = torch.nn.Parameter(torch.zeros(self.condition_dim))
        torch.nn.init.normal_(self.free_token, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.attached_token, mean=0.0, std=0.02)
        self.condition_norm = torch.nn.LayerNorm(self.condition_dim)
        self.backbone = DiffusionBackbone(self.condition_dim, spec["training"])

    def normalize_observation_local(self, raw_history):
        mean = self.obs_mean.to(device=raw_history.device, dtype=raw_history.dtype)
        std = self.obs_std.to(device=raw_history.device, dtype=raw_history.dtype)
        return (raw_history - mean) / std.clamp_min(1.0e-6)

    def relative_features(self, raw_history):
        prev = raw_history[:, 0]
        cur = raw_history[:, -1]
        tcp = cur[:, 18:21]
        prev_tcp = prev[:, 18:21]
        red = cur[:, 25:28]
        prev_red = prev[:, 25:28]
        blue = cur[:, 32:35]
        prev_blue = prev[:, 32:35]
        drawer_position = cur[:, 39:40]
        drawer_velocity = cur[:, 40:41]
        red_goal = cur[:, 41:44]
        blue_goal = cur[:, 44:47]

        rel_tcp_blue = tcp - blue
        prev_rel_tcp_blue = prev_tcp - prev_blue
        finger_width = cur[:, 7:8] + cur[:, 8:9]
        prev_finger_width = prev[:, 7:8] + prev[:, 8:9]
        finger_velocity = cur[:, 16:17] + cur[:, 17:18]
        tcp_delta = tcp - prev_tcp
        blue_delta = blue - prev_blue
        red_delta = red - prev_red
        zeros = drawer_position.new_zeros(drawer_position.shape)
        drawer_center = torch.cat((0.19 - drawer_position, zeros, zeros + 0.035), dim=-1)

        xy_distance = torch.linalg.norm(rel_tcp_blue[:, 0:2], dim=-1, keepdim=True)
        xyz_distance = torch.linalg.norm(rel_tcp_blue, dim=-1, keepdim=True)
        tcp_speed = torch.linalg.norm(tcp_delta, dim=-1, keepdim=True)
        blue_speed = torch.linalg.norm(blue_delta, dim=-1, keepdim=True)

        pieces = [
            rel_tcp_blue / 0.15,
            prev_rel_tcp_blue / 0.15,
            (rel_tcp_blue - prev_rel_tcp_blue) / 0.05,
            (tcp - red) / 0.20,
            (red - red_goal) / 0.20,
            (blue - blue_goal) / 0.30,
            (blue - drawer_center) / 0.30,
            (tcp - blue_goal) / 0.30,
            tcp_delta / 0.05,
            blue_delta / 0.05,
            red_delta / 0.05,
            (finger_width - 0.05) / 0.04,
            (prev_finger_width - 0.05) / 0.04,
            (finger_width - prev_finger_width) / 0.02,
            finger_velocity / 0.20,
            tcp[:, 2:3] / 0.30,
            blue[:, 2:3] / 0.30,
            red[:, 2:3] / 0.10,
            xy_distance / 0.20,
            xyz_distance / 0.30,
            tcp_speed / 0.10,
            blue_speed / 0.10,
            drawer_position / 0.30,
            drawer_velocity / 0.10,
        ]
        return torch.cat(pieces, dim=-1)

    def encode_condition(self, raw_history):
        norm_history = self.normalize_observation_local(raw_history).reshape(raw_history.shape[0], -1)
        rel = self.relative_features(raw_history)
        features = torch.cat((norm_history, rel), dim=-1)
        base = self.encoder(features)
        attachment_logit = self.attachment_head(base)
        attachment_probability = torch.sigmoid(attachment_logit)
        phase_token = (1.0 - attachment_probability) * self.free_token + attachment_probability * self.attached_token
        condition = self.condition_norm(base + phase_token)
        return condition, attachment_logit

    def forward(self, noisy_action, timestep, raw_history):
        condition, unused_logit = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, condition)

    def attachment_targets(self, batch):
        raw_history = batch["raw_obs"]
        future = batch["future_obs"]
        future_mask = batch["future_mask"].squeeze(-1) > 0.5
        cur = raw_history[:, -1]

        cur_tcp = cur[:, 18:21]
        cur_blue = cur[:, 32:35]
        cur_rel = cur_tcp - cur_blue
        cur_xy_close = torch.linalg.norm(cur_rel[:, 0:2], dim=-1) < 0.055
        cur_gap_mean = 0.5 * (cur[:, 7] + cur[:, 8])
        observed_closed = cur_gap_mean < 0.025
        lifted_now = (cur_blue[:, 2] > 0.040) & observed_closed & (torch.linalg.norm(cur_rel, dim=-1) < 0.090)

        future_blue_z = future[:, :, 34]
        masked_z = future_blue_z.masked_fill(~future_mask, -1000000.0)
        max_future_z = masked_z.max(dim=1).values
        has_future = future_mask.any(dim=1)
        max_future_z = torch.where(has_future, max_future_z, cur_blue[:, 2])
        threshold = torch.maximum(cur_blue[:, 2] + 0.008, cur_blue[:, 2].new_full(cur_blue[:, 2].shape, 0.026))
        future_lift = max_future_z > threshold

        future_rel = future[:, :, 18:21] - future[:, :, 32:35]
        rel_error = torch.linalg.norm(future_rel - cur_rel[:, None, :], dim=-1)
        stable_candidate = (rel_error < 0.060) & future_mask
        stable_future = stable_candidate.any(dim=1)

        target = (lifted_now | (observed_closed & cur_xy_close & future_lift & stable_future)).to(raw_history.dtype)
        return target.unsqueeze(-1)

    def prior_loss(self, batch, predicted_noise):
        condition, attachment_logit = self.encode_condition(batch["raw_obs"])
        attachment_target = self.attachment_targets(batch)
        attachment_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            attachment_logit, attachment_target
        )

        alpha_bar = batch["alpha_bar"].to(device=predicted_noise.device, dtype=predicted_noise.dtype).reshape(-1, 1, 1)
        alpha_bar = alpha_bar.clamp(1.0e-5, 0.99999)
        clean_estimate = (batch["noisy_action"] - torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
        clean_gripper = clean_estimate[:, :, 7].clamp(-1.5, 1.5)
        gripper_target = batch["encoded_action"][:, :, 7]
        mask = batch["mask"].squeeze(-1)

        future = batch["future_obs"]
        future_tcp_blue_xy = torch.linalg.norm(future[:, :, 18:20] - future[:, :, 32:34], dim=-1)
        future_finger_mean = 0.5 * (future[:, :, 7] + future[:, :, 8])
        near_contact_or_carry = (
            ((future_tcp_blue_xy < 0.065) & (future[:, :, 20] < 0.080) & (future[:, :, 34] < 0.060))
            | (future[:, :, 34] > 0.040)
            | (future_finger_mean < 0.028)
        ).to(clean_gripper.dtype)
        phase_weight = (1.0 + 2.0 * near_contact_or_carry) * batch["future_mask"].squeeze(-1)
        phase_weight = phase_weight * mask
        gripper_error = torch.nn.functional.smooth_l1_loss(clean_gripper, gripper_target, reduction="none")
        gripper_loss = (gripper_error * phase_weight).sum() / phase_weight.sum().clamp_min(1.0)

        return self.attachment_loss_weight * attachment_loss + self.gripper_phase_loss_weight * gripper_loss


def build_model(spec):
    return BlueAttachmentLatentDiffusionPolicy(spec)


def compute_loss(model, batch, spec):
    predicted_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = epsilon_loss(predicted_noise, batch["noise"], batch["mask"])
    prior_loss = model.prior_loss(batch, predicted_noise)
    total_loss = diffusion_loss + prior_loss
    return {"loss": total_loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
