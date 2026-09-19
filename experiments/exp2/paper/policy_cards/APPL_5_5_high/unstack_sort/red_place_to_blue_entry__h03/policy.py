import torch
import appl.public as public


class SequentialObjectAttentionPolicy(torch.nn.Module):
    """Learned epsilon-prediction diffusion policy with a red-to-blue attention prior."""

    def __init__(self, spec):
        super().__init__()
        normalizer = spec.get("normalizer", spec.get("shared_normalizer"))
        if normalizer is None:
            raise ValueError("spec must contain the shared observation/action normalizer")
        self.spec = spec
        self.condition_dim = 256
        self.object_feature_dim = 28
        self.object_embed_dim = 96

        self.register_buffer("obs_mean", torch.tensor(normalizer["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.tensor(normalizer["std"], dtype=torch.float32))
        self.register_buffer("xyz_scale", torch.tensor([0.30, 0.30, 0.20], dtype=torch.float32))
        self.register_buffer("fine_xyz_scale", torch.tensor([0.08, 0.08, 0.06], dtype=torch.float32))

        self.object_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.object_feature_dim, 128),
            torch.nn.Mish(),
            torch.nn.Linear(128, 128),
            torch.nn.Mish(),
            torch.nn.Linear(128, self.object_embed_dim),
            torch.nn.Mish(),
        )
        self.history_encoder = torch.nn.Sequential(
            torch.nn.Linear(2 * 47, 160),
            torch.nn.Mish(),
            torch.nn.Linear(160, 128),
            torch.nn.Mish(),
        )
        self.attention_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * 47 + 18, 128),
            torch.nn.Mish(),
            torch.nn.Linear(128, 64),
            torch.nn.Mish(),
            torch.nn.Linear(64, 2),
        )
        precondition_dim = 128 + 3 * self.object_embed_dim + 2 + 18
        self.phase_head = torch.nn.Sequential(
            torch.nn.Linear(precondition_dim, 96),
            torch.nn.Mish(),
            torch.nn.Linear(96, 3),
        )
        self.condition_projector = torch.nn.Sequential(
            torch.nn.Linear(precondition_dim, 256),
            torch.nn.Mish(),
            torch.nn.Linear(256, self.condition_dim),
            torch.nn.LayerNorm(self.condition_dim),
        )
        self.backbone = public.DiffusionBackbone(self.condition_dim, spec["training"])

    def norm_history(self, raw_history):
        mean = self.obs_mean.to(device=raw_history.device, dtype=raw_history.dtype)
        std = self.obs_std.to(device=raw_history.device, dtype=raw_history.dtype)
        return (raw_history - mean) / std

    def score_at_goal(self, obj_pos, goal):
        xy = torch.linalg.norm(obj_pos[:, :2] - goal[:, :2], dim=-1, keepdim=True)
        z = (obj_pos[:, 2:3] - goal[:, 2:3]).abs()
        xy_score = torch.sigmoid((0.055 - xy) / 0.012)
        z_score = torch.sigmoid((0.030 - z) / 0.008)
        return xy_score * z_score, xy, z

    def object_features(self, current, previous, obj_start, goal_start, type_index):
        scale = self.xyz_scale.to(device=current.device, dtype=current.dtype)
        fine = self.fine_xyz_scale.to(device=current.device, dtype=current.dtype)
        tcp = current[:, 18:21]
        prev_tcp = previous[:, 18:21]
        obj = current[:, obj_start:obj_start + 3]
        prev_obj = previous[:, obj_start:obj_start + 3]
        quat = current[:, obj_start + 3:obj_start + 7]
        goal = current[:, goal_start:goal_start + 3]
        finger_mean = 0.5 * (current[:, 7:8] + current[:, 8:9])
        prev_finger_mean = 0.5 * (previous[:, 7:8] + previous[:, 8:9])
        at_goal, xy, zerr = self.score_at_goal(obj, goal)
        tcp_obj = (tcp - obj) / scale
        obj_goal = (obj - goal) / scale
        tcp_goal = (tcp - goal) / scale
        obj_vel = (obj - prev_obj) / fine
        tcp_vel = (tcp - prev_tcp) / fine
        tcp_obj_dist = torch.linalg.norm(tcp - obj, dim=-1, keepdim=True) / 0.30
        lifted = torch.sigmoid((obj[:, 2:3] - 0.050) / 0.020)
        finger_scaled = (finger_mean - 0.028) / 0.020
        finger_delta = (finger_mean - prev_finger_mean) / 0.010
        type_red = torch.ones_like(finger_mean) if type_index == 0 else torch.zeros_like(finger_mean)
        type_blue = 1.0 - type_red
        return torch.cat([
            tcp_obj, obj_goal, tcp_goal, obj_vel, tcp_vel, quat,
            xy / 0.30, zerr / 0.20, tcp_obj_dist, at_goal, lifted,
            finger_scaled, finger_delta, type_red, type_blue,
        ], dim=-1)

    def switch_features(self, current, previous):
        tcp = current[:, 18:21]
        red = current[:, 25:28]
        blue = current[:, 32:35]
        red_goal = current[:, 41:44]
        blue_goal = current[:, 44:47]
        finger_mean = 0.5 * (current[:, 7:8] + current[:, 8:9])
        prev_finger_mean = 0.5 * (previous[:, 7:8] + previous[:, 8:9])
        red_goal_score, red_xy, red_z = self.score_at_goal(red, red_goal)
        blue_goal_score, blue_xy, blue_zerr = self.score_at_goal(blue, blue_goal)
        open_score = torch.sigmoid((finger_mean - 0.030) / 0.004)
        closed_score = torch.sigmoid((0.026 - finger_mean) / 0.004)
        blue_lift = torch.sigmoid((blue[:, 2:3] - 0.050) / 0.020)
        red_lift = torch.sigmoid((red[:, 2:3] - 0.050) / 0.020)
        tcp_red = torch.linalg.norm(tcp - red, dim=-1, keepdim=True)
        tcp_blue = torch.linalg.norm(tcp - blue, dim=-1, keepdim=True)
        red_stable_delta = torch.linalg.norm(red - previous[:, 25:28], dim=-1, keepdim=True) / 0.050
        blue_delta = torch.linalg.norm(blue - previous[:, 32:35], dim=-1, keepdim=True) / 0.050
        finger_delta = (finger_mean - prev_finger_mean) / 0.010
        return torch.cat([
            red_xy / 0.30, red_z / 0.20, blue_xy / 0.40, blue_zerr / 0.25,
            red_goal_score, blue_goal_score, open_score, closed_score,
            blue_lift, red_lift, tcp_red / 0.40, tcp_blue / 0.40,
            (finger_mean - 0.028) / 0.020, finger_delta,
            red_stable_delta, blue_delta,
            (tcp[:, 2:3] - red[:, 2:3]) / 0.30,
            (tcp[:, 2:3] - blue[:, 2:3]) / 0.30,
        ], dim=-1)

    def encode(self, raw_history):
        norm_hist = self.norm_history(raw_history)
        hist_flat = norm_hist.reshape(raw_history.shape[0], -1)
        previous = raw_history[:, 0]
        current = raw_history[:, -1]
        red_feat = self.object_features(current, previous, 25, 41, 0)
        blue_feat = self.object_features(current, previous, 32, 44, 1)
        red_emb = self.object_encoder(red_feat)
        blue_emb = self.object_encoder(blue_feat)
        hist_emb = self.history_encoder(hist_flat)
        switch_feat = self.switch_features(current, previous)
        logits = self.attention_mlp(torch.cat([hist_flat, switch_feat], dim=-1))
        probs = torch.softmax(logits, dim=-1)
        attended = probs[:, 0:1] * red_emb + probs[:, 1:2] * blue_emb
        precondition = torch.cat([hist_emb, red_emb, blue_emb, attended, probs, switch_feat], dim=-1)
        condition = self.condition_projector(precondition)
        phase_logits = self.phase_head(precondition)
        return condition, logits, probs, phase_logits

    def forward(self, noisy_action, timestep, raw_history):
        condition, logits, probs, phase_logits = self.encode(raw_history)
        return self.backbone(noisy_action, timestep, condition)

    def auxiliary_predictions(self, raw_history):
        condition, logits, probs, phase_logits = self.encode(raw_history)
        return logits, probs, phase_logits


def soft_attention_targets(raw_history, future_obs=None, future_mask=None):
    current = raw_history[:, -1]
    red = current[:, 25:28]
    blue = current[:, 32:35]
    red_goal = current[:, 41:44]
    finger_mean = 0.5 * (current[:, 7:8] + current[:, 8:9])
    red_xy = torch.linalg.norm(red[:, :2] - red_goal[:, :2], dim=-1, keepdim=True)
    red_z = (red[:, 2:3] - red_goal[:, 2:3]).abs()
    red_at_goal = torch.sigmoid((0.055 - red_xy) / 0.012) * torch.sigmoid((0.030 - red_z) / 0.008)
    gripper_open = torch.sigmoid((finger_mean - 0.030) / 0.004)
    blue_lift = torch.sigmoid((blue[:, 2:3] - 0.050) / 0.020)
    current_blue_target = 1.0 - (1.0 - red_at_goal * gripper_open) * (1.0 - blue_lift)
    if future_obs is not None and future_mask is not None:
        mask = future_mask.to(device=current.device, dtype=current.dtype)
        f_red = future_obs[:, :, 25:28]
        f_blue = future_obs[:, :, 32:35]
        f_red_goal = future_obs[:, :, 41:44]
        f_finger = 0.5 * (future_obs[:, :, 7:8] + future_obs[:, :, 8:9])
        f_red_xy = torch.linalg.norm(f_red[:, :, :2] - f_red_goal[:, :, :2], dim=-1, keepdim=True)
        f_red_z = (f_red[:, :, 2:3] - f_red_goal[:, :, 2:3]).abs()
        f_red_at_goal = torch.sigmoid((0.055 - f_red_xy) / 0.012) * torch.sigmoid((0.030 - f_red_z) / 0.008)
        f_open = torch.sigmoid((f_finger - 0.030) / 0.004)
        f_blue_lift = torch.sigmoid((f_blue[:, :, 2:3] - 0.050) / 0.020)
        f_switch = 1.0 - (1.0 - f_red_at_goal * f_open) * (1.0 - f_blue_lift)
        f_switch = f_switch * mask
        valid = mask.sum(dim=1).clamp_min(1.0)
        future_max = f_switch.max(dim=1).values
        future_mean = f_switch.sum(dim=1) / valid
        current_blue_target = torch.maximum(current_blue_target, 0.65 * future_max + 0.20 * future_mean)
    return current_blue_target.clamp(0.0, 1.0)


def phase_targets(raw_history, future_obs=None, future_mask=None):
    current = raw_history[:, -1]
    red = current[:, 25:28]
    blue = current[:, 32:35]
    red_goal = current[:, 41:44]
    finger_mean = 0.5 * (current[:, 7:8] + current[:, 8:9])
    red_xy = torch.linalg.norm(red[:, :2] - red_goal[:, :2], dim=-1, keepdim=True)
    red_z = (red[:, 2:3] - red_goal[:, 2:3]).abs()
    red_done_open = (torch.sigmoid((0.055 - red_xy) / 0.012) *
                     torch.sigmoid((0.030 - red_z) / 0.008) *
                     torch.sigmoid((finger_mean - 0.030) / 0.004))
    blue_lift_now = torch.sigmoid((blue[:, 2:3] - 0.050) / 0.020)
    blue_lift_soon = blue_lift_now
    if future_obs is not None and future_mask is not None:
        mask = future_mask.to(device=current.device, dtype=current.dtype)
        f_blue = future_obs[:, :, 32:35]
        f_lift = torch.sigmoid((f_blue[:, :, 2:3] - 0.050) / 0.020) * mask
        blue_lift_soon = torch.maximum(blue_lift_soon, f_lift.max(dim=1).values)
    target_blue = soft_attention_targets(raw_history, future_obs, future_mask)
    return torch.cat([red_done_open, target_blue, blue_lift_soon], dim=-1).clamp(0.0, 1.0)


def build_model(spec):
    return SequentialObjectAttentionPolicy(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = public.epsilon_loss(prediction, batch["noise"], batch["mask"])
    att_logits, att_probs, phase_logits = model.auxiliary_predictions(batch["raw_obs"])
    future_obs = batch["future_obs"] if "future_obs" in batch else None
    future_mask = batch["future_mask"] if "future_mask" in batch else None
    blue_target = soft_attention_targets(batch["raw_obs"], future_obs, future_mask)
    red_target = 1.0 - blue_target
    log_probs = torch.log_softmax(att_logits, dim=-1)
    attention_ce = -(red_target * log_probs[:, 0:1] + blue_target * log_probs[:, 1:2]).mean()
    phase_target = phase_targets(batch["raw_obs"], future_obs, future_mask)
    phase_bce = torch.nn.functional.binary_cross_entropy_with_logits(phase_logits, phase_target)
    current = batch["raw_obs"][:, -1]
    red = current[:, 25:28]
    red_goal = current[:, 41:44]
    finger_mean = 0.5 * (current[:, 7:8] + current[:, 8:9])
    red_xy = torch.linalg.norm(red[:, :2] - red_goal[:, :2], dim=-1, keepdim=True)
    red_z = (red[:, 2:3] - red_goal[:, 2:3]).abs()
    red_done_open = (torch.sigmoid((0.055 - red_xy) / 0.012) *
                     torch.sigmoid((0.030 - red_z) / 0.008) *
                     torch.sigmoid((finger_mean - 0.030) / 0.004)).detach()
    consistency = -(red_done_open * torch.log(att_probs[:, 1:2].clamp_min(1e-6)) +
                    (1.0 - red_done_open) * torch.log(att_probs[:, 0:1].clamp_min(1e-6))).mean()
    prior_loss = 0.06 * attention_ce + 0.03 * phase_bce + 0.02 * consistency
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
