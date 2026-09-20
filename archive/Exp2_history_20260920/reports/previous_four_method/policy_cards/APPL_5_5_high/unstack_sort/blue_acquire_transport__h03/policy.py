import torch
import torch.nn.functional as F

from appl.public import DiffusionBackbone, epsilon_loss


def make_mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(torch.nn.SiLU())
    return torch.nn.Sequential(*layers)


class CompletedObjectInvariantDiffusion(torch.nn.Module):
    """Action diffusion model with a learned completed-red invariant prior."""

    def __init__(self, spec):
        super().__init__()
        normalizer = spec.get("normalizer", spec.get("shared_normalizer"))
        if normalizer is None:
            raise ValueError("spec must contain the shared normalizer")
        self.fields = {k: tuple(v) for k, v in spec["fields"].items()}
        training = spec["training"]
        self.horizon = int(training.get("horizon", 16))
        self.action_dim = 8
        self.obs_feature_dim = 149
        self.cond_dim = 256
        self.aux_channels = 10

        self.register_buffer("obs_mean", torch.tensor(normalizer["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.tensor(normalizer["std"], dtype=torch.float32).clamp_min(1.0e-6))
        self.register_buffer("action_min", torch.tensor(normalizer["action_min"], dtype=torch.float32))
        self.register_buffer("action_scale", torch.tensor(normalizer["action_scale"], dtype=torch.float32).clamp_min(1.0e-6))
        self.register_buffer("xyz_scale", torch.tensor([0.50, 0.50, 0.30], dtype=torch.float32))
        self.register_buffer("goal_tol", torch.tensor([0.06, 0.06, 0.011], dtype=torch.float32))

        self.obs_encoder = make_mlp([self.obs_feature_dim, 384, 384, self.cond_dim])
        self.invariant_encoder = make_mlp([10, 128, self.cond_dim])
        self.cond_norm = torch.nn.LayerNorm(self.cond_dim)
        self.backbone = DiffusionBackbone(self.cond_dim, training)
        self.action_encoder = make_mlp([self.horizon * self.action_dim, 256, 256])
        self.aux_head = make_mlp([self.cond_dim + 256, 512, 512, self.horizon * self.aux_channels])

    def field_slice(self, x, name):
        s, e = self.fields[name]
        return x[..., s:e]

    @staticmethod
    def safe_norm(x):
        return torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + 1.0e-8)

    def red_goal_score(self, red_xyz, red_goal):
        scaled = (red_xyz - red_goal) / self.goal_tol.to(device=red_xyz.device, dtype=red_xyz.dtype)
        return torch.exp(-torch.sum(scaled * scaled, dim=-1, keepdim=True))

    def make_features(self, raw_history):
        dtype = raw_history.dtype
        obs_mean = self.obs_mean.to(device=raw_history.device, dtype=dtype)
        obs_std = self.obs_std.to(device=raw_history.device, dtype=dtype)
        xyz_scale = self.xyz_scale.to(device=raw_history.device, dtype=dtype)

        norm_hist = (raw_history - obs_mean) / obs_std
        flat_norm = norm_hist.reshape(raw_history.shape[0], -1)
        prev = raw_history[:, 0]
        last = raw_history[:, -1]

        def xyz(state, name):
            return self.field_slice(state, name)[..., :3]

        prev_tcp = xyz(prev, "tcp_pose")
        prev_red = xyz(prev, "red_pose")
        prev_blue = xyz(prev, "blue_pose")
        last_tcp = xyz(last, "tcp_pose")
        last_red = xyz(last, "red_pose")
        last_blue = xyz(last, "blue_pose")
        red_goal = self.field_slice(last, "red_goal")
        blue_goal = self.field_slice(last, "blue_goal")

        def rel_pack(tcp, red, blue):
            rels = [
                (red - red_goal) / xyz_scale,
                (blue - blue_goal) / xyz_scale,
                (tcp - red) / xyz_scale,
                (tcp - blue) / xyz_scale,
                (blue - red) / xyz_scale,
                (tcp - blue_goal) / xyz_scale,
            ]
            return rels, torch.cat(rels, dim=-1)

        rels_prev, rel_prev = rel_pack(prev_tcp, prev_red, prev_blue)
        rels_last, rel_last = rel_pack(last_tcp, last_red, last_blue)
        deltas = torch.cat(
            [
                (last_tcp - prev_tcp) / xyz_scale,
                (last_red - prev_red) / xyz_scale,
                (last_blue - prev_blue) / xyz_scale,
            ],
            dim=-1,
        )
        norm_scalars = torch.cat([self.safe_norm(r) for r in rels_last], dim=-1)
        grip_mean = ((last[:, 7:9].mean(dim=-1, keepdim=True)) - 0.02) / 0.02
        grip_balance = (last[:, 7:8] - last[:, 8:9]) / 0.02
        red_score = self.red_goal_score(last_red, red_goal)
        blue_score = self.red_goal_score(last_blue, blue_goal)
        scalars = torch.cat([norm_scalars, grip_mean, grip_balance, red_score, blue_score], dim=-1)

        features = torch.cat([flat_norm, rel_prev, rel_last, deltas, scalars], dim=-1)
        invariant_features = torch.cat(
            [
                (last_red - red_goal) / xyz_scale,
                (last_tcp - last_red) / xyz_scale,
                (last_blue - last_red) / xyz_scale,
                red_score,
            ],
            dim=-1,
        )
        return features, invariant_features, red_score

    def encode_condition(self, raw_history):
        features, invariant_features, red_score = self.make_features(raw_history)
        base = self.obs_encoder(features)
        inv = self.invariant_encoder(invariant_features)
        cond = base + red_score.clamp(0.0, 1.0) * inv
        return self.cond_norm(cond)

    def forward(self, noisy_action, timestep, raw_history):
        condition = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, condition)

    def predict_aux(self, raw_history, encoded_action):
        condition = self.encode_condition(raw_history)
        bounded_action = 1.5 * torch.tanh(encoded_action / 1.5)
        action_feat = self.action_encoder(bounded_action.reshape(bounded_action.shape[0], -1))
        out = self.aux_head(torch.cat([condition, action_feat], dim=-1))
        out = out.reshape(encoded_action.shape[0], self.horizon, self.aux_channels)
        return {
            "red_delta": out[..., 0:3],
            "tcp_delta": out[..., 3:6],
            "blue_to_goal": out[..., 6:9],
            "red_goal_logit": out[..., 9:10],
        }


def build_model(spec):
    return CompletedObjectInvariantDiffusion(spec)


def masked_mse(pred, target, mask):
    denom = (mask.sum() * pred.shape[-1]).clamp_min(1.0)
    return (((pred - target) ** 2) * mask).sum() / denom


def masked_bce_with_logits(logits, target, mask):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def compute_loss(model, batch, spec):
    raw_obs = batch["raw_obs"]
    noisy_action = batch["noisy_action"]
    timesteps = batch["timesteps"]
    noise = batch["noise"]
    action_mask = batch["mask"]

    pred_noise = model(noisy_action, timesteps, raw_obs)
    diffusion_loss = epsilon_loss(pred_noise, noise, action_mask)

    alpha_bar = batch["alpha_bar"].to(device=noisy_action.device, dtype=noisy_action.dtype).reshape(-1, 1, 1)
    sqrt_alpha = torch.sqrt(alpha_bar.clamp_min(1.0e-8))
    sqrt_one_minus = torch.sqrt((1.0 - alpha_bar).clamp_min(0.0))
    clean_estimate = (noisy_action - sqrt_one_minus * pred_noise) / sqrt_alpha

    future = batch["future_obs"]
    future_mask = batch["future_mask"]
    current = raw_obs[:, -1]
    xyz_scale = model.xyz_scale.to(device=raw_obs.device, dtype=raw_obs.dtype)
    goal_tol = model.goal_tol.to(device=raw_obs.device, dtype=raw_obs.dtype)

    def xyz(state, name):
        return model.field_slice(state, name)[..., :3]

    current_red = xyz(current, "red_pose")
    current_tcp = xyz(current, "tcp_pose")
    red_goal = model.field_slice(current, "red_goal")
    blue_goal = model.field_slice(current, "blue_goal")
    future_red = xyz(future, "red_pose")
    future_tcp = xyz(future, "tcp_pose")
    future_blue = xyz(future, "blue_pose")

    target_red_delta = (future_red - current_red[:, None, :]) / xyz_scale
    target_tcp_delta = (future_tcp - current_tcp[:, None, :]) / xyz_scale
    target_blue_to_goal = (future_blue - blue_goal[:, None, :]) / xyz_scale
    red_err = (future_red - red_goal[:, None, :]) / goal_tol
    red_goal_target = (torch.sum(red_err * red_err, dim=-1, keepdim=True) < 3.0).to(dtype=raw_obs.dtype)

    teacher_aux = model.predict_aux(raw_obs, batch["encoded_action"])
    red_sup = masked_mse(teacher_aux["red_delta"], target_red_delta, future_mask)
    tcp_sup = masked_mse(teacher_aux["tcp_delta"], target_tcp_delta, future_mask)
    blue_sup = masked_mse(teacher_aux["blue_to_goal"], target_blue_to_goal, future_mask)
    red_goal_sup = masked_bce_with_logits(teacher_aux["red_goal_logit"], red_goal_target, future_mask)

    prior_aux = model.predict_aux(raw_obs, clean_estimate)
    red_static = masked_mse(prior_aux["red_delta"], torch.zeros_like(prior_aux["red_delta"]), future_mask)
    red_goal_keep = masked_bce_with_logits(
        prior_aux["red_goal_logit"], torch.ones_like(prior_aux["red_goal_logit"]), future_mask
    )

    tcp_abs = current_tcp[:, None, :] + prior_aux["tcp_delta"] * xyz_scale
    blue_abs = blue_goal[:, None, :] + prior_aux["blue_to_goal"] * xyz_scale
    red_abs = current_red[:, None, :]

    tcp_xy_dist = torch.sqrt(torch.sum((tcp_abs[..., :2] - red_abs[..., :2]) ** 2, dim=-1, keepdim=True) + 1.0e-8)
    tcp_z_rel = tcp_abs[..., 2:3] - red_abs[..., 2:3]
    tcp_low_gate = torch.sigmoid((0.10 - tcp_z_rel) * 25.0)
    tcp_intrusion = F.softplus((0.08 - tcp_xy_dist) * 30.0) / 30.0
    tcp_clearance = ((tcp_intrusion ** 2) * tcp_low_gate * future_mask).sum() / future_mask.sum().clamp_min(1.0)

    blue_xy_dist = torch.sqrt(torch.sum((blue_abs[..., :2] - red_abs[..., :2]) ** 2, dim=-1, keepdim=True) + 1.0e-8)
    blue_z_rel = blue_abs[..., 2:3] - red_abs[..., 2:3]
    blue_low_gate = torch.sigmoid((0.08 - blue_z_rel) * 25.0)
    blue_intrusion = F.softplus((0.07 - blue_xy_dist) * 30.0) / 30.0
    blue_clearance = ((blue_intrusion ** 2) * blue_low_gate * future_mask).sum() / future_mask.sum().clamp_min(1.0)

    prior_loss = (
        0.08 * (red_sup + tcp_sup + blue_sup)
        + 0.02 * red_goal_sup
        + 0.05 * red_static
        + 0.01 * red_goal_keep
        + 0.02 * tcp_clearance
        + 0.01 * blue_clearance
    )
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
