import math
import torch
import appl.public as public


class DrawerFrameBlueInsertionPolicy(torch.nn.Module):
    """Learned DDPM epsilon model with drawer-frame blue-object conditioning.

    The model remains an action diffusion policy: forward() predicts diffusion
    noise for normalized Panda joint/gripper action sequences.  The prior is
    implemented as trainable conditioning and an auxiliary trainable future-state
    predictor that uses the denoised action estimate during training.
    """

    def __init__(self, spec):
        super().__init__()
        cfg = spec.get("candidate_config", {})
        training = spec["training"]
        normalizer = spec["normalizer"]

        self.condition_dim = int(cfg.get("condition_dim", 256))
        self.prior_loss_weight = float(cfg.get("future_blue_aux_weight", 0.05))
        self.inside_loss_weight = float(cfg.get("inside_bce_weight", 0.15))
        self.clean_action_bound = float(cfg.get("aux_clean_action_bound", 1.5))

        self.register_buffer("obs_mean", torch.tensor(normalizer["mean"], dtype=torch.float32), persistent=False)
        self.register_buffer("obs_std", torch.tensor(normalizer["std"], dtype=torch.float32), persistent=False)
        self.register_buffer("action_min", torch.tensor(normalizer["action_min"], dtype=torch.float32), persistent=False)
        self.register_buffer("action_scale", torch.tensor(normalizer["action_scale"], dtype=torch.float32), persistent=False)

        # Fixed task geometry, in metres, from the assigned completion contract.
        self.register_buffer("drawer_origin", torch.tensor([0.19, 0.0, 0.035], dtype=torch.float32), persistent=False)
        self.register_buffer("blue_aux_scale", torch.tensor([0.172, 0.182, 0.15], dtype=torch.float32), persistent=False)
        self.register_buffer("rel_scale_20", torch.tensor([0.20, 0.20, 0.20], dtype=torch.float32), persistent=False)
        self.register_buffer("rel_scale_30", torch.tensor([0.30, 0.30, 0.30], dtype=torch.float32), persistent=False)
        self.register_buffer("red_scale", torch.tensor([0.12, 0.12, 0.10], dtype=torch.float32), persistent=False)
        self.register_buffer("cavity_half", torch.tensor([0.172, 0.182], dtype=torch.float32), persistent=False)
        self.object_half = 0.02

        # Two normalized observations (2*47) plus explicit causal drawer-frame
        # geometric features.  The 78 prior features are constructed in
        # drawer_frame_features().
        condition_input_dim = 2 * 47 + 78
        hidden = int(cfg.get("condition_hidden", 384))
        self.condition_encoder = torch.nn.Sequential(
            torch.nn.Linear(condition_input_dim, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, self.condition_dim),
            torch.nn.LayerNorm(self.condition_dim),
            torch.nn.SiLU(),
        )

        self.backbone = public.DiffusionBackbone(self.condition_dim, training)

        # Auxiliary action-conditioned predictor.  It is used only for training
        # loss; deployment still calls forward() and samples actions with DDPM.
        aux_dim = int(cfg.get("auxiliary_hidden", 96))
        self.action_aux_encoder = torch.nn.Sequential(
            torch.nn.Linear(9, aux_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(aux_dim, aux_dim),
            torch.nn.SiLU(),
        )
        self.cond_aux_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.condition_dim, aux_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(aux_dim, aux_dim),
            torch.nn.SiLU(),
        )
        self.future_blue_head = torch.nn.Sequential(
            torch.nn.Linear(aux_dim * 3 + 1, aux_dim * 2),
            torch.nn.SiLU(),
            torch.nn.Linear(aux_dim * 2, aux_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(aux_dim, 4),
        )

    def normalize_observation_local(self, raw_history):
        mean = self.obs_mean.to(device=raw_history.device, dtype=raw_history.dtype)
        std = self.obs_std.to(device=raw_history.device, dtype=raw_history.dtype)
        return (raw_history - mean) / std

    def drawer_center(self, obs):
        # obs shape [...,47]. Drawer center expression: [0.19-drawer_position, 0, 0.035].
        drawer_pos = obs[..., 39:40]
        x = self.drawer_origin[0].to(obs) - drawer_pos
        y = torch.zeros_like(x)
        z = torch.full_like(x, 0.035)
        return torch.cat([x, y, z], dim=-1)

    def normalize_quat(self, q):
        return q / q.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)

    def xy_extent_for_block(self, quat):
        # Axis-aligned XY half-extent of a cube with half-size 0.02 m under a
        # wxyz quaternion.  This is used for predicates/features, not as a hard controller.
        q = self.normalize_quat(quat)
        w, x, y, z = q.unbind(dim=-1)
        r00 = 1.0 - 2.0 * (y * y + z * z)
        r01 = 2.0 * (x * y - z * w)
        r02 = 2.0 * (x * z + y * w)
        r10 = 2.0 * (x * y + z * w)
        r11 = 1.0 - 2.0 * (x * x + z * z)
        r12 = 2.0 * (y * z - x * w)
        ex = self.object_half * (r00.abs() + r01.abs() + r02.abs())
        ey = self.object_half * (r10.abs() + r11.abs() + r12.abs())
        return torch.stack([ex, ey], dim=-1)

    def inside_margins(self, obs):
        blue_xyz = obs[..., 32:35]
        blue_quat = obs[..., 35:39]
        center = self.drawer_center(obs)
        rel_xy = blue_xyz[..., 0:2] - center[..., 0:2]
        extent_xy = self.xy_extent_for_block(blue_quat)
        half = self.cavity_half.to(device=obs.device, dtype=obs.dtype)
        xy_margin = (half - (rel_xy.abs() + extent_xy)) / half
        z = blue_xyz[..., 2:3]
        z_low = (z - 0.053) / 0.10
        z_high = (0.074 - z) / 0.10
        return torch.cat([xy_margin, z_low, z_high], dim=-1)

    def drawer_frame_features(self, raw_history):
        # raw_history shape [B,2,47].  All scales are task-geometry or broad
        # shared-workspace scales, not per-skill fitted statistics.
        obs = raw_history
        tcp = obs[..., 18:21]
        red = obs[..., 25:28]
        red_quat = obs[..., 28:32]
        blue = obs[..., 32:35]
        drawer_pos = obs[..., 39:40]
        drawer_vel = obs[..., 40:41]
        red_goal = obs[..., 41:44]
        blue_goal = obs[..., 44:47]
        qpos = obs[..., 0:9]
        center = self.drawer_center(obs)

        rel20 = self.rel_scale_20.to(device=obs.device, dtype=obs.dtype)
        rel30 = self.rel_scale_30.to(device=obs.device, dtype=obs.dtype)
        red_scale = self.red_scale.to(device=obs.device, dtype=obs.dtype)

        blue_rel_drawer = (blue - center) / rel20
        blue_rel_goal = (blue - blue_goal) / rel20
        tcp_rel_blue = (tcp - blue) / rel20
        tcp_rel_drawer = (tcp - center) / rel30
        red_rel_pad = (red - red_goal) / red_scale
        goal_rel_drawer = (blue_goal - center) / rel20

        finger_left = qpos[..., 7:8]
        finger_right = qpos[..., 8:9]
        finger_width = finger_left + finger_right
        finger_balance = finger_left - finger_right
        fingers = torch.cat([finger_width / 0.08, finger_balance / 0.02], dim=-1)

        drawer_features = torch.cat([(drawer_pos - 0.26) / 0.15, drawer_vel / 0.15], dim=-1)
        inside_margins = self.inside_margins(obs)

        red_extent = self.xy_extent_for_block(red_quat)
        red_xy_margin = (torch.tensor([0.06, 0.06], device=obs.device, dtype=obs.dtype)
                         - ((red[..., 0:2] - red_goal[..., 0:2]).abs() + red_extent)) / 0.06
        red_z_margin = torch.minimum(red[..., 2:3] - 0.014, 0.031 - red[..., 2:3]) / 0.02
        red_margins = torch.cat([red_xy_margin, red_z_margin], dim=-1)

        tcp_blue_dist = (tcp - blue).norm(dim=-1, keepdim=True) / 0.05
        close_width = (finger_width - 0.036) / 0.04
        attach_features = torch.cat([tcp_blue_dist, close_width], dim=-1)

        per_step = torch.cat([
            blue_rel_drawer,
            blue_rel_goal,
            tcp_rel_blue,
            tcp_rel_drawer,
            red_rel_pad,
            goal_rel_drawer,
            fingers,
            drawer_features,
            inside_margins,
            red_margins,
            attach_features,
        ], dim=-1)  # [B,2,31]

        first = obs[:, 0]
        last = obs[:, 1]
        delta_tcp = (last[:, 18:21] - first[:, 18:21]) / rel20
        delta_blue = (last[:, 32:35] - first[:, 32:35]) / rel20
        delta_drawer = (last[:, 39:40] - first[:, 39:40]) / 0.10
        delta_fingers = (last[:, 7:9] - first[:, 7:9]) / 0.04
        delta_qpos = last[:, 0:7] - first[:, 0:7]
        delta = torch.cat([delta_tcp, delta_blue, delta_drawer, delta_fingers, delta_qpos], dim=-1)  # [B,16]

        return torch.cat([per_step.reshape(obs.shape[0], -1), delta], dim=-1)

    def encode_condition(self, raw_history):
        norm = self.normalize_observation_local(raw_history).reshape(raw_history.shape[0], -1)
        geom = self.drawer_frame_features(raw_history)
        return self.condition_encoder(torch.cat([norm, geom], dim=-1))

    def forward(self, noisy_action, timestep, raw_history):
        condition = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, condition)

    def predict_future_blue(self, clean_action_encoded, raw_history):
        # Predict scaled drawer-frame future blue xyz and an inside-cavity logit
        # from the denoised action estimate.  This supplies the prior loss with
        # gradients to the denoiser through clean_action_encoded.
        condition = self.encode_condition(raw_history)
        bsz, horizon, _ = clean_action_encoded.shape
        dtype = clean_action_encoded.dtype
        device = clean_action_encoded.device
        if horizon > 1:
            step_fraction = torch.arange(horizon, device=device, dtype=dtype).view(1, horizon, 1) / float(horizon - 1)
        else:
            step_fraction = torch.zeros((1, horizon, 1), device=device, dtype=dtype)
        step_fraction = step_fraction.expand(bsz, horizon, 1)
        action_in = torch.cat([clean_action_encoded, step_fraction], dim=-1)
        action_latent = self.action_aux_encoder(action_in)
        cumulative_action_latent = torch.cumsum(action_latent, dim=1)
        cond_latent = self.cond_aux_encoder(condition).unsqueeze(1).expand(-1, horizon, -1)
        head_in = torch.cat([cond_latent, action_latent, cumulative_action_latent, step_fraction], dim=-1)
        return self.future_blue_head(head_in)

    def future_targets(self, future_obs):
        center = self.drawer_center(future_obs)
        blue = future_obs[..., 32:35]
        scale = self.blue_aux_scale.to(device=future_obs.device, dtype=future_obs.dtype)
        rel = (blue - center) / scale
        margins = self.inside_margins(future_obs)
        inside = ((margins[..., 0:1] > 0.0) &
                  (margins[..., 1:2] > 0.0) &
                  (margins[..., 2:3] > 0.0) &
                  (margins[..., 3:4] > 0.0)).to(dtype=future_obs.dtype)
        return rel, inside


def build_model(spec):
    return DrawerFrameBlueInsertionPolicy(spec)


def compute_loss(model, batch, spec):
    predicted_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = public.epsilon_loss(predicted_noise, batch["noise"], batch["mask"])

    alpha_bar = batch["alpha_bar"].to(device=predicted_noise.device, dtype=predicted_noise.dtype).view(-1, 1, 1)
    sqrt_alpha = alpha_bar.sqrt().clamp_min(1.0e-6)
    sqrt_one_minus = (1.0 - alpha_bar).clamp_min(0.0).sqrt()
    clean_estimate = (batch["noisy_action"] - sqrt_one_minus * predicted_noise) / sqrt_alpha
    # Keep auxiliary gradients numerically bounded at very noisy timesteps.
    bound = model.clean_action_bound
    clean_for_aux = torch.tanh(clean_estimate / bound) * bound

    aux_pred = model.predict_future_blue(clean_for_aux, batch["raw_obs"])
    target_rel, target_inside = model.future_targets(batch["future_obs"])

    mask = batch["mask"]
    if "future_mask" in batch:
        mask = mask * batch["future_mask"]
    noise_weight = alpha_bar.clamp(0.05, 1.0)
    weighted_mask = mask.to(dtype=predicted_noise.dtype) * noise_weight

    rel_loss_raw = torch.nn.functional.smooth_l1_loss(aux_pred[..., 0:3], target_rel, beta=0.5, reduction="none")
    rel_loss = (rel_loss_raw * weighted_mask).sum() / (weighted_mask.sum() * 3.0).clamp_min(1.0)

    bce_raw = torch.nn.functional.binary_cross_entropy_with_logits(aux_pred[..., 3:4], target_inside, reduction="none")
    inside_loss = (bce_raw * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)

    prior_loss = model.prior_loss_weight * (rel_loss + model.inside_loss_weight * inside_loss)
    total_loss = diffusion_loss + prior_loss
    return {
        "loss": total_loss,
        "diffusion_loss": diffusion_loss,
        "prior_loss": prior_loss,
    }
