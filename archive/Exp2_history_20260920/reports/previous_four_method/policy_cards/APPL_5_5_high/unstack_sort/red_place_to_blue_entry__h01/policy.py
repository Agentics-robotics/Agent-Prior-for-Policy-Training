import torch
import appl.public as public


class RedGoalRelativeDiffusionPolicy(torch.nn.Module):
    """Goal-relative learned action diffusion model for red placement and blue-entry transition."""

    def __init__(self, spec):
        super().__init__()
        cfg = spec.get("candidate_config", {})
        self.horizon = int(spec["training"].get("horizon", 16))
        self.condition_dim = int(cfg.get("condition_dim", 256))
        self.relative_scale_m = float(cfg.get("relative_scale_m", 0.25))
        self.delta_scale_m = float(cfg.get("delta_scale_m", 0.05))
        self.finger_scale_m = float(cfg.get("finger_scale_m", 0.08))

        n = spec["normalizer"]
        self.register_buffer("obs_mean", torch.tensor(n["mean"], dtype=torch.float32), persistent=False)
        self.register_buffer("obs_std", torch.tensor(n["std"], dtype=torch.float32), persistent=False)

        # Full normalized two-step observation (2*47) plus hand-authored causal relative features.
        # Per history step: seven 3-D relative vectors, four scalar distances/heights, and
        # four gripper/object-contact scalars = 29 features. Current-minus-previous deltas add
        # three 3-D motion vectors, one finger-width change, and one 3-D goal-error change = 13.
        self.relative_feature_dim_per_step = 29
        self.delta_feature_dim = 13
        feature_dim = 2 * 47 + 2 * self.relative_feature_dim_per_step + self.delta_feature_dim
        hidden = int(cfg.get("encoder_hidden_dim", 256))
        self.feature_norm = torch.nn.LayerNorm(feature_dim)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden),
            torch.nn.Mish(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Mish(),
            torch.nn.Linear(hidden, self.condition_dim),
        )
        self.condition_norm = torch.nn.LayerNorm(self.condition_dim)
        self.backbone = public.DiffusionBackbone(self.condition_dim, spec["training"])

        aux_hidden = int(cfg.get("aux_hidden_dim", 128))
        # Predicts future red-goal error and phase labels from the causal condition. This head is
        # training-only, but it shares the encoder used by the diffusion model.
        self.aux_cond_head = torch.nn.Sequential(
            torch.nn.Linear(self.condition_dim, aux_hidden),
            torch.nn.Mish(),
            torch.nn.Linear(aux_hidden, self.horizon * 5),
        )
        # A second auxiliary head reads the denoised action estimate, giving the prior loss a direct
        # differentiable path to the epsilon prediction while remaining a learned diffusion model.
        self.action_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.horizon * 8, aux_hidden),
            torch.nn.Mish(),
            torch.nn.Linear(aux_hidden, aux_hidden),
            torch.nn.Mish(),
        )
        self.aux_action_head = torch.nn.Sequential(
            torch.nn.Linear(self.condition_dim + aux_hidden, aux_hidden),
            torch.nn.Mish(),
            torch.nn.Linear(aux_hidden, self.horizon * 5),
        )

    def norm_obs(self, raw_history):
        return (raw_history - self.obs_mean.to(raw_history.device, raw_history.dtype)) / self.obs_std.to(raw_history.device, raw_history.dtype)

    def relative_one(self, obs):
        # Fixed M1_v2 state schema, world frame, metres and wxyz quaternions.
        qpos = obs[:, 0:9]
        tcp = obs[:, 18:21]
        red = obs[:, 25:28]
        blue = obs[:, 32:35]
        red_goal = obs[:, 41:44]
        blue_goal = obs[:, 44:47]
        rs = self.relative_scale_m
        fs = self.finger_scale_m

        e_red_goal = (red_goal - red) / rs
        e_blue_goal = (blue_goal - blue) / rs
        e_tcp_red = (tcp - red) / rs
        e_tcp_blue = (tcp - blue) / rs
        e_red_blue = (red - blue) / rs
        e_tcp_red_goal = (tcp - red_goal) / rs
        e_tcp_blue_goal = (tcp - blue_goal) / rs

        red_xy_dist = torch.linalg.norm(red_goal[:, 0:2] - red[:, 0:2], dim=-1, keepdim=True) / rs
        blue_xy_dist = torch.linalg.norm(blue_goal[:, 0:2] - blue[:, 0:2], dim=-1, keepdim=True) / rs
        red_z_err = (red_goal[:, 2:3] - red[:, 2:3]) / rs
        blue_z_err = (blue_goal[:, 2:3] - blue[:, 2:3]) / rs
        finger_width = (qpos[:, 7:8] + qpos[:, 8:9]) / fs
        finger_mean = ((qpos[:, 7:8] + qpos[:, 8:9]) * 0.5) / fs
        # Smooth causal phase indicators, not gates or controllers.
        red_near_goal_soft = torch.exp(-4.0 * red_xy_dist.clamp_max(4.0))
        red_high_soft = torch.sigmoid((red[:, 2:3] - red_goal[:, 2:3] - 0.04) / 0.02)

        return torch.cat([
            e_red_goal, e_blue_goal, e_tcp_red, e_tcp_blue, e_red_blue,
            e_tcp_red_goal, e_tcp_blue_goal,
            red_xy_dist, blue_xy_dist, red_z_err, blue_z_err,
            finger_width, finger_mean, red_near_goal_soft, red_high_soft,
        ], dim=-1)

    def make_features(self, raw_history):
        norm_hist = self.norm_obs(raw_history).reshape(raw_history.shape[0], -1)
        prev = raw_history[:, 0]
        curr = raw_history[:, -1]
        rel_prev = self.relative_one(prev)
        rel_curr = self.relative_one(curr)

        ds = self.delta_scale_m
        red_delta = (curr[:, 25:28] - prev[:, 25:28]) / ds
        blue_delta = (curr[:, 32:35] - prev[:, 32:35]) / ds
        tcp_delta = (curr[:, 18:21] - prev[:, 18:21]) / ds
        # Finger-width and current red-goal-error changes help disambiguate closing/lifting,
        # placing/releasing, and the subsequent open-gripper blue approach.
        finger_delta = ((curr[:, 7:9] - prev[:, 7:9]).sum(dim=-1, keepdim=True)) / 0.01
        red_goal_delta = ((curr[:, 41:44] - curr[:, 25:28]) - (prev[:, 41:44] - prev[:, 25:28])) / ds
        delta = torch.cat([red_delta, blue_delta, tcp_delta, finger_delta, red_goal_delta], dim=-1)
        feat = torch.cat([norm_hist, rel_prev, rel_curr, delta], dim=-1)
        return self.feature_norm(feat)

    def condition(self, raw_history):
        cond = self.encoder(self.make_features(raw_history))
        return self.condition_norm(cond)

    def forward(self, noisy_action, timestep, raw_history):
        cond = self.condition(raw_history)
        return self.backbone(noisy_action, timestep, cond)

    def predict_aux(self, raw_history, denoised_action_estimate=None):
        cond = self.condition(raw_history)
        cond_pred = self.aux_cond_head(cond).view(raw_history.shape[0], self.horizon, 5)
        action_pred = None
        if denoised_action_estimate is not None:
            # Bound the noisy clean estimate before the auxiliary action encoder. This is not used
            # for sampling; it only stabilizes the training-only prior objective.
            a = torch.tanh(denoised_action_estimate).reshape(raw_history.shape[0], -1)
            af = self.action_encoder(a)
            action_pred = self.aux_action_head(torch.cat([cond, af], dim=-1)).view(raw_history.shape[0], self.horizon, 5)
        return cond_pred, action_pred


def build_model(spec):
    return RedGoalRelativeDiffusionPolicy(spec)


def masked_huber(error, mask, delta=0.1):
    abs_err = error.abs()
    quad = torch.minimum(abs_err, torch.tensor(delta, device=error.device, dtype=error.dtype))
    lin = abs_err - quad
    loss = 0.5 * quad.square() + delta * lin
    denom = (mask.sum() * error.shape[-1]).clamp_min(1.0)
    return (loss * mask).sum() / denom


def masked_bce_with_logits(logits, target, mask):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * mask.squeeze(-1)).sum() / mask.sum().clamp_min(1.0)


def compute_loss(model, batch, spec):
    pred_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = public.epsilon_loss(pred_noise, batch["noise"], batch["mask"])

    cfg = spec.get("candidate_config", {})
    future_weight = float(cfg.get("future_aux_weight", 0.05))
    action_weight = float(cfg.get("action_aux_weight", 0.02))
    phase_weight = float(cfg.get("phase_aux_weight", 0.25))
    open_weight = float(cfg.get("open_aux_weight", 0.10))
    rel_scale = float(cfg.get("relative_scale_m", 0.25))

    future = batch["future_obs"]
    future_mask = batch.get("future_mask", batch["mask"])
    red_future = future[:, :, 25:28]
    red_goal_future = future[:, :, 41:44]
    red_err_target = (red_goal_future - red_future) / rel_scale
    red_abs_err = (red_goal_future - red_future).abs()
    # Contract-informed label used only as a training target for the learned phase head.
    red_at_goal = ((red_abs_err[:, :, 0] <= 0.04) &
                   (red_abs_err[:, :, 1] <= 0.04) &
                   (red_abs_err[:, :, 2] <= 0.011)).to(future.dtype)
    finger_width = future[:, :, 7] + future[:, :, 8]
    gripper_open = (finger_width >= 0.06).to(future.dtype)

    cond_pred, unused_action_pred = model.predict_aux(batch["raw_obs"], None)
    cond_offset = cond_pred[:, :, 0:3]
    cond_goal_logit = cond_pred[:, :, 3]
    cond_open_logit = cond_pred[:, :, 4]
    cond_loss = masked_huber(cond_offset - red_err_target, future_mask, delta=0.15)
    cond_loss = cond_loss + phase_weight * masked_bce_with_logits(cond_goal_logit, red_at_goal, future_mask)
    cond_loss = cond_loss + open_weight * masked_bce_with_logits(cond_open_logit, gripper_open, future_mask)

    alpha_bar = batch["alpha_bar"].to(pred_noise.dtype).view(-1, 1, 1).clamp_min(1.0e-4)
    x0_est = (batch["noisy_action"] - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
    unused_cond_pred, action_pred = model.predict_aux(batch["raw_obs"], x0_est)
    act_offset = action_pred[:, :, 0:3]
    act_goal_logit = action_pred[:, :, 3]
    act_open_logit = action_pred[:, :, 4]
    # Low-alpha clean estimates are intentionally downweighted to avoid high-noise amplification.
    weighted_mask = future_mask * torch.sqrt(alpha_bar).detach()
    action_loss = masked_huber(act_offset - red_err_target, weighted_mask, delta=0.15)
    action_loss = action_loss + phase_weight * masked_bce_with_logits(act_goal_logit, red_at_goal, weighted_mask)
    action_loss = action_loss + open_weight * masked_bce_with_logits(act_open_logit, gripper_open, weighted_mask)

    prior_loss = future_weight * cond_loss + action_weight * action_loss
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
