import torch
import appl.public as public


class GoalAnchoredBluePlacementPolicy(torch.nn.Module):
    """Learned diffusion policy with explicit goal-relative blue placement features."""

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        cfg = spec.get("candidate_config", {})
        if "config" in cfg and isinstance(cfg.get("config"), dict):
            cfg = cfg["config"]
        self.config = cfg
        self.obs_dim = int(spec.get("observation_dimension", 47))
        self.history = int(spec.get("training", {}).get("observation_steps", 2))
        self.horizon = int(spec.get("training", {}).get("horizon", 16))
        self.cond_dim = int(cfg.get("condition_dim", 256))
        hidden = int(cfg.get("encoder_hidden", 256))

        derived_per_step = 26
        encoder_in = self.history * self.obs_dim + self.obs_dim + self.history * derived_per_step
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(encoder_in, hidden),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, self.cond_dim),
        )
        self.backbone = public.DiffusionBackbone(self.cond_dim, spec["training"])

        aux_hidden = int(cfg.get("aux_hidden", 192))
        aux_in = self.cond_dim + 24
        self.aux_trunk = torch.nn.Sequential(
            torch.nn.Linear(aux_in, aux_hidden),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(aux_hidden),
            torch.nn.Linear(aux_hidden, aux_hidden),
            torch.nn.SiLU(),
        )
        self.final_blue_error_head = torch.nn.Linear(aux_hidden, 3)
        self.min_distance_head = torch.nn.Linear(aux_hidden, 1)
        self.success_logit_head = torch.nn.Linear(aux_hidden, 1)

    def field_range(self, name):
        f = self.spec["fields"][name]
        return int(f[0]), int(f[1])

    def metric_scales(self, device, dtype):
        pos = torch.tensor([0.35, 0.45, 0.30], device=device, dtype=dtype)
        z = torch.tensor([0.30], device=device, dtype=dtype)
        return pos, z

    def derived_step_features(self, raw_step):
        dtype = raw_step.dtype
        device = raw_step.device
        q0, q1 = self.field_range("qpos")
        tcp0, tcp1 = self.field_range("tcp_pose")
        red0, red1 = self.field_range("red_pose")
        blue0, blue1 = self.field_range("blue_pose")
        rg0, rg1 = self.field_range("red_goal")
        bg0, bg1 = self.field_range("blue_goal")
        qpos = raw_step[:, q0:q1]
        tcp = raw_step[:, tcp0:tcp1]
        red = raw_step[:, red0:red1]
        blue = raw_step[:, blue0:blue1]
        red_goal = raw_step[:, rg0:rg1]
        blue_goal = raw_step[:, bg0:bg1]
        tcp_xyz = tcp[:, :3]
        red_xyz = red[:, :3]
        blue_xyz = blue[:, :3]
        pos_scale, z_scale = self.metric_scales(device, dtype)

        blue_err = (blue_xyz - blue_goal) / pos_scale
        tcp_goal_err = (tcp_xyz - blue_goal) / pos_scale
        tcp_blue_err = (tcp_xyz - blue_xyz) / pos_scale
        red_goal_err = (red_xyz - red_goal) / pos_scale
        tcp_red_err = (tcp_xyz - red_xyz) / pos_scale
        blue_red_rel = (blue_xyz - red_xyz) / pos_scale

        finger_width = (qpos[:, 7:8] + qpos[:, 8:9]) / torch.tensor(0.08, device=device, dtype=dtype)
        tcp_above_blue = (tcp_xyz[:, 2:3] - blue_xyz[:, 2:3]) / z_scale
        tcp_above_goal = (tcp_xyz[:, 2:3] - blue_goal[:, 2:3]) / z_scale
        contain = torch.cat([
            torch.abs(blue_xyz[:, 0:1] - blue_goal[:, 0:1]) / torch.tensor(0.04, device=device, dtype=dtype),
            torch.abs(blue_xyz[:, 1:2] - blue_goal[:, 1:2]) / torch.tensor(0.04, device=device, dtype=dtype),
            torch.abs(blue_xyz[:, 2:3] - blue_goal[:, 2:3]) / torch.tensor(0.011, device=device, dtype=dtype),
        ], dim=-1)
        xy_dist = torch.linalg.norm((blue_xyz[:, :2] - blue_goal[:, :2]) / torch.tensor(0.06, device=device, dtype=dtype), dim=-1, keepdim=True)
        at_goal = ((contain[:, 0:1] <= 1.0) & (contain[:, 1:2] <= 1.0) & (contain[:, 2:3] <= 1.0)).to(dtype)
        return torch.cat([
            blue_err, tcp_goal_err, tcp_blue_err, red_goal_err, tcp_red_err,
            blue_red_rel, finger_width, tcp_above_blue, tcp_above_goal,
            contain, xy_dist, at_goal,
        ], dim=-1)

    def encode_condition(self, raw_history):
        norm = public.normalize_observation(raw_history, self.spec)
        flat_norm = norm.reshape(norm.shape[0], -1)
        delta = norm[:, -1, :] - norm[:, 0, :]
        derived = []
        for i in range(raw_history.shape[1]):
            derived.append(self.derived_step_features(raw_history[:, i, :]))
        x = torch.cat([flat_norm, delta] + derived, dim=-1)
        return self.encoder(x)

    def forward(self, noisy_action, timestep, raw_history):
        condition = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, condition)

    def auxiliary_predictions(self, noisy_action, timestep, raw_history, predicted_noise, alpha_bar, action_mask=None):
        condition = self.encode_condition(raw_history)
        if alpha_bar is None:
            alpha = torch.ones(noisy_action.shape[0], device=noisy_action.device, dtype=noisy_action.dtype)
        else:
            alpha = alpha_bar.to(device=noisy_action.device, dtype=noisy_action.dtype).reshape(-1)
        sqrt_alpha = torch.sqrt(alpha.clamp_min(1.0e-5)).view(-1, 1, 1)
        sqrt_one_minus = torch.sqrt((1.0 - alpha).clamp_min(0.0)).view(-1, 1, 1)
        x0_est = (noisy_action - sqrt_one_minus * predicted_noise) / sqrt_alpha
        x0_est = torch.clamp(x0_est, -2.0, 2.0)
        if action_mask is None:
            mean_action = x0_est.mean(dim=1)
        else:
            m = action_mask.to(dtype=x0_est.dtype, device=x0_est.device)
            mean_action = (x0_est * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        first_action = x0_est[:, 0, :]
        last_action = x0_est[:, -1, :]
        aux_in = torch.cat([condition, mean_action, first_action, last_action], dim=-1)
        h = self.aux_trunk(aux_in)
        final_err = self.final_blue_error_head(h)
        min_dist = torch.nn.functional.softplus(self.min_distance_head(h))
        success_logit = self.success_logit_head(h)
        return {"final_blue_error": final_err, "min_distance": min_dist, "success_logit": success_logit}


def build_model(spec):
    return GoalAnchoredBluePlacementPolicy(spec)


def last_valid_future(future_obs, future_mask):
    b, h, d = future_obs.shape
    lengths = future_mask.squeeze(-1).to(dtype=torch.long).sum(dim=1).clamp_min(1)
    idx = (lengths - 1).view(b, 1, 1).expand(b, 1, d)
    return future_obs.gather(1, idx).squeeze(1)


def blue_goal_labels(batch, spec):
    raw_hist = batch["raw_obs"]
    future = batch["future_obs"]
    fields = spec["fields"]
    blue0, blue1 = fields["blue_pose"]
    bg0, bg1 = fields["blue_goal"]
    blue0 = int(blue0)
    bg0 = int(bg0)
    bg1 = int(bg1)
    goal = raw_hist[:, -1, bg0:bg1]
    current_blue = raw_hist[:, -1, blue0:blue0 + 3]
    future_blue = future[:, :, blue0:blue0 + 3]
    last = last_valid_future(future, batch["future_mask"])
    last_blue = last[:, blue0:blue0 + 3]

    current_err = current_blue - goal
    future_err = future_blue - goal[:, None, :]
    last_err = last_blue - goal

    xy_ok_future = (torch.abs(future_err[:, :, 0]) <= 0.04) & (torch.abs(future_err[:, :, 1]) <= 0.04)
    z_ok_future = torch.abs(future_err[:, :, 2]) <= 0.011
    valid = batch["future_mask"].squeeze(-1).to(dtype=torch.bool, device=future.device)
    future_at = xy_ok_future & z_ok_future & valid
    xy_ok_current = (torch.abs(current_err[:, 0]) <= 0.04) & (torch.abs(current_err[:, 1]) <= 0.04)
    z_ok_current = torch.abs(current_err[:, 2]) <= 0.011
    current_at = xy_ok_current & z_ok_current
    any_at = (future_at.any(dim=1) | current_at).to(dtype=future.dtype).view(-1, 1)

    dist = torch.linalg.norm(future_err, dim=-1)
    big = torch.full_like(dist, 10.0)
    dist = torch.where(valid, dist, big)
    current_dist = torch.linalg.norm(current_err, dim=-1, keepdim=True)
    min_future = torch.minimum(dist.min(dim=1, keepdim=True).values, current_dist)
    return last_err, min_future, any_at


def compute_loss(model, batch, spec):
    pred = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = public.epsilon_loss(pred, batch["noise"], batch["mask"])

    cfg = spec.get("candidate_config", {})
    if "config" in cfg and isinstance(cfg.get("config"), dict):
        cfg = cfg["config"]
    aux = model.auxiliary_predictions(
        batch["noisy_action"], batch["timesteps"], batch["raw_obs"], pred,
        batch.get("alpha_bar", None), batch.get("mask", None)
    )
    last_err, min_dist, any_at = blue_goal_labels(batch, spec)
    device = pred.device
    dtype = pred.dtype
    err_scale = torch.tensor([0.10, 0.10, 0.12], device=device, dtype=dtype)
    target_err = last_err / err_scale
    target_min = min_dist / torch.tensor(0.20, device=device, dtype=dtype)

    alpha = batch.get("alpha_bar", None)
    if alpha is None:
        reliability = torch.ones(pred.shape[0], 1, device=device, dtype=dtype)
    else:
        reliability = alpha.to(device=device, dtype=dtype).reshape(-1, 1).clamp(0.05, 1.0)

    pose_per = torch.nn.functional.smooth_l1_loss(aux["final_blue_error"], target_err, reduction="none").mean(dim=-1, keepdim=True)
    min_per = torch.nn.functional.smooth_l1_loss(aux["min_distance"], target_min, reduction="none")
    bce_per = torch.nn.functional.binary_cross_entropy_with_logits(aux["success_logit"], any_at, reduction="none")
    terminal_per = aux["final_blue_error"].square().mean(dim=-1, keepdim=True) * any_at

    lambda_pose = float(cfg.get("lambda_pose", 0.06))
    lambda_min = float(cfg.get("lambda_min_distance", 0.02))
    lambda_success = float(cfg.get("lambda_success", 0.03))
    lambda_terminal = float(cfg.get("lambda_terminal_goal", 0.02))
    prior_per = lambda_pose * pose_per + lambda_min * min_per + lambda_success * bce_per + lambda_terminal * terminal_per
    prior_loss = (reliability * prior_per).mean()
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
