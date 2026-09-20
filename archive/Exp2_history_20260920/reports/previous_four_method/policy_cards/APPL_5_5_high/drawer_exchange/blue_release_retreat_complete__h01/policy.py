import torch
from appl.public import DiffusionBackbone, normalize_observation, epsilon_loss


class PredicateInsertionEncoder(torch.nn.Module):
    """Causal state encoder for the blue release/retreat completion prior.

    The encoder uses the shared observation normalizer for the full state and adds
    evaluator-relevant relative geometry in world and drawer frames. All added
    metre features are scaled by task-level geometric scales, not by the narrow
    empirical range of this late skill slice.
    """

    def __init__(self, spec, condition_dim=256):
        super().__init__()
        self.condition_dim = condition_dim
        self.input_dim = 160
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 256),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 256),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, condition_dim),
            torch.nn.SiLU(),
        )

    def single_geometry(self, obs):
        qpos = obs[..., 0:9]
        tcp = obs[..., 18:25]
        red = obs[..., 25:32]
        blue = obs[..., 32:39]
        drawer_pos = obs[..., 39:40]
        drawer_vel = obs[..., 40:41]
        red_goal = obs[..., 41:44]
        blue_goal = obs[..., 44:47]

        drawer_center_x = 0.19 - drawer_pos
        drawer_center = torch.cat(
            [drawer_center_x, torch.zeros_like(drawer_center_x), torch.full_like(drawer_center_x, 0.035)],
            dim=-1,
        )

        blue_xyz = blue[..., 0:3]
        red_xyz = red[..., 0:3]
        tcp_xyz = tcp[..., 0:3]

        blue_goal_err = (blue_xyz - blue_goal) / 0.15
        blue_drawer_err = (blue_xyz - drawer_center) / 0.20
        tcp_blue = (tcp_xyz - blue_xyz) / 0.25
        tcp_goal = (tcp_xyz - blue_goal) / 0.25
        red_goal_err = (red_xyz - red_goal) / 0.20

        drawer_margin = (drawer_pos - 0.26) / 0.15
        drawer_v = drawer_vel / 0.20
        finger_sum = (qpos[..., 7:8] + qpos[..., 8:9]) / 0.08

        blue_mx = (0.172 - torch.abs(blue_xyz[..., 0:1] - drawer_center[..., 0:1]) - 0.020) / 0.172
        blue_my = (0.182 - torch.abs(blue_xyz[..., 1:2] - drawer_center[..., 1:2]) - 0.020) / 0.182
        blue_z_low = (blue_xyz[..., 2:3] - 0.053) / 0.05
        blue_z_high = (0.074 - blue_xyz[..., 2:3]) / 0.05

        red_mx = (0.060 - torch.abs(red_xyz[..., 0:1] - red_goal[..., 0:1]) - 0.020) / 0.060
        red_my = (0.060 - torch.abs(red_xyz[..., 1:2] - red_goal[..., 1:2]) - 0.020) / 0.060
        red_z_band = torch.minimum((red_xyz[..., 2:3] - 0.014) / 0.03, (0.031 - red_xyz[..., 2:3]) / 0.03)

        drawer_open = (drawer_pos > 0.26).to(obs.dtype)
        red_on_pad = ((red_mx > 0.0) & (red_my > 0.0) & (red_z_band > 0.0)).to(obs.dtype)
        blue_inside = ((blue_mx > 0.0) & (blue_my > 0.0) & (blue_z_low > 0.0) & (blue_z_high > 0.0)).to(obs.dtype)

        return torch.cat(
            [
                blue_goal_err,
                blue_drawer_err,
                tcp_blue,
                tcp_goal,
                red_goal_err,
                drawer_margin,
                drawer_v,
                finger_sum,
                blue_mx,
                blue_my,
                blue_z_low,
                blue_z_high,
                red_mx,
                red_my,
                red_z_band,
                drawer_open,
                red_on_pad,
                blue_inside,
            ],
            dim=-1,
        )

    def forward(self, raw_history, spec):
        norm = normalize_observation(raw_history, spec).reshape(raw_history.shape[0], -1)
        geom0 = self.single_geometry(raw_history[:, 0, :])
        geom1 = self.single_geometry(raw_history[:, 1, :])

        tcp_delta = (raw_history[:, 1, 18:21] - raw_history[:, 0, 18:21]) / 0.10
        blue_delta = (raw_history[:, 1, 32:35] - raw_history[:, 0, 32:35]) / 0.10
        red_delta = (raw_history[:, 1, 25:28] - raw_history[:, 0, 25:28]) / 0.10
        drawer_delta = (raw_history[:, 1, 39:40] - raw_history[:, 0, 39:40]) / 0.10
        delta = torch.cat([tcp_delta, blue_delta, red_delta, drawer_delta], dim=-1)

        features = torch.cat([norm, geom0, geom1, delta], dim=-1)
        return self.net(features)


class ActionConditionedCompletionHead(torch.nn.Module):
    """Training-only auxiliary predictor from denoised actions and condition."""

    def __init__(self, condition_dim=256, hidden=128, horizon=16):
        super().__init__()
        self.horizon = horizon
        self.action_proj = torch.nn.Linear(8, hidden)
        self.cond_proj = torch.nn.Linear(condition_dim, hidden)
        self.pos = torch.nn.Parameter(torch.zeros(1, horizon, hidden))
        self.trunk = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
        )
        self.completion = torch.nn.Linear(hidden, 1)
        self.geometry = torch.nn.Linear(hidden, 8)

    def forward(self, condition, denoised_action):
        h = self.action_proj(denoised_action) + self.cond_proj(condition).unsqueeze(1)
        h = h + self.pos[:, : denoised_action.shape[1], :]
        h = self.trunk(h)
        return self.completion(h).squeeze(-1), self.geometry(h)


class BlueReleaseRetreatDiffusionPolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        cfg = spec.get("candidate_config", {}) if isinstance(spec, dict) else {}
        condition_dim = int(cfg.get("condition_dim", 256))
        self.spec = spec
        self.encoder = PredicateInsertionEncoder(spec, condition_dim=condition_dim)
        self.backbone = DiffusionBackbone(condition_dim, spec["training"])
        horizon = int(spec["training"].get("horizon", 16))
        self.aux = ActionConditionedCompletionHead(condition_dim=condition_dim, horizon=horizon)
        self.prior_bce_weight = float(cfg.get("prior_bce_weight", 0.05))
        self.prior_geom_weight = float(cfg.get("prior_geom_weight", 0.02))

    def condition(self, raw_history, spec):
        return self.encoder(raw_history, spec)

    def forward(self, noisy_action, timestep, raw_history):
        cond = self.condition(raw_history, self.spec)
        return self.backbone(noisy_action, timestep, cond)

    def auxiliary_loss(self, predicted_noise, batch, spec):
        noisy = batch["noisy_action"]
        alpha_bar = batch["alpha_bar"].to(device=noisy.device, dtype=noisy.dtype).view(-1, 1, 1)
        sqrt_ab = torch.sqrt(alpha_bar.clamp_min(1.0e-6))
        sqrt_om = torch.sqrt((1.0 - alpha_bar).clamp_min(0.0))
        x0 = (noisy - sqrt_om * predicted_noise) / sqrt_ab
        x0 = x0.clamp(-2.0, 2.0)

        cond = self.condition(batch["raw_obs"], spec)
        completion_logits, geom_pred = self.aux(cond, x0)

        future = batch["future_obs"]
        fmask = batch["future_mask"].squeeze(-1).to(dtype=noisy.dtype)
        labels = completion_labels(future).to(dtype=noisy.dtype)
        gate = alpha_bar.view(-1, 1).detach().clamp(0.05, 1.0)
        w = fmask * gate
        denom = w.sum().clamp_min(1.0)

        bce = torch.nn.functional.binary_cross_entropy_with_logits(completion_logits, labels, reduction="none")
        bce_loss = (bce * w).sum() / denom

        geom_tgt = future_geometry_targets(future).to(dtype=noisy.dtype)
        gm = w.unsqueeze(-1)
        geom_loss = ((geom_pred - geom_tgt).square() * gm).sum() / (gm.sum() * geom_pred.shape[-1]).clamp_min(1.0)
        return self.prior_bce_weight * bce_loss + self.prior_geom_weight * geom_loss


def completion_labels(obs):
    red = obs[..., 25:28]
    blue = obs[..., 32:35]
    drawer_pos = obs[..., 39:40]
    red_goal = obs[..., 41:44]

    drawer_center_x = 0.19 - drawer_pos
    drawer_open = drawer_pos > 0.26
    red_ok = (
        (torch.abs(red[..., 0:1] - red_goal[..., 0:1]) + 0.020 <= 0.060)
        & (torch.abs(red[..., 1:2] - red_goal[..., 1:2]) + 0.020 <= 0.060)
        & (red[..., 2:3] >= 0.014)
        & (red[..., 2:3] <= 0.031)
    )
    blue_ok = (
        (torch.abs(blue[..., 0:1] - drawer_center_x) + 0.020 <= 0.172)
        & (torch.abs(blue[..., 1:2]) + 0.020 <= 0.182)
        & (blue[..., 2:3] > 0.053)
        & (blue[..., 2:3] < 0.074)
    )
    return (drawer_open & red_ok & blue_ok).squeeze(-1)


def future_geometry_targets(obs):
    qpos = obs[..., 0:9]
    tcp = obs[..., 18:21]
    blue = obs[..., 32:35]
    drawer_pos = obs[..., 39:40]
    blue_goal = obs[..., 44:47]
    blue_goal_err = (blue - blue_goal) / 0.15
    tcp_blue = (tcp - blue) / 0.25
    drawer_margin = (drawer_pos - 0.26) / 0.15
    finger_sum = (qpos[..., 7:8] + qpos[..., 8:9]) / 0.08
    return torch.cat([blue_goal_err, tcp_blue, drawer_margin, finger_sum], dim=-1)


def build_model(spec):
    return BlueReleaseRetreatDiffusionPolicy(spec)


def compute_loss(model, batch, spec):
    predicted = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diff = epsilon_loss(predicted, batch["noise"], batch["mask"])
    prior = model.auxiliary_loss(predicted, batch, spec)
    return {"loss": diff + prior, "diffusion_loss": diff, "prior_loss": prior}
