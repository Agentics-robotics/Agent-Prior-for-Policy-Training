import math
import torch
from appl.public import DiffusionBackbone, epsilon_loss


TCP_SLICE = slice(18, 25)
RED_SLICE = slice(25, 32)
BLUE_SLICE = slice(32, 39)
DRAWER_INDEX = 39
DRAWER_VEL_INDEX = 40
RED_GOAL_SLICE = slice(41, 44)
BLUE_GOAL_SLICE = slice(44, 47)


def normalizer_from_spec(spec):
    if "normalizer" in spec:
        return spec["normalizer"]
    return spec["shared_normalizer"]


def config_from_spec(spec):
    cfg = {}
    if "candidate_config" in spec and isinstance(spec["candidate_config"], dict):
        inner = spec["candidate_config"].get("config", {})
        if isinstance(inner, dict):
            cfg.update(inner)
    return cfg


class TemporalAuxHead(torch.nn.Module):
    """Predicts future geometric state labels from the denoised action sequence.

    The head is deliberately separate from the epsilon output but receives the
    clean-action estimate, so its supervised clearance losses back-propagate into
    the diffusion denoiser through x0(noisy_action, predicted_epsilon).
    """

    def __init__(self, condition_dim, hidden_dim=192, out_dim=8):
        super().__init__()
        self.in_proj = torch.nn.Linear(condition_dim + 8 + 4, hidden_dim)
        self.conv1 = torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.out = torch.nn.Linear(hidden_dim, out_dim)
        self.act = torch.nn.SiLU()

    def forward(self, clean_action, condition):
        bsz, horizon, _ = clean_action.shape
        dtype = clean_action.dtype
        device = clean_action.device
        if horizon > 1:
            p = torch.arange(horizon, device=device, dtype=dtype) / float(horizon - 1)
        else:
            p = torch.zeros(horizon, device=device, dtype=dtype)
        pos = torch.stack((p, p * p, torch.sin(math.pi * p), torch.cos(math.pi * p)), dim=-1)
        pos = pos.unsqueeze(0).expand(bsz, horizon, 4)
        cond = condition.unsqueeze(1).expand(bsz, horizon, condition.shape[-1])
        x = torch.cat((clean_action, cond, pos), dim=-1)
        h = self.act(self.in_proj(x)).transpose(1, 2)
        r = h
        h = self.conv2(self.act(self.conv1(h)))
        h = self.act(h + r)
        r = h
        h = self.conv4(self.act(self.conv3(h)))
        h = self.act(h + r)
        h = h.transpose(1, 2)
        return self.out(h)


class OpenDrawerClearancePolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        cfg = config_from_spec(spec)
        normalizer = normalizer_from_spec(spec)
        obs_mean = torch.as_tensor(normalizer["mean"], dtype=torch.float32)
        obs_std = torch.as_tensor(normalizer["std"], dtype=torch.float32)
        self.register_buffer("obs_mean", obs_mean)
        self.register_buffer("obs_std", obs_std)
        self.condition_dim = int(cfg.get("condition_dim", 256))
        self.engineered_dim = 30
        feature_dim = 2 * 47 + 47 + self.engineered_dim
        hidden = int(cfg.get("encoder_hidden_dim", 256))
        self.encoder = torch.nn.Sequential(
            torch.nn.LayerNorm(feature_dim),
            torch.nn.Linear(feature_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, self.condition_dim),
            torch.nn.LayerNorm(self.condition_dim),
            torch.nn.SiLU(),
        )
        self.backbone = DiffusionBackbone(self.condition_dim, spec["training"])
        aux_hidden = int(cfg.get("aux_hidden_dim", 192))
        self.aux_head = TemporalAuxHead(self.condition_dim, aux_hidden, 8)

    def normalize_obs_history(self, raw_history):
        return (raw_history - self.obs_mean.to(raw_history.dtype)) / self.obs_std.to(raw_history.dtype)

    def engineered_features(self, raw_history):
        cur = raw_history[:, -1]
        prev = raw_history[:, 0]
        tcp = cur[:, TCP_SLICE]
        red = cur[:, RED_SLICE]
        blue = cur[:, BLUE_SLICE]
        ptcp = prev[:, TCP_SLICE]
        pred = prev[:, RED_SLICE]
        drawer = cur[:, DRAWER_INDEX:DRAWER_INDEX + 1]
        drawer_vel = cur[:, DRAWER_VEL_INDEX:DRAWER_VEL_INDEX + 1]
        zeros = torch.zeros_like(drawer)
        drawer_xy = torch.cat((0.19 - drawer, zeros), dim=-1)

        rel_tcp_red = (tcp[:, :3] - red[:, :3]) / 0.30
        prev_rel_tcp_red = (ptcp[:, :3] - pred[:, :3]) / 0.30
        rel_tcp_drawer = (tcp[:, :2] - drawer_xy) / 0.30
        rel_red_drawer = (red[:, :2] - drawer_xy) / 0.30
        rel_blue_drawer = (blue[:, :2] - drawer_xy) / 0.30

        fingers = cur[:, 7:9]
        finger_avg = fingers.mean(dim=-1, keepdim=True)
        finger_diff = fingers[:, 0:1] - fingers[:, 1:2]
        height_terms = torch.cat(
            (
                tcp[:, 2:3] / 0.30,
                red[:, 2:3] / 0.30,
                (tcp[:, 2:3] - red[:, 2:3]) / 0.30,
                ptcp[:, 2:3] / 0.30,
                pred[:, 2:3] / 0.30,
                finger_avg / 0.04,
                finger_diff / 0.04,
                drawer / 0.30,
                drawer_vel / 0.10,
            ),
            dim=-1,
        )

        red_goal = cur[:, RED_GOAL_SLICE]
        blue_goal = cur[:, BLUE_GOAL_SLICE]
        goal_terms = torch.cat(((red[:, :3] - red_goal) / 0.30, (blue[:, :3] - blue_goal) / 0.30), dim=-1)
        clearance_terms = torch.cat(((0.19 - drawer) / 0.30, (tcp[:, 2:3] - 0.075) / 0.30, (red[:, 2:3] - 0.075) / 0.30), dim=-1)
        return torch.cat(
            (
                rel_tcp_red,
                prev_rel_tcp_red,
                rel_tcp_drawer,
                rel_red_drawer,
                rel_blue_drawer,
                height_terms,
                goal_terms,
                clearance_terms,
            ),
            dim=-1,
        )

    def encode_condition(self, raw_history):
        norm = self.normalize_obs_history(raw_history)
        flat = norm.reshape(norm.shape[0], -1)
        delta = norm[:, 1] - norm[:, 0]
        engineered = self.engineered_features(raw_history)
        features = torch.cat((flat, delta, engineered), dim=-1)
        return self.encoder(features)

    def forward(self, noisy_action, timestep, raw_history):
        condition = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, condition)

    def predict_future_aux(self, clean_action, raw_history):
        condition = self.encode_condition(raw_history)
        return self.aux_head(clean_action, condition)

    def make_aux_targets(self, future_obs):
        mean = self.obs_mean.to(future_obs.dtype)
        std = self.obs_std.to(future_obs.dtype)
        tcp = (future_obs[..., 18:21] - mean[18:21]) / std[18:21]
        red = (future_obs[..., 25:28] - mean[25:28]) / std[25:28]
        fingers = (future_obs[..., 7:9] - mean[7:9]) / std[7:9]
        return torch.cat((tcp, red, fingers), dim=-1)

    def decode_aux_positions(self, aux_pred):
        mean = self.obs_mean.to(aux_pred.dtype)
        std = self.obs_std.to(aux_pred.dtype)
        tcp = aux_pred[..., 0:3] * std[18:21] + mean[18:21]
        red = aux_pred[..., 3:6] * std[25:28] + mean[25:28]
        fingers = aux_pred[..., 6:8] * std[7:9] + mean[7:9]
        return tcp, red, fingers


def build_model(spec):
    return OpenDrawerClearancePolicy(spec)


def compute_loss(model, batch, spec):
    pred_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = epsilon_loss(pred_noise, batch["noise"], batch["mask"])

    alpha_bar = batch["alpha_bar"].to(pred_noise.dtype).reshape(-1, 1, 1)
    sqrt_ab = torch.sqrt(alpha_bar.clamp_min(1.0e-6))
    sqrt_om = torch.sqrt((1.0 - alpha_bar).clamp_min(0.0))
    clean_est = (batch["noisy_action"] - sqrt_om * pred_noise) / sqrt_ab
    clean_est = clean_est.clamp(-1.20, 1.20)

    aux_pred = model.predict_future_aux(clean_est, batch["raw_obs"])
    aux_target = model.make_aux_targets(batch["future_obs"])
    valid = batch["future_mask"].to(aux_pred.dtype)
    alpha_weight = alpha_bar.clamp(0.05, 1.0)
    weighted_valid = valid * alpha_weight

    aux_dim = aux_pred.shape[-1]
    aux_mse = ((aux_pred - aux_target).square() * weighted_valid).sum() / (weighted_valid.sum() * aux_dim).clamp_min(1.0)

    pred_tcp, pred_red, pred_fingers = model.decode_aux_positions(aux_pred)
    target_tcp = batch["future_obs"][..., 18:21]
    target_red = batch["future_obs"][..., 25:28]
    target_fingers = batch["future_obs"][..., 7:9]

    tcp_high_gate = torch.sigmoid((target_tcp[..., 2:3] - 0.11) / 0.025)
    red_lift_gate = torch.sigmoid((target_red[..., 2:3] - 0.10) / 0.025)
    tcp_under = torch.relu((target_tcp[..., 2:3] - 0.012) - pred_tcp[..., 2:3]) / 0.15
    red_under = torch.relu((target_red[..., 2:3] - 0.008) - pred_red[..., 2:3]) / 0.15
    clearance_terms = tcp_high_gate * tcp_under.square() + red_lift_gate * red_under.square()
    clearance_loss = (clearance_terms * weighted_valid).sum() / weighted_valid.sum().clamp_min(1.0)

    current = batch["raw_obs"][:, -1]
    drawer = current[:, DRAWER_INDEX:DRAWER_INDEX + 1].to(aux_pred.dtype)
    drawer_center_x = (0.19 - drawer).unsqueeze(1)
    open_gate = torch.sigmoid((drawer - 0.26) / 0.02).unsqueeze(1)
    dx = (pred_tcp[..., 0:1] - drawer_center_x) / 0.20
    dy = pred_tcp[..., 1:2] / 0.18
    drawer_xy_risk = torch.exp(-(dx.square() + dy.square()))
    tcp_red_dx = (pred_tcp[..., 0:1] - pred_red[..., 0:1]) / 0.04
    tcp_red_dy = (pred_tcp[..., 1:2] - pred_red[..., 1:2]) / 0.04
    over_red = torch.exp(-(tcp_red_dx.square() + tcp_red_dy.square()))
    free_space_risk = (1.0 - over_red).clamp(0.0, 1.0)
    safe_tcp_z = 0.115
    geom_clear = (torch.relu(safe_tcp_z - pred_tcp[..., 2:3]) / 0.15).square()
    geom_loss = (geom_clear * drawer_xy_risk * free_space_risk * open_gate * weighted_valid).sum() / weighted_valid.sum().clamp_min(1.0)

    pred_finger_avg = pred_fingers.mean(dim=-1, keepdim=True)
    target_finger_avg = target_fingers.mean(dim=-1, keepdim=True)
    close_gate = torch.sigmoid((0.028 - target_finger_avg) / 0.004)
    open_finger_gate = torch.sigmoid((target_finger_avg - 0.035) / 0.004)
    close_over = torch.relu(pred_finger_avg - (target_finger_avg + 0.003)) / 0.02
    open_under = torch.relu((target_finger_avg - 0.003) - pred_finger_avg) / 0.02
    finger_loss = ((close_gate * close_over.square() + open_finger_gate * open_under.square()) * weighted_valid).sum() / weighted_valid.sum().clamp_min(1.0)

    cfg = config_from_spec(spec)
    aux_w = float(cfg.get("aux_state_weight", 0.08))
    clearance_w = float(cfg.get("clearance_weight", 0.15))
    geom_w = float(cfg.get("drawer_geometry_weight", 0.03))
    finger_w = float(cfg.get("finger_phase_weight", 0.03))
    prior_loss = aux_w * aux_mse + clearance_w * clearance_loss + geom_w * geom_loss + finger_w * finger_loss
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
