import math
import torch
import appl.public as public


class ReleaseGatedDiffusionPolicy(torch.nn.Module):
    """A learned DDPM denoiser with a supported-release conditioning prior.

    The model remains an epsilon-predicting action diffusion model. The prior is
    implemented as learned causal support/event heads whose probabilities are fed
    to the diffusion denoiser, plus auxiliary losses that train those heads and
    discourage denoised open-gripper actions when the causal observation predicts
    that the blue block is not yet table-supported and aligned with its goal.
    """

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        cfg = spec.get("candidate_config", {}).get("config", {})
        self.prior_weights = {
            "support_bce": float(cfg.get("support_bce_weight", 0.05)),
            "event_ce": float(cfg.get("event_ce_weight", 0.05)),
            "unsupported_open": float(cfg.get("unsupported_open_weight", 0.08)),
            "gripper_bce": float(cfg.get("gripper_bce_weight", 0.02)),
        }
        self.condition_dim = int(cfg.get("condition_dim", 256))
        hidden = int(cfg.get("encoder_hidden", 256))
        latent = int(cfg.get("latent_dim", 192))
        self.latent_dim = latent

        # raw normalized history: 2 * 47. Engineered causal geometric features:
        # six 3-vectors plus width, previous width, width change, z error,
        # xy error norm, and two finger velocities = 25.
        encoder_in = 2 * 47 + 25
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(encoder_in, hidden),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, latent),
            torch.nn.SiLU(),
        )
        self.support_head = torch.nn.Linear(latent, 1)
        self.event_head = torch.nn.Linear(latent, 4)
        # The final condition exposes both the continuous latent and the learned
        # release state probabilities. The projection keeps the public backbone
        # condition dimension within the interface limit.
        self.condition_projector = torch.nn.Sequential(
            torch.nn.Linear(latent + 1 + 4 + 25, self.condition_dim),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(self.condition_dim),
        )
        self.backbone = public.DiffusionBackbone(self.condition_dim, spec["training"])

    def normalized_history(self, raw_history):
        return public.normalize_observation(raw_history, self.spec).reshape(raw_history.shape[0], -1)

    def engineered_features(self, raw_history):
        # All features use only the two causal observations. Scaling constants
        # are broad task/world scales, not per-skill fitted normalizers.
        prev = raw_history[:, 0, :]
        cur = raw_history[:, -1, :]
        tcp_c = cur[:, 18:21]
        tcp_p = prev[:, 18:21]
        red_c = cur[:, 25:28]
        blue_c = cur[:, 32:35]
        blue_p = prev[:, 32:35]
        red_goal = cur[:, 41:44]
        blue_goal = cur[:, 44:47]
        width_c = (cur[:, 7:8] + cur[:, 8:9])
        width_p = (prev[:, 7:8] + prev[:, 8:9])
        finger_vel = cur[:, 16:18]

        blue_rel = (blue_c - blue_goal) / 0.30
        red_rel = (red_c - red_goal) / 0.30
        tcp_blue = (tcp_c - blue_c) / 0.20
        tcp_goal = (tcp_c - blue_goal) / 0.40
        d_blue = (blue_c - blue_p) / 0.05
        d_tcp = (tcp_c - tcp_p) / 0.05
        z_err = (blue_c[:, 2:3] - blue_goal[:, 2:3]) / 0.30
        xy_err_norm = torch.linalg.vector_norm(blue_c[:, 0:2] - blue_goal[:, 0:2], dim=-1, keepdim=True) / 0.30
        features = torch.cat([
            blue_rel, red_rel, tcp_blue, tcp_goal, d_blue, d_tcp,
            width_c / 0.08, width_p / 0.08, (width_c - width_p) / 0.02,
            z_err, xy_err_norm, finger_vel / 0.20,
        ], dim=-1)
        return features

    def encode(self, raw_history):
        norm = self.normalized_history(raw_history)
        geom = self.engineered_features(raw_history)
        latent = self.encoder(torch.cat([norm, geom], dim=-1))
        support_logit = self.support_head(latent).squeeze(-1)
        event_logits = self.event_head(latent)
        support_prob = torch.sigmoid(support_logit).unsqueeze(-1)
        event_prob = torch.softmax(event_logits, dim=-1)
        condition = self.condition_projector(torch.cat([latent, support_prob, event_prob, geom], dim=-1))
        return condition, support_logit, event_logits

    def forward(self, noisy_action, timestep, raw_history):
        condition, ignored_support, ignored_event = self.encode(raw_history)
        return self.backbone(noisy_action, timestep, condition)


def current_labels(raw_obs, native_action):
    """Return support labels and four release-event labels from causal state.

    Events are: 0 carry_closed, 1 descend_closed, 2 release_open, 3 retreat_open.
    These labels are only used during training of learned heads; at deployment the
    heads infer the state from raw observation history.
    """
    cur = raw_obs[:, -1, :]
    blue = cur[:, 32:35]
    goal = cur[:, 44:47]
    tcp = cur[:, 18:21]
    width = cur[:, 7] + cur[:, 8]
    xy_err = torch.linalg.vector_norm(blue[:, 0:2] - goal[:, 0:2], dim=-1)
    z_err = blue[:, 2] - goal[:, 2]
    supported = ((xy_err < 0.040) & (z_err.abs() < 0.018)).to(raw_obs.dtype)

    # Use observed fingers and the demonstrated gripper target to detect the
    # release transition. The commanded action is a training label only.
    commanded_open = native_action[:, 0, 7] > 0.0
    fingers_open = width > 0.060
    is_open = commanded_open | fingers_open
    high = z_err > 0.075
    retreat = (tcp[:, 2] - blue[:, 2]) > 0.070

    labels = torch.zeros(raw_obs.shape[0], device=raw_obs.device, dtype=torch.long)
    labels[(~is_open) & (~high)] = 1
    labels[is_open & (~retreat)] = 2
    labels[is_open & retreat] = 3
    return supported, labels


def future_action_open_labels(native_action):
    return (native_action[:, :, 7] > 0.0).to(native_action.dtype)


def build_model(spec):
    return ReleaseGatedDiffusionPolicy(spec)


def compute_loss(model, batch, spec):
    pred_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = public.epsilon_loss(pred_noise, batch["noise"], batch["mask"])

    # Learned causal support/event heads.
    ignored_condition, support_logit, event_logits = model.encode(batch["raw_obs"])
    support_label, event_label = current_labels(batch["raw_obs"], batch["native_action"])
    support_loss = torch.nn.functional.binary_cross_entropy_with_logits(support_logit, support_label)
    event_loss = torch.nn.functional.cross_entropy(event_logits, event_label)

    # Low-noise denoised action estimate for differentiable release-consistency
    # losses. This is not used as a controller; it regularizes the denoiser.
    alpha_bar = batch["alpha_bar"].to(pred_noise.dtype).reshape(-1, 1, 1).clamp_min(1.0e-4)
    x0 = (batch["noisy_action"] - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
    x0_cmd = x0[:, :, 7].clamp(-1.5, 1.5)
    mask = batch["mask"].squeeze(-1).to(x0_cmd.dtype)
    low_noise_weight = (alpha_bar.reshape(-1, 1).clamp(0.0, 1.0) ** 0.5).detach()

    # Discourage predicted opening when the learned causal support predictor says
    # the block is not supported/aligned. Detaching the probability prevents the
    # denoiser penalty from being minimized by simply declaring every state safe;
    # the support head itself is trained by the BCE above.
    p_support = torch.sigmoid(support_logit).detach().reshape(-1, 1)
    open_excess = torch.relu(x0_cmd + 0.20)
    unsupported_open_loss = ((1.0 - p_support) * open_excess.square() * mask * low_noise_weight).sum() / (mask.sum().clamp_min(1.0))

    # A small gripper-mode loss sharpens the release boundary learned from the
    # demonstrations without replacing diffusion supervision of all actions.
    desired_open = future_action_open_labels(batch["native_action"])
    grip_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        3.0 * x0_cmd, desired_open, reduction="none")
    gripper_loss = (grip_bce * mask * low_noise_weight).sum() / (mask.sum().clamp_min(1.0))

    w = model.prior_weights
    prior_loss = (w["support_bce"] * support_loss +
                  w["event_ce"] * event_loss +
                  w["unsupported_open"] * unsupported_open_loss +
                  w["gripper_bce"] * gripper_loss)
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
