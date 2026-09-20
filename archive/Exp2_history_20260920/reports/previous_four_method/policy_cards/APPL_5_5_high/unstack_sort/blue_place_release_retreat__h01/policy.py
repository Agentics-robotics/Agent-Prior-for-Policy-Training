import math
import torch

from appl.public import DiffusionBackbone, normalize_observation, epsilon_loss


class GoalBasinEncoder(torch.nn.Module):
    """Causal state encoder for the blue-place goal basin.

    The encoder receives the two raw observations supplied by the framework.  It
    concatenates shared-normalized observations with broad-scale world-frame
    relative geometry that is specific to the handoff: blue-to-blue-goal,
    tcp-to-blue, tcp-to-blue-goal, red-to-red-goal, gripper aperture and coarse
    finite differences.  The constants below are task/robot scale constants, not
    per-slice empirical statistics.
    """

    def __init__(self, spec, condition_dim):
        super().__init__()
        obs_dim = int(spec.get("observation_dimension", 47))
        self.obs_dim = obs_dim
        self.history = int(spec["training"].get("observation_steps", 2))
        engineered_per_step = 37
        delta_dim = 17
        # two normalized observations, engineered features for each observation,
        # and one set of cross-history delta features from last minus previous.
        input_dim = self.history * obs_dim + self.history * engineered_per_step + delta_dim
        hidden = 256
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.Mish(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Mish(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, condition_dim),
            torch.nn.Mish(),
        )

    def features_one(self, obs):
        # Field layout from INTERFACE.md.
        qpos = obs[:, 0:9]
        qvel = obs[:, 9:18]
        tcp_pos = obs[:, 18:21]
        tcp_quat = obs[:, 21:25]
        red_pos = obs[:, 25:28]
        red_quat = obs[:, 28:32]
        blue_pos = obs[:, 32:35]
        blue_quat = obs[:, 35:39]
        red_goal = obs[:, 41:44]
        blue_goal = obs[:, 44:47]

        # Broad physical scaling in metres/radians.  These are deliberately not
        # narrow per-skill standard deviations.
        e_blue = (blue_goal - blue_pos) / torch.tensor([0.50, 0.50, 0.30], device=obs.device, dtype=obs.dtype)
        e_red = (red_goal - red_pos) / torch.tensor([0.50, 0.50, 0.30], device=obs.device, dtype=obs.dtype)
        tcp_to_blue = (blue_pos - tcp_pos) / torch.tensor([0.30, 0.30, 0.30], device=obs.device, dtype=obs.dtype)
        tcp_to_goal = (blue_goal - tcp_pos) / torch.tensor([0.50, 0.50, 0.30], device=obs.device, dtype=obs.dtype)

        finger_width = (qpos[:, 7:8] + qpos[:, 8:9]) / 0.08
        finger_balance = (qpos[:, 7:8] - qpos[:, 8:9]) / 0.02
        finger_vel = (qvel[:, 7:8] + qvel[:, 8:9]) / 0.20
        blue_height = (blue_pos[:, 2:3] - blue_goal[:, 2:3]) / 0.30
        tcp_height = (tcp_pos[:, 2:3] - blue_goal[:, 2:3]) / 0.30
        blue_xy_norm = torch.linalg.vector_norm(blue_goal[:, 0:2] - blue_pos[:, 0:2], dim=-1, keepdim=True) / 0.50
        tcp_blue_norm = torch.linalg.vector_norm(blue_pos - tcp_pos, dim=-1, keepdim=True) / 0.30
        red_xy_norm = torch.linalg.vector_norm(red_goal[:, 0:2] - red_pos[:, 0:2], dim=-1, keepdim=True) / 0.50

        # Soft causal predicates: continuous cues, not hard gates.
        blue_on_pad_soft = torch.exp(-20.0 * blue_xy_norm) * torch.exp(-8.0 * torch.abs(blue_height))
        red_on_pad_soft = torch.exp(-20.0 * red_xy_norm) * torch.exp(-8.0 * torch.abs((red_pos[:, 2:3] - red_goal[:, 2:3]) / 0.30))

        # Include the object quaternions and tcp quaternion directly.  They are
        # already bounded unit components and mostly cue yaw/contact changes.
        return torch.cat([
            e_blue, e_red, tcp_to_blue, tcp_to_goal,
            finger_width, finger_balance, finger_vel,
            blue_height, tcp_height, blue_xy_norm, tcp_blue_norm, red_xy_norm,
            blue_on_pad_soft, red_on_pad_soft,
            tcp_quat, red_quat[:, 0:2], blue_quat[:, 0:2],
            qvel[:, 0:7] / 1.0,
        ], dim=-1)

    def forward(self, raw_history, spec):
        norm = normalize_observation(raw_history, spec).reshape(raw_history.shape[0], -1)
        feats = []
        for i in range(raw_history.shape[1]):
            feats.append(self.features_one(raw_history[:, i, :]))
        last = raw_history[:, -1, :]
        prev = raw_history[:, -2, :] if raw_history.shape[1] > 1 else raw_history[:, -1, :]
        delta_blue = (last[:, 32:35] - prev[:, 32:35]) / torch.tensor([0.10, 0.10, 0.10], device=last.device, dtype=last.dtype)
        delta_tcp = (last[:, 18:21] - prev[:, 18:21]) / torch.tensor([0.10, 0.10, 0.10], device=last.device, dtype=last.dtype)
        delta_q = (last[:, 0:7] - prev[:, 0:7]) / 0.25
        delta_finger = (last[:, 7:9] - prev[:, 7:9]) / 0.04
        delta = torch.cat([delta_blue, delta_tcp, delta_q, delta_finger, last[:, 39:41]], dim=-1)
        return self.net(torch.cat([norm] + feats + [delta], dim=-1))


class GoalBasinDiffusionPolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        cfg = spec.get("candidate_config", {})
        condition_dim = int(cfg.get("condition_dim", 256))
        self.spec = spec
        self.horizon = int(spec["training"].get("horizon", 16))
        self.action_dim = 8
        self.encoder = GoalBasinEncoder(spec, condition_dim)
        self.backbone = DiffusionBackbone(condition_dim, spec["training"])

        action_hidden = int(cfg.get("action_summary_dim", 128))
        self.action_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.horizon * self.action_dim + 1, action_hidden),
            torch.nn.Mish(),
            torch.nn.LayerNorm(action_hidden),
            torch.nn.Linear(action_hidden, action_hidden),
            torch.nn.Mish(),
        )
        aux_in = condition_dim + action_hidden
        self.state_value = torch.nn.Sequential(
            torch.nn.Linear(condition_dim, 128),
            torch.nn.Mish(),
            torch.nn.Linear(128, 1),
        )
        self.action_value = torch.nn.Sequential(
            torch.nn.Linear(aux_in, 192),
            torch.nn.Mish(),
            torch.nn.Linear(192, 1),
        )
        self.terminal_error = torch.nn.Sequential(
            torch.nn.Linear(aux_in, 192),
            torch.nn.Mish(),
            torch.nn.Linear(192, 3),
        )

    def condition(self, raw_history, spec):
        return self.encoder(raw_history, spec)

    def forward(self, noisy_action, timestep, raw_history):
        # The fixed sampler calls model.forward(noisy_action, timestep, raw_history)
        # only, so build_model stores the immutable runtime spec for normalization.
        cond = self.encoder(raw_history, self.spec)
        return self.backbone(noisy_action, timestep, cond)

    def aux_predictions(self, raw_history, x0_estimate, mask, spec):
        cond = self.encoder(raw_history, spec)
        if mask is None:
            masked = x0_estimate
            valid_fraction = torch.ones(x0_estimate.shape[0], 1, device=x0_estimate.device, dtype=x0_estimate.dtype)
        else:
            masked = x0_estimate * mask
            valid_fraction = mask.mean(dim=(1, 2), keepdim=False).reshape(-1, 1)
        flat = masked.reshape(masked.shape[0], -1)
        act = self.action_encoder(torch.cat([flat, valid_fraction], dim=-1))
        joined = torch.cat([cond, act], dim=-1)
        return self.state_value(cond).squeeze(-1), self.action_value(joined).squeeze(-1), self.terminal_error(joined)


def build_model(spec):
    return GoalBasinDiffusionPolicy(spec)


def future_success_and_terminal(batch):
    fut = batch["future_obs"]
    fmask = batch["future_mask"].squeeze(-1)
    blue = fut[:, :, 32:35]
    red = fut[:, :, 25:28]
    red_goal = fut[:, :, 41:44]
    blue_goal = fut[:, :, 44:47]

    blue_xy = torch.amax(torch.abs(blue[:, :, 0:2] - blue_goal[:, :, 0:2]), dim=-1) <= 0.06
    red_xy = torch.amax(torch.abs(red[:, :, 0:2] - red_goal[:, :, 0:2]), dim=-1) <= 0.06
    blue_z = torch.abs(blue[:, :, 2] - blue_goal[:, :, 2]) <= 0.011
    red_z = torch.abs(red[:, :, 2] - red_goal[:, :, 2]) <= 0.011
    success_each = (blue_xy & red_xy & blue_z & red_z).to(fut.dtype) * fmask
    success_any = torch.amax(success_each, dim=1)

    raw_valid_count = fmask.sum(dim=1).to(torch.long)
    valid_count = raw_valid_count.clamp_min(1)
    last_index = valid_count - 1
    b = torch.arange(fut.shape[0], device=fut.device)
    terminal_blue = blue[b, last_index, :]
    terminal_goal = blue_goal[b, last_index, :]
    # Predict blue position error in broad task coordinates.
    scale = torch.tensor([0.12, 0.12, 0.30], device=fut.device, dtype=fut.dtype)
    terminal_error = (terminal_blue - terminal_goal) / scale
    terminal_valid = (raw_valid_count > 0).to(fut.dtype)
    return success_any, terminal_error, terminal_valid


def compute_loss(model, batch, spec):
    pred = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = epsilon_loss(pred, batch["noise"], batch["mask"])

    alpha_bar = batch["alpha_bar"].reshape(-1, 1, 1).to(dtype=batch["noisy_action"].dtype)
    sqrt_ab = torch.sqrt(alpha_bar.clamp_min(1.0e-6))
    sqrt_om = torch.sqrt((1.0 - alpha_bar).clamp_min(0.0))
    x0_est = (batch["noisy_action"] - sqrt_om * pred) / sqrt_ab
    # The auxiliary heads see a bounded clean-action estimate to avoid allowing
    # very high-noise samples to dominate the representation gradients.
    x0_aux = torch.clamp(x0_est, -1.5, 1.5)

    state_logit, action_logit, terminal_pred = model.aux_predictions(batch["raw_obs"], x0_aux, batch["mask"], spec)
    success_target, terminal_target, terminal_valid = future_success_and_terminal(batch)

    bce_state = torch.nn.functional.binary_cross_entropy_with_logits(state_logit, success_target, reduction="none")
    bce_action = torch.nn.functional.binary_cross_entropy_with_logits(action_logit, success_target, reduction="none")
    # Continuous low-noise weighting for action-conditioned auxiliaries; the
    # state-only critic remains fully supervised at all diffusion timesteps.
    low_noise_w = torch.sqrt(batch["alpha_bar"].to(dtype=bce_action.dtype)).clamp(0.05, 1.0)
    bce_action = (bce_action * low_noise_w).mean()
    bce_state = bce_state.mean()

    huber = torch.nn.functional.smooth_l1_loss(terminal_pred, terminal_target, reduction="none").mean(dim=-1)
    huber = (huber * low_noise_w * terminal_valid).sum() / (terminal_valid.sum().clamp_min(1.0))

    cfg = spec.get("candidate_config", {})
    w_state = float(cfg.get("state_value_loss_weight", 0.15))
    w_action = float(cfg.get("action_value_loss_weight", 0.10))
    w_terminal = float(cfg.get("terminal_error_loss_weight", 0.05))
    prior_loss = w_state * bce_state + w_action * bce_action + w_terminal * huber
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
