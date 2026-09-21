import torch
from appl.public import DiffusionBackbone, normalize_observation, epsilon_loss


class SupportPreservationDiffusion(torch.nn.Module):
    """Learned epsilon-prediction diffusion policy with a support-preservation prior.

    The DDPM action model is a standard conditional U-Net. The conditioning vector
    is learned from the two causal state observations plus world-frame relative
    features needed for the prior: tcp-red, red-blue, goal offsets and gripper
    aperture. A learned auxiliary decoder maps a candidate denoised action horizon
    and the same causal condition to future object/TCP deltas; its supervised and
    physical losses provide the support-preservation prior during training.
    """

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        cfg = spec.get("candidate_config", {})
        self.horizon = int(spec["training"].get("horizon", 16))
        self.condition_dim = int(cfg.get("condition_dim", 256))
        self.position_scale = float(cfg.get("relative_position_scale_m", 0.5))

        fields = spec["fields"]
        self.qpos_slice = slice(fields["qpos"][0], fields["qpos"][1])
        self.qvel_slice = slice(fields["qvel"][0], fields["qvel"][1])
        self.tcp_slice = slice(fields["tcp_pose"][0], fields["tcp_pose"][1])
        self.red_slice = slice(fields["red_pose"][0], fields["red_pose"][1])
        self.blue_slice = slice(fields["blue_pose"][0], fields["blue_pose"][1])
        self.red_goal_slice = slice(fields["red_goal"][0], fields["red_goal"][1])
        self.blue_goal_slice = slice(fields["blue_goal"][0], fields["blue_goal"][1])

        engineered_dim = 47
        encoder_in = 2 * int(spec.get("observation_dimension", 47)) + engineered_dim
        self.condition_encoder = torch.nn.Sequential(
            torch.nn.Linear(encoder_in, self.condition_dim),
            torch.nn.LayerNorm(self.condition_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.condition_dim, self.condition_dim),
            torch.nn.LayerNorm(self.condition_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.condition_dim, self.condition_dim),
        )
        self.backbone = DiffusionBackbone(self.condition_dim, spec["training"])

        self.action_gru = torch.nn.GRU(input_size=8, hidden_size=64, num_layers=1, batch_first=True)
        self.future_decoder = torch.nn.Sequential(
            torch.nn.Linear(self.condition_dim + 64 + 3, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 10),
        )

    def engineered_features(self, raw_history):
        prev = raw_history[:, 0]
        cur = raw_history[:, -1]
        tcp = cur[:, self.tcp_slice]
        red = cur[:, self.red_slice]
        blue = cur[:, self.blue_slice]
        red_goal = cur[:, self.red_goal_slice]
        blue_goal = cur[:, self.blue_goal_slice]
        qpos = cur[:, self.qpos_slice]
        prev_tcp = prev[:, self.tcp_slice]
        prev_red = prev[:, self.red_slice]
        prev_blue = prev[:, self.blue_slice]
        prev_qpos = prev[:, self.qpos_slice]

        tcp_xyz = tcp[:, 0:3]
        red_xyz = red[:, 0:3]
        blue_xyz = blue[:, 0:3]
        red_goal_xyz = red_goal[:, 0:3]
        blue_goal_xyz = blue_goal[:, 0:3]
        scale = self.position_scale

        rel = torch.cat([
            (tcp_xyz - red_xyz) / scale,
            (tcp_xyz - blue_xyz) / scale,
            (red_xyz - blue_xyz) / scale,
            (red_goal_xyz - red_xyz) / scale,
            (blue_goal_xyz - blue_xyz) / scale,
            (tcp_xyz - prev_tcp[:, 0:3]) / scale,
            (red_xyz - prev_red[:, 0:3]) / scale,
            (blue_xyz - prev_blue[:, 0:3]) / scale,
        ], dim=-1)

        aperture = 0.5 * (qpos[:, 7:8] + qpos[:, 8:9])
        prev_aperture = 0.5 * (prev_qpos[:, 7:8] + prev_qpos[:, 8:9])
        fingers = torch.cat([
            qpos[:, 7:8] / 0.08,
            qpos[:, 8:9] / 0.08,
            aperture / 0.08,
            prev_aperture / 0.08,
            (aperture - prev_aperture) / 0.08,
        ], dim=-1)

        quat_and_goals = torch.cat([
            tcp[:, 3:7], red[:, 3:7], blue[:, 3:7],
            red_goal_xyz / scale, blue_goal_xyz / scale,
        ], dim=-1)
        return torch.cat([rel, fingers, quat_and_goals], dim=-1)

    def encode_condition(self, raw_history):
        norm = normalize_observation(raw_history, self.spec).reshape(raw_history.shape[0], -1)
        engineered = self.engineered_features(raw_history)
        return self.condition_encoder(torch.cat([norm, engineered], dim=-1))

    def forward(self, noisy_action, timestep, raw_history):
        cond = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, cond)

    def predict_future_deltas(self, clean_encoded_action, raw_history):
        """Predict [red_delta_xyz, blue_delta_xyz, tcp_delta_xyz, aperture_delta]."""
        cond = self.encode_condition(raw_history)
        action_context, unused_state = self.action_gru(clean_encoded_action)
        del unused_state
        b, h, dim = clean_encoded_action.shape
        del dim
        if h <= 1:
            tau = torch.zeros((b, h, 1), device=clean_encoded_action.device, dtype=clean_encoded_action.dtype)
        else:
            tau = torch.linspace(0.0, 1.0, h, device=clean_encoded_action.device, dtype=clean_encoded_action.dtype).view(1, h, 1).expand(b, h, 1)
        time_features = torch.cat([tau, torch.sin(3.141592653589793 * tau), torch.cos(3.141592653589793 * tau)], dim=-1)
        cond_seq = cond.unsqueeze(1).expand(-1, h, -1)
        decoder_in = torch.cat([cond_seq, action_context, time_features], dim=-1)
        return self.future_decoder(decoder_in)


def build_model(spec):
    return SupportPreservationDiffusion(spec)


def masked_mean(value, mask, weight=None):
    m = mask
    if weight is not None:
        m = m * weight
    expanded = m.expand_as(value)
    return (value * expanded).sum() / expanded.sum().clamp_min(1.0)


def future_targets(batch, spec):
    fields = spec["fields"]
    future = batch["future_obs"]
    cur = batch["raw_obs"][:, -1]
    red_f = future[:, :, fields["red_pose"][0]:fields["red_pose"][0] + 3]
    blue_f = future[:, :, fields["blue_pose"][0]:fields["blue_pose"][0] + 3]
    tcp_f = future[:, :, fields["tcp_pose"][0]:fields["tcp_pose"][0] + 3]
    red_c = cur[:, fields["red_pose"][0]:fields["red_pose"][0] + 3]
    blue_c = cur[:, fields["blue_pose"][0]:fields["blue_pose"][0] + 3]
    tcp_c = cur[:, fields["tcp_pose"][0]:fields["tcp_pose"][0] + 3]
    qpos_f = future[:, :, fields["qpos"][0]:fields["qpos"][1]]
    qpos_c = cur[:, fields["qpos"][0]:fields["qpos"][1]]
    aperture_f = 0.5 * (qpos_f[:, :, 7:8] + qpos_f[:, :, 8:9])
    aperture_c = 0.5 * (qpos_c[:, 7:8] + qpos_c[:, 8:9])
    return torch.cat([
        red_f - red_c.unsqueeze(1),
        blue_f - blue_c.unsqueeze(1),
        tcp_f - tcp_c.unsqueeze(1),
        aperture_f - aperture_c.unsqueeze(1),
    ], dim=-1)


def auxiliary_losses(model, predicted, target, raw_obs, mask):
    red_scale = 0.25
    blue_scale = 0.05
    tcp_scale = 0.25
    grip_scale = 0.05
    red_loss = ((predicted[:, :, 0:3] - target[:, :, 0:3]) / red_scale).square()
    blue_loss = ((predicted[:, :, 3:6] - target[:, :, 3:6]) / blue_scale).square()
    tcp_loss = ((predicted[:, :, 6:9] - target[:, :, 6:9]) / tcp_scale).square()
    grip_loss = ((predicted[:, :, 9:10] - target[:, :, 9:10]) / grip_scale).square()
    supervised = (masked_mean(red_loss, mask) + 1.5 * masked_mean(blue_loss, mask)
                  + 0.5 * masked_mean(tcp_loss, mask) + 0.2 * masked_mean(grip_loss, mask))

    cur = raw_obs[:, -1]
    red0 = cur[:, model.red_slice][:, 0:3]
    blue0 = cur[:, model.blue_slice][:, 0:3]
    pred_red = red0.unsqueeze(1) + predicted[:, :, 0:3]
    pred_blue = blue0.unsqueeze(1) + predicted[:, :, 3:6]

    blue_xy = predicted[:, :, 3:5]
    blue_preserve = masked_mean((blue_xy / 0.02).square(), mask)

    clearance = pred_red[:, :, 2:3] - pred_blue[:, :, 2:3]
    h_min = 0.10
    insufficient_clearance = torch.relu(h_min - clearance) / h_min
    start_xy = red0[:, 0:2].unsqueeze(1)
    lateral_from_start = torch.linalg.vector_norm(pred_red[:, :, 0:2] - start_xy, dim=-1, keepdim=True) / 0.08
    clearance_gate = masked_mean((insufficient_clearance * lateral_from_start).square(), mask)
    return supervised, blue_preserve, clearance_gate


def compute_loss(model, batch, spec):
    predicted_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = epsilon_loss(predicted_noise, batch["noise"], batch["mask"])

    alpha_bar = batch["alpha_bar"].view(-1, 1, 1).to(dtype=batch["noisy_action"].dtype)
    sqrt_ab = alpha_bar.clamp_min(1.0e-4).sqrt()
    sqrt_one_minus = (1.0 - alpha_bar).clamp_min(0.0).sqrt()
    x0_est = (batch["noisy_action"] - sqrt_one_minus * predicted_noise) / sqrt_ab
    x0_aux = 2.0 * torch.tanh(x0_est / 2.0)

    target = future_targets(batch, spec)
    future_mask = batch["future_mask"]

    pred_from_true_action = model.predict_future_deltas(batch["encoded_action"], batch["raw_obs"])
    pred_from_denoised = model.predict_future_deltas(x0_aux, batch["raw_obs"])

    sup_true, blue_true, gate_true = auxiliary_losses(model, pred_from_true_action, target, batch["raw_obs"], future_mask)
    sup_pred, blue_pred, gate_pred = auxiliary_losses(model, pred_from_denoised, target, batch["raw_obs"], future_mask)

    denoise_weight = alpha_bar.detach().clamp_min(0.05)
    pred_target_error = (pred_from_denoised - target).square()
    pred_consistency = masked_mean(pred_target_error, future_mask, denoise_weight)

    cfg = spec.get("candidate_config", {})
    aux_w = float(cfg.get("auxiliary_weight", 0.04))
    blue_w = float(cfg.get("blue_preservation_weight", 0.02))
    clear_w = float(cfg.get("clearance_gate_weight", 0.01))
    pred_w = float(cfg.get("denoised_auxiliary_weight", 0.02))

    prior_loss = (aux_w * (0.7 * sup_true + 0.3 * sup_pred)
                  + pred_w * pred_consistency
                  + blue_w * (0.3 * blue_true + blue_pred)
                  + clear_w * (0.3 * gate_true + gate_pred))
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
