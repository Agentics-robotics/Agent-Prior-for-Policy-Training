import math
import torch

from appl.public import DiffusionBackbone, epsilon_loss


class RedCarryReleasePolicy(torch.nn.Module):
    """Diffusion policy with a learned carry/release object-dynamics prior.

    The diffusion model remains an epsilon-predicting action denoiser.  The prior
    is expressed in two places: (1) the global conditioning vector contains a
    learned attachment probability and TCP-to-red offset inferred only from the
    two causal observations, and (2) training uses an action-conditioned red
    object rollout auxiliary whose gradients flow through the DDPM clean-action
    estimate.
    """

    def __init__(self, spec):
        super().__init__()
        cfg = spec.get("candidate_config", {})
        self.condition_dim = int(cfg.get("condition_dim", 256))
        self.hidden_dim = int(cfg.get("prior_hidden_dim", 256))
        self.dyn_hidden_dim = int(cfg.get("dynamics_hidden_dim", 256))
        self.pos_scale_m = float(cfg.get("position_feature_scale_m", 0.25))
        self.delta_scale_m = float(cfg.get("delta_feature_scale_m", 0.10))
        self.offset_limit_m = float(cfg.get("offset_limit_m", 0.08))
        self.carry_delta_limit_m = float(cfg.get("carry_delta_limit_m", 0.08))
        self.free_delta_limit_m = float(cfg.get("free_delta_limit_m", 0.02))
        self.residual_delta_limit_m = float(cfg.get("residual_delta_limit_m", 0.01))

        n = spec["normalizer"]
        self.register_buffer("obs_mean", torch.tensor(n["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.tensor(n["std"], dtype=torch.float32))
        self.register_buffer("action_min", torch.tensor(n["action_min"], dtype=torch.float32))
        self.register_buffer("action_scale", torch.tensor(n["action_scale"], dtype=torch.float32))
        # Shared complete-demonstration observation scales for red position.
        self.register_buffer("red_pos_norm", torch.tensor([n["std"][25], n["std"][26], n["std"][27]], dtype=torch.float32))

        # normalized history (2*47) plus explicit object-centric causal features.
        feature_dim = 125
        self.feature_encoder = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, self.hidden_dim),
            torch.nn.Mish(),
            torch.nn.LayerNorm(self.hidden_dim),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.Mish(),
        )
        self.initial_attach_head = torch.nn.Linear(self.hidden_dim, 1)
        self.offset_head = torch.nn.Linear(self.hidden_dim, 3)
        self.release_readiness_head = torch.nn.Linear(self.hidden_dim, 1)
        self.condition_projector = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim + 5, self.condition_dim),
            torch.nn.Mish(),
            torch.nn.LayerNorm(self.condition_dim),
            torch.nn.Linear(self.condition_dim, self.condition_dim),
        )
        self.backbone = DiffusionBackbone(self.condition_dim, spec["training"])

        # Learned rollout used only by the differentiable auxiliary loss.
        dyn_input_dim = 8 + 8 + 3 + 3 + 1
        self.dyn_init = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim + 5, self.dyn_hidden_dim),
            torch.nn.Mish(),
            torch.nn.Linear(self.dyn_hidden_dim, self.dyn_hidden_dim),
            torch.nn.Tanh(),
        )
        self.dyn_cell = torch.nn.GRUCell(dyn_input_dim, self.dyn_hidden_dim)
        self.step_attach_head = torch.nn.Linear(self.dyn_hidden_dim, 1)
        self.carry_delta_head = torch.nn.Linear(self.dyn_hidden_dim, 3)
        self.free_delta_head = torch.nn.Linear(self.dyn_hidden_dim, 3)
        self.residual_delta_head = torch.nn.Linear(self.dyn_hidden_dim, 3)

    def normalise_obs(self, raw_history):
        return (raw_history - self.obs_mean.to(raw_history.dtype)) / self.obs_std.to(raw_history.dtype)

    def denormalise_action_local(self, encoded_action):
        return (encoded_action + 1.0) * self.action_scale.to(encoded_action.dtype) / 2.0 + self.action_min.to(encoded_action.dtype)

    def object_features(self, raw_history):
        raw = raw_history
        dtype = raw.dtype
        norm_flat = self.normalise_obs(raw).reshape(raw.shape[0], -1)
        prev = raw[:, 0]
        cur = raw[:, 1]

        tcp_prev = prev[:, 18:21]
        tcp_cur = cur[:, 18:21]
        red_prev = prev[:, 25:28]
        red_cur = cur[:, 25:28]
        blue_cur = cur[:, 32:35]
        red_goal = cur[:, 41:44]

        pscale = torch.as_tensor(self.pos_scale_m, device=raw.device, dtype=dtype)
        dscale = torch.as_tensor(self.delta_scale_m, device=raw.device, dtype=dtype)
        rel_cur = (red_cur - tcp_cur) / pscale
        rel_prev = (red_prev - tcp_prev) / pscale
        tcp_delta = (tcp_cur - tcp_prev) / dscale
        red_delta = (red_cur - red_prev) / dscale
        red_to_goal = (red_goal - red_cur) / pscale
        tcp_to_goal = (red_goal - tcp_cur) / pscale
        tcp_to_blue = (blue_cur - tcp_cur) / pscale
        red_to_blue = (blue_cur - red_cur) / pscale

        finger_cur = cur[:, 7:9]
        finger_prev = prev[:, 7:9]
        width_cur = finger_cur.sum(dim=-1, keepdim=True) / 0.08
        width_prev = finger_prev.sum(dim=-1, keepdim=True) / 0.08
        width_change = (finger_cur.sum(dim=-1, keepdim=True) - finger_prev.sum(dim=-1, keepdim=True)) / 0.05
        finger_scaled = finger_cur / 0.04
        drawer_scaled = torch.cat([cur[:, 39:40] / 0.30, cur[:, 40:41] / 0.10], dim=-1)

        return torch.cat([
            norm_flat,
            rel_cur, rel_prev, tcp_delta, red_delta,
            red_to_goal, tcp_to_goal, tcp_to_blue, red_to_blue,
            width_cur, width_prev, width_change,
            finger_scaled, drawer_scaled,
        ], dim=-1)

    def encode_condition(self, raw_history):
        feat = self.object_features(raw_history)
        hidden = self.feature_encoder(feat)
        attach_logit = self.initial_attach_head(hidden)
        attach_prob = torch.sigmoid(attach_logit)
        offset_m = self.offset_limit_m * torch.tanh(self.offset_head(hidden))
        release_readiness = torch.sigmoid(self.release_readiness_head(hidden))
        prior_latent = torch.cat([hidden, attach_prob, offset_m / self.offset_limit_m, release_readiness], dim=-1)
        cond = self.condition_projector(prior_latent)
        aux = {
            "hidden": hidden,
            "attach_logit": attach_logit,
            "attach_prob": attach_prob,
            "offset_m": offset_m,
            "release_readiness": release_readiness,
            "prior_latent": prior_latent,
        }
        return cond, aux

    def forward(self, noisy_action, timestep, raw_history):
        cond, unused_aux = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, cond)

    def rollout_red_dynamics(self, raw_history, encoded_action):
        """Predict future red positions from causal state and a clean action sequence.

        This is not an executable controller.  It is a learned differentiable
        dynamics head used to train the attachment/release representation.
        """
        cond, aux = self.encode_condition(raw_history)
        native_action = self.denormalise_action_local(encoded_action)
        cur = raw_history[:, 1]
        red_pos = cur[:, 25:28]
        tcp_pos = cur[:, 18:21]
        red_goal = cur[:, 41:44]
        offset = aux["offset_m"]
        attach_prob = aux["attach_prob"]
        h = self.dyn_init(aux["prior_latent"])

        pred_pos = []
        attach_logits = []
        carry_deltas = []
        free_deltas = []
        residual_deltas = []
        prev_attach = attach_prob
        for j in range(encoded_action.shape[1]):
            rel_goal = (red_goal - red_pos) / self.pos_scale_m
            rel_offset = (red_pos - tcp_pos - offset) / self.pos_scale_m
            step_in = torch.cat([
                encoded_action[:, j],
                native_action[:, j],
                rel_goal,
                rel_offset,
                prev_attach,
            ], dim=-1)
            h = self.dyn_cell(step_in, h)
            step_logit = self.step_attach_head(h)
            step_attach = torch.sigmoid(step_logit)
            carry_delta = self.carry_delta_limit_m * torch.tanh(self.carry_delta_head(h))
            free_delta = self.free_delta_limit_m * torch.tanh(self.free_delta_head(h))
            residual_delta = self.residual_delta_limit_m * torch.tanh(self.residual_delta_head(h))
            red_delta = step_attach * (carry_delta + residual_delta) + (1.0 - step_attach) * free_delta
            red_pos = red_pos + red_delta
            # The true TCP is not available at deployment; this proxy simply keeps
            # the relative-offset feature numerically meaningful inside the learned
            # rollout and is trained only through object losses.
            tcp_pos = tcp_pos + carry_delta
            pred_pos.append(red_pos)
            attach_logits.append(step_logit)
            carry_deltas.append(carry_delta)
            free_deltas.append(free_delta)
            residual_deltas.append(residual_delta)
            prev_attach = step_attach

        return {
            "pred_red_pos": torch.stack(pred_pos, dim=1),
            "attach_logits": torch.stack(attach_logits, dim=1),
            "carry_deltas": torch.stack(carry_deltas, dim=1),
            "free_deltas": torch.stack(free_deltas, dim=1),
            "residual_deltas": torch.stack(residual_deltas, dim=1),
            "initial_attach_logit": aux["attach_logit"],
            "initial_offset_m": aux["offset_m"],
        }


def masked_mean(x, mask, denom):
    return (x * mask).sum() / (mask.sum() * denom).clamp_min(1.0)


def build_model(spec):
    return RedCarryReleasePolicy(spec)


def compute_loss(model, batch, spec):
    pred_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = epsilon_loss(pred_noise, batch["noise"], batch["mask"])

    alpha_bar = batch["alpha_bar"].reshape(-1, 1, 1).to(pred_noise.dtype)
    sqrt_alpha = torch.sqrt(alpha_bar.clamp_min(1.0e-6))
    sqrt_one_minus = torch.sqrt((1.0 - alpha_bar).clamp_min(0.0))
    x0_est = (batch["noisy_action"] - sqrt_one_minus * pred_noise) / sqrt_alpha
    x0_est = torch.clamp(x0_est, -1.5, 1.5)

    # Stabilise the auxiliary against very noisy DDPM steps while preserving a
    # nonzero gradient path from the object loss to the denoiser's epsilon output.
    blend = (0.25 * torch.sqrt(alpha_bar.clamp_min(0.0))).to(pred_noise.dtype)
    clean_for_prior = batch["encoded_action"] + blend * (x0_est - batch["encoded_action"])

    rollout = model.rollout_red_dynamics(batch["raw_obs"], clean_for_prior)
    mask = batch["future_mask"].to(pred_noise.dtype)
    target_red = batch["future_obs"][:, :, 25:28]
    pred_red = rollout["pred_red_pos"]
    red_norm = model.red_pos_norm.to(pred_noise.dtype).reshape(1, 1, 3).clamp_min(0.05)
    pos_sq = ((pred_red - target_red) / red_norm).square()
    pos_loss = masked_mean(pos_sq, mask, torch.as_tensor(3.0, device=pred_noise.device, dtype=pred_noise.dtype))

    future_width = batch["future_obs"][:, :, 7:9].sum(dim=-1, keepdim=True)
    attach_target = (future_width < 0.055).to(pred_noise.dtype)
    attach_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        rollout["attach_logits"], attach_target, reduction="none")
    attach_loss = masked_mean(attach_bce, mask, torch.as_tensor(1.0, device=pred_noise.device, dtype=pred_noise.dtype))

    cur_width = batch["raw_obs"][:, 1, 7:9].sum(dim=-1, keepdim=True)
    cur_attach = (cur_width < 0.055).to(pred_noise.dtype)
    initial_attach_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        rollout["initial_attach_logit"], cur_attach, reduction="mean")
    current_offset = batch["raw_obs"][:, 1, 25:28] - batch["raw_obs"][:, 1, 18:21]
    offset_sq = ((rollout["initial_offset_m"] - current_offset) / model.offset_limit_m).square()
    offset_loss = (offset_sq * cur_attach).sum() / (cur_attach.sum() * 3.0).clamp_min(1.0)

    prev_red = torch.cat([batch["raw_obs"][:, 1:2, 25:28], target_red[:, :-1]], dim=1)
    target_delta = target_red - prev_red
    carry_sq = ((rollout["carry_deltas"] - target_delta) / red_norm).square()
    carry_loss = (carry_sq * mask * attach_target).sum() / ((mask * attach_target).sum() * 3.0).clamp_min(1.0)
    free_stationary_sq = (rollout["free_deltas"] / red_norm).square()
    free_loss = (free_stationary_sq * mask * (1.0 - attach_target)).sum() / ((mask * (1.0 - attach_target)).sum() * 3.0).clamp_min(1.0)
    residual_loss = ((rollout["residual_deltas"] / red_norm).square() * mask).sum() / (mask.sum() * 3.0).clamp_min(1.0)

    cfg = spec.get("candidate_config", {})
    prior_weight = float(cfg.get("prior_weight", 0.05))
    prior_loss = (
        float(cfg.get("red_position_loss_weight", 1.0)) * pos_loss
        + float(cfg.get("attachment_bce_weight", 0.35)) * attach_loss
        + float(cfg.get("initial_attachment_weight", 0.15)) * initial_attach_loss
        + float(cfg.get("offset_loss_weight", 0.20)) * offset_loss
        + float(cfg.get("carry_delta_loss_weight", 0.40)) * carry_loss
        + float(cfg.get("free_stationary_loss_weight", 0.10)) * free_loss
        + float(cfg.get("residual_regularizer_weight", 0.02)) * residual_loss
    )
    loss = diffusion_loss + prior_weight * prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
