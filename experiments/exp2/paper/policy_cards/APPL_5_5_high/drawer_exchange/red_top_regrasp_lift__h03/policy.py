import math
import torch
import appl.public as public


# Field slices in the fixed 47-D state vector.
QPOS = slice(0, 9)
QVEL = slice(9, 18)
TCP = slice(18, 25)
RED = slice(25, 32)
BLUE = slice(32, 39)
DRAWER_POS = 39
DRAWER_VEL = 40
RED_GOAL = slice(41, 44)


class AttachmentSwitchDiffusionPolicy(torch.nn.Module):
    """Mode-latent conditional DDPM epsilon model for the red top regrasp/lift slice."""

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        cfg = spec.get("candidate_config", {}).get("config", {})
        self.condition_dim = int(cfg.get("condition_dim", 320))
        self.feature_dim = 130
        hidden = int(cfg.get("encoder_hidden", 256))
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
        )
        self.mode_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, 96),
            torch.nn.SiLU(),
            torch.nn.Linear(96, 3),
        )
        self.attach_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, 96),
            torch.nn.SiLU(),
            torch.nn.Linear(96, 1),
        )
        self.lift_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, 96),
            torch.nn.SiLU(),
            torch.nn.Linear(96, 1),
        )
        self.rel_offset_head = torch.nn.Sequential(
            torch.nn.Linear(hidden, 96),
            torch.nn.SiLU(),
            torch.nn.Linear(96, 3),
        )
        self.mode_embeddings = torch.nn.Parameter(torch.randn(3, 32) * 0.02)
        self.condition_projector = torch.nn.Sequential(
            torch.nn.Linear(hidden + 3 + 32 + 1 + 1, self.condition_dim),
            torch.nn.LayerNorm(self.condition_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(self.condition_dim, self.condition_dim),
        )
        self.backbone = public.DiffusionBackbone(self.condition_dim, spec["training"])

    def observation_features(self, raw_history):
        # Use the shared full-demonstration normalizer for the raw state history.
        norm_history = public.normalize_observation(raw_history, self.spec)
        flat_norm = norm_history.reshape(norm_history.shape[0], -1)

        tcp_pos = raw_history[:, :, 18:21]
        red_pos = raw_history[:, :, 25:28]
        rel_pos = tcp_pos - red_pos
        rel_scaled = (rel_pos / 0.15).reshape(raw_history.shape[0], -1)

        red_delta = (red_pos[:, 1] - red_pos[:, 0]) / 0.05
        tcp_delta = (tcp_pos[:, 1] - tcp_pos[:, 0]) / 0.05
        rel_delta = (rel_pos[:, 1] - rel_pos[:, 0]) / 0.05

        fingers = raw_history[:, :, 7:9]
        finger_flat = (fingers / 0.04).reshape(raw_history.shape[0], -1)
        finger_delta = (fingers[:, 1] - fingers[:, 0]) / 0.02
        last_fingers = fingers[:, 1]
        finger_mean = last_fingers.mean(dim=-1, keepdim=True) / 0.04
        finger_width = last_fingers.sum(dim=-1, keepdim=True) / 0.08
        finger_asym = (last_fingers[:, 0:1] - last_fingers[:, 1:2]) / 0.02
        finger_qvel = raw_history[:, 1, 16:18] / 0.10

        last_rel = rel_pos[:, 1]
        rel_xy_norm = torch.linalg.norm(last_rel[:, :2], dim=-1, keepdim=True) / 0.15
        rel_z = last_rel[:, 2:3] / 0.15
        rel_norm = torch.linalg.norm(last_rel, dim=-1, keepdim=True) / 0.20

        red_z = raw_history[:, 1, 27:28] / 0.15
        tcp_z = raw_history[:, 1, 20:21] / 0.40
        drawer_pos = raw_history[:, 1, 39:40] / 0.30
        drawer_vel = raw_history[:, 1, 40:41] / 0.10
        red_to_goal = (raw_history[:, 1, 41:44] - red_pos[:, 1]) / 0.20

        return torch.cat([
            flat_norm,
            rel_scaled,
            red_delta,
            tcp_delta,
            rel_delta,
            finger_flat,
            finger_delta,
            finger_mean,
            finger_width,
            finger_asym,
            finger_qvel,
            rel_xy_norm,
            rel_z,
            rel_norm,
            red_z,
            tcp_z,
            drawer_pos,
            drawer_vel,
            red_to_goal,
        ], dim=-1)

    def make_condition(self, raw_history):
        features = self.observation_features(raw_history)
        encoded = self.encoder(features)
        mode_logits = self.mode_head(encoded)
        mode_prob = torch.softmax(mode_logits, dim=-1)
        attach_logit = self.attach_head(encoded).squeeze(-1)
        attach_prob = torch.sigmoid(attach_logit).unsqueeze(-1)
        lift_raw = self.lift_head(encoded).squeeze(-1)
        lift_prob = torch.sigmoid(lift_raw).unsqueeze(-1)
        rel_pred = self.rel_offset_head(encoded)
        mode_emb = mode_prob @ self.mode_embeddings
        condition = self.condition_projector(torch.cat([
            encoded,
            mode_prob,
            mode_emb,
            attach_prob,
            lift_prob,
        ], dim=-1))
        aux = {
            "mode_logits": mode_logits,
            "mode_prob": mode_prob,
            "attach_logit": attach_logit,
            "lift_raw": lift_raw,
            "rel_offset_pred": rel_pred,
        }
        return condition, aux

    def forward(self, noisy_action, timestep, raw_history, return_aux=False):
        condition, aux = self.make_condition(raw_history)
        eps = self.backbone(noisy_action, timestep, condition)
        if return_aux:
            return eps, aux
        return eps


def build_model(spec):
    return AttachmentSwitchDiffusionPolicy(spec)


def attachment_targets(batch):
    raw = batch["raw_obs"]
    future = batch["future_obs"]
    future_mask = batch["future_mask"].squeeze(-1)
    dtype = raw.dtype
    device = raw.device

    last = raw[:, 1]
    red0 = last[:, 25:28]
    tcp0 = last[:, 18:21]
    rel0 = tcp0 - red0
    finger_mean = last[:, 7:9].mean(dim=-1)

    # Continuous labels: the hard heuristic is converted to soft scores for supervision.
    closed_score = torch.sigmoid((torch.as_tensor(0.027, device=device, dtype=dtype) - finger_mean) / 0.003)
    rel_dist = torch.linalg.norm(rel0, dim=-1)
    prox_score = torch.sigmoid((torch.as_tensor(0.040, device=device, dtype=dtype) - rel_dist) / 0.010)

    red_future = future[:, :, 25:28]
    tcp_future = future[:, :, 18:21]
    valid = future_mask.clamp(0, 1)
    valid_count = valid.sum(dim=1).clamp_min(1.0)

    dz = red_future[:, :, 2] - red0[:, None, 2]
    very_neg = torch.full_like(dz, -1.0e6)
    max_dz = torch.where(valid > 0.5, dz, very_neg).max(dim=1).values.clamp_min(0.0)
    lift_score = torch.sigmoid((max_dz - 0.025) / 0.012)

    rel_future = tcp_future - red_future
    rel_mean = (rel_future * valid[:, :, None]).sum(dim=1) / valid_count[:, None]
    rel_var = ((rel_future - rel_mean[:, None, :]).square() * valid[:, :, None]).sum(dim=1) / valid_count[:, None]
    rel_rms = torch.sqrt(rel_var.sum(dim=-1).clamp_min(1.0e-12))
    stable_score = torch.sigmoid((0.012 - rel_rms) / 0.004)

    attach_label = (closed_score * prox_score * lift_score * stable_score).clamp(0.0, 1.0)
    contact_label = (closed_score * prox_score * (1.0 - 0.7 * lift_score) * (1.0 - 0.5 * attach_label)).clamp(0.0, 1.0)
    free_label = (1.0 - torch.maximum(attach_label, contact_label)).clamp(0.0, 1.0)
    mode_target = torch.stack([free_label, contact_label, attach_label], dim=-1)
    mode_target = mode_target / mode_target.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)

    rel_target = rel0 / 0.15
    lift_target = (max_dz / 0.25).clamp(0.0, 1.0)
    close_prob = (mode_target[:, 1] + mode_target[:, 2]).clamp(0.0, 1.0)
    gripper_target = 1.0 - 2.0 * close_prob
    return {
        "attach": attach_label,
        "mode": mode_target,
        "rel": rel_target,
        "lift": lift_target,
        "close_prob": close_prob,
        "gripper": gripper_target,
    }


def compute_loss(model, batch, spec):
    pred_noise, aux = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"], return_aux=True)
    diffusion_loss = public.epsilon_loss(pred_noise, batch["noise"], batch["mask"])

    targets = attachment_targets(batch)
    mode_logp = torch.log_softmax(aux["mode_logits"], dim=-1)
    mode_loss = -(targets["mode"] * mode_logp).sum(dim=-1).mean()
    attach_loss = torch.nn.functional.binary_cross_entropy_with_logits(aux["attach_logit"], targets["attach"])
    lift_loss = torch.nn.functional.smooth_l1_loss(torch.sigmoid(aux["lift_raw"]), targets["lift"])

    rel_weight = targets["close_prob"].detach()
    rel_error = torch.nn.functional.smooth_l1_loss(aux["rel_offset_pred"], targets["rel"], reduction="none").mean(dim=-1)
    rel_loss = (rel_error * (0.25 + rel_weight)).mean()

    # Low-weight denoised-action consistency with the inferred mode.
    alpha_bar = batch["alpha_bar"].reshape(-1, 1, 1).to(dtype=batch["noisy_action"].dtype, device=batch["noisy_action"].device)
    sqrt_ab = torch.sqrt(alpha_bar.clamp_min(1.0e-8))
    sqrt_om = torch.sqrt((1.0 - alpha_bar).clamp_min(0.0))
    x0_pred = (batch["noisy_action"] - sqrt_om * pred_noise) / sqrt_ab
    x0_gripper = x0_pred[:, :8, 7].clamp(-2.0, 2.0)
    grip_target = targets["gripper"][:, None].expand_as(x0_gripper)
    grip_mask = batch["mask"][:, :8, 0]
    ab_weight = ((alpha_bar[:, 0, 0] - 0.20) / 0.80).clamp(0.0, 1.0)[:, None]
    grip_error = torch.nn.functional.smooth_l1_loss(x0_gripper, grip_target, reduction="none")
    grip_loss = (grip_error * grip_mask * ab_weight).sum() / (grip_mask * ab_weight).sum().clamp_min(1.0)

    cfg = spec.get("candidate_config", {}).get("config", {})
    prior_loss = (
        float(cfg.get("mode_loss_weight", 0.05)) * mode_loss
        + float(cfg.get("attach_loss_weight", 0.05)) * attach_loss
        + float(cfg.get("lift_loss_weight", 0.02)) * lift_loss
        + float(cfg.get("rel_loss_weight", 0.01)) * rel_loss
        + float(cfg.get("gripper_consistency_weight", 0.01)) * grip_loss
    )
    total_loss = diffusion_loss + prior_loss
    return {
        "loss": total_loss,
        "diffusion_loss": diffusion_loss,
        "prior_loss": prior_loss,
    }
