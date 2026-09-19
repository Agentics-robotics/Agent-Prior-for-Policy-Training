import math
import torch
import appl.public as public


EPSILON = 1.0e-6


def vector_norm(x, dim=-1, keepdim=False):
    return torch.sqrt(torch.clamp((x * x).sum(dim=dim, keepdim=keepdim), min=EPSILON))


class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, final_activation=False):
        super().__init__()
        layers = []
        last = in_dim
        for width in hidden_dims:
            layers.append(torch.nn.Linear(last, width))
            layers.append(torch.nn.LayerNorm(width))
            layers.append(torch.nn.Mish())
            last = width
        layers.append(torch.nn.Linear(last, out_dim))
        if final_activation:
            layers.append(torch.nn.Mish())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AttachmentAwareAcquirePolicy(torch.nn.Module):
    """Diffusion policy with a learned causal attachment-state representation."""

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        cfg = spec.get("candidate_config", {})
        self.horizon = int(spec["training"].get("horizon", 16))
        self.condition_dim = int(cfg.get("condition_dim", 256))
        self.hidden_dim = int(cfg.get("encoder_hidden_dim", 256))

        obs_dim = int(spec.get("observation_dimension", 47))
        with torch.no_grad():
            dummy = torch.zeros(1, 2, obs_dim)
            feature_dim = int(self.make_features(dummy).shape[-1])
        self.feature_dim = feature_dim

        self.encoder = MLP(self.feature_dim, [self.hidden_dim, self.hidden_dim], self.hidden_dim)
        self.active_head = torch.nn.Linear(self.hidden_dim, 2)
        self.attach_head = torch.nn.Linear(self.hidden_dim, 2)

        cond_in = self.hidden_dim + 9
        self.condition_projector = MLP(cond_in, [self.hidden_dim], self.condition_dim)
        self.backbone = public.DiffusionBackbone(self.condition_dim, spec["training"])

        self.action_encoder = MLP(self.horizon * 8, [self.hidden_dim], 128)
        self.future_relative_head = MLP(self.condition_dim + 128, [self.hidden_dim, self.hidden_dim],
                                        self.horizon * 2 * 3)

    def make_features(self, raw_history):
        """Causal features from the two available observations using broad metric scales."""
        raw = raw_history
        norm_obs = public.normalize_observation(raw, self.spec).reshape(raw.shape[0], -1)
        prev = raw[:, 0]
        cur = raw[:, -1]

        qpos = cur[:, 0:9]
        qvel = cur[:, 9:18]
        prev_qpos = prev[:, 0:9]
        tcp = cur[:, 18:21]
        prev_tcp = prev[:, 18:21]
        red = cur[:, 25:28]
        prev_red = prev[:, 25:28]
        blue = cur[:, 32:35]
        prev_blue = prev[:, 32:35]
        red_goal = cur[:, 41:44]
        blue_goal = cur[:, 44:47]

        width = qpos[:, 7:8] + qpos[:, 8:9]
        prev_width = prev_qpos[:, 7:8] + prev_qpos[:, 8:9]
        d_width = width - prev_width
        finger_vel = qvel[:, 7:8] + qvel[:, 8:9]
        finger_feats = torch.cat([qpos[:, 7:9] / 0.04, width / 0.08, prev_width / 0.08,
                                  d_width / 0.02, finger_vel / 0.1], dim=-1)

        rel_red = red - tcp
        rel_blue = blue - tcp
        prev_rel_red = prev_red - prev_tcp
        prev_rel_blue = prev_blue - prev_tcp
        d_tcp = tcp - prev_tcp
        d_red = red - prev_red
        d_blue = blue - prev_blue
        d_rel_red = rel_red - prev_rel_red
        d_rel_blue = rel_blue - prev_rel_blue

        red_tcp_xy = vector_norm(rel_red[:, 0:2], dim=-1, keepdim=True)
        blue_tcp_xy = vector_norm(rel_blue[:, 0:2], dim=-1, keepdim=True)
        red_tcp_xyz = vector_norm(rel_red, dim=-1, keepdim=True)
        blue_tcp_xyz = vector_norm(rel_blue, dim=-1, keepdim=True)
        red_goal_delta = red - red_goal
        blue_goal_delta = blue - blue_goal
        red_goal_dist = vector_norm(red_goal_delta, dim=-1, keepdim=True)
        blue_goal_dist = vector_norm(blue_goal_delta, dim=-1, keepdim=True)

        red_comotion = d_red - d_tcp
        blue_comotion = d_blue - d_tcp
        comotion_feats = torch.cat([red_comotion / 0.05, blue_comotion / 0.05,
                                    vector_norm(red_comotion, dim=-1, keepdim=True) / 0.05,
                                    vector_norm(blue_comotion, dim=-1, keepdim=True) / 0.05], dim=-1)

        height_feats = torch.cat([tcp[:, 2:3] / 0.35, red[:, 2:3] / 0.35, blue[:, 2:3] / 0.35,
                                  (red[:, 2:3] - 0.02) / 0.30, (blue[:, 2:3] - 0.02) / 0.30], dim=-1)
        distance_feats = torch.cat([red_tcp_xy / 0.30, red_tcp_xyz / 0.35,
                                    blue_tcp_xy / 0.30, blue_tcp_xyz / 0.35], dim=-1)
        active_rule_feats = torch.cat([(red_goal_dist - blue_goal_dist) / 0.5,
                                       red_goal_dist / 0.5, blue_goal_dist / 0.5], dim=-1)

        engineered = torch.cat([
            finger_feats,
            tcp / 0.5, red / 0.5, blue / 0.5,
            rel_red / 0.35, rel_blue / 0.35,
            prev_rel_red / 0.35, prev_rel_blue / 0.35,
            d_tcp / 0.05, d_red / 0.05, d_blue / 0.05,
            d_rel_red / 0.05, d_rel_blue / 0.05,
            distance_feats,
            red_goal_delta / 0.5, blue_goal_delta / 0.5,
            red_goal_dist / 0.5, blue_goal_dist / 0.5,
            height_feats,
            comotion_feats,
            active_rule_feats,
        ], dim=-1)
        return torch.cat([norm_obs, engineered], dim=-1)

    def encode_condition(self, raw_history):
        features = self.make_features(raw_history)
        hidden = self.encoder(features)
        active_logits = self.active_head(hidden)
        attach_logits = self.attach_head(hidden)
        active_probs = torch.softmax(active_logits, dim=-1)
        attach_probs = torch.sigmoid(attach_logits)
        active_attachment = (active_probs * attach_probs).sum(dim=-1, keepdim=True)
        cond_input = torch.cat([hidden, active_logits, active_probs,
                                attach_logits, attach_probs, active_attachment], dim=-1)
        condition = self.condition_projector(cond_input)
        aux = {
            "active_logits": active_logits,
            "attach_logits": attach_logits,
            "active_probs": active_probs,
            "attach_probs": attach_probs,
            "active_attachment": active_attachment,
        }
        return condition, aux

    def predict_future_relative(self, condition, clean_action_estimate):
        b = clean_action_estimate.shape[0]
        action_flat = clean_action_estimate.reshape(b, self.horizon * 8)
        action_feat = self.action_encoder(action_flat)
        pred = self.future_relative_head(torch.cat([condition, action_feat], dim=-1))
        return pred.reshape(b, self.horizon, 2, 3)

    def forward(self, noisy_action, timestep, raw_history):
        condition, unused_aux = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, condition)


def attachment_targets(raw_obs, future_obs, future_mask):
    cur = raw_obs[:, -1]
    tcp = cur[:, 18:21]
    red = cur[:, 25:28]
    blue = cur[:, 32:35]
    red_goal = cur[:, 41:44]
    width = cur[:, 7:8] + cur[:, 8:9]

    fut_tcp = future_obs[:, :, 18:21]
    fut_red = future_obs[:, :, 25:28]
    fut_blue = future_obs[:, :, 32:35]
    m = future_mask[:, :, 0].clamp(0.0, 1.0)
    count = m.sum(dim=1).clamp_min(1.0)

    def one_object(obj, fut_obj):
        rel = obj - tcp
        fut_rel = fut_obj - fut_tcp
        rel_err = vector_norm(fut_rel - rel[:, None, :], dim=-1)
        mean_rel_err = (rel_err * m).sum(dim=1, keepdim=True) / count[:, None]
        masked_z = fut_obj[:, :, 2] * m + (-10.0) * (1.0 - m)
        max_future_z = masked_z.max(dim=1).values[:, None]
        masked_tcp_z = fut_tcp[:, :, 2] * m + (-10.0) * (1.0 - m)
        max_tcp_z = masked_tcp_z.max(dim=1).values[:, None]

        lateral = vector_norm(rel[:, 0:2], dim=-1, keepdim=True)
        z_abs = torch.abs(rel[:, 2:3])
        closed = torch.sigmoid((0.055 - width) / 0.006)
        near = torch.sigmoid((0.055 - lateral) / 0.012) * torch.sigmoid((0.045 - z_abs) / 0.012)
        stable = torch.sigmoid((0.050 - mean_rel_err) / 0.015)
        lifted_now = torch.sigmoid((obj[:, 2:3] - 0.055) / 0.015)
        lifted_future = torch.sigmoid((max_future_z - 0.055) / 0.015)
        tcp_lift_future = torch.sigmoid(((max_tcp_z - tcp[:, 2:3]) - 0.015) / 0.010)
        lift_or_future = torch.maximum(lifted_now, lifted_future * tcp_lift_future)
        return (closed * near * stable * lift_or_future).clamp(0.0, 1.0)

    red_attach = one_object(red, fut_red)
    blue_attach = one_object(blue, fut_blue)
    attach_targets = torch.cat([red_attach, blue_attach], dim=-1)

    red_goal_dist = vector_norm(red - red_goal, dim=-1)
    active_index = (red_goal_dist < 0.065).long()
    return active_index, attach_targets


def compute_loss(model, batch, spec):
    raw_obs = batch["raw_obs"]
    noisy_action = batch["noisy_action"]
    noise = batch["noise"]
    timesteps = batch["timesteps"]
    mask = batch["mask"]

    condition, aux = model.encode_condition(raw_obs)
    predicted_noise = model.backbone(noisy_action, timesteps, condition)
    diffusion_loss = public.epsilon_loss(predicted_noise, noise, mask)

    active_index, attach_t = attachment_targets(raw_obs, batch["future_obs"], batch["future_mask"])
    active_loss = torch.nn.functional.cross_entropy(aux["active_logits"], active_index, reduction="mean")
    attach_loss_all = torch.nn.functional.binary_cross_entropy_with_logits(
        aux["attach_logits"], attach_t, reduction="mean")
    active_attach_logit = aux["attach_logits"].gather(1, active_index[:, None])
    active_attach_target = attach_t.gather(1, active_index[:, None])
    attach_loss_active = torch.nn.functional.binary_cross_entropy_with_logits(
        active_attach_logit, active_attach_target, reduction="mean")

    alpha = batch["alpha_bar"].reshape(-1, 1, 1).clamp(1.0e-4, 0.9999)
    clean_est = (noisy_action - torch.sqrt(1.0 - alpha) * predicted_noise) / torch.sqrt(alpha)
    clean_est = clean_est.clamp(-1.5, 1.5)
    clean_est = clean_est.detach() + 0.05 * (clean_est - clean_est.detach())
    pred_rel = model.predict_future_relative(condition, clean_est)

    fut_tcp = batch["future_obs"][:, :, 18:21]
    fut_red_rel = batch["future_obs"][:, :, 25:28] - fut_tcp
    fut_blue_rel = batch["future_obs"][:, :, 32:35] - fut_tcp
    target_rel = torch.stack([fut_red_rel, fut_blue_rel], dim=2)
    obj_weight = torch.nn.functional.one_hot(active_index, num_classes=2).to(pred_rel.dtype)
    obj_weight = obj_weight[:, None, :, None]
    attach_weight = active_attach_target.detach()[:, None, None, :]
    valid_weight = batch["future_mask"][:, :, None, :].to(pred_rel.dtype)
    snr_weight = batch["alpha_bar"].detach().reshape(-1, 1, 1, 1).clamp(0.05, 1.0)
    rel_weight = obj_weight * attach_weight * valid_weight * snr_weight
    rel_sq = (pred_rel - target_rel).square()
    rel_loss = (rel_sq * rel_weight).sum() / (rel_weight.sum() * 3.0).clamp_min(1.0)

    cfg = spec.get("candidate_config", {})
    prior_weight = float(cfg.get("prior_loss_weight", 0.20))
    active_w = float(cfg.get("active_ce_weight", 0.35))
    attach_w = float(cfg.get("attachment_bce_weight", 1.0))
    rel_w = float(cfg.get("future_relative_weight", 0.20))
    prior_loss = prior_weight * (active_w * active_loss + attach_w * (0.5 * attach_loss_all + attach_loss_active)
                                 + rel_w * rel_loss)
    total = diffusion_loss + prior_loss
    return {"loss": total, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}


def build_model(spec):
    return AttachmentAwareAcquirePolicy(spec)
