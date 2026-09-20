import math
import torch
from appl.public import DiffusionBackbone, normalize_observation, epsilon_loss


class WaypointConditionedDiffusion(torch.nn.Module):
    """Learned DDPM epsilon model with an explicit clearance-waypoint prior."""

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.obs_dim = int(spec.get("observation_dimension", 47))
        self.hist = int(spec.get("training", {}).get("observation_steps", 2))
        self.feature_dim = self.hist * self.obs_dim + 72
        self.cond_dim = 256

        self.waypoint_head = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, 192),
            torch.nn.Mish(),
            torch.nn.Linear(192, 128),
            torch.nn.Mish(),
            torch.nn.Linear(128, 8),
        )
        self.condition_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim + 8, 256),
            torch.nn.Mish(),
            torch.nn.Linear(256, self.cond_dim),
            torch.nn.LayerNorm(self.cond_dim),
        )
        self.backbone = DiffusionBackbone(self.cond_dim, spec["training"])

    def slice_field(self, x, name):
        fields = self.spec.get("fields", {})
        if name in fields:
            a, b = fields[name]
            return x[..., int(a):int(b)]
        defaults = {
            "qpos": (0, 9),
            "qvel": (9, 18),
            "tcp_pose": (18, 25),
            "red_pose": (25, 32),
            "blue_pose": (32, 39),
            "red_goal": (41, 44),
            "blue_goal": (44, 47),
        }
        a, b = defaults[name]
        return x[..., a:b]

    def geometry_features(self, raw_history):
        last = raw_history[:, -1, :]
        prev = raw_history[:, 0, :]
        qpos = self.slice_field(last, "qpos")
        tcp = self.slice_field(last, "tcp_pose")[:, :3]
        tcp_prev = self.slice_field(prev, "tcp_pose")[:, :3]
        red = self.slice_field(last, "red_pose")[:, :3]
        blue = self.slice_field(last, "blue_pose")[:, :3]
        red_goal = self.slice_field(last, "red_goal")
        blue_goal = self.slice_field(last, "blue_goal")

        grip_sum = (qpos[:, 7:8] + qpos[:, 8:9]).clamp(0.0, 0.10)
        red_goal_vec = red_goal - red
        blue_goal_vec = blue_goal - blue
        red_goal_xy = torch.linalg.vector_norm(red_goal_vec[:, :2], dim=-1, keepdim=True)
        blue_goal_xy = torch.linalg.vector_norm(blue_goal_vec[:, :2], dim=-1, keepdim=True)

        red_done_soft = torch.sigmoid((0.070 - red_goal_xy) / 0.015) * torch.sigmoid((0.080 - red[:, 2:3]) / 0.015)
        red_done_hard = ((red_goal_xy < 0.075) & (red[:, 2:3] < 0.080)).to(raw_history.dtype)
        w_blue = red_done_hard
        w_red = 1.0 - w_blue
        active = w_red * red + w_blue * blue
        active_goal = w_red * red_goal + w_blue * blue_goal
        inactive = w_red * blue + w_blue * red
        inactive_goal = w_red * blue_goal + w_blue * red_goal

        off = torch.zeros_like(active)
        off[:, 0:1] = -0.009
        tcp_object = active + off
        tcp_goal = active_goal + off

        z_hover = torch.full_like(active[:, 2:3], 0.270)
        z_grasp = torch.clamp(active[:, 2:3] + 0.004, min=0.024, max=0.070)
        z_lift = torch.full_like(active[:, 2:3], 0.285)
        z_goal = torch.full_like(active[:, 2:3], 0.300)
        wp_hover = torch.cat([tcp_object[:, :2], z_hover], dim=-1)
        wp_grasp = torch.cat([tcp_object[:, :2], z_grasp], dim=-1)
        wp_lift = torch.cat([tcp_object[:, :2], z_lift], dim=-1)
        wp_goal = torch.cat([tcp_goal[:, :2], z_goal], dim=-1)
        wps = torch.stack([wp_hover, wp_grasp, wp_lift, wp_goal], dim=1)

        tcp_obj_xy = torch.linalg.vector_norm((tcp - tcp_object)[:, :2], dim=-1, keepdim=True)
        active_goal_vec = active_goal - active
        active_goal_xy = torch.linalg.vector_norm(active_goal_vec[:, :2], dim=-1, keepdim=True)

        open_score = torch.sigmoid((grip_sum - 0.060) / 0.006)
        lifted_score = torch.sigmoid((active[:, 2:3] - 0.095) / 0.025)
        far_xy_score = torch.sigmoid((tcp_obj_xy - 0.045) / 0.015)
        near_goal_score = torch.sigmoid((0.080 - active_goal_xy) / 0.020)

        phase_hover = open_score * (1.0 - lifted_score) * far_xy_score
        phase_grasp = open_score * (1.0 - lifted_score) * (1.0 - far_xy_score)
        phase_lift = (1.0 - open_score) * (1.0 - lifted_score) * (1.0 - near_goal_score)
        phase_goal = lifted_score + (1.0 - open_score) * near_goal_score
        phase_raw = torch.cat([phase_hover, phase_grasp, phase_lift, phase_goal], dim=-1).clamp_min(1.0e-4)
        phase = phase_raw / phase_raw.sum(dim=-1, keepdim=True)
        analytic_next = (wps * phase.unsqueeze(-1)).sum(dim=1)
        phase_values = torch.as_tensor([0.05, 0.30, 0.62, 0.90], device=raw_history.device, dtype=raw_history.dtype).view(1, 4)
        progress = (phase * phase_values).sum(dim=-1, keepdim=True)

        norm_hist = normalize_observation(raw_history, self.spec).reshape(raw_history.shape[0], -1)
        pos_scale = 0.50
        rel_scale = 0.30
        scalar_features = torch.cat([
            red_goal_xy / rel_scale,
            blue_goal_xy / rel_scale,
            tcp_obj_xy / rel_scale,
            active_goal_xy / rel_scale,
            active[:, 2:3] / rel_scale,
            tcp[:, 2:3] / rel_scale,
        ], dim=-1)
        role = torch.cat([w_red, w_blue, red_done_soft, grip_sum / 0.080], dim=-1)
        abs_pos = torch.cat([tcp, red, blue, red_goal, blue_goal], dim=-1) / pos_scale
        rels = torch.cat([
            (red - tcp) / rel_scale,
            (blue - tcp) / rel_scale,
            red_goal_vec / rel_scale,
            blue_goal_vec / rel_scale,
            (active - tcp) / rel_scale,
            active_goal_vec / rel_scale,
            (active_goal - tcp) / rel_scale,
            (inactive - tcp) / rel_scale,
            (inactive_goal - inactive) / rel_scale,
        ], dim=-1)
        wp_rel = ((wps - tcp.unsqueeze(1)) / rel_scale).reshape(raw_history.shape[0], -1)
        tcp_delta = (tcp - tcp_prev) / 0.10
        geo = torch.cat([role, abs_pos, rels, scalar_features, wp_rel, phase, progress, tcp_delta], dim=-1)
        features = torch.cat([norm_hist, geo], dim=-1)
        return features, analytic_next, phase, progress, wps, active, active_goal

    def condition(self, raw_history):
        features, analytic_next, phase, progress, wps, active, active_goal = self.geometry_features(raw_history)
        head = self.waypoint_head(features)
        residual = 0.060 * torch.tanh(head[:, 0:3])
        pred_next = analytic_next + residual
        pred_phase = torch.softmax(head[:, 3:7], dim=-1)
        pred_progress = torch.sigmoid(head[:, 7:8])
        tcp = self.slice_field(raw_history[:, -1, :], "tcp_pose")[:, :3]
        cond_extra = torch.cat([(pred_next - tcp) / 0.30, pred_phase, pred_progress], dim=-1)
        cond = self.condition_encoder(torch.cat([features, cond_extra], dim=-1))
        return cond

    def aux_predictions(self, raw_history):
        features, analytic_next, phase, progress, wps, active, active_goal = self.geometry_features(raw_history)
        head = self.waypoint_head(features)
        pred_next = analytic_next + 0.060 * torch.tanh(head[:, 0:3])
        return {
            "head": head,
            "pred_next": pred_next,
            "target_next": analytic_next.detach(),
            "target_phase": phase.detach(),
            "target_progress": progress.detach(),
            "active_z": active[:, 2:3].detach(),
            "goal_xy_dist": torch.linalg.vector_norm((active_goal - active)[:, :2], dim=-1, keepdim=True).detach(),
        }

    def forward(self, noisy_action, timestep, raw_history):
        cond = self.condition(raw_history)
        return self.backbone(noisy_action, timestep, cond)


def build_model(spec):
    return WaypointConditionedDiffusion(spec)


def compute_loss(model, batch, spec):
    pred = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    mask = batch["mask"]
    diffusion_loss = epsilon_loss(pred, batch["noise"], mask)

    aux = model.aux_predictions(batch["raw_obs"])
    waypoint_loss = ((aux["pred_next"] - aux["target_next"]) / 0.10).square().mean()
    log_phase = torch.log_softmax(aux["head"][:, 3:7], dim=-1)
    phase_loss = -(aux["target_phase"] * log_phase).sum(dim=-1).mean()
    progress_loss = (torch.sigmoid(aux["head"][:, 7:8]) - aux["target_progress"]).square().mean()

    carry_weight = torch.sigmoid((aux["active_z"] - 0.070) / 0.020) + torch.sigmoid((0.12 - aux["goal_xy_dist"]) / 0.030)
    carry_weight = carry_weight.clamp(0.0, 1.0)
    clearance_loss = (carry_weight * torch.relu(0.115 - aux["pred_next"][:, 2:3]).square() / (0.05 * 0.05)).mean()

    alpha = batch["alpha_bar"].reshape(-1, 1, 1).to(pred.dtype).clamp_min(1.0e-5)
    clean_action_hat = (batch["noisy_action"] - torch.sqrt(1.0 - alpha) * pred) / torch.sqrt(alpha)
    gripper_err = (clean_action_hat[:, :, 7:8] - batch["encoded_action"][:, :, 7:8]).square()
    gripper_loss = (gripper_err * mask * alpha).sum() / (mask.sum() * alpha.mean().clamp_min(1.0e-5)).clamp_min(1.0)

    prior_loss = 0.05 * waypoint_loss + 0.01 * phase_loss + 0.02 * progress_loss + 0.02 * clearance_loss + 0.01 * gripper_loss
    total = diffusion_loss + prior_loss
    return {"loss": total, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
