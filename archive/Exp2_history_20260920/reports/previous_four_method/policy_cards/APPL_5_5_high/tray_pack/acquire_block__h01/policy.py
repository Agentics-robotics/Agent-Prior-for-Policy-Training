import torch
import appl.public as public


# This policy implements the assigned active-object-relative acquisition prior as a
# learned conditional diffusion model. The diffusion sampler, optimizer and data
# pipeline are supplied by the framework; this file defines the trainable
# representation, backbone conditioning, and auxiliary differentiable losses.


POSITION_SCALE_ACTIVE = 0.40
POSITION_SCALE_INACTIVE = 0.60
HEIGHT_SCALE = 0.30
FINGER_OPEN_WIDTH = 0.080
FINGER_CLOSED_WIDTH = 0.0365


def make_mlp(sizes, final_activation=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or final_activation:
            layers.append(torch.nn.SiLU())
    return torch.nn.Sequential(*layers)


def quat_conjugate(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_multiply(a, b):
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def norm_xy(v):
    return torch.sqrt(torch.clamp((v * v).sum(dim=-1, keepdim=True), min=1.0e-12))


class ActiveObjectRelativeDiffusionPolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = spec["normalizer"]
        self.register_buffer("obs_mean", torch.tensor(n["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.tensor(n["std"], dtype=torch.float32))

        self.history = int(spec.get("training", {}).get("observation_steps", 2))
        self.obs_dim = int(spec.get("observation_dimension", 47))
        condition_dim = 256

        # Per active-object candidate, for each of the two causal observations:
        # qpos_n(9), qvel_n(9), tcp-active(3), goal-active(3), inactive-active(3),
        # tcp-goal(3), active xyz at coarse scale(3), tcp quat(4), active quat(4),
        # active-frame tcp quat(4), finger width(1), goal distance(1), active height(1).
        self.candidate_step_dim = 48
        candidate_dim = self.history * self.candidate_step_dim
        abs_dim = self.history * self.obs_dim
        role_dim = self.obs_dim + 21

        # The same candidate encoder is applied to the red-active and blue-active
        # hypotheses. Weight sharing is the architectural prior.
        self.candidate_encoder = make_mlp([candidate_dim, 192, 128, 128])
        self.absolute_encoder = make_mlp([abs_dim, 192, 128, 96])
        self.role_mlp = make_mlp([role_dim, 128, 64, 1])
        self.condition_mlp = make_mlp([96 + 128 + 128 + 5, 256, condition_dim])

        self.phase_head = make_mlp([condition_dim, 96, 2])
        self.geometry_head = make_mlp([condition_dim, 96, 6])
        self.backbone = public.DiffusionBackbone(condition_dim, spec["training"])
        self.last_aux = None

    def normalize_obs(self, raw):
        return (raw - self.obs_mean.to(device=raw.device, dtype=raw.dtype)) / self.obs_std.to(device=raw.device, dtype=raw.dtype)

    def candidate_features(self, raw_history, norm_history, active_slice_start, inactive_slice_start, goal_slice_start):
        qpos_n = norm_history[..., 0:9]
        qvel_n = norm_history[..., 9:18]
        tcp_xyz = raw_history[..., 18:21]
        tcp_q = raw_history[..., 21:25]
        active_xyz = raw_history[..., active_slice_start:active_slice_start + 3]
        active_q = raw_history[..., active_slice_start + 3:active_slice_start + 7]
        inactive_xyz = raw_history[..., inactive_slice_start:inactive_slice_start + 3]
        goal_xyz = raw_history[..., goal_slice_start:goal_slice_start + 3]

        rel_tcp = (tcp_xyz - active_xyz) / POSITION_SCALE_ACTIVE
        rel_goal = (goal_xyz - active_xyz) / POSITION_SCALE_ACTIVE
        rel_inactive = (inactive_xyz - active_xyz) / POSITION_SCALE_INACTIVE
        rel_tcp_goal = (tcp_xyz - goal_xyz) / POSITION_SCALE_ACTIVE
        active_coarse = active_xyz / POSITION_SCALE_INACTIVE
        rel_q = quat_multiply(quat_conjugate(active_q), tcp_q)

        finger_width = (raw_history[..., 7:8] + raw_history[..., 8:9]) / FINGER_OPEN_WIDTH
        goal_dist = norm_xy(goal_xyz[..., :2] - active_xyz[..., :2]) / POSITION_SCALE_ACTIVE
        active_h = active_xyz[..., 2:3] / HEIGHT_SCALE

        per_step = torch.cat([
            qpos_n, qvel_n, rel_tcp, rel_goal, rel_inactive, rel_tcp_goal,
            active_coarse, tcp_q, active_q, rel_q, finger_width, goal_dist, active_h,
        ], dim=-1)
        return per_step.reshape(per_step.shape[0], -1)

    def role_input(self, raw_history, norm_history):
        cur = raw_history[:, -1]
        cur_n = norm_history[:, -1]
        tcp = cur[:, 18:21]
        red = cur[:, 25:28]
        blue = cur[:, 32:35]
        red_goal = cur[:, 41:44]
        blue_goal = cur[:, 44:47]
        finger_width = cur[:, 7:8] + cur[:, 8:9]

        red_goal_delta = (red_goal - red) / POSITION_SCALE_ACTIVE
        blue_goal_delta = (blue_goal - blue) / POSITION_SCALE_ACTIVE
        tcp_red = (tcp - red) / POSITION_SCALE_ACTIVE
        tcp_blue = (tcp - blue) / POSITION_SCALE_ACTIVE
        red_dist = norm_xy(red_goal[:, :2] - red[:, :2])
        blue_dist = norm_xy(blue_goal[:, :2] - blue[:, :2])
        red_height_err = torch.abs(red[:, 2:3] - red_goal[:, 2:3])
        blue_height_err = torch.abs(blue[:, 2:3] - blue_goal[:, 2:3])
        red_goal_score = torch.exp(-((red_dist / 0.07) ** 2 + (red_height_err / 0.06) ** 2))
        blue_goal_score = torch.exp(-((blue_dist / 0.07) ** 2 + (blue_height_err / 0.06) ** 2))
        stage_features = torch.cat([
            red_goal_delta, blue_goal_delta, tcp_red, tcp_blue,
            red_dist / 0.5, blue_dist / 0.5,
            red[:, 2:3] / HEIGHT_SCALE, blue[:, 2:3] / HEIGHT_SCALE,
            finger_width / FINGER_OPEN_WIDTH,
            red_height_err / 0.2, blue_height_err / 0.2,
            red_goal_score, blue_goal_score,
        ], dim=-1)
        return torch.cat([cur_n, stage_features], dim=-1)

    def condition(self, raw_history):
        raw_history = raw_history.to(dtype=self.obs_mean.dtype) if raw_history.dtype != self.obs_mean.dtype else raw_history
        norm_history = self.normalize_obs(raw_history)
        abs_flat = norm_history.reshape(norm_history.shape[0], -1)
        abs_enc = self.absolute_encoder(abs_flat)

        red_feat = self.candidate_features(raw_history, norm_history, 25, 32, 41)
        blue_feat = self.candidate_features(raw_history, norm_history, 32, 25, 44)
        red_enc = self.candidate_encoder(red_feat)
        blue_enc = self.candidate_encoder(blue_feat)

        role_in = self.role_input(raw_history, norm_history)
        role_logit = self.role_mlp(role_in)
        p_blue = torch.sigmoid(role_logit)
        active_enc = (1.0 - p_blue) * red_enc + p_blue * blue_enc
        inactive_enc = p_blue * red_enc + (1.0 - p_blue) * blue_enc

        cur = raw_history[:, -1]
        red = cur[:, 25:28]
        blue = cur[:, 32:35]
        red_goal = cur[:, 41:44]
        blue_goal = cur[:, 44:47]
        red_goal_score = torch.exp(-((norm_xy(red_goal[:, :2] - red[:, :2]) / 0.07) ** 2 +
                                     (torch.abs(red[:, 2:3] - red_goal[:, 2:3]) / 0.06) ** 2))
        blue_goal_score = torch.exp(-((norm_xy(blue_goal[:, :2] - blue[:, :2]) / 0.07) ** 2 +
                                      (torch.abs(blue[:, 2:3] - blue_goal[:, 2:3]) / 0.06) ** 2))
        finger_width = (cur[:, 7:8] + cur[:, 8:9]) / FINGER_OPEN_WIDTH
        role_summary = torch.cat([p_blue, role_logit / 5.0, red_goal_score, blue_goal_score, finger_width], dim=-1)

        cond = self.condition_mlp(torch.cat([abs_enc, active_enc, inactive_enc, role_summary], dim=-1))
        phase_logits = self.phase_head(cond)
        geom_pred = self.geometry_head(cond)
        self.last_aux = {
            "role_logit": role_logit,
            "p_blue": p_blue,
            "phase_logits": phase_logits,
            "geom_pred": geom_pred,
        }
        return cond

    def forward(self, noisy_action, timestep, raw_history):
        cond = self.condition(raw_history)
        return self.backbone(noisy_action, timestep, cond)


def blue_active_label(raw_history):
    cur = raw_history[:, -1]
    red = cur[:, 25:28]
    red_goal = cur[:, 41:44]
    dist_xy = norm_xy(red_goal[:, :2] - red[:, :2])
    near_xy = dist_xy < 0.075
    placed_height = (red[:, 2:3] > 0.025) & (red[:, 2:3] < 0.090)
    return (near_xy & placed_height).to(dtype=raw_history.dtype)


def select_active(raw_history, blue_label):
    cur = raw_history[:, -1]
    red = cur[:, 25:28]
    blue = cur[:, 32:35]
    red_goal = cur[:, 41:44]
    blue_goal = cur[:, 44:47]
    active = (1.0 - blue_label) * red + blue_label * blue
    active_goal = (1.0 - blue_label) * red_goal + blue_label * blue_goal
    tcp = cur[:, 18:21]
    return active, active_goal, tcp


def build_model(spec):
    return ActiveObjectRelativeDiffusionPolicy(spec)


def compute_loss(model, batch, spec):
    pred_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = public.epsilon_loss(pred_noise, batch["noise"], batch["mask"])

    aux = model.last_aux
    blue_label = blue_active_label(batch["raw_obs"]).to(device=pred_noise.device, dtype=pred_noise.dtype)
    role_loss = torch.nn.functional.binary_cross_entropy_with_logits(aux["role_logit"], blue_label)

    active, active_goal, tcp = select_active(batch["raw_obs"].to(device=pred_noise.device, dtype=pred_noise.dtype), blue_label)
    finger_width = batch["raw_obs"][:, -1, 7:8].to(device=pred_noise.device, dtype=pred_noise.dtype) + \
        batch["raw_obs"][:, -1, 8:9].to(device=pred_noise.device, dtype=pred_noise.dtype)
    lift_target = torch.clamp((active[:, 2:3] - 0.02) / 0.25, 0.0, 1.0)
    closed_target = torch.clamp((FINGER_OPEN_WIDTH - finger_width) / (FINGER_OPEN_WIDTH - FINGER_CLOSED_WIDTH), 0.0, 1.0)
    phase_target = torch.cat([lift_target, closed_target], dim=-1)
    phase_loss = torch.nn.functional.mse_loss(torch.sigmoid(aux["phase_logits"]), phase_target)

    geom_target = torch.cat([(tcp - active) / POSITION_SCALE_ACTIVE,
                             (active_goal - active) / POSITION_SCALE_ACTIVE], dim=-1)
    geom_loss = torch.nn.functional.mse_loss(aux["geom_pred"], geom_target)

    prior_loss = 0.05 * role_loss + 0.02 * phase_loss + 0.01 * geom_loss
    loss = diffusion_loss + prior_loss
    return {
        "loss": loss,
        "diffusion_loss": diffusion_loss,
        "prior_loss": prior_loss,
    }
