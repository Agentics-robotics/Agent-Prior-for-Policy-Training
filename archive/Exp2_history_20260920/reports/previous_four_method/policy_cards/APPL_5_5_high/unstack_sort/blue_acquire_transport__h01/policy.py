import torch
from appl.public import DiffusionBackbone, epsilon_loss


class IdentityConditionedTopGraspPolicy(torch.nn.Module):
    """Learned diffusion policy with a blue-target object-centric grasp/transport prior."""

    def __init__(self, spec):
        super().__init__()
        training = spec["training"]
        if "normalizer" in spec:
            normalizer = spec["normalizer"]
        else:
            normalizer = spec["shared_normalizer"]
        if "candidate_config" in spec:
            candidate_config = spec["candidate_config"]
        else:
            candidate_config = {}

        self.horizon = int(training.get("horizon", 16))
        self.condition_dim = int(candidate_config.get("condition_dim", 256))
        self.aux_regression_weight = float(candidate_config.get("aux_regression_weight", 0.05))
        self.aux_classification_weight = float(candidate_config.get("aux_classification_weight", 0.01))

        self.register_buffer("obs_mean", torch.as_tensor(normalizer["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.as_tensor(normalizer["std"], dtype=torch.float32))

        # Shared object encoder: the same weights encode a block whether it is the
        # blue target or the red finished/context block. Learned role and identity
        # embeddings select the blue-target reuse of the top-grasp geometry.
        object_input_dim = 23
        object_dim = 64
        self.object_encoder = torch.nn.Sequential(
            torch.nn.Linear(object_input_dim, 64),
            torch.nn.Mish(),
            torch.nn.Linear(64, object_dim),
            torch.nn.Mish(),
        )
        self.target_role_embedding = torch.nn.Parameter(0.02 * torch.randn(object_dim))
        self.context_role_embedding = torch.nn.Parameter(0.02 * torch.randn(object_dim))
        self.blue_identity_embedding = torch.nn.Parameter(0.02 * torch.randn(object_dim))
        self.red_identity_embedding = torch.nn.Parameter(0.02 * torch.randn(object_dim))

        # 2 normalized observations + normalized finite difference + relative and
        # phase features + two 64-d object tokens.
        global_input_dim = 94 + 47 + 36 + 2 * object_dim
        self.condition_encoder = torch.nn.Sequential(
            torch.nn.Linear(global_input_dim, self.condition_dim),
            torch.nn.LayerNorm(self.condition_dim),
            torch.nn.Mish(),
            torch.nn.Linear(self.condition_dim, self.condition_dim),
            torch.nn.LayerNorm(self.condition_dim),
            torch.nn.Mish(),
        )

        self.backbone = DiffusionBackbone(self.condition_dim, training)

        # Auxiliary heads predict future blue lift/goal approach and grasp cues
        # from the same condition embedding used by the diffusion U-Net.
        self.aux_head = torch.nn.Sequential(
            torch.nn.Linear(self.condition_dim, 128),
            torch.nn.Mish(),
            torch.nn.Linear(128, self.horizon * 9),
        )

    def normalize_history(self, raw_history):
        mean = self.obs_mean.to(device=raw_history.device, dtype=raw_history.dtype)
        std = self.obs_std.to(device=raw_history.device, dtype=raw_history.dtype).clamp_min(1.0e-6)
        return (raw_history - mean) / std

    def make_object_inputs(self, obj_pose, obj_goal, prev_obj_pose, tcp_xyz, finger_summary):
        obj_xyz = obj_pose[:, 0:3]
        obj_quat = obj_pose[:, 3:7]
        obj_delta = obj_xyz - prev_obj_pose[:, 0:3]
        return torch.cat([
            obj_xyz / 0.50,
            obj_quat,
            obj_goal / 0.50,
            (tcp_xyz - obj_xyz) / 0.35,
            (obj_goal - obj_xyz) / 0.35,
            obj_delta / 0.05,
            finger_summary,
        ], dim=-1)

    def encode_condition(self, raw_history):
        norm = self.normalize_history(raw_history)
        norm_flat = norm.reshape(norm.shape[0], -1)
        norm_delta = norm[:, 1, :] - norm[:, 0, :]

        prev = raw_history[:, 0, :]
        cur = raw_history[:, 1, :]

        qpos = cur[:, 0:9]
        tcp = cur[:, 18:25]
        red = cur[:, 25:32]
        blue = cur[:, 32:39]
        red_goal = cur[:, 41:44]
        blue_goal = cur[:, 44:47]

        prev_tcp = prev[:, 18:25]
        prev_red = prev[:, 25:32]
        prev_blue = prev[:, 32:39]

        tcp_xyz = tcp[:, 0:3]
        red_xyz = red[:, 0:3]
        blue_xyz = blue[:, 0:3]
        width = qpos[:, 7:8] + qpos[:, 8:9]
        finger_summary = torch.cat([
            width / 0.08,
            qpos[:, 7:8] / 0.04,
            qpos[:, 8:9] / 0.04,
            (0.08 - width) / 0.08,
        ], dim=-1)

        tcp_blue = (tcp_xyz - blue_xyz) / 0.35
        blue_to_goal = (blue_goal - blue_xyz) / 0.35
        blue_red = (blue_xyz - red_xyz) / 0.35
        red_to_goal = (red_goal - red_xyz) / 0.35
        tcp_to_blue_goal = (tcp_xyz - blue_goal) / 0.35
        tcp_delta = (tcp_xyz - prev_tcp[:, 0:3]) / 0.05
        blue_delta = (blue_xyz - prev_blue[:, 0:3]) / 0.05
        red_delta = (red_xyz - prev_red[:, 0:3]) / 0.05

        dist_tcp_blue_xy = torch.linalg.norm(tcp_xyz[:, 0:2] - blue_xyz[:, 0:2], dim=-1, keepdim=True)
        dist_blue_goal_xy = torch.linalg.norm(blue_goal[:, 0:2] - blue_xyz[:, 0:2], dim=-1, keepdim=True)
        dist_red_goal = torch.linalg.norm(red_goal - red_xyz, dim=-1, keepdim=True)
        centered = torch.exp(-0.5 * (dist_tcp_blue_xy / 0.035).square())
        near_goal = torch.exp(-0.5 * (dist_blue_goal_xy / 0.08).square())
        red_done = torch.exp(-0.5 * (dist_red_goal / 0.04).square())
        z_offset = (tcp_xyz[:, 2:3] - blue_xyz[:, 2:3]) / 0.30
        blue_lift = (blue_xyz[:, 2:3] - 0.02) / 0.30
        closedness = (0.08 - width) / 0.08
        high_carry = torch.sigmoid((blue_xyz[:, 2:3] - 0.12) / 0.03)
        low_contact = torch.exp(-0.5 * (z_offset / 0.06).square())
        phase_features = torch.cat([
            dist_tcp_blue_xy / 0.35,
            z_offset,
            blue_lift,
            near_goal,
            red_done,
            centered,
            closedness,
            high_carry * low_contact,
        ], dim=-1)

        rel_features = torch.cat([
            tcp_blue,
            blue_to_goal,
            blue_red,
            red_to_goal,
            tcp_to_blue_goal,
            tcp_delta,
            blue_delta,
            red_delta,
            finger_summary,
            phase_features,
        ], dim=-1)

        blue_input = self.make_object_inputs(blue, blue_goal, prev_blue, tcp_xyz, finger_summary)
        red_input = self.make_object_inputs(red, red_goal, prev_red, tcp_xyz, finger_summary)
        blue_token = self.object_encoder(blue_input) + self.target_role_embedding + self.blue_identity_embedding
        red_token = self.object_encoder(red_input) + self.context_role_embedding + self.red_identity_embedding

        features = torch.cat([norm_flat, norm_delta, rel_features, blue_token, red_token], dim=-1)
        return self.condition_encoder(features)

    def forward(self, noisy_action, timestep, raw_history):
        condition = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, condition)

    def predict_aux(self, raw_history):
        condition = self.encode_condition(raw_history)
        return self.aux_head(condition).reshape(raw_history.shape[0], self.horizon, 9)

    def auxiliary_loss(self, batch):
        raw_obs = batch["raw_obs"]
        future = batch["future_obs"]
        if "future_mask" in batch:
            mask = batch["future_mask"]
        else:
            mask = batch["mask"]
        if future.shape[1] != self.horizon:
            horizon = min(future.shape[1], self.horizon)
            aux = self.predict_aux(raw_obs)[:, :horizon, :]
            future = future[:, :horizon, :]
            mask = mask[:, :horizon, :]
        else:
            aux = self.predict_aux(raw_obs)

        cur = raw_obs[:, 1, :]
        future_blue = future[:, :, 32:35]
        future_tcp = future[:, :, 18:21]
        future_qpos = future[:, :, 0:9]
        future_blue_goal = future[:, :, 44:47]
        cur_blue_z = cur[:, None, 34:35]

        target_reg = torch.cat([
            future_blue[:, :, 2:3] / 0.30,
            (future_blue_goal[:, :, 0:2] - future_blue[:, :, 0:2]) / 0.35,
            (future_blue[:, :, 2:3] - cur_blue_z) / 0.30,
            (future_tcp - future_blue) / 0.25,
        ], dim=-1)
        pred_reg = aux[:, :, 0:7]
        denom_reg = (mask.sum() * target_reg.shape[-1]).clamp_min(1.0)
        reg_loss = ((pred_reg - target_reg).square() * mask).sum() / denom_reg

        future_width = future_qpos[:, :, 7:8] + future_qpos[:, :, 8:9]
        lifted_label = ((future_blue[:, :, 2:3] > 0.06) & (future_width < 0.055)).to(dtype=aux.dtype)
        closed_label = (future_width < 0.055).to(dtype=aux.dtype)
        lift_bce = torch.nn.functional.binary_cross_entropy_with_logits(aux[:, :, 7:8], lifted_label, reduction="none")
        closed_bce = torch.nn.functional.binary_cross_entropy_with_logits(aux[:, :, 8:9], closed_label, reduction="none")
        denom_cls = mask.sum().clamp_min(1.0)
        cls_loss = ((lift_bce + closed_bce) * mask).sum() / denom_cls

        return self.aux_regression_weight * reg_loss + self.aux_classification_weight * cls_loss


def build_model(spec):
    return IdentityConditionedTopGraspPolicy(spec)


def compute_loss(model, batch, spec):
    predicted_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = epsilon_loss(predicted_noise, batch["noise"], batch["mask"])
    prior_loss = model.auxiliary_loss(batch)
    loss = diffusion_loss + prior_loss
    return {
        "loss": loss,
        "diffusion_loss": diffusion_loss,
        "prior_loss": prior_loss,
    }
