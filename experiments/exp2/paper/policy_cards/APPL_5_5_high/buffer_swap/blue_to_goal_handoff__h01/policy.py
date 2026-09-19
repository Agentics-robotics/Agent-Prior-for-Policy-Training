import torch
from appl.public import DiffusionBackbone, epsilon_loss


class ReflectionBlueTransferPolicy(torch.nn.Module):
    """Learned diffusion policy with an active-object/reflection representation prior.

    The model keeps the fixed action diffusion interface. The prior is encoded in
    the conditioning network: a shared relational encoder is applied both to the
    demonstrated blue-active view and to a y-reflected, red/blue-swapped view.
    The diffusion U-Net then predicts normalized action noise from this learned
    condition. A small auxiliary task-space predictor shares the condition and is
    used only during training to make the spatial representation predictive and
    approximately reflection-consistent.
    """

    def __init__(self, spec):
        super().__init__()
        self.horizon = int(spec["training"]["horizon"])
        mean = torch.as_tensor(spec["normalizer"]["mean"], dtype=torch.float32)
        std = torch.as_tensor(spec["normalizer"]["std"], dtype=torch.float32)
        self.register_buffer("obs_mean", mean)
        self.register_buffer("obs_std", std)
        self.register_buffer("xyz_center", torch.tensor([-0.35, 0.0, 0.16], dtype=torch.float32))
        self.register_buffer("xyz_scale", torch.tensor([0.50, 0.50, 0.30], dtype=torch.float32))
        self.register_buffer("rel_scale", torch.tensor([0.40, 0.40, 0.30], dtype=torch.float32))

        self.raw_mlp = torch.nn.Sequential(
            torch.nn.Linear(94, 160),
            torch.nn.Mish(),
            torch.nn.Linear(160, 96),
            torch.nn.Mish(),
        )
        self.rel_mlp = torch.nn.Sequential(
            torch.nn.Linear(108, 160),
            torch.nn.Mish(),
            torch.nn.Linear(160, 80),
            torch.nn.Mish(),
        )
        self.phase_mlp = torch.nn.Sequential(
            torch.nn.Linear(25, 64),
            torch.nn.Mish(),
            torch.nn.Linear(64, 32),
            torch.nn.Mish(),
        )
        self.fuse = torch.nn.Sequential(
            torch.nn.Linear(288, 256),
            torch.nn.Mish(),
            torch.nn.Linear(256, 256),
            torch.nn.Mish(),
        )
        self.backbone = DiffusionBackbone(256, spec["training"])
        self.aux_head = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.Mish(),
            torch.nn.Linear(256, self.horizon * 10),
        )

    def norm_obs(self, raw_history):
        return (raw_history - self.obs_mean.to(raw_history.dtype)) / self.obs_std.to(raw_history.dtype)

    def norm_xyz(self, xyz):
        return (xyz - self.xyz_center.to(xyz.dtype)) / self.xyz_scale.to(xyz.dtype)

    def relative_xyz(self, a, b):
        return (a - b) / self.rel_scale.to(a.dtype)

    def mirror_swap_history(self, raw_history):
        """Reflect y and swap red/blue object and goal fields for the prior branch.

        Joint coordinates are intentionally not transformed: the repository gives
        no robot kinematic mirror map, and actions remain absolute joint targets.
        """
        x = raw_history.clone()
        for idx in (19, 26, 33, 42, 45):
            x[..., idx] = -x[..., idx]
        red = x[..., 25:32].clone()
        blue = x[..., 32:39].clone()
        x[..., 25:32] = blue
        x[..., 32:39] = red
        rg = x[..., 41:44].clone()
        bg = x[..., 44:47].clone()
        x[..., 41:44] = bg
        x[..., 44:47] = rg
        return x

    def relation_features(self, raw_history):
        feats = []
        for k in range(raw_history.shape[1]):
            s = raw_history[:, k]
            qpos = s[:, 0:9]
            qvel = s[:, 9:18]
            tcp = s[:, 18:21]
            tcp_q = s[:, 21:25]
            red = s[:, 25:28]
            red_q = s[:, 28:32]
            blue = s[:, 32:35]
            blue_q = s[:, 35:39]
            red_goal = s[:, 41:44]
            blue_goal = s[:, 44:47]
            active = blue
            active_q = blue_q
            active_goal = blue_goal
            passive = red
            passive_q = red_q
            passive_goal = red_goal
            dist = torch.cat([
                torch.linalg.norm(tcp - active, dim=-1, keepdim=True) / 0.40,
                torch.linalg.norm(active_goal - active, dim=-1, keepdim=True) / 0.50,
                torch.linalg.norm(tcp - passive, dim=-1, keepdim=True) / 0.40,
                torch.linalg.norm(passive_goal - passive, dim=-1, keepdim=True) / 0.50,
                torch.linalg.norm(active - passive, dim=-1, keepdim=True) / 0.50,
            ], dim=-1)
            finger = (qpos[:, 7:9] - 0.025) / 0.025
            finger_v = qvel[:, 7:9] / 0.20
            step_feat = torch.cat([
                self.relative_xyz(tcp, active),
                self.relative_xyz(active_goal, active),
                self.relative_xyz(passive, active),
                self.relative_xyz(passive_goal, active_goal),
                self.relative_xyz(tcp, passive),
                self.relative_xyz(passive_goal, passive),
                self.norm_xyz(active),
                self.norm_xyz(passive),
                self.norm_xyz(tcp),
                self.norm_xyz(active_goal),
                self.norm_xyz(passive_goal),
                tcp_q,
                active_q,
                passive_q,
                finger,
                finger_v,
                dist,
            ], dim=-1)
            feats.append(step_feat)
        return torch.cat(feats, dim=-1)

    def phase_features(self, raw_history):
        s0 = raw_history[:, 0]
        s1 = raw_history[:, 1]
        tcp0 = s0[:, 18:21]
        tcp1 = s1[:, 18:21]
        red0 = s0[:, 25:28]
        red1 = s1[:, 25:28]
        blue0 = s0[:, 32:35]
        blue1 = s1[:, 32:35]
        red_goal = s1[:, 41:44]
        blue_goal = s1[:, 44:47]
        qpos = s1[:, 0:9]
        qvel = s1[:, 9:18]
        finger_width = (qpos[:, 7:8] + qpos[:, 8:9])
        open_score = (finger_width - 0.036) / 0.044
        closed_score = (0.040 - finger_width) / 0.040
        small_scale = torch.tensor([0.05, 0.05, 0.05], device=raw_history.device, dtype=raw_history.dtype)
        tcp_delta = (tcp1 - tcp0) / small_scale
        blue_delta = (blue1 - blue0) / small_scale
        red_delta = (red1 - red0) / small_scale
        xy_scale = torch.tensor([0.40, 0.40], device=raw_history.device, dtype=raw_history.dtype)
        phase = torch.cat([
            (finger_width - 0.04) / 0.04,
            open_score,
            closed_score,
            (tcp1[:, 2:3] - 0.16) / 0.18,
            (blue1[:, 2:3] - 0.10) / 0.20,
            (red1[:, 2:3] - 0.10) / 0.20,
            torch.linalg.norm(tcp1 - blue1, dim=-1, keepdim=True) / 0.40,
            torch.linalg.norm(blue_goal - blue1, dim=-1, keepdim=True) / 0.50,
            torch.linalg.norm(tcp1 - red1, dim=-1, keepdim=True) / 0.40,
            torch.linalg.norm(red_goal - red1, dim=-1, keepdim=True) / 0.50,
            (blue_goal[:, :2] - blue1[:, :2]) / xy_scale,
            (red_goal[:, :2] - red1[:, :2]) / xy_scale,
            tcp_delta,
            blue_delta,
            red_delta,
            qvel[:, 7:9] / 0.20,
        ], dim=-1)
        return phase

    def encode_condition(self, raw_history):
        if raw_history.dtype != self.obs_mean.dtype:
            raw_history = raw_history.to(dtype=self.obs_mean.dtype)
        raw_flat = self.norm_obs(raw_history).reshape(raw_history.shape[0], -1)
        raw_code = self.raw_mlp(raw_flat)
        rel_code = self.rel_mlp(self.relation_features(raw_history))
        mirror = self.mirror_swap_history(raw_history)
        mirror_code = self.rel_mlp(self.relation_features(mirror))
        phase_code = self.phase_mlp(self.phase_features(raw_history))
        return self.fuse(torch.cat([raw_code, rel_code, mirror_code, phase_code], dim=-1))

    def forward(self, noisy_action, timestep, raw_history):
        cond = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, cond)

    def predict_aux(self, raw_history):
        cond = self.encode_condition(raw_history)
        return self.aux_head(cond).reshape(raw_history.shape[0], self.horizon, 10)

    def aux_from_future_obs(self, future_obs):
        tcp = self.norm_xyz(future_obs[..., 18:21])
        red = self.norm_xyz(future_obs[..., 25:28])
        blue = self.norm_xyz(future_obs[..., 32:35])
        finger = ((future_obs[..., 7:8] + future_obs[..., 8:9]) - 0.04) / 0.04
        return torch.cat([tcp, blue, red, finger], dim=-1)

    def mirror_aux(self, aux):
        y = aux.clone()
        y[..., 1] = -y[..., 1]
        y[..., 4] = -y[..., 4]
        y[..., 7] = -y[..., 7]
        blue = y[..., 3:6].clone()
        red = y[..., 6:9].clone()
        y[..., 3:6] = red
        y[..., 6:9] = blue
        return y


def build_model(spec):
    return ReflectionBlueTransferPolicy(spec)


def compute_loss(model, batch, spec):
    pred = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = epsilon_loss(pred, batch["noise"], batch["mask"])

    aux_pred = model.predict_aux(batch["raw_obs"])
    aux_target = model.aux_from_future_obs(batch["future_obs"])
    mask = batch["future_mask"]
    aux_mse = ((aux_pred - aux_target).square() * mask).sum() / (mask.sum() * aux_pred.shape[-1]).clamp_min(1.0)

    mirror_obs = model.mirror_swap_history(batch["raw_obs"])
    aux_mirror = model.predict_aux(mirror_obs)
    aux_consistency_target = model.mirror_aux(aux_pred)
    consistency = ((aux_mirror - aux_consistency_target).square() * mask).sum() / (mask.sum() * aux_pred.shape[-1]).clamp_min(1.0)

    prior_loss = 0.03 * aux_mse + 0.01 * consistency
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
