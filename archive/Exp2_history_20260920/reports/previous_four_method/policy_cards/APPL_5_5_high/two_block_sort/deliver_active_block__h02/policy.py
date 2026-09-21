import torch
from appl.public import DiffusionBackbone, epsilon_loss


STAGE_LIFT = 0
STAGE_HIGH_TRANSLATE = 1
STAGE_DESCEND = 2
STAGE_RELEASE = 3
STAGE_RETREAT = 4


class VerticalStagingDiffusionPolicy(torch.nn.Module):
    """Stage-conditioned learned diffusion model for block delivery."""

    def __init__(self, spec):
        super().__init__()
        normalizer = spec["normalizer"]
        self.register_buffer("obs_mean", torch.tensor(normalizer["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.tensor(normalizer["std"], dtype=torch.float32))
        self.register_buffer("action_min", torch.tensor(normalizer["action_min"], dtype=torch.float32))
        self.register_buffer("action_scale", torch.tensor(normalizer["action_scale"], dtype=torch.float32))

        self.feature_dim = 176
        hidden_dim = 256
        stage_dim = 32
        cond_dim = 384

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.feature_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
        )
        self.stage_head = torch.nn.Linear(hidden_dim, 5)
        self.stage_embedding = torch.nn.Embedding(5, stage_dim)
        self.condition_projector = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + stage_dim + 5, cond_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(cond_dim),
        )
        self.backbone = DiffusionBackbone(cond_dim, spec["training"])

        self.future_feature_indices = torch.tensor(
            [18, 19, 20, 25, 26, 27, 32, 33, 34, 7, 8], dtype=torch.long
        )
        self.future_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + 8 + 1, 256),
            torch.nn.GELU(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 11),
        )

    def normalize_obs(self, raw):
        return (raw - self.obs_mean.to(device=raw.device, dtype=raw.dtype)) / self.obs_std.to(device=raw.device, dtype=raw.dtype)

    def denormalize_action(self, encoded):
        amin = self.action_min.to(device=encoded.device, dtype=encoded.dtype)
        scale = self.action_scale.to(device=encoded.device, dtype=encoded.dtype)
        return (encoded + 1.0) * scale / 2.0 + amin

    def xyz_scale(self, raw):
        return raw.new_tensor([0.5, 0.5, 0.3])

    def make_features(self, raw_history):
        b = raw_history.shape[0]
        norm = self.normalize_obs(raw_history)
        pieces = [norm.reshape(b, -1)]
        xyz_scale = self.xyz_scale(raw_history)

        for k in range(raw_history.shape[1]):
            s = raw_history[:, k]
            tcp = s[:, 18:21]
            red = s[:, 25:28]
            blue = s[:, 32:35]
            red_goal = s[:, 41:44]
            blue_goal = s[:, 44:47]
            grip = s[:, 7:9]
            pieces.extend([
                (red_goal - red) / xyz_scale,
                (blue_goal - blue) / xyz_scale,
                (tcp - red) / xyz_scale,
                (tcp - blue) / xyz_scale,
                (tcp - red_goal) / xyz_scale,
                (tcp - blue_goal) / xyz_scale,
                red / xyz_scale,
                blue / xyz_scale,
                tcp / xyz_scale,
                torch.linalg.vector_norm(red[:, :2] - red_goal[:, :2], dim=-1, keepdim=True) / 0.5,
                torch.linalg.vector_norm(blue[:, :2] - blue_goal[:, :2], dim=-1, keepdim=True) / 0.5,
                torch.linalg.vector_norm(tcp[:, :2] - red[:, :2], dim=-1, keepdim=True) / 0.5,
                torch.linalg.vector_norm(tcp[:, :2] - blue[:, :2], dim=-1, keepdim=True) / 0.5,
                red[:, 2:3] / 0.3,
                blue[:, 2:3] / 0.3,
                tcp[:, 2:3] / 0.3,
                grip.sum(dim=-1, keepdim=True) / 0.08,
                (grip[:, 0:1] - grip[:, 1:2]) / 0.02,
            ])

        first = raw_history[:, 0]
        last = raw_history[:, -1]
        pieces.extend([
            (last[:, 18:21] - first[:, 18:21]) / xyz_scale,
            (last[:, 25:28] - first[:, 25:28]) / xyz_scale,
            (last[:, 32:35] - first[:, 32:35]) / xyz_scale,
            (last[:, 7:9].sum(dim=-1, keepdim=True) - first[:, 7:9].sum(dim=-1, keepdim=True)) / 0.02,
        ])
        return torch.cat(pieces, dim=-1)

    def encode_condition(self, raw_history):
        features = self.make_features(raw_history)
        hidden = self.encoder(features)
        logits = self.stage_head(hidden)
        probs = torch.softmax(logits, dim=-1)
        emb_weight = self.stage_embedding.weight.to(device=raw_history.device, dtype=hidden.dtype)
        soft_stage_embedding = probs @ emb_weight
        cond = self.condition_projector(torch.cat([hidden, soft_stage_embedding, probs], dim=-1))
        return cond, hidden, logits

    def stage_logits(self, raw_history):
        return self.encode_condition(raw_history)[2]

    def forward(self, noisy_action, timestep, raw_history):
        cond, hidden, logits = self.encode_condition(raw_history)
        return self.backbone(noisy_action, timestep, cond)

    def future_targets(self, future_obs):
        norm = self.normalize_obs(future_obs)
        idx = self.future_feature_indices.to(device=future_obs.device)
        return torch.index_select(norm, dim=-1, index=idx)

    def predict_future_features(self, raw_history, clean_encoded_action):
        cond, hidden, logits = self.encode_condition(raw_history)
        b, h, action_dim = clean_encoded_action.shape
        if h <= 1:
            slot = clean_encoded_action.new_zeros((1, h, 1))
        else:
            slot = torch.arange(h, device=clean_encoded_action.device, dtype=clean_encoded_action.dtype).view(1, h, 1) / float(h - 1)
        hidden_seq = hidden[:, None, :].expand(b, h, hidden.shape[-1])
        inp = torch.cat([hidden_seq, clean_encoded_action, slot.expand(b, h, 1)], dim=-1)
        return self.future_head(inp)


def build_model(spec):
    return VerticalStagingDiffusionPolicy(spec)


def stage_labels_from_state(state):
    red = state[..., 25:28]
    blue = state[..., 32:35]
    tcp = state[..., 18:21]
    red_goal = state[..., 41:44]
    blue_goal = state[..., 44:47]
    grip_total = state[..., 7] + state[..., 8]

    red_xy_err = torch.linalg.vector_norm(red[..., :2] - red_goal[..., :2], dim=-1)
    blue_xy_err = torch.linalg.vector_norm(blue[..., :2] - blue_goal[..., :2], dim=-1)
    red_at_goal = (red_xy_err < 0.055) & (red[..., 2] < 0.045)
    blue_at_goal = (blue_xy_err < 0.055) & (blue[..., 2] < 0.045)

    use_red = torch.logical_not(red_at_goal)
    obj_z = torch.where(use_red, red[..., 2], blue[..., 2])
    xy_err = torch.where(use_red, red_xy_err, blue_xy_err)

    stage = torch.full_like(obj_z, STAGE_LIFT, dtype=torch.long)
    stage = torch.where((obj_z > 0.18) & (xy_err > 0.055), torch.full_like(stage, STAGE_HIGH_TRANSLATE), stage)
    stage = torch.where((xy_err <= 0.070) & (obj_z > 0.045), torch.full_like(stage, STAGE_DESCEND), stage)
    stage = torch.where((xy_err <= 0.065) & (obj_z <= 0.055) & (tcp[..., 2] <= 0.080), torch.full_like(stage, STAGE_RELEASE), stage)

    any_placed = red_at_goal | blue_at_goal
    open_fingers = grip_total > 0.065
    retreat = any_placed & open_fingers & (tcp[..., 2] > 0.060)
    stage = torch.where(retreat, torch.full_like(stage, STAGE_RETREAT), stage)
    return stage


def masked_mean(value, mask, feature_count=1):
    return (value * mask).sum() / (mask.sum() * feature_count).clamp_min(1.0)


def compute_loss(model, batch, spec):
    pred_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    mask = batch["mask"]
    diffusion_loss = epsilon_loss(pred_noise, batch["noise"], mask)

    current_state = batch["raw_obs"][:, -1]
    stage_target = stage_labels_from_state(current_state)
    stage_logits = model.stage_logits(batch["raw_obs"])
    log_probs = torch.log_softmax(stage_logits, dim=-1)
    stage_loss = -log_probs.gather(1, stage_target.view(-1, 1)).mean()

    alpha_bar = batch["alpha_bar"].to(device=pred_noise.device, dtype=pred_noise.dtype).view(-1, 1, 1)
    sqrt_ab = torch.sqrt(alpha_bar.clamp_min(1.0e-5))
    sqrt_om = torch.sqrt((1.0 - alpha_bar).clamp_min(0.0))
    clean_est = (batch["noisy_action"] - sqrt_om * pred_noise) / sqrt_ab
    clean_est = clean_est.clamp(-2.0, 2.0)
    native_est = model.denormalize_action(clean_est)

    future_state = batch["future_obs"]
    future_mask = batch["future_mask"]
    future_stage = stage_labels_from_state(future_state)
    open_target = torch.where(
        future_stage >= STAGE_RELEASE,
        torch.ones_like(native_est[..., 7]),
        -torch.ones_like(native_est[..., 7]),
    )
    gripper_loss = masked_mean((native_est[..., 7:8] - open_target.unsqueeze(-1)).square(), future_mask, 1)

    pred_future = model.predict_future_features(batch["raw_obs"], clean_est)
    target_future = model.future_targets(future_state)
    noise_weight = alpha_bar.detach().clamp(0.05, 1.0)
    state_loss = ((pred_future - target_future).square() * future_mask * noise_weight).sum() / (
        (future_mask * noise_weight).sum() * pred_future.shape[-1]
    ).clamp_min(1.0)

    prior_loss = 0.10 * stage_loss + 0.02 * gripper_loss + 0.05 * state_loss
    total_loss = diffusion_loss + prior_loss
    return {"loss": total_loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
