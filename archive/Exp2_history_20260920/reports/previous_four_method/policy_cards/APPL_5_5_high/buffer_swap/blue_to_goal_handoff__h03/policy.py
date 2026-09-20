import math
import torch
import appl.public as public


MODE_ENTRY_APPROACH_BLUE = 0
MODE_BLUE_GRASP_LIFT = 1
MODE_BLUE_CARRY = 2
MODE_BLUE_PLACE_RELEASE = 3
MODE_RETREAT = 4
MODE_RED_HANDOFF = 5
MODE_COUNT = 6


def xy_distance(a, b):
    d = a[..., :2] - b[..., :2]
    return torch.sqrt((d * d).sum(dim=-1) + 1.0e-8)


def weak_mode_labels(raw_obs):
    """Vectorized weak contact-mode labels from observed poses.

    The labels are used only for auxiliary training. They are successful-demo
    event proxies, not hard runtime guards and not contact measurements.
    """
    tcp = raw_obs[..., 18:21]
    red = raw_obs[..., 25:28]
    blue = raw_obs[..., 32:35]
    blue_goal = raw_obs[..., 44:47]
    finger_width = raw_obs[..., 7] + raw_obs[..., 8]

    tcp_blue_xy = xy_distance(tcp, blue)
    tcp_red_xy = xy_distance(tcp, red)
    blue_goal_xy = xy_distance(blue, blue_goal)

    closed = finger_width < 0.055
    open_gripper = finger_width > 0.065
    near_blue = tcp_blue_xy < 0.040
    near_red = tcp_red_xy < 0.045
    blue_lifted = blue[..., 2] > 0.055
    red_lifted = red[..., 2] > 0.055
    blue_xy_at_goal = blue_goal_xy < 0.065
    blue_placed = blue_xy_at_goal & ((blue[..., 2] - blue_goal[..., 2]).abs() < 0.035)

    label = torch.zeros(raw_obs.shape[:-1], dtype=torch.long, device=raw_obs.device)
    label = torch.where(closed & near_blue, torch.full_like(label, MODE_BLUE_GRASP_LIFT), label)
    label = torch.where(blue_lifted & (~blue_xy_at_goal), torch.full_like(label, MODE_BLUE_CARRY), label)
    place_region = blue_xy_at_goal & (blue_lifted | near_blue | blue_placed)
    label = torch.where(place_region, torch.full_like(label, MODE_BLUE_PLACE_RELEASE), label)
    retreat_region = blue_placed & (open_gripper | (tcp[..., 2] > 0.10))
    label = torch.where(retreat_region, torch.full_like(label, MODE_RETREAT), label)
    red_handoff_region = blue_placed & (red_lifted | (near_red & (tcp[..., 2] < 0.22)))
    label = torch.where(red_handoff_region, torch.full_like(label, MODE_RED_HANDOFF), label)
    return label


class ExpertResidual(torch.nn.Module):
    def __init__(self, condition_dim, channels=64, time_dim=32):
        super().__init__()
        self.time_dim = time_dim
        self.conv_in = torch.nn.Conv1d(8, channels, 3, padding=1)
        self.film = torch.nn.Linear(condition_dim + time_dim, channels * 2)
        self.conv_mid = torch.nn.Conv1d(channels, channels, 3, padding=1)
        self.conv_out = torch.nn.Conv1d(channels, 8, 1)
        self.act = torch.nn.Mish()
        torch.nn.init.zeros_(self.conv_out.weight)
        torch.nn.init.zeros_(self.conv_out.bias)

    def time_embedding(self, timestep, batch, device, dtype):
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=device, dtype=torch.long)
        else:
            timestep = timestep.to(device)
        if timestep.dim() == 0:
            timestep = timestep[None]
        timestep = timestep.expand(batch).to(dtype=dtype)
        half = self.time_dim // 2
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=device, dtype=dtype)
            / max(half - 1, 1)
        )
        args = timestep[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.time_dim:
            emb = torch.cat([emb, torch.zeros(batch, 1, device=device, dtype=dtype)], dim=-1)
        return emb

    def forward(self, noisy_action, timestep, condition):
        b = noisy_action.shape[0]
        x = noisy_action.transpose(1, 2)
        h = self.conv_in(x)
        temb = self.time_embedding(timestep, b, noisy_action.device, noisy_action.dtype)
        film = self.film(torch.cat([condition, temb], dim=-1))
        scale, bias = film.chunk(2, dim=-1)
        h = h * (1.0 + torch.tanh(scale)[:, :, None]) + bias[:, :, None]
        h = self.act(h)
        h = self.act(self.conv_mid(h))
        return self.conv_out(h).transpose(1, 2)


class ContactModeDiffusionPolicy(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = spec["normalizer"]
        self.register_buffer("obs_mean", torch.tensor(n["mean"], dtype=torch.float32))
        self.register_buffer("obs_std", torch.tensor(n["std"], dtype=torch.float32))
        self.condition_dim = 256
        self.mode_count = MODE_COUNT
        self.engineered_dim = 24
        self.encoder_in_dim = 47 + self.engineered_dim
        self.gru = torch.nn.GRU(self.encoder_in_dim, 128, batch_first=True)
        self.mode_head = torch.nn.Sequential(
            torch.nn.Linear(128 + self.encoder_in_dim, 128),
            torch.nn.Mish(),
            torch.nn.Linear(128, MODE_COUNT),
        )
        self.mode_embedding = torch.nn.Parameter(torch.randn(MODE_COUNT, 32) * 0.02)
        self.condition_head = torch.nn.Sequential(
            torch.nn.Linear(128 + self.encoder_in_dim + 32 + MODE_COUNT, 256),
            torch.nn.Mish(),
            torch.nn.Linear(256, self.condition_dim),
            torch.nn.LayerNorm(self.condition_dim),
        )
        self.backbone = public.DiffusionBackbone(self.condition_dim, spec["training"])
        self.experts = torch.nn.ModuleList(
            [ExpertResidual(self.condition_dim, channels=64, time_dim=32) for _ in range(MODE_COUNT)]
        )
        self.residual_scale = 0.25

    def normalize_obs(self, raw_history):
        return (raw_history - self.obs_mean.to(raw_history.dtype)) / self.obs_std.to(raw_history.dtype)

    def engineered_features(self, raw_history):
        tcp = raw_history[..., 18:21]
        red = raw_history[..., 25:28]
        blue = raw_history[..., 32:35]
        red_goal = raw_history[..., 41:44]
        blue_goal = raw_history[..., 44:47]
        finger_l = raw_history[..., 7:8]
        finger_r = raw_history[..., 8:9]
        finger_width = finger_l + finger_r
        finger_asym = finger_l - finger_r

        xyz_scale = raw_history.new_tensor([0.40, 0.40, 0.30])
        z_scale = raw_history.new_tensor(0.30)
        xy_scale = raw_history.new_tensor(0.40)

        tcp_blue = (tcp - blue) / xyz_scale
        tcp_red = (tcp - red) / xyz_scale
        blue_to_goal = (blue - blue_goal) / xyz_scale
        red_to_goal = (red - red_goal) / xyz_scale
        red_to_blue = (red - blue) / xyz_scale
        heights = torch.stack(
            [tcp[..., 2] - 0.02, red[..., 2] - 0.02, blue[..., 2] - 0.02], dim=-1
        ) / z_scale
        fingers = torch.cat([(finger_width - 0.055) / 0.04, finger_asym / 0.02], dim=-1)
        dists = torch.stack(
            [
                xy_distance(tcp, blue) / xy_scale,
                xy_distance(tcp, red) / xy_scale,
                xy_distance(blue, blue_goal) / xy_scale,
                xy_distance(red, red_goal) / xy_scale,
            ],
            dim=-1,
        )
        return torch.cat(
            [tcp_blue, tcp_red, blue_to_goal, red_to_goal, red_to_blue, heights, fingers, dists], dim=-1
        )

    def encode_condition(self, raw_history):
        norm = self.normalize_obs(raw_history)
        eng = self.engineered_features(raw_history)
        enc_in = torch.cat([norm, eng], dim=-1)
        seq_out, h_last = self.gru(enc_in)
        h = h_last[-1]
        last = enc_in[:, -1, :]
        summary = torch.cat([h, last], dim=-1)
        mode_logits = self.mode_head(summary)
        mode_probs = torch.softmax(mode_logits, dim=-1)
        mode_context = mode_probs @ self.mode_embedding
        condition = self.condition_head(torch.cat([summary, mode_context, mode_probs], dim=-1))
        return condition, mode_logits, mode_probs

    def forward_with_info(self, noisy_action, timestep, raw_history):
        condition, mode_logits, mode_probs = self.encode_condition(raw_history)
        base = self.backbone(noisy_action, timestep, condition)
        residual = 0.0
        for m, expert in enumerate(self.experts):
            residual = residual + mode_probs[:, m].view(-1, 1, 1) * expert(noisy_action, timestep, condition)
        pred = base + self.residual_scale * residual
        return pred, {"mode_logits": mode_logits, "mode_probs": mode_probs, "condition": condition}

    def forward(self, noisy_action, timestep, raw_history):
        pred, aux = self.forward_with_info(noisy_action, timestep, raw_history)
        return pred


def build_model(spec):
    return ContactModeDiffusionPolicy(spec)


def masked_cross_entropy(logits, labels, mask):
    losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    mask = mask.to(losses.dtype)
    return (losses * mask).sum() / mask.sum().clamp_min(1.0)


def compute_loss(model, batch, spec):
    pred, info = model.forward_with_info(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = public.epsilon_loss(pred, batch["noise"], batch["mask"])

    current_labels = weak_mode_labels(batch["raw_obs"][:, -1, :])
    current_ce = torch.nn.functional.cross_entropy(info["mode_logits"], current_labels)

    future_obs = batch["future_obs"]
    b, h, d = future_obs.shape
    current_obs = batch["raw_obs"][:, -1:, :]
    all_obs = torch.cat([current_obs, future_obs], dim=1)
    future_hist = torch.stack([all_obs[:, :-1, :], all_obs[:, 1:, :]], dim=2).reshape(b * h, 2, d)
    cond_unused, future_logits_flat, future_probs_flat = model.encode_condition(future_hist)
    future_labels = weak_mode_labels(future_obs).reshape(b * h)
    future_mask = batch["future_mask"].reshape(b * h)
    future_ce = masked_cross_entropy(future_logits_flat, future_labels, future_mask)

    future_probs = future_probs_flat.reshape(b, h, MODE_COUNT)
    prob_seq = torch.cat([info["mode_probs"].unsqueeze(1), future_probs], dim=1)
    pair_mask = batch["future_mask"].squeeze(-1)
    smooth = ((prob_seq[:, 1:, :] - prob_seq[:, :-1, :]).square().sum(dim=-1) * pair_mask).sum()
    smooth = smooth / pair_mask.sum().clamp_min(1.0)

    prior_loss = 0.10 * current_ce + 0.05 * future_ce + 0.005 * smooth
    loss = diffusion_loss + prior_loss
    return {"loss": loss, "diffusion_loss": diffusion_loss, "prior_loss": prior_loss}
