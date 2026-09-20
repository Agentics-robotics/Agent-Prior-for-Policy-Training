import torch

from appl.public import DiffusionBackbone, normalize_observation, epsilon_loss


class EventGatedGraspLiftPolicy(torch.nn.Module):
    """Learned event-gated diffusion policy for red unstack lift.

    The model keeps the action generator a DDPM epsilon-prediction network, while
    the causal state history is encoded through an ordered three-mode latent
    representation: approach_open, close_contact, lift_carry.  The mode is not a
    scripted controller output; it is a trainable classifier/embedding used as
    global conditioning for the diffusion denoiser and trained with auxiliary
    labels inferred from the demonstrations.
    """

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        cand = spec.get("candidate_config", {})
        if isinstance(cand, dict) and "config" in cand:
            cand = cand["config"]
        if not isinstance(cand, dict):
            cand = {}

        self.history_steps = int(spec.get("training", {}).get("observation_steps", 2))
        self.horizon = int(spec.get("training", {}).get("horizon", 16))
        self.obs_dim = int(spec.get("observation_dimension", 47))
        self.mode_count = 3
        self.engineered_dim = 35
        hidden_dim = int(cand.get("hidden_dim", 128))
        cond_dim = int(cand.get("condition_dim", 256))
        mode_embed_dim = int(cand.get("mode_embedding_dim", 24))

        self.obs_pre = torch.nn.Sequential(
            torch.nn.Linear(self.obs_dim + self.engineered_dim, hidden_dim),
            torch.nn.Mish(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Mish(),
        )
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.history_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Mish(),
        )

        self.phase_head = torch.nn.Linear(hidden_dim, self.mode_count)
        self.transition_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + self.mode_count, hidden_dim),
            torch.nn.Mish(),
            torch.nn.Linear(hidden_dim, self.mode_count),
        )
        self.mode_embedding = torch.nn.Embedding(self.mode_count, mode_embed_dim)

        cond_input = hidden_dim + self.mode_count + mode_embed_dim + self.mode_count + 1
        self.condition_head = torch.nn.Sequential(
            torch.nn.Linear(cond_input, cond_dim),
            torch.nn.Mish(),
            torch.nn.LayerNorm(cond_dim),
            torch.nn.Linear(cond_dim, cond_dim),
            torch.nn.Mish(),
        )
        self.prototype_head = torch.nn.Sequential(
            torch.nn.Linear(cond_input, hidden_dim),
            torch.nn.Mish(),
            torch.nn.Linear(hidden_dim, self.horizon * 8),
        )

        self.backbone = DiffusionBackbone(cond_dim, spec["training"])

    def engineered_features(self, raw_history):
        qpos = raw_history[..., 0:9]
        qvel = raw_history[..., 9:18]
        tcp = raw_history[..., 18:25]
        red = raw_history[..., 25:32]
        blue = raw_history[..., 32:39]
        red_goal = raw_history[..., 41:44]
        blue_goal = raw_history[..., 44:47]

        tcp_xyz = tcp[..., 0:3]
        red_xyz = red[..., 0:3]
        blue_xyz = blue[..., 0:3]
        finger_left = qpos[..., 7:8]
        finger_right = qpos[..., 8:9]
        finger_width = finger_left + finger_right
        finger_asym = finger_left - finger_right
        finger_vel_sum = qvel[..., 7:8] + qvel[..., 8:9]
        finger_vel_asym = qvel[..., 7:8] - qvel[..., 8:9]
        tcp_red = tcp_xyz - red_xyz
        red_blue = red_xyz - blue_xyz
        red_to_goal = red_goal - red_xyz
        blue_to_goal = blue_goal - blue_xyz

        d_red = torch.zeros_like(red_xyz)
        d_tcp = torch.zeros_like(tcp_xyz)
        d_blue = torch.zeros_like(blue_xyz)
        d_width = torch.zeros_like(finger_width)
        if raw_history.shape[1] > 1:
            d_red[:, 1:] = red_xyz[:, 1:] - red_xyz[:, :-1]
            d_tcp[:, 1:] = tcp_xyz[:, 1:] - tcp_xyz[:, :-1]
            d_blue[:, 1:] = blue_xyz[:, 1:] - blue_xyz[:, :-1]
            d_width[:, 1:] = finger_width[:, 1:] - finger_width[:, :-1]
            d_red[:, 0:1] = d_red[:, 1:2]
            d_tcp[:, 0:1] = d_tcp[:, 1:2]
            d_blue[:, 0:1] = d_blue[:, 1:2]
            d_width[:, 0:1] = d_width[:, 1:2]

        return torch.cat([
            finger_left, finger_right, finger_width, finger_asym,
            finger_vel_sum, finger_vel_asym,
            tcp_red, red_blue, red_to_goal, blue_to_goal,
            d_red, d_tcp, d_blue, d_width,
            qvel[..., 0:7],
        ], dim=-1)

    def encode_context(self, raw_history):
        norm = normalize_observation(raw_history, self.spec)
        eng = self.engineered_features(raw_history)
        x = torch.cat([norm, eng], dim=-1)
        x = self.obs_pre(x)
        unused_out, h = self.gru(x)
        h = self.history_head(h[-1])

        phase_logits = self.phase_head(h)
        phase_prob = torch.softmax(phase_logits, dim=-1)
        embeds = self.mode_embedding.weight.unsqueeze(0).expand(h.shape[0], -1, -1)
        soft_embed = (phase_prob.unsqueeze(-1) * embeds).sum(dim=1)
        order_values = torch.linspace(0.0, 1.0, self.mode_count, device=h.device, dtype=h.dtype)
        order_scalar = (phase_prob * order_values.unsqueeze(0)).sum(dim=-1, keepdim=True)
        transition_logits = self.transition_head(torch.cat([h, phase_prob], dim=-1))
        transition_prob = torch.softmax(transition_logits, dim=-1)

        cond_basis = torch.cat([h, phase_prob, soft_embed, transition_prob, order_scalar], dim=-1)
        condition = self.condition_head(cond_basis)
        proto = self.prototype_head(cond_basis).view(h.shape[0], self.horizon, 8)
        return {
            "condition": condition,
            "phase_logits": phase_logits,
            "phase_prob": phase_prob,
            "transition_logits": transition_logits,
            "transition_prob": transition_prob,
            "prototype": proto,
        }

    def forward(self, noisy_action, timestep, raw_history):
        enc = self.encode_context(raw_history)
        return self.backbone(noisy_action, timestep, enc["condition"])

    def auxiliary_predictions(self, raw_history):
        return self.encode_context(raw_history)


def build_model(spec):
    return EventGatedGraspLiftPolicy(spec)


def mode_labels_from_state(state, future_obs=None, future_mask=None):
    finger_width = state[:, 7] + state[:, 8]
    red_z = state[:, 27]
    blue_z = state[:, 34]
    lift_height = red_z - blue_z

    lifted_now = (red_z > 0.095) | (lift_height > 0.070)
    if future_obs is not None:
        red_future = future_obs[:, :, 27]
        if future_mask is not None:
            valid = future_mask[:, :, 0] > 0.5
            very_low = torch.full_like(red_future, -1.0e6)
            red_future = torch.where(valid, red_future, very_low)
        future_max = red_future.max(dim=1).values
        lifted_soon = (future_max > 0.135) & (finger_width < 0.052)
        lifted_now = lifted_now | lifted_soon

    closed_or_closing = finger_width < 0.055
    labels = torch.zeros(state.shape[0], device=state.device, dtype=torch.long)
    labels = torch.where(closed_or_closing, torch.ones_like(labels), labels)
    labels = torch.where(lifted_now, torch.full_like(labels, 2), labels)
    return labels


def future_mode_labels(future_obs, future_mask):
    h = future_obs.shape[1]
    idx = 7 if h > 7 else h - 1
    state = future_obs[:, idx, :]
    return mode_labels_from_state(state, None, None)


def compute_loss(model, batch, spec):
    pred_noise = model(batch["noisy_action"], batch["timesteps"], batch["raw_obs"])
    diffusion_loss = epsilon_loss(pred_noise, batch["noise"], batch["mask"])

    aux = model.auxiliary_predictions(batch["raw_obs"])
    current_state = batch["raw_obs"][:, -1, :]
    phase_labels = mode_labels_from_state(
        current_state, batch.get("future_obs", None), batch.get("future_mask", None)
    )
    next_labels = future_mode_labels(batch["future_obs"], batch["future_mask"])

    phase_loss = torch.nn.functional.cross_entropy(aux["phase_logits"], phase_labels)
    transition_loss = torch.nn.functional.cross_entropy(aux["transition_logits"], next_labels)

    proto = aux["prototype"]
    target = batch["encoded_action"]
    mask = batch["mask"]
    if proto.shape[1] != target.shape[1]:
        if proto.shape[1] > target.shape[1]:
            proto = proto[:, :target.shape[1], :]
        else:
            repeat = target.shape[1] - proto.shape[1]
            proto = torch.cat([proto, proto[:, -1:, :].expand(-1, repeat, -1)], dim=1)
    prototype_loss = ((proto - target).square() * mask).sum() / (mask.sum() * target.shape[-1]).clamp_min(1.0)

    phase_prob = aux["phase_prob"]
    trans_prob = aux["transition_prob"]
    backward = torch.zeros((), device=phase_prob.device, dtype=phase_prob.dtype)
    for i in range(1, 3):
        backward = backward + phase_prob[:, i:i+1] * trans_prob[:, :i].sum(dim=-1, keepdim=True)
    order_loss = backward.mean()

    cand = spec.get("candidate_config", {})
    if isinstance(cand, dict) and "config" in cand:
        cand = cand["config"]
    if not isinstance(cand, dict):
        cand = {}
    w_phase = float(cand.get("phase_loss_weight", 0.060))
    w_transition = float(cand.get("transition_loss_weight", 0.040))
    w_proto = float(cand.get("prototype_loss_weight", 0.020))
    w_order = float(cand.get("order_loss_weight", 0.010))

    prior_loss = w_phase * phase_loss + w_transition * transition_loss + w_proto * prototype_loss + w_order * order_loss
    loss = diffusion_loss + prior_loss
    return {
        "loss": loss,
        "diffusion_loss": diffusion_loss,
        "prior_loss": prior_loss,
        "phase_loss": phase_loss.detach(),
        "transition_loss": transition_loss.detach(),
        "prototype_loss": prototype_loss.detach(),
        "order_loss": order_loss.detach(),
    }
