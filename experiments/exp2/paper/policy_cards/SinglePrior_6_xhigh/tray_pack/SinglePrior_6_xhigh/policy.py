import torch
from appl.public import DiffusionBackbone, epsilon_loss


def rotation_matrix(quaternion):
    """Convert wxyz quaternions to sign-invariant rotation matrices."""
    q = quaternion / quaternion.square().sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    w, x, y, z = q.unbind(dim=-1)
    rows = (
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    )
    return torch.stack(rows, dim=-1).reshape(q.shape[:-1] + (3, 3))


def object_positions(raw):
    return torch.stack((raw[..., 25:28], raw[..., 32:35]), dim=-2)


def object_goals(raw):
    return torch.stack((raw[..., 41:44], raw[..., 44:47]), dim=-2)


def transport_relations(raw):
    """World-frame tool-to-object and object-to-goal vectors, for both blocks."""
    position = object_positions(raw)
    tool = raw[..., 18:21].unsqueeze(-2)
    return torch.cat((position - tool, object_goals(raw) - position), dim=-1)


class TransportRelationPolicy(torch.nn.Module):
    """One epsilon-predicting DP with a shared two-object temporal encoder."""
    def __init__(self, spec):
        super().__init__()
        normalizer = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(normalizer['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(normalizer['std'], dtype=torch.float32))
        config = spec.get('candidate_config', {})
        self.prior_weight = float(config.get('prior_weight', 0.05))
        self.horizon = int(spec['training']['horizon'])
        self.position_scale = 0.25
        self.local_scale = 0.04
        self.velocity_scale = 0.5
        self.finger_velocity_scale = 0.2
        self.forecast_scale = 0.10
        self.observation_dt = 0.05

        # Identical feature map and identical weights for red and blue.
        # Two 38-dimensional frames plus eleven measured motion features.
        self.object_encoder = torch.nn.Sequential(
            torch.nn.Linear(87, 192),
            torch.nn.SiLU(),
            torch.nn.Linear(192, 192),
            torch.nn.SiLU(),
            torch.nn.Linear(192, 128),
            torch.nn.LayerNorm(128),
        )
        # Retain ordered block identities and absolute robot configuration.
        self.scene_encoder = torch.nn.Sequential(
            torch.nn.Linear(94 + 2 * 128, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 128),
            torch.nn.LayerNorm(128),
        )
        self.backbone = DiffusionBackbone(94 + 2 * 128 + 128, spec['training'])

        # This head is used only for representation learning, never for control.
        # One shared head predicts both blocks' future relation displacements.
        self.relation_forecaster = torch.nn.Sequential(
            torch.nn.Linear(256, 192),
            torch.nn.SiLU(),
            torch.nn.Linear(192, self.horizon * 6),
        )
        torch.nn.init.normal_(self.relation_forecaster[-1].weight, mean=0.0, std=0.01)
        torch.nn.init.zeros_(self.relation_forecaster[-1].bias)

    def geometry_history(self, raw):
        # Shapes: batch, observed time, block, feature.
        position = object_positions(raw)
        goal = object_goals(raw)
        tool = raw[..., 18:21].unsqueeze(-2)
        relative = position - tool
        residual = goal - position
        other_relative = position.flip(dims=(-2,)) - position
        tool_rotation = rotation_matrix(raw[..., 21:25]).unsqueeze(-3)
        block_quaternion = torch.stack((raw[..., 28:32], raw[..., 35:39]), dim=-2)
        block_rotation = rotation_matrix(block_quaternion)
        inverse_tool = tool_rotation.transpose(-1, -2)
        local_relative = torch.matmul(inverse_tool, relative.unsqueeze(-1)).squeeze(-1)
        relative_rotation = torch.matmul(inverse_tool, block_rotation)
        # First two columns of a rotation matrix retain its orientation.
        relative_rotation_six = relative_rotation[..., :, :2].flatten(start_dim=-2)
        block_rotation_six = block_rotation[..., :, :2].flatten(start_dim=-2)
        fingers = (raw[..., 7:9] / 0.04).unsqueeze(-2).expand(-1, -1, 2, -1)
        features = torch.cat((
            position / self.position_scale,
            goal / self.position_scale,
            relative / self.position_scale,
            residual / self.position_scale,
            other_relative / self.position_scale,
            local_relative / self.position_scale,
            torch.tanh(relative / self.local_scale),
            torch.tanh(residual / self.local_scale),
            relative_rotation_six,
            block_rotation_six,
            fingers,
        ), dim=-1)
        # Differences are causal measurements, not an internal episode clock.
        velocity_denominator = self.observation_dt * self.velocity_scale
        object_motion = (position[:, 1] - position[:, 0]) / velocity_denominator
        tool_motion = (tool[:, 1] - tool[:, 0]).expand(-1, 2, -1) / velocity_denominator
        relative_motion = (relative[:, 1] - relative[:, 0]) / velocity_denominator
        finger_motion = ((raw[:, 1, 7:9] - raw[:, 0, 7:9]) /
                         (self.observation_dt * self.finger_velocity_scale))
        finger_motion = finger_motion.unsqueeze(1).expand(-1, 2, -1)
        return torch.cat((features[:, 0], features[:, 1], object_motion,
                          tool_motion, relative_motion, finger_motion), dim=-1)

    def encode_history(self, raw_history):
        normalized = (raw_history - self.obs_mean) / self.obs_scale
        flat_raw = normalized.flatten(start_dim=1)
        tokens = self.object_encoder(self.geometry_history(raw_history))
        flat_tokens = tokens.flatten(start_dim=1)
        scene = self.scene_encoder(torch.cat((flat_raw, flat_tokens), dim=-1))
        condition = torch.cat((flat_raw, flat_tokens, scene), dim=-1)
        return condition, tokens, scene

    def predict_relations(self, tokens, scene):
        context = torch.cat((tokens, scene.unsqueeze(1).expand(-1, 2, -1)), dim=-1)
        prediction = self.relation_forecaster(context)
        return prediction.reshape(prediction.shape[0], 2, self.horizon, 6).transpose(1, 2)

    def forward(self, noisy_action, timestep, raw_history):
        condition, _, _ = self.encode_history(raw_history)
        return self.backbone(noisy_action, timestep, condition)


def build_model(spec):
    return TransportRelationPolicy(spec)


def compute_loss(model, batch, spec):
    # The same encoder and denoiser as forward; reuse the encoder for the head.
    condition, tokens, scene = model.encode_history(batch['raw_obs'])
    predicted_noise = model.backbone(batch['noisy_action'], batch['timesteps'], condition)
    diffusion_loss = epsilon_loss(predicted_noise, batch['noise'], batch['mask'])

    # Slot j labels the state AFTER action slot j, including the past-action
    # slot zero. There is no shift, phase crop, future input, or target action
    # substitution. Anchoring the forecast at the current relations makes
    # stationary blocks easy to represent without fitting absolute coordinates.
    prediction = model.predict_relations(tokens, scene)
    with torch.no_grad():
        current = transport_relations(batch['raw_obs'][:, -1]).unsqueeze(1)
        future = transport_relations(batch['future_obs'])
        target = (future - current) / model.forecast_scale
    valid = batch['future_mask'].to(dtype=prediction.dtype).unsqueeze(-1)
    error = torch.nn.functional.smooth_l1_loss(prediction, target, reduction='none', beta=0.1)
    prior_loss = (error * valid).sum() / (valid.sum() * 12).clamp_min(1.0)
    loss = diffusion_loss + model.prior_weight * prior_loss
    return {'loss': loss, 'diffusion_loss': diffusion_loss, 'prior_loss': prior_loss}
