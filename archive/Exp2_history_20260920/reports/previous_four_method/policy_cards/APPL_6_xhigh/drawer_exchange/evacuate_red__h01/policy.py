import math
import torch
from torch import nn
from appl.public import DiffusionBackbone, epsilon_loss


def rotation_matrix(q):
    """Unit wxyz quaternion to a sign-invariant world rotation matrix."""
    q = q / q.square().sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)
    ), dim=-1).reshape(q.shape[:-1] + (3, 3))


def projection(size, width):
    # Previous state, current state and their difference have separate weights.
    return nn.Sequential(nn.Linear(3 * size, 192), nn.SiLU(),
                         nn.Linear(192, width), nn.LayerNorm(width))


class RelationBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0,
                                               batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.SiLU(),
                                nn.Linear(2 * width, width))

    def forward(self, tokens):
        h = self.norm1(tokens)
        tokens = tokens + self.attention(h, h, h, need_weights=False)[0]
        return tokens + self.ff(self.norm2(tokens))


class RoleDiffusionPolicy(nn.Module):
    def __init__(self, spec):
        super().__init__()
        config = spec.get('candidate_config', {})
        width = int(config.get('token_width', 128))
        heads = int(config.get('attention_heads', 4))
        layers = int(config.get('attention_layers', 2))
        condition_dim = int(config.get('condition_dimension', 384))
        self.horizon = int(spec['training']['horizon'])
        self.auxiliary_weight = float(config.get('auxiliary_weight', 0.5))
        self.width = width
        normalizer = spec['normalizer']
        mean = torch.tensor(normalizer['mean'], dtype=torch.float32)
        scale = torch.tensor(normalizer['std'], dtype=torch.float32)
        self.register_buffer('obs_mean', mean)
        self.register_buffer('obs_scale', scale)
        # Workspace scales come ONLY from the shared complete-demo normalizer.
        space = torch.stack((scale[18:21], scale[25:28], scale[32:35])).amax(0)
        self.register_buffer('space_scale', space)
        self.register_buffer('local_scale', space.max())
        self.register_buffer('drawer_origin', torch.tensor([0.19, 0.0, 0.035]))
        self.register_buffer('red_half_xy', torch.tensor([0.06, 0.06]))
        self.register_buffer('blue_half_xy', torch.tensor([0.172, 0.182]))
        self.robot_projection = projection(32, width)
        self.drawer_projection = projection(21, width)
        self.object_projection = projection(43, width)
        self.pad_projection = projection(15, width)
        self.role_embedding = nn.Parameter(0.02 * torch.randn(5, width))
        self.blocks = nn.ModuleList([RelationBlock(width, heads) for _ in range(layers)])
        self.focus_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        # Keep every role as well as the adaptive focus. Nothing is cropped out.
        self.condition_projection = nn.Sequential(
            nn.Linear(6 * width, condition_dim), nn.SiLU(),
            nn.Linear(condition_dim, condition_dim), nn.LayerNorm(condition_dim))
        self.backbone = DiffusionBackbone(condition_dim, spec['training'])
        self.slot_embedding = nn.Parameter(0.02 * torch.randn(self.horizon, 32))
        self.displacement_head = nn.Sequential(
            nn.Linear(2 * width + 32, 192), nn.SiLU(),
            nn.Linear(192, 128), nn.SiLU(), nn.Linear(128, 3))

    def object_features(self, p, rot, goal, target_center, half_xy,
                        lower_z, upper_z, tcp, tcp_rot, drawer_center):
        world_relative = p - tcp
        tool_relative = torch.matmul(tcp_rot.transpose(-1, -2),
                                     world_relative.unsqueeze(-1)).squeeze(-1)
        relative_rot = torch.matmul(tcp_rot.transpose(-1, -2), rot)
        extent = 0.02 * rot.abs().sum(dim=-1)
        # These are observed geometric features, not enforced constraints.
        xy_slack = half_xy - (p[..., :2] - target_center[..., :2]).abs() - extent[..., :2]
        return torch.cat((
            p / self.space_scale,
            rot.flatten(-2),
            world_relative / self.space_scale,
            tool_relative / self.local_scale,
            relative_rot.flatten(-2),
            (goal - p) / self.space_scale,
            (p - drawer_center) / self.space_scale,
            goal / self.space_scale,
            extent / self.space_scale,
            xy_slack / self.space_scale[:2],
            (p[..., 2:3] - lower_z) / self.space_scale[2],
            (upper_z - p[..., 2:3]) / self.space_scale[2]
        ), dim=-1)

    @staticmethod
    def history_features(x):
        before, current = x[:, -2], x[:, -1]
        return torch.cat((before, current, current - before), dim=-1)

    def encode(self, raw_history):
        raw = raw_history
        normalized = (raw - self.obs_mean) / self.obs_scale
        tcp, red, blue = raw[..., 18:21], raw[..., 25:28], raw[..., 32:35]
        tcp_rot = rotation_matrix(raw[..., 21:25])
        red_rot = rotation_matrix(raw[..., 28:32])
        blue_rot = rotation_matrix(raw[..., 35:39])
        pad, blue_goal = raw[..., 41:44], raw[..., 44:47]
        d = raw[..., 39:40]
        zero = torch.zeros_like(d)
        drawer_center = self.drawer_origin + torch.cat((-d, zero, zero), dim=-1)
        robot_features = torch.cat((normalized[..., :18], tcp / self.space_scale,
                                    tcp_rot.flatten(-2), normalized[..., 39:41]), dim=-1)
        drawer_features = torch.cat((
            normalized[..., 39:41], drawer_center / self.space_scale,
            (drawer_center - tcp) / self.space_scale, blue_goal / self.space_scale,
            (blue_goal - tcp) / self.space_scale, (red - drawer_center) / self.space_scale,
            (blue - drawer_center) / self.space_scale,
            (d - 0.26) / self.obs_scale[39:40]), dim=-1)
        red_features = self.object_features(red, red_rot, pad, pad, self.red_half_xy,
                                            0.014, 0.031, tcp, tcp_rot, drawer_center)
        blue_features = self.object_features(blue, blue_rot, blue_goal, drawer_center,
                                             self.blue_half_xy, 0.053, 0.074,
                                             tcp, tcp_rot, drawer_center)
        pad_features = torch.cat((pad / self.space_scale,
                                  (red - pad) / self.space_scale,
                                  (tcp - pad) / self.space_scale,
                                  (blue - pad) / self.space_scale,
                                  (drawer_center - pad) / self.space_scale), dim=-1)
        tokens = torch.stack((
            self.robot_projection(self.history_features(robot_features)),
            self.drawer_projection(self.history_features(drawer_features)),
            self.object_projection(self.history_features(red_features)),
            self.object_projection(self.history_features(blue_features)),
            self.pad_projection(self.history_features(pad_features))
        ), dim=1) + self.role_embedding.unsqueeze(0)
        for block in self.blocks:
            tokens = block(tokens)
        h = self.focus_norm(tokens)
        # A contextual robot query uses finger configuration, motion and both objects.
        # There is no elapsed-time phase input or deterministic grasp switch.
        query = self.query(h[:, 0])
        scores = (self.key(h) * query.unsqueeze(1)).sum(-1) / math.sqrt(self.width)
        weights = scores.softmax(dim=-1)
        focus = (weights.unsqueeze(-1) * self.value(h)).sum(dim=1)
        condition = self.condition_projection(torch.cat((tokens.flatten(1), focus), dim=-1))
        return condition, tokens, focus

    def forward(self, noisy_action, timestep, raw_history):
        condition, _, _ = self.encode(raw_history)
        return self.backbone(noisy_action, timestep, condition)

    def predict_displacements(self, tokens, focus):
        # Training-only prediction from causal representations, not from future inputs.
        batch_size = tokens.shape[0]
        objects = tokens[:, None, 2:4, :].expand(-1, self.horizon, -1, -1)
        attended = focus[:, None, None, :].expand(-1, self.horizon, 2, -1)
        slots = self.slot_embedding[None, :, None, :].expand(batch_size, -1, 2, -1)
        return self.displacement_head(torch.cat((objects, attended, slots), dim=-1))


def build_model(spec):
    return RoleDiffusionPolicy(spec)


def compute_loss(model, batch, spec):
    # Exactly the same causal encoding and epsilon path as deployment forward.
    condition, tokens, focus = model.encode(batch['raw_obs'])
    predicted_noise = model.backbone(batch['noisy_action'], batch['timesteps'], condition)
    diffusion_loss = epsilon_loss(predicted_noise, batch['noise'], batch['mask'])
    prediction = model.predict_displacements(tokens, focus)
    current = torch.stack((batch['raw_obs'][:, -1, 25:28],
                           batch['raw_obs'][:, -1, 32:35]), dim=1)
    future = torch.stack((batch['future_obs'][..., 25:28],
                          batch['future_obs'][..., 32:35]), dim=2)
    target = (future - current[:, None]) / model.space_scale
    valid = (batch['future_mask'] * batch['mask']).unsqueeze(-1)
    displacement_loss = ((prediction - target).square() * valid).sum() / (valid.sum() * 6).clamp_min(1)
    prior_loss = model.auxiliary_weight * displacement_loss
    return {'loss': diffusion_loss + prior_loss,
            'diffusion_loss': diffusion_loss, 'prior_loss': prior_loss}
