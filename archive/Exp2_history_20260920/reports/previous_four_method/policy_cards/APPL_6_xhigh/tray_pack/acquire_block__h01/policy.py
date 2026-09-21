"""Role-factored relative-geometry action diffusion for the full acquisition slice.

Only causal raw_history enters the denoiser. Object pose/goal bundles share all
encoder weights and geometric scales. Latent role queries are learned through
noise prediction; they are not a scripted phase controller or external intent.
"""
import torch
from torch import nn
from appl.public import DiffusionBackbone, epsilon_loss


def quaternion_matrix(q):
    """Unit-normalized wxyz quaternion to local-to-world rotation matrix."""
    q = q / q.square().sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    w, x, y, z = q.unbind(dim=-1)
    entries = (
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - w * z),
        2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    )
    return torch.stack(entries, dim=-1).reshape(q.shape[:-1] + (3, 3))


def history_features(x):
    """Ordered two-state history and a causal per-observation difference."""
    return torch.cat((x[:, 0], x[:, 1], x[:, 1] - x[:, 0]), dim=-1)


class SharedRoleGeometry(nn.Module):
    def __init__(self, spec):
        super().__init__()
        mean = torch.tensor(spec['normalizer']['mean'], dtype=torch.float32)
        scale = torch.tensor(spec['normalizer']['std'], dtype=torch.float32)
        self.register_buffer('observation_mean', mean)
        self.register_buffer('observation_scale', scale)
        # One isotropic length unit, derived only from the frozen full-demo fit.
        # It is common to both objects and to world/TCP-frame displacements.
        lengths = torch.cat((scale[18:21], scale[25:28], scale[32:35]))
        self.register_buffer('geometry_length', lengths.max().reshape(1))
        self.register_buffer('world_center', mean[18:21].clone())

        # Per state: 36 geometry features. Two states plus their difference.
        self.slot_encoder = nn.Sequential(
            nn.Linear(108, 192), nn.SiLU(),
            nn.Linear(192, 128), nn.LayerNorm(128), nn.SiLU(),
        )
        # qpos/qvel (18), normalized TCP xyz (3), TCP rotation (9), height (1).
        self.robot_encoder = nn.Sequential(
            nn.Linear(93, 192), nn.SiLU(),
            nn.Linear(192, 128), nn.LayerNorm(128), nn.SiLU(),
        )
        # No object-position embeddings: pose AND matching goal travel together.
        self.object_interaction = nn.MultiheadAttention(128, 4, dropout=0.0, batch_first=True)
        self.object_norm = nn.LayerNorm(128)
        self.object_ff = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 128))
        self.object_ff_norm = nn.LayerNorm(128)

        # Three unlabelled role queries can simultaneously represent an incoming
        # payload, a prospective source and scene context. These names are design
        # motivation, not guaranteed or supervised semantic assignments.
        self.role_embeddings = nn.Parameter(torch.randn(3, 128) * 0.02)
        self.scene_queries = nn.Sequential(
            nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 384)
        )
        self.role_attention = nn.MultiheadAttention(128, 4, dropout=0.0, batch_first=True)
        self.role_norm = nn.LayerNorm(128)
        self.fusion = nn.Sequential(
            nn.Linear(640, 384), nn.SiLU(),
            nn.Linear(384, 256), nn.LayerNorm(256), nn.SiLU(),
        )
        self.denoiser = DiffusionBackbone(384, spec['training'])

    def geometric_features(self, raw_history):
        normalized = (raw_history - self.observation_mean) / self.observation_scale
        tcp = raw_history[..., 18:21]
        tcp_rotation = quaternion_matrix(raw_history[..., 21:25])
        world_to_tcp = tcp_rotation.transpose(-1, -2).unsqueeze(2)
        poses = torch.stack((raw_history[..., 25:32], raw_history[..., 32:39]), dim=2)
        goals = torch.stack((raw_history[..., 41:44], raw_history[..., 44:47]), dim=2)
        positions = poses[..., :3]
        object_rotations = quaternion_matrix(poses[..., 3:7])
        offset = positions - tcp.unsqueeze(2)
        local_offset = torch.matmul(world_to_tcp, offset.unsqueeze(-1)).squeeze(-1)
        relative_rotation = torch.matmul(world_to_tcp, object_rotations).flatten(-2)
        goal_error = goals - positions
        local_goal_error = torch.matmul(world_to_tcp, goal_error.unsqueeze(-1)).squeeze(-1)
        heights = torch.stack((
            positions[..., 2],
            tcp[..., 2].unsqueeze(-1).expand_as(positions[..., 2]),
            goals[..., 2],
        ), dim=-1)
        other_offset = positions.flip(2) - positions
        other_goal_offset = goals.flip(2) - goals
        length = self.geometry_length
        slots = torch.cat((
            offset / length,                         # 3
            local_offset / length,                   # 3
            relative_rotation,                       # 9
            goal_error / length,                     # 3
            local_goal_error / length,               # 3
            (positions - self.world_center) / length,# 3
            (goals - self.world_center) / length,    # 3
            heights / length,                        # 3
            other_offset / length,                   # 3
            other_goal_offset / length,              # 3
        ), dim=-1)
        robot = torch.cat((
            normalized[..., :18],
            normalized[..., 18:21],
            tcp_rotation.flatten(-2),
            tcp[..., 2:3] / length,
        ), dim=-1)
        return history_features(slots), history_features(robot)

    def encode(self, raw_history):
        slot_features, robot_features = self.geometric_features(raw_history)
        objects = self.slot_encoder(slot_features)
        robot = self.robot_encoder(robot_features)
        message, _ = self.object_interaction(objects, objects, objects, need_weights=False)
        objects = self.object_norm(objects + message)
        objects = self.object_ff_norm(objects + self.object_ff(objects))
        scene = objects.mean(dim=1)
        query = self.scene_queries(torch.cat((robot, scene), dim=-1))
        query = query.reshape(raw_history.shape[0], 3, 128) + self.role_embeddings.unsqueeze(0)
        roles, _ = self.role_attention(query, objects, objects, need_weights=False)
        roles = self.role_norm(roles + query)
        fused = self.fusion(torch.cat((robot, scene, roles.flatten(1)), dim=-1))
        # The fixed-base stream is retained explicitly as well as in fusion.
        return torch.cat((robot, fused), dim=-1)

    def forward(self, noisy_action, timestep, raw_history):
        return self.denoiser(noisy_action, timestep, self.encode(raw_history))


def build_model(spec):
    return SharedRoleGeometry(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['raw_obs'])
    diffusion_loss = epsilon_loss(prediction, batch['noise'], batch['mask'])
    # The prior is architectural parameter tying and relative conditioning.
    # No constant pose-only penalty is presented as a trainable geometric loss.
    prior_loss = diffusion_loss * 0.0
    return {
        'loss': diffusion_loss + prior_loss,
        'diffusion_loss': diffusion_loss,
        'prior_loss': prior_loss,
    }
