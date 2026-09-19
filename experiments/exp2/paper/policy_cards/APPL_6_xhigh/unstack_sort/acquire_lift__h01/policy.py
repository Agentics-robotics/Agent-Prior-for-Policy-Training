import torch
from torch import nn
from appl.public import DiffusionBackbone, epsilon_loss


def rotation_matrix(quaternion):
    """World rotation from wxyz; identical for q and -q."""
    q = quaternion / quaternion.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
    w, x, y, z = q.unbind(-1)
    entries = (
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
    )
    return torch.stack(entries, dim=-1).reshape(q.shape[:-1] + (3, 3))


def mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(),
                         nn.Linear(hidden_dim, output_dim))


def causal_displacement(x):
    return torch.cat((torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]), dim=1)


class RelationRound(nn.Module):
    """One directed other-object message per node, shared across identities."""
    def __init__(self, width):
        super().__init__()
        self.message = mlp(2 * width + 17, 128, width)
        self.update = mlp(3 * width, 128, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, nodes, edges, robot):
        other = nodes.flip(2)
        message = self.message(torch.cat((nodes, other, edges), dim=-1))
        global_nodes = robot.unsqueeze(2).expand_as(nodes)
        update = self.update(torch.cat((nodes, message, global_nodes), dim=-1))
        return self.norm(nodes + update)


class RoleRelativeDiffusion(nn.Module):
    def __init__(self, spec):
        super().__init__()
        n = spec['normalizer']
        mean = torch.tensor(n['mean'], dtype=torch.float32)
        scale = torch.tensor(n['std'], dtype=torch.float32)
        self.register_buffer('obs_mean', mean)
        self.register_buffer('obs_scale', scale)
        # Derived only from the complete-demonstration common normalizer.
        spatial_scale = torch.stack((scale[18:21], scale[25:28], scale[32:35])).amax(0)
        self.register_buffer('spatial_scale', spatial_scale)
        self.register_buffer('object_center', (mean[25:28] + mean[32:35]) / 2)
        self.register_buffer('object_identity', torch.eye(2, dtype=torch.float32))
        self.robot_encoder = mlp(33, 128, 96)
        self.node_encoder = mlp(40, 128, 96)
        self.rounds = nn.ModuleList([RelationRound(96), RelationRound(96)])
        # These are latent attention slots, NOT supervised selected/held labels.
        # Null attention lets a slot represent absence without deleting either node.
        self.slot_logits = mlp(192, 96, 3)
        self.null_logits = nn.Linear(96, 3)
        self.null_tokens = nn.Parameter(torch.zeros(3, 96))
        self.frame_encoder = mlp(480, 256, 192)
        self.temporal_encoder = mlp(576, 384, 256)
        self.denoiser = DiffusionBackbone(256, spec['training'])

    def encode(self, raw_history):
        raw = raw_history
        normalized = (raw - self.obs_mean) / self.obs_scale
        scale = self.spatial_scale
        pos = torch.stack((raw[..., 25:28], raw[..., 32:35]), dim=2)
        goal = torch.stack((raw[..., 41:44], raw[..., 44:47]), dim=2)
        quat = torch.stack((raw[..., 28:32], raw[..., 35:39]), dim=2)
        rot = rotation_matrix(quat)
        tcp_pos = raw[..., 18:21]
        tcp_rot = rotation_matrix(raw[..., 21:25])
        obj_motion = causal_displacement(pos)
        tcp_motion = causal_displacement(tcp_pos)
        robot_features = torch.cat((normalized[..., :18], normalized[..., 18:21],
                                    tcp_rot.flatten(-2), tcp_motion / scale), dim=-1)
        robot = self.robot_encoder(robot_features)
        robot_at_nodes = robot.unsqueeze(2).expand(-1, -1, 2, -1)
        fingers = torch.cat((normalized[..., 7:9], normalized[..., 16:18]), dim=-1)
        fingers = fingers.unsqueeze(2).expand(-1, -1, 2, -1)
        identity = self.object_identity.view(1, 1, 2, 2).expand(raw.shape[0], raw.shape[1], -1, -1)
        tcp_relative_rotation = torch.matmul(tcp_rot.transpose(-1, -2).unsqueeze(2), rot)
        node_features = torch.cat((
            (pos - self.object_center) / scale,
            rot.flatten(-2),
            (goal - pos) / scale,
            (tcp_pos.unsqueeze(2) - pos) / scale,
            tcp_relative_rotation.flatten(-2),
            (pos[..., 2:3] - 0.02) / scale[2],
            obj_motion / scale,
            (tcp_motion.unsqueeze(2) - obj_motion) / scale,
            identity,
            fingers,
        ), dim=-1)
        nodes = self.node_encoder(node_features)
        other_pos = pos.flip(2)
        relative_position = (other_pos - pos) / scale
        relative_rotation = torch.matmul(rot.transpose(-1, -2), rot.flip(2))
        # A zero vertical gap and small lateral offset are support geometry cues,
        # not a measured contact flag. 0.04 m is the supplied cube side length.
        support_gap = (pos[..., 2:3] - other_pos[..., 2:3] - 0.04) / scale[2]
        lateral_distance_squared = relative_position[..., :2].square().sum(-1, keepdim=True)
        edges = torch.cat((relative_position, relative_rotation.flatten(-2),
                           support_gap, lateral_distance_squared,
                           (obj_motion.flip(2) - obj_motion) / scale), dim=-1)
        for relation_round in self.rounds:
            nodes = relation_round(nodes, edges, robot)
        logits = self.slot_logits(torch.cat((nodes, robot_at_nodes), dim=-1)).transpose(-1, -2)
        logits = torch.cat((logits, self.null_logits(robot).unsqueeze(-1)), dim=-1)
        weights = logits.softmax(dim=-1)
        slots = torch.einsum('btkj,btjd->btkd', weights[..., :2], nodes)
        slots = slots + weights[..., 2:3] * self.null_tokens.view(1, 1, 3, 96)
        # The mean path always retains both objects, including a still-held red
        # during the blue invocation. No attention decision gates low-level actions.
        frame = self.frame_encoder(torch.cat((slots.flatten(-2), nodes.mean(2), robot), dim=-1))
        condition = self.temporal_encoder(torch.cat((frame[:, 0], frame[:, 1],
                                                      frame[:, 1] - frame[:, 0]), dim=-1))
        return condition

    def forward(self, noisy_action, timestep, raw_history):
        return self.denoiser(noisy_action, timestep, self.encode(raw_history))


def build_model(spec):
    return RoleRelativeDiffusion(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['raw_obs'])
    diffusion_loss = epsilon_loss(prediction, batch['noise'], batch['mask'])
    # This candidate's prior is architectural. No observation-only penalty is
    # represented as a training objective, and no future state is an input.
    prior_loss = diffusion_loss * 0.0
    return {'loss': diffusion_loss, 'diffusion_loss': diffusion_loss,
            'prior_loss': prior_loss}
