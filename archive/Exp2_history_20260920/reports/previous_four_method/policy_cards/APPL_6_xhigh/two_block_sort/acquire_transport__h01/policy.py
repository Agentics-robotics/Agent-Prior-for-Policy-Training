import torch
from torch import nn
from appl.public import DiffusionBackbone, epsilon_loss


def quaternion_matrix(q):
    """World rotation from wxyz; exactly insensitive to quaternion sign."""
    q = q / q.square().sum(dim=-1, keepdim=True).clamp_min(1.0e-12).sqrt()
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)
    ), dim=-1).reshape(q.shape[:-1] + (3, 3))


def containment_margins(position, goal, rotation):
    """Signed world-axis margins, not action constraints or contact tests."""
    extent = 0.02 * rotation[..., :2, :].abs().sum(dim=-1)
    xy = 0.06 - extent - (position[..., :2] - goal[..., :2]).abs()
    z = 0.011 - (position[..., 2:3] - goal[..., 2:3]).abs()
    return torch.cat((xy, z), dim=-1)


class GraphBlock(nn.Module):
    def __init__(self, width=192, heads=4):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.SiLU(),
                                nn.Linear(4 * width, width))

    def forward(self, nodes):
        b, n, d = nodes.shape
        qkv = self.qkv(self.norm1(nodes)).reshape(b, n, 3, self.heads, self.head_width)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * (self.head_width ** -0.5), dim=-1)
        message = torch.matmul(weights, v).transpose(1, 2).reshape(b, n, d)
        nodes = nodes + self.projection(message)
        return nodes + self.ff(self.norm2(nodes))


class RoleGraphDiffusion(nn.Module):
    def __init__(self, spec):
        super().__init__()
        normalizer = spec['normalizer']
        mean = torch.tensor(normalizer['mean'], dtype=torch.float32)
        scale = torch.tensor(normalizer['std'], dtype=torch.float32)
        self.register_buffer('obs_mean', mean)
        self.register_buffer('obs_scale', scale)
        # Derived ONLY from the supplied complete-demonstration normalizer.
        # The same spatial scale is used for both slots and their relations.
        spatial_scale = torch.maximum(scale[18:21], torch.maximum(scale[25:28], scale[32:35]))
        self.register_buffer('spatial_scale', spatial_scale)
        self.register_buffer('spatial_center', mean[18:21].clone())
        self.register_buffer('order_tag', torch.tensor([-1.0, 1.0]))
        self.register_buffer('gravity', torch.tensor([0.0, 0.0, -1.0]))
        width = 192
        self.object_encoder = nn.Sequential(nn.Linear(102, width), nn.SiLU(),
                                            nn.Linear(width, width), nn.LayerNorm(width))
        self.robot_encoder = nn.Sequential(nn.Linear(99, width), nn.SiLU(),
                                           nn.Linear(width, width), nn.LayerNorm(width))
        self.scene_graph = nn.Sequential(GraphBlock(width), GraphBlock(width))
        self.object_role_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2))
        self.null_role_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.role_fusion = nn.Sequential(nn.Linear(width + 2, width), nn.SiLU(), nn.Linear(width, width))
        self.bound_graph = GraphBlock(width)
        self.null_predecessor = nn.Parameter(torch.zeros(width))
        self.condition_encoder = nn.Sequential(
            nn.Linear(4 * width + 1 + 99, 384), nn.SiLU(),
            nn.Linear(384, 256), nn.LayerNorm(256))
        self.denoiser = DiffusionBackbone(256, spec['training'])

    def features(self, history):
        b, t, _ = history.shape
        norm = (history - self.obs_mean) / self.obs_scale
        tcp_position = history[..., 18:21]
        tcp_rotation = quaternion_matrix(history[..., 21:25])
        poses = torch.stack((history[..., 25:32], history[..., 32:39]), dim=2)
        goals = torch.stack((history[..., 41:44], history[..., 44:47]), dim=2)
        position = poses[..., :3]
        rotation = quaternion_matrix(poses[..., 3:7])
        relative_rotation = torch.matmul(tcp_rotation.unsqueeze(2).transpose(-1, -2), rotation)
        margins = containment_margins(position, goals, rotation)
        tags = self.order_tag.reshape(1, 1, 2, 1).expand(b, t, 2, 1)
        obj = torch.cat((
            (position - self.spatial_center) / self.spatial_scale,
            (goals - self.spatial_center) / self.spatial_scale,
            (goals - position) / self.spatial_scale,
            (position - tcp_position.unsqueeze(2)) / self.spatial_scale,
            rotation.flatten(-2), relative_rotation.flatten(-2),
            margins / self.spatial_scale, tags
        ), dim=-1)
        robot = torch.cat((norm[..., :18], norm[..., 18:21], tcp_rotation.flatten(-2),
                           self.gravity.reshape(1, 1, 3).expand(b, t, 3)), dim=-1)
        # Displacements are per observation interval, not claimed metric velocities.
        obj_history = torch.cat((obj[:, -1], obj[:, -2], obj[:, -1] - obj[:, -2]), dim=-1)
        robot_history = torch.cat((robot[:, -1], robot[:, -2], robot[:, -1] - robot[:, -2]), dim=-1)
        return obj_history, robot_history

    def encode_scene(self, history, permute=False):
        obj_features, robot_features = self.features(history)
        if permute:
            swap = torch.rand(history.shape[0], 1, 1, device=history.device) < 0.5
            # Pose, goal, geometry and task-order tag travel together. No joint swap.
            obj_features = torch.where(swap, obj_features.flip(1), obj_features)
        obj_nodes = self.object_encoder(obj_features)
        robot_node = self.robot_encoder(robot_features).unsqueeze(1)
        nodes = self.scene_graph(torch.cat((robot_node, obj_nodes), dim=1))
        role_scores = self.object_role_head(nodes[:, 1:])
        requested_logits = role_scores[..., 0]
        predecessor_logits = torch.cat((role_scores[..., 1], self.null_role_head(nodes[:, 0])), dim=1)
        return nodes, robot_features, requested_logits, predecessor_logits

    def condition(self, history):
        nodes, robot_features, requested_logits, predecessor_logits = self.encode_scene(history, permute=self.training)
        requested = torch.softmax(requested_logits, dim=-1)
        predecessor = torch.softmax(predecessor_logits, dim=-1)
        role_flags = torch.stack((requested, predecessor[:, :2]), dim=-1)
        objects = nodes[:, 1:] + self.role_fusion(torch.cat((nodes[:, 1:], role_flags), dim=-1))
        nodes = self.bound_graph(torch.cat((nodes[:, :1], objects), dim=1))
        objects = nodes[:, 1:]
        requested_context = (objects * requested.unsqueeze(-1)).sum(dim=1)
        predecessor_context = (objects * predecessor[:, :2].unsqueeze(-1)).sum(dim=1)
        predecessor_context = predecessor_context + predecessor[:, 2:3] * self.null_predecessor
        # The mean context and robot attention retain BOTH objects even when one
        # role is highly confident. Requested blue never erases a held red node.
        combined = torch.cat((nodes[:, 0], requested_context, predecessor_context,
                              objects.mean(dim=1), predecessor[:, 2:3], robot_features), dim=-1)
        return self.condition_encoder(combined)

    def forward(self, noisy_action, timestep, raw_history):
        return self.denoiser(noisy_action, timestep, self.condition(raw_history))

    def role_loss(self, history):
        # The fixed deployment signature lacks the original caller role IDs.
        # Use causal, training-only pseudo-labels valid for the demonstrated order.
        # Red aligned in XY can STILL be held above its pad: z is not tested here.
        with torch.no_grad():
            current = history[:, -1]
            red_rotation = quaternion_matrix(current[:, 28:32])
            red_margins = containment_margins(current[:, 25:28], current[:, 41:44], red_rotation)
            second_occurrence = (red_margins[:, :2] >= 0).all(dim=-1)
            requested_target = second_occurrence.long()
            predecessor_target = torch.where(second_occurrence,
                                             torch.zeros_like(requested_target),
                                             torch.full_like(requested_target, 2))
        # Canonical slot order for these labels; shared weights and graph pooling
        # make the stochastic internal permutation in forward role-consistent.
        _, _, requested_logits, predecessor_logits = self.encode_scene(history, permute=False)
        req_loss = nn.functional.cross_entropy(requested_logits, requested_target)
        pred_loss = nn.functional.cross_entropy(predecessor_logits, predecessor_target)
        return 0.5 * (req_loss + pred_loss)


def build_model(spec):
    return RoleGraphDiffusion(spec)


def compute_loss(model, batch, spec):
    prediction = model(batch['noisy_action'], batch['timesteps'], batch['raw_obs'])
    diffusion_loss = epsilon_loss(prediction, batch['noise'], batch['mask'])
    prior_loss = model.role_loss(batch['raw_obs'])
    return {'loss': diffusion_loss + 0.1 * prior_loss,
            'diffusion_loss': diffusion_loss, 'prior_loss': prior_loss}
