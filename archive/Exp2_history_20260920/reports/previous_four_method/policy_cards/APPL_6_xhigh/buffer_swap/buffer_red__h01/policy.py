import torch
from torch import nn
from appl.public import DiffusionBackbone, epsilon_loss


def mlp(din, hidden, dout):
    return nn.Sequential(nn.Linear(din, hidden), nn.SiLU(), nn.Linear(hidden, dout))


def footprint_xy(quaternion):
    # World XY projection of a cube with 0.02 m half-side; wxyz convention.
    q = quaternion / quaternion.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
    w, x, y, z = q.unbind(-1)
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    return 0.02 * torch.stack((r00.abs() + r01.abs() + r02.abs(),
                               r10.abs() + r11.abs() + r12.abs()), -1)


class GraphRound(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.message = mlp(2 * width + 11, 128, width)
        self.score = mlp(2 * width + 11, 64, 1)
        self.update = mlp(2 * width, 128, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, nodes, edges):
        n = nodes.shape[-2]
        receiver = nodes.unsqueeze(-2).expand(-1, n, n, -1)
        sender = nodes.unsqueeze(-3).expand(-1, n, n, -1)
        pair = torch.cat((receiver, sender, edges), -1)
        weight = self.score(pair).softmax(dim=-2)
        message = (weight * self.message(pair)).sum(-2)
        return self.norm(nodes + self.update(torch.cat((nodes, message), -1)))


class OccupancyDiffusion(nn.Module):
    def __init__(self, spec):
        super().__init__()
        norm = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(norm['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(norm['std'], dtype=torch.float32))
        self.register_buffer('identities', torch.eye(5))
        self.node_encoder = mlp(18, 96, 64)
        self.rounds = nn.ModuleList([GraphRound(), GraphRound()])
        self.robot_encoder = mlp(50, 192, 128)
        self.role_head = mlp(768, 192, 5)
        self.role_embedding = nn.Parameter(torch.randn(5, 32) * 0.02)
        self.role_film = nn.Linear(32, 128)
        self.role_norm = nn.LayerNorm(64)
        self.graph_readout = mlp(640, 256, 192)
        # The blocker-goal layout excludes red/TCP, preventing direct copying of
        # the supported-red label and allowing supervision to transfer earlier.
        self.stage_head = mlp(192, 128, 6)
        self.diffusion = DiffusionBackbone(352, spec['training'])

    def scene(self, raw):
        b, t = raw.shape[:2]
        position = torch.stack((raw[..., 18:21], raw[..., 25:28], raw[..., 32:35],
                                raw[..., 41:44], raw[..., 44:47]), -2)
        unit = torch.zeros_like(raw[..., 21:25])
        unit[..., 0] = 1
        quaternion = torch.stack((raw[..., 21:25], raw[..., 28:32], raw[..., 35:39],
                                  unit, unit), -2)
        extent = torch.stack((torch.zeros_like(raw[..., 18:20]),
                              footprint_xy(raw[..., 28:32]),
                              footprint_xy(raw[..., 35:39]),
                              torch.ones_like(raw[..., 18:20]) * 0.06,
                              torch.ones_like(raw[..., 18:20]) * 0.06), -2)
        scale = self.obs_scale[18:21]
        normalized_position = (position - self.obs_mean[18:21]) / scale
        delta = (position[:, -1] - position[:, 0]) / scale
        motion = delta.unsqueeze(1).expand(-1, t, -1, -1)
        identity = self.identities.view(1, 1, 5, 5).expand(b, t, -1, -1)
        height = (position[..., 2:3] - raw[..., 43:44].unsqueeze(-2)) / scale[2]
        feature = torch.cat((normalized_position, quaternion, identity, motion,
                             extent / scale[:2], height), -1)
        # Directed j-minus-i displacement includes every TCP/object offset.
        displacement = (position.unsqueeze(-3) - position.unsqueeze(-2)) / scale
        distance = displacement.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
        absolute_xy = (position.unsqueeze(-3) - position.unsqueeze(-2))[..., :2].abs()
        receiver_extent = extent.unsqueeze(-2)
        sender_extent = extent.unsqueeze(-3)
        containment = (sender_extent - receiver_extent - absolute_xy) / scale[:2]
        overlap = (sender_extent + receiver_extent - absolute_xy) / scale[:2]
        relative_motion = motion.unsqueeze(-3) - motion.unsqueeze(-2)
        edges = torch.cat((displacement, distance, containment, overlap, relative_motion), -1)
        return feature, edges, position, extent

    def encode(self, raw):
        b, t = raw.shape[:2]
        features, edges, position, extent = self.scene(raw)
        initial = self.node_encoder(features).reshape(b * t, 5, 64)
        full_edges = edges.reshape(b * t, 5, 5, 11)
        nodes = initial
        layout = initial[:, 2:]
        layout_edges = full_edges[:, 2:, 2:]
        for layer in self.rounds:
            nodes = layer(nodes, full_edges)
            layout = layer(layout, layout_edges)
        normalized = (raw - self.obs_mean) / self.obs_scale
        robot = self.robot_encoder(normalized[..., :25].reshape(b, 50))
        graph = nodes.reshape(b, t, 5, 64)
        logits = self.role_head(torch.cat((graph.reshape(b, 640), robot), -1))
        probabilities = logits.softmax(-1)
        film = self.role_film(probabilities @ self.role_embedding)
        gain, bias = film.chunk(2, -1)
        role_graph = self.role_norm(graph * (1 + 0.1 * gain.tanh()[:, None, None, :])
                                   + bias[:, None, None, :])
        summary = self.graph_readout(role_graph.reshape(b, 640))
        layout_summary = layout.reshape(b, t, 3, 64).mean(1).reshape(b, 192)
        staging = self.stage_head(layout_summary)
        mean = staging[:, :3]
        sigma = 0.05 + 0.95 * staging[:, 3:].sigmoid()
        site = self.obs_mean[18:21] + self.obs_scale[18:21] * mean
        latest = position[:, -1]
        relative_site = (site[:, None, :] - latest) / self.obs_scale[18:21]
        site_extent = extent[:, -1, 1]
        goal_margin = (0.06 + site_extent[:, None, :]
                       - (site[:, None, :2] - latest[:, 3:, :2]).abs()) / self.obs_scale[18:20]
        blue_margin = (site_extent + extent[:, -1, 2]
                       - (site[:, :2] - latest[:, 2, :2]).abs()) / self.obs_scale[18:20]
        condition = torch.cat((summary, robot, probabilities, mean, sigma.log(),
                               relative_site.reshape(b, 15), goal_margin.reshape(b, 4),
                               blue_margin), -1)
        aux = {'role_logits': logits, 'site': site, 'site_mean': mean, 'site_sigma': sigma,
               'goal_margin': goal_margin, 'blue_margin': blue_margin}
        return condition, aux

    def predict_with_aux(self, noisy_action, timestep, raw_history):
        condition, aux = self.encode(raw_history)
        return self.diffusion(noisy_action, timestep, condition), aux

    def forward(self, noisy_action, timestep, raw_history):
        prediction, _ = self.predict_with_aux(noisy_action, timestep, raw_history)
        return prediction


def build_model(spec):
    return OccupancyDiffusion(spec)


def outside_regions(red, extent, red_goal, blue_goal):
    goals = torch.stack((red_goal, blue_goal), -2)
    gap = (red[..., None, :2] - goals[..., :2]).abs() - (0.06 + extent[..., None, :])
    # Nonintersection of XY bounding boxes, not the task-success predicate.
    return (gap.amax(-1) > 0).all(-1)


def staging_labels(batch, model):
    future = batch['future_obs']
    red = future[..., 25:28]
    extent = footprint_xy(future[..., 28:32])
    free = outside_regions(red, extent, future[..., 41:44], future[..., 44:47])
    low = (red[..., 2] - future[..., 43]).abs() < 0.006
    separation = (future[..., 18:21] - red).square().sum(-1).sqrt()
    open_or_away = (future[..., 7:9].sum(-1) > 0.06) | (separation > 0.06)
    previous = torch.cat((batch['raw_obs'][:, -1:, 25:28], red[:, :-1]), 1)
    stationary = (red - previous).square().sum(-1) < 0.003 ** 2
    valid = batch['future_mask'][..., 0] > 0
    previous_valid = torch.cat((torch.ones_like(valid[:, :1]), valid[:, :-1]), 1)
    weight = (valid & previous_valid & free & low & open_or_away & stationary).to(red.dtype)
    count = weight.sum(1)
    target_world = (red * weight[..., None]).sum(1) / count.clamp_min(1)[:, None]
    target = (target_world - model.obs_mean[18:21]) / model.obs_scale[18:21]
    return target, (count > 0).to(red.dtype)


def role_labels(batch):
    raw = batch['raw_obs']
    current = raw[:, -1]
    red = current[:, 25:28]
    blue = current[:, 32:35]
    tcp = current[:, 18:21]
    red_distance = (tcp - red).square().sum(-1).sqrt()
    blue_distance = (tcp - blue).square().sum(-1).sqrt()
    free = outside_regions(red, footprint_xy(current[:, 28:32]),
                           current[:, 41:44], current[:, 44:47])
    supported = (red[:, 2] - current[:, 43]).abs() < 0.008
    closed = current[:, 7:9].sum(-1) < 0.05
    # Slot 1 is the action at the current state, not the preceding action.
    commanded_close = (batch['native_action'][:, 1, 7] < 0) & (batch['mask'][:, 1, 0] > 0)
    closing = closed | commanded_close
    red_high = red[:, 2] > current[:, 43] + 0.015
    descending_red = red[:, 2] - raw[:, 0, 27] < -0.0001
    carrying = ((red_distance < 0.055) & closing) | red_high
    releasing = free & (red_distance < 0.10) & (descending_red | (red[:, 2] < current[:, 43] + 0.04))
    blue_approach = free & supported & (red_distance >= 0.10)
    blue_acquire = free & supported & (blue_distance < 0.055) & closing
    label = torch.zeros(current.shape[0], device=current.device, dtype=torch.long)
    label = torch.where(carrying, torch.ones_like(label), label)
    label = torch.where(releasing, torch.full_like(label, 2), label)
    label = torch.where(blue_approach, torch.full_like(label, 3), label)
    label = torch.where(blue_acquire, torch.full_like(label, 4), label)
    return label


def compute_loss(model, batch, spec):
    prediction, aux = model.predict_with_aux(batch['noisy_action'], batch['timesteps'], batch['raw_obs'])
    diffusion_loss = epsilon_loss(prediction, batch['noise'], batch['mask'])
    with torch.no_grad():
        target, valid = staging_labels(batch, model)
        role = role_labels(batch)
    sigma = aux['site_sigma']
    # Gaussian position NLL, shifted by a constant so the minimum is zero.
    nll = (0.5 * ((aux['site_mean'] - target) / sigma).square()
           + (sigma / 0.05).log()).mean(-1)
    staging_loss = (nll * valid).sum() / valid.sum().clamp_min(1)
    role_loss = nn.functional.cross_entropy(aux['role_logits'], role)
    goal_penalty = aux['goal_margin'].amin(-1).relu().square().mean()
    blue_penalty = aux['blue_margin'].amin(-1).relu().square().mean()
    support_penalty = ((aux['site'][:, 2] - batch['raw_obs'][:, -1, 43])
                       / model.obs_scale[20]).square().mean()
    free_site_loss = goal_penalty + blue_penalty + support_penalty
    prior_loss = 0.05 * staging_loss + 0.10 * role_loss + 0.02 * free_site_loss
    return {'loss': diffusion_loss + prior_loss, 'diffusion_loss': diffusion_loss,
            'prior_loss': prior_loss, 'staging_loss': staging_loss,
            'role_loss': role_loss, 'free_site_loss': free_site_loss}
