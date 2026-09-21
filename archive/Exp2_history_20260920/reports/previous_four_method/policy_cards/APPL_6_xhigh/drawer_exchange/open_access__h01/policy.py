import torch
from torch import nn
from appl.public import DiffusionBackbone, epsilon_loss


def mlp(n_in, n_hidden, n_out):
    return nn.Sequential(nn.Linear(n_in, n_hidden), nn.SiLU(),
                         nn.Linear(n_hidden, n_out))


def rotation_matrix_features(quaternion):
    # wxyz. All products are unchanged when the quaternion sign is reversed.
    q = quaternion / quaternion.square().sum(-1, keepdim=True).sqrt().clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y - z*w),
                        2 * (x*z + y*w), 2 * (x*y + z*w),
                        1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
                        2 * (x*z - y*w), 2 * (y*z + x*w),
                        1 - 2 * (x*x + y*y)), dim=-1)


class AxialEdge(nn.Module):
    def __init__(self):
        super().__init__()
        self.axial = mlp(1, 32, 32)
        self.transverse = mlp(2, 32, 32)

    def forward(self, relative_position):
        return torch.cat((self.axial(relative_position[..., :1]),
                          self.transverse(relative_position[..., 1:])), dim=-1)


class TypedGraphRound(nn.Module):
    def __init__(self):
        super().__init__()
        # robot, drawer, red, blue; each edge has its own two directed maps.
        self.pairs = ((0, 1), (2, 1), (3, 1), (2, 0), (3, 0))
        self.to_first = nn.ModuleList([mlp(192, 128, 64) for _ in self.pairs])
        self.to_second = nn.ModuleList([mlp(192, 128, 64) for _ in self.pairs])
        self.update = nn.ModuleList([mlp(128, 128, 64) for _ in range(4)])
        self.norm = nn.ModuleList([nn.LayerNorm(64) for _ in range(4)])
        self.degrees = (3.0, 3.0, 2.0, 2.0)

    def forward(self, nodes, edges):
        incoming = [torch.zeros_like(n) for n in nodes]
        for k, (i, j) in enumerate(self.pairs):
            # The edge descriptor has a fixed typed direction (first minus second).
            # Separate maps learn the interpretation at the two endpoints.
            a = torch.cat((nodes[i], nodes[j], edges[k]), dim=-1)
            b = torch.cat((nodes[j], nodes[i], edges[k]), dim=-1)
            incoming[i] = incoming[i] + self.to_first[k](a)
            incoming[j] = incoming[j] + self.to_second[k](b)
        return [self.norm[i](nodes[i] + self.update[i](torch.cat(
                    (nodes[i], incoming[i] / self.degrees[i]), dim=-1)))
                for i in range(4)]


class PrismaticEncoder(nn.Module):
    def __init__(self, spec):
        super().__init__()
        mean = torch.tensor(spec['normalizer']['mean'], dtype=torch.float32)
        scale = torch.tensor(spec['normalizer']['std'], dtype=torch.float32)
        self.register_buffer('obs_mean', mean)
        self.register_buffer('obs_scale', scale)
        self.register_buffer('drawer_origin', torch.tensor([0.19, 0.0, 0.035]))
        # Conservative half-range sums for differences, using only the complete-
        # demonstration shared normalizer. Constant drawer Y/Z add no range.
        center_range = torch.stack((scale[39], scale[39]*0, scale[39]*0))
        edge_scale = torch.stack((scale[18:21] + center_range,
                                  scale[25:28] + center_range,
                                  scale[32:35] + center_range,
                                  scale[25:28] + scale[18:21],
                                  scale[32:35] + scale[18:21]))
        self.register_buffer('edge_scale', edge_scale)
        self.node_encoders = nn.ModuleList([
            mlp(30, 128, 64),  # qpos, qvel, TCP world position and rotation
            mlp(5, 64, 64),   # articulation, speed, world center
            mlp(12, 64, 64),  # red world position and rotation
            mlp(12, 64, 64),  # blue world position and rotation
        ])
        self.drawer_edges = nn.ModuleList([AxialEdge() for _ in range(3)])
        self.object_edges = nn.ModuleList([mlp(3, 64, 64) for _ in range(2)])
        self.rounds = nn.ModuleList([TypedGraphRound(), TypedGraphRound()])
        # Ordered causal frames and their latent difference, without a clock,
        # persistent phase state, source indices or trajectory identifiers.
        self.temporal = mlp(768, 384, 256)

    def forward(self, raw_history):
        x = (raw_history - self.obs_mean) / self.obs_scale
        d = raw_history[..., 39:40]
        zero = torch.zeros_like(d)
        center = self.drawer_origin + torch.cat((-d, zero, zero), dim=-1)
        tcp = raw_history[..., 18:21]
        red = raw_history[..., 25:28]
        blue = raw_history[..., 32:35]
        world_center = (center - self.obs_mean[18:21]) / self.obs_scale[18:21]
        node_inputs = [
            torch.cat((x[..., :18], x[..., 18:21],
                       rotation_matrix_features(raw_history[..., 21:25])), dim=-1),
            torch.cat((x[..., 39:41], world_center), dim=-1),
            torch.cat((x[..., 25:28],
                       rotation_matrix_features(raw_history[..., 28:32])), dim=-1),
            torch.cat((x[..., 32:35],
                       rotation_matrix_features(raw_history[..., 35:39])), dim=-1),
        ]
        nodes = [enc(value) for enc, value in zip(self.node_encoders, node_inputs)]
        relatives = (tcp-center, red-center, blue-center, red-tcp, blue-tcp)
        edges = [self.drawer_edges[i](relatives[i] / self.edge_scale[i])
                 for i in range(3)]
        edges = edges + [self.object_edges[i](relatives[i+3] / self.edge_scale[i+3])
                         for i in range(2)]
        for layer in self.rounds:
            nodes = layer(nodes, edges)
        frames = torch.cat(nodes, dim=-1)
        previous, current = frames[:, 0], frames[:, 1]
        return self.temporal(torch.cat((previous, current, current-previous), dim=-1))


class PrismaticDiffusionPolicy(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.encoder = PrismaticEncoder(spec)
        self.denoiser = DiffusionBackbone(256, spec['training'])
        self.drawer_head = mlp(256, 128, spec['training']['horizon'])
        self.prior_weight = 0.25

    def forward(self, noisy_action, timestep, raw_history):
        condition = self.encoder(raw_history)
        return self.denoiser(noisy_action, timestep, condition)

    def predict_drawer_displacements(self, raw_history):
        return self.drawer_head(self.encoder(raw_history))


def build_model(spec):
    return PrismaticDiffusionPolicy(spec)


def compute_loss(model, batch, spec):
    predicted_noise = model(batch['noisy_action'], batch['timesteps'], batch['raw_obs'])
    diffusion_loss = epsilon_loss(predicted_noise, batch['noise'], batch['mask'])
    predictions = model.predict_drawer_displacements(batch['raw_obs'])
    # Action slot 0 is t-1 and its post-state is the already observed state t.
    # Slot 1 is t and its post-state is t+1: this is the one-step auxiliary.
    # The remaining slots extend the same displacement prediction across the
    # training chunk. No future observation ever enters the encoder or sampler.
    current_d = batch['raw_obs'][:, -1, 39:40]
    targets = ((batch['future_obs'][..., 39] - current_d)
               / model.encoder.obs_scale[39]).detach()
    valid = (batch['future_mask'][..., 0] * batch['mask'][..., 0]).clone()
    valid[:, 0] = 0
    prior_loss = ((predictions-targets).square()*valid).sum() / valid.sum().clamp_min(1)
    loss = diffusion_loss + model.prior_weight * prior_loss
    return {'loss': loss, 'diffusion_loss': diffusion_loss, 'prior_loss': prior_loss}
