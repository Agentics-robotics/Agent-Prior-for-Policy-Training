import math
import torch
from appl.public import epsilon_loss


# All poses are world-frame; quaternion order is wxyz.
def rotation_matrix(q):
    q = q / q.square().sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)
    ), dim=-1).reshape(q.shape[:-1] + (3, 3))


def feature_mlp(input_dim, width):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, width), torch.nn.GELU(),
        torch.nn.Linear(width, width), torch.nn.LayerNorm(width))


class ActionBlock(torch.nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.self_norm = torch.nn.LayerNorm(width)
        self.cross_norm = torch.nn.LayerNorm(width)
        self.ff_norm = torch.nn.LayerNorm(width)
        self.self_attention = torch.nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.cross_attention = torch.nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.ff = torch.nn.Sequential(torch.nn.Linear(width, 4*width), torch.nn.GELU(),
                                      torch.nn.Linear(4*width, width))
        self.time_modulation = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(width, 6*width))
        torch.nn.init.zeros_(self.time_modulation[-1].weight)
        torch.nn.init.zeros_(self.time_modulation[-1].bias)

    def forward(self, x, memory, time):
        b1, s1, b2, s2, b3, s3 = self.time_modulation(time).chunk(6, dim=-1)
        q = self.self_norm(x) * (1 + s1[:, None]) + b1[:, None]
        x = x + self.self_attention(q, q, q, need_weights=False)[0]
        q = self.cross_norm(x) * (1 + s2[:, None]) + b2[:, None]
        x = x + self.cross_attention(q, memory, memory, need_weights=False)[0]
        q = self.ff_norm(x) * (1 + s3[:, None]) + b3[:, None]
        return x + self.ff(q)


class ContainmentDiffusion(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        width, heads = 192, 6
        self.horizon = int(spec['training']['horizon'])
        normalizer = spec['normalizer']
        mean = torch.tensor(normalizer['mean'], dtype=torch.float32)
        scale = torch.tensor(normalizer['std'], dtype=torch.float32)
        self.register_buffer('obs_mean', mean)
        self.register_buffer('obs_scale', scale)
        self.register_buffer('pad', torch.tensor([-0.18, -0.30, 0.02]))
        self.register_buffer('red_half_xy', torch.tensor([0.06, 0.06]))
        self.register_buffer('blue_half_xy', torch.tensor([0.172, 0.182]))
        # These are shared original-demonstration half-ranges, not skill fits.
        red_scale = scale[25:28].clone()
        blue_scale = scale[32:35].clone()
        blue_scale[0] = torch.maximum(blue_scale[0], scale[39])
        self.register_buffer('red_target_scale', red_scale)
        self.register_buffer('blue_target_scale', blue_scale)
        self.register_buffer('red_tcp_scale', torch.maximum(scale[25:28], scale[18:21]))
        self.register_buffer('blue_tcp_scale', torch.maximum(scale[32:35], scale[18:21]))
        margin_scale = torch.stack((red_scale[0], red_scale[1], red_scale[2], red_scale[2],
                                    blue_scale[0], blue_scale[1], blue_scale[2], blue_scale[2], scale[39]))
        self.register_buffer('margin_scale', margin_scale)
        self.register_buffer('margin_loss_weights', torch.tensor([1., 1., 2., 2., 1., 1., 2., 2., 1.]))
        self.register_buffer('future_indices', torch.tensor([1, 4, 8, 15], dtype=torch.long))
        self.register_buffer('time_frequencies', torch.exp(-math.log(10000.0) * torch.arange(64).float() / 63.0))

        # Each causal token includes its own inter-observation displacement.
        self.robot_encoder = feature_mlp(2*30, width)
        self.object_encoder = feature_mlp(2*25, width)
        self.geometry_encoder = feature_mlp(2*18, width)
        self.margin_encoder = feature_mlp(2*9, width)
        self.token_type = torch.nn.Parameter(torch.randn(1, 1, 5, width) * 0.02)
        self.history_position = torch.nn.Parameter(torch.randn(1, 2, 1, width) * 0.02)
        self.observation_layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(d_model=width, nhead=heads, dim_feedforward=4*width,
                dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
            for _ in range(2)])
        self.memory_norm = torch.nn.LayerNorm(width)
        self.action_projection = torch.nn.Linear(8, width)
        self.action_position = torch.nn.Parameter(torch.randn(1, self.horizon, width) * 0.02)
        self.time_encoder = torch.nn.Sequential(torch.nn.Linear(128, 2*width), torch.nn.SiLU(),
                                                torch.nn.Linear(2*width, width))
        self.action_layers = torch.nn.ModuleList([ActionBlock(width, heads) for _ in range(4)])
        self.output_norm = torch.nn.LayerNorm(width)
        self.epsilon_head = torch.nn.Linear(width, 8)
        self.future_head = torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.SiLU(),
                                                torch.nn.Linear(width, 9))
        torch.nn.init.normal_(self.future_head[-1].weight, std=0.01)
        torch.nn.init.zeros_(self.future_head[-1].bias)

    def geometry(self, raw):
        red_r = rotation_matrix(raw[..., 28:32])
        blue_r = rotation_matrix(raw[..., 35:39])
        red_extent = 0.02 * red_r.abs().sum(dim=-1)
        blue_extent = 0.02 * blue_r.abs().sum(dim=-1)
        d = raw[..., 39]
        cavity = torch.stack((0.19 - d, torch.zeros_like(d), torch.full_like(d, 0.035)), dim=-1)
        red_xy = self.red_half_xy - (raw[..., 25:27] - self.pad[:2]).abs() - red_extent[..., :2]
        blue_xy = self.blue_half_xy - (raw[..., 32:34] - cavity[..., :2]).abs() - blue_extent[..., :2]
        red_z = raw[..., 27:28]
        blue_z = raw[..., 34:35]
        margins = torch.cat((red_xy, red_z - 0.014, 0.031 - red_z,
                             blue_xy, blue_z - 0.053, 0.074 - blue_z,
                             raw[..., 39:40] - 0.26), dim=-1)
        return cavity, red_r, blue_r, red_extent, blue_extent, margins / self.margin_scale

    def normalized_margins(self, raw):
        return self.geometry(raw)[-1]

    @staticmethod
    def with_displacement(x):
        displacement = torch.cat((torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]), dim=1)
        return torch.cat((x, displacement), dim=-1)

    def encode_history(self, raw):
        n = (raw - self.obs_mean) / self.obs_scale
        cavity, red_r, blue_r, red_extent, blue_extent, margins = self.geometry(raw)
        tcp = raw[..., 18:21]
        tcp_r = rotation_matrix(raw[..., 21:25])
        robot = torch.cat((n[..., :18], n[..., 18:21], tcp_r.flatten(-2)), dim=-1)
        red = torch.cat((n[..., 25:28], red_r.flatten(-2), red_extent / self.obs_scale[25:28],
                         (raw[..., 25:28] - tcp) / self.red_tcp_scale,
                         (raw[..., 25:28] - self.pad) / self.red_target_scale, margins[..., :4]), dim=-1)
        blue = torch.cat((n[..., 32:35], blue_r.flatten(-2), blue_extent / self.obs_scale[32:35],
                          (raw[..., 32:35] - tcp) / self.blue_tcp_scale,
                          (raw[..., 32:35] - cavity) / self.blue_target_scale, margins[..., 4:8]), dim=-1)
        red_half = (self.red_half_xy / self.red_target_scale[:2]).expand(raw.shape[0], raw.shape[1], 2)
        blue_half = (self.blue_half_xy / self.blue_target_scale[:2]).expand(raw.shape[0], raw.shape[1], 2)
        pad_position = ((self.pad - self.obs_mean[25:28]) / self.obs_scale[25:28]).expand_as(tcp)
        geometry = torch.cat((n[..., 39:41], (cavity - tcp) / self.blue_tcp_scale,
                              (cavity - self.obs_mean[32:35]) / self.obs_scale[32:35],
                              (self.pad - tcp) / self.red_tcp_scale, pad_position, red_half, blue_half), dim=-1)
        tokens = torch.stack((
            self.robot_encoder(self.with_displacement(robot)),
            self.object_encoder(self.with_displacement(red)),
            self.object_encoder(self.with_displacement(blue)),
            self.geometry_encoder(self.with_displacement(geometry)),
            self.margin_encoder(self.with_displacement(margins))), dim=2)
        tokens = tokens + self.token_type + self.history_position
        memory = tokens.flatten(1, 2)
        for layer in self.observation_layers:
            memory = layer(memory)
        return self.memory_norm(memory), margins[:, -1]

    def denoise(self, noisy_action, timestep, raw_history):
        memory, current_margins = self.encode_history(raw_history)
        t = torch.as_tensor(timestep, device=noisy_action.device, dtype=noisy_action.dtype).reshape(-1)
        t = t.expand(noisy_action.shape[0])
        angles = t[:, None] * self.time_frequencies[None]
        time = self.time_encoder(torch.cat((angles.sin(), angles.cos()), dim=-1))
        x = self.action_projection(noisy_action) + self.action_position[:, :noisy_action.shape[1]] + time[:, None]
        for layer in self.action_layers:
            x = layer(x, memory, time)
        latent = self.output_norm(x)
        return self.epsilon_head(latent), latent, current_margins

    def forward(self, noisy_action, timestep, raw_history):
        return self.denoise(noisy_action, timestep, raw_history)[0]


def build_model(spec):
    return ContainmentDiffusion(spec)


def compute_loss(model, batch, spec):
    predicted_noise, latent, current = model.denoise(batch['noisy_action'], batch['timesteps'], batch['raw_obs'])
    diffusion = epsilon_loss(predicted_noise, batch['noise'], batch['mask'])
    # Slot j is the state at t+j, AFTER action t-1+j. Do not train on trivial slot 0.
    idx = model.future_indices
    predicted_future = current[:, None, :] + model.future_head(latent.index_select(1, idx))
    with torch.no_grad():
        target = model.normalized_margins(batch['future_obs'].index_select(1, idx))
        valid = batch['future_mask'].index_select(1, idx) * batch['mask'].index_select(1, idx)
    error = torch.nn.functional.smooth_l1_loss(predicted_future, target, reduction='none', beta=0.1)
    weights = model.margin_loss_weights
    prior = (error * valid * weights).sum() / (valid.sum() * weights.sum()).clamp_min(1.0)
    loss = diffusion + 0.2 * prior
    return {'loss': loss, 'diffusion_loss': diffusion, 'prior_loss': prior}
