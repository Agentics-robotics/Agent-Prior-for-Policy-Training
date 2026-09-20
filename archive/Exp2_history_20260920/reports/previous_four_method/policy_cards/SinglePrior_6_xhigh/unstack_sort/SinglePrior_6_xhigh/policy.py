import torch
from appl.public import DiffusionBackbone, epsilon_loss


def mlp(input_dim, hidden_dim, output_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.Mish(),
        torch.nn.Linear(hidden_dim, output_dim),
    )


def rotation_columns(quaternion):
    """First two rotation-matrix columns, using the interface's wxyz convention."""
    q = quaternion / quaternion.square().sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack((
        1 - 2 * (y * y + z * z),
        2 * (x * y + w * z),
        2 * (x * z - w * y),
        2 * (x * y - w * z),
        1 - 2 * (x * x + z * z),
        2 * (y * z + w * x),
    ), dim=-1)


def object_positions(raw):
    return torch.stack((raw[..., 25:28], raw[..., 32:35]), dim=-2)


def object_goals(raw):
    return torch.stack((raw[..., 41:44], raw[..., 44:47]), dim=-2)


def task_relations(raw):
    """Six physical relation coordinates per object; no phase or success labels."""
    p = object_positions(raw)
    tcp = raw[..., 18:21].unsqueeze(-2)
    return torch.cat((p - tcp, object_goals(raw) - p), dim=-1)


class RelationalPredictivePolicy(torch.nn.Module):
    """One epsilon-predicting U-Net with causally forecast relation conditioning."""

    def __init__(self, spec):
        super().__init__()
        normalizer = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(normalizer['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(normalizer['std'], dtype=torch.float32))
        self.register_buffer('object_color', torch.tensor([-1.0, 1.0]).reshape(1, 2, 1))
        self.horizon = int(spec['training']['horizon'])
        cfg = spec.get('candidate_config', {})
        self.prior_weight = float(cfg.get('prior_weight', 0.1))
        self.aperture_weight = float(cfg.get('aperture_weight', 0.25))
        self.forecast_gain = float(cfg.get('forecast_gain', 0.25))

        # Raw state preserves the robot's absolute joint/action coordinate frame.
        self.robot_encoder = torch.nn.Sequential(mlp(94, 256, 128), torch.nn.LayerNorm(128))
        # 2 * 41 history features, 11 motion features, and one color coordinate.
        self.object_encoder = torch.nn.Sequential(mlp(94, 256, 128), torch.nn.LayerNorm(128))
        # The same message function is used in both directions, with robot context.
        self.message = mlp(384, 256, 128)
        self.node_norm = torch.nn.LayerNorm(128)

        # Forecast changes, not unconditional goal achievement. The horizon slots
        # match the supplied action/future observation labels, including slot zero.
        self.relation_forecast = mlp(256, 256, self.horizon * 6)
        self.aperture_forecast = mlp(256, 128, self.horizon)
        for head in (self.relation_forecast, self.aperture_forecast):
            torch.nn.init.normal_(head[-1].weight, mean=0.0, std=0.001)
            torch.nn.init.zeros_(head[-1].bias)
        self.relation_projection = mlp(self.horizon * 6, 128, 128)
        self.aperture_projection = mlp(self.horizon, 64, 128)
        self.backbone = DiffusionBackbone(384, spec['training'])

    def object_features(self, raw_history):
        b = raw_history.shape[0]
        p = object_positions(raw_history)                         # B, 2, 2, 3
        goal = object_goals(raw_history)
        tcp = raw_history[..., 18:21].unsqueeze(2).expand(-1, -1, 2, -1)
        peer = p.flip(2)
        quaternion = torch.stack((raw_history[..., 28:32], raw_history[..., 35:39]), dim=2)
        obj_rotation = rotation_columns(quaternion)
        tcp_rotation = rotation_columns(raw_history[..., 21:25]).unsqueeze(2).expand(-1, -1, 2, -1)
        fingers = raw_history[..., 7:9].unsqueeze(2).expand(-1, -1, 2, -1)
        to_tcp = (p - tcp) / 0.25
        to_goal = (goal - p) / 0.25
        to_peer = (peer - p) / 0.25
        squared_distances = torch.stack((
            to_tcp.square().sum(dim=-1),
            to_goal.square().sum(dim=-1),
            to_peer.square().sum(dim=-1),
        ), dim=-1)
        frame = torch.cat((
            to_tcp, to_goal, to_peer, (goal - tcp) / 0.25,
            p / 0.5, goal / 0.5,
            obj_rotation, tcp_rotation, obj_rotation.flip(2),
            fingers / 0.04, squared_distances,
        ), dim=-1)                                               # B, 2, 2, 41
        temporal = frame.permute(0, 2, 1, 3).reshape(b, 2, 82)
        # 0.025 m per observation interval equals 0.5 m/s at 20 Hz.
        # Physical scales avoid dividing by narrow empirical pose subranges.
        motion = torch.cat((
            (p[:, 1] - p[:, 0]) / 0.025,
            (tcp[:, 1] - tcp[:, 0]) / 0.025,
            (peer[:, 1] - peer[:, 0]) / 0.025,
            (fingers[:, 1] - fingers[:, 0]) / 0.005,
        ), dim=-1)
        return torch.cat((temporal, motion, self.object_color.expand(b, -1, -1)), dim=-1)

    def encode(self, raw_history):
        b = raw_history.shape[0]
        normalized = (raw_history - self.obs_mean) / self.obs_scale
        robot = self.robot_encoder(normalized.reshape(b, 94))
        nodes = self.object_encoder(self.object_features(raw_history))
        robot_pair = robot.unsqueeze(1).expand(-1, 2, -1)
        messages = self.message(torch.cat((nodes, nodes.flip(1), robot_pair), dim=-1))
        nodes = self.node_norm(nodes + messages)
        relation_delta = self.relation_forecast(torch.cat((nodes, robot_pair), dim=-1))
        relation_delta = relation_delta.reshape(b, 2, self.horizon, 6)
        aperture_delta = self.aperture_forecast(torch.cat((robot, nodes.mean(dim=1)), dim=-1))

        # Forecasts are causal learned features, not externally supplied futures.
        # The direct residual paths retain information and diffusion multimodality
        # when the deterministic forecast averages ambiguous demonstration pauses.
        node_condition = nodes + self.forecast_gain * self.relation_projection(
            relation_delta.reshape(b, 2, self.horizon * 6))
        robot_condition = robot + self.forecast_gain * self.aperture_projection(aperture_delta)
        condition = torch.cat((robot_condition, node_condition.reshape(b, 256)), dim=-1)
        return condition, relation_delta, aperture_delta

    def forward(self, noisy_action, timestep, raw_history):
        condition, _, _ = self.encode(raw_history)
        return self.backbone(noisy_action, timestep, condition)


def build_model(spec):
    return RelationalPredictivePolicy(spec)


def compute_loss(model, batch, spec):
    raw = batch['raw_obs']
    condition, relation_delta, aperture_delta = model.encode(raw)
    predicted_noise = model.backbone(batch['noisy_action'], batch['timesteps'], condition)
    diffusion_loss = epsilon_loss(predicted_noise, batch['noise'], batch['mask'])

    # future_obs is exclusively a supervised target in training. Slot j is the
    # state AFTER action t-1+j; slot zero therefore normally equals current state.
    future = batch['future_obs']
    now_relation = task_relations(raw[:, -1])                      # B, 2, 6
    future_relation = task_relations(future).permute(0, 2, 1, 3)   # B, 2, H, 6
    target_relation_delta = ((future_relation - now_relation.unsqueeze(2)) / 0.1).detach()
    now_aperture = raw[:, -1, 7:9].mean(dim=-1)
    future_aperture = future[..., 7:9].mean(dim=-1)
    target_aperture_delta = ((future_aperture - now_aperture.unsqueeze(1)) / 0.02).detach()

    valid = batch['future_mask'] * batch['mask']                   # B, H, 1
    relation_error = torch.nn.functional.smooth_l1_loss(
        relation_delta, target_relation_delta, reduction='none', beta=0.25)
    relation_loss = (relation_error * valid.unsqueeze(1)).sum() / (valid.sum() * 12).clamp_min(1)
    aperture_error = torch.nn.functional.smooth_l1_loss(
        aperture_delta, target_aperture_delta, reduction='none', beta=0.25)
    aperture_loss = (aperture_error * valid.squeeze(-1)).sum() / valid.sum().clamp_min(1)
    prior_loss = relation_loss + model.aperture_weight * aperture_loss
    loss = diffusion_loss + model.prior_weight * prior_loss
    return {'loss': loss, 'diffusion_loss': diffusion_loss, 'prior_loss': prior_loss}
