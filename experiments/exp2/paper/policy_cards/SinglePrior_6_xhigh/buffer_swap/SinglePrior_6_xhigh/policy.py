import torch
from appl.public import DiffusionBackbone, epsilon_loss


class Dense(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, normalize=False):
        super().__init__()
        layers = [torch.nn.Linear(input_dim, hidden_dim), torch.nn.Mish(),
                  torch.nn.Linear(hidden_dim, output_dim)]
        if normalize:
            layers.extend([torch.nn.LayerNorm(output_dim), torch.nn.Mish()])
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class OccupancyForecastPolicy(torch.nn.Module):
    """One epsilon model with causal, role-aware object relational conditioning."""

    def __init__(self, spec):
        super().__init__()
        normalizer = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(normalizer['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(normalizer['std'], dtype=torch.float32))
        self.register_buffer('role_identity', torch.eye(2, dtype=torch.float32))
        self.register_buffer('goal_distance_scale', torch.tensor([0.06, 0.06, 0.03], dtype=torch.float32))
        self.register_buffer('forecast_slots', torch.tensor([4, 8, 15], dtype=torch.long))
        self.register_buffer('forecast_indices', torch.tensor(
            [18, 19, 20, 25, 26, 27, 32, 33, 34, 7, 8], dtype=torch.long))
        self.register_buffer('forecast_scale', torch.tensor([0.10] * 9 + [0.04] * 2, dtype=torch.float32))
        self.prior_weight = float(spec.get('candidate_config', {}).get('forecast_weight', 0.10))
        self.robot_encoder = Dense(94, 192, 96, normalize=True)
        self.object_encoder = Dense(68, 192, 96, normalize=True)
        self.message_encoder = Dense(288, 192, 96)
        self.message_norm = torch.nn.LayerNorm(96)
        self.forecast_head = Dense(288, 192, 33)
        # The same learned forecasts, never future labels, enter train and inference.
        self.backbone = DiffusionBackbone(415, spec['training'])

    def object_features(self, raw_history):
        batch_size = raw_history.shape[0]
        tcp = raw_history[..., 18:21].unsqueeze(-2)
        objects = torch.stack([raw_history[..., 25:28], raw_history[..., 32:35]], dim=-2)
        goals = torch.stack([raw_history[..., 41:44], raw_history[..., 44:47]], dim=-2)
        rotations = torch.stack([raw_history[..., 28:32], raw_history[..., 35:39]], dim=-2)
        other_objects = objects.flip(-2)
        other_goals = goals.flip(-2)
        relative_tcp = objects - tcp
        to_goal = goals - objects
        blocker = other_objects - goals
        to_other = other_objects - objects
        goal_from_tcp = goals - tcp
        to_other_goal = other_goals - objects
        metric_vectors = torch.cat([relative_tcp, to_goal, blocker, to_other,
                                    goal_from_tcp, to_other_goal], dim=-1) / 0.25
        contact = torch.exp(-0.5 * (relative_tcp / 0.04).square().sum(dim=-1, keepdim=True))
        own_goal_proximity = torch.exp(-0.5 * (to_goal / self.goal_distance_scale).square().sum(dim=-1, keepdim=True))
        blocker_proximity = torch.exp(-0.5 * (blocker / self.goal_distance_scale).square().sum(dim=-1, keepdim=True))
        other_goal_proximity = torch.exp(-0.5 * (to_other_goal / self.goal_distance_scale).square().sum(dim=-1, keepdim=True))
        height = (objects[..., 2:3] - goals[..., 2:3]) / 0.25
        aperture = (raw_history[..., 7:9] / 0.04).unsqueeze(-2).expand(-1, -1, 2, -1)
        identity = self.role_identity.reshape(1, 1, 2, 2).expand(batch_size, 2, -1, -1)
        per_frame = torch.cat([metric_vectors, rotations, height, contact,
                               own_goal_proximity, blocker_proximity, other_goal_proximity,
                               aperture, identity], dim=-1)
        # Two ordered frames plus finite differences in stable metric units.
        history_features = per_frame.permute(0, 2, 1, 3).reshape(batch_size, 2, 62)
        object_step = (objects[:, 1] - objects[:, 0]) / 0.025
        relative_step = (relative_tcp[:, 1] - relative_tcp[:, 0]) / 0.025
        return torch.cat([history_features, object_step, relative_step], dim=-1)

    def encode(self, raw_history):
        batch_size = raw_history.shape[0]
        normalized = ((raw_history - self.obs_mean) / self.obs_scale).reshape(batch_size, 94)
        robot = self.robot_encoder(normalized)
        objects = self.object_encoder(self.object_features(raw_history))
        robot_context = robot.unsqueeze(1).expand(-1, 2, -1)
        messages = self.message_encoder(torch.cat([objects, objects.flip(1), robot_context], dim=-1))
        objects = self.message_norm(objects + messages)
        scene = torch.cat([robot, objects.reshape(batch_size, 192)], dim=-1)
        forecast = self.forecast_head(scene).reshape(batch_size, 3, 11)
        condition = torch.cat([normalized, scene, forecast.reshape(batch_size, 33)], dim=-1)
        return condition, forecast

    def forward(self, noisy_action, timestep, raw_history):
        condition, _ = self.encode(raw_history)
        return self.backbone(noisy_action, timestep, condition)

    def training_prediction(self, noisy_action, timestep, raw_history):
        condition, forecast = self.encode(raw_history)
        return self.backbone(noisy_action, timestep, condition), forecast

    def forecast_target(self, raw_history, future_obs):
        selected_future = future_obs.index_select(1, self.forecast_slots)
        selected_future = selected_future.index_select(2, self.forecast_indices)
        present = raw_history[:, -1].index_select(1, self.forecast_indices)
        translations = selected_future[..., :9] - present[:, None, :9]
        # Finger apertures are absolute; Cartesian labels are displacements.
        return torch.cat([translations, selected_future[..., 9:]], dim=-1) / self.forecast_scale


def build_model(spec):
    return OccupancyForecastPolicy(spec)


def compute_loss(model, batch, spec):
    predicted_noise, forecast = model.training_prediction(
        batch['noisy_action'], batch['timesteps'], batch['raw_obs'])
    diffusion_loss = epsilon_loss(predicted_noise, batch['noise'], batch['mask'])
    target = model.forecast_target(batch['raw_obs'], batch['future_obs'])
    future_mask = batch['future_mask'].index_select(1, model.forecast_slots)
    forecast_error = torch.nn.functional.smooth_l1_loss(
        forecast, target, reduction='none', beta=0.25)
    prior_loss = (forecast_error * future_mask).sum() / (future_mask.sum() * 11).clamp_min(1)
    loss = diffusion_loss + model.prior_weight * prior_loss
    return {'loss': loss, 'diffusion_loss': diffusion_loss, 'prior_loss': prior_loss}
