import torch
from appl.public import DiffusionBackbone, epsilon_loss


def mlp(in_dim, hidden_dim, out_dim, normalize=False):
    layers = [torch.nn.Linear(in_dim, hidden_dim), torch.nn.SiLU(),
              torch.nn.Linear(hidden_dim, out_dim)]
    if normalize:
        layers.append(torch.nn.LayerNorm(out_dim))
    layers.append(torch.nn.SiLU())
    return torch.nn.Sequential(*layers)


class RelationalAnticipationPolicy(torch.nn.Module):
    """One causal epsilon predictor with supervised relational anticipation."""

    def __init__(self, spec):
        super().__init__()
        normalizer = spec['normalizer']
        config = spec.get('candidate_config', {})
        self.prior_weight = float(config.get('forecast_loss_weight', 0.1))
        self.register_buffer('obs_mean', torch.tensor(normalizer['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(normalizer['std'], dtype=torch.float32))
        self.register_buffer('drawer_origin', torch.tensor([0.19, 0.0, 0.035], dtype=torch.float32))
        self.register_buffer('drawer_axis', torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32))
        # R has seven Cartesian vectors, one opening, and two finger positions.
        self.register_buffer('relation_scale', torch.tensor([0.25] * 21 + [0.30, 0.04, 0.04]))
        self.register_buffer('motion_scale', torch.tensor([0.025] * 21 + [0.025, 0.01, 0.01]))
        self.register_buffer('forecast_scale', torch.tensor([0.05] * 21 + [0.05, 0.02, 0.02]))
        self.register_buffer('forecast_slots', torch.tensor([4, 8, 15], dtype=torch.long))

        # Shared geometric processing, but distinct learned roles and ordered tokens.
        self.object_roles = torch.nn.Parameter(torch.randn(2, 8) * 0.02)
        self.object_encoder = mlp(50, 128, 128, normalize=True)
        self.global_encoder = mlp(94, 192, 128, normalize=True)
        self.fusion = mlp(384, 256, 128, normalize=True)
        self.forecaster = torch.nn.Sequential(
            torch.nn.Linear(176, 256), torch.nn.SiLU(),
            torch.nn.Linear(256, 72))
        torch.nn.init.normal_(self.forecaster[-1].weight, std=0.005)
        torch.nn.init.zeros_(self.forecaster[-1].bias)

        # 94 raw-normalized history + 24 relations + 24 changes + 128 latent
        # + 3 * 24 forecast coordinates. No forecast label enters this condition.
        self.denoiser = DiffusionBackbone(342, spec['training'])

    def drawer_center(self, state):
        return self.drawer_origin + state[..., 39:40] * self.drawer_axis

    def relations(self, state):
        """Continuous world-frame relations; no thresholded phase assignment."""
        tcp = state[..., 18:21]
        red = state[..., 25:28]
        blue = state[..., 32:35]
        center = self.drawer_center(state)
        return torch.cat((red - tcp, state[..., 41:44] - red, red - center,
                          blue - tcp, state[..., 44:47] - blue, blue - center,
                          tcp - center, state[..., 39:40], state[..., 7:9]), dim=-1)

    def rotation_columns(self, quaternion):
        # First two rotation-matrix columns; invariant to quaternion sign.
        q = quaternion / quaternion.square().sum(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
        w, x, y, z = q.unbind(dim=-1)
        return torch.stack((1 - 2 * (y * y + z * z),
                            2 * (x * y + w * z),
                            2 * (x * z - w * y),
                            2 * (x * y - w * z),
                            1 - 2 * (x * x + z * z),
                            2 * (y * z + w * x)), dim=-1)

    def object_features(self, state):
        positions = torch.stack((state[..., 25:28], state[..., 32:35]), dim=-2)
        goals = torch.stack((state[..., 41:44], state[..., 44:47]), dim=-2)
        quaternions = torch.stack((state[..., 28:32], state[..., 35:39]), dim=-2)
        tcp = state[..., 18:21].unsqueeze(-2)
        center = self.drawer_center(state).unsqueeze(-2)
        return torch.cat(((positions - tcp) / 0.25,
                          (goals - positions) / 0.25,
                          (positions - center) / 0.25,
                          positions[..., 2:3] / 0.25,
                          self.rotation_columns(quaternions)), dim=-1)

    def condition_and_forecast(self, raw_history):
        batch_size = raw_history.shape[0]
        normalized = ((raw_history - self.obs_mean) / self.obs_scale).reshape(batch_size, 94)
        objects = self.object_features(raw_history)
        previous, current = objects[:, 0], objects[:, 1]
        # Metre-valued components only: one-frame displacement scaled by 2.5 cm.
        change = (current[..., :10] - previous[..., :10]) * 10.0
        roles = self.object_roles.unsqueeze(0).expand(batch_size, -1, -1)
        object_input = torch.cat((previous, current, change, roles), dim=-1)
        object_tokens = self.object_encoder(object_input).reshape(batch_size, 256)
        global_token = self.global_encoder(normalized)
        latent = self.fusion(torch.cat((object_tokens, global_token), dim=-1))

        relation_history = self.relations(raw_history)
        relation_now = relation_history[:, 1] / self.relation_scale
        relation_change = (relation_history[:, 1] - relation_history[:, 0]) / self.motion_scale
        forecast_input = torch.cat((latent, relation_now, relation_change), dim=-1)
        forecast = self.forecaster(forecast_input).reshape(batch_size, 3, 24)
        condition = torch.cat((normalized, relation_now, relation_change,
                               latent, forecast.reshape(batch_size, 72)), dim=-1)
        return condition, forecast

    def predict_with_forecast(self, noisy_action, timestep, raw_history):
        condition, forecast = self.condition_and_forecast(raw_history)
        predicted_noise = self.denoiser(noisy_action, timestep, condition)
        return predicted_noise, forecast

    def forward(self, noisy_action, timestep, raw_history):
        predicted_noise, _ = self.predict_with_forecast(noisy_action, timestep, raw_history)
        return predicted_noise


def build_model(spec):
    return RelationalAnticipationPolicy(spec)


def compute_loss(model, batch, spec):
    predicted_noise, forecast = model.predict_with_forecast(
        batch['noisy_action'], batch['timesteps'], batch['raw_obs'])
    diffusion_loss = epsilon_loss(predicted_noise, batch['noise'], batch['mask'])
    # At observation t, future slot j is state t+j, after action t-1+j.
    # These are labels only. They never enter forward or its conditioning path.
    with torch.no_grad():
        future = batch['future_obs'].index_select(1, model.forecast_slots)
        target = (model.relations(future) -
                  model.relations(batch['raw_obs'][:, -1]).unsqueeze(1)) / model.forecast_scale
        valid = batch['future_mask'].index_select(1, model.forecast_slots).to(forecast.dtype)
    error = torch.nn.functional.smooth_l1_loss(forecast, target, reduction='none', beta=1.0)
    prior_loss = (error * valid).sum() / (valid.sum() * forecast.shape[-1]).clamp_min(1.0)
    loss = diffusion_loss + model.prior_weight * prior_loss
    return {'loss': loss, 'diffusion_loss': diffusion_loss, 'prior_loss': prior_loss}
