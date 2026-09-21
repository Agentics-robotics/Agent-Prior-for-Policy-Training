import torch
from torch import nn
from appl.public import DiffusionBackbone, epsilon_loss


def rotation_matrix(quaternion):
    """Convert wxyz observations to sign-invariant rotation features."""
    q = quaternion / quaternion.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)
    ), dim=-1).reshape(q.shape[:-1] + (3, 3))


def objects_and_goals(raw):
    positions = torch.stack((raw[..., 25:28], raw[..., 32:35]), dim=-2)
    goals = torch.stack((raw[..., 41:44], raw[..., 44:47]), dim=-2)
    return positions, goals


def metric_relations(raw):
    """For each object: TCP minus object, followed by object minus its goal."""
    positions, goals = objects_and_goals(raw)
    tcp = raw[..., 18:21].unsqueeze(-2)
    return torch.cat((tcp - positions, positions - goals), dim=-1)


class RelationalForecastPolicy(nn.Module):
    """One epsilon network with a causal, object-factorized predictive condition."""
    def __init__(self, spec):
        super().__init__()
        normalizer = spec['normalizer']
        self.register_buffer('obs_mean', torch.tensor(normalizer['mean'], dtype=torch.float32))
        self.register_buffer('obs_scale', torch.tensor(normalizer['std'], dtype=torch.float32))
        self.register_buffer('object_identity', torch.eye(2, dtype=torch.float32))
        self.register_buffer('forecast_slots', torch.tensor([1, 5, 9, 15], dtype=torch.long))
        self.prior_weight = 0.1
        self.relation_forecast_scale = 0.1
        self.finger_forecast_scale = 0.04

        # 39 features per observation, eleven motion features and two identities.
        self.object_encoder = nn.Sequential(
            nn.Linear(91, 192), nn.SiLU(),
            nn.Linear(192, 128), nn.LayerNorm(128), nn.SiLU()
        )
        # Pooled shared semantics plus ordered original observations allow the
        # learned forecast to account for both objects, without an object selector.
        self.scene_encoder = nn.Sequential(
            nn.Linear(94 + 128, 192), nn.SiLU(),
            nn.Linear(192, 128), nn.LayerNorm(128), nn.SiLU()
        )
        self.relation_forecaster = nn.Sequential(
            nn.Linear(256, 192), nn.SiLU(), nn.Linear(192, 24)
        )
        self.finger_forecaster = nn.Sequential(
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 4)
        )
        # Original history (94), two tokens (256), relational forecasts (48)
        # and finger forecasts (4). The absolute action interface is unchanged.
        self.backbone = DiffusionBackbone(402, spec['training'])

    def object_features(self, raw):
        positions, goals = objects_and_goals(raw)
        tcp = raw[..., 18:21].unsqueeze(2).expand_as(positions)
        object_quaternions = torch.stack((raw[..., 28:32], raw[..., 35:39]), dim=2)
        object_rotations = rotation_matrix(object_quaternions)
        tcp_rotations = rotation_matrix(raw[..., 21:25]).unsqueeze(2)
        relative_rotations = tcp_rotations.transpose(-1, -2) @ object_rotations
        fingers = (raw[..., 7:9] / 0.04).unsqueeze(2).expand(-1, -1, 2, -1)

        # Smooth state representation except for abs at geometric boundaries;
        # these signed margins are features, never a success test or action gate.
        projected_half_width = 0.02 * object_rotations.abs().sum(dim=-1)[..., :2]
        xy_margins = (0.06 - projected_half_width - (positions - goals).abs()[..., :2]) / 0.06
        tcp_distance = (tcp - positions).square().sum(-1, keepdim=True).add(1e-12).sqrt() / 0.25
        goal_distance = (goals - positions).square().sum(-1, keepdim=True).add(1e-12).sqrt() / 0.25
        per_frame = torch.cat((
            positions / 0.5,
            goals / 0.5,
            (tcp - positions) / 0.25,
            (goals - positions) / 0.25,
            (goals - tcp) / 0.25,
            object_rotations.flatten(-2),
            relative_rotations.flatten(-2),
            fingers, xy_margins, tcp_distance, goal_distance
        ), dim=-1)
        frame_pair = per_frame.permute(0, 2, 1, 3).flatten(2)
        object_motion = positions[:, 1] - positions[:, 0]
        tcp_motion = tcp[:, 1] - tcp[:, 0]
        finger_motion = ((raw[:, 1, 7:9] - raw[:, 0, 7:9]) / 0.02)
        finger_motion = finger_motion.unsqueeze(1).expand(-1, 2, -1)
        identity = self.object_identity.unsqueeze(0).expand(raw.shape[0], -1, -1)
        return torch.cat((frame_pair, object_motion / 0.025,
                          tcp_motion / 0.025, (object_motion - tcp_motion) / 0.025,
                          finger_motion, identity), dim=-1)

    def encode_history(self, raw_history):
        normalized = ((raw_history - self.obs_mean) / self.obs_scale).flatten(1)
        tokens = self.object_encoder(self.object_features(raw_history))
        scene = self.scene_encoder(torch.cat((normalized, tokens.mean(dim=1)), dim=-1))
        forecast_inputs = torch.cat((tokens, scene.unsqueeze(1).expand(-1, 2, -1)), dim=-1)
        relation_delta = self.relation_forecaster(forecast_inputs).reshape(-1, 2, 4, 6)
        finger_delta = self.finger_forecaster(scene).unsqueeze(-1)
        condition = torch.cat((normalized, tokens.flatten(1),
                               relation_delta.flatten(1), finger_delta.flatten(1)), dim=-1)
        return condition, relation_delta, finger_delta

    def forward(self, noisy_action, timestep, raw_history, return_aux=False):
        condition, relation_delta, finger_delta = self.encode_history(raw_history)
        epsilon = self.backbone(noisy_action, timestep, condition)
        if return_aux:
            return epsilon, relation_delta, finger_delta
        return epsilon


def build_model(spec):
    return RelationalForecastPolicy(spec)


def compute_loss(model, batch, spec):
    predicted_noise, relation_delta, finger_delta = model(
        batch['noisy_action'], batch['timesteps'], batch['raw_obs'], return_aux=True
    )
    diffusion_loss = epsilon_loss(predicted_noise, batch['noise'], batch['mask'])

    # Future observations only provide training labels. They never enter forward.
    # With action slots t-1,...,t+14, slots 1,5,9,15 label states t+1,t+5,t+9,t+15.
    future = batch['future_obs'].index_select(1, model.forecast_slots)
    current = batch['raw_obs'][:, -1]
    target_relation = ((metric_relations(future) - metric_relations(current).unsqueeze(1))
                       / model.relation_forecast_scale).permute(0, 2, 1, 3)
    target_finger = ((future[..., 7:9].mean(-1, keepdim=True)
                      - current[..., 7:9].mean(-1, keepdim=True).unsqueeze(1))
                     / model.finger_forecast_scale)
    valid = (batch['future_mask'] * batch['mask']).index_select(1, model.forecast_slots)
    relation_error = torch.nn.functional.smooth_l1_loss(
        relation_delta, target_relation, reduction='none', beta=0.5
    )
    finger_error = torch.nn.functional.smooth_l1_loss(
        finger_delta, target_finger, reduction='none', beta=0.5
    )
    relation_loss = (relation_error * valid.unsqueeze(1)).sum() / (valid.sum() * 12).clamp_min(1)
    finger_loss = (finger_error * valid).sum() / valid.sum().clamp_min(1)
    # No reconstruction of x0 is used: the auxiliary gradient has no inverse-SNR
    # amplification, and the frozen epsilon/DDPM objective remains unchanged.
    prior_loss = model.prior_weight * (relation_loss + 0.5 * finger_loss)
    return {'loss': diffusion_loss + prior_loss,
            'diffusion_loss': diffusion_loss, 'prior_loss': prior_loss}
