"""Assembly candidates, implementing frozen proposal v1 plus its sole amendment.

Only the common numerical observations enter the runtime paths. Retrospective
milestones and future observations are used exclusively by ``supervision``.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from round2.learning import Policy
from .plugins import BaselinePlugin


_POSITION_SLICES = ((0, 3), (4, 7), (18, 21), (22, 25), (36, 39), (39, 42))
_RETAINED = np.r_[0:11, 18:29, 36:42]
_ENCODED_POSITION_SLICES = ((0, 3), (4, 7), (11, 14), (15, 18), (22, 25), (25, 28))


def _raw_array(raw):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape[-1] != 42 or not np.isfinite(raw).all():
        raise ValueError('Assembly requires finite common 42D observations')
    return raw


def _canonical_quaternion(value):
    q = np.asarray(value, dtype=np.float32).copy()
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q /= np.maximum(norm, np.float32(1e-8))
    q = np.where(norm < 1e-8, np.array([1, 0, 0, 0], np.float32), q)
    first_index = np.argmax(q != 0, axis=-1)
    first = np.take_along_axis(q, first_index[..., None], axis=-1)
    return q * np.where(first < 0, np.float32(-1), np.float32(1))


def transport_basis(current_raw):
    """World-column right-handed basis; +z stays vertical in both branches."""
    raw = _raw_array(current_raw)
    delta = raw[..., 36:38] - raw[..., 39:41]
    rho = np.linalg.norm(delta, axis=-1)
    horizontal = delta / np.maximum(rho[..., None], np.float32(1e-8))
    horizontal = np.where((rho >= .020)[..., None], horizontal,
                          np.array([1, 0], dtype=np.float32))
    first = np.concatenate([horizontal, np.zeros_like(rho[..., None])], axis=-1)
    vertical = np.zeros_like(first)
    vertical[..., 2] = 1
    second = np.cross(vertical, first)
    return np.stack([first, second, vertical], axis=-1).astype(np.float32)


def _features(raw, basis=None, origin=None):
    raw = _raw_array(raw)
    result = raw[..., _RETAINED].copy()
    for source, target in zip(_POSITION_SLICES, _ENCODED_POSITION_SLICES, strict=True):
        value = raw[..., source[0]:source[1]]
        if basis is not None:
            value = np.einsum('...i,...ij->...j', value - origin, basis)
        result[..., target[0]:target[1]] = value / np.float32(.10)
    result[..., 7:11] = _canonical_quaternion(raw[..., 7:11])
    result[..., 18:22] = _canonical_quaternion(raw[..., 25:29])
    hand, handle = raw[..., :3], raw[..., 4:7]
    goal, ring = raw[..., 36:39], raw[..., 39:42]
    vectors = [handle-hand, ring-hand, goal-ring, handle-ring]
    velocities = [(hand-raw[..., 18:21])/np.float32(.0125),
                  (handle-raw[..., 22:25])/np.float32(.0125)]
    if basis is not None:
        vectors = [np.einsum('...i,...ij->...j', value, basis) for value in vectors]
        velocities = [np.einsum('...i,...ij->...j', value, basis) for value in velocities]
    rho = np.linalg.norm((ring-goal)[..., :2], axis=-1, keepdims=True)
    height = (ring-goal)[..., 2:3]
    result = np.concatenate([result, *[value/np.float32(.10) for value in vectors],
                             rho/np.float32(.10), height/np.float32(.10),
                             *velocities], axis=-1)
    if basis is not None:
        first_xy = np.broadcast_to(basis[..., :2, 0], raw.shape[:-1] + (2,))
        result = np.concatenate([result, hand/np.float32(.10), goal/np.float32(.10),
                                 ring/np.float32(.10), first_xy], axis=-1)
    expected = 59 if basis is not None else 48
    if result.shape[-1] != expected:
        raise AssertionError('Frozen assembly feature dimension mismatch')
    return result.astype(np.float32)


class EventConditionedPolicy(Policy):
    """One diffusion network, soft event conditioning and a training-only head."""

    def __init__(self, obs_dim, config):
        if obs_dim != 48 or config['n_obs_steps'] != 2:
            raise ValueError('P3 requires two 48D relational observations')
        # 2 * 66 = 132 global conditioning dimensions; no extra U-Net is built.
        super().__init__(66, config)
        self.obs_dim = obs_dim
        self.encoder = nn.Sequential(nn.Linear(96, 128), nn.SiLU(),
                                     nn.Linear(128, 128), nn.SiLU())
        self.event_head = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 4))
        self.future_head = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 12))

    def condition(self, obs_history):
        encoded = self.encoder(obs_history.flatten(start_dim=1))
        logits = self.event_head(encoded)
        probabilities = logits.softmax(dim=-1)
        return torch.cat([encoded, probabilities], dim=-1), encoded, logits

    def forward(self, noisy_actions, timesteps, obs_history):
        condition, encoded, logits = self.condition(obs_history)
        return dict(epsilon=self.net(noisy_actions, timesteps, global_cond=condition),
                    event_logits=logits,
                    future_relations=self.future_head(encoded).reshape(-1, 2, 6))

    def enable_inference_compilation(self):
        # Compile the denoiser only. predict_action explicitly supplies learned
        # 132D conditioning, so the base Policy's raw-flatten bypass cannot occur.
        super().enable_inference_compilation()

    @torch.no_grad()
    def predict_action(self, obs_history, generator):
        scheduler = self.inference_scheduler
        scheduler.set_timesteps(self.config['inference_steps'], device=obs_history.device)
        condition, _, _ = self.condition(obs_history)
        sample = torch.randn((len(obs_history), self.config['prediction_horizon'], 4),
                             generator=generator, device=generator.device,
                             dtype=obs_history.dtype).to(obs_history.device)
        # The condition is computed once and remains fixed for this whole chunk.
        for timestep in scheduler.timesteps:
            if self._compiled_inference_net is None:
                epsilon = self.net(sample, timestep, global_cond=condition)
            else:
                torch.compiler.cudagraph_mark_step_begin()
                epsilon = self._compiled_inference_net(sample, timestep, global_cond=condition)
            sample = scheduler.step(epsilon, timestep, sample, eta=0.,
                                    use_clipped_model_output=False, generator=generator,
                                    return_dict=True).prev_sample
        return sample


class AssemblyPlugin(BaselinePlugin):
    def __init__(self, task, schema, config):
        super().__init__(task, schema, config)
        self.candidate_id = config['candidate_id']
        if task != 'assembly' or self.candidate_id not in ('P1', 'P2', 'P3'):
            raise ValueError('AssemblyPlugin only implements assembly P1/P2/P3')
        if schema['raw_dim'] != 42:
            raise ValueError('Frozen common assembly schema is 42D')

    def observation(self, raw):
        raw = _raw_array(raw)
        if self.candidate_id == 'P2':
            return _features(raw, transport_basis(raw), raw[..., 39:42])
        return _features(raw)

    def observation_history(self, raw_history):
        raw = _raw_array(raw_history)
        if raw.ndim < 2 or raw.shape[-2] != 2:
            raise ValueError('Assembly requires two chronological whole observations')
        if self.candidate_id != 'P2':
            return _features(raw)
        anchor = raw[..., -1, :]
        basis = transport_basis(anchor)[..., None, :, :]
        origin = anchor[..., None, 39:42]
        return _features(raw, basis, origin)

    def passthrough(self, dimension):
        # Frozen proposal uses fixed scales and zero fitted mean/unit fitted std.
        expected = 59 if self.candidate_id == 'P2' else 48
        if dimension != expected:
            raise ValueError('Assembly normalization dimension mismatch')
        return list(range(dimension))

    def action_encode(self, actions, current_raw):
        result = np.asarray(actions, np.float32).copy()
        if self.candidate_id == 'P2':
            result[..., :3] = np.einsum('...hi,...ij->...hj', result[..., :3],
                                        transport_basis(current_raw))
        return result

    def action_decode(self, actions, current_raw):
        result = np.asarray(actions, np.float32).copy()
        if self.candidate_id == 'P2':
            result[..., :3] = np.einsum('...hi,...ji->...hj', result[..., :3],
                                        transport_basis(current_raw))
        # Shared Loaded.actions applies the sole world xyz/gripper cube clipping.
        return result

    def build_policy(self, cfg):
        if self.candidate_id == 'P3':
            return EventConditionedPolicy(cfg['obs_dim'], cfg)
        return Policy(cfg['obs_dim'], cfg)

    @staticmethod
    def milestone_indices(observations):
        raw = _raw_array(observations)
        final = len(raw)-1

        def event(mask, start):
            return next((i for i in range(start, final-1) if mask[i:i+3].all()), final+1)

        close = event(raw[:, 3] <= .85, 0)
        lifted = event(raw[:, 41] >= raw[:, 38]+.050, close)
        radius = np.linalg.norm(raw[:, 39:41]-raw[:, 36:38], axis=-1)
        aligned = event((radius <= .020) & (raw[:, 41] >= raw[:, 38]+.030), lifted)
        return close, lifted, aligned

    def supervision(self, episodes, window_index):
        if self.candidate_id != 'P3':
            return {}
        milestones = [self.milestone_indices(episode['obs']) for episode in episodes]
        count = len(window_index)
        labels = np.empty(count, np.int64)
        future = np.zeros((count, 2, 6), np.float32)
        eligible = np.zeros((count, 2), np.float32)
        for index, (episode_index, time) in enumerate(window_index):
            labels[index] = sum(time >= boundary for boundary in milestones[episode_index])
            obs = episodes[episode_index]['obs']
            for j, horizon in enumerate((4, 16)):
                target_index = time+horizon
                if target_index < len(obs):
                    target = obs[target_index]
                    future[index, j] = np.r_[target[39:42]-target[36:39],
                                              target[4:7]-target[0:3]]/np.float32(.10)
                    eligible[index, j] = 1
        return dict(event_class=labels, future_relations=future, future_valid=eligible)

    def extra_loss(self, output, batch, step):
        if self.candidate_id != 'P3':
            return None, {}
        targets = batch['targets']
        event_loss = F.cross_entropy(output['event_logits'], targets['event_class'])
        valid = targets['future_valid'][..., None]
        component_loss = F.smooth_l1_loss(output['future_relations'],
                                        targets['future_relations'], beta=1., reduction='none')
        future_loss = (component_loss*valid).sum()/(valid.sum()*6).clamp_min(1)
        ramp = min(1., (step+1)/2000.)
        total = ramp*(.10*event_loss+.05*future_loss)
        return total, dict(event_cross_entropy=event_loss, future_relation_smooth_l1=future_loss,
                           auxiliary_ramp=ramp)

    @torch.no_grad()
    def diagnostics(self, raw_history, policy):
        result = dict(route='flat', execution_horizon=4, candidate=self.candidate_id)
        if self.candidate_id == 'P2':
            raw = _raw_array(raw_history)[-1]
            result.update(coordinate_basis=transport_basis(raw).tolist(),
                          basis_origin_world_m=raw[39:42].tolist())
        if self.candidate_id == 'P3':
            device = next(policy.parameters()).device
            history = torch.as_tensor(self.observation_history(raw_history), device=device)[None]
            _, _, logits = policy.condition(history)
            result.update(route='learned_soft_events',
                          event_probabilities=logits.softmax(dim=-1)[0].cpu().tolist())
        return result
