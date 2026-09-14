"""The three saved stick-push proposals; numerical observations only.

P1 exposes fixed-scale world relations, P2 uses a frozen horizontal goal frame,
and P3 jointly diffuses native actions and supervised future entity motions.
All clipping is delegated to round3.learning.Loaded, after world decoding.
"""
import numpy as np
import torch

from relative_dp.model import masked_epsilon_loss
from round2.learning import Policy
from .plugins import BaselinePlugin


WORLD_ORIGIN = np.array([0., .6, .1], dtype=np.float32)
POSITION_SLICES = (slice(0, 3), slice(4, 7), slice(11, 14),
                   slice(18, 21), slice(22, 25), slice(29, 32), slice(36, 39))
QUATERNION_SLICES = (slice(7, 11), slice(14, 18), slice(25, 29), slice(32, 36))


def goal_frame(raw):
    """Return current container origin and R columns e1,e2,world-z."""
    raw = np.asarray(raw, dtype=np.float32)
    origin = raw[..., 11:14].copy()
    xy = raw[..., 36:38] - origin[..., :2]
    length = np.linalg.norm(xy, axis=-1, keepdims=True)
    direction = xy / np.maximum(length, np.float32(1e-6))
    direction = np.where(length >= 1e-6, direction,
                         np.array([1., 0.], dtype=np.float32))
    rotation = np.zeros((*raw.shape[:-1], 3, 3), dtype=np.float32)
    rotation[..., 0, 0] = direction[..., 0]
    rotation[..., 1, 0] = direction[..., 1]
    rotation[..., 0, 1] = -direction[..., 1]
    rotation[..., 1, 1] = direction[..., 0]
    rotation[..., 2, 2] = 1.
    return origin, rotation


def _local(vector, rotation):
    return np.einsum('...i,...ij->...j', vector, rotation)


def _canonical_quaternion(value):
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    unit = value / np.maximum(norm, np.float32(1e-8))
    return np.where(unit[..., 3:4] < 0., -unit, unit)


def relation_features(raw, origin=None, rotation=None):
    """Encode raw39 as66 world features or71 current-frame features."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape[-1] != 39:
        raise ValueError('Stick-push requires the declared raw39 schema')
    encoded = raw.copy()
    hand, stick, container, goal = (raw[..., :3], raw[..., 4:7],
                                   raw[..., 11:14], raw[..., 36:39])
    deltas = (stick - hand, container - stick, goal - container,
              container - hand, goal - stick, goal - hand)
    motions = (hand - raw[..., 18:21], stick - raw[..., 22:25],
               container - raw[..., 29:32])
    framed = rotation is not None
    for section in POSITION_SLICES:
        vector = raw[..., section] - (origin if framed else WORLD_ORIGIN)
        encoded[..., section] = (_local(vector, rotation) if framed else vector) / .1
    for section in QUATERNION_SLICES:
        # These are WORLD xyzw orientations, even in P2; zeros remain absent.
        encoded[..., section] = _canonical_quaternion(raw[..., section])
    relations = [(_local(v, rotation) if framed else v) / .1 for v in deltas]
    differences = [(_local(v, rotation) if framed else v) / .01 for v in motions]
    values = [encoded, *relations, *differences]
    if framed:
        context = np.concatenate([(origin - WORLD_ORIGIN) / .1,
                                  rotation[..., :2, 0]], axis=-1)
        values.append(np.broadcast_to(context, (*raw.shape[:-1], 5)))
    return np.concatenate(values, axis=-1).astype(np.float32)


class JointMotionPolicy(Policy):
    """Joint13-channel DDIM; generated motion channels never control actions."""

    @torch.no_grad()
    def predict_action(self, obs_history, generator):
        scheduler = self.inference_scheduler
        scheduler.set_timesteps(self.config['inference_steps'], device=obs_history.device)
        sample = torch.randn((len(obs_history), self.config['prediction_horizon'], 13),
                             generator=generator, device=generator.device,
                             dtype=obs_history.dtype).to(obs_history.device)
        for timestep in scheduler.timesteps:
            if self._compiled_inference_net is None:
                epsilon = self(sample, timestep, obs_history)
            else:
                torch.compiler.cudagraph_mark_step_begin()
                epsilon = self._compiled_inference_net(
                    sample, timestep, global_cond=obs_history.flatten(start_dim=1))
            sample = scheduler.step(epsilon, timestep, sample, eta=0.,
                                    use_clipped_model_output=False, generator=generator,
                                    return_dict=True).prev_sample
        # Native world clipping happens only after action_decode in shared Loaded.
        return sample[..., :4]


class StickPlugin(BaselinePlugin):
    def __init__(self, task, schema, config):
        super().__init__(task, schema, config)
        self.candidate = config['candidate_id']
        if task != 'stick-push' or self.candidate not in ('P1', 'P2', 'P3'):
            raise ValueError('This module implements only stick-push P1/P2/P3')

    def passthrough(self, dimension):
        expected = 71 if self.candidate == 'P2' else 66
        if dimension != expected:
            raise ValueError('Unexpected stick feature dimension')
        return list(range(dimension))

    def observation(self, raw):
        if self.candidate == 'P2':
            origin, rotation = goal_frame(raw)
            return relation_features(raw, origin, rotation)
        return relation_features(raw)

    def observation_history(self, raw):
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape[-2:] != (2, 39):
            raise ValueError('Expected two whole raw39 observations')
        if self.candidate != 'P2':
            return relation_features(raw)
        # Singleton history axis broadcasts one latest anchor across BOTH rows.
        origin, rotation = goal_frame(raw[..., -1, :])
        return relation_features(raw, origin[..., None, :], rotation[..., None, :, :])

    def action_encode(self, actions, current_raw):
        encoded = np.asarray(actions, dtype=np.float32).copy()
        if encoded.shape[-1] != 4:
            raise ValueError('Action encoding requires four native channels')
        if self.candidate == 'P2':
            _, rotation = goal_frame(current_raw)
            encoded[..., :3] = _local(encoded[..., :3], rotation[..., None, :, :])
        return encoded

    def action_decode(self, actions, current_raw):
        decoded = np.asarray(actions, dtype=np.float32).copy()
        if decoded.shape[-1] != 4:
            raise ValueError('Joint sampler must discard generated motion channels')
        if self.candidate == 'P2':
            _, rotation = goal_frame(current_raw)
            decoded[..., :3] = np.einsum('...j,...ij->...i', decoded[..., :3],
                                        rotation[..., None, :, :])
        return decoded

    def supervision(self, episodes, window_index):
        if self.candidate != 'P3':
            return {}
        motion = np.zeros((len(window_index), 16, 9), dtype=np.float32)
        for i, (episode_index, t) in enumerate(window_index):
            episode = episodes[episode_index]
            obs = np.asarray(episode['obs'], dtype=np.float32)
            valid = min(16, len(episode['actions']) - t)
            if obs.shape != (len(episode['actions']) + 1, 39) or valid <= 0:
                raise ValueError('Invalid current-D_N episode/window')
            for j, section in enumerate((slice(0, 3), slice(4, 7), slice(11, 14))):
                motion[i, :valid, 3*j:3*j+3] = (
                    obs[t+1:t+valid+1, section] - obs[t, section]) / .1
        return {'future_motion': motion}

    def training_actions(self, actions, targets):
        if self.candidate != 'P3':
            return actions
        motion = np.asarray(targets['future_motion'], dtype=np.float32)
        if motion.shape != (*actions.shape[:-1], 9):
            raise ValueError('Future motion targets must align to action windows')
        return np.concatenate([actions, motion], axis=-1).astype(np.float32)

    def model_config(self, cfg):
        return {'action_dim': 13 if self.candidate == 'P3' else 4}

    def build_policy(self, cfg):
        policy_class = JointMotionPolicy if self.candidate == 'P3' else Policy
        return policy_class(cfg['obs_dim'], cfg)

    def extra_loss(self, output, batch, step):
        if self.candidate != 'P3':
            return None, {}
        epsilon = output['epsilon'] if isinstance(output, dict) else output
        motion_loss = masked_epsilon_loss(epsilon[..., 4:13], batch['noise'][..., 4:13],
                                         batch['mask'])
        return .15 * motion_loss, {'future_motion_epsilon_mse': motion_loss}

    def diagnostics(self, raw_history, policy):
        result = dict(route='flat', execution_horizon=4, candidate=self.candidate,
                      policy_inputs='two_common_raw39_numerical_observations')
        if self.candidate == 'P2':
            origin, rotation = goal_frame(np.asarray(raw_history)[-1])
            result.update(frame_origin_world_m=origin.tolist(),
                          frame_forward_world=rotation[:, 0].tolist(),
                          frame_frozen_for_chunk=True)
        if self.candidate == 'P3':
            result['generated_motion_channels_discarded'] = 9
        return result
