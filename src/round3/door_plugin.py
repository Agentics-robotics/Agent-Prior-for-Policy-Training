"""Frozen D2-designed door priors: cabinet coordinates and joint future geometry.

Implementation of round3/design_records/door/proposal.json version 1. Future
observations appear exclusively in supervision(), never runtime conditioning.
"""
import numpy as np
import torch

from relative_dp.model import masked_epsilon_loss
from round2.learning import Policy
from .plugins import BaselinePlugin


def _yaw(raw):
    """Return normalized cabinet (sin, cos), with identity for invalid zero yaw."""
    value = np.asarray(raw, np.float32)[..., 39:41]
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return np.where(norm > 1e-8, value / np.maximum(norm, 1e-8),
                    np.array([0., 1.], np.float32))


def _rotate(vector, yaw, inverse=False):
    """Rotate vectors in coherent cabinet axes; broadcast over chunk/history."""
    vector = np.asarray(vector, np.float32)
    yaw = np.asarray(yaw, np.float32)
    while yaw.ndim < vector.ndim:
        yaw = np.expand_dims(yaw, axis=-2)
    sine, cosine = yaw[..., 0], yaw[..., 1]
    if inverse:
        sine = -sine
    x, y, z = vector[..., 0], vector[..., 1], vector[..., 2]
    return np.stack([cosine*x + sine*y, -sine*x + cosine*y, z], axis=-1).astype(np.float32)


def _door_direction(raw):
    """World xy projection of quaternion-rotated local +y, q ordered xyzw."""
    q = np.asarray(raw, np.float32)[..., 7:11]
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    x, y, z, w = [q[..., i] for i in range(4)]
    direction = np.stack([2*(x*y-w*z), 1-2*(x*x+z*z)], axis=-1)
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    return np.where(norm > 1e-8, direction / np.maximum(norm, 1e-8),
                    np.array([0., 1.], np.float32)).astype(np.float32)


def _direction3(raw):
    direction = _door_direction(raw)
    return np.concatenate([direction, np.zeros_like(direction[..., :1])], axis=-1)


class DoorPolicy(Policy):
    """Same unbounded DDIM sampling, with the declared joint target dimension."""

    @torch.no_grad()
    def predict_action(self, obs_history, generator):
        scheduler = self.inference_scheduler
        scheduler.set_timesteps(self.config['inference_steps'], device=obs_history.device)
        sample = torch.randn(
            (len(obs_history), self.config['prediction_horizon'], self.config['action_dim']),
            generator=generator, device=generator.device, dtype=obs_history.dtype,
        ).to(obs_history.device)
        for timestep in scheduler.timesteps:
            if self._compiled_inference_net is None:
                epsilon = self(sample, timestep, obs_history)
            else:
                torch.compiler.cudagraph_mark_step_begin()
                epsilon = self._compiled_inference_net(
                    sample, timestep, global_cond=obs_history.flatten(start_dim=1))
            sample = scheduler.step(
                epsilon, timestep, sample, eta=0., use_clipped_model_output=False,
                generator=generator, return_dict=True,
            ).prev_sample
        return sample


class DoorPlugin(BaselinePlugin):
    """P1 cabinet representation; P2 joint dynamics; P3 their combination."""

    def __init__(self, task, schema, config):
        super().__init__(task, schema, config)
        if task != 'door':
            raise ValueError('DoorPlugin only supports the declared door schema')
        self.candidate = config['candidate_id']
        if self.candidate not in ('P1', 'P2', 'P3'):
            raise ValueError('Unknown frozen door candidate')
        self.cabinet = self.candidate in ('P1', 'P3')
        self.joint = self.candidate in ('P2', 'P3')

    def passthrough(self, dimension):
        # Fixed scales are part of the original design; no fitted normalization.
        return list(range(dimension))

    def _observation(self, raw, yaw):
        raw = np.asarray(raw, np.float32)
        if raw.shape[-1] != 41:
            raise ValueError('Expected common raw41 door observation')
        if not self.cabinet:
            return np.concatenate([raw, _door_direction(raw)], axis=-1)
        relations = [
            _rotate(raw[..., :3] - raw[..., 4:7], yaw) / .1,
            _rotate(raw[..., 36:39] - raw[..., 4:7], yaw) / .5,
            _rotate(raw[..., :3] - raw[..., 18:21], yaw) / .01,
            _rotate(raw[..., 4:7] - raw[..., 22:25], yaw) / .01,
            _rotate(_direction3(raw), yaw)[..., :2],
        ]
        return np.concatenate([raw, *relations], axis=-1).astype(np.float32)

    def observation(self, raw):
        return self._observation(raw, _yaw(raw))

    def observation_history(self, raw_history):
        raw_history = np.asarray(raw_history, np.float32)
        if raw_history.shape[-2:] != (2, 41):
            raise ValueError('Expected two whole common raw41 observations')
        return self._observation(raw_history, _yaw(raw_history[..., -1, :]))

    def action_encode(self, actions, current_raw):
        encoded = np.asarray(actions, np.float32).copy()
        if encoded.shape[-1] != 4:
            raise ValueError('Expected four native action channels')
        if self.cabinet:
            encoded[..., :3] = _rotate(encoded[..., :3], _yaw(current_raw))
        return encoded

    def action_decode(self, actions, current_raw):
        # Auxiliary futures are jointly sampled but never used as a controller.
        decoded = np.asarray(actions, np.float32)[..., :4].copy()
        if self.cabinet:
            decoded[..., :3] = _rotate(decoded[..., :3], _yaw(current_raw), inverse=True)
        return decoded

    def supervision(self, episodes, window_index):
        if not self.joint:
            return {}
        targets = np.zeros((len(window_index), 16, 8), np.float32)
        for row, (episode_index, start) in enumerate(window_index):
            episode = episodes[episode_index]
            raw = np.asarray(episode['obs'], np.float32)
            valid = min(16, len(episode['actions']) - start)
            current = raw[start]
            future = raw[start+1:start+valid+1]
            displacement = future[:, 4:7] - current[4:7]
            gap = future[:, :3] - future[:, 4:7]
            direction = _direction3(future)
            if self.cabinet:
                yaw = _yaw(current)
                displacement = _rotate(displacement, yaw)
                gap = _rotate(gap, yaw)
                direction = _rotate(direction, yaw)
            targets[row, :valid] = np.concatenate(
                [displacement/.1, gap/.1, direction[:, :2]], axis=-1)
        return {'future_geometry': targets}

    def training_actions(self, encoded_actions, targets):
        encoded_actions = np.asarray(encoded_actions, np.float32)
        if not self.joint:
            return encoded_actions
        future = np.asarray(targets['future_geometry'], np.float32)
        if future.shape != (*encoded_actions.shape[:2], 8):
            raise ValueError('Future geometry and action window alignment differ')
        return np.concatenate([encoded_actions, future], axis=-1)

    def model_config(self, cfg):
        return {'action_dim': 12 if self.joint else 4}

    def build_policy(self, cfg):
        expected_dim = 12 if self.joint else 4
        if cfg['action_dim'] != expected_dim:
            raise ValueError('Incorrect frozen candidate diffusion dimension')
        return DoorPolicy(cfg['obs_dim'], cfg)

    def extra_loss(self, output, batch, step):
        if not self.joint:
            return None, {}
        epsilon = output['epsilon'] if isinstance(output, dict) else output
        auxiliary = masked_epsilon_loss(epsilon[..., 4:12], batch['noise'][..., 4:12], batch['mask'])
        return .2*auxiliary, {'future_geometry_epsilon_mse': auxiliary}

    def diagnostics(self, raw_history, policy):
        return {'route': 'flat', 'execution_horizon': 4,
                'candidate_id': self.candidate,
                'action_frame': 'cabinet' if self.cabinet else 'world',
                'joint_future_channels_discarded': 8 if self.joint else 0}
