"""Drawer P1/P2/P3, implemented after the isolated version-1 proposal freeze.

All auxiliary future fields are diffusion channels. They are discarded before
execution; there is no geometric action controller or runtime outcome oracle.
"""

import numpy as np
import torch

from relative_dp.model import masked_epsilon_loss
from round2.learning import Policy
from .plugins import BaselinePlugin


def _yaw(raw):
    """Normalized sine/cosine of the declared cabinet yaw."""
    pair = np.asarray(raw, dtype=np.float32)[..., 39:41]
    norm = np.linalg.norm(pair, axis=-1, keepdims=True)
    if not np.isfinite(pair).all() or np.any(norm < 1e-6):
        raise ValueError('Drawer cabinet yaw must be finite and nonzero')
    pair = pair / norm
    return pair[..., 0], pair[..., 1]


def _local(vector, anchor):
    s, c = _yaw(anchor)
    x, y, z = np.moveaxis(np.asarray(vector, dtype=np.float32), -1, 0)
    return np.stack((c * x + s * y, -s * x + c * y, z), axis=-1)


def _world(vector, anchor):
    s, c = _yaw(anchor)
    x, y, z = np.moveaxis(np.asarray(vector, dtype=np.float32), -1, 0)
    return np.stack((c * x - s * y, s * x + c * y, z), axis=-1)


class DrawerPolicy(Policy):
    """Unclipped DDIM with the proposal's complete joint channel dimension."""

    @torch.no_grad()
    def predict_action(self, obs_history, generator):
        scheduler = self.inference_scheduler
        scheduler.set_timesteps(self.config['inference_steps'], device=obs_history.device)
        sample = torch.randn(
            (len(obs_history), self.config['prediction_horizon'], self.config['action_dim']),
            generator=generator, device=generator.device,
            dtype=obs_history.dtype).to(obs_history.device)
        for timestep in scheduler.timesteps:
            if self._compiled_inference_net is None:
                epsilon = self(sample, timestep, obs_history)
            else:
                torch.compiler.cudagraph_mark_step_begin()
                epsilon = self._compiled_inference_net(
                    sample, timestep, global_cond=obs_history.flatten(start_dim=1))
            sample = scheduler.step(
                epsilon, timestep, sample, eta=0., use_clipped_model_output=False,
                generator=generator, return_dict=True).prev_sample
        return sample


class DrawerPlugin(BaselinePlugin):
    HORIZON = 16
    DIMENSIONS = {'P1': (25, 4), 'P2': (53, 5), 'P3': (25, 8)}

    def __init__(self, task, schema, config):
        super().__init__(task, schema, config)
        self.candidate_id = config['candidate_id']
        if task != 'drawer' or self.candidate_id not in self.DIMENSIONS:
            raise ValueError('DrawerPlugin requires drawer P1/P2/P3')
        self.observation_dim, self.action_dim = self.DIMENSIONS[self.candidate_id]

    def _local_observation(self, raw, anchor):
        raw = np.asarray(raw, dtype=np.float32)
        anchor = np.asarray(anchor, dtype=np.float32)
        origin = anchor[..., 4:7]
        s, c = _yaw(anchor)
        yaw = np.broadcast_to(np.stack((s, c), axis=-1), raw.shape[:-1] + (2,))
        return np.concatenate((
            _local(raw[..., :3] - origin, anchor) / .2,
            _local(raw[..., 4:7] - origin, anchor) / .16,
            _local(raw[..., 36:39] - origin, anchor) / .16,
            _local(raw[..., :3] - raw[..., 18:21], anchor) / .01,
            _local(raw[..., 4:7] - raw[..., 22:25], anchor) / .01,
            raw[..., :3], raw[..., 4:7], raw[..., 3:4], raw[..., 21:22], yaw,
        ), axis=-1).astype(np.float32)

    def observation(self, raw):
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape[-1] != 41:
            raise ValueError('Expected common drawer raw dimension41')
        if self.candidate_id != 'P2':
            return self._local_observation(raw, raw)
        return np.concatenate((
            raw, (raw[..., :3] - raw[..., 4:7]) / .2,
            (raw[..., 36:39] - raw[..., 4:7]) / .16,
            (raw[..., :3] - raw[..., 18:21]) / .01,
            (raw[..., 4:7] - raw[..., 22:25]) / .01,
        ), axis=-1).astype(np.float32)

    def observation_history(self, raw_history):
        raw = np.asarray(raw_history, dtype=np.float32)
        if raw.shape[-2:] != (2, 41):
            raise ValueError('Expected two whole drawer raw observations')
        if self.candidate_id == 'P2':
            return self.observation(raw)
        return self._local_observation(raw, raw[..., -1:, :])

    def passthrough(self, dimension):
        if dimension != self.observation_dim:
            raise ValueError('Unexpected encoded drawer dimension')
        return list(range(dimension))

    def action_encode(self, actions, current_raw):
        result = np.asarray(actions, dtype=np.float32).copy()
        if result.shape[-1] != 4:
            raise ValueError('Native training actions must have4 coordinates')
        if self.candidate_id != 'P2':
            anchor = np.asarray(current_raw, dtype=np.float32)[..., None, :]
            result[..., :3] = _local(result[..., :3], anchor)
        return result

    def action_decode(self, actions, current_raw):
        # The shared Loaded.actions method performs native-world cube clipping.
        result = np.asarray(actions, dtype=np.float32)[..., :4].copy()
        if self.candidate_id != 'P2':
            anchor = np.asarray(current_raw, dtype=np.float32)[..., None, :]
            result[..., :3] = _world(result[..., :3], anchor)
        return result

    def supervision(self, episodes, window_index):
        if self.candidate_id == 'P1':
            return {}
        future = np.zeros((len(window_index), self.HORIZON, self.action_dim - 4),
                          dtype=np.float32)
        for row, (episode_index, t) in enumerate(window_index):
            episode = episodes[episode_index]
            obs = np.asarray(episode['obs'], dtype=np.float32)
            valid = min(self.HORIZON, len(episode['actions']) - t)
            if not 0 < valid <= self.HORIZON:
                raise ValueError('Supervision window must begin at a valid action')
            anchor = obs[t]
            next_obs = obs[t + 1:t + valid + 1]
            remaining = _local(anchor[36:39] - next_obs[:, 4:7], anchor)[:, 1:2] / .16
            if self.candidate_id == 'P2':
                future[row, :valid] = remaining
            else:
                gap = _local(next_obs[:, :3] - next_obs[:, 4:7], anchor) / .2
                future[row, :valid] = np.concatenate((gap, remaining), axis=-1)
        return {'future_relations': future}

    def training_actions(self, encoded_actions, targets):
        if self.candidate_id == 'P1':
            return np.asarray(encoded_actions, dtype=np.float32)
        return np.concatenate((encoded_actions, targets['future_relations']), axis=-1).astype(np.float32)

    def model_config(self, cfg):
        return {'action_dim': self.action_dim}

    def build_policy(self, cfg):
        if cfg['obs_dim'] != self.observation_dim or cfg['action_dim'] != self.action_dim:
            raise ValueError('Policy dimensions must match the frozen drawer proposal')
        return DrawerPolicy(cfg['obs_dim'], cfg)

    def extra_loss(self, output, batch, step):
        if self.candidate_id == 'P1':
            return None, {}
        epsilon = output['epsilon'] if isinstance(output, dict) else output
        future_loss = masked_epsilon_loss(epsilon[..., 4:], batch['noise'][..., 4:], batch['mask'])
        return .25 * future_loss, {'future_relation_epsilon_mse': future_loss}

    def diagnostics(self, raw_history, policy):
        return dict(route='flat', execution_horizon=4, candidate_id=self.candidate_id,
                    action_frame='world' if self.candidate_id == 'P2' else 'cabinet_current_anchor',
                    generated_channels=self.action_dim, executed_channels=4)
