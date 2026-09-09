"""Peg P1/P2/P3, implemented after the isolated version-1 proposal was saved.

P1 adds fixed metric relations; P2 jointly denoises actions and future physical
relations; P3 combines them. Future targets exist only during training.
"""

import numpy as np
import torch

from relative_dp.model import masked_epsilon_loss
from round2.learning import Policy
from .plugins import BaselinePlugin


class PegJointPolicy(Policy):
    """Use the shared unbounded DDIM sampler with the declared joint dimension."""

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
                    sample, timestep, global_cond=obs_history.flatten(start_dim=1)
                )
            sample = scheduler.step(
                epsilon, timestep, sample, eta=0., use_clipped_model_output=False,
                generator=generator, return_dict=True,
            ).prev_sample
        return sample


class PegPlugin(BaselinePlugin):
    """Exactly the three mechanisms in design_records/peg-insert-side/proposal.json."""

    def __init__(self, task, schema, config):
        super().__init__(task, schema, config)
        if task != 'peg-insert-side' or schema.get('raw_dim') != 42:
            raise ValueError('Peg candidates require the frozen peg-insert-side raw42 schema')
        self.candidate_id = config.get('candidate_id')
        if self.candidate_id not in ('P1', 'P2', 'P3'):
            raise ValueError('Peg candidate must be P1, P2, or P3')
        self.relations = self.candidate_id in ('P1', 'P3')
        self.joint = self.candidate_id in ('P2', 'P3')

    def observation(self, raw):
        raw = np.asarray(raw, np.float32)
        if raw.shape[-1] != 42:
            raise ValueError('Expected common peg observation dimension42')
        if not self.relations:
            return raw.copy()
        hand, grasp = raw[..., :3], raw[..., 4:7]
        goal, head = raw[..., 36:39], raw[..., 39:42]
        grasp_gap, insertion_gap = grasp - hand, goal - head
        features = (
            raw,
            grasp_gap / .1,
            insertion_gap / .1,
            (head - grasp) / .1,
            (goal - hand) / .1,
            (hand - raw[..., 18:21]) / .01,
            (grasp - raw[..., 22:25]) / .01,
            np.linalg.norm(grasp_gap[..., :2], axis=-1, keepdims=True) / .1,
            np.linalg.norm(insertion_gap[..., 1:3], axis=-1, keepdims=True) / .1,
        )
        return np.concatenate(features, axis=-1).astype(np.float32, copy=False)

    def passthrough(self, dimension):
        expected = 62 if self.relations else 42
        if dimension != expected:
            raise ValueError(f'Expected encoded dimension{expected}, got{dimension}')
        # The proposal fixes every feature's units and scaling, without DN fitting.
        return list(range(dimension))

    def action_encode(self, actions, current_raw):
        actions = np.asarray(actions, np.float32)
        if actions.shape[-1] != 4:
            raise ValueError('Action encoding accepts exactly4 native world control channels')
        return actions.copy()

    def action_decode(self, actions, current_raw):
        actions = np.asarray(actions, np.float32)
        expected = 10 if self.joint else 4
        if actions.shape[-1] != expected:
            raise ValueError(f'Expected{expected} sampled channels, got{actions.shape[-1]}')
        # Shared Loaded.actions applies the native world cube clip AFTER this decode.
        return actions[..., :4].copy()

    def supervision(self, episodes, window_index):
        if not self.joint:
            return {}
        targets = np.zeros((len(window_index), 16, 6), dtype=np.float32)
        for row, (episode_index, start) in enumerate(window_index):
            episode = episodes[episode_index]
            obs = np.asarray(episode['obs'], np.float32)
            length = len(episode['actions'])
            if obs.shape != (length + 1, 42) or not 0 <= start < length:
                raise ValueError('Future targets need one next observation per valid action')
            valid = min(16, length - start)
            future = obs[start + 1:start + valid + 1]
            targets[row, :valid, :3] = (future[:, 39:42] - obs[start, 39:42]) / .1
            targets[row, :valid, 3:] = (future[:, 4:7] - future[:, :3]) / .1
        return {'future_relations': targets}

    def training_actions(self, encoded_actions, targets):
        actions = np.asarray(encoded_actions, np.float32)
        if actions.ndim != 3 or actions.shape[1:] != (16, 4):
            raise ValueError('Expected encoded action windows[windows,16,4]')
        if not self.joint:
            return actions.copy()
        future = np.asarray(targets['future_relations'], np.float32)
        if future.shape != (*actions.shape[:2], 6):
            raise ValueError('Future supervision does not match action windows')
        return np.concatenate((actions, future), axis=-1)

    def model_config(self, cfg):
        return {'action_dim': 10 if self.joint else 4}

    def build_policy(self, cfg):
        if cfg['action_dim'] != (10 if self.joint else 4):
            raise ValueError('Model configuration differs from proposed joint dimension')
        policy_type = PegJointPolicy if self.joint else Policy
        return policy_type(cfg['obs_dim'], cfg)

    def extra_loss(self, output, batch, step):
        if not self.joint:
            return None, {}
        epsilon = output['epsilon'] if isinstance(output, dict) else output
        auxiliary = masked_epsilon_loss(epsilon[..., 4:10], batch['noise'][..., 4:10], batch['mask'])
        return .2 * auxiliary, {'future_relation_epsilon_loss': auxiliary}

    def diagnostics(self, raw_history, policy):
        return dict(
            route='flat', execution_horizon=4, candidate_id=self.candidate_id,
            observation_features=62 if self.relations else 42,
            sampled_channels=10 if self.joint else 4,
            auxiliary_channels_executed=False,
        )
