"""Small implementation contract. Candidate modules are added only after proposals."""
import importlib
import numpy as np
from round2.learning import Policy


class BaselinePlugin:
    def __init__(self, task, schema, config):
        self.task, self.schema, self.config = task, schema, config

    def observation(self, raw):
        return np.asarray(raw, np.float32).copy()

    def passthrough(self, dimension):
        # Preserve native quaternion/absent-object values as in the validated R2 B0.
        values = list(range(7,11)) + list(range(14,18)) + list(range(25,29)) + list(range(32,36))
        if self.task in ('drawer','door'):
            values += list(range(11,18)) + list(range(29,36)) + [39,40]
        return [i for i in values if i < dimension]

    def action_encode(self, actions, current_raw):
        return np.asarray(actions,np.float32).copy()

    def action_decode(self, actions, current_raw):
        return np.asarray(actions,np.float32).copy()

    def supervision(self, episodes, window_index):
        return {}

    def build_policy(self, cfg):
        return Policy(cfg['obs_dim'],cfg)

    def extra_loss(self, output, batch, step):
        return None, {}

    def diagnostics(self, raw_history, policy):
        return dict(route='flat', execution_horizon=4)


def load_plugin(task, schema, implementation):
    module = implementation.get('plugin_module','round3.plugins')
    factory = getattr(importlib.import_module(module), implementation.get('plugin_class','BaselinePlugin'))
    return factory(task,schema,implementation.get('plugin_config',{}))
