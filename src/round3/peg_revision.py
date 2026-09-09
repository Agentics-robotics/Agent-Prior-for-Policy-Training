"""P4 hand-relative conditioning, saved before implementation in revision.json.

Only four position blocks in P1's retained raw conditioning are replaced.
The physical world actions, appended P1 features and diffusion policy are inherited.
"""

import numpy as np

from .peg_plugin import PegPlugin


class PegRevisionPlugin(PegPlugin):
    """P1 with hand-relative grasp, previous grasp, goal and observed head."""

    def __init__(self, task, schema, config):
        if config.get('candidate_id') != 'P4':
            raise ValueError('Peg revision requires candidate P4')
        super().__init__(task, schema, dict(config, candidate_id='P1'))
        self.candidate_id = 'P4'
        self.config = config

    def observation(self, raw):
        raw = np.asarray(raw, np.float32)
        # P1 computes its appended relations from the original world observation.
        # Its concatenate allocates independent storage; never mutate raw inputs.
        encoded = super().observation(raw)
        encoded[..., 4:7] = (raw[..., 4:7] - raw[..., :3]) / .1
        encoded[..., 22:25] = (raw[..., 22:25] - raw[..., 18:21]) / .1
        encoded[..., 36:39] = (raw[..., 36:39] - raw[..., :3]) / .1
        encoded[..., 39:42] = (raw[..., 39:42] - raw[..., :3]) / .1
        return encoded
