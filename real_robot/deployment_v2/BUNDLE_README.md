# Push training_v2

This archive includes the exact API source, both final EMA checkpoints, separate
policy prior/usage documents, structured handoff information and a local host
interface. It contains no original recordings, training cache or API credentials.

Start with [the host guide](real_robot/deployment_v2/README.md) and give the
deployment agent [POLICY_CATALOG.md](real_robot/policies/push_v2/source/POLICY_CATALOG.md).
Read [actual training coverage](real_robot/policies/push_v2/TRAINING.md) alongside
the API-authored designs; ten source cuts had no eligible training examples.
Use `from real_robot.deployment_v2 import PushPolicy` for these checkpoints.

Install the locked inference environment from this extracted directory:

```bash
pixi install --manifest-path real_robot/deployment/pixi.toml --locked
pixi run --manifest-path real_robot/deployment/pixi.toml --locked python your_host_script.py
```

Both policies use independently trained Gaussian-mixture action models; diffusion
was permitted but not selected by the API. This run has 6,314 eligible examples
from 12/22 cuts. Trained weights and callable interfaces do not establish robot
performance, real-time control or generalization to unseen letters/words.
The receiving computer owns robot-side communication and actuation.
