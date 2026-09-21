# Local trained-policy interface

For downloading to another computer, use the newer
[portable deployment interface](../deployment/README.md). It uses Git-visible
API source, two inference checkpoints and an independent locked environment.
The original training-run interface below is retained for provenance and local
reproduction; it still reads the historical `runs/` package.

This directory is the developer-owned execution interface. Scientific policy and
conversion code is API-authored and retained under
[`train_v1/package_00/source`](../runs/push_letters/train_v1/package_00/source/).
Read its [CALLING.md](../runs/push_letters/train_v1/package_00/source/CALLING.md)
for the exact raw-observation, scene-instance, goal and controller contracts.
See [PUSH_TRAIN.md](../reports/PUSH_TRAIN.md) for actual data coverage and status.

From the repository root, inspect execution with the locked environment:

```bash
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked python -m real_robot.policy_training status
```

After the relevant policy has a completed `training/<policy_id>/result.json`,
the Python host interface is:

```python
from real_robot.policy_training.inference import PolicyProcess

worker = PolicyProcess(
    config="real_robot/configs/push_train_v1.json",
    policy_id="contour_push",  # or independently trained "visual_push"
    output="real_robot/runs/push_letters/train_v1/local_calls/call_001",
    gpu=0,                    # choose an available allocated GPU: 0 or 1
    revision=0,
)
try:
    # observation is ONE current paired record, including original RGB arrays,
    # robot state, timestamp fields and the live calibration metadata.
    inventory = worker.inventory(observation, scene_version="current_scene")

    # The HLA selects a current inventory instance and constructs the call
    # using that instance's returned template and one fixed spatial goal.
    # call_from_high_level_agent follows the API-authored CALLING.md schema.
    answer = worker.act(observation, call_from_high_level_agent, reset=True)
    # For subsequent observations in that same invocation, leave reset=False.
finally:
    worker.close()
```

The snippet documents the interface; `observation` and
`call_from_high_level_agent` must come from the caller. It does not invent a
controller or a goal. The output directory must be new; existing logs are never
silently overwritten. Run the host script through the same locked Pixi
environment. Each worker loads its own final EMA checkpoint and normalization,
checks source/checkpoint hashes, and retains memory across its `act` requests.
Do not share invocation memory across policies or changed scene/instance/goals.

`inventory` returns observed numeric templates with scene-scoped instance IDs;
it does not recognize letter names. `act` returns the API's status, diagnostics
and optional candidate seven-command vector. Optional `controller_contract`
requests command-description conversion with the API's explicit checks. Neither
operation sends hardware commands. Original training trajectories are not granted
to the inference worker; current observations are passed over local numeric IPC.

The completed [library.json](../runs/push_letters/train_v1/library.json) records
both checkpoint paths and hashes, exact API source provenance and training
receipts. It is published only after both fixed-budget trainings and checkpoint
reload checks succeed. During an active run this file may not exist yet.
