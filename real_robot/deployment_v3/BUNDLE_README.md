# Push training_v3 inference bundle

This archive contains the three final EMA checkpoints, exact API source and
agent-facing prior/usage/handoff documents, a local inference adapter, and a
locked Pixi environment. It has no training recordings, caches, runs or credentials.

Read [the installation and agent guide](real_robot/deployment_v3/README.md).
From this directory, on Linux x86_64 with NVIDIA GPU and the documented system prerequisites:

```bash
pixi install --manifest-path real_robot/deployment/pixi.toml --locked
pixi run --manifest-path real_robot/deployment/pixi.toml --locked python -m real_robot.deployment_v3 check --gpu 0
```

Use `real_robot.deployment_v3.PushPolicy`. The independent models are
`tool_waypoint_v1`, `piece_contact_graph_v1` and `piece_goal_field_v1`.
`real_robot/policies/push_v3/source/POLICY_CATALOG.md` is the agent's entry.
Candidates retain recorded six-dimensional v/w encoding. The hardware decoder
remains disabled; robot integration belongs to the receiving host.
