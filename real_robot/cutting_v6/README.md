# cut_v6: planner-supported learning design

Developer interface, reviewed 2026-09-21. Scientific grouping, representations,
targets, priors and policy/executor contracts are authored by Runtime API.

Completed: one learned `shape_push_v1` policy/prior, one supplied executor group,
14 materialized segments and 47 sparse decision anchors. All 24 Astra/xhigh calls
and 51 published files passed verification. See the
[Chinese review](../reports/PUSH_CUT_V6_REVIEW.md) and
[completion receipt](../runs/push_letters/cut_v6_with_example/completion_receipt.json).
Do not restart this completed run. No implementation/training is pending under
the cut-stage authorization.

The active configuration is
[`push_cut_v6_with_example.json`](../configs/push_cut_v6_with_example.json).
It combines the [general prompt](../prompts/general_cut_and_prior_v5.md) and
[Push task specification](../task_specifications/push_letters_v2.md).
At the user's request the generic output examples include the short phrase
"Spatial goals (e.g., a contact point)". Policy count and output schemas are
left to the API. This is not an unprompted discovery of contact-point outputs.

The new [data contract](DATA_CONTRACT.md) supports:

- learned data groups with dense or explicitly indexed sparse decisions;
- supplied-executor groups with retained raw evidence and no learned policy;
- independent raw context/target evidence and supervision anchors;
- a policy catalog plus a separate system execution catalog.

Source arrays and media pairings are preserved exactly. Future target derivation,
preprocessing and executor bindings are designs at this stage. Old training and
deployment packages do not automatically implement these new schemas.

The user-declared planner/controller capability is a design input. A concrete
remote planner integration and its calibration have not been verified here.
No training or robot execution belongs to this round.

Run artifacts are under `real_robot/runs/push_letters/cut_v6_with_example`;
published cuts are under `real_robot/data/push_letters/cut_v6`. To inspect the
active run without restarting it:

```bash
/home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked --no-install python -m real_robot.execution.cut_v6 status --config real_robot/configs/push_cut_v6_with_example.json
```

The initial no-example attempt remains under
`real_robot/runs/push_letters/cut_v6`. It was interrupted on the user's prompt
amendment before submission. Its nine completed calls and one in-flight call
remain in the original journal; the interruption receipt retains known costs
and unresolved provider outcome. Do not resume it or replay its pending request.
The replacement uses a fresh session and does not receive its design output.

The shared older cut implementation, prior prompts and published designs remain
unchanged. The independent v6 schema passed nine new tests together with twelve
existing data/transport interface checks. These are structural and provenance
checks, not policy performance tests.
