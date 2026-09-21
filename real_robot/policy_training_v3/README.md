# cut_v5 implementation/training executor

Developer-owned version of policy_training, retained independently for the
three cut_v5 priors. Scientific source is generated exclusively by Runtime API
under runs/push_letters/training_v3/design_*/_session/work and submitted package_*/source.

The interface uses recorded [action.v, action.w] provenance labels, per-policy
sampling weights constrained to the original dataset groups, independent
20,000-step training and a candidate-action-only inference process.
See INTERFACE.md and ../configs/push_training_v3.json.

Run all commands in the locked Exp2 Pixi environment with PYTHONPATH including
the repository root and src. Entry points are:

- python -m real_robot.policy_training_v3.design --config ... --revision 0
- python -m real_robot.policy_training_v3 execute --config ... --revision 0
- python -m real_robot.policy_training_v3 status --config ...

No automatic provider retry or training restart. Implementation errors are
retained and may support an explicitly recorded API repair revision. Source,
executor and input identities are frozen in every submitted package/run.
