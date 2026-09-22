# Flip egg cut_v1 interface

Isolated fork of the current Push cut_v6 developer interface, with the exact
general v5 prompt and a separate Flip egg task specification. Scientific cuts,
priors, model IO and supervision are authored only by Runtime API. Fork
provenance is retained in `../runs/flip_egg/cut_v1/setup/fork_provenance.json`.

Use the locked Exp2 Pixi environment and module
`real_robot.flip_egg_cut_v1.runner`, commands prepare/run/verify/status.
Raw data remain external and read-only. No training or robot execution.
