# Active Exp2 rebuild

Execute APPL_EXP2_REBUILD_AND_EXECUTE.md. This is the single active Exp2 project:
src/appl, experiments/exp2, environments/exp2, data/exp2, runs/exp2.
Old frameworks/ is archived at archive/frameworks/; never import archived code.
Data/assets/calibration inputs live in the real directory data/exp2, outside the archive.
Preserve Exp1 source/data/results/root lock and paper. Work on dev.
Use /home/users/oscar/.pixi/bin/pixi run --manifest-path environments/exp2/pixi.toml --locked.
Physical GPUs 4–7 only; every GPU process uses appl.gpu device isolation.
Candidate code and MuJoCo scene modelling decisions come from real Runtime API.
Developer implements baseline, training, contracts, controlled tools, diagnostics.
No hidden-target feedback in design; no automatic provider retries.
Diagnose before formal runs. Failed diagnostic gates are reported, not waived.
Shared implementation bugs are fixed here with incident/version accounting.
