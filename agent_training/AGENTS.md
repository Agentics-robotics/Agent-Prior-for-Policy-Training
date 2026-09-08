# Project execution

All Python and project commands run through Pixi: `pixi run ...`.
Pixi binary on this host: `/home/users/oscar/.pixi/bin/pixi`.
Do not create Conda/venv environments or install with bare pip.
The immutable experiment specification is `ROUND1_SPEC.txt`.
Preserve frozen data, split IDs, formal configurations, and prior run records.
Never report training or evaluation as complete without validated artifacts.
Use GPU 1 (selected by Pixi activation) and one formal training process.
