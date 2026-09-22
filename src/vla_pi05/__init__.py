"""pi0.5 (openpi / LeRobot PyTorch port) fine-tuning and inference on real Franka data.

Layout:
    raw_episode.py     -- the ONLY module that knows the raw recording format
    manifest.py        -- (instruction, trajectory segment) pairs; the thing to update when cut data lands
    action_space.py    -- state/action vector definitions and resampling to the training rate
    convert_dataset.py -- manifest -> LeRobotDataset (what lerobot-train consumes)
    train.py           -- 4-GPU launcher around lerobot-train (accelerate)
    inference.py       -- checkpoint -> action chunks from raw observations
    serve.py/client.py -- websocket policy server/client (openpi_client wire-compatible)
    eval_offline.py    -- open-loop evaluation on held-out segments
"""

__version__ = "0.1.0"
