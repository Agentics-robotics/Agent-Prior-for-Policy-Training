"""Record actual environment and dependency provenance."""
import importlib.metadata as md
import platform
import subprocess
import sys
import time
import os
from pathlib import Path
from .utils import ROOT, atomic_json, sha256
from .config import METAWORLD_COMMIT, DP_COMMIT

def doctor():
    import torch
    import metaworld
    import mujoco
    start = time.perf_counter()
    packages = {}
    for name in ("metaworld","mujoco","torch","diffusers","gymnasium","numpy","scipy","einops","imageio","imageio-ffmpeg","pytest","matplotlib"):
        d = md.distribution(name)
        packages[name] = {"version": d.version, "license": d.metadata.get("License-Expression") or d.metadata.get("License"), "direct_url": d.read_text("direct_url.json")}
    source_root = Path(metaworld.__file__).parent
    source_files = ["sawyer_xyz_env.py", "envs/sawyer_drawer_open_v3.py", "envs/sawyer_door_v3.py", "policies/sawyer_drawer_open_v3_policy.py", "policies/sawyer_door_open_v3_policy.py"]
    record = dict(timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                  platform=platform.platform(), python=sys.version, executable=sys.executable,
                  packages=packages, metaworld_commit=METAWORLD_COMMIT, diffusion_policy_commit=DP_COMMIT,
                  metaworld_source_hashes={p: sha256(source_root/p) for p in source_files},
                  cuda_available=torch.cuda.is_available(), torch_cuda=torch.version.cuda,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  mujoco_gl=os.environ.get("MUJOCO_GL"),
                  gpus=[dict(index=i,name=torch.cuda.get_device_name(i),total_memory=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())],
                  nvidia_smi=subprocess.run(["nvidia-smi"],text=True,capture_output=True).stdout)
    assert ".pixi/envs/" in sys.executable, "Project commands must run inside Pixi"
    assert packages["mujoco"]["version"] == "3.3.0"
    assert packages["torch"]["version"].split("+")[0] == "2.7.1"
    assert packages["diffusers"]["version"] == "0.35.1"
    record["elapsed_seconds"] = time.perf_counter()-start
    atomic_json(ROOT/"artifacts/environment.json",record)
    print(f"Doctor: Python {platform.python_version()}, torch {torch.__version__}, CUDA {torch.cuda.is_available()}, MuJoCo {mujoco.__version__}")
    print(f"Environment record: {ROOT/'artifacts/environment.json'}")
    return record
