"""Filesystem defaults for the VLA pipeline. Every path can be overridden by an env var."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Raw teleop recordings (one folder per task, one sub-folder per episode).
RAW_DATA_ROOT = Path(os.environ.get("VLA_RAW_DATA_ROOT", "/home/storage/tianrunhu/real_robot_data"))

# Everything the pipeline writes (datasets, pretrained snapshots, runs) lives on the data drive.
WORK_ROOT = Path(os.environ.get("VLA_WORK_ROOT", "/home/storage/tianrunhu/vla_pi05"))
DATASETS_ROOT = WORK_ROOT / "datasets"
RUNS_ROOT = WORK_ROOT / "runs"
ASSETS_ROOT = WORK_ROOT / "assets"
PRETRAINED_ROOT = WORK_ROOT / "pretrained"

# Small, versioned inputs live in the repo.
CONFIG_ROOT = REPO_ROOT / "configs" / "vla"
MANIFESTS_ROOT = CONFIG_ROOT / "manifests"

# Physical GPUs reserved for VLA work (user request 2026-09-20). GPUs 0-3 belong to Round 3.
DEFAULT_GPUS = os.environ.get("VLA_GPUS", "4,5,6,7")

DEFAULT_PRETRAINED_REPO = os.environ.get("VLA_PRETRAINED_REPO", "lerobot/pi05_base")
PALIGEMMA_TOKENIZER_REPO = "google/paligemma-3b-pt-224"
PALIGEMMA_TOKENIZER_GCS = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"

# Camera naming follows pi0.5's pretrained input features so no rename map is needed.
# key = feature suffix under `observation.images.`, value = folder name inside a raw episode.
DEFAULT_CAMERA_MAP = {"base_0_rgb": "third", "left_wrist_0_rgb": "wrist"}


def ensure_dirs() -> None:
    for p in (DATASETS_ROOT, RUNS_ROOT, ASSETS_ROOT, PRETRAINED_ROOT):
        p.mkdir(parents=True, exist_ok=True)
