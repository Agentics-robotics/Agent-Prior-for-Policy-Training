"""Convert manifest segments into a LeRobotDataset (v3) that ``lerobot-train`` consumes.

Per resampled frame the dataset stores:

    observation.images.base_0_rgb        HxWx3 uint8 (video)   third-person camera
    observation.images.left_wrist_0_rgb  HxWx3 uint8 (video)   wrist camera
    observation.state                    float32[state_dim]
    action                               float32[action_dim]
    task                                 str (language instruction)

Camera keys deliberately match the pretrained pi0.5 input features, so the checkpoint's
config needs no rename map (the third pretrained slot, right_wrist_0_rgb, is padded as
an empty camera by the policy). ``meta/vla_conversion.json`` records the manifest, the
action-space layout and the raw episode of every dataset episode, so eval and inference
can reconstruct exactly what the numbers mean.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .action_space import ActionSpaceConfig, segment_arrays
from .manifest import Manifest, Segment
from .paths import DATASETS_ROOT, DEFAULT_CAMERA_MAP
from .raw_episode import RawEpisode

log = logging.getLogger(__name__)

SIDECAR_NAME = "vla_conversion.json"
ROBOT_TYPE = "franka_panda"


@dataclass
class ConvertConfig:
    name: str = "franka_pi05"
    out_root: Path = DATASETS_ROOT
    image_hw: tuple[int, int] = (144, 256)  # 1280x720 / 5, aspect preserved; pi0.5 letterboxes to 224
    camera_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CAMERA_MAP))
    action_space: ActionSpaceConfig = field(default_factory=ActionSpaceConfig)
    use_videos: bool = True
    image_writer_threads: int = 8
    decode_threads: int = 8
    splits: tuple[str, ...] = ("train", "val")
    overwrite: bool = False
    max_segments: int | None = None  # for smoke tests

    @property
    def fps(self) -> int:
        return int(self.action_space.target_fps)

    def dataset_dir(self, split: str) -> Path:
        return Path(self.out_root) / f"{self.name}_{split}"

    def repo_id(self, split: str) -> str:
        return f"local/{self.name}_{split}"

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["out_root"] = str(self.out_root)
        d["action_space"] = self.action_space.to_dict()
        return d


def dataset_features(cfg: ConvertConfig) -> dict:
    h, w = cfg.image_hw
    feats: dict[str, dict] = {}
    for key in cfg.camera_map:  # insertion order = camera order seen by the policy
        feats[f"observation.images.{key}"] = {
            "dtype": "video" if cfg.use_videos else "image",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        }
    feats["observation.state"] = {
        "dtype": "float32",
        "shape": (cfg.action_space.state_dim,),
        "names": cfg.action_space.state_names,
    }
    feats["action"] = {
        "dtype": "float32",
        "shape": (cfg.action_space.action_dim,),
        "names": cfg.action_space.action_names,
    }
    return feats


def load_segment_frames(ep: RawEpisode, seg: Segment, cfg: ConvertConfig) -> dict:
    """Resampled arrays plus decoded images for one segment (images: {cam_key: (N,H,W,3)})."""
    start, end = seg.resolve(ep.path.parent.parent, ep.n_steps)
    arrays = segment_arrays(ep, start, end, cfg.action_space)
    idx = arrays["indices"]

    def _load(job):
        cam_key, raw_cam, i = job
        return cam_key, i, ep.load_image(raw_cam, int(i), cfg.image_hw)

    jobs = [(k, raw, i) for k, raw in cfg.camera_map.items() for i in idx]
    images = {k: np.empty((len(idx), *cfg.image_hw, 3), dtype=np.uint8) for k in cfg.camera_map}
    pos = {int(i): n for n, i in enumerate(idx)}
    with ThreadPoolExecutor(max_workers=cfg.decode_threads) as pool:
        for cam_key, i, img in pool.map(_load, jobs):
            images[cam_key][pos[int(i)]] = img
    arrays["images"] = images
    return arrays


def convert_split(manifest: Manifest, cfg: ConvertConfig, split: str) -> Path | None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    segments = manifest.split(split)
    if cfg.max_segments:
        segments = segments[: cfg.max_segments]
    if not segments:
        log.warning("split %s has no segments; skipping", split)
        return None
    out_dir = cfg.dataset_dir(split)
    if out_dir.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"{out_dir} exists (use overwrite=True)")
        shutil.rmtree(out_dir)

    ds = LeRobotDataset.create(
        repo_id=cfg.repo_id(split),
        fps=cfg.fps,
        features=dataset_features(cfg),
        root=out_dir,
        robot_type=ROBOT_TYPE,
        use_videos=cfg.use_videos,
        image_writer_threads=cfg.image_writer_threads,
    )
    records = []
    raw_root = Path(manifest.raw_root)
    for ep_index, seg in enumerate(segments):
        ep = RawEpisode.load(raw_root / seg.episode)
        data = load_segment_frames(ep, seg, cfg)
        n = len(data["indices"])
        for key in ("state", "action"):
            if not np.isfinite(data[key]).all():
                bad = np.where(~np.isfinite(data[key]).all(axis=0))[0].tolist()
                names = cfg.action_space.state_names
                raise ValueError(f"{seg.id}: non-finite {key} values in dims {[names[i] for i in bad]}")
        for k in range(n):
            frame = {
                "observation.state": data["state"][k],
                "action": data["action"][k],
                "task": seg.instruction,
            }
            for cam_key in cfg.camera_map:
                frame[f"observation.images.{cam_key}"] = data["images"][cam_key][k]
            ds.add_frame(frame)
        ds.save_episode()
        records.append(
            {
                "dataset_episode_index": ep_index,
                "segment_id": seg.id,
                "episode": seg.episode,
                "task": seg.task,
                "instruction": seg.instruction,
                "raw_start": int(data["indices"][0]),
                "raw_end_exclusive": int(data["indices"][-1]) + 1,
                "raw_rate_hz": ep.rate_hz,
                "n_frames": n,
                "raw_indices_first_last": [int(data["indices"][0]), int(data["indices"][-1])],
            }
        )
        log.info("[%s] %d/%d %s -> %d frames @ %d Hz", split, ep_index + 1, len(segments), seg.id, n, cfg.fps)
    ds.finalize()

    sidecar = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "split": split,
        "repo_id": cfg.repo_id(split),
        "convert_config": cfg.to_dict(),
        "features": {k: {**v, "shape": list(v["shape"])} for k, v in dataset_features(cfg).items()},
        "manifest": {
            "digest": manifest.digest(),
            "raw_root": manifest.raw_root,
            "instruction_source": manifest.instruction_source,
            "created": manifest.created,
        },
        "episodes": records,
        "n_frames_total": int(sum(r["n_frames"] for r in records)),
    }
    (out_dir / "meta" / SIDECAR_NAME).write_text(json.dumps(sidecar, indent=1))
    return out_dir


def convert(manifest: Manifest, cfg: ConvertConfig) -> dict[str, Path | None]:
    problems = manifest.validate(check_files=True)
    if problems:
        raise ValueError("manifest problems:\n  " + "\n  ".join(problems))
    Path(cfg.out_root).mkdir(parents=True, exist_ok=True)
    return {split: convert_split(manifest, cfg, split) for split in cfg.splits}


def load_sidecar(dataset_dir: str | Path) -> dict:
    p = Path(dataset_dir) / "meta" / SIDECAR_NAME
    if not p.is_file():
        raise FileNotFoundError(f"{p} not found; was this dataset produced by vla_pi05.convert_dataset?")
    return json.loads(p.read_text())
