"""Reader for one raw teleop episode.

This is the ONLY module that knows the on-disk layout of the recordings. When the cut
data arrives in a different format, adapt `RawEpisode.load` (and `discover_episodes`)
and nothing downstream has to change.

Current layout (Franka Panda, spacemouse teleop, recorded on the camera PC):

    <task>/<episode>/meta.json            notes, rate_hz, n_steps, cameras, gripper_max_width_m, ...
    <task>/<episode>/obs.npz              t, q, dq, tau_ext, T_base_flange, T_base_ee,
                                          gripper_position, gripper_width_m, action_json, ...
    <task>/<episode>/third/000000.jpg     third-person RGB, 1280x720, one file per step
    <task>/<episode>/wrist/000000.jpg     wrist RGB, 1280x720
    <task>/<episode>/{third,wrist}_depth  16-bit depth PNGs (unused by pi0.5)
    <task>/<episode>/frames.json          optional per-frame camera timing (30 Hz recordings only)

Sample i of obs.npz corresponds to image file i of every camera.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from .paths import RAW_DATA_ROOT

IMAGE_EXT = ".jpg"
REQUIRED_OBS_KEYS = ("t", "q", "gripper_width_m", "T_base_ee")


@dataclass
class RawEpisode:
    path: Path
    meta: dict = field(repr=False)
    obs: dict = field(repr=False)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path) -> "RawEpisode":
        path = Path(path)
        meta_path = path / "meta.json"
        obs_path = path / "obs.npz"
        if not meta_path.is_file() or not obs_path.is_file():
            raise FileNotFoundError(f"{path} is not a raw episode (need meta.json and obs.npz)")
        meta = json.loads(meta_path.read_text())
        with np.load(obs_path, allow_pickle=False) as z:
            obs = {k: z[k] for k in z.files}
        missing = [k for k in REQUIRED_OBS_KEYS if k not in obs]
        if missing:
            raise ValueError(f"{path}: obs.npz misses keys {missing}")
        ep = cls(path=path, meta=meta, obs=obs)
        ep._check_consistency()
        return ep

    def _check_consistency(self) -> None:
        n = self.n_steps
        for key, arr in self.obs.items():
            if arr.shape[0] != n:
                raise ValueError(f"{self.path}: obs[{key}] has {arr.shape[0]} rows, expected {n}")
        for cam in self.cameras:
            for idx in (0, n - 1):
                p = self.image_path(cam, idx)
                if not p.is_file():
                    raise FileNotFoundError(f"{self.path}: missing image {p}")

    # --------------------------------------------------------------- identity
    @property
    def episode_id(self) -> str:
        return self.path.name

    @property
    def task(self) -> str:
        return self.path.parent.name

    @property
    def relpath(self) -> str:
        """Path relative to the raw root, e.g. 'flip_egg/episode_2026091916095801'."""
        try:
            return str(self.path.relative_to(RAW_DATA_ROOT))
        except ValueError:
            return f"{self.task}/{self.episode_id}"

    # ----------------------------------------------------------------- scalars
    @property
    def n_steps(self) -> int:
        return int(self.obs["t"].shape[0])

    @property
    def rate_hz(self) -> float:
        return float(self.meta.get("rate_hz", 0.0)) or self.estimated_rate_hz

    @cached_property
    def estimated_rate_hz(self) -> float:
        t = self.obs["t"]
        if len(t) < 2:
            return 0.0
        return float(1.0 / np.median(np.diff(t)))

    @property
    def notes(self) -> str:
        return str(self.meta.get("notes", "") or "").strip()

    @property
    def cameras(self) -> list[str]:
        cams = list(self.meta.get("cameras") or [])
        if not cams:
            cams = sorted(p.name for p in self.path.iterdir() if p.is_dir() and not p.name.endswith("_depth"))
        return cams

    @property
    def gripper_max_width_m(self) -> float:
        return float(self.meta.get("gripper_max_width_m", 0.14))

    @property
    def complete(self) -> bool:
        return bool(self.meta.get("complete", True))

    # ------------------------------------------------------------------ arrays
    @property
    def t(self) -> np.ndarray:
        return self.obs["t"]

    @property
    def q(self) -> np.ndarray:
        return self.obs["q"]

    @property
    def dq(self) -> np.ndarray | None:
        return self.obs.get("dq")

    @property
    def T_base_ee(self) -> np.ndarray:
        return self.obs["T_base_ee"]

    @property
    def gripper_width_m(self) -> np.ndarray:
        return self.obs["gripper_width_m"]

    @cached_property
    def gripper_cmd(self) -> np.ndarray:
        """Commanded gripper opening in [0, 1] parsed from action_json (NaN where absent)."""
        out = np.full(self.n_steps, np.nan, dtype=np.float64)
        aj = self.obs.get("action_json")
        if aj is None:
            return out
        for i, s in enumerate(aj):
            try:
                g = json.loads(str(s)).get("gripper")
            except (json.JSONDecodeError, AttributeError):
                g = None
            if g is not None:
                out[i] = float(g)
        return out

    # ------------------------------------------------------------------ images
    def image_path(self, camera: str, index: int) -> Path:
        return self.path / camera / f"{index:06d}{IMAGE_EXT}"

    def load_image(self, camera: str, index: int, size_hw: tuple[int, int] | None = None) -> np.ndarray:
        """Return an RGB uint8 HxWx3 array, optionally downscaled to (H, W)."""
        from PIL import Image

        with Image.open(self.image_path(camera, index)) as im:
            if size_hw is not None:
                # JPEG draft mode decodes at a reduced scale, which is much faster than a full decode.
                im.draft("RGB", (size_hw[1], size_hw[0]))
            im = im.convert("RGB")
            if size_hw is not None and (im.height, im.width) != tuple(size_hw):
                im = im.resize((size_hw[1], size_hw[0]), Image.BILINEAR)
            return np.asarray(im, dtype=np.uint8)

    # ------------------------------------------------------------------ summary
    def summary(self) -> dict:
        return {
            "episode": self.relpath,
            "task": self.task,
            "n_steps": self.n_steps,
            "rate_hz": self.rate_hz,
            "duration_s": float(self.t[-1] - self.t[0]) if self.n_steps > 1 else 0.0,
            "notes": self.notes,
            "cameras": self.cameras,
            "complete": self.complete,
        }


def discover_episodes(raw_root: str | Path = RAW_DATA_ROOT, tasks: list[str] | None = None) -> list[Path]:
    """All `<task>/<episode>` folders containing meta.json + obs.npz, sorted."""
    raw_root = Path(raw_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw data root {raw_root} does not exist")
    task_dirs = [p for p in sorted(raw_root.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    if tasks:
        task_dirs = [p for p in task_dirs if p.name in set(tasks)]
    episodes = []
    for td in task_dirs:
        for ep in sorted(td.iterdir()):
            if ep.is_dir() and (ep / "meta.json").is_file() and (ep / "obs.npz").is_file():
                episodes.append(ep)
    return episodes
