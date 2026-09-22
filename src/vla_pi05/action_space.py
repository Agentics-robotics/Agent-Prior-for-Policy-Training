"""State / action vector definitions and resampling of a raw segment to the training rate.

The policy is trained and run at ``target_fps`` (10 Hz by default). Raw recordings are
10 Hz or 30 Hz; frames are picked by nearest robot timestamp on a uniform 1/target_fps grid.

Action spaces (all include the gripper as the last dimension, normalized opening in [0, 1]):

    joint_abs   action[k] = absolute joint positions at the next resampled frame (7) + gripper
    joint_delta action[k] = q[k+1] - q[k] (7) + absolute gripper
    ee_abs      action[k] = next EE position xyz (3) + rotation as 6D (first two rotation
                matrix columns) (6) + gripper, all in the robot base frame

State is always [q (7), gripper (1)] for joint spaces and [xyz, rot6d, gripper] for ee_abs.
The exact layout is written next to the dataset (``vla_conversion.json``) and read back by
the inference wrapper so the robot side never has to guess.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .raw_episode import RawEpisode

JOINT_NAMES = [f"joint_{i}" for i in range(7)]
EE_NAMES = ["ee_x", "ee_y", "ee_z", "ee_r00", "ee_r10", "ee_r20", "ee_r01", "ee_r11", "ee_r21"]
GRIPPER_NAME = "gripper"
ACTION_SPACES = ("joint_abs", "joint_delta", "ee_abs")


@dataclass
class ActionSpaceConfig:
    name: str = "joint_abs"
    target_fps: int = 10
    gripper_source: str = "width"  # "width" (measured opening) or "cmd" (teleop command, NaN-filled)
    chunk_size: int = 50  # informational; pi0.5 predicts this many steps per inference

    def __post_init__(self) -> None:
        if self.name not in ACTION_SPACES:
            raise ValueError(f"action space must be one of {ACTION_SPACES}, got {self.name!r}")
        if self.gripper_source not in ("width", "cmd"):
            raise ValueError("gripper_source must be 'width' or 'cmd'")
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")

    @property
    def state_names(self) -> list[str]:
        base = EE_NAMES if self.name == "ee_abs" else JOINT_NAMES
        return [*base, GRIPPER_NAME]

    @property
    def action_names(self) -> list[str]:
        return self.state_names

    @property
    def state_dim(self) -> int:
        return len(self.state_names)

    @property
    def action_dim(self) -> int:
        return len(self.action_names)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state_names"] = self.state_names
        d["action_names"] = self.action_names
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ActionSpaceConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ------------------------------------------------------------------ resampling
def resample_indices(t: np.ndarray, target_fps: float, start: int = 0, end: int | None = None) -> np.ndarray:
    """Indices of raw frames closest to a uniform grid at ``target_fps`` within [start, end)."""
    end = len(t) if end is None else end
    if end - start < 1:
        raise ValueError("empty range")
    tt = np.asarray(t[start:end], dtype=np.float64)
    if len(tt) == 1:
        return np.array([start])
    if np.any(np.diff(tt) < 0):
        raise ValueError("timestamps must be non-decreasing")
    grid = np.arange(tt[0], tt[-1] + 1e-9, 1.0 / target_fps)
    pos = np.searchsorted(tt, grid)
    pos = np.clip(pos, 1, len(tt) - 1)
    left, right = pos - 1, pos
    choose_left = (grid - tt[left]) <= (tt[right] - grid)
    idx = np.where(choose_left, left, right)
    idx = np.unique(idx)  # no duplicate frames when the raw rate is below target
    return idx + start


# ------------------------------------------------------------------ vectors
def rot6d_from_matrix(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6): first two columns, column-major (Zhou et al. 2019)."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


NO_GRIPPER_VALUE = 0.0  # used when an episode has no gripper signal at all (e.g. rod tool for pushing)


def gripper_normalized(ep: RawEpisode, idx: np.ndarray, source: str) -> np.ndarray:
    """Gripper opening in [0, 1]; never NaN.

    Preference order: requested source, the other source, then a constant. Episodes recorded
    with a tool instead of the gripper report NaN width and position -1, so both sources are
    absent and the dimension becomes the constant ``NO_GRIPPER_VALUE``.
    """
    width = ep.gripper_width_m[idx] / ep.gripper_max_width_m
    cmd = ep.gripper_cmd[idx]
    g = cmd if source == "cmd" else width
    other = width if source == "cmd" else cmd
    g = np.where(np.isnan(g), other, g)
    g = np.where(np.isnan(g), NO_GRIPPER_VALUE, g)
    return np.clip(g, 0.0, 1.0).astype(np.float32)


def build_state(ep: RawEpisode, idx: np.ndarray, cfg: ActionSpaceConfig) -> np.ndarray:
    """(N, state_dim) float32 state vectors at the resampled indices."""
    g = gripper_normalized(ep, idx, cfg.gripper_source)[:, None]
    if cfg.name == "ee_abs":
        T = ep.T_base_ee[idx]
        core = np.concatenate([T[:, :3, 3], rot6d_from_matrix(T[:, :3, :3])], axis=1)
    else:
        core = ep.q[idx]
    return np.concatenate([core, g], axis=1).astype(np.float32)


def build_actions(state: np.ndarray, cfg: ActionSpaceConfig) -> np.ndarray:
    """Actions aligned with states: action[k] targets state[k+1]; the last action holds."""
    nxt = np.concatenate([state[1:], state[-1:]], axis=0)
    if cfg.name == "joint_delta":
        act = nxt.copy()
        act[:, :-1] = nxt[:, :-1] - state[:, :-1]
        return act.astype(np.float32)
    return nxt.astype(np.float32)


def actions_to_targets(actions: np.ndarray, current_state: np.ndarray, cfg: ActionSpaceConfig) -> np.ndarray:
    """Turn a predicted chunk into absolute targets the robot can track (inverse of build_actions)."""
    actions = np.asarray(actions, dtype=np.float32)
    if cfg.name != "joint_delta":
        return actions
    out = actions.copy()
    out[:, :-1] = current_state[None, :-1] + np.cumsum(actions[:, :-1], axis=0)
    return out


def segment_arrays(ep: RawEpisode, start: int, end: int, cfg: ActionSpaceConfig) -> dict:
    idx = resample_indices(ep.t, cfg.target_fps, start, end)
    state = build_state(ep, idx, cfg)
    action = build_actions(state, cfg)
    return {"indices": idx, "state": state, "action": action, "timestamps": ep.t[idx] - ep.t[idx[0]]}
