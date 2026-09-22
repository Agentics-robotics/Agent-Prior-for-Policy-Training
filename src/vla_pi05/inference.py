"""Run a fine-tuned pi0.5 checkpoint on raw observations.

``Pi05Runner`` hides LeRobot's processor pipelines: give it uint8 RGB images, the raw
state vector and an instruction, get back an unnormalized action chunk (chunk_size x
action_dim) in the dataset's action space. The layout of state/action is read from the
``vla_conversion.json`` sidecar that ``vla_pi05.train`` copies next to each run, so the
runner also knows how to turn a chunk into absolute joint targets (``chunk_to_targets``).

Typical robot loop at 10 Hz (chunk of 50 = 5 s; re-plan every ``replan_every`` steps):

    runner = Pi05Runner.from_run(run_dir)
    chunk = runner.predict({"base_0_rgb": img_third, "left_wrist_0_rgb": img_wrist}, state, "flip the fried egg")
    targets = runner.chunk_to_targets(chunk, state)   # (50, 8): 7 joint positions + gripper opening in [0,1]
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .action_space import ActionSpaceConfig, actions_to_targets
from .convert_dataset import SIDECAR_NAME
from .paths import DEFAULT_CAMERA_MAP

log = logging.getLogger(__name__)


@dataclass
class ObservationSpec:
    camera_keys: list[str]  # e.g. ["base_0_rgb", "left_wrist_0_rgb"] (order matters)
    camera_map: dict[str, str]  # policy camera -> raw camera folder
    image_hw: tuple[int, int]
    action_space: ActionSpaceConfig
    fps: int

    @classmethod
    def from_sidecar(cls, sidecar: dict) -> "ObservationSpec":
        cc = sidecar["convert_config"]
        return cls(
            camera_keys=list(cc["camera_map"].keys()),
            camera_map=dict(cc["camera_map"]),
            image_hw=tuple(cc["image_hw"]),
            action_space=ActionSpaceConfig.from_dict(cc["action_space"]),
            fps=int(cc["action_space"]["target_fps"]),
        )

    @classmethod
    def default(cls) -> "ObservationSpec":
        asc = ActionSpaceConfig()
        return cls(list(DEFAULT_CAMERA_MAP), dict(DEFAULT_CAMERA_MAP), (144, 256), asc, asc.target_fps)

    def to_dict(self) -> dict:
        return {
            "camera_keys": self.camera_keys,
            "camera_map": self.camera_map,
            "image_hw": list(self.image_hw),
            "action_space": self.action_space.to_dict(),
            "fps": self.fps,
        }


def is_peft_checkpoint(checkpoint_dir: Path) -> bool:
    return (Path(checkpoint_dir) / "adapter_config.json").is_file()


def load_policy(checkpoint_dir: str | Path, device: str = "cuda:0"):
    """Load a LeRobot pi0.5 checkpoint, full or PEFT/LoRA.

    A LoRA checkpoint holds only ``adapter_model.safetensors``; the base weights come from the
    ``base_model_name_or_path`` recorded in ``adapter_config.json`` (our local pi05_base dir).
    The adapter is merged into the base weights so inference runs at full-model speed.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    checkpoint_dir = Path(checkpoint_dir)
    cfg = PreTrainedConfig.from_pretrained(str(checkpoint_dir))
    cfg.device = device
    if not is_peft_checkpoint(checkpoint_dir):
        return PI05Policy.from_pretrained(str(checkpoint_dir), config=cfg)

    from peft import PeftConfig, PeftModel

    peft_cfg = PeftConfig.from_pretrained(str(checkpoint_dir))
    base = peft_cfg.base_model_name_or_path
    if not base or not (Path(base) / "config.json").is_file():
        raise FileNotFoundError(f"base model for LoRA checkpoint not found: {base!r}")
    cfg.use_peft = False
    cfg.pretrained_path = None
    log.info("loading base pi0.5 from %s and LoRA adapter from %s", base, checkpoint_dir)
    policy = PI05Policy.from_pretrained(base, config=cfg)
    peft_model = PeftModel.from_pretrained(policy, str(checkpoint_dir), config=peft_cfg, is_trainable=False)
    merged = peft_model.merge_and_unload()
    if not hasattr(merged, "predict_action_chunk"):
        raise RuntimeError("merge_and_unload did not return the underlying PI05Policy")
    return merged


def find_sidecar(checkpoint_dir: Path) -> Path | None:
    """Walk up from a checkpoint dir to the run dir that holds vla_conversion.json."""
    p = Path(checkpoint_dir).resolve()
    for _ in range(6):
        cand = p / SIDECAR_NAME
        if cand.is_file():
            return cand
        cand = p / "meta" / SIDECAR_NAME
        if cand.is_file():
            return cand
        p = p.parent
    return None


class Pi05Runner:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        spec: ObservationSpec | None = None,
        device: str = "cuda:0",
        num_inference_steps: int | None = None,
        compile_model: bool = False,
    ):
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device)
        sidecar_path = find_sidecar(self.checkpoint_dir)
        if spec is None:
            if sidecar_path is None:
                log.warning("no %s near %s; using default observation spec", SIDECAR_NAME, checkpoint_dir)
                spec = ObservationSpec.default()
            else:
                spec = ObservationSpec.from_sidecar(json.loads(sidecar_path.read_text()))
        self.spec = spec

        self.policy = load_policy(self.checkpoint_dir, device)
        cfg = self.policy.config
        cfg.device = device
        if num_inference_steps is not None:
            cfg.num_inference_steps = num_inference_steps
        if compile_model:
            cfg.compile_model = True
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.policy.to(self.device)
        self.policy.eval()
        # Full device string ("cuda:1"), not just the type: the processor must move inputs to the
        # same card as the model when several policies share one host.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=str(self.checkpoint_dir),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self.chunk_size = int(cfg.chunk_size)
        self.action_dim = int(cfg.output_features["action"].shape[0])
        self.state_dim = self.spec.action_space.state_dim
        self._torch = torch
        self.reset()

    @classmethod
    def from_run(cls, run_dir: str | Path, **kw) -> "Pi05Runner":
        from .train import latest_checkpoint

        return cls(latest_checkpoint(run_dir), **kw)

    # ------------------------------------------------------------------ api
    def reset(self) -> None:
        self.policy.reset()
        self._queue: list[np.ndarray] = []

    def metadata(self) -> dict:
        return {
            "checkpoint": str(self.checkpoint_dir),
            "chunk_size": self.chunk_size,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "spec": self.spec.to_dict(),
        }

    def _batch(self, images: dict[str, np.ndarray], state: np.ndarray, instruction: str) -> dict:
        torch = self._torch
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] != self.state_dim:
            raise ValueError(f"state has {state.shape[0]} dims, expected {self.state_dim} ({self.spec.action_space.state_names})")
        batch: dict = {
            "observation.state": torch.from_numpy(state),
            "task": instruction,
        }
        for key in self.spec.camera_keys:
            img = images.get(key)
            if img is None and key in self.spec.camera_map:
                img = images.get(self.spec.camera_map[key])  # accept raw folder names too
            if img is None:
                continue  # missing camera -> policy pads an empty one
            img = np.asarray(img)
            if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"image {key} must be HxWx3 uint8, got {img.shape} {img.dtype}")
            t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
            batch[f"observation.images.{key}"] = t
        return batch

    def predict(self, images: dict[str, np.ndarray], state: np.ndarray, instruction: str) -> np.ndarray:
        """Full action chunk (chunk_size, action_dim), unnormalized, in the dataset action space."""
        torch = self._torch
        t0 = time.perf_counter()
        batch = self.preprocessor(self._batch(images, state, instruction))
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(batch)
        actions = self.postprocessor(actions)
        out = actions[0].detach().float().cpu().numpy()
        self.last_infer_s = time.perf_counter() - t0
        return out

    def act(self, images, state, instruction, replan_every: int | None = None) -> np.ndarray:
        """One action per call; re-plans when the local queue is exhausted (open-loop chunking)."""
        n = replan_every or self.policy.config.n_action_steps
        if not self._queue:
            chunk = self.predict(images, state, instruction)
            self._queue = list(chunk[:n])
        return self._queue.pop(0)

    def chunk_to_targets(self, chunk: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """Absolute targets for the controller (undoes delta encodings)."""
        return actions_to_targets(chunk, np.asarray(current_state, dtype=np.float32), self.spec.action_space)


# ---------------------------------------------------------------- observation packing
def obs_from_message(msg: dict) -> tuple[dict[str, np.ndarray], np.ndarray, str]:
    """Accept both our nested format and openpi's flat ``observation/<key>`` format."""
    if "images" in msg:
        images = dict(msg["images"])
        state = np.asarray(msg["state"], dtype=np.float32)
        instruction = str(msg.get("instruction") or msg.get("prompt") or "")
        return images, state, instruction
    images = {}
    state = None
    for k, v in msg.items():
        if k.startswith("observation/") and k != "observation/state":
            images[k.split("/", 1)[1]] = v
        elif k in ("observation/state", "state"):
            state = np.asarray(v, dtype=np.float32)
    if state is None:
        raise KeyError("message has no state (expected 'state' or 'observation/state')")
    return images, state, str(msg.get("prompt") or msg.get("instruction") or "")
