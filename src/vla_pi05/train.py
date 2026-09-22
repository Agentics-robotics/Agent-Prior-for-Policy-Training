"""Multi-GPU fine-tuning launcher: wraps ``lerobot-train`` under ``accelerate launch``.

We do not re-implement the training loop; LeRobot's trainer already provides DDP/FSDP,
checkpointing, resume and logging. This module owns the *policy* of how we run it:
memory-safe presets for 24 GB cards, reproducible run records, and the dataset/pretrained
plumbing produced by the other modules.

Presets (effective batch = batch_size x #GPUs):

    expert_only  freeze the PaliGemma VLM, train the 300M action expert + projections. DDP.
                 Fits comfortably on 24 GB with batch 8/GPU. Recommended first run.
    lora         LoRA (r=32) on the action-expert attention + projections. DDP.
    full         every parameter trainable, FSDP-sharded across the 4 GPUs. Experimental.

Resume:  ``vla-train --run-name X --resume`` continues from ``checkpoints/last``.
Dry run: ``--dry-run`` prints the exact command instead of executing it.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .assets import prepare_pretrained, pretrained_info
from .convert_dataset import SIDECAR_NAME, load_sidecar
from .paths import CONFIG_ROOT, DATASETS_ROOT, DEFAULT_GPUS, REPO_ROOT, RUNS_ROOT

log = logging.getLogger(__name__)

ACCELERATE_CONFIGS = {"ddp": CONFIG_ROOT / "accelerate_ddp.yaml", "fsdp": CONFIG_ROOT / "accelerate_fsdp.yaml"}

LORA_TARGET_REGEX = (
    r"(.*\.(gemma_expert\.model|paligemma\.model\.language_model)\.layers\.\d+\.self_attn\.(q|k|v|o)_proj"
    r"|model\.(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out))"
)

PRESETS: dict[str, dict] = {
    "expert_only": {
        "policy": {
            "train_expert_only": True,
            "freeze_vision_encoder": True,
            "dtype": "bfloat16",
            "gradient_checkpointing": True,
        },
        "batch_size": 8,
        "steps": 20_000,
        "accelerate": "ddp",
    },
    "lora": {
        "policy": {
            "freeze_vision_encoder": True,
            "dtype": "bfloat16",
            "gradient_checkpointing": True,
        },
        # openpi-style LoRA: attention projections of the PaliGemma LLM and of the action expert,
        # plus the (small) action/time projection layers trained fully. Vision tower stays frozen.
        "peft": {
            "method_type": "LORA",
            "r": 32,
            "lora_alpha": 64,
            "target_modules": LORA_TARGET_REGEX,
            "full_training_modules": "[]",
        },
        "batch_size": 8,
        "steps": 20_000,
        "accelerate": "ddp",
    },
    "full": {
        "policy": {
            "dtype": "bfloat16",
            "gradient_checkpointing": True,
        },
        "batch_size": 4,
        "steps": 20_000,
        "accelerate": "fsdp",
    },
}


@dataclass
class TrainArgs:
    dataset: str = "franka_pi05"  # dataset name as given to `vla-convert --name`
    dataset_dir: Path | None = None  # explicit train dataset dir (overrides `dataset`)
    run_name: str | None = None
    preset: str = "expert_only"
    pretrained: str | None = None  # local policy dir; default: prepare_pretrained()
    gpus: str = os.environ.get("CUDA_VISIBLE_DEVICES", DEFAULT_GPUS)
    steps: int | None = None
    batch_size: int | None = None
    lr: float | None = None
    num_workers: int = 4
    save_freq: int = 2_000
    log_freq: int = 50
    seed: int = 1000
    video_backend: str | None = None
    compile_model: bool = False
    wandb: bool = True  # real runs log to wandb; smoke runs never do
    wandb_project: str = "vla_pi05"
    image_aug: bool = True  # LeRobot photometric image transforms (brightness/contrast/...)
    resume: bool = False
    smoke: bool = False
    dry_run: bool = False
    extra: list[str] = field(default_factory=list)  # raw extra args forwarded to lerobot-train

    def resolved_dataset_dir(self) -> Path:
        if self.dataset_dir:
            return Path(self.dataset_dir)
        return DATASETS_ROOT / f"{self.dataset}_train"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _kv(prefix: str, d: dict) -> list[str]:
    out = []
    for k, v in d.items():
        if isinstance(v, bool):
            v = "true" if v else "false"
        out.append(f"--{prefix}.{k}={v}")
    return out


def wandb_logged_in() -> bool:
    """True when a wandb API key is available (env var or ~/.netrc)."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc = Path.home() / ".netrc"
    try:
        return netrc.is_file() and "api.wandb.ai" in netrc.read_text()
    except OSError:
        return False


def build_command(args: TrainArgs) -> tuple[list[str], dict, Path]:
    """Return (argv, run_record, run_dir)."""
    if args.preset not in PRESETS:
        raise ValueError(f"unknown preset {args.preset!r}; choose from {list(PRESETS)}")
    preset = PRESETS[args.preset]
    ds_dir = args.resolved_dataset_dir()
    sidecar = load_sidecar(ds_dir)
    info = json.loads((ds_dir / "meta" / "info.json").read_text())
    repo_id = sidecar["repo_id"]

    pretrained_dir = Path(args.pretrained) if args.pretrained else prepare_pretrained()
    if not (pretrained_dir / "config.json").is_file():
        raise FileNotFoundError(f"pretrained policy dir {pretrained_dir} has no config.json")

    gpu_list = [g for g in args.gpus.split(",") if g != ""]
    n_proc = len(gpu_list)
    steps = args.steps or preset["steps"]
    batch = args.batch_size or preset["batch_size"]
    if args.smoke:
        n_proc = 1
        gpu_list = gpu_list[:1]
        steps = min(steps, 10)
        batch = min(batch, 2)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.dataset}_{args.preset}_{stamp}"
    if args.smoke and not args.run_name:
        run_name = f"smoke_{run_name}"
    run_dir = RUNS_ROOT / run_name
    output_dir = run_dir / "train"

    accel_cfg = ACCELERATE_CONFIGS[preset["accelerate"] if n_proc > 1 else "ddp"]
    argv = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        f"--config_file={accel_cfg}",
        f"--num_processes={n_proc}",
        "-m",
        "lerobot.scripts.lerobot_train",
    ]
    if args.resume:
        last_cfg = output_dir / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        if not last_cfg.is_file():
            raise FileNotFoundError(f"cannot resume: {last_cfg} not found")
        argv += [f"--config_path={last_cfg}", "--resume=true"]
    else:
        argv += [
            f"--dataset.repo_id={repo_id}",
            f"--dataset.root={ds_dir}",
            f"--policy.path={pretrained_dir}",
            "--policy.device=cuda",
            "--policy.push_to_hub=false",
            f"--output_dir={output_dir}",
            f"--job_name={run_name}",
            f"--batch_size={batch}",
            f"--steps={steps}",
            f"--save_freq={min(args.save_freq, steps)}",
            f"--log_freq={args.log_freq}",
            f"--num_workers={args.num_workers}",
            f"--seed={args.seed}",
        ]
        use_wandb = args.wandb and not args.smoke
        argv += [f"--wandb.enable={'true' if use_wandb else 'false'}", f"--wandb.project={args.wandb_project}"]
        if use_wandb and not wandb_logged_in():
            log.warning("wandb: no API key found (run `wandb login` or set WANDB_API_KEY); logging offline, sync later with `wandb sync`")
            argv.append("--wandb.mode=offline")
        if args.image_aug:
            argv.append("--dataset.image_transforms.enable=true")
        argv += _kv("policy", preset["policy"])
        if "peft" in preset:
            argv += _kv("peft", preset["peft"])
        if args.lr is not None:
            argv.append(f"--policy.optimizer_lr={args.lr}")
        if args.compile_model:
            argv.append("--policy.compile_model=true")
        if args.video_backend:
            argv.append(f"--dataset.video_backend={args.video_backend}")
    argv += list(args.extra)

    record = {
        "run_name": run_name,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "preset": args.preset,
        "preset_spec": preset,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in dataclasses.asdict(args).items()},
        "gpus": gpu_list,
        "num_processes": n_proc,
        "effective_batch_size": batch * n_proc,
        "steps": steps,
        "dataset_dir": str(ds_dir),
        "dataset_repo_id": repo_id,
        "dataset_frames": info.get("total_frames"),
        "dataset_episodes": info.get("total_episodes"),
        "dataset_fps": info.get("fps"),
        "manifest_digest": sidecar["manifest"]["digest"],
        "pretrained_dir": str(pretrained_dir),
        "pretrained_info": pretrained_info(pretrained_dir),
        "command": shlex.join(argv),
        "cuda_visible_devices": ",".join(gpu_list),
        "wandb": {"enabled": args.wandb and not args.smoke, "project": args.wandb_project, "logged_in": wandb_logged_in()},
    }
    return argv, record, run_dir


def launch(args: TrainArgs) -> int:
    argv, record, run_dir = build_command(args)
    if args.dry_run:
        print(json.dumps(record, indent=2))
        print("\n" + record["command"])
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "vla_run.json").write_text(json.dumps(record, indent=2))
    src_sidecar = Path(record["dataset_dir"]) / "meta" / SIDECAR_NAME
    shutil.copyfile(src_sidecar, run_dir / SIDECAR_NAME)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = record["cuda_visible_devices"]
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")
    log_path = run_dir / ("train_resume.log" if args.resume else "train.log")
    log.info("launching %s (log: %s)", record["run_name"], log_path)
    log.info("%s", record["command"])
    with open(log_path, "a") as lf:
        lf.write(f"\n=== {record['created']} {record['command']}\n")
        lf.flush()
        proc = subprocess.Popen(argv, cwd=REPO_ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
        rc = proc.wait()
    record["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    record["return_code"] = rc
    (run_dir / "vla_run.json").write_text(json.dumps(record, indent=2))
    if rc != 0:
        log.error("training exited with code %s; see %s", rc, log_path)
    else:
        log.info("training finished; checkpoints under %s", run_dir / "train" / "checkpoints")
    return rc


def latest_checkpoint(run_dir: str | Path) -> Path:
    """`.../checkpoints/last/pretrained_model` of a run directory (or a checkpoints dir)."""
    run_dir = Path(run_dir)
    candidates = [
        run_dir / "train" / "checkpoints" / "last" / "pretrained_model",
        run_dir / "checkpoints" / "last" / "pretrained_model",
        run_dir / "last" / "pretrained_model",
        run_dir / "pretrained_model",
        run_dir,
    ]
    for c in candidates:
        if (c / "config.json").is_file() and (
            (c / "model.safetensors").is_file() or (c / "adapter_model.safetensors").is_file()
        ):
            return c
    raise FileNotFoundError(f"no checkpoint found under {run_dir}")
