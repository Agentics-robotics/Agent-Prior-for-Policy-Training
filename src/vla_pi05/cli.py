"""Command line entry: ``python -m vla_pi05.cli <command>`` (wrapped by the pixi `vla-*` tasks)."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .paths import DATASETS_ROOT, DEFAULT_GPUS, MANIFESTS_ROOT, RAW_DATA_ROOT, RUNS_ROOT, ensure_dirs


def _log():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _pin_single_gpu(device: str) -> str:
    """Expose exactly one physical GPU and return the device string to use (``cuda:0``).

    pi0.5 inference through LeRobot/PEFT returns wrong or NaN actions when the model lives on a
    non-zero CUDA index inside a multi-GPU process (verified 2026-09-21: identical checkpoint,
    correct on cuda:0 or when a card is exposed alone; wrong on cuda:1..3). Restricting
    CUDA_VISIBLE_DEVICES before torch initialises sidesteps that. Must run before any torch import.
    """
    if not device.startswith("cuda"):
        return device
    visible = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", DEFAULT_GPUS).split(",") if g != ""]
    idx = int(device.split(":")[1]) if ":" in device else 0
    if idx >= len(visible):
        raise SystemExit(f"{device} requested but only {len(visible)} GPUs visible: {visible}")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible[idx]
    logging.getLogger(__name__).info("pinned physical GPU %s as cuda:0", visible[idx])
    return "cuda:0"


def _resolve_checkpoint(args) -> Path:
    from .train import latest_checkpoint

    if getattr(args, "checkpoint", None):
        return Path(args.checkpoint)
    if getattr(args, "run", None):
        run = Path(args.run)
        if not run.is_absolute() and not run.exists():
            run = RUNS_ROOT / run
        return latest_checkpoint(run)
    raise SystemExit("give --run <run dir or name> or --checkpoint <pretrained_model dir>")


# ----------------------------------------------------------------- commands
def cmd_doctor(args):
    from .doctor import run

    return run(check_hub=not args.no_hub)


def cmd_inspect_raw(args):
    from .raw_episode import RawEpisode, discover_episodes

    eps = discover_episodes(args.raw_root, args.tasks)
    for p in eps[: args.limit]:
        print(json.dumps(RawEpisode.load(p).summary()))
    print(f"{len(eps)} episodes total", file=sys.stderr)
    return 0


def cmd_manifest(args):
    from .manifest import build_manifest

    imap = {}
    for item in args.instruction or []:
        task, _, text = item.partition("=")
        imap[task] = text
    m = build_manifest(
        raw_root=args.raw_root,
        tasks=args.tasks,
        val_fraction=args.val_fraction,
        seed=args.seed,
        instruction_source=args.instruction_source,
        instruction_map=imap or None,
        include_incomplete=args.include_incomplete,
    )
    out = Path(args.out) if args.out else MANIFESTS_ROOT / f"{args.name}.json"
    m.save(out)
    print(json.dumps(m.stats(), indent=1))
    print(f"wrote {out}")
    return 0


def cmd_manifest_cut(args):
    from .cut_import import manifest_from_cut

    m = manifest_from_cut(
        cut_root=args.cut_root,
        raw_task=args.raw_task,
        val_ids=tuple(args.val or ()),
        with_goal_coords=not args.no_goal_coords,
        trim_exclusions=not args.keep_exclusions,
    )
    problems = m.validate(check_files=True)
    if problems:
        raise SystemExit("manifest problems:\n  " + "\n  ".join(problems))
    out = Path(args.out) if args.out else MANIFESTS_ROOT / f"{args.name}.json"
    m.save(out)
    for seg in m.segments:
        print(f"{seg.split:5s} {seg.id:14s} [{seg.start:5d},{seg.end:5d})  {seg.instruction}")
    print(json.dumps(m.stats(), indent=1))
    print(f"wrote {out}")
    return 0


def cmd_convert(args):
    from .action_space import ActionSpaceConfig
    from .convert_dataset import ConvertConfig, convert
    from .manifest import Manifest

    manifest_path = Path(args.manifest)
    if not manifest_path.exists() and (MANIFESTS_ROOT / f"{args.manifest}.json").exists():
        manifest_path = MANIFESTS_ROOT / f"{args.manifest}.json"
    m = Manifest.load(manifest_path)
    cam_map = dict(kv.split("=") for kv in args.camera) if args.camera else None
    cfg = ConvertConfig(
        name=args.name,
        out_root=Path(args.out_root),
        image_hw=(args.image_hw[0], args.image_hw[1]),
        action_space=ActionSpaceConfig(name=args.action_space, target_fps=args.fps, gripper_source=args.gripper_source),
        use_videos=not args.images,
        splits=tuple(args.splits),
        overwrite=args.overwrite,
        max_segments=args.max_segments,
        **({"camera_map": cam_map} if cam_map else {}),
    )
    out = convert(m, cfg)
    print(json.dumps({k: str(v) for k, v in out.items()}, indent=1))
    return 0


def cmd_prepare_pretrained(args):
    from .assets import prepare_pretrained, pretrained_info

    d = prepare_pretrained(repo_id=args.repo, prefer_hub_tokenizer=not args.no_hub_tokenizer, force=args.force)
    print(json.dumps({"policy_dir": str(d), **pretrained_info(d)}, indent=1))
    return 0


def cmd_train(args):
    from .train import TrainArgs, launch

    targs = TrainArgs(
        dataset=args.dataset,
        dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
        run_name=args.run_name,
        preset=args.preset,
        pretrained=args.pretrained,
        gpus=args.gpus,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        save_freq=args.save_freq,
        log_freq=args.log_freq,
        seed=args.seed,
        video_backend=args.video_backend,
        compile_model=args.compile,
        wandb=not args.no_wandb,
        image_aug=not args.no_image_aug,
        resume=args.resume,
        smoke=args.smoke,
        dry_run=args.dry_run,
        extra=args.extra,
    )
    return launch(targs)


def cmd_eval(args):
    args.device = _pin_single_gpu(args.device)
    from .eval_offline import evaluate

    ckpt = _resolve_checkpoint(args)
    ds_dir = Path(args.dataset_dir) if args.dataset_dir else DATASETS_ROOT / f"{args.dataset}_val"
    out = Path(args.out) if args.out else ckpt.parent.parent.parent.parent / "eval" / ds_dir.name
    m = evaluate(ckpt, ds_dir, out, stride=args.stride, batch_size=args.batch_size, max_samples=args.max_samples, device=args.device, num_inference_steps=args.num_inference_steps)
    print(json.dumps({k: v for k, v in m.items() if k not in ("examples",)}, indent=1))
    return 0


def cmd_serve(args):
    args.device = _pin_single_gpu(args.device)
    from .inference import Pi05Runner
    from .serve import PolicyServer

    runner = Pi05Runner(_resolve_checkpoint(args), device=args.device, num_inference_steps=args.num_inference_steps, compile_model=args.compile)
    PolicyServer(runner, host=args.host, port=args.port).serve_forever()
    return 0


def cmd_infer_raw(args):
    """Run one inference on a frame of a raw episode (sanity check of the runner)."""
    args.device = _pin_single_gpu(args.device)

    import numpy as np

    from .action_space import build_state
    from .inference import Pi05Runner
    from .raw_episode import RawEpisode

    runner = Pi05Runner(_resolve_checkpoint(args), device=args.device)
    ep_path = Path(args.episode)
    if not ep_path.is_absolute():
        ep_path = RAW_DATA_ROOT / ep_path
    ep = RawEpisode.load(ep_path)
    idx = np.array([args.index])
    state = build_state(ep, idx, runner.spec.action_space)[0]
    images = {k: ep.load_image(raw, args.index, runner.spec.image_hw) for k, raw in runner.spec.camera_map.items()}
    instruction = args.instruction or ep.notes or ep.task.replace("_", " ")
    chunk = runner.predict(images, state, instruction)
    targets = runner.chunk_to_targets(chunk, state)
    np.set_printoptions(precision=3, suppress=True, linewidth=160)
    print("instruction:", instruction)
    print("state      :", state)
    print("chunk[0:5] :\n", targets[:5])
    print(f"infer time : {runner.last_infer_s * 1000:.0f} ms")
    return 0


# ----------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vla_pi05", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("doctor", help="check environment, GPUs, data, assets")
    s.add_argument("--no-hub", action="store_true", help="skip the Hugging Face gated-tokenizer probe")
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("inspect-raw", help="list raw episodes")
    s.add_argument("--raw-root", default=str(RAW_DATA_ROOT))
    s.add_argument("--tasks", nargs="*")
    s.add_argument("--limit", type=int, default=1000)
    s.set_defaults(fn=cmd_inspect_raw)

    s = sub.add_parser("manifest", help="build a placeholder manifest (one segment per raw episode)")
    s.add_argument("--name", default="all_tasks")
    s.add_argument("--out")
    s.add_argument("--raw-root", default=str(RAW_DATA_ROOT))
    s.add_argument("--tasks", nargs="*")
    s.add_argument("--val-fraction", type=float, default=0.1)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--instruction-source", choices=["notes", "task"], default="notes")
    s.add_argument("--instruction", nargs="*", metavar="TASK=TEXT", help="override instruction per task")
    s.add_argument("--include-incomplete", action="store_true")
    s.set_defaults(fn=cmd_manifest)

    s = sub.add_parser("manifest-cut", help="build a manifest with prompts from the cut_v3 push segments")
    s.add_argument("--cut-root", default="/home/storage/oscar/cut_v3")
    s.add_argument("--name", default="push_cut_v3")
    s.add_argument("--out")
    s.add_argument("--raw-task", default="push_letters")
    s.add_argument("--val", nargs="*", default=["b_d_stage", "b_i_align"], help="cut segment ids held out for eval")
    s.add_argument("--no-goal-coords", action="store_true", help="omit the numeric goal centre from the prompt")
    s.add_argument("--keep-exclusions", action="store_true", help="do not trim the boundary exclusion windows")
    s.set_defaults(fn=cmd_manifest_cut)

    s = sub.add_parser("convert", help="manifest -> LeRobotDataset train/val")
    s.add_argument("--manifest", required=True, help="path or manifest name under configs/vla/manifests")
    s.add_argument("--name", default="franka_pi05")
    s.add_argument("--out-root", default=str(DATASETS_ROOT))
    s.add_argument("--fps", type=int, default=10)
    s.add_argument("--image-hw", type=int, nargs=2, default=[144, 256])
    s.add_argument("--action-space", choices=["joint_abs", "joint_delta", "ee_abs"], default="joint_abs")
    s.add_argument("--gripper-source", choices=["width", "cmd"], default="width")
    s.add_argument("--camera", nargs="*", metavar="POLICY_KEY=RAW_FOLDER")
    s.add_argument("--images", action="store_true", help="store PNG frames instead of video")
    s.add_argument("--splits", nargs="*", default=["train", "val"])
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--max-segments", type=int)
    s.set_defaults(fn=cmd_convert)

    s = sub.add_parser("prepare-pretrained", help="download pi05_base and resolve the tokenizer")
    s.add_argument("--repo", default="lerobot/pi05_base")
    s.add_argument("--no-hub-tokenizer", action="store_true", help="always use the local SentencePiece-converted tokenizer")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_prepare_pretrained)

    s = sub.add_parser("train", help="launch fine-tuning on the 4 GPUs")
    s.add_argument("--dataset", default="franka_pi05")
    s.add_argument("--dataset-dir")
    s.add_argument("--run-name")
    s.add_argument("--preset", choices=["expert_only", "lora", "full"], default="expert_only")
    s.add_argument("--pretrained", help="local policy dir (default: prepared lerobot/pi05_base)")
    s.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", DEFAULT_GPUS), help="physical GPU ids (default: CUDA_VISIBLE_DEVICES of the vla env = 4,5,6,7)")
    s.add_argument("--steps", type=int)
    s.add_argument("--batch-size", type=int, help="per GPU")
    s.add_argument("--lr", type=float)
    s.add_argument("--num-workers", type=int, default=4)
    s.add_argument("--save-freq", type=int, default=2000)
    s.add_argument("--log-freq", type=int, default=50)
    s.add_argument("--seed", type=int, default=1000)
    s.add_argument("--video-backend", choices=["torchcodec", "pyav"])
    s.add_argument("--compile", action="store_true")
    s.add_argument("--no-wandb", action="store_true", help="disable wandb logging (on by default for non-smoke runs)")
    s.add_argument("--no-image-aug", action="store_true", help="disable photometric image augmentation")
    s.add_argument("--resume", action="store_true")
    s.add_argument("--smoke", action="store_true", help="1 GPU, 10 steps, tiny batch")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("extra", nargs="*", help="extra args forwarded to lerobot-train (after --)")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("eval", help="open-loop evaluation on the val split")
    s.add_argument("--run")
    s.add_argument("--checkpoint")
    s.add_argument("--dataset", default="franka_pi05")
    s.add_argument("--dataset-dir")
    s.add_argument("--out")
    s.add_argument("--stride", type=int, default=10)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--max-samples", type=int)
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--num-inference-steps", type=int)
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("serve", help="websocket policy server for the robot PC")
    s.add_argument("--run")
    s.add_argument("--checkpoint")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--num-inference-steps", type=int)
    s.add_argument("--compile", action="store_true")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("infer-raw", help="single inference on a raw episode frame")
    s.add_argument("--run")
    s.add_argument("--checkpoint")
    s.add_argument("--episode", required=True, help="e.g. flip_egg/episode_2026091916095801")
    s.add_argument("--index", type=int, default=0)
    s.add_argument("--instruction")
    s.add_argument("--device", default="cuda:0")
    s.set_defaults(fn=cmd_infer_raw)
    return p


def main(argv=None) -> int:
    _log()
    ensure_dirs()
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
