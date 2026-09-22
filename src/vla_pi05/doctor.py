"""Environment and asset checks for the VLA pipeline (`pixi run -e vla vla-doctor`)."""
from __future__ import annotations

import importlib
import json
import shutil
import sys
from pathlib import Path

from .paths import DATASETS_ROOT, PRETRAINED_ROOT, RAW_DATA_ROOT, RUNS_ROOT, WORK_ROOT


def _row(rows, name, ok, detail=""):
    rows.append((name, "OK" if ok else "FAIL", detail))
    return ok


def run(check_hub: bool = True) -> int:
    rows = []
    hard_fail = False
    _row(rows, "python", sys.version_info >= (3, 12), sys.version.split()[0])
    try:
        import torch

        n = torch.cuda.device_count()
        gpus = []
        for i in range(n):
            free, total = torch.cuda.mem_get_info(i)
            gpus.append(f"{i}:{torch.cuda.get_device_name(i)} {free / 2**30:.1f}/{total / 2**30:.1f}GiB free")
        hard_fail |= not _row(rows, "torch+cuda", torch.cuda.is_available() and n > 0, f"torch {torch.__version__}; " + "; ".join(gpus))
    except Exception as e:
        hard_fail |= not _row(rows, "torch+cuda", False, repr(e))
    for mod in ("lerobot", "transformers", "accelerate", "peft", "av", "torchcodec", "websockets", "msgpack", "sentencepiece"):
        try:
            m = importlib.import_module(mod)
            _row(rows, f"pkg {mod}", True, getattr(m, "__version__", ""))
        except Exception as e:
            hard_fail |= not _row(rows, f"pkg {mod}", False, repr(e)[:80]) if mod in ("lerobot", "transformers", "accelerate") else False
    try:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: F401

        _row(rows, "PI05Policy import", True)
    except Exception as e:
        hard_fail |= not _row(rows, "PI05Policy import", False, repr(e)[:100])
    try:
        from lerobot.utils.import_utils import get_safe_default_video_backend

        _row(rows, "video backend", True, get_safe_default_video_backend())
    except Exception as e:
        _row(rows, "video backend", False, repr(e)[:80])
    _row(rows, "ffmpeg binary", shutil.which("ffmpeg") is not None, shutil.which("ffmpeg") or "not on PATH (only needed for some encoders)")

    # raw data
    try:
        from .raw_episode import discover_episodes

        eps = discover_episodes(RAW_DATA_ROOT)
        per = {}
        for e in eps:
            per[e.parent.name] = per.get(e.parent.name, 0) + 1
        _row(rows, "raw data", len(eps) > 0, f"{RAW_DATA_ROOT}: {len(eps)} episodes {per}")
    except Exception as e:
        _row(rows, "raw data", False, repr(e)[:100])

    # work root + disk
    try:
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(WORK_ROOT)
        _row(rows, "work root", usage.free > 50 * 2**30, f"{WORK_ROOT}: {usage.free / 2**30:.0f} GiB free")
        dsets = sorted(p.name for p in DATASETS_ROOT.iterdir()) if DATASETS_ROOT.is_dir() else []
        _row(rows, "datasets", True, ", ".join(dsets) or "none converted yet")
        runs = sorted(p.name for p in RUNS_ROOT.iterdir()) if RUNS_ROOT.is_dir() else []
        _row(rows, "runs", True, ", ".join(runs) or "none")
    except Exception as e:
        _row(rows, "work root", False, repr(e)[:100])

    # pretrained + tokenizer
    try:
        from .assets import TOKENIZER_DIR, hub_tokenizer_accessible

        markers = list(PRETRAINED_ROOT.glob("*/vla_pretrained.json")) if PRETRAINED_ROOT.is_dir() else []
        detail = "; ".join(f"{m.parent.name}: tokenizer={json.loads(m.read_text()).get('tokenizer_name')}" for m in markers)
        _row(rows, "pretrained prepared", bool(markers), detail or "run `vla_pi05.cli prepare-pretrained`")
        local_tok = (TOKENIZER_DIR / "tokenizer.json").is_file()
        hub_ok = hub_tokenizer_accessible() if check_hub else None
        _row(rows, "tokenizer", local_tok or bool(hub_ok), f"hub(gated)={'ok' if hub_ok else ('skipped' if hub_ok is None else 'no access')} local={'ok' if local_tok else 'missing'}")
    except Exception as e:
        _row(rows, "pretrained/tokenizer", False, repr(e)[:100])

    width = max(len(r[0]) for r in rows)
    for name, status, detail in rows:
        print(f"{name:<{width}}  {status:<4}  {detail}")
    return 1 if hard_fail else 0
