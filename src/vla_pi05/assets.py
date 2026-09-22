"""Pretrained assets: the PaliGemma tokenizer and a local pi0.5 base checkpoint directory.

Two facts drive this module:

1. ``lerobot/pi05_base`` (weights + processor configs) is public, but its preprocessor
   references the tokenizer of ``google/paligemma-3b-pt-224``, which is a *gated* repo.
   Without an approved Hugging Face token the tokenizer download fails.
2. Google publishes the identical SentencePiece model ungated at
   ``gs://big_vision/paligemma_tokenizer.model``. We convert it to a Hugging Face fast
   tokenizer (token ids verified equal to SentencePiece) and point the processor at it.

``prepare_pretrained`` therefore builds ``<PRETRAINED_ROOT>/<name>/`` containing symlinks to
the downloaded weights/config plus a patched ``policy_preprocessor.json`` whose
``tokenizer_processor.tokenizer_name`` is a local path. Training and inference use that
directory as ``--policy.path``; everything downstream is offline-reproducible.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import urllib.request
from pathlib import Path

from .paths import (
    ASSETS_ROOT,
    DEFAULT_PRETRAINED_REPO,
    PALIGEMMA_TOKENIZER_GCS,
    PALIGEMMA_TOKENIZER_REPO,
    PRETRAINED_ROOT,
)

log = logging.getLogger(__name__)

TOKENIZER_DIR = ASSETS_ROOT / "paligemma_tokenizer"
_SPM_BOS_ID, _SPM_EOS_ID, _SPM_PAD_ID, _SPM_VOCAB = 2, 1, 0, 257152
_CHECK_TEXT = "Task: flip the fried egg, State: 12 200 3;\nAction: "


# ------------------------------------------------------------------ tokenizer
def hub_tokenizer_accessible() -> bool:
    """True when the gated PaliGemma tokenizer can be loaded (token with accepted licence)."""
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(PALIGEMMA_TOKENIZER_REPO)
        return True
    except Exception as e:  # gated -> OSError/401
        log.info("hub tokenizer not accessible: %s", str(e).splitlines()[0][:120])
        return False


def build_local_tokenizer(out_dir: Path = TOKENIZER_DIR, force: bool = False) -> Path:
    """Download the public SentencePiece model and convert it to a HF fast tokenizer directory."""
    out_dir = Path(out_dir)
    if not force and (out_dir / "tokenizer.json").is_file() and (out_dir / "tokenizer_config.json").is_file():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    spm_path = out_dir / "tokenizer.model"
    if force or not spm_path.is_file():
        log.info("downloading %s", PALIGEMMA_TOKENIZER_GCS)
        urllib.request.urlretrieve(PALIGEMMA_TOKENIZER_GCS, spm_path)

    from transformers import GemmaTokenizerFast
    from transformers.convert_slow_tokenizer import SentencePieceExtractor

    kw = SentencePieceExtractor(str(spm_path)).extract(None)
    tok = GemmaTokenizerFast(
        **kw,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        add_bos_token=True,
        add_eos_token=False,
    )
    tok.save_pretrained(out_dir)
    verify_tokenizer(out_dir, spm_path)
    return out_dir


def verify_tokenizer(tok_dir: Path, spm_path: Path | None = None) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    if len(tok) != _SPM_VOCAB or tok.bos_token_id != _SPM_BOS_ID or tok.pad_token_id != _SPM_PAD_ID:
        raise RuntimeError(f"tokenizer in {tok_dir} has unexpected vocab/special ids")
    ids = tok(_CHECK_TEXT)["input_ids"]
    if spm_path is not None and Path(spm_path).is_file():
        import sentencepiece as spm

        ref = spm.SentencePieceProcessor(model_file=str(spm_path)).encode(_CHECK_TEXT, add_bos=True)
        if ids != ref:
            raise RuntimeError("converted tokenizer does not reproduce SentencePiece ids")
    elif ids[:3] != [2, 7071, 235292]:
        raise RuntimeError("tokenizer produced unexpected ids for the check prompt")


def resolve_tokenizer(prefer_hub: bool = True) -> str:
    """Name-or-path to hand to lerobot's tokenizer step."""
    env = os.environ.get("VLA_TOKENIZER_PATH")
    if env:
        return env
    if prefer_hub and hub_tokenizer_accessible():
        return PALIGEMMA_TOKENIZER_REPO
    return str(build_local_tokenizer())


# ------------------------------------------------------------------ pretrained
def download_pretrained(repo_id: str = DEFAULT_PRETRAINED_REPO) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id))


def prepare_pretrained(
    repo_id: str = DEFAULT_PRETRAINED_REPO,
    out_dir: Path | None = None,
    prefer_hub_tokenizer: bool = True,
    force: bool = False,
) -> Path:
    """Local policy directory usable as ``--policy.path`` with a resolvable tokenizer."""
    out_dir = Path(out_dir) if out_dir else PRETRAINED_ROOT / repo_id.replace("/", "__")
    marker = out_dir / "vla_pretrained.json"
    if marker.is_file() and not force:
        return out_dir
    src = download_pretrained(repo_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_name = resolve_tokenizer(prefer_hub=prefer_hub_tokenizer)

    for name in ("config.json", "model.safetensors", "policy_postprocessor.json", "README.md"):
        s = src / name
        d = out_dir / name
        if not s.exists():
            continue
        if d.is_symlink() or d.exists():
            d.unlink()
        os.symlink(s.resolve(), d)

    pre = json.loads((src / "policy_preprocessor.json").read_text())
    patched = 0
    for step in pre.get("steps", []):
        if step.get("registry_name") == "tokenizer_processor":
            step.setdefault("config", {})["tokenizer_name"] = tokenizer_name
            patched += 1
    (out_dir / "policy_preprocessor.json").write_text(json.dumps(pre, indent=2))
    marker.write_text(
        json.dumps(
            {
                "repo_id": repo_id,
                "snapshot": str(src),
                "tokenizer_name": tokenizer_name,
                "tokenizer_steps_patched": patched,
            },
            indent=2,
        )
    )
    return out_dir


def pretrained_info(policy_dir: Path) -> dict:
    marker = Path(policy_dir) / "vla_pretrained.json"
    return json.loads(marker.read_text()) if marker.is_file() else {}


def copy_tree_no_symlinks(src: Path, dst: Path) -> None:
    """Materialize a policy dir (resolving symlinks) e.g. for shipping a checkpoint elsewhere."""
    dst.mkdir(parents=True, exist_ok=True)
    for p in Path(src).iterdir():
        if p.is_dir():
            copy_tree_no_symlinks(p, dst / p.name)
        else:
            shutil.copyfile(p.resolve(), dst / p.name)
