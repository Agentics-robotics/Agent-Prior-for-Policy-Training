"""Read-only dataset capabilities; semantic conversions belong to Runtime API."""

from pathlib import Path
import copy
import json

import numpy as np
from PIL import Image

from appl.io import atomic

_context = None


def configure(sources, segments, output, assets, source=None):
    global _context
    _context = dict(
        sources=sources, segments=segments, output=Path(output), assets=Path(assets),
        source=Path(source) if source is not None else None,
    )


def segments():
    return copy.deepcopy(_context["segments"])


def records(trajectory_id):
    return _context["sources"].episode(trajectory_id)["arrays"]


def rgb(trajectory_id, sample_index, camera):
    path = _context["sources"].media_path(trajectory_id, camera, int(sample_index))
    with Image.open(path) as value:
        return np.array(value.convert("RGB"))


def depth(trajectory_id, sample_index, camera):
    path = _context["sources"].media_path(trajectory_id, camera, int(sample_index), depth=True)
    with Image.open(path) as value:
        return np.array(value)


def camera_intrinsics(trajectory_id):
    path = _context["sources"].episode(trajectory_id)["path"] / "camera_intrinsics.json"
    return json.loads(path.read_text())


def metadata(trajectory_id):
    return copy.deepcopy(_context["sources"].episode(trajectory_id)["meta"])


def progress(payload):
    atomic(_context["output"] / "preparation_progress.json", payload)
    print(json.dumps(payload, allow_nan=False), flush=True)


def asset_path(name):
    if Path(name).name != name:
        raise ValueError("Asset name must be flat")
    path = _context["assets"] / name
    if not path.is_file() or path.is_symlink():
        raise ValueError("Asset was not explicitly downloaded")
    return str(path)


def model_path(name):
    if Path(name).name != name:
        raise ValueError("Model name must be flat")
    path = _context["assets"] / name
    if not path.is_dir() or path.is_symlink() or not (path / "asset_manifest.json").exists():
        raise ValueError("Model snapshot was not explicitly downloaded")
    return str(path)


def source_json(name):
    if Path(name).name != name or not name.endswith(".json"):
        raise ValueError("Source resource must be a flat JSON filename")
    path = _context["source"] / name
    if path.is_symlink():
        raise ValueError("No resource symlinks")
    return json.loads(path.read_text())


def write_report(name, payload):
    if Path(name).name != name or not name.endswith(".json"):
        raise ValueError("Report name must be flat JSON")
    atomic(_context["output"] / "evidence" / name, payload)


def write_image(name, array):
    if Path(name).name != name or not name.endswith(".png"):
        raise ValueError("Evidence image name must be flat PNG")
    if not isinstance(array, np.ndarray) or array.dtype != np.uint8 or array.ndim not in (2, 3):
        raise ValueError("Evidence image must be uint8 pixels")
    path = _context["output"] / "evidence" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
