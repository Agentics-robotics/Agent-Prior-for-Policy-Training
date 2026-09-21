"""Read-only dataset capabilities; semantic conversions belong to Runtime API."""

from pathlib import Path
import copy
import json

import numpy as np
from PIL import Image

from appl.io import atomic

_context = None


def configure(sources, segments, output, assets):
    global _context
    _context = dict(
        sources=sources, segments=segments, output=Path(output), assets=Path(assets)
    )


def segments():
    return copy.deepcopy(_context["segments"])


def records(trajectory_id):
    return _context["sources"].episode(trajectory_id)["arrays"]


def rgb(trajectory_id, sample_index, camera):
    path = _context["sources"].media_path(trajectory_id, camera, int(sample_index))
    with Image.open(path) as value:
        return np.array(value.convert("RGB"))


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
