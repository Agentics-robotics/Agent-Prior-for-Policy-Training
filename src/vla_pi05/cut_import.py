"""Import the decoupled/regrouped push cuts (``cut_v3``) into a training manifest with prompts.

The cut dataset (schema ``real_robot.cut_dataset.v2``) describes 22 segments over the two raw
``push_letters`` episodes. Each segment fixes ONE selected wooden piece and ONE hindsight image
goal (``goal box [u0,v0,u1,v1]`` in the 1280x720 third camera at the segment's last frame), and
lists 1-second windows at the boundaries that must not receive action loss.

Prompt design. pi0.5 conditions on a short natural-language string, so the per-segment markdown
is compressed to the two conditions the policy actually needs:

    <verb> the <piece> <target phrase>, goal at x<gx> y<gy>

* piece   -- which physical instance, by shape (the letters are visual mnemonics for shapes)
             plus a location cue when two pieces share a shape (the I and H crossbar pieces).
* target  -- where it goes, relative to the arrangement (word row, above the O, between R and D ...).
             ``stage``/``align`` verbs distinguish temporary staging from final alignment.
* goal    -- the numeric deployment condition: goal-box centre in percent of image width/height.
             Drop with ``with_goal_coords=False`` if the caller cannot supply an image goal.

Exclusion windows are handled by trimming (or splitting) the supervised interval so no chunk starts
inside them; context loss is at most one second at the segment boundaries.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .manifest import Manifest, Segment
from .paths import RAW_DATA_ROOT

DEFAULT_CUT_ROOT = Path("/home/storage/oscar/cut_v3")
IMAGE_W, IMAGE_H = 1280, 720

# segment_id -> (verb, piece, target phrase). Hand-written from heuristic.md / dataset.json
# (objective + label_derivation), kept short and consistent across the two episodes.
PROMPTS: dict[str, tuple[str, str, str]] = {
    # episode A: episode_2026091922002701
    "a_w": ("push", "the W-shaped zigzag piece", "from the upper middle to the lower-right word row"),
    "a_i_stage1": ("stage", "the I-shaped bar piece next to the O", "a short way aside"),
    "a_o": ("push", "the O-shaped ring piece", "to the right and set it above the W"),
    "a_r_stage": ("stage", "the R-shaped piece", "upward near the H"),
    "a_i_stage2": ("stage", "the I-shaped bar piece", "further up and left to clear space around the H"),
    "a_h_stage": ("stage", "the H-shaped piece", "down at the lower left"),
    "a_d_stage": ("stage", "the D-shaped piece", "up at the upper right"),
    "a_r": ("push", "the staged R-shaped piece", "into the row above the O"),
    "a_d_align": ("align", "the D-shaped piece", "at the top of the word column"),
    "a_l": ("push", "the L-shaped piece", "into the gap between the R and the D"),
    "a_i_align": ("push", "the staged I-shaped bar piece", "to the left of the R and L"),
    "a_h_align": ("push", "the staged H-shaped piece", "up and align it next to the I"),
    # episode B: episode_2026091922062101
    "b_w": ("push", "the W-shaped zigzag piece", "from the bottom to the lower-right word row"),
    "b_i_stage": ("stage", "the central I-shaped bar piece", "a short way to the left"),
    "b_d_stage": ("stage", "the D-shaped piece", "from the lower left up to the upper right"),
    "b_o": ("push", "the O-shaped ring piece", "down and set it above the W"),
    "b_r": ("push", "the R-shaped piece", "into the row above the O"),
    "b_d_align": ("align", "the D-shaped piece", "at the top of the word column"),
    "b_l": ("push", "the L-shaped piece", "into the gap between the R and the D"),
    "b_h_stage": ("stage", "the upper-left H-shaped piece", "down at the lower left"),
    "b_i_align": ("push", "the staged I-shaped bar piece", "up to the left of the R and L"),
    "b_h_align": ("push", "the staged H-shaped piece", "right and up next to the I"),
}

DEFAULT_VAL_IDS = ("b_d_stage", "b_i_align")


def load_cut_segments(cut_root: str | Path = DEFAULT_CUT_ROOT) -> list[dict]:
    cut_root = Path(cut_root)
    ds_files = sorted(cut_root.glob("datasets/*/dataset.json"))
    if not ds_files:
        raise FileNotFoundError(f"no datasets/*/dataset.json under {cut_root}")
    segments = []
    for f in ds_files:
        d = json.loads(f.read_text())
        for s in d["segments"]:
            s = dict(s)
            s["_dataset"] = d.get("skill_id", f.parent.name)
            segments.append(s)
    return segments


def goal_box(seg: dict) -> list[int] | None:
    m = re.search(r"goal box \[([\d,\s]+)\]", seg.get("label_derivation", ""))
    if not m:
        return None
    vals = [int(v) for v in m.group(1).replace(" ", "").split(",")]
    return vals if len(vals) == 4 else None


def goal_center_pct(box: list[int]) -> tuple[int, int]:
    u = (box[0] + box[2]) / 2 / IMAGE_W
    v = (box[1] + box[3]) / 2 / IMAGE_H
    return int(round(100 * u)), int(round(100 * v))


def build_instruction(seg: dict, with_goal_coords: bool = True) -> str:
    sid = seg["segment_id"]
    if sid not in PROMPTS:
        raise KeyError(f"no prompt entry for cut segment {sid!r}; add it to cut_import.PROMPTS")
    verb, piece, target = PROMPTS[sid]
    text = f"{verb} {piece} {target}"
    box = goal_box(seg)
    if with_goal_coords and box:
        gx, gy = goal_center_pct(box)
        text += f", goal at x{gx} y{gy}"
    return text


def supervised_intervals(seg: dict, trim_exclusions: bool = True) -> list[tuple[int, int]]:
    """[start, end) ranges that keep action loss, after removing the exclusion windows."""
    start, stop = int(seg.get("supervised_start", seg["start"])), int(seg.get("supervised_stop", seg["stop"]))
    if not trim_exclusions:
        return [(start, stop)]
    keep = [(start, stop)]
    for ex in seg.get("supervision_exclusions", []):
        a, b = int(ex["start"]), int(ex["stop"])
        nxt = []
        for s, e in keep:
            if b <= s or a >= e:
                nxt.append((s, e))
                continue
            if a > s:
                nxt.append((s, a))
            if b < e:
                nxt.append((b, e))
        keep = nxt
    return [(s, e) for s, e in keep if e - s > 0]


def manifest_from_cut(
    cut_root: str | Path = DEFAULT_CUT_ROOT,
    raw_task: str = "push_letters",
    raw_root: str | Path = RAW_DATA_ROOT,
    val_ids: tuple[str, ...] = DEFAULT_VAL_IDS,
    with_goal_coords: bool = True,
    trim_exclusions: bool = True,
    min_frames: int = 30,
) -> Manifest:
    cut_root = Path(cut_root)
    segs = load_cut_segments(cut_root)
    out: list[Segment] = []
    for seg in segs:
        instruction = build_instruction(seg, with_goal_coords)
        split = "val" if seg["segment_id"] in set(val_ids) else "train"
        parts = supervised_intervals(seg, trim_exclusions)
        for k, (s, e) in enumerate(parts):
            if e - s < min_frames:
                continue
            out.append(
                Segment(
                    episode=f"{raw_task}/{seg['trajectory_id']}",
                    start=s,
                    end=e,
                    instruction=instruction,
                    split=split,
                    task=raw_task,
                    id=seg["segment_id"] if len(parts) == 1 else f"{seg['segment_id']}/{k}",
                    meta={
                        "cut_segment_id": seg["segment_id"],
                        "cut_interval": [int(seg["start"]), int(seg["stop"])],
                        "goal_box_1280x720": goal_box(seg),
                        "objective": seg.get("objective", ""),
                        "training_condition": seg.get("training_condition", ""),
                        "cut_dataset": seg.get("_dataset"),
                    },
                )
            )
    m = Manifest(segments=out, raw_root=str(raw_root), instruction_source=f"cut:{cut_root}")
    m.notes = (
        f"{len(segs)} cut segments from {cut_root} -> {len(out)} supervised intervals; "
        f"prompts from vla_pi05.cut_import.PROMPTS; goal coords={'on' if with_goal_coords else 'off'}; "
        f"val={list(val_ids)}"
    )
    return m
