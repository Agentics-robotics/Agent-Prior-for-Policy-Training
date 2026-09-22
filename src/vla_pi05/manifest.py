"""Segment manifest: the list of (language instruction, trajectory segment) pairs to train on.

This is the interface between the raw recordings and the dataset converter, and the
file to regenerate once the recordings are cut and labelled. A segment references a
raw episode folder plus a half-open frame range ``[start, end)`` at the raw rate.

Until language labels exist, ``build_manifest`` uses one segment per raw episode and a
placeholder instruction (the teleop ``notes`` string, else the task folder name).
The placeholder is recorded in ``instruction_source`` so nobody mistakes it for a label.

Manifest JSON::

    {
      "version": 1,
      "created": "2026-09-20T12:00:00",
      "raw_root": "/home/storage/tianrunhu/real_robot_data",
      "instruction_source": "notes",
      "segments": [
        {"id": "flip_egg/episode_2026091916095801#0", "episode": "flip_egg/episode_2026091916095801",
         "start": 0, "end": 1811, "instruction": "flip the fried egg", "split": "train", "task": "flip_egg"}
      ]
    }

A CSV with columns ``episode,start,end,instruction[,split]`` is also accepted (``from_csv``),
which is the easiest thing for a labelling tool to emit.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .paths import RAW_DATA_ROOT
from .raw_episode import RawEpisode, discover_episodes

SPLITS = ("train", "val")


@dataclass
class Segment:
    episode: str  # path relative to raw_root, e.g. "flip_egg/episode_2026091916095801"
    start: int = 0
    end: int | None = None  # exclusive; None = episode end
    instruction: str = ""
    split: str = "train"
    task: str = ""
    id: str = ""
    meta: dict = field(default_factory=dict)  # free-form provenance (goal box, source cut id, ...)

    def __post_init__(self) -> None:
        self.episode = self.episode.strip("/")
        if not self.task:
            self.task = self.episode.split("/")[0]
        if not self.id:
            self.id = f"{self.episode}#{self.start}"
        if self.split not in SPLITS:
            raise ValueError(f"segment {self.id}: split must be one of {SPLITS}, got {self.split!r}")
        if self.start < 0 or (self.end is not None and self.end <= self.start):
            raise ValueError(f"segment {self.id}: invalid range [{self.start}, {self.end})")

    def resolve(self, raw_root: str | Path, n_steps: int) -> tuple[int, int]:
        end = n_steps if self.end is None else min(self.end, n_steps)
        if self.start >= end:
            raise ValueError(f"segment {self.id}: range [{self.start}, {end}) is empty for {n_steps} steps")
        return self.start, end

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class Manifest:
    segments: list[Segment]
    raw_root: str = str(RAW_DATA_ROOT)
    instruction_source: str = "manual"
    version: int = 1
    created: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))
    notes: str = ""

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "created": self.created,
            "raw_root": self.raw_root,
            "instruction_source": self.instruction_source,
            "notes": self.notes,
            "segments": [s.to_dict() for s in self.segments],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        path = Path(path)
        if path.suffix.lower() == ".csv":
            return cls.from_csv(path)
        d = json.loads(path.read_text())
        segs = [Segment(**{k: v for k, v in s.items() if k in Segment.__dataclass_fields__}) for s in d["segments"]]
        return cls(
            segments=segs,
            raw_root=d.get("raw_root", str(RAW_DATA_ROOT)),
            instruction_source=d.get("instruction_source", "manual"),
            version=int(d.get("version", 1)),
            created=d.get("created", ""),
            notes=d.get("notes", ""),
        )

    @classmethod
    def from_csv(cls, path: str | Path, raw_root: str | Path = RAW_DATA_ROOT) -> "Manifest":
        segs = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                end = row.get("end", "")
                segs.append(
                    Segment(
                        episode=row["episode"],
                        start=int(row.get("start") or 0),
                        end=int(end) if end not in ("", None) else None,
                        instruction=(row.get("instruction") or "").strip(),
                        split=(row.get("split") or "train").strip(),
                    )
                )
        return cls(segments=segs, raw_root=str(raw_root), instruction_source="csv")

    # --------------------------------------------------------------- helpers
    def digest(self) -> str:
        payload = json.dumps([s.to_dict() for s in self.segments], sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]

    def split(self, name: str) -> list[Segment]:
        return [s for s in self.segments if s.split == name]

    def tasks(self) -> list[str]:
        return sorted({s.task for s in self.segments})

    def stats(self) -> dict:
        per = defaultdict(lambda: {"train": 0, "val": 0})
        for s in self.segments:
            per[s.task][s.split] += 1
        return {"n_segments": len(self.segments), "per_task": dict(per), "digest": self.digest()}

    def validate(self, check_files: bool = True) -> list[str]:
        """Return a list of problems (empty = ok). Missing instructions are reported."""
        problems = []
        seen = set()
        for s in self.segments:
            if s.id in seen:
                problems.append(f"duplicate segment id {s.id}")
            seen.add(s.id)
            if not s.instruction:
                problems.append(f"{s.id}: empty instruction")
            if check_files:
                ep_dir = Path(self.raw_root) / s.episode
                if not (ep_dir / "meta.json").is_file():
                    problems.append(f"{s.id}: episode folder missing ({ep_dir})")
        return problems


def assign_splits(episodes: list[Path], val_fraction: float, seed: int) -> dict[Path, str]:
    """Deterministic per-task held-out episodes. At least one val episode per task when possible."""
    by_task: dict[str, list[Path]] = defaultdict(list)
    for ep in episodes:
        by_task[ep.parent.name].append(ep)
    out: dict[Path, str] = {}
    rng = random.Random(seed)
    for task, eps in sorted(by_task.items()):
        eps = sorted(eps)
        n_val = int(round(len(eps) * val_fraction)) if val_fraction > 0 else 0
        if val_fraction > 0 and len(eps) >= 3:
            n_val = max(n_val, 1)
        n_val = min(n_val, max(len(eps) - 1, 0))
        idx = list(range(len(eps)))
        rng.shuffle(idx)
        val_idx = set(idx[:n_val])
        for i, ep in enumerate(eps):
            out[ep] = "val" if i in val_idx else "train"
    return out


def build_manifest(
    raw_root: str | Path = RAW_DATA_ROOT,
    tasks: list[str] | None = None,
    val_fraction: float = 0.1,
    seed: int = 0,
    instruction_source: str = "notes",
    instruction_map: dict[str, str] | None = None,
    include_incomplete: bool = False,
) -> Manifest:
    """One segment per raw episode with a placeholder instruction.

    instruction_source:
        "notes" -> teleop notes string from meta.json (falls back to task name)
        "task"  -> task folder name with underscores replaced by spaces
    instruction_map: optional {task: instruction} override applied last.
    """
    raw_root = Path(raw_root)
    episodes = discover_episodes(raw_root, tasks)
    splits = assign_splits(episodes, val_fraction, seed)
    segments = []
    skipped = []
    for ep_path in episodes:
        ep = RawEpisode.load(ep_path)
        if not ep.complete and not include_incomplete:
            skipped.append(ep.relpath)
            continue
        task_text = ep.task.replace("_", " ")
        if instruction_source == "notes":
            instruction = ep.notes or task_text
        elif instruction_source == "task":
            instruction = task_text
        else:
            raise ValueError(f"unknown instruction_source {instruction_source!r}")
        if instruction_map and ep.task in instruction_map:
            instruction = instruction_map[ep.task]
        segments.append(
            Segment(
                episode=str(ep_path.relative_to(raw_root)),
                start=0,
                end=ep.n_steps,
                instruction=instruction,
                split=splits[ep_path],
                task=ep.task,
            )
        )
    notes = "placeholder instructions; replace with cut+labelled segments when available"
    if skipped:
        notes += f"; skipped incomplete episodes: {skipped}"
    return Manifest(segments=segments, raw_root=str(raw_root), instruction_source=instruction_source, notes=notes)
