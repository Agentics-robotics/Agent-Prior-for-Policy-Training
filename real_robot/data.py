"""Index-preserving readers and exact sequence publication; no semantic choices."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import io
import json

import numpy as np
from PIL import Image

from appl.io import atomic, digest, read

ROOT = Path(__file__).resolve().parents[1]


def config(path):
    c = read(path)
    if c["schema"] != "real_robot.cut_config.v1":
        raise ValueError("Unsupported configuration")
    if (c["model"], c["reasoning_effort"]) != ("gpt-6-astra", "xhigh"):
        raise ValueError("New real-robot work requires gpt-6-astra/xhigh")
    return c


def clean(value):
    """Only API inspection replaces nonfinite numbers with explicit nulls."""
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return round(float(value), 6) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


CONTRACT = {
    "row_semantics": "N paired observation/action records; segments [start,stop). No timestamp realignment or fabricated N+1 observation.",
    "inspection_precision": "Numerical tool summaries round to 6 decimal places; published NPZ arrays are exact original slices.",
    "available_state": [
        "q",
        "dq",
        "tau_ext",
        "T_base_flange",
        "T_base_ee",
        "gripper_position",
        "gripper_width_m",
    ],
    "available_media": [
        "third RGB",
        "wrist RGB",
        "third_depth raw PNG",
        "wrist_depth raw PNG",
    ],
    "available_metadata": "Per-episode camera calibration/intrinsics, TCP setting, recorded teleoperation configuration and frame timing.",
    "unavailable_labels": [
        "verified target word",
        "letter identities/poses/masks",
        "contact ground truth",
        "verified success labels",
    ],
    "actions": "action_json contains recorded track_twist v,w,spacemouse,gripper,dq,target_pose. Command dq differs from measured observation dq. Null is missing, never an inferred zero.",
    "metadata_control": "metadata declares translation frame=base, rotation_frame=ee, teleop command=joint_velocity at 20 Hz; recording rate=30 Hz. Full execution contract needs recording-controller verification before later deployment.",
    "gripper_feedback": "These recordings have gripper_position=-1 and gripper_width_m=NaN; expose missing feedback, never feed NaN as a valid feature. Images show the installed tool; API must inspect it.",
    "depth": "Original raw depth PNG retained. Do not infer its unit scale, RGB registration or metric reconstruction accuracy from the filename alone.",
    "future_label_boundary": "No windows cross original segment boundaries. Future-derived labels may be training-only; deployment input must be causal.",
}


class Sources:
    def __init__(self, cfg):
        self.cfg = cfg
        self.base = Path(cfg["source_root"]).resolve(strict=True)
        self.episodes = {}
        for tid in cfg["trajectory_ids"]:
            if Path(tid).name != tid:
                raise ValueError("Invalid trajectory identifier")
            ep = self.base / tid
            with np.load(ep / "obs.npz", allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            n = len(arrays["sample_index"])
            frames = read(ep / "frames.json")
            rows = frames["samples"]
            if any(len(v) != n for v in arrays.values()) or len(rows) != n:
                raise ValueError("Inconsistent paired row counts")
            if not np.array_equal(arrays["sample_index"], np.arange(n)):
                raise ValueError("Noncontiguous sample indices")
            for field, key in [
                ("index", "sample_index"),
                ("robot_time", "t"),
                ("recv_time", "recv_time"),
            ]:
                if not np.array_equal(np.array([r[field] for r in rows]), arrays[key]):
                    raise ValueError(
                        f"Frame-to-observation mapping mismatch: {tid}/{field}"
                    )
            self.episodes[tid] = dict(
                path=ep, arrays=arrays, frames=frames, meta=read(ep / "meta.json"), n=n
            )

    def episode(self, tid):
        if tid not in self.episodes:
            raise ValueError("Unknown or unauthorized trajectory")
        return self.episodes[tid]

    def indices(self, tid, indices):
        ep = self.episode(tid)
        if any(type(i) is not int or not 0 <= i < ep["n"] for i in indices):
            raise ValueError("Sample index outside authorized trajectory")
        return ep

    def interval(self, s):
        ep = self.episode(s["trajectory_id"])
        if not 0 <= s["start"] < s["stop"] <= ep["n"]:
            raise ValueError("Invalid half-open paired-sample range")
        return ep

    def media_path(self, tid, camera, i, depth=False):
        ep = self.indices(tid, [i])
        if camera not in ("third", "wrist"):
            raise ValueError("Unsupported camera")
        directory = camera + ("_depth" if depth else "")
        path = ep["path"] / directory / (f"{i:06d}" + (".png" if depth else ".jpg"))
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing original media or unexpected symlink")
        return path

    def catalog(self):
        return [
            dict(
                trajectory_id=tid,
                records=ep["n"],
                duration_s=ep["meta"]["duration_s"],
                cameras=["third", "wrist"],
                recording_complete=ep["meta"]["complete"],
                task_success="not annotated",
            )
            for tid, ep in self.episodes.items()
        ]

    def steps(self, tid, indices):
        ep = self.indices(tid, indices)
        a = ep["arrays"]
        result = []
        for i in indices:
            state = {k: a[k][i] for k in CONTRACT["available_state"]}
            result.append(
                dict(
                    sample_index=i,
                    robot_elapsed_s=round(float(a["t"][i] - a["t"][0]), 6),
                    state=clean(state),
                    action=clean(json.loads(str(a["action_json"][i]))),
                    frame_association=ep["frames"]["samples"][i]["frames"],
                )
            )
        return dict(
            trajectory_id=tid, rows=result, precision=CONTRACT["inspection_precision"]
        )

    def motion(self, tid, start, stop, bins):
        ep = self.interval(dict(trajectory_id=tid, start=start, stop=stop))
        if not 1 <= bins <= 64 or bins > stop - start:
            raise ValueError("Use 1..64 bins, no more than the sample count")
        a = ep["arrays"]
        rows = []
        edges = np.linspace(start, stop, bins + 1, dtype=int)
        for lo, hi in zip(edges[:-1], edges[1:]):
            xyz = a["T_base_ee"][lo:hi, :3, 3]
            dt = np.diff(a["t"][lo:hi])
            speed = (
                np.linalg.norm(np.diff(xyz, axis=0), axis=1) / dt
                if len(dt)
                else np.array([0.0])
            )
            rows.append(
                dict(
                    start=int(lo),
                    stop=int(hi),
                    first_tcp_xyz=clean(xyz[0]),
                    last_tcp_xyz=clean(xyz[-1]),
                    tcp_xyz_min=clean(xyz.min(0)),
                    tcp_xyz_max=clean(xyz.max(0)),
                    tcp_speed_m_s_p50_p95=clean(np.percentile(speed, [50, 95])),
                    measured_dq_abs_max=clean(np.max(np.abs(a["dq"][lo:hi]), axis=0)),
                )
            )
        return dict(
            trajectory_id=tid,
            bins=rows,
            meaning="Deterministic measurements only, not semantic boundaries or contact labels.",
        )

    def image(self, tid, camera, i, resolution, crop):
        path = self.media_path(tid, camera, i)
        with Image.open(path) as original:
            original_size = original.size
            image = original.convert("RGB")
        box = (
            [0, 0, *original_size]
            if crop is None
            else [crop[k] for k in ("left", "top", "right", "bottom")]
        )
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= original_size[0] and 0 <= y0 < y1 <= original_size[1]):
            raise ValueError("Crop coordinates must be within the original image")
        image = image.crop(box)
        if resolution == "preview":
            image.thumbnail(
                (self.cfg["preview_width"], self.cfg["preview_height"]),
                Image.Resampling.LANCZOS,
            )
        elif resolution != "original":
            raise ValueError("Unsupported evidence resolution")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        meta = dict(
            trajectory_id=tid,
            sample_index=i,
            camera=camera,
            original_sha256=digest(path),
            original_size=list(original_size),
            original_crop=box,
            returned_size=list(image.size),
            resolution=resolution,
            coordinate_mapping=dict(
                original_x_offset=x0,
                original_y_offset=y0,
                original_x_per_returned_pixel=(x1 - x0) / image.width,
                original_y_per_returned_pixel=(y1 - y0) / image.height,
            ),
        )
        return buf.getvalue(), meta

    def inventory(self, output):
        """Hash original content; collect no inferred task/skill labels."""
        output = Path(output)
        records = {}
        paths = []
        for tid, ep in self.episodes.items():
            small = {
                name: digest(ep["path"] / name)
                for name in (
                    "obs.npz",
                    "frames.json",
                    "meta.json",
                    "camera_intrinsics.json",
                )
            }
            records[tid] = dict(path=str(ep["path"]), records=ep["n"], files=small)
            for camera in ("third", "wrist"):
                for depth in (False, True):
                    for i in range(ep["n"]):
                        paths.append(self.media_path(tid, camera, i, depth))

        def entry(path):
            return str(path.relative_to(self.base)), dict(
                sha256=digest(path), bytes=path.stat().st_size
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            media = dict(pool.map(entry, paths))
        manifest = dict(
            schema="real_robot.source_manifest.v1",
            source_root=str(self.base),
            episodes=records,
            media=media,
            pairing_verified=True,
            contract=CONTRACT,
            catalog=self.catalog(),
        )
        target = output / "source_manifest.json"
        if target.exists() and read(target) != manifest:
            raise ValueError("Original source identity changed")
        atomic(target, manifest)
        atomic(
            output / "data_audit.json",
            dict(
                passed=True,
                episodes=len(records),
                paired_records=sum(ep["n"] for ep in self.episodes.values()),
                original_media_files=len(media),
                original_media_bytes=sum(x["bytes"] for x in media.values()),
                source_manifest_sha256=digest(target),
                timestamp_realignment=False,
                original_data_mutations=0,
                contract=CONTRACT,
            ),
        )
        return manifest

    def verify_metadata(self, manifest):
        for tid, record in manifest["episodes"].items():
            for name, expected in record["files"].items():
                if digest(self.episode(tid)["path"] / name) != expected:
                    raise ValueError("Source metadata or arrays changed")

    def verify_media(self, manifest, paths=None):
        names = list(manifest["media"]) if paths is None else paths

        def check(name):
            if digest(self.base / name) != manifest["media"][name]["sha256"]:
                raise ValueError("Original media changed: " + name)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(check, names))

    def publish_segment(self, segment, target, source_manifest):
        ep = self.interval(segment)
        lo, hi = segment["start"], segment["stop"]
        target = Path(target)
        target.mkdir(parents=True, exist_ok=False)
        arrays = {k: v[lo:hi] for k, v in ep["arrays"].items()}
        np.savez_compressed(target / "records.npz", **arrays)
        with np.load(target / "records.npz", allow_pickle=False) as z:
            for k, original in arrays.items():
                equal = (
                    np.array_equal(z[k], original, equal_nan=True)
                    if original.dtype.kind == "f"
                    else np.array_equal(z[k], original)
                )
                if not equal:
                    raise ValueError("Materialized slice differs from original")
        samples = []
        for i in range(lo, hi):
            images = {}
            for camera in ("third", "wrist"):
                for depth in (False, True):
                    path = self.media_path(segment["trajectory_id"], camera, i, depth)
                    rel = str(path.relative_to(self.base))
                    images[camera + ("_depth" if depth else "")] = dict(
                        source_relative_path=rel,
                        sha256=source_manifest["media"][rel]["sha256"],
                    )
            samples.append(dict(source_sample_index=i, media=images))
        atomic(
            target / "samples.json",
            dict(
                source_root=str(self.base),
                samples=samples,
                source_frame_records=ep["frames"]["samples"][lo:hi],
                successor_source_index=hi if hi < ep["n"] else None,
                continuity="Only within this source segment. No cross-segment windows.",
            ),
        )
        return dict(
            **segment,
            records=hi - lo,
            records_sha256=digest(target / "records.npz"),
            samples_sha256=digest(target / "samples.json"),
            directory=str(target.relative_to(ROOT)),
        )
