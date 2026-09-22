"""Unit tests for the vla_pi05 pipeline pieces that do not need a GPU or the pi0.5 weights.

Run with:  pixi run -e vla vla-test
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# These modules live in the separate `vla` Pixi environment; skip when collected from the default one.
pytest.importorskip("msgpack")

from vla_pi05 import msgpack_numpy
from vla_pi05.action_space import (
    ActionSpaceConfig,
    actions_to_targets,
    build_actions,
    build_state,
    resample_indices,
    segment_arrays,
)
from vla_pi05.convert_dataset import ConvertConfig, dataset_features
from vla_pi05.manifest import Manifest, Segment, assign_splits, build_manifest
from vla_pi05.raw_episode import RawEpisode, discover_episodes


# ------------------------------------------------------------------ fixtures
def make_raw_episode(root: Path, task: str, name: str, n: int, rate_hz: float, notes: str = "do the thing") -> Path:
    from PIL import Image

    ep = root / task / name
    for cam in ("third", "wrist"):
        (ep / cam).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            Image.fromarray(np.full((36, 64, 3), i % 255, dtype=np.uint8)).save(ep / cam / f"{i:06d}.jpg")
    t = 1000.0 + np.arange(n) / rate_hz
    q = np.stack([np.linspace(0, 1, n) + j for j in range(7)], axis=1)
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, 0, 3] = np.linspace(0.3, 0.5, n)
    width = np.linspace(0.02, 0.14, n)
    aj = np.array([json.dumps({"type": "track_twist", "gripper": (i / n) if i % 2 else None}) for i in range(n)])
    np.savez(
        ep / "obs.npz",
        t=t,
        q=q,
        dq=np.zeros((n, 7)),
        T_base_flange=T,
        T_base_ee=T,
        gripper_position=np.zeros(n),
        gripper_width_m=width,
        action_json=aj,
    )
    (ep / "meta.json").write_text(
        json.dumps({"notes": notes, "rate_hz": rate_hz, "n_steps": n, "cameras": ["wrist", "third"], "gripper_max_width_m": 0.14, "complete": True})
    )
    return ep


@pytest.fixture
def raw_root(tmp_path: Path) -> Path:
    make_raw_episode(tmp_path, "flip_egg", "episode_0001", 90, 30.0, "flip the fried egg")
    make_raw_episode(tmp_path, "flip_egg", "episode_0002", 60, 30.0, "flip the fried egg")
    make_raw_episode(tmp_path, "flip_egg", "episode_0003", 75, 30.0, "flip the fried egg")
    make_raw_episode(tmp_path, "scoop", "episode_0001", 40, 10.0, "scoop the particles into the bowl")
    return tmp_path


# ------------------------------------------------------------------ raw episode
def test_raw_episode_loads_and_checks(raw_root):
    eps = discover_episodes(raw_root)
    assert [e.name for e in eps] == ["episode_0001", "episode_0002", "episode_0003", "episode_0001"]
    ep = RawEpisode.load(eps[0])
    assert ep.n_steps == 90 and ep.rate_hz == 30.0 and ep.task == "flip_egg"
    assert ep.notes == "flip the fried egg"
    img = ep.load_image("third", 5, (18, 32))
    assert img.shape == (18, 32, 3) and img.dtype == np.uint8
    assert np.isnan(ep.gripper_cmd[0]) and not np.isnan(ep.gripper_cmd[1])
    assert abs(ep.estimated_rate_hz - 30.0) < 1e-6


def test_raw_episode_rejects_missing_images(raw_root):
    ep_dir = discover_episodes(raw_root)[0]
    (ep_dir / "third" / "000089.jpg").unlink()
    with pytest.raises(FileNotFoundError):
        RawEpisode.load(ep_dir)


# ------------------------------------------------------------------ resampling / action space
def test_resample_30hz_to_10hz():
    t = np.arange(90) / 30.0
    idx = resample_indices(t, 10)
    assert idx[0] == 0 and len(idx) == 30
    assert np.all(np.diff(idx) == 3)


def test_resample_10hz_identity_and_ranges():
    t = np.arange(40) / 10.0
    idx = resample_indices(t, 10)
    assert np.array_equal(idx, np.arange(40))
    idx = resample_indices(t, 10, start=10, end=20)
    assert idx[0] == 10 and idx[-1] == 19
    with pytest.raises(ValueError):
        resample_indices(t, 10, start=5, end=5)


def test_resample_never_upsamples():
    t = np.arange(20) / 5.0  # 5 Hz source
    idx = resample_indices(t, 10)
    assert len(np.unique(idx)) == len(idx) <= 20


def test_state_action_layouts(raw_root):
    ep = RawEpisode.load(raw_root / "flip_egg" / "episode_0001")
    for name, dim in (("joint_abs", 8), ("joint_delta", 8), ("ee_abs", 10)):
        cfg = ActionSpaceConfig(name=name, target_fps=10)
        assert cfg.state_dim == dim and cfg.action_dim == dim
        arrs = segment_arrays(ep, 0, ep.n_steps, cfg)
        assert arrs["state"].shape == (30, dim) and arrs["action"].shape == (30, dim)
        assert arrs["state"].dtype == np.float32
        assert 0.0 <= arrs["state"][:, -1].min() and arrs["state"][:, -1].max() <= 1.0
        # action[k] targets state[k+1]; last holds
        targets = actions_to_targets(arrs["action"], arrs["state"][0], cfg)
        if name == "joint_delta":
            recon = arrs["state"][0, :-1] + np.cumsum(arrs["action"][:, :-1], axis=0)
            np.testing.assert_allclose(recon, targets[:, :-1], atol=1e-5)
            np.testing.assert_allclose(targets[:-1, :-1], arrs["state"][1:, :-1], atol=1e-4)
        else:
            np.testing.assert_array_equal(targets[:-1], arrs["state"][1:])
            np.testing.assert_array_equal(targets[-1], arrs["state"][-1])


def test_gripper_cmd_source_fills_gaps(raw_root):
    ep = RawEpisode.load(raw_root / "flip_egg" / "episode_0001")
    idx = np.arange(ep.n_steps)
    s = build_state(ep, idx, ActionSpaceConfig(gripper_source="cmd"))
    assert not np.isnan(s).any()


def test_action_space_config_roundtrip():
    cfg = ActionSpaceConfig(name="ee_abs", target_fps=10)
    d = cfg.to_dict()
    assert d["state_names"][-1] == "gripper" and len(d["action_names"]) == 10
    assert ActionSpaceConfig.from_dict(d) == cfg
    with pytest.raises(ValueError):
        ActionSpaceConfig(name="nope")


# ------------------------------------------------------------------ manifest
def test_build_manifest_and_roundtrip(raw_root, tmp_path):
    m = build_manifest(raw_root, val_fraction=0.34, seed=0)
    st = m.stats()
    assert st["n_segments"] == 4
    assert st["per_task"]["flip_egg"] == {"train": 2, "val": 1}
    assert st["per_task"]["scoop"] == {"train": 1, "val": 0}  # too few episodes to hold one out
    assert {s.instruction for s in m.segments} == {"flip the fried egg", "scoop the particles into the bowl"}
    p = m.save(tmp_path / "m.json")
    m2 = Manifest.load(p)
    assert m2.digest() == m.digest() and m2.instruction_source == "notes"
    assert m2.validate(check_files=True) == []


def test_manifest_csv_and_validation(raw_root, tmp_path):
    csv = tmp_path / "cuts.csv"
    csv.write_text(
        "episode,start,end,instruction,split\n"
        "flip_egg/episode_0001,0,45,move to the egg,train\n"
        "flip_egg/episode_0001,45,90,flip the egg,train\n"
        "flip_egg/episode_0002,0,,,val\n"
    )
    m = Manifest.from_csv(csv, raw_root=raw_root)
    assert [s.id for s in m.segments] == ["flip_egg/episode_0001#0", "flip_egg/episode_0001#45", "flip_egg/episode_0002#0"]
    assert m.segments[2].end is None
    problems = m.validate(check_files=True)
    assert problems == ["flip_egg/episode_0002#0: empty instruction"]
    with pytest.raises(ValueError):
        Segment(episode="x/y", start=10, end=5)


def test_assign_splits_deterministic(raw_root):
    eps = discover_episodes(raw_root)
    a = assign_splits(eps, 0.34, seed=3)
    b = assign_splits(eps, 0.34, seed=3)
    assert a == b and list(a.values()).count("val") == 1


def test_instruction_map_override(raw_root):
    m = build_manifest(raw_root, instruction_source="task", instruction_map={"scoop": "scoop it"})
    by_task = {s.task: s.instruction for s in m.segments}
    assert by_task == {"flip_egg": "flip egg", "scoop": "scoop it"}


# ------------------------------------------------------------------ dataset features
def test_dataset_features_follow_pi05_naming():
    cfg = ConvertConfig(name="x", image_hw=(144, 256))
    f = dataset_features(cfg)
    assert list(f)[:2] == ["observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb"]
    assert f["observation.images.base_0_rgb"]["shape"] == (144, 256, 3)
    assert f["observation.state"]["names"][-1] == "gripper" and f["action"]["shape"] == (8,)


# ------------------------------------------------------------------ wire codec
def test_msgpack_numpy_roundtrip():
    obs = {"images": {"base_0_rgb": np.random.randint(0, 255, (144, 256, 3), dtype=np.uint8)}, "state": np.arange(8, dtype=np.float32), "instruction": "flip", "n": np.float64(1.5)}
    back = msgpack_numpy.unpackb(msgpack_numpy.packb(obs))
    np.testing.assert_array_equal(back["images"]["base_0_rgb"], obs["images"]["base_0_rgb"])
    assert back["state"].dtype == np.float32 and back["instruction"] == "flip" and back["n"] == 1.5
    with pytest.raises(ValueError):
        msgpack_numpy.packb(np.array([object()]))


# ------------------------------------------------------------------ train launcher
def test_build_command_composes_lerobot_args(tmp_path, monkeypatch):
    from vla_pi05 import train as tr

    ds = tmp_path / "ds_train"
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "info.json").write_text(json.dumps({"total_frames": 100, "total_episodes": 2, "fps": 10}))
    (ds / "meta" / "vla_conversion.json").write_text(json.dumps({"repo_id": "local/ds_train", "manifest": {"digest": "abc"}, "convert_config": {}}))
    pre = tmp_path / "pretrained"
    pre.mkdir()
    (pre / "config.json").write_text("{}")
    monkeypatch.setattr(tr, "RUNS_ROOT", tmp_path / "runs")
    args = tr.TrainArgs(dataset_dir=ds, pretrained=str(pre), preset="expert_only", gpus="4,5,6,7", run_name="r1")
    argv, rec, run_dir = tr.build_command(args)
    cmd = " ".join(argv)
    assert "--num_processes=4" in cmd and "accelerate_ddp.yaml" in cmd
    assert "--policy.train_expert_only=true" in cmd and "--policy.dtype=bfloat16" in cmd
    assert "--policy.push_to_hub=false" in cmd and "--policy.device=cuda" in cmd
    assert rec["effective_batch_size"] == 32 and run_dir == tmp_path / "runs" / "r1"
    smoke = tr.TrainArgs(dataset_dir=ds, pretrained=str(pre), smoke=True, gpus="4,5,6,7")
    argv, rec, _ = tr.build_command(smoke)
    assert "--num_processes=1" in " ".join(argv) and rec["steps"] == 10 and rec["gpus"] == ["4"]
    full = tr.TrainArgs(dataset_dir=ds, pretrained=str(pre), preset="full", gpus="4,5,6,7")
    argv, _, _ = tr.build_command(full)
    assert "accelerate_fsdp.yaml" in " ".join(argv)
    lora = tr.TrainArgs(dataset_dir=ds, pretrained=str(pre), preset="lora", gpus="4,5")
    argv, _, _ = tr.build_command(lora)
    assert "--peft.method_type=LORA" in " ".join(argv) and "--peft.r=32" in " ".join(argv)


# ------------------------------------------------------------------ cut import
def test_cut_import_intervals_and_prompt():
    from vla_pi05.cut_import import build_instruction, goal_box, goal_center_pct, supervised_intervals

    seg = {
        "segment_id": "a_i_stage1",
        "trajectory_id": "episode_x",
        "start": 1720,
        "stop": 1920,
        "supervised_start": 1720,
        "supervised_stop": 1920,
        "label_derivation": "G at 1919, goal box [275,310,465,432]. Select the lower crossbar piece.",
        "supervision_exclusions": [{"start": 1720, "stop": 1750}, {"start": 1890, "stop": 1920}],
    }
    assert goal_box(seg) == [275, 310, 465, 432]
    assert goal_center_pct([275, 310, 465, 432]) == (29, 52)
    assert supervised_intervals(seg) == [(1750, 1890)]
    assert supervised_intervals(seg, trim_exclusions=False) == [(1720, 1920)]
    mid = dict(seg, supervision_exclusions=[{"start": 1800, "stop": 1830}])
    assert supervised_intervals(mid) == [(1720, 1800), (1830, 1920)]
    text = build_instruction(seg)
    assert text == "stage the I-shaped bar piece next to the O a short way aside, goal at x29 y52"
    assert build_instruction(seg, with_goal_coords=False).endswith("aside")
    assert len(text.split()) < 20
    with pytest.raises(KeyError):
        build_instruction(dict(seg, segment_id="unknown"))


def test_cut_prompt_table_is_complete_and_short():
    from vla_pi05.cut_import import PROMPTS

    assert len(PROMPTS) == 22
    for sid, (verb, piece, target) in PROMPTS.items():
        assert verb in ("push", "stage", "align"), sid
        assert len(f"{verb} {piece} {target}".split()) <= 18, sid


def test_gripper_missing_signal_becomes_constant(raw_root):
    from vla_pi05.action_space import NO_GRIPPER_VALUE, gripper_normalized

    ep = RawEpisode.load(raw_root / "flip_egg" / "episode_0001")
    ep.obs["gripper_width_m"] = np.full(ep.n_steps, np.nan)
    ep.obs["action_json"] = np.array([json.dumps({"gripper": None})] * ep.n_steps)
    ep.__dict__.pop("gripper_cmd", None)  # reset cached property
    idx = np.arange(ep.n_steps)
    for src in ("width", "cmd"):
        g = gripper_normalized(ep, idx, src)
        assert np.isfinite(g).all() and np.all(g == NO_GRIPPER_VALUE)
