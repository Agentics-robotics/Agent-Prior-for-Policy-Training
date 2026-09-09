"""Scalar goal-observable MetaWorld Task adapter, pinned to audited v3 source.

We use the public Task/set_task interface with the same payload as MetaWorld's
_encode_task. A canonical simulator/model reset precedes native reset so reused
instances cannot retain a preceding episode's mutable model placement. Targets,
reward caches, and packed history are then initialized by native reset_model.
No post-reset object relocation or action rescaling is performed.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import pickle
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .representation import OBS_DIM
from .utils import ROOT, atomic_json, sha256

METAWORLD_COMMIT = "6e01ad7e2ffb2302e4dca04f796fcd8837df8540"
SCHEMA_VERSION = "relative-dp-round1-v1"
MAX_EPISODE_STEPS = 500
TASK_NAMES = {"drawer": "drawer-open-v3", "door": "door-open-v3"}
REPLAY_ATOL = 1e-9


def task_key(task_name: str) -> str:
    if task_name in TASK_NAMES:
        return task_name
    return {value: key for key, value in TASK_NAMES.items()}[task_name]


def make_env(task_name: str, render_mode: str | None = None):
    from metaworld.envs.sawyer_drawer_open_v3 import SawyerDrawerOpenEnvV3
    from metaworld.envs.sawyer_door_v3 import SawyerDoorEnvV3

    key = task_key(task_name)
    cls = {"drawer": SawyerDrawerOpenEnvV3, "door": SawyerDoorEnvV3}[key]
    env = cls(render_mode=render_mode, camera_name="corner2", width=480, height=480)
    env._round1_key = key
    env._round1_initial_body_pos = env.model.body_pos.copy()
    env._round1_initial_site_pos = env.model.site_pos.copy()
    env._round1_initial_eq_data = env.model.eq_data.copy()
    env.max_path_length = min(MAX_EPISODE_STEPS, env.max_path_length)
    return env


def make_expert(task_name: str):
    from metaworld.policies.sawyer_drawer_open_v3_policy import SawyerDrawerOpenV3Policy
    from metaworld.policies.sawyer_door_open_v3_policy import SawyerDoorOpenV3Policy

    return {"drawer": SawyerDrawerOpenV3Policy, "door": SawyerDoorOpenV3Policy}[task_key(task_name)]()


def get_base_position(env) -> np.ndarray:
    """Actual stationary mechanism base in world coordinates (not handle)."""
    return env.data.body(env._round1_key).xpos.copy()


def reset_from_record(env, record: dict[str, Any]) -> tuple[np.ndarray, dict]:
    from metaworld.types import Task

    params = record["task_params"]
    position = np.asarray(params["base_position"], dtype=np.float64)
    if task_key(record.get("task_name", env._round1_key)) != env._round1_key:
        raise ValueError("Record belongs to a different environment")
    if np.any(position < env._random_reset_space.low) or np.any(position > env._random_reset_space.high):
        raise ValueError("Task parameters outside native reset range")
    env.model.body_pos[:] = env._round1_initial_body_pos
    env.model.site_pos[:] = env._round1_initial_site_pos
    env.model.eq_data[:] = env._round1_initial_eq_data
    mujoco.mj_resetData(env.model, env.data)
    mujoco.mj_forward(env.model, env.data)
    env._did_see_sim_exception = False
    env._last_stable_obs = None
    env._prev_obs = env._get_curr_obs_combined_no_goal().copy()
    env.set_task(Task(TASK_NAMES[env._round1_key], pickle.dumps({
        "env_cls": type(env), "rand_vec": position.copy(), "partially_observable": False,
    })))
    # Native reset(seed=...) explicitly ignores seed in this pinned version.
    env.seed(int(record["seed"]))
    obs, info = env.reset()
    if obs.shape != (OBS_DIM,):
        raise RuntimeError(f"Source-verified schema changed: {obs.shape}")
    np.testing.assert_allclose(get_base_position(env), position, atol=REPLAY_ATOL, rtol=0)
    np.testing.assert_allclose(obs[36:39], env._target_pos, atol=REPLAY_ATOL, rtol=0)
    np.testing.assert_array_equal(obs[:18], obs[18:36])
    if "initial_base_position" in record:
        np.testing.assert_allclose(get_base_position(env), record["initial_base_position"], atol=REPLAY_ATOL, rtol=0)
    return obs.copy(), info


def snapshot_initial_state(env, obs: np.ndarray) -> dict[str, np.ndarray]:
    """Diagnostic complete integration state plus non-MjData reset dependencies.

The supported restore route is canonical Task+seed reset, proved against these
snapshots and short action replay; qpos/qvel alone are never used as a restore.
"""
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.model, state_spec), dtype=np.float64)
    mujoco.mj_getState(env.model, env.data, state, state_spec)
    result = {
        "snapshot_integration_state": state,
        "snapshot_state_spec": np.asarray(int(state_spec)),
        "snapshot_model_body_pos": env.model.body_pos.copy(),
        "snapshot_model_site_pos": env.model.site_pos.copy(),
        "snapshot_model_eq_data": env.model.eq_data.copy(),
        "snapshot_prev_obs": env._prev_obs.copy(),
        "snapshot_target_pos": env._target_pos.copy(),
        "snapshot_initial_obs": obs.copy(),
        "snapshot_curr_path_length": np.asarray(env.curr_path_length),
        "snapshot_init_tcp": env.init_tcp.copy(),
        "snapshot_init_left_pad": env.init_left_pad.copy(),
        "snapshot_init_right_pad": env.init_right_pad.copy(),
    }
    for name in ("maxDist", "maxPullDist", "target_reward", "objHeight", "obj_init_pos", "obj_init_angle"):
        if hasattr(env, name):
            result[f"snapshot_{name}"] = np.asarray(getattr(env, name)).copy()
    return result


def initial_state_hash(snapshot: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(snapshot.items()):
        digest.update(key.encode())
        digest.update(np.asarray(value, dtype=np.float64).tobytes())
    return digest.hexdigest()


def verify_replay(record: dict, actions: np.ndarray | None = None, steps: int = 8) -> dict:
    """Compare two independent instances and a reused instance after stepping."""
    if actions is None:
        actions = np.random.default_rng(87343).uniform(-0.5, 0.5, (steps, 4)).astype(np.float32)
    envs = [make_env(record["task_name"]) for _ in range(2)]
    try:
        observations, snapshots = [], []
        for env in envs:
            obs, _ = reset_from_record(env, record)
            snapshots.append(snapshot_initial_state(env, obs))
            trajectory = [obs]
            for action in actions[:steps]:
                obs, _, terminated, truncated, _ = env.step(action.copy())
                trajectory.append(obs)
                if terminated or truncated:
                    break
            observations.append(np.asarray(trajectory))
        # A reset after nonzero controls must reproduce the same initial state.
        reused_obs, _ = reset_from_record(envs[0], record)
        reused = snapshot_initial_state(envs[0], reused_obs)
        max_error = float(np.max(np.abs(observations[0] - observations[1])))
        np.testing.assert_allclose(observations[0], observations[1], atol=REPLAY_ATOL, rtol=0)
        for name in snapshots[0]:
            np.testing.assert_allclose(snapshots[0][name], snapshots[1][name], atol=REPLAY_ATOL, rtol=0, err_msg=name)
            np.testing.assert_allclose(snapshots[0][name], reused[name], atol=REPLAY_ATOL, rtol=0, err_msg=f"reused:{name}")
        return {"passed": True, "steps": min(steps, len(actions)), "atol": REPLAY_ATOL,
                "max_observation_error": max_error, "initial_state_hash": initial_state_hash(snapshots[0])}
    finally:
        for env in envs:
            env.close()


def observation_schema(output: Path | None = None) -> dict:
    """Audit runtime dimensions/fields against the pinned installed source."""
    import metaworld

    source_root = Path(metaworld.__file__).parent
    schema = {
        "schema_version": SCHEMA_VERSION, "metaworld_commit": METAWORLD_COMMIT,
        "metaworld_version": importlib.metadata.version("metaworld"),
        "observation_dim": OBS_DIM, "raw_selection": "entire native goal-observable vector",
        "storage": "NPZ obs float32; reset snapshots float64; policy casts to float32",
        "native_history": "current18 + previous18 + current goal3; native reset repeats current18 into previous18",
        "dp_history": "two consecutive full native39 observations; earliest repeated at t=0",
        "input_adaptation": "No history padding change; canonical simulator/model reset only",
        "relative_mapping": "For each 18-block hand -= its handle; current goal -= current handle; all other fields preserved",
        "step_return_order": ["observation", "reward", "terminated", "truncated", "info"],
        "task_interface": "public Task/set_task, env_cls + rand_vec + partially_observable=False; env.seed(seed), native reset()",
        "reset_procedure": "restore initial body_pos/site_pos/eq_data, mj_resetData+forward, clear exception/history caches, set Task, seed, native reset",
        "reproduction": "Task+seed+canonical reset with full mjSTATE_INTEGRATION/model/target/history diagnostic snapshot; independent and reused replay tolerance 1e-9",
        "tasks": {},
    }
    fields = []
    for prefix, offset in (("current", 0), ("previous", 18)):
        for name, begin, end, meaning in (
            ("hand_xyz", 0, 3, "world position of MuJoCo hand body COM; not tcp_center"),
            ("gripper", 3, 4, "distance between rightclaw and leftclaw body COM / 0.1m, clipped [0,1]"),
            ("handle_xyz", 4, 7, "task-specific interaction point in world coordinates"),
            ("object_quaternion", 7, 11, "task-specific quaternion ordering documented below; unchanged"),
            ("absent_object2_xyz", 11, 14, "zero placeholder; not a physical position; unchanged"),
            ("absent_object2_quaternion", 14, 18, "zero placeholder; unchanged"),
        ):
            fields.append({"name": f"{prefix}.{name}", "slice": [offset + begin, offset + end], "meaning": meaning})
    fields.append({"name": "current.goal_xyz", "slice": [36, 39], "meaning": "native _target_pos; visible world xyz"})
    schema["fields"] = fields
    for key, task_name in TASK_NAMES.items():
        env = make_env(task_name)
        try:
            midpoint = (env._random_reset_space.low + env._random_reset_space.high) / 2
            record = {"task_name": task_name, "seed": 123, "task_params": {"base_position": midpoint.tolist()}}
            obs, _ = reset_from_record(env, record)
            np.testing.assert_allclose(obs[:3], env.get_endeff_pos(), atol=REPLAY_ATOL, rtol=0)
            np.testing.assert_allclose(obs[4:7], env._get_pos_objects(), atol=REPLAY_ATOL, rtol=0)
            np.testing.assert_allclose(obs[7:11], env._get_quat_objects(), atol=REPLAY_ATOL, rtol=0)
            np.testing.assert_array_equal(obs[11:18], np.zeros(7))
            np.testing.assert_array_equal(obs[29:36], np.zeros(7))
            source = source_root / "envs" / ("sawyer_drawer_open_v3.py" if key == "drawer" else "sawyer_door_v3.py")
            schema["tasks"][key] = {
                "task_name": task_name, "goal_observable": not env._partially_observable,
                "runtime_observation_dim": int(obs.size), "runtime_reset_observation": obs.tolist(),
                "native_reset_low": env._random_reset_space.low.tolist(),
                "native_reset_high": env._random_reset_space.high.tolist(),
                "handle": "drawer_link body COM + [0,-0.16,0]m" if key == "drawer" else "handle geometry center, no extra offset",
                "quaternion": "MuJoCo drawer_link body xquat: w,x,y,z" if key == "drawer" else "scipy Rotation(handle xmat).as_quat(): x,y,z,w",
                "goal": "base+[0,-0.36,0.09]m" if key == "drawer" else "base+[-0.3,-0.45,0]m",
                "expert_offsets": "handle+[0,0,-0.02] target adjustment" if key == "drawer" else "expert subtracts 0.05m from handle x in its private observation copy; representation retains native handle",
                "action_dim": int(env.action_space.shape[0]),
                "action_low": env.action_space.low.tolist(), "action_high": env.action_space.high.tolist(),
                "action_meaning": "normalized Cartesian mocap increments xyz, native scale then workspace clip; gripper control [a3,-a3]",
                "action_scale_metres": float(env.action_scale), "extra_action_scaling": False,
                "frame_skip": int(env.frame_skip), "physics_timestep_seconds": float(env.model.opt.timestep),
                "control_timestep_seconds": float(env.dt), "control_frequency_hz": float(1 / env.dt),
                "max_episode_steps": int(env.max_path_length), "native_terminated_on_success": False,
                "native_success": "norm(observed_handle-target)<=0.03" if key == "drawer" else "abs(observed_handle.x-target.x)<=0.08",
                "source_file": str(source), "source_sha256": sha256(source),
                "source_url": f"https://github.com/Farama-Foundation/Metaworld/blob/{METAWORLD_COMMIT}/metaworld/envs/{source.name}",
                "runtime_checks_passed": True,
            }
        finally:
            env.close()
    schema["base_source_sha256"] = sha256(source_root / "sawyer_xyz_env.py")
    schema["base_source_url"] = f"https://github.com/Farama-Foundation/Metaworld/blob/{METAWORLD_COMMIT}/metaworld/sawyer_xyz_env.py"
    if output is not None:
        if output.exists():
            # Schema describes immutable semantics, never silently overwrite one.
            previous = json.loads(output.read_text())
            if previous != schema:
                raise RuntimeError(f"Existing schema differs: {output}")
        else:
            atomic_json(output, schema)
    return schema
