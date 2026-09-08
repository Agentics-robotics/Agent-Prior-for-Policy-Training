import numpy as np
import pytest

from relative_dp.collection import build_sampling_plan, run_expert_episode
from relative_dp.environment import (TASK_NAMES, get_base_position, make_env,
                                    observation_schema, reset_from_record, verify_replay)


@pytest.fixture(scope="module")
def schema():
    return observation_schema()


def test_source_schema_runtime_and_independent_splits(schema):
    plan = build_sampling_plan(schema)
    assert plan == build_sampling_plan(schema)
    assert schema["observation_dim"] == 39
    for key, task in plan["tasks"].items():
        assert schema["tasks"][key]["goal_observable"]
        assert schema["tasks"][key]["action_dim"] == 4
        assert schema["tasks"][key]["max_episode_steps"] == 500
        assert schema["tasks"][key]["action_scale_metres"] == .01
        records = sum(task["splits"].values(), [])
        assert len(records) == len({r["task_parameter_hash"] for r in records})
        env = make_env(key)
        try:
            for split, conditions in task["splits"].items():
                for record in (conditions[0], conditions[-1]):
                    obs, _ = reset_from_record(env, record)
                    np.testing.assert_array_equal(obs[:18], obs[18:36])
                    np.testing.assert_allclose(get_base_position(env), record["task_params"]["base_position"], atol=1e-9, rtol=0)
                    x = get_base_position(env)[0]
                    ranges = plan["ranges"][key]
                    if split == "test_ood":
                        lo, hi = ranges["ood_x"][0 if record["ood_side"] == "left" else 1]
                    else:
                        lo, hi = ranges["train_dev_iid_x"]
                    assert lo <= x <= hi
        finally:
            env.close()


@pytest.mark.parametrize("key", TASK_NAMES)
def test_experts_replay_and_no_observation_mutation(key, schema):
    plan = build_sampling_plan(schema)
    env = make_env(key)
    try:
        for index in range(3):
            record = plan["tasks"][key]["splits"]["train_pool"][index]
            arrays, summary = run_expert_episode(env, record)
            assert summary["expert_success"], summary
            assert arrays["obs"].shape == (len(arrays["actions"]) + 1, 39)
            assert np.all(arrays["actions"] <= 1) and np.all(arrays["actions"] >= -1)
            np.testing.assert_array_equal(arrays["obs"][0], arrays["snapshot_initial_obs"].astype(np.float32))
            assert arrays["success"][-1]
            replay = verify_replay(record, arrays["actions"][:8])
            assert replay["passed"]
            obs, _ = reset_from_record(env, record)
            for step, action in enumerate(arrays["actions"][:8]):
                obs, _, terminated, truncated, _ = env.step(action)
                np.testing.assert_allclose(obs, arrays["obs"][step + 1], atol=1e-6, rtol=0)
                assert not terminated and not truncated
    finally:
        env.close()
