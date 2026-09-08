"""Protocol tests catch action/history offset and checkpoint-selection errors."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from relative_dp import environment
from relative_dp.evaluate import _read_cached, rollout, sampling_seed, select_checkpoint
from relative_dp.utils import atomic_json


class FakePolicy:
    device = torch.device("cpu")

    def __init__(self):
        self.conditions = []

    def predict_actions(self, observations, generator):
        self.conditions.append(observations.copy())
        # Deliberately outside native bounds to check actual env.step clipping.
        return np.full((16, 4), 2, dtype=np.float32)


class FakeEnv:
    action_space = SimpleNamespace(low=np.full(4,-1), high=np.full(4,1))

    def __init__(self, success_step=None, truncation_step=None):
        self.steps = 0
        self.success_step = success_step
        self.truncation_step = truncation_step

    def step(self, action):
        np.testing.assert_array_equal(action, np.ones(4))
        self.steps += 1
        return np.full(39,self.steps), 1.5, False, self.steps == self.truncation_step, {"success": self.steps == self.success_step}


@pytest.fixture
def record(monkeypatch):
    def reset(env, record):
        env.steps = 0
        return np.zeros(39), {}
    monkeypatch.setattr(environment,"reset_from_record",reset)
    monkeypatch.setattr(environment,"get_base_position",lambda env: np.zeros(3))
    return {"episode_id":"independent-1","seed":812,"initial_base_position":[0,0,0]}


def test_chunk_stops_at_first_success_without_extra_action(record):
    policy, env = FakePolicy(), FakeEnv(success_step=2)
    result = rollout(policy,env,record,seed=7)
    assert result["success"] and result["first_success_step"] == 2
    assert result["episode_length"] == env.steps == 2
    assert len(policy.conditions) == 1
    assert result["return"] == 3
    assert result["exception"] is None


def test_chunk_stops_at_truncation_and_uses_current_history(record):
    policy, env = FakePolicy(), FakeEnv(truncation_step=6)
    result = rollout(policy,env,record,seed=7)
    assert result["truncated"] and not result["success"]
    assert result["episode_length"] == env.steps == 6
    assert len(policy.conditions) == 2
    np.testing.assert_array_equal(policy.conditions[0],np.zeros((2,39)))
    np.testing.assert_array_equal(policy.conditions[1],np.stack([np.full(39,3),np.full(39,4)]))


def test_fixed_dev_selection_ties():
    def candidate(step, rate, mean):
        return {"step":step,"summary":{"success_rate":rate,"mean_success_steps":mean}}
    assert select_checkpoint([candidate(20000,.5,12),candidate(10000,.5,12),candidate(5000,.5,13)])["step"] == 10000
    assert select_checkpoint([candidate(20000,.6,100),candidate(5000,.5,12)])["step"] == 20000
    assert select_checkpoint([candidate(20000,0,None),candidate(5000,0,None),candidate(10000,0,None)])["step"] == 5000


def test_episode_sampling_streams_do_not_depend_on_order():
    first = sampling_seed("drawer","test_iid","iid-0")
    assert first == sampling_seed("drawer","test_iid","iid-0")
    assert len({first,sampling_seed("drawer","dev","iid-0"),sampling_seed("door","test_iid","iid-0")}) == 3
    torch.randn(10, generator=torch.Generator().manual_seed(first))
    a = torch.randn(5,generator=torch.Generator().manual_seed(first))
    b = torch.randn(5,generator=torch.Generator().manual_seed(first))
    torch.testing.assert_close(a,b)


def test_resume_rejects_stale_identity_or_wrong_initial_condition(tmp_path):
    path = tmp_path/"episodes.json"
    identity = {"checkpoint_hash":"frozen", "data_manifest_hash":"data"}
    records = [{"episode_id":"dev-0"},{"episode_id":"dev-1"}]
    atomic_json(path,{"identity":identity,"complete":False,"episodes":[{"episode_id":"dev-0","exception":None}]})
    assert len(_read_cached(path,identity,records)["episodes"]) == 1
    with pytest.raises(ValueError,match="Stale"):
        _read_cached(path,{**identity,"checkpoint_hash":"different"},records)
    with pytest.raises(ValueError,match="order mismatch"):
        _read_cached(path,identity,list(reversed(records)))
