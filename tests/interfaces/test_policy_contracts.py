import json
from pathlib import Path

import pytest

from experiment_interfaces.envs import EnvSpec
from experiment_interfaces.policies import validate_device


@pytest.mark.parametrize("visible,device", [("0", "cuda:0"), ("4,3", "cuda:0"), ("4,4", "cuda:0"),
                                           ("4,5", "cuda:2"), ("GPU-uuid", "cuda:0")])
def test_unauthorized_or_ambiguous_gpu_visibility_is_rejected(monkeypatch, visible, device):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(ValueError):
        validate_device(device)

