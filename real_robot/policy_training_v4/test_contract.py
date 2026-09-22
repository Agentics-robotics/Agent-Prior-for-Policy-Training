"""Synthetic provenance/interface fixtures; never sent to the design agent."""

from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from .engine import validate_prepared


@pytest.fixture
def sample():
    cfg=dict(policies=["shape_push_v1"],policy_datasets={"shape_push_v1":"shape_push_decisions"},training=dict(max_cache_bytes=64000000000))
    spec=dict(trajectory_ids=["test_episode"])
    s=dict(trajectory_id="test_episode",segment_id="learn",dataset_group="shape_push_decisions",start=0,stop=8,supervised_start=1,supervised_stop=7,supervision_kind="sparse",decision_indices=[1,3,5],supervision_exclusions=[])
    ex=dict(s,segment_id="executor",dataset_group="geometric_motion_executor",supervision_kind="executor_only",decision_indices=[],supervised_start=None,supervised_stop=None)
    arrays=dict(example_episode=np.array([0,0]),example_source_index=np.array([1,3]),example_segment=np.array([0,0]),observation_start=np.array([0,1]),observation_stop=np.array([2,4]),target_start=np.array([1,3]),target_stop=np.array([3,6]),policy_weight=np.ones((2,1)),geometric_target=np.ones((2,5)))
    audit=[dict(trajectory_id="test_episode",segment_id="learn",source_index=i,status="retained" if i in (1,3) else "excluded",reason="TEST FIXTURE") for i in (1,3,5)]
    prepared=dict(arrays=arrays,metadata=dict(num_examples=2,coverage={"fixture":"TEST FIXTURE"},label_provenance="TEST FIXTURE",anchor_audit=audit))
    return prepared,cfg,spec,[s,ex],SimpleNamespace()


def test_sparse_geometric_targets_do_not_require_ee_action(sample):
    v=validate_prepared(*sample)
    assert v["frozen_sparse_anchors"]==3 and v["excluded_sparse_anchors"]==1
    assert v["segment_examples"]=={"learn":2,"executor":0}


def test_context_row_cannot_become_decision(sample):
    sample[0]["arrays"]["example_source_index"][0]=2
    with pytest.raises(ValueError,match="frozen segment"):
        validate_prepared(*sample)


def test_executor_only_cannot_become_supervision(sample):
    sample[0]["arrays"]["example_segment"][0]=1
    with pytest.raises(ValueError,match="frozen segment"):
        validate_prepared(*sample)


def test_future_observation_rejected(sample):
    sample[0]["arrays"]["observation_stop"][0]=3
    with pytest.raises(ValueError,match="causal"):
        validate_prepared(*sample)


def test_target_crossing_segment_rejected(sample):
    sample[0]["arrays"]["target_stop"][0]=9
    with pytest.raises(ValueError,match="Target evidence"):
        validate_prepared(*sample)


def test_every_rejected_anchor_requires_accounting(sample):
    sample[0]["metadata"]["anchor_audit"].pop()
    with pytest.raises(ValueError,match="Anchor audit"):
        validate_prepared(*sample)


def test_audit_cannot_claim_unused_anchor_retained(sample):
    sample[0]["metadata"]["anchor_audit"][-1]["status"]="retained"
    with pytest.raises(ValueError,match="Anchor audit"):
        validate_prepared(*sample)


def test_group_sampling_and_nonfinite_targets_rejected(sample):
    original=deepcopy(sample)
    sample[3][0]["dataset_group"]="geometric_motion_executor"
    with pytest.raises(ValueError,match="another frozen dataset"):
        validate_prepared(*sample)
    original[0]["arrays"]["geometric_target"][0,0]=np.nan
    with pytest.raises(ValueError,match="must be finite"):
        validate_prepared(*original)


def test_duplicate_source_anchor_rejected(sample):
    a=sample[0]["arrays"]
    a["example_source_index"][1]=1
    with pytest.raises(ValueError,match="Duplicate identical"):
        validate_prepared(*sample)
