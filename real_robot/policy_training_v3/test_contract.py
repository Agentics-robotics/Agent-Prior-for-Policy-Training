"""Developer provenance tests; no candidate scientific model is authored."""
import json
import unittest
from types import SimpleNamespace
import numpy as np
from .engine import validate_prepared
from .design import DATASETS, POLICIES


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(policies=list(POLICIES), policy_datasets=DATASETS,
                        training=dict(max_cache_bytes=64000000000))
        self.spec = dict(trajectory_ids=["episode"])
        base = dict(trajectory_id="episode", start=0, stop=4,
                    supervised_start=1, supervised_stop=3, supervision_exclusions=[])
        self.segments = [dict(base, segment_id="tool", dataset_group="tool_position"),
                         dict(base, segment_id="piece", dataset_group="piece_relocation")]
        self.sources = SimpleNamespace(episode=lambda tid: dict(arrays=dict(
            action_json=np.array([json.dumps(dict(v=[1., 2., 3.], w=[4., 5., 6.], dq=None))] * 4))))
        self.value = dict(arrays=dict(
            example_episode=np.array([0, 0]), example_source_index=np.array([1, 1]),
            example_segment=np.array([0, 1]),
            native_action=np.array([[1., 2., 3., 4., 5., 6.]] * 2),
            policy_weight=np.array([[1., 0., 0.], [0., 1., 1.]])),
            metadata=dict(num_examples=2, coverage={}))

    def validate(self):
        return validate_prepared(self.value, self.cfg, self.spec, self.segments, self.sources)

    def test_vw_with_missing_dq_and_reused_interval(self):
        self.assertEqual(self.validate()["policy_examples"], dict.fromkeys(POLICIES, 1))

    def test_cross_group_sampling_rejected(self):
        self.value["arrays"]["policy_weight"][0, 1] = 1
        with self.assertRaisesRegex(ValueError, "another frozen dataset"):
            self.validate()

    def test_changed_native_label_rejected(self):
        self.value["arrays"]["native_action"][0, 2] = 100
        with self.assertRaisesRegex(ValueError, "differs from recorded"):
            self.validate()

    def test_context_is_not_supervision(self):
        self.value["arrays"]["example_source_index"][0] = 0
        with self.assertRaisesRegex(ValueError, "frozen segment/masks"):
            self.validate()

    def test_exclusion_is_not_supervision(self):
        self.segments[0]["supervision_exclusions"] = [dict(start=1, stop=2)]
        with self.assertRaisesRegex(ValueError, "frozen segment/masks"):
            self.validate()

    def test_duplicate_anchor_rejected(self):
        self.value["arrays"]["example_segment"][1] = 0
        with self.assertRaisesRegex(ValueError, "Duplicate identical"):
            self.validate()

    def test_unrepresented_policy_rejected(self):
        self.value["arrays"]["policy_weight"][:, 2] = 0
        with self.assertRaisesRegex(ValueError, "positive row/column"):
            self.validate()

    def test_weights_must_be_finite(self):
        self.value["arrays"]["policy_weight"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite nonnegative"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
