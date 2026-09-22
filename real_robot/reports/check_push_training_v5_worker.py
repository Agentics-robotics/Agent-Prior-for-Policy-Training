"""Independent process/serialization check using the API's recorded fixture.

Developer-owned interface verification, not policy code or robot evaluation.
"""
import fcntl

import numpy as np

from appl.io import ROOT, atomic, digest, read
from real_robot.training_pipeline.inference import PolicyProcess


def main():
    config = ROOT / "real_robot/configs/push_training_v5.json"
    cfg = read(config)
    root = ROOT / cfg["run"]
    state = read(root / "workflow.json")
    assert state["status"] == "complete"
    revision = state["revision"]
    package = root / f"package_{revision:02d}"
    fixture = read(package / "prepared/metadata.json")["calling_fixture"]
    chart = np.eye(4)
    chart[2, 3] = .068
    # These acknowledgements are synthetic test inputs, not commissioned facts.
    obs = dict(scene_revision=1, time_s=10., calibration_id="offline_host_fixture",
        T_base_ee=fixture["T_base_ee"], T_base_P=chart.tolist(),
        uncertainty_m=.008, chart_validated=True,
        instances=[dict(instance_id="recorded_support", support_points_P=fixture["support"],
                        geometry_complete=False)],
        previous_delta_P_m=fixture["previous_delta"], history_dt_s=.2)
    call = dict(instance_id="recorded_support", goal_R=fixture["R"], goal_t=fixture["t"],
                scene_revision=1, now_s=10.01, contact_confirmed=True)
    contract = dict(commissioned=True, scene_revision=1, calibration_id="offline_host_fixture",
        T_base_P=chart.tolist(), guard_profile_id="synthetic", tool_geometry_id="synthetic",
        max_speed_m_s=.01)
    lock_path = ROOT / "real_robot/runs/.training_pipeline_locks/gpu_4.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        worker = PolicyProcess(config, "shape_push_v1", root / "host_worker_check",
                               gpu=4, revision=revision)
        try:
            first = worker.act(obs, call, reset=True, executor_contract={})
            assert first["status"] == "proposed_continuation"
            delta = np.asarray(first["decision"]["delta_P_m"])
            assert delta.shape == (2,) and np.isfinite(delta).all()
            assert np.linalg.norm(delta) <= .015 + 1e-8
            assert first["executor_request"]["status"] == "blocked_uncommissioned"
            second = worker.act(obs, call, executor_contract=contract)
            request = second["executor_request"]
            assert request["status"] == "requires_planner_validation"
            start = np.asarray(request["start_T_base_ee"])
            end = np.asarray(request["end_T_base_ee"])
            np.testing.assert_array_equal(start[:3, :3], end[:3, :3])
            assert start[2, 3] == end[2, 3]
            np.testing.assert_allclose(end[:2, 3] - start[:2, 3], delta, atol=1e-8)
            missing_geometry = worker.act(obs, dict(call, contact_confirmed=False))
            assert missing_geometry["status"] == "needs_full_geometry_for_initialization"
            interrupted = worker.act(obs, dict(call, interrupted=True))
            assert interrupted["status"] == "interrupted" and interrupted["decision"] is None
        finally:
            worker.close()
    result = dict(passed=True, policy_id="shape_push_v1", source=fixture["source"],
        cases=[first, second, missing_geometry, interrupted], actual_separate_worker=True,
        synthetic_flags_and_limits=True, hardware_execution=False, performance_evaluation=False,
        continuation_adapter_preserves_height_and_orientation=True,
        checkpoint_sha256=digest(package / "training/shape_push_v1/last.pt"))
    atomic(root / "host_worker_check/result.json", result)
    print(dict(passed=True, statuses=[x["status"] for x in result["cases"]]), flush=True)


if __name__ == "__main__":
    main()
