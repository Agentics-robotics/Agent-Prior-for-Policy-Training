"""Independent IPC/checkpoint check; no scientific edits or physical execution."""
import fcntl
import numpy as np

from appl.io import ROOT, atomic, digest, read
from real_robot.training_pipeline.inference import PolicyProcess


def main():
    config = ROOT / "real_robot/configs/push_training_v6.json"
    cfg = read(config)
    root = ROOT / cfg["run"]
    state = read(root / "workflow.json")
    assert state["status"] == "complete"
    package = root / f"package_{state['revision']:02d}"
    cache = package / "prepared"
    manifest = read(cache / "result.json")

    def array(name):
        return np.load(cache / manifest["arrays"][name]["file"], allow_pickle=False)

    xy, goal = array("xy")[0], array("goal_xy")[0]
    x, y = xy - xy.mean(0), goal - goal.mean(0)
    angle = float(np.arctan2(np.sum(x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]),
                             np.sum(x * y)))
    # Current contour is recorded; the caller goal is a hindsight test fixture.
    # All commissioning flags, contact height and transforms below are synthetic.
    obs = dict(tracks=[dict(instance_id="fixture_actor", xy=xy.tolist(),
        normal=array("normal")[0].tolist(), loop=array("loop")[0].tolist())],
        workspace=[[.1, -.7], [1., -.7], [1., .7], [.1, .7]],
        tool_radius_m=.003, contact_height_m=.045, calibration_valid=True,
        calibration_id="offline_fixture_only", uncertainty_m=.006,
        age_s=.02, revision=11, stable=False, geometry_complete_validated=True)
    call = dict(instance_id="fixture_actor", scene_revision=11, reference_id="fixture",
        goal_delta=(goal.mean(0) - xy.mean(0)).tolist() + [angle], max_stroke_m=.01)
    contract = dict(commissioned=True, scene_revision=11,
        T_flange_tool=np.eye(4).tolist(), R_base_tool=np.eye(3).tolist(),
        tool_collision_model_id="synthetic_fixture", guard_profile_id="synthetic_fixture")
    lock_path = ROOT / "real_robot/runs/.training_pipeline_locks/gpu_4.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        worker = PolicyProcess(config, "shape_push_v1", root / "host_worker_check",
                               gpu=4, revision=state["revision"])
        try:
            first = worker.act(obs, call, reset=True, executor_contract={})
            assert first["status"] == "proposed"
            d = first["decision"]
            point, direction = np.asarray(d["contact_point_m"]), np.asarray(d["direction_unit"])
            assert np.isfinite(point).all() and np.isfinite(direction).all()
            assert np.min(np.linalg.norm(xy - point, axis=1)) < 1e-6
            assert abs(np.linalg.norm(direction) - 1) < 1e-5
            assert 0 < d["stroke_length_m"] <= .01 + 1e-8
            assert first["executor_request"] == dict(status="calibration_required", execute=False)
            second = worker.act(obs, call, executor_contract=contract)
            request = second["executor_request"]
            assert request["status"] == "objective_only" and not request["execute"]
            start = np.asarray(request["T_base_flange_contact"])
            end = np.asarray(request["T_base_flange_end"])
            np.testing.assert_array_equal(start[:3, :3], end[:3, :3])
            assert start[2, 3] == end[2, 3]
            np.testing.assert_allclose(end[:2, 3] - start[:2, 3],
                                       direction * d["stroke_length_m"], atol=1e-8)
            stale = worker.act(obs, dict(call, scene_revision=10))
            assert stale["status"] == "stale_scene" and stale["decision"] is None
            interrupted = worker.act(obs, dict(call, interrupted=True))
            assert interrupted["status"] == "interrupted" and interrupted["decision"] is None
            incomplete = worker.act(dict(obs, geometry_complete_validated=False), call)
            assert incomplete["status"] == "needs_view" and incomplete["decision"] is None
        finally:
            worker.close()
    result = dict(passed=True, actual_separate_worker=True,
        cases=[first, second, stale, interrupted, incomplete], synthetic_executor_flags=True,
        synthetic_single_object_scene=True, synthetic_scene_validation=True,
        historical_goal_fixture=True, hardware_execution=False, performance_evaluation=False,
        checkpoint_sha256=digest(package / "training/shape_push_v1/last.pt"),
        fixture=dict(observation=obs, call=call, executor_contract=contract))
    atomic(root / "host_worker_check/result.json", result)
    print(dict(passed=True, statuses=[x["status"] for x in result["cases"]]), flush=True)


if __name__ == "__main__":
    main()
