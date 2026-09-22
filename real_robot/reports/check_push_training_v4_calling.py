"""Developer-owned local interface checks; no policy design or robot execution.

Uses one retained training scene and its supplied geometric goal to check the
public worker contract, not held-out performance. Synthetic common-clock times
are explicit fixtures; original recording timestamps remain untouched.
"""

import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from appl.io import ROOT, atomic, digest, read
from real_robot.data import Sources
from real_robot.policy_training_v4.inference import PolicyProcess, wire


def main():
    config = ROOT / "real_robot/configs/push_training_v4.json"
    cfg = read(config)
    run = ROOT / cfg["run"]
    package = run / "package_00"
    metadata = read(package / "prepared/metadata.json")
    audit = next(a for a in metadata["anchor_audit"] if a["status"] == "retained")
    sources = Sources(read(ROOT / cfg["cut_config"]))
    tid, index = audit["trajectory_id"], audit["source_index"]
    episode = sources.episode(tid)
    image_path = sources.media_path(tid, "third", index)
    rgb = np.array(Image.open(image_path).convert("RGB"))
    inferred = metadata["calibration_inference"]
    calibration = {
        "id": "offline_inference_interface_fixture_not_commissioned",
        "T_base_P": np.eye(4).tolist(),
        "third": episode["meta"]["calibration"]["third"],
        "top_height_m": inferred["top_plane_base"][3],
        "sigma_m": inferred["geometry_sigma_m"],
    }
    context = {
        "calibration": calibration,
        "roi_pixels": [[265, 38], [755, 25], [1065, 719], [35, 719]],
        "scene_revision": "interface_fixture_0",
    }
    observation = {
        "third_rgb": rgb,
        "t": 100.0,
        "recv_time": 100.02,
        "third_image_age_s": 0.02,
        "geometry_context": context,
    }
    output = run / "calling_check"
    worker = PolicyProcess(config, "shape_push_v1", output, gpu=4, revision=0)
    results = {}
    try:
        inventory = worker.inventory(observation, context["scene_revision"])
        assert inventory["status"] == "geometry_only" and inventory["instances"]
        annotations = read(package / "source/annotations.json")
        letter = "A" if tid == sources.cfg["trajectory_ids"][0] else "B"
        annotation = next(a for a in annotations[letter] if a[0] == index)
        hint = np.array(annotation[3]) * 2
        actor = min(inventory["instances"], key=lambda a: np.linalg.norm(np.array(a["seed_pixel_xy"]) - hint))
        angle = audit["goal_yaw_rad"]
        goal = np.eye(3)
        goal[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        goal[:2, 2] = audit["goal_delta_m"]
        call = {
            **context,
            "instance_id": "interface_actor",
            "reference_id": "current_contour_fixture",
            "goal_transform_P": goal.tolist(),
            "reference_contours_P": actor["contours_P"],
            "seed_pixel_xy": actor["seed_pixel_xy"],
            "workspace_polygon_P": [[-1, -1], [2, -1], [2, 2], [-1, 2]],
            "limits": {"max_stroke_m": 0.02},
            "tolerances": {"position_m": 0.01, "yaw_rad": 0.15},
            "baseline_direction_P": [1.0, 0.0],
            "tool": {"radius_m": inferred["tool_radius_estimate_m"]},
        }
        results["inventory"] = inventory
        results["proposal"] = worker.act(observation, call, reset=True, executor_contract={})
        proposal = results["proposal"]
        assert proposal["status"] == "proposed" and proposal["decision"] is not None
        assert proposal["executor_request"]["enabled"] is False
        decision = proposal["decision"]
        assert len(decision["contact_point_P"]) == len(decision["inward_push_unit_P"]) == 2
        assert 0 < decision["stroke_length_m"] <= 0.02
        assert abs(np.linalg.norm(decision["inward_push_unit_P"]) - 1) < 1e-5
        stale = dict(observation, recv_time=102.0)
        results["stale"] = worker.act(stale, call, reset=True)
        assert results["stale"]["status"] == "stale_scene"
        invalid = copy.deepcopy(call)
        invalid["goal_transform_P"] = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        results["reflection"] = worker.act(observation, invalid, reset=True)
        assert results["reflection"]["status"] == "invalid_goal"
        results["interrupt"] = worker.act(observation, dict(call, interrupt=True), reset=True)
        assert results["interrupt"]["status"] == "interrupted"
        results["missing_goal"] = worker.act(observation, {"word": "test"}, reset=True)
        assert results["missing_goal"]["status"] == "missing_geometric_goal"
        arrays = episode["arrays"]
        original_clock = dict(observation, t=float(arrays["t"][index]), recv_time=float(arrays["recv_time"][index]))
        results["original_clock"] = worker.act(original_clock, call, reset=True)
        assert results["original_clock"]["status"] == "stale_scene"
        receipt = {
            "passed": True,
            "hardware_io": False,
            "generalization_evaluation": False,
            "source_scene": {"trajectory_id": tid, "index": index, "rgb_sha256": digest(image_path)},
            "fixture": {
                "clock": "Synthetic common-clock values only for interface verification; not corrected recording timestamps",
                "calibration": "API-inferred offline values, not commissioned robot calibration",
                "workspace": "Broad numeric interface fixture, not verified robot workspace",
                "goal": "Supplied goal from the retained training label; no held-out performance evidence",
                "original_recv_minus_t_seconds": float(arrays["recv_time"][index] - arrays["t"][index]),
            },
            "call": call,
            "results": wire(results),
        }
        atomic(output / "result.json", receipt)
        print(json.dumps({"passed": True, "receipt": str(output / "result.json"), "statuses": {k: v.get("status") for k, v in results.items()}}))
    finally:
        worker.close()


if __name__ == "__main__":
    main()
