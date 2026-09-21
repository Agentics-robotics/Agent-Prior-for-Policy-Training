"""Bounded live-input interface check of trained policies; no rollout or scoring."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from appl.io import ROOT, atomic, read
from real_robot.data import Sources
from real_robot.policy_training_v3.inference import PolicyProcess, wire


def observation(cfg, index):
    cut = read(ROOT / cfg["cut_config"])
    cut["trajectory_ids"] = cut["trajectory_ids"][:1]
    sources = Sources(cut)
    tid = cut["trajectory_ids"][0]
    episode = sources.episode(tid)
    keys = ["q", "dq", "tau_ext", "T_base_ee", "T_base_flange",
            "gripper_position", "gripper_width_m", "t", "recv_time",
            "img_t_wrist", "img_t_third", "img_age_wrist", "img_age_third"]
    ob = {k: episode["arrays"][k][index] for k in keys if k in episode["arrays"]}
    ob.update(metadata=episode["meta"], sample_index=index)
    for camera in ("third", "wrist"):
        with Image.open(sources.media_path(tid, camera, index)) as im:
            ob[camera + "_rgb"] = np.array(im.convert("RGB"))
    return ob, tid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--index", type=int, default=300)
    args = parser.parse_args()
    config = ROOT / "real_robot/configs/push_training_v3.json"
    cfg = read(config)
    run = ROOT / cfg["run"]
    ob, tid = observation(cfg, args.index)
    out = run / f"interface_check_{args.revision:02d}"
    out.mkdir(exist_ok=False)
    cases = []
    for pid in cfg["policies"]:
        worker = PolicyProcess(config, pid, out / pid, gpu=args.gpu, revision=args.revision)
        try:
            invalid = worker.inventory({}, "invalid-observation")
            assert invalid["status"] == "PERCEPTION_UNCERTAIN" and not invalid["instances"]
            scene = worker.inventory(ob, "bounded-interface-fixture")
            assert scene["hardware_io"] is False
            base = dict(policy_id=pid, call_id="bounded-interface-fixture",
                        now_s=float(ob["t"]), timeout_s=30., arrival_mode="pass_through",
                        protected_instance_ids=[], last_executed_action=[0.] * 6)
            if pid == "tool_waypoint_v1":
                base["goal_tcp_base"] = np.asarray(ob["T_base_ee"]).reshape(4, 4).tolist()
            elif scene["instances"]:
                selected = scene["instances"][0]
                base.update(instance_id="interface-fixture-instance",
                            instance_seed_W=selected["seed_W"], inventory_t=float(ob["t"]),
                            goal_SE2_W=np.eye(3).tolist())
            else:
                cases.append(dict(policy_id=pid, status="unavailable_on_fixture",
                                  scene=scene, reason="No current component to bind; no alternative fixture search"))
                continue
            valid = worker.act(ob, base, reset=True, controller_contract={})
            assert valid["hardware_io"] is False and valid["status"] and valid["diagnostics"]
            candidate = valid["action"]
            if candidate is not None:
                assert np.asarray(candidate).shape == (6,) and np.isfinite(candidate).all()
                assert valid["decoded"]["enabled"] is False and valid["decoded"]["command"] is None
            stale = worker.act(ob, dict(base, now_s=float(ob["t"]) + 2.))
            assert stale["action"] is None and stale["status"] == "PERCEPTION_UNCERTAIN"
            aborted = worker.act(ob, dict(base, abort=True))
            assert aborted["action"] is None and aborted["status"] == "ABORTED"
            bad_goal = dict(base)
            goal_key = "goal_tcp_base" if pid == "tool_waypoint_v1" else "goal_SE2_W"
            invalid_transform = np.asarray(base[goal_key]).copy()
            invalid_transform[0, 0] = 2.0
            bad_goal[goal_key] = invalid_transform.tolist()
            rejected_goal = worker.act(ob, bad_goal, reset=True)
            assert rejected_goal["action"] is None and rejected_goal["status"] == "PERCEPTION_UNCERTAIN"
            atomic(out / pid / "responses.json", wire(dict(scene=scene, call=base,
                    valid=valid, stale=stale, aborted=aborted, invalid_goal=rejected_goal)))
            cases.append(dict(policy_id=pid, status=valid["status"],
                              scene_instances=len(scene["instances"]),
                              finite_candidate=candidate is not None,
                              unverified_decoder_disabled=candidate is not None,
                              stale_input_rejected=True, invalid_goal_rejected=True,
                              interruption_supported=True))
        finally:
            worker.close()
    result = dict(status="completed", revision=args.revision, cases=cases,
                  paired_observation=dict(trajectory=tid, sample_index=args.index),
                  fixture_goal="Current TCP/current piece footprint; interface exercise only",
                  source_action_labels_provided=False, future_observations_provided=False,
                  candidate_optimizer_updates=0, performance_evaluation=False, robot_execution=False)
    atomic(run / "interface_check.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
