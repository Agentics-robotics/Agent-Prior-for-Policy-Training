"""Measure submitted segment support; never change API cuts, priors or labels."""

import ast
from collections import Counter
import json

import numpy as np

from appl.io import atomic, digest, read
from real_robot.data import ROOT, Sources, config


def main():
    cfg = config(ROOT / "real_robot/configs/push_cut_v4.json")
    source = Sources(cfg)
    plan_path = ROOT / cfg["output"] / "plan.json"
    plan = read(plan_path)
    rows = []
    for skill in plan["skills"]:
        for segment in skill["segments"]:
            arrays = source.episodes[segment["trajectory_id"]]["arrays"]
            lo, hi = segment["supervised_start"], segment["supervised_stop"]
            t = arrays["t"]
            xyz = arrays["T_base_ee"][lo:hi, :3, 3]
            valid = []
            for raw in arrays["action_json"][lo:hi]:
                command = np.asarray(json.loads(str(raw)).get("dq"), dtype=float)
                valid.append(command.shape == (7,) and bool(np.isfinite(command).all()))
            last = json.loads(str(arrays["action_json"][hi - 1]))
            last_dq = np.asarray(last.get("dq"), dtype=float)
            tail = xyz[-min(16, len(xyz)):]
            dt = np.diff(t[hi - len(tail):hi])
            assert len(dt) and (dt > 0).all()
            rows.append(dict(
                skill=skill["skill_id"], segment=segment["segment_id"],
                episode=segment["trajectory_id"], supervised=[lo, hi],
                duration_first_to_last_s=float(t[hi - 1] - t[lo]),
                context_before_s=float(t[lo] - t[segment["start"]]),
                context_after_s=float(t[segment["stop"] - 1] - t[hi - 1]),
                finite_command_rows=sum(valid),
                start_xyz_m=xyz[0].tolist(), end_xyz_m=xyz[-1].tolist(),
                z_range_m=[float(xyz[:, 2].min()), float(xyz[:, 2].max())],
                net_dz_m=float(xyz[-1, 2] - xyz[0, 2]),
                positive_z_path_m=float(np.maximum(np.diff(xyz[:, 2]), 0).sum()),
                negative_z_path_m=float(-np.minimum(np.diff(xyz[:, 2]), 0).sum()),
                end_command_dq=last.get("dq"),
                end_command_norm=float(np.linalg.norm(last_dq))
                if last_dq.shape == (7,) and np.isfinite(last_dq).all() else None,
                end_command_v=last.get("v"), end_command_w=last.get("w"),
                tail_tcp_speed_m_s_mean=float(np.mean(np.linalg.norm(np.diff(tail, axis=0), axis=1) / dt)),
                training_condition=segment["training_condition"],
            ))

    access = [row for row in rows if row["skill"] == "tool_access"]
    legacy_path = ROOT / "real_robot/policy_training/design.py"
    tree = ast.parse(legacy_path.read_text())
    legacy = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "POLICIES" for target in node.targets))
    manifest = read(ROOT / cfg["output"] / "manifest.json")
    for name, expected in manifest["files"].items():
        assert digest(ROOT / cfg["output"] / name) == expected, name
    output = ROOT / cfg["run"] / "interface_audit"
    output.mkdir(exist_ok=True)
    atomic(output / "segment_motion_evidence.json", dict(
        date="2026-09-21", plan_sha256=digest(plan_path),
        purpose="Read-only interface/support audit; recorded EE motion is not object pose or contact truth",
        tail_window="Up to 16 final supervised rows; mean consecutive EE position speed using original t",
        command_norm_units="Native recorded command units; controller semantics are not newly verified",
        segments=rows,
    ))
    result = dict(
        date="2026-09-21", plan_sha256=digest(plan_path),
        access_segments=len(access),
        access_finite_command_rows=sum(row["finite_command_rows"] for row in access),
        access_endpoints_with_nonzero_command=sum(row["end_command_norm"] is not None and row["end_command_norm"] > 0 for row in access),
        all_segment_endpoints_with_nonzero_command=sum(row["end_command_norm"] is not None and row["end_command_norm"] > 0 for row in rows),
        legacy_implementation_policy_ids=list(legacy),
        cut_v4_policy_ids=[h["policy_id"] for s in plan["skills"] for h in s["heuristics"]],
        legacy_design_sha256=digest(legacy_path),
        published_files_unchanged=len(manifest["files"]),
        no_new_api_training_or_robot_execution=True,
    )
    calls = []
    for path in sorted((ROOT / cfg["run"] / "_session/api").glob("*.response.json")):
        response = read(path)["response"]
        for item in response.get("output", []):
            if item.get("type") == "function_call":
                calls.append(dict(name=item["name"], args=json.loads(item["arguments"])))
    result["evidence_tool_calls"] = dict(Counter(c["name"] for c in calls))
    result["motion_inspections"] = [c["args"] for c in calls if c["name"] == "read_motion"]
    result["recorded_poses_and_state_inspection"] = []
    for tid, episode in source.episodes.items():
        pose = episode["arrays"]["T_base_ee"]
        inspected = {i for c in calls if c["name"] == "read_steps"
                     and c["args"]["trajectory_id"] == tid for i in c["args"]["indices"]}
        result["recorded_poses_and_state_inspection"].append(dict(
            trajectory_id=tid, T_base_ee_shape=list(pose.shape),
            finite_pose_rows=int(np.isfinite(pose).all(axis=(1, 2)).sum()),
            unique_explicit_state_inspection_rows=len(inspected),
        ))
    atomic(output / "audit_facts.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
