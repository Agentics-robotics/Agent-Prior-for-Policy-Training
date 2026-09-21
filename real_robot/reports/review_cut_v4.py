"""Read-only scientific audit and timeline of the exact API cut_v4 submission."""

from pathlib import Path
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from appl.io import atomic, digest, read
from real_robot.data import ROOT, Sources, config


def intervals(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def main():
    cfg = config(ROOT / "real_robot/configs/push_cut_v4.json")
    run, output = ROOT / cfg["run"], ROOT / cfg["output"]
    validation = read(run / "validation.json")
    assert validation["passed"]
    plan = read(output / "plan.json")
    source = Sources(cfg)
    assigned = {tid: np.zeros(ep["n"], dtype=np.int32) for tid, ep in source.episodes.items()}
    supervised = {tid: np.zeros(ep["n"], dtype=np.int32) for tid, ep in source.episodes.items()}
    command_valid = {}
    for tid, ep in source.episodes.items():
        valid = []
        for raw in ep["arrays"]["action_json"]:
            command = json.loads(str(raw)).get("dq")
            values = np.asarray(command, dtype=np.float64)
            valid.append(values.shape == (7,) and bool(np.isfinite(values).all()))
        command_valid[tid] = np.asarray(valid)

    group_rows, segment_rows = [], []
    palette = plt.get_cmap("tab10")
    fig, axes = plt.subplots(len(source.episodes), 1, figsize=(15, 3 + 0.48 * len(plan["skills"]) * len(source.episodes)), squeeze=False)
    episode_axes = dict(zip(source.episodes, axes[:, 0]))
    for group_index, skill in enumerate(plan["skills"]):
        group_assigned = {tid: np.zeros_like(mask, dtype=bool) for tid, mask in assigned.items()}
        group_supervised = {tid: np.zeros_like(mask, dtype=bool) for tid, mask in assigned.items()}
        for segment in skill["segments"]:
            tid, lo, hi = segment["trajectory_id"], segment["start"], segment["stop"]
            slo, shi = segment["supervised_start"], segment["supervised_stop"]
            mask = np.zeros_like(assigned[tid], dtype=bool)
            mask[slo:shi] = True
            excluded = 0
            for exclusion in segment["supervision_exclusions"]:
                a, b = exclusion["start"], exclusion["stop"]
                mask[a:b] = False
                excluded += b - a
            assigned[tid][lo:hi] += 1
            supervised[tid] += mask
            group_assigned[tid][lo:hi] = True
            group_supervised[tid] |= mask
            segment_rows.append(dict(skill_id=skill["skill_id"], segment_id=segment["segment_id"], trajectory_id=tid, start=lo, stop=hi, materialized_records=hi-lo, supervised_start=slo, supervised_stop=shi, leading_context_records=slo-lo, trailing_context_records=hi-shi, explicitly_masked_records=excluded, supervision_candidates=int(mask.sum()), finite_command_candidates=int((mask & command_valid[tid]).sum()), objective=segment["objective"], condition=segment["training_condition"]))
            ax = episode_axes[tid]
            color = palette(group_index % 10)
            ax.broken_barh([(lo, hi-lo)], (group_index-0.3, 0.6), color=color, alpha=0.22)
            ax.broken_barh([(int(a), int(b-a)) for a, b in intervals(mask)], (group_index-0.2, 0.4), color=color, alpha=0.95)
        group_rows.append(dict(skill_id=skill["skill_id"], name=skill["name"], subgoal=skill["subgoal"], segments=len(skill["segments"]), policies=[h["policy_id"] for h in skill["heuristics"]], materialized_assignment_records=sum(s["stop"]-s["start"] for s in skill["segments"]), assigned_unique=sum(int(v.sum()) for v in group_assigned.values()), supervised_unique=sum(int(v.sum()) for v in group_supervised.values()), finite_command_supervised_unique=sum(int((v & command_valid[tid]).sum()) for tid, v in group_supervised.items())))

    coverage = {}
    for tid in assigned:
        counts, sup = assigned[tid], supervised[tid]
        actual = dict(source_records=len(counts), assigned_unique=int((counts>0).sum()), reused_unique=int((counts>1).sum()), assignment_records=int(counts.sum()), supervised_unique=int((sup>0).sum()), context_only_unique=int(((counts>0)&(sup==0)).sum()), finite_command_supervised_unique=int(((sup>0)&command_valid[tid]).sum()), missing_command_supervised_unique=int(((sup>0)&~command_valid[tid]).sum()))
        frozen = validation["coverage"][tid]
        for key in ("source_records", "assigned_unique", "reused_unique", "assignment_records", "supervised_unique", "context_only_unique"):
            assert actual[key] == frozen[key], (tid, key, actual[key], frozen[key])
        coverage[tid] = actual

    for tid, ax in episode_axes.items():
        ax.set_title(tid)
        ax.set_yticks(range(len(plan["skills"])), labels=[s["skill_id"] for s in plan["skills"]])
        ax.set_ylim(-0.7, len(plan["skills"])-0.3)
        ax.invert_yaxis()
        ax.set_xlim(0, len(assigned[tid]))
        ax.set_xlabel("Original paired sample index")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("API cut_v4: subtask datasets and overlap\nLight: retained interval; dark: action-supervision candidates before input preprocessing", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figure = ROOT / "real_robot/reports/PUSH_CUT_V4_TIMELINE.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)

    previous_freeze = read(ROOT / "real_robot/runs/push_letters/cut_v3/interface_freeze.json")
    for name, expected in previous_freeze["files"].items():
        assert digest(ROOT / name) == expected, name
    previous_manifest = read(ROOT / "real_robot/data/push_letters/cut_v3/manifest.json")
    for name, expected in previous_manifest["files"].items():
        assert digest(ROOT / "real_robot/data/push_letters/cut_v3" / name) == expected, name
    review = dict(review_date="2026-09-21",authorship="Developer deterministic audit of unchanged API submission",passed=True,plan_hash=validation["plan_hash"],groups=group_rows,coverage=coverage,segments=len(segment_rows),policies=sum(len(s["heuristics"]) for s in plan["skills"]),prior_frozen_files_verified=len(previous_freeze["files"]),prior_published_files_verified=len(previous_manifest["files"]),input_preprocessing_implemented=False,training_runs=0,coverage_meaning="Supervision candidates under API interval masks; finite native commands are counted separately. No derived-input preprocessing or learned coverage is claimed.",timeline=str(figure.relative_to(ROOT)))
    atomic(run / "review.json", review)
    with (run / "segment_inventory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(segment_rows[0]))
        writer.writeheader()
        writer.writerows(segment_rows)
    print(json.dumps(review, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
