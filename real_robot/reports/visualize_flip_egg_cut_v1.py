"""Render submitted API intervals and supervision without altering the design."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def main():
    cfg = json.loads((ROOT / "real_robot/configs/flip_egg_cut_v1.json").read_text())
    output = ROOT / cfg["output"]
    plan = json.loads((output / "plan.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    times = {}
    for tid in cfg["trajectory_ids"]:
        with np.load(Path(cfg["source_root"]) / tid / "obs.npz", allow_pickle=False) as z:
            t = z["t"] - z["t"][0]
            # Plotting the last half-open bin uses its recorded median spacing.
            # This visual edge does not create an additional observation.
            times[tid] = np.r_[t, t[-1] + np.median(np.diff(t))]
    groups = plan["skills"]
    palette = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(14, max(9, len(times) * 0.53)))
    width = 0.72 / len(groups)
    total_anchors = 0
    for row, (tid, t) in enumerate(times.items()):
        ax.hlines(row, 0, t[-1], color="#d7dde4", lw=0.6, zorder=0)
        for j, group in enumerate(groups):
            y = row - 0.36 + j * width
            color = palette(j % 10)
            for segment in group["segments"]:
                if segment["trajectory_id"] != tid:
                    continue
                start, stop = segment["start"], segment["stop"]
                ax.broken_barh([(t[start], t[stop] - t[start])], (y, width * 0.83),
                               facecolors=color, alpha=0.24, edgecolors=color,
                               linewidth=0.5)
                kind = segment["supervision_kind"]
                if kind == "sparse":
                    indices = segment["decision_indices"]
                    total_anchors += len(indices)
                    ax.scatter(t[indices], np.full(len(indices), y + width * 0.4),
                               marker="|", color=color, s=33, linewidths=1.1, zorder=3)
                elif kind == "dense":
                    mask = np.zeros(len(t) - 1, dtype=bool)
                    mask[segment["supervised_start"]:segment["supervised_stop"]] = True
                    for exclusion in segment["supervision_exclusions"]:
                        mask[exclusion["start"]:exclusion["stop"]] = False
                    total_anchors += int(mask.sum())
                    edges = np.diff(np.r_[False, mask, False].astype(int))
                    for lo, hi in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                        ax.broken_barh([(t[lo], t[hi] - t[lo])], (y, width * 0.83),
                                       facecolors=color, alpha=0.9)
        for exclusion in plan["exclusions"]:
            if exclusion["trajectory_id"] == tid:
                lo, hi = exclusion["start"], exclusion["stop"]
                ax.broken_barh([(t[lo], t[hi] - t[lo])], (row - 0.36, 0.72),
                               facecolors="#dddddd", hatch="///", alpha=0.4)
    ax.set_yticks(range(len(times)), [tid.removeprefix("episode_") for tid in times], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Recorded robot time from episode start (seconds)")
    ax.set_ylabel("Original episode (separate sequences)")
    ax.grid(axis="x", alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Flip egg: API-submitted cuts and learning decisions", loc="left", weight="bold", pad=45)
    ax.legend(handles=[Patch(facecolor=palette(j % 10), label=g["skill_id"] +
                             (" [executor]" if g["component_kind"] == "supplied_executor" else " [learned]"))
                       for j, g in enumerate(groups)],
              loc="lower left", bbox_to_anchor=(0, 1.005), frameon=False,
              ncol=min(3, len(groups)), fontsize=8)
    fig.text(0.12, 0.013, "Light bars: retained evidence/context. Dark bars: dense supervision. Ticks: sparse decision anchors.\n"
             "Intervals and labels are unchanged API output. Overlap is shown on separate group lanes; no trajectories are concatenated.",
             fontsize=8, color="#505b66")
    fig.tight_layout(rect=(0, 0.052, 1, 1))
    destination = ROOT / "real_robot/reports/flip_egg_cut_v1"
    destination.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(destination / f"timeline.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    expected = sum(c["supervision_assignment_records"] for c in manifest["checks"]["coverage"].values())
    assert total_anchors == expected
    (destination / "figure_receipt.json").write_text(json.dumps(dict(
        source_plan_hash=manifest["plan_hash"], episodes=len(times), groups=len(groups),
        plotted_supervision_assignments=total_anchors, expected_supervision_assignments=expected,
        time_axis="Original robot timestamps, per-episode origin; last bin edge uses median recorded spacing for display only",
        scientific_design_mutations=0,
    ), indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
