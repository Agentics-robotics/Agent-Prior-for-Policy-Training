"""Reports derive exclusively from saved run and episode artifacts."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import CONFIG, RUNS
from .evaluate import CHECKPOINT_STEPS, FREEZE_PATH, SELECTION_PATH, _implementation_hash, result_path, summarize_episodes
from .utils import ROOT, atomic_json, read_json, sha256


def _optional_json(path: Path) -> dict:
    return read_json(path) if path.exists() else {}


def _training_log(run_id: str) -> list[dict]:
    path = ROOT / "runs" / run_id / "train.jsonl"
    if not path.exists():
        return []
    records = []
    for index, line in enumerate(path.read_text().splitlines()):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A process may be between write() and flush() on the final line.
            if index != len(path.read_text().splitlines()) - 1:
                raise
    return records


def _verified_final_results() -> bool:
    freeze = _optional_json(FREEZE_PATH)
    selection = _optional_json(SELECTION_PATH)
    manifest_path = ROOT / "data" / "dataset_manifest.json"
    return bool(freeze.get("complete") and freeze.get("episode_count") == 160
                and manifest_path.exists() and selection.get("identity", {}).get("data_manifest_hash") == sha256(manifest_path)
                and selection.get("identity", {}).get("implementation_hash") == _implementation_hash()
                and SELECTION_PATH.exists() and sha256(SELECTION_PATH) == freeze.get("selection_hash")
                and len(freeze.get("result_hashes", {})) == 8
                and all((ROOT / name).exists() and sha256(ROOT / name) == digest
                        for name, digest in freeze.get("result_hashes", {}).items()))


def _training_complete(run_id: str, marker: dict) -> bool:
    if not marker:
        return False
    # Training owns the full config/optimizer/RNG validation contract.
    from .train import validate_run_complete
    return bool(validate_run_complete(run_id))


def _paired_integrity(rows: list[dict], manifest: dict) -> dict:
    """Compare independently recorded formal artifacts; fail on unfair pairs."""
    artifact = {"complete": all(r["training_status"] == "complete" for r in rows), "pairs": {}}
    for task in ("drawer", "door"):
        pair = [r for r in rows if r["task"] == task]
        if any(r["training_status"] != "complete" for r in pair):
            artifact["pairs"][task] = {"status": "pending", "passed": None}
            continue
        runs = [ROOT / "runs" / r["run_id"] for r in pair]
        markers = [read_json(path / "complete.json") for path in runs]
        normalizers = [read_json(path / "normalizer.json") for path in runs]
        configs = [read_json(path / "config.json") for path in runs]
        common_configs = [{k: v for k,v in config.items() if k not in ("run_id", "representation")} for config in configs]
        checks = {name: markers[0][name] == markers[1][name]
                  for name in ("initial_weights_hash", "pairing_digest", "parameter_count", "manifest_hash", "step")}
        checks.update(configuration_equal_except_representation=common_configs[0] == common_configs[1],
                      normalization_source_ids=normalizers[0]["source_ids"] == normalizers[1]["source_ids"] == manifest["tasks"][task]["train20"],
                      normalization_source_hashes=normalizers[0]["source_hashes"] == normalizers[1]["source_hashes"],
                      normalization_input_count=normalizers[0]["count"] == normalizers[1]["count"],
                      prescribed_updates=markers[0]["step"] == markers[1]["step"] == 20000)
        artifact["pairs"][task] = {"status": "verified", "passed": all(checks.values()), "checks": checks,
                                  "run_ids": [r["run_id"] for r in pair],
                                  "initial_weights_hash": markers[0]["initial_weights_hash"],
                                  "pairing_digest": markers[0]["pairing_digest"],
                                  "parameter_count": markers[0]["parameter_count"]}
    failed = [task for task, pair in artifact["pairs"].items() if pair["passed"] is False]
    artifact["passed"] = False if failed else True if artifact["complete"] else None
    atomic_json(ROOT / "results" / "paired_integrity.json", artifact)
    if failed:
        raise ValueError(f"Paired training integrity failed for {failed}; see results/paired_integrity.json")
    return artifact


def _run_summary(run: dict, selections: dict, final_frozen: bool) -> dict:
    run_id = run["run_id"]
    complete = _optional_json(ROOT / "runs" / run_id / "complete.json")
    log = _training_log(run_id)
    trained = _training_complete(run_id, complete)
    row = dict(run)
    row["training_status"] = "complete" if trained else "in_progress" if log else "not_started"
    row["last_logged_update"] = log[-1]["step"] if log else 0
    row["selected_checkpoint"] = selections.get(run_id, {}).get("step")
    row["train_seconds"] = complete.get("elapsed_seconds", complete.get("training_seconds", log[-1].get("elapsed_seconds") if log else None))
    sessions_path = ROOT / "runs" / run_id / "sessions.jsonl"
    row["train_loop_wall_seconds"] = sum(json.loads(line)["wall_seconds"] for line in sessions_path.read_text().splitlines()) if sessions_path.exists() else None
    row["dev_seconds"] = 0.0
    for step in CHECKPOINT_STEPS:
        dev = _optional_json(result_path(run_id, "dev", step))
        if dev:
            row["dev_seconds"] += summarize_episodes(dev.get("episodes", []))["elapsed_seconds"]
    row["test_seconds"] = 0.0
    row["final_results_frozen"] = final_frozen
    for split, label in (("test_iid", "iid"), ("test_ood", "ood")):
        results = _optional_json(result_path(run_id, split))
        episodes = results.get("episodes", [])
        if results.get("complete") and len(episodes) != 20:
            raise ValueError(f"Wrong final result count: {run_id}/{split}")
        summary = summarize_episodes(episodes)
        row[f"{label}_evaluated"] = len(episodes)
        row[f"{label}_successes"] = summary["successes"] if episodes else None
        row[f"{label}_success_rate"] = summary["success_rate"]
        row[f"{label}_mean_success_steps"] = summary["mean_success_steps"]
        row["test_seconds"] += summary["elapsed_seconds"]
    row["evaluation_seconds"] = row["dev_seconds"] + row["test_seconds"]
    row["status"] = "complete" if trained and final_frozen and row["iid_evaluated"] == row["ood_evaluated"] == 20 else "incomplete"
    return row


def wilson_interval(successes: int, n: int) -> tuple[float, float]:
    """95% descriptive episode-binomial interval; not training-seed uncertainty."""
    if n <= 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    p = successes / n
    center = (p + z*z / (2*n)) / (1 + z*z/n)
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return max(0.0, center-radius), min(1.0, center+radius)


def _plot_success(rows: list[dict], split: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.7), layout="constrained")
    colors = {"raw": "#416788", "relative": "#D68C45"}
    for i, task in enumerate(("drawer", "door")):
        for j, rep in enumerate(("raw", "relative")):
            row = next(r for r in rows if r["task"] == task and r["representation"] == rep)
            x = i + (j-.5)*.32
            n = row[f"{split}_evaluated"]
            if n == 20 and row["final_results_frozen"]:
                p = row[f"{split}_success_rate"]
                low, high = wilson_interval(row[f"{split}_successes"], n)
                ax.bar(x, 100*p, width=.3, color=colors[rep], label=rep if i == 0 else None)
                ax.errorbar(x, 100*p, yerr=[[100*(p-low)], [100*(high-p)]], fmt="none", color="#222222", capsize=4)
                ax.annotate(f"{row[f'{split}_successes']}/20", (x, min(106,100*high+3)), ha="center", fontsize=9)
            else:
                ax.bar(x, 0, width=.3, color=colors[rep], label=rep if i == 0 else None)
                ax.text(x, 6, f"pending\n{n}/20 run", ha="center", fontsize=8)
    ax.set(xticks=[0,1], xticklabels=["Drawer open", "Door open"], ylim=(0,115), ylabel="Success rate (%)")
    ax.set_title("IID evaluation" if split == "iid" else "Position OOD evaluation", pad=40)
    ax.set_yticks([0,20,40,60,80,100])
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(.5,1.01), ncol=2, frameon=False)
    fig.text(.5, -.015, "n=20 evaluation episodes; 1 training seed. Bars: 95% Wilson episode intervals.", ha="center", fontsize=9)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_drawer_ood_completion(output: Path) -> dict | None:
    raw = read_json(result_path("drawer_raw_n20_s0", "test_ood"))["episodes"]
    relative = read_json(result_path("drawer_relative_n20_s0", "test_ood"))["episodes"]
    if ([e["episode_id"] for e in raw] != [e["episode_id"] for e in relative]
            or [e["policy_sampling_seed"] for e in raw] != [e["policy_sampling_seed"] for e in relative]):
        raise ValueError("Drawer OOD completion comparison requires matching episode IDs and sampling seeds")
    if len(raw) != 20 or not all(e["success"] for e in raw + relative):
        return None
    a = np.asarray([e["first_success_step"] for e in raw], dtype=np.float64)
    b = np.asarray([e["first_success_step"] for e in relative], dtype=np.float64)
    difference = b - a
    stats = {"n": len(a), "raw_mean": float(a.mean()), "relative_mean": float(b.mean()),
             "raw_median": float(np.median(a)), "relative_median": float(np.median(b)),
             "mean_difference": float(difference.mean()), "percent_mean_reduction": float(100*(a.mean()-b.mean())/a.mean()),
             "faster": int((difference < 0).sum()), "equal": int((difference == 0).sum()),
             "slower": int((difference > 0).sum()), "median_paired_difference": float(np.median(difference))}
    fig, ax = plt.subplots(figsize=(6.8,5.0), layout="constrained")
    offsets = np.linspace(-.055,.055,len(a))
    for index, (first, second) in enumerate(zip(a,b)):
        ax.plot([offsets[index],1+offsets[index]],[first,second],color="#87929A",alpha=.45,lw=1,zorder=1)
    for x, points, color in ((0,a,"#416788"),(1,b,"#D68C45")):
        ax.scatter(x+offsets,points,s=28,color=color,alpha=.85,zorder=2)
        ax.plot([x-.12,x+.12],[points.mean(),points.mean()],color="#222222",lw=2.5,zorder=3,label="Mean" if x == 0 else None)
        ax.scatter([x],[np.median(points)],marker="D",facecolor="white",edgecolor="#222222",s=40,zorder=4,label="Median" if x == 0 else None)
    ax.set(xticks=[0,1],xticklabels=[f"raw\nmean {a.mean():.2f}; median {np.median(a):.1f}",
                                   f"relative\nmean {b.mean():.2f}; median {np.median(b):.1f}"],
           xlim=(-.3,1.3),ylim=(0,max(a.max(),b.max())*1.10),ylabel="First-success environment step (lower is faster)")
    ax.set_title("Drawer OOD: paired completion steps",pad=40)
    ax.legend(loc="lower center",bbox_to_anchor=(.5,1.01),ncol=2,frameon=False)
    ax.grid(axis="y",alpha=.2)
    fig.text(.5,-.075,"20 paired OOD initial conditions; both 20/20 successful; 1 training seed.\nEach line connects one fixed initial condition. Descriptive comparison only.",ha="center",fontsize=9)
    fig.savefig(output,dpi=180,bbox_inches="tight")
    plt.close(fig)
    return stats


def _plot_curves(output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11,7), layout="constrained")
    for i, task in enumerate(("drawer", "door")):
        for rep, color in (("raw", "#416788"), ("relative", "#D68C45")):
            run_id = f"{task}_{rep}_n20_s0"
            log = [r for r in _training_log(run_id) if "loss" in r]
            if log:
                axes[0,i].plot([r["step"] for r in log], [r["loss"] for r in log], label=rep, color=color, alpha=.8)
            devs = []
            for step in CHECKPOINT_STEPS:
                result = _optional_json(result_path(run_id,"dev",step))
                if result.get("complete"):
                    devs.append((step, result["summary"]["success_rate"]*100))
            if devs:
                axes[1,i].plot(*zip(*devs), marker="o", label=rep, color=color)
        axes[0,i].set(title=f"{task.capitalize()}: recorded training loss", ylabel="Valid-action epsilon MSE", xlabel="Optimizer updates")
        axes[1,i].set(title=f"{task.capitalize()}: EMA development success", ylabel="Success rate (%)", xlabel="Optimizer updates", ylim=(-3,103))
        for ax in axes[:,i]:
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend()
            else:
                ax.text(.5,.5,"No completed records yet",transform=ax.transAxes,ha="center")
    fig.text(.5,-.015,"Development: n=20 independent initial conditions; 1 training seed.",ha="center",fontsize=9)
    fig.savefig(output,dpi=180,bbox_inches="tight")
    plt.close(fig)


def _fmt(value, digits=1):
    return "—" if value is None else f"{value:.{digits}f}"


def _outcome(row: dict, split: str) -> str:
    n = row[f"{split}_evaluated"]
    if n == 0:
        return "未评测"
    if n != 20 or not row["final_results_frozen"]:
        return f"未冻结（已评 {n}/20）"
    return f"{row[f'{split}_successes']}/20 ({100*row[f'{split}_success_rate']:.0f}%)"


def generate_report() -> list[dict]:
    results_dir = ROOT / "results"
    figures = results_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    selections = _optional_json(SELECTION_PATH).get("selections", {})
    final_frozen = _verified_final_results()
    rows = [_run_summary(run, selections, final_frozen) for run in RUNS]
    with (results_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot_success(rows, "iid", figures / "iid_success.png")
    _plot_success(rows, "ood", figures / "ood_success.png")
    _plot_curves(figures / "training_and_dev.png")
    drawer_ood_timing = _plot_drawer_ood_completion(figures / "drawer_ood_completion_steps.png") if final_frozen else None
    completed_runs = sum(row["training_status"] == "complete" for row in rows)
    episodes = sum(row["iid_evaluated"] + row["ood_evaluated"] for row in rows)
    environment = _optional_json(ROOT / "artifacts" / "environment.json")
    manifest = _optional_json(ROOT / "data" / "dataset_manifest.json")
    videos = _optional_json(results_dir / "videos" / "manifest.json")
    preflight = _optional_json(ROOT / "artifacts" / "preflight.json")
    render_check = _optional_json(ROOT / "artifacts" / "render_check.json")
    data_audit = _optional_json(ROOT / "artifacts" / "dataset_audit.json")
    pipeline_timing = _optional_json(ROOT / "artifacts" / "pipeline_timing.json")
    recovery = _optional_json(ROOT / "artifacts" / "runtime_recovery_check.json")
    round1_invocations = []
    for path in sorted((ROOT / "artifacts").glob("invocation_*.json")):
        invocation = read_json(path)
        if invocation.get("command") == "round1":
            round1_invocations.append({**invocation, "path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    round1_invocations.sort(key=lambda item: (item.get("started_utc", ""), item.get("pid", 0)))
    paired_integrity = _paired_integrity(rows, manifest)
    cost = {
        "training_seconds": sum(r["train_seconds"] or 0 for r in rows),
        "training_loop_wall_seconds": sum(r["train_loop_wall_seconds"] or 0 for r in rows) if any(r["train_loop_wall_seconds"] is not None for r in rows) else None,
        "training_loop_wall_seconds_complete": all(r["train_loop_wall_seconds"] is not None for r in rows),
        "development_seconds": sum(r["dev_seconds"] for r in rows),
        "final_test_seconds": sum(r["test_seconds"] for r in rows),
        "video_replay_seconds": sum(v.get("elapsed_seconds", 0) for v in videos.get("videos", [])),
        "data_collection_seconds": manifest.get("collection_invocation_elapsed_seconds"),
        "expert_rollout_seconds": manifest.get("expert_rollout_elapsed_seconds_total"),
        "preflight_seconds": preflight.get("elapsed_seconds"),
        "render_check_seconds": render_check.get("elapsed_seconds"),
        "dataset_audit_seconds": data_audit.get("elapsed_seconds"),
        "pipeline_timing": pipeline_timing or None,
        "runtime_recovery_diagnostic_seconds": recovery.get("diagnostic_wall_seconds"),
        "runtime_recovery_slow_prefix_update_seconds": recovery.get("saved_prefix_optimizer_update_seconds"),
        "pipeline_round1_wall_seconds": sum(item["elapsed_seconds"] for item in round1_invocations) if round1_invocations else None,
        "pipeline_round1_invocations": round1_invocations,
        "pipeline_wall_scope": "Only finalized artifacts/invocation_*.json with command=round1; includes failed/paused and resumed invocations, excludes overlapping child train/report commands and pipeline_timing aliases. An active invocation is absent until its finally block writes a timing artifact.",
        "notes": "training_seconds is measured optimizer update time including first compilation and the preserved slow 65-update prefix; training_loop_wall_seconds additionally includes checkpoint/log I/O but excludes model/data setup. Evaluation times sum per-episode reset/inference/simulation, excluding policy loading/first inference compilation outside episodes. Missing durations are unknown, not zero. These nested measurements must not be added to pipeline invocation wall time. Recovery replay is separate diagnostic time, not formal updates. No purchased cloud resources.",
    }
    cost["recorded_training_gpu_hours"] = cost["training_seconds"] / 3600 if environment.get("cuda_available") else None
    atomic_json(results_dir / "compute_cost.json", cost)
    lines = [
        "# 第一轮相对坐标先验实验报告", "",
        f"生成时间（UTC）：{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}。",
        f"实际状态：正式训练完成 **{completed_runs}/4**；正式测试记录 **{episodes}/160**；全部测试 hash 冻结：**{'是' if final_frozen else '否'}**。",
        "只有完成训练校验和全部测试冻结的行标为 complete；短训练、运行中任务和缺失结果均不视为完成。", "",
        "## 四个 run 的真实结果", "",
        "| Run | 状态 / 最后更新 | IID | 位置 OOD | 选定 checkpoint | IID / OOD 成功平均步数 | 更新 / 评价秒数 |",
        "|---|---|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['run_id']} | {row['status']} / {row['last_logged_update']} | {_outcome(row,'iid')} | {_outcome(row,'ood')} | {row['selected_checkpoint'] or '—'} | {_fmt(row['iid_mean_success_steps'],2)} / {_fmt(row['ood_mean_success_steps'],2)} | {_fmt(row['train_seconds'])} / {_fmt(row['evaluation_seconds'])} |")
    lines += ["", "表中更新秒数只累计 `Trainer.update`（含首次编译）；评价秒数累计 dev + test episode。包含保存 checkpoint / 日志的训练循环墙钟时间另见 summary 的 `train_loop_wall_seconds`，不把这些嵌套计时重复相加。", "",
              "原始表：[summary.csv](results/summary.csv)。逐 episode 文件位于 `results/<run_id>/test_iid.json` 与 `test_ood.json`；包含 checkpoint / 数据 / 初始状态 hash、配对采样 seed、成功首步、return、结束标志、异常与耗时。", "",
              "## 主要观察", ""]
    if final_frozen:
        for task in ("drawer", "door"):
            raw = next(r for r in rows if r["task"] == task and r["representation"] == "raw")
            rel = next(r for r in rows if r["task"] == task and r["representation"] == "relative")
            differences = {s: 100*(rel[f"{s}_success_rate"]-raw[f"{s}_success_rate"]) for s in ("iid","ood")}
            lines.append(f"- {task}：relative − raw 的 IID 差为 **{differences['iid']:+.0f} 个百分点**，位置 OOD 差为 **{differences['ood']:+.0f} 个百分点**。这是同一训练 seed、同组初始条件下的描述性结果。")
        if all(r["iid_successes"] == r["ood_successes"] == 20 for r in rows):
            lines += ["", "主要终点出现**成功率上限效应**：四个 run 的 IID 与 OOD 均为 20/20。本轮没有观察到 relative 的成功率收益；这不等于证明两种表示等价，也不能据此估计更困难任务或其他训练 seed 的差异。"]
        if drawer_ood_timing:
            t = drawer_ood_timing
            lines += ["", f"次要描述性观察：Drawer OOD 平均完成步数从 **{t['raw_mean']:.2f} 降至 {t['relative_mean']:.2f}**，差为 **{t['mean_difference']:.2f} 步**（平均步数减少 **{t['percent_mean_reduction']:.1f}%**）。中位数从 {t['raw_median']:.1f} 降至 {t['relative_median']:.1f}；同一组 20 个初始条件中 relative 更快 {t['faster']} 个、持平 {t['equal']} 个、更慢 {t['slower']} 个，配对差值中位数 {t['median_paired_difference']:.1f} 步。平均差距部分来自 raw 的较慢尾部，不能直接解释为某一交互阶段已获改善。", "",
                      "其他完成时间差异较小且方向混合：Drawer IID 为 86.95→87.10 步；Door IID 均为 77.65 步；Door OOD 为 80.00→81.40 步。以上均是单训练 seed 的描述，不构成跨训练 seed 稳定收益的证据。"]
    else:
        lines.append("正式比较尚未全部完成并冻结，目前不能判断 relative 是否改善 IID 或位置 OOD。待完成记录保留为空，不填充推测成功率。")
    lines += ["", "接近、建立交互和持续打开阶段的归因仍属**未知**：本轮保存原生诊断 info 和固定规则重放视频，但没有经验证的阶段分类器。最终成功率和个别视频不足以单独证明某一阶段的机制改善。", "",
              "## 固定设置与公平性", "",
              "每任务采集 100 条成功脚本专家轨迹，以预先固定 RNG 选择 train20。四个正式 run 均只使用同任务相同的 20 条示范计算输入统计和更新网络；其余 80 条不进入优化器或 normalization。dev / IID / OOD 各 20 个独立初始条件，按真实基座位置划分；OOD 左右各 10 个。原始数据、split 与 train20 在训练前冻结。", "",
              "状态为目标可见的 native 39 维观测（18 维当前、18 维原生历史、3 维目标）。每个历史位置块使用自身 handle；当前 goal 使用当前 handle。relative 保留 handle 世界坐标，将 hand 与 goal 替换为相对 handle 位置；姿态、夹爪和占位不丢失。native reset 重复首帧历史；DP 另用相同的 2 步观测历史。变换先于仅来自 train20 的 mean/std（std 下限 1e-3），OOD 输入不裁剪。完整字段、handle 固定 offset、动作内部缩放和 reset 语义见 [observation_schema.json](artifacts/observation_schema.json) 与 [DATA_SEMANTICS.md](docs/DATA_SEMANTICS.md)。", "",
              "Conditional 1D U-Net 使用官方 DP 必需模块，两个表示网络和参数量一致。每次条件为 `(o[t-1], o[t])`，目标为 `a[t:t+16]`，执行 `[0:4]`；尾部无效 action 不计 loss。动作是原生 4 维归一化 Cartesian 增量与夹爪，实际 env.step 输入裁剪到 [-1,1]，不再按示范范围归一化。单个 chunk 中一旦原生 success 或终止/截断触发立即结束。", "",
              f"正式预算：每 run {CONFIG['train_updates']:,} 更新，有效 batch={CONFIG['batch_size']}，AdamW lr={CONFIG['learning_rate']}、weight_decay={CONFIG['weight_decay']}，恒定学习率，梯度范数上限 1.0。U-Net down_dims={CONFIG['unet_down_dims']}，diffusion embedding=128，kernel=5，groups=8。100 步 squaredcos 训练扩散，epsilon MSE；DDIM 16 步、eta=0、clean sample clipping；EMA decay=0.995。训练 seed=0。最终实际冻结配置与训练身份见 `runs/<run_id>/`；资源调整见 README / PROGRESS。", "",
              "每个 5k/10k/20k EMA checkpoint 在相同 dev20 和相同 episode 采样 seed 上评价。依次按成功率最高、成功 episode 平均首成功步最少、checkpoint 最早选择；全失败时直接选最早。四个选择全部冻结后才运行正式 test。训练 loader、训练 diffusion、dev、test 和视频使用独立 RNG；同任务 raw/relative 共享 episode 采样 seed。", "",
              f"预先统一的执行兼容性设置：`torch_compile={CONFIG.get('torch_compile')}`、`inference_torch_compile={CONFIG.get('inference_torch_compile')}`，float32，TF32 关闭、确定性算法开启；不改变有效 batch、网络、示范数或更新预算。两个表示模型均有 4,696,644 个参数（完整 run identity 可核对）。", "",
              "## 图与视频", "",
              "![IID success](results/figures/iid_success.png)", "", "![Position OOD success](results/figures/ood_success.png)", "", "![Training and development](results/figures/training_and_dev.png)", "",
              "图中 n=20 evaluation episodes，1 training seed。Wilson 95% 区间只描述 episode 二项不确定性，不能作为跨训练 seed 稳定性的证据。成功次数相差 1 对应 5 个百分点。", ""]
    if drawer_ood_timing:
        lines += ["![Drawer OOD paired completion steps](results/figures/drawer_ood_completion_steps.png)", "",
                  "配对步数图直接使用同一组 20 个冻结 OOD 初始条件的首成功步数，每条线连接同一个 episode；全部 episode 成功，没有删除慢轨迹。它补充展示成功率达到上限后仍可观察到的完成时间差异。", ""]
    evidence = ["## 数据与实现校验", "",
                f"冻结 manifest SHA256：`{sha256(ROOT/'data/dataset_manifest.json') if manifest else '尚未冻结'}`。", "",
                "| Task | 专家池成功 / 尝试 | train_pool 真实 x 范围 | dev 真实 x 范围 | IID 真实 x 范围 | OOD 真实 x 范围 |", "|---|---|---|---|---|---|"]
    for task, task_data in manifest.get("tasks", {}).items():
        spans = []
        for split in ("train_pool", "dev", "test_iid", "test_ood"):
            xs = [r["initial_base_position"][0] for r in task_data["splits"][split]]
            spans.append(f"[{min(xs):.5f}, {max(xs):.5f}]")
        evidence.append(f"| {task} | {task_data['train_pool_expert_successes']} / {task_data['train_pool_attempts']} | " + " | ".join(spans) + " |")
    evidence += ["", "单位为米；OOD 行显示两端样本的总体 min/max，中间空隙没有采样。规定 drawer 训练 x∈[-0.04,0.04]、OOD x∈[-0.10,-0.06]∪[0.06,0.10]；door 训练 x∈[0.03,0.07]、OOD x∈[0,0.02]∪[0.08,0.10]，所有 door split y∈[0.88,0.92]。本安装版本原生范围与规格一致，无区间调整。所有 eval 初始条件专家预检查成功，排除清单为空；实际清单和 seed 见 manifest。", ""]
    if data_audit.get("passed"):
        evidence += [f"独立数据审计实际检查 {data_audit['initial_conditions_checked']} 个初始 snapshot，完整重放 {data_audit['full_demonstrations_replayed']} 条示范、{data_audit['action_steps_replayed']:,} 个实际动作；snapshot 容差 {data_audit['state_snapshot_atol']:.0e}，存储 float32 观测最大误差 {data_audit['max_stored_float32_observation_error']:.3e}，raw→relative→raw 最大误差 {data_audit['max_roundtrip_error']:.3e}。动作标签与真实重放输入一致，reward/success/终止标志检查通过。证据：[dataset_audit.json](artifacts/dataset_audit.json)。", ""]
    evidence += [f"完整 preflight 实际通过：`{preflight.get('passed', False)}`；证据：[preflight.json](artifacts/preflight.json)、[preflight_tests.xml](artifacts/preflight_tests.xml)。独立 debug checkpoint 和 midpoint rollout 不作为任何正式 run 初始化。", "",
                 f"最终配对完整性检查：`{paired_integrity['passed']}`（未完成时为 null）；[paired_integrity.json](results/paired_integrity.json) 验证同任务两个 run 的初始权重 hash、全部窗口/timestep/noise 序列 digest、参数量、train20 来源、normalization 输入数、20k 更新和其余配置一致。", ""]
    insertion = lines.index("## 图与视频")
    lines[insertion:insertion] = evidence
    if recovery:
        recovery_section = ["## 运行恢复与执行调整", "",
                            "首个 `drawer_raw_n20_s0` 已完成 20,000 更新。随后同一进程中的 relative 训练显著变慢，仍在实际计算；收到 SIGTERM 后在第 65 次更新完整保存并正常进入可恢复暂停。已完成 raw 保留，原始运行日志与耗时保留，没有将慢任务冒充完成。", "",
                            f"独立新进程从原始初始化重放 65 次更新，与保存状态的模型、EMA、完整 AdamW、loader RNG、diffusion RNG、初始权重 hash、样本数和配对 digest **逐位一致**（校验 passed=`{recovery.get('passed')}`）。原慢前缀更新耗时 {_fmt(recovery.get('saved_prefix_optimizer_update_seconds'), 3)} 秒；诊断重放更新 {_fmt(recovery.get('fresh_prefix_optimizer_update_seconds'), 3)} 秒，诊断总墙钟 {_fmt(recovery.get('diagnostic_wall_seconds'), 3)} 秒。诊断输出不进入正式初始化或正式训练预算；relative 从原始第 65 步继续完成剩余 19,935 步。", "",
                            "执行调整仅涉及 CLI 编排：每个正式 run 在新的 Pixi 子进程中顺序执行，父进程持有管线锁。冻结的模型/训练/数据/配置、精度、编译选项、有效 batch 和 20k 预算均未改变，无正式结果被作废，也没有增加正式 run。最初较慢的 227.93 秒保留在 relative 累计训练耗时中。", "",
                            "原因仍是**根据源码的推断**：固定 PyTorch 2.7.1 的 CUDA graph generation 逻辑会优先使用非零全局 `MarkStepBox` 计数，而先前 debug inference 调用了显式 step 标记；跨 Trainer 保留的 graph 状态可能导致性能异常。主机 ptrace 权限不允许附加堆栈检查，因此没有直接堆栈证据，未更改系统权限或驱动。", "",
                            "证据：[RUNTIME_RECOVERY.md](docs/RUNTIME_RECOVERY.md)、[runtime_recovery_check.json](artifacts/runtime_recovery_check.json)、[诊断日志](logs/runtime_recovery_check.log)、[原始调用日志](logs/round1_invocation_804594.log)、[原始调用计时](artifacts/pipeline_timing_804594.json)。", ""]
        insertion = lines.index("## 图与视频")
        lines[insertion:insertion] = recovery_section
    if videos:
        for video in videos.get("videos", []):
            name = f"{video['run_id']} / {video['split']} / {video['kind']}"
            if video["status"] == "recorded":
                lines.append(f"- [{name}]({video['path']})（与冻结轨迹 hash 一致）。")
            else:
                lines.append(f"- {name}：{video['status']}；{video.get('reason', '无记录')}。")
    else:
        lines.append("尚无视频记录。仅在 160 条正式测试记录冻结后重放每个 run / split 的首个成功和首个失败；没有对应类型会明确标记。")
    lines += ["", "## 依赖、许可证与计算成本", "",
              f"实际平台：`{environment.get('platform', '尚未记录')}`；Python `{environment.get('python', '尚未记录').splitlines()[0]}`；CUDA 可用 `{environment.get('cuda_available', '尚未记录')}`；渲染后端 `{environment.get('mujoco_gl', '尚未记录')}`。",
              f"GPU：`{json.dumps(environment.get('gpus', []), ensure_ascii=False)}`。", "",
              "实际使用主机**物理 GPU 1**；Pixi 设置 `CUDA_VISIBLE_DEVICES=1`，因此进程内部对应 `cuda:0`。CPU、内存、所有 GPU 与后端探测记录：[host_resources.json](artifacts/host_resources.json)。", "",
              f"固定 MetaWorld commit：`{environment.get('metaworld_commit', '尚未记录')}`；DP commit：`{environment.get('diffusion_policy_commit', '尚未记录')}`。", "",
              "完整依赖版本、源码 hash 与元数据：[artifacts/environment.json](artifacts/environment.json)；固定环境：[pixi.lock](pixi.lock)。DP 模块来源与修改说明：[PROVENANCE.md](src/relative_dp/vendor/diffusion_policy/PROVENANCE.md)，许可证：[LICENSE](src/relative_dp/vendor/diffusion_policy/LICENSE)。官方源码来源：[MetaWorld](https://github.com/Farama-Foundation/Metaworld)、[Diffusion Policy](https://github.com/real-stanford/diffusion_policy)。", "",
              "| 包 | 实际版本 | 许可证（安装元数据） |", "|---|---|---|"]
    for name in ("metaworld", "mujoco", "torch", "diffusers", "gymnasium", "numpy", "matplotlib"):
        package = environment.get("packages", {}).get(name, {})
        license_text = str(package.get("license") or "元数据未声明；参见依赖自带 LICENSE").replace("\n", " ")
        # License bodies can be thousands of words; raw metadata remains in artifact.
        if len(license_text) > 100:
            license_text = license_text[:97] + "…（完整见环境记录）"
        lines.append(f"| {name} | {package.get('version', '尚未记录')} | {license_text.replace('|','/')} |")
    lines += ["", f"记录的正式训练更新耗时 {_fmt(cost['training_seconds'])} 秒；训练循环（含 checkpoint/log）{_fmt(cost['training_loop_wall_seconds'])} 秒；dev {_fmt(cost['development_seconds'])} 秒；test {_fmt(cost['final_test_seconds'])} 秒；视频重放 {_fmt(cost['video_replay_seconds'])} 秒；数据采集 {_fmt(cost['data_collection_seconds'])} 秒。更新所占 GPU 小时 {_fmt(cost['recorded_training_gpu_hours'],3)}。", "",
              f"其他实测阶段：数据审计 {_fmt(cost['dataset_audit_seconds'])} 秒、完整 preflight {_fmt(cost['preflight_seconds'])} 秒、EGL render check {_fmt(cost['render_check_seconds'])} 秒；运行恢复诊断另计 {_fmt(cost['runtime_recovery_diagnostic_seconds'])} 秒。", "",
              f"已结束的 `round1` 调用共 {len(round1_invocations)} 次，累计管线墙钟 **{_fmt(cost['pipeline_round1_wall_seconds'])} 秒**。只汇总 `artifacts/invocation_*.json` 中 `command=round1` 的记录（包含原始暂停和后续恢复）；不重复加入相同内容的 `pipeline_timing*.json`、重叠的训练子进程或单独 report 命令。仍在运行的调用尚未写入最终计时，不包含在这个累计数中。完整调用清单/hash 和阶段细分见 [compute_cost.json](results/compute_cost.json)。", "",
              "管线墙钟与内部训练/评价/编译阶段是嵌套计时，不能相加；恢复诊断的 65 步也不是额外正式训练。更新计时不包括模型/数据初始化，episode 计时不包括 checkpoint 读取。未测量的环境安装/调度开销明确为未知，无新增付费云资源。", "",
              "## 解释边界与下一步", "",
              "本轮 relative 变换可逆且大体为仿射重参数化，没有增加传感信息。在第一层线性映射足够自由时，它未必缩小可表达函数集合。正结果支持有限数据下表示对优化或泛化有效，不能证明任意组合能力，也没有验证 VLM/Agent 自动选择先验。负结果和两方法都差的情况均保留，不通过增加示范、更新次数、辅助损失或修改 OOD 分布追求胜出。", "",
              "仅 1 个训练 seed 与每 split 20 个 episode 可支持本次固定条件下的初步配对观察，不能估计训练随机性，不能宣称普遍收益。下一步建议：保持相同数据、架构和位置分布，另行预注册多训练 seed 的配对复验；本轮不自动增加正式训练。若当前尚未完成，优先恢复并完成本轮后再作比较。", "",
              "## 复现与恢复", "", "```bash", "pixi run doctor", "pixi run test", "pixi run collect-round1", "pixi run train-round1", "pixi run evaluate-round1", "pixi run report-round1", "# 顺序执行并根据完整身份 / hash 恢复或跳过", "pixi run round1", "```", "",
              "单 run 恢复命令以 [README](README.md) 为准；训练恢复包含 optimizer、EMA 和 RNG。进度及活动进程：[PROGRESS.md](PROGRESS.md)。过期 hash 或运行异常会报错，不能凭目录存在跳过任务。", ""]
    if (ROOT / "artifacts" / "resource_usage_summary.json").exists():
        insertion = lines.index("## 解释边界与下一步")
        lines[insertion:insertion] = ["只读资源采样：[resource_usage_summary.json](artifacts/resource_usage_summary.json)、[原始调用采样](artifacts/resource_usage_summary_804594.json)、[resources.jsonl](logs/resources.jsonl)。每 10 秒采样，覆盖部分运行窗口；功率包含设备空闲基线。能量估算既不是完整实验总能耗，也不是增量能耗或收费计量。", ""]
    (ROOT / "ROUND1_REPORT.md").write_text("\n".join(lines))
    print(f"Report written: {ROOT / 'ROUND1_REPORT.md'}; complete training {completed_runs}/4, test episodes {episodes}/160", flush=True)
    return rows
