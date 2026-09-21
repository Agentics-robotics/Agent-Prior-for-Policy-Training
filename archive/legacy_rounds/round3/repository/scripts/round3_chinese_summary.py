"""Render a concise Chinese overview only after validated final test reporting.

Run: pixi run python scripts/round3_chinese_summary.py
The sole pre-gate result read is reports/test_summary.json's final marker. No
other test-derived artifact is accessed until validate_test_gate succeeds.
This CPU-only consumer never trains, evaluates, selects, or changes proposals.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "round3"
SPLITS = ("IID", "C", "E")
LABELS = {"B0": "普通 DP（B0）", "initial_selected": "初始开发选优", "final_system": "最终系统"}


class NotReady(ValueError):
    """No publishable final overview can be produced from the inputs."""


def require(condition, message):
    if not condition:
        raise NotReady(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def final_summary_before_gate(path, gate, reader=read_json):
    """Check the final marker first, then the gate; injectable for guard tests."""
    require(path.is_file(), f"等待最终报告：缺少 {path}；未读取锁定测试文件。")
    summary = reader(path)
    require(summary.get("final") is True, "等待最终报告：test_summary.json 的 final 必须为 true。")
    gate()  # This must precede every other result/selection/report/visual read.
    return summary


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def close(left, right):
    return math.isfinite(float(left)) and math.isclose(float(left), float(right), abs_tol=1e-9)


def row_id(task, candidate, n):
    return f"r3_{task}_{candidate}_n{n}_s0"


def validate_tables(summary, selection, config):
    """Reject pending/inconsistent aggregates, while retaining explicit invalids."""
    tasks, ns = config["tasks"], config["demonstration_counts"]
    require(summary.get("stage") == "test" and summary.get("selection_frozen") is True,
            "需要冻结选择后的 test 汇总。")
    require(selection.get("frozen") is True and not selection.get("unavailable"), "开发选择尚未齐备并冻结。")
    require(not summary.get("pending_training"), "正式训练仍有待完成项目。")
    rows = summary["results"]
    by_id = {row["run_id"]: row for row in rows}
    require(len(by_id) == len(rows) == summary["planned_runs"], "计划数与成绩行不一致。")
    initial_ids = {row_id(t, c, n) for t in tasks for n in ns for c in config["candidates"]}
    require({r["run_id"] for r in rows if r["candidate_id"] != "P4"} == initial_ids,
            "初始主矩阵成绩行不完整。")
    require(len(initial_ids) == config["initial_runs"] == 96, "本脚本要求 Round 3 的 96 项主矩阵。")
    evaluated = [r for r in rows if r["status"] == "evaluated"]
    invalid = [r for r in rows if r["status"] == "invalid"]
    require(len(evaluated) + len(invalid) == len(rows), "成绩仍有 pending 项目。")
    require(all(r.get("reason") for r in invalid), "无效配置缺少原因。")
    require({r["run_id"] for r in invalid} == {r["run_id"] for r in summary["invalid"]}, "无效配置清单不一致。")
    require(len(evaluated) == summary["evaluated_runs"] == summary["completed_formal_runs"], "已训练/已评测数量不一致。")
    require(sum(r["candidate_id"] != "P4" for r in evaluated) == summary["completed_main_runs"], "主矩阵完成数不一致。")
    require(sum(r["candidate_id"] == "P4" for r in evaluated) == summary["completed_p4_runs"], "P4 完成数不一致。")
    require(summary["planned_runs"] <= config["maximum_formal_runs"], "超出正式训练预算。")
    episodes = 0
    for row in evaluated:
        for split in SPLITS:
            count, successes = row[split + "_episodes"], row[split + "_successes"]
            require(count == config["test_episodes"][split] and 0 <= successes <= count,
                    f"{row['run_id']} {split} 测试回合数异常。")
            require(close(row[split + "_success_rate"], successes / count), "成功率与计数不一致。")
            episodes += count
        require(close(row["ood_success_rate"], (row["C_success_rate"] + row["E_success_rate"]) / 2), "OOD 均值不一致。")
        for key in ("train_wall_seconds", "gpu_active_work_seconds", "parameter_count", "mean_latency_ms", "mean_replans"):
            require(math.isfinite(float(row[key])) and row[key] >= 0, f"缺少真实成本字段 {key}。")
        baseline = by_id[row_id(row["task"], "B0", row["train_n"])]
        if baseline["status"] == "evaluated":
            delta = summary["paired_ood_deltas"].get(row["run_id"])
            require(delta is not None, f"缺少配对 OOD 区间：{row['run_id']}")
            expected = 100 * (row["ood_success_rate"] - baseline["ood_success_rate"])
            require(close(delta["delta_pp"], expected) and close(row["ood_delta_B0_pp"], expected), "配对 OOD 差值不一致。")
            low, high = delta["paired_bootstrap_95ci_pp"]
            require(-100 <= low <= high <= 100 and delta["draws"] > 0 and delta.get("scope"), "配对区间无效。")
    require(episodes == summary["scored_episodes"], "测试回合总数不一致。")
    require(set(selection["groups"]) == {f"{t}:n{n}" for t in tasks for n in ns}, "需要完整的 24 组开发选择。")
    for task in tasks:
        for n in ns:
            group = selection["groups"][f"{task}:n{n}"]
            for kind in ("initial_selected", "final_system"):
                chosen = group[kind]
                eligible = [r for r in group["candidates"] if kind == "final_system" or r["candidate_id"] in ("P1", "P2", "P3")]
                expected = min(eligible, key=lambda r: (-r["ood_success_rate"], -r["iid_success_rate"], r["parameter_count"], r["candidate_id"])) if eligible else None
                require(chosen == expected, "冻结选择不符合预定开发规则。")
                if chosen:
                    row = by_id.get(chosen["run_id"])
                    require(row is not None and row["status"] == "evaluated" and row["task"] == task and row["train_n"] == n,
                            f"所选系统缺少完整测试：{task} N{n} {kind}")
                    require(chosen["candidate_id"] == row["candidate_id"], "所选候选 ID 不一致。")
                    require(row["candidate_id"] != "P4" or n in (5, 20), "P4 只能参与 N5/N20。")
            require(group["b0_fallback"] == bool(group["final_system"] and group["final_system"]["candidate_id"] == "B0"), "B0 回退标志不一致。")
    feedback = {r["task"]: r for r in summary["feedback_accounting"]}
    require(set(feedback) == set(tasks), "反馈决策未覆盖全部任务。")
    for task, item in feedback.items():
        p4 = [r for r in rows if r["task"] == task and r["candidate_id"] == "P4"]
        require(item["decision"] in ("P4", "no_revision") and item["feedback_cycles_used"] == 1,
                f"{task} 需要一次已完成的反馈决定。")
        require(item["p4_planned_trainings"] == len(p4) and item["p4_completed_trainings"] == sum(r["status"] == "evaluated" for r in p4), "P4 成本账目不一致。")
        require((not p4) if item["decision"] == "no_revision" else {r["train_n"] for r in p4} == {5, 20}, "反馈决定与 P4 矩阵不一致。")
        for total, field in (("p4_train_wall_seconds", "train_wall_seconds"), ("p4_gpu_active_work_seconds", "gpu_active_work_seconds")):
            require(close(item[total], sum(r[field] for r in p4 if r["status"] == "evaluated")), "P4 训练耗时合计不一致。")
        if p4:
            require(item.get("p4_base_candidate_id") in config["candidates"] and item.get("p4_principal_revision_intent"), "P4 缺少实际修订基线与主要意图。")
            sessions = item["p4_debug_sessions"]
            require(close(item["p4_debug_session_updates"], sum(r["end_step"] - r["start_step"] for r in sessions)) and
                    close(item["p4_debug_session_wall_seconds"], sum(r["wall_seconds"] for r in sessions)), "P4 调试账目不一致。")
    expected_effects = {(t, n, kind) for t, item in feedback.items() if item["decision"] == "P4"
                        for n in (5, 20) for kind in ("base-candidate", "initial-selected")}
    effects = summary.get("feedback_effects", [])
    require(len(effects) == len(expected_effects) and
            {(e["task"], e["train_n"], e["reference_kind"]) for e in effects} == expected_effects, "P4 对照表不完整。")
    for effect in effects:
        task, n = effect["task"], effect["train_n"]
        chosen = selection["groups"][f"{task}:n{n}"]["initial_selected"]
        reference_id = row_id(task, feedback[task]["p4_base_candidate_id"], n) if effect["reference_kind"] == "base-candidate" else chosen["run_id"] if chosen else None
        require(effect["reference_run_id"] == reference_id and effect["p4_run_id"] == row_id(task, "P4", n), "P4 修订基线与开发选优对照混淆。")
        p4, reference = by_id[effect["p4_run_id"]], by_id.get(reference_id)
        for prefix, row in (("p4", p4), ("reference", reference)):
            require(close(effect[prefix + "_ood_success_rate"], row["ood_success_rate"]) if row and row["status"] == "evaluated" else
                    effect[prefix + "_ood_success_rate"] is None, "P4 对照成功率与成绩不一致。")
        delta = effect["paired_ood_delta"]
        if p4["status"] == "evaluated" and reference and reference["status"] == "evaluated":
            require(effect["reference_candidate_id"] == reference["candidate_id"], "P4 对照候选 ID 不一致。")
            require(delta and close(delta["delta_pp"], 100*(p4["ood_success_rate"]-reference["ood_success_rate"])), "P4 对照差值不一致。")
            low, high = delta["paired_bootstrap_95ci_pp"]
            require(-100 <= low <= high <= 100 and delta["draws"] > 0 and delta.get("scope"), "P4 配对区间无效。")
        else:
            require(delta is None, "无有效配对模型时不得报告 P4 效果。")
    return by_id


def load_sources():
    def gate():
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        sys.path.insert(0, str(ROOT / "src"))
        from round3.evaluate import validate_test_gate
        validate_test_gate()

    summary = final_summary_before_gate(R3 / "reports/test_summary.json", gate)
    # Everything below is reached only after the real project gate validates.
    source_paths = [R3 / p for p in ("reports/test_summary.json", "selection.json", "run_manifest.json",
                    "reports/integrity.json", "ROUND3_REPORT.md", "reports/test_all_results.csv",
                    "reports/test_offline_diagnostics.json", "reports/dataset_accounting.json",
                    "videos/manifest.json", "context_exposure.md", "audits/gpu_expansion_0_to_5.json",
                    "audits/root_migration_git_visibility.json")]
    for name in ("gpu_reduction_to_0_3.json", "gpu_concurrency_2_per_device.json"):
        execution_audit = R3 / "audits" / name
        if execution_audit.is_file():
            source_paths.append(execution_audit)
    require(all(p.is_file() and p.stat().st_size for p in source_paths), "最终报告、CSV、视频清单或审计附件尚不齐备。")
    integrity = read_json(R3 / "reports/integrity.json")
    require(integrity.get("final") is True and integrity.get("stage") == "test" and
            integrity.get("all_completed_results_validated") is True, "主报告验证尚未完成。")
    require(integrity["report_sha256"] == digest(R3 / "ROUND3_REPORT.md"), "主报告与完整性记录不匹配。")
    selection = read_json(R3 / "selection.json")
    manifest = read_json(R3 / "run_manifest.json")
    config = manifest["config"]
    validate_tables(summary, selection, config)
    require({r["run_id"] for r in manifest["runs"]} == {r["run_id"] for r in summary["results"]}, "当前运行清单与最终汇总不一致。")
    from round3.evaluate import _checked_result
    freeze_hash = digest(R3 / "global_freeze.json")
    for row in summary["results"]:
        if row["status"] != "evaluated":
            continue
        path = R3 / "locked_test_results" / row["run_id"] / "complete.json"
        require(path.is_file(), f"测试完成产物缺失：{row['run_id']}")
        result = _checked_result(path)  # Validates stored trajectory hashes and metrics.
        identity = result["identity"]
        require(identity["stage"] == "test" and identity["step"] == config["updates_per_system"] and
                identity["global_freeze_hash"] == freeze_hash and
                all(identity[k] == row[k] for k in ("run_id", "task", "candidate_id", "train_n")), "测试身份与冻结汇总不一致。")
        for split in SPLITS:
            metric = result["metrics"][split]
            require(all(close(row[split + "_" + k], metric[v]) for k, v in
                        (("success_rate", "success_rate"), ("successes", "successes"), ("episodes", "n"))), "测试汇总与已验证结果不一致。")
        train_path = R3 / "checkpoints" / row["run_id"] / "complete.json"
        training = read_json(train_path)
        require(training["step"] == config["updates_per_system"] and training["debug"] is False, "训练完成产物无效。")
        require(close(row["train_wall_seconds"], training["total_session_wall_seconds"]) and
                close(row["gpu_active_work_seconds"], training["gpu_active_work_seconds"]), "训练成本与完成产物不一致。")
        source_paths.extend((path, train_path))
    stream = io.StringIO()
    fields = list(dict.fromkeys(k for row in summary["results"] for k in row))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary["results"])
    require(list(csv.DictReader(io.StringIO(stream.getvalue()))) ==
            list(csv.DictReader(io.StringIO((R3 / "reports/test_all_results.csv").read_text()))), "完整 CSV 与测试汇总不一致。")
    expected_figures = {f"round3/figures/test/{t}_{kind}.png" for t in config["tasks"] for kind in ("data_curve", "layout_coverage")}
    expected_figures |= {"round3/figures/test/ood_delta_heatmap.png", "round3/figures/test/feedback_revision_comparison.png"}
    require(expected_figures <= set(summary["figures"]), "最终图表未全部生成。")
    videos = read_json(R3 / "videos/manifest.json")["records"]
    require(videos == summary["videos"] and not any(v.get("status") == "pending" for v in videos), "视频仍在生成或与汇总不一致。")
    for task in config["tasks"]:
        require(any(v.get("path") == f"round3/videos/{task}_fixed_C_N20.mp4" for v in videos), f"缺少 {task} 固定配对视频。")
    for value in summary["figures"]:
        path = ROOT / value
        require(path.is_file() and path.stat().st_size > 0, f"图表缺失：{value}")
        source_paths.append(path)
    for video in videos:
        if video.get("path"):
            path = ROOT / video["path"]
            require(path.is_file() and digest(path) == video["video_sha256"], f"视频缺失或校验失败：{path}")
            source_paths.append(path)
    proposals, bundles, revisions = {}, {}, {}
    for task in config["tasks"]:
        for destination, path in ((proposals, R3 / "design_records" / task / "proposal.json"),
                                  (revisions, R3 / "design_records" / task / "revision.json"),
                                  (bundles, R3 / "evidence" / task / "bundle.json")):
            require(path.is_file(), f"缺少设计证据：{path}")
            destination[task] = read_json(path)
            source_paths.append(path)
        decision = next(r["decision"] for r in summary["feedback_accounting"] if r["task"] == task)
        require(revisions[task].get("decision", revisions[task].get("choice")) == decision, "反馈决定与最终汇总不一致。")
        if decision == "P4":
            item = next(r for r in summary["feedback_accounting"] if r["task"] == task)
            candidate = revisions[task]["candidate"]
            require(item["p4_base_candidate_id"] == candidate["base_candidate_id"] and
                    item["p4_principal_revision_intent"] == candidate["principal_revision_intent"], "P4 基线或修订意图与冻结记录不一致。")
            sessions = []
            for n in (5, 20):
                path = R3 / "debug" / row_id(task, "P4", n) / "sessions.jsonl"
                if path.is_file():
                    sessions.extend(dict(path=str(path.relative_to(ROOT)), **json.loads(line)) for line in path.read_text().splitlines() if line.strip())
                    source_paths.append(path)
            require(sessions == item["p4_debug_sessions"], "P4 调试成本与原始会话账目不一致。")
    return summary, selection, config, proposals, bundles, source_paths


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def rates(row):
    if row is None or row["status"] != "evaluated":
        return "NA（无有效模型）"
    return " / ".join(f"{100 * row[key]:.1f}" for key in ("IID_success_rate", "C_success_rate", "E_success_rate", "ood_success_rate"))


def delta_text(summary, row):
    value = summary["paired_ood_deltas"].get(row["run_id"]) if row else None
    if not value:
        return "NA"
    low, high = value["paired_bootstrap_95ci_pp"]
    return f"{value['delta_pp']:+.1f} [{low:+.1f}, {high:+.1f}]"


def render(summary, selection, config, proposals, bundles, reduction_audit=False, concurrency_audit=False):
    """Pure rendering from supplied dictionaries; no filesystem or policy calls."""
    rows = validate_tables(summary, selection, config)
    tasks, ns = config["tasks"], config["demonstration_counts"]
    workers_per_gpu = config.get("workers_per_gpu", 1)
    maximum_workers = config.get("maximum_simultaneous_workers", config["maximum_simultaneous_gpus"] * workers_per_gpu)

    def selected(task, n, kind):
        if kind == "B0":
            return rows[row_id(task, "B0", n)]
        item = selection["groups"][f"{task}:n{n}"][kind]
        return rows[item["run_id"]] if item else None

    backend = "、".join(sorted({p["agent_backend"] for p in proposals.values()}))
    lines = ["# Round 3 中文结果摘要", "",
             f"已完成并验证锁定测试：正式训练 **{summary['completed_formal_runs']}/{summary['planned_runs']}** 次，"
             f"其中初始主矩阵 **{summary['completed_main_runs']}/{config['initial_runs']}** 次、额外 P4 **{summary['completed_p4_runs']}** 次；"
             f"无效配置 {len(summary['invalid'])} 项。完成 {summary['evaluated_runs']} 个模型的锁定测试，共 **{summary['scored_episodes']}** 个计分回合。"
             f"每次训练 {config['updates_per_system']:,} 次更新，主评测使用 {config['primary_checkpoint']}。", "",
             f"实际设计后端为 **{cell(backend)}**，初始提案由保存记录中的真实 Codex 子代理产生，协调 Codex 实现并执行。"
             "协调会话接触过 Round 1/2 的 D20 摘要、历史成绩及专家/环境源码；drawer、door 已是历史开发任务。"
             "初始设计材料量见下表，反馈额外暴露量另列。因此结果属于 **Codex 辅助设计的探索实验**，不能声称整个设计过程仅看过两条示范。"
             "[完整上下文披露](context_exposure.md)。", "",
             f"模型标识：`{cell(config['design_model'])}`；会话标识：`{cell(config['session_id'])}`；"
             f"token 数：`{cell(config['token_count'])}`；API 费用：`{cell(config['api_cost'])}`。未记录的精确版本与成本不作估计。", "",
             f"运行通过 Pixi，使用物理 GPU {', '.join(map(str, config['gpu_devices']))}（最多 {config['maximum_simultaneous_gpus']} 卡同时运行，"
             f"每卡最多 {workers_per_gpu} 个正式训练进程，总计最多 {maximum_workers} 个并发正式训练进程）。"
             "这些是当前运行配置；历史上曾扩展到六卡，详见[历史扩容审计](audits/gpu_expansion_0_to_5.json)。"
             + ("随后按用户要求收回 GPU 4/5，见[设备缩减与恢复点审计](audits/gpu_reduction_to_0_3.json)。" if reduction_audit else "")
             + ("各卡增加并发训练进程的授权与实施见[每卡并发审计](audits/gpu_concurrency_2_per_device.json)。" if concurrency_audit else "")
             + "项目已迁到仓库根目录，见[迁移审计](audits/root_migration_git_visibility.json)。", "",
             "| 任务 | 实际初始设计者 | 初设材料：完整示范 / 帧 | 设计理由与证据 |",
             "|---|---|---:|---|"]
    for task in tasks:
        designer = proposals[task].get("designer", "unavailable")
        if isinstance(designer, dict):
            designer = next((designer[k] for k in ("name", "actual_subagent_name", "subagent_name") if k in designer), "unavailable")
        lines.append(f"| {task} | {cell(designer)} | {bundles[task]['demo_count']} / {bundles[task]['frame_count']} | [原始提案](design_records/{task}/proposal.json) |")
    lines += ["", "初始开发选优（initial-selected）只在 P1/P2/P3 中选择；最终系统（final-system）可选择 B0/P1/P2/P3，以及实际训练的 N5/N20 P4。"
              "两者均使用该 N 的开发集：OOD 最高，其次 IID 更高、参数更少、候选 ID 字典序更小。"
              "以下直接采用[测试前冻结的选择](selection.json)，没有按测试成绩重选。候选 ID 只在各任务内表示具体设计。", "",
              "每格成功率依次为 **IID / C / E / OOD（%）**；OOD=(C+E)/2。C 为组合留出，E 为位置/方向外推；"
              "Δ 相对同任务同 N 的 B0，单位为百分点，方括号为汇总文件提供的配对 bootstrap 95% 区间。", "",
              "| 任务 | N | B0 | 初始开发选优：ID；成功率 | Δ B0 [95%区间] | 最终系统：ID；成功率 | Δ B0 [95%区间] | B0 回退 |",
              "|---|---:|---|---|---|---|---|---|"]
    for task in tasks:
        for n in ns:
            base, initial, final = [selected(task, n, kind) for kind in LABELS]
            cells = [f"{r['candidate_id'] if r else 'NA'}；{rates(r)}" for r in (initial, final)]
            fallback = selection["groups"][f"{task}:n{n}"]["b0_fallback"]
            lines.append(f"| {task} | {n} | {rates(base)} | {cells[0]} | {delta_text(summary, initial)} | {cells[1]} | {delta_text(summary, final)} | {'是' if fallback else '否'} |")
    lines += ["", "以下先在任务内计算，再对六任务等权平均。若某组存在无效模型则不以剩余任务冒充完整六任务均值。", "",
              "| N | 方法 | 完整任务数 | IID / C / E / OOD（%） | OOD Δ B0（百分点） |", "|---:|---|---:|---|---:|"]
    for n in ns:
        for kind, label in LABELS.items():
            group = [selected(t, n, kind) for t in tasks]
            valid = [r for r in group if r and r["status"] == "evaluated"]
            means = "NA"
            change = "NA"
            if len(valid) == len(tasks):
                mean = {k: sum(r[k] for r in valid) / len(tasks) for k in ("IID_success_rate", "C_success_rate", "E_success_rate", "ood_success_rate")}
                means = rates(dict(mean, status="evaluated"))
                if all(selected(t, n, "B0")["status"] == "evaluated" for t in tasks):
                    base_mean = sum(selected(t, n, "B0")["ood_success_rate"] for t in tasks) / len(tasks)
                    change = f"{100 * (mean['ood_success_rate'] - base_mean):+.2f}"
            lines.append(f"| {n} | {label} | {len(valid)}/{len(tasks)} | {means} | {change} |")
    lines += ["", "| OOD 对照范围 | 正向 | 负向 | 持平 | 可配对总数 |", "|---|---:|---:|---:|---:|"]
    comparisons = {"全部初始候选 P1/P2/P3": [r for r in rows.values() if r["candidate_id"] in ("P1", "P2", "P3")]}
    comparisons.update({label: [selected(t, n, kind) for t in tasks for n in ns] for kind, label in LABELS.items() if kind != "B0"})
    for label, group in comparisons.items():
        deltas = [summary["paired_ood_deltas"][r["run_id"]]["delta_pp"] for r in group if r and r["run_id"] in summary["paired_ood_deltas"]]
        counts = Counter("正向" if d > 1e-9 else "负向" if d < -1e-9 else "持平" for d in deltas)
        lines.append(f"| {label} | {counts['正向']} | {counts['负向']} | {counts['持平']} | {len(deltas)} |")
    lines += ["", "正负号仅描述这些固定模型的观测差异，并不表示统计显著或识别了某个模块的因果作用；区间包含训练随机性以外的评测状态抽样不确定性。"
              "IID 与 OOD 同步上升时，收益也可能包括拟合改善。全部候选的分布成绩、Wilson 区间及失败配置见[完整 CSV](reports/test_all_results.csv)和[主报告](ROUND3_REPORT.md)。", "",
              "| 任务 | 一次反馈决定 | 开发回合 / 反馈帧 | P4 完成 / 计划 | P4 开发回合 | P4 训练墙钟 / 进程同步计时（小时） |", "|---|---|---:|---:|---:|---:|"]
    for item in summary["feedback_accounting"]:
        decision = "不修订（no_revision）" if item["decision"] == "no_revision" else "修订 P4"
        task = item["task"]
        lines.append(f"| {task} | [{decision}](design_records/{task}/revision.json) | {item['feedback_development_episodes']} / {item['feedback_packet_frames']} | {item['p4_completed_trainings']} / {item['p4_planned_trainings']} | {item['p4_development_episodes']} | {item['p4_train_wall_seconds']/3600:.3f} / {item['p4_gpu_active_work_seconds']/3600:.3f} |")
    p4_tasks = [r["task"] for r in summary["feedback_accounting"] if r["decision"] == "P4"]
    if p4_tasks:
        lines += ["", "P4 单列，未并入初次设计曲线；每个可实现的修订增加 N5/N20 两次独立从头训练，单模型预算不变。"
                  "实际修订基线（base-candidate）指冻结提案中被改动的候选；初始开发选优（initial-selected）是该 N 的程序化选优，两者可能不同。"
                  "下面分别报告 P4 相对二者的观测差值与直接配对 snapshot bootstrap 区间，未从两个 B0 区间推造新区间。", "",
                  "| 任务 | N | 对照类别 | 对照候选 | P4 IID / C / E / OOD（%） | P4 Δ 对照 [95%区间] | P4 Δ B0 [95%区间] |", "|---|---:|---|---|---|---|---|"]
        for effect in summary["feedback_effects"]:
            task, n = effect["task"], effect["train_n"]
            p4 = rows[effect["p4_run_id"]]
            delta = effect["paired_ood_delta"]
            change = f"{delta['delta_pp']:+.1f} [{delta['paired_bootstrap_95ci_pp'][0]:+.1f}, {delta['paired_bootstrap_95ci_pp'][1]:+.1f}]" if delta else "NA（无有效配对模型）"
            label = "实际修订基线" if effect["reference_kind"] == "base-candidate" else "初始开发选优"
            lines.append(f"| {task} | {n} | {label} | {effect['reference_candidate_id'] or 'NA'} | {rates(p4)} | {change} | {delta_text(summary, p4)} |")
        lines += ["", "P4 调试开销额外单列，属于主报告的全局调试账目，不计入正式训练次数或耗时。"
                  "以下仅统计已完成的调试会话；零 optimizer 更新的 CPU 特征/反向传播审计、其他未汇总检查与未完成会话另行记录，未记录耗时不作估计。", "",
                  "| 任务 | 实际修订基线 | 唯一主要修订意图 | P4 调试更新 | P4 调试墙钟（秒） |", "|---|---|---|---:|---:|"]
        for item in summary["feedback_accounting"]:
            if item["decision"] == "P4":
                lines.append(f"| {item['task']} | {item['p4_base_candidate_id']} | {cell(item['p4_principal_revision_intent'])} | {item['p4_debug_session_updates']} | {item['p4_debug_session_wall_seconds']:.3f} |")
    no_revision = [r["task"] for r in summary["feedback_accounting"] if r["decision"] == "no_revision"]
    if no_revision:
        lines += ["", f"{', '.join(no_revision)} 决定 no_revision，因此没有 P4 干预或额外 P4 训练，无法从这些任务估计反馈修订的效果。"]
    completed = [r for r in rows.values() if r["status"] == "evaluated"]
    lines += ["", f"已完成正式模型的训练墙钟之和 **{sum(r['train_wall_seconds'] for r in completed)/3600:.3f} 小时**，"
              f"逐训练进程的同步训练计时之和 **{sum(r['gpu_active_work_seconds'] for r in completed)/3600:.3f} 进程小时**（原始字段 `gpu_active_work_seconds`）。"
              "同卡并发进程的计时时段会重叠，因此此合计不代表物理 GPU 占用小时、利用率或独占 kernel 时间；也不等于实验经过时间。"
              "P4 表中使用相同计时口径。上述正式训练合计覆盖最终完成的模型，包含这些模型中断后恢复的已记录训练会话；"
              "尚未完成的模型、调试、评测和渲染另行记账。主报告中排除 running/interrupted work 的表述指尚未完成的模型，不能理解为扣除已完成模型的历史中断会话。"
              f"预算上限 {config['maximum_formal_runs']} 次正式训练，每次系统总 batch 为 {config['batch_chunks_per_system_update']} 个 chunks。"
              "相同更新数不等于相同 FLOPs；真实参数量、推理延迟、重规划次数和每模型成本见完整 CSV；"
              "示范 transition 与辅助标签数量见[数据账目](reports/dataset_accounting.json)。", "",
              f"训练 seed 仅为 {config['train_seeds']}，且只有一组嵌套示范子集。配对区间仅反映固定模型的测试 snapshot 抽样，不覆盖训练 seed 或示范选择随机性；"
              "各 N 的策略仅用其完整 D_N 与相应归一化统计。部署策略仍使用共同数值状态（state-input），设计时看图不构成视觉泛化实验。"
              "候选比较没有形成完整因子实验，不能将差异归因为单一先验，也不能据此证明优于人类设计/随机搜索、跨任务通用或各任务必须使用不同先验。", "",
              "完整附件：[主报告](ROUND3_REPORT.md)、[全部候选 CSV](reports/test_all_results.csv)、[配对区间与原始汇总](reports/test_summary.json)、"
              "[离线失败诊断](reports/test_offline_diagnostics.json)、[实施审计](audits/)、[最终图表](figures/test/)、"
              "[视频清单](videos/manifest.json)。视频为保存动作的真实模拟器重放，增加零次策略推理和零个计分回合。", ""]
    return "\n".join(lines)


def main():
    try:
        summary, selection, config, proposals, bundles, sources = load_sources()
        signatures = {str(p): digest(p) for p in sources}
        content = render(summary, selection, config, proposals, bundles,
                         reduction_audit=R3 / "audits/gpu_reduction_to_0_3.json" in sources,
                         concurrency_audit=R3 / "audits/gpu_concurrency_2_per_device.json" in sources)
        require(all(digest(Path(p)) == h for p, h in signatures.items()), "生成期间输入发生变化；请待最终报告稳定后重试。")
        output = R3 / "ROUND3_SUMMARY_ZH.md"
        temporary = output.with_name(output.name + ".partial")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(output)
        print(json.dumps({"status": "complete", "path": str(output), "source_test_summary_sha256": signatures[str(R3 / 'reports/test_summary.json')]}, ensure_ascii=False))
        return 0
    except (ValueError, AssertionError, KeyError, OSError, TypeError) as exc:
        print(json.dumps({"status": "not_ready", "reason": str(exc), "summary_written": False}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
