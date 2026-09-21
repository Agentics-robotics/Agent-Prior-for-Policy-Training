"""Audit completed cut_v5 policy training and write a developer evidence report."""
import argparse
import json
from pathlib import Path
import numpy as np
from appl.io import ROOT, atomic, digest, read
from real_robot.policy_training_v3.design import configuration
from real_robot.policy_training_v3.runner import publish


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", type=int, required=True)
    args = parser.parse_args()
    cfg = configuration(ROOT / "real_robot/configs/push_training_v3.json")
    run = ROOT / cfg["run"]
    library = publish(cfg, args.revision)
    package = run / f"package_{args.revision:02d}"
    source = package / "source"
    metadata = read(source / "package.json")
    cache = read(package / "prepared/result.json")
    preparation_review = read(run / f"preparation_review_{args.revision:02d}.json")
    for name, entry in cache["arrays"].items():
        assert digest(package / "prepared" / entry["file"]) == entry["sha256"], name
    assert digest(package / "prepared/metadata.json") == cache["metadata_sha256"]
    old = {}
    for cut in ("cut_v3", "cut_v4", "cut_v5"):
        folder = ROOT / "real_robot/data/push_letters" / cut
        manifest = read(folder / "manifest.json")
        for name, sha in manifest["files"].items():
            assert digest(folder / name) == sha, name
        old[cut] = len(manifest["files"])
    origin = read(run / "setup/framework_origin.json")
    for name, sha in origin["source_files"].items():
        assert digest(ROOT / origin["source_directory"] / name) == sha
    plan = read(ROOT / cfg["cut_output"] / "plan.json")
    segments = {g["skill_id"]: [s["segment_id"] for s in g["segments"]] for g in plan["skills"]}
    models = []
    arrays = {name: np.load(package / "prepared" / entry["file"], mmap_mode="r", allow_pickle=False)
              for name, entry in cache["arrays"].items()}
    for entry in library["policies"]:
        policy_id = entry["policy_id"]
        result = read(entry["training_result"])
        counts = cache["policy_segment_examples"][policy_id]
        group = cfg["policy_datasets"][policy_id]
        expected = segments[group]
        assert result["training_examples"] == sum(counts.values()) > 0
        assert result["initial_weights_sha256"] != result["final_weights_sha256"]
        assert all(c["l2_norm"] > 0 for c in result["finite_nonzero_gradient_checks"])
        eligible = arrays["policy_weight"][:, cfg["policies"].index(policy_id)] > 0
        feature_coverage = {key: int(np.count_nonzero(arrays[key][eligible] > 0))
                            for key in ("goal_valid", "geometry_valid", "response_valid")
                            if key in arrays and arrays[key].shape == eligible.shape}
        models.append(dict(
            policy_id=policy_id, dataset_group=group,
            model_family=metadata["policies"][policy_id]["model_family"],
            diffusion_consideration=metadata["policies"][policy_id]["diffusion_consideration"],
            optimizer_steps=result["optimizer_steps"], training_examples=result["training_examples"],
            covered_segments=sum(counts[s] > 0 for s in expected), assigned_segments=len(expected),
            empty_segments=[s for s in expected if not counts[s]],
            feature_coverage=feature_coverage,
            object_feature_masks_applicable=group == "piece_relocation",
            missing_object_goal_segments=[
                dict(segment_id=s["segment_id"], action_rows=s["retained_action"])
                for s in preparation_review["API_segment_coverage"]
                if group == "piece_relocation" and s["segment_id"] in expected
                and not s["goal_eligible"]],
            trainable_parameters=result["trainable_parameters"],
            reload_max_action_error=result["reload_max_action_error"],
            checkpoint_sha256=result["checkpoint_sha256"],
            elapsed_seconds=result["elapsed_seconds"], device=result["device"]))
    receipt = dict(
        date="2026-09-21", status="complete", revision=args.revision,
        policies=models, API_accounting=library["API_accounting"],
        API_authored_files=len(library["API_source_authorship"]),
        old_cut_published_files_unchanged=old, old_training_framework_unchanged=True,
        prepared_cache_hashes_verified=True,
        performance_evaluation=False, robot_execution=False,
        interface_check=read(run / "interface_check.json") if (run / "interface_check.json").exists() else None,
        limitations=metadata["limitations"], adaptations=metadata["adaptations"],
        provenance="Scientific package/metadata by Runtime API; execution and this report by RepositoryAgent")
    atomic(run / "completion_receipt.json", receipt)
    lines = ["# Push training_v3 — cut_v5 三策略训练", "",
        "核对日期：2026-09-21。按用户授权，由 gpt-6-astra/xhigh 编写策略，鼓励 Diffusion Policy并要求说明其他模型的选择理由。三个模型独立完成固定 20,000 步训练，保留最终 EMA。", "",
        "| Policy | API 选择的模型 | 样本数 | 覆盖片段 | 参数数 | EMA 重载最大误差 |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for m in models:
        lines.append(f"| {m['policy_id']} | {m['model_family']} | {m['training_examples']:,} | {m['covered_segments']}/{m['assigned_segments']} | {m['trainable_parameters']:,} | {m['reload_max_action_error']} |")
    lines += ["", "训练只采样各自原始数据组中的有效样本；完整逐片段覆盖和过滤原因保存在 prepared/result.json 与 metadata.json。输入预处理、模型、损失、目标/动作转换、prior 和调用文档均为 API 原文。", "",
              "动作样本覆盖与可信的物体目标/几何/响应监督覆盖不同。以下为预处理缓存中 API 有效掩码的实际计数；有效仍只代表其自动标注规则通过，并非人工验证的真值。这三个物体掩码不适用于工具 waypoint policy，它使用独立的 TCP 目标。", ""]
    for m in models:
        if m["object_feature_masks_applicable"]:
            lines.append(f"- {m['policy_id']}: {json.dumps(m['feature_coverage'], ensure_ascii=False)}。缺少可信物体目标的片段：{json.dumps(m['missing_object_goal_segments'], ensure_ascii=False)}。这些行保留动作监督，以目标缺失掩码和终点 TCP 条件参与训练。")
        else:
            lines.append(f"- {m['policy_id']}: 使用 TCP 目标，以上物体掩码不适用。")
    lines += ["", "两个搬运模型使用物体相对的局部几何与运动表示，但最终仍输出六维原始 v/w，包含受限残差；接触阶段没有硬性投影成纯二维动作。工具换位模型、轮廓接触图模型和目标场伺服模型各自拥有独立权重。", ""]
    lines += ["", "## API 模型选择理由", ""]
    for m in models:
        lines += [f"### {m['policy_id']}", "", m["diffusion_consideration"], ""]
    account = library["API_accounting"]
    rel = Path("../runs/push_letters/training_v3") / package.name / "source"
    lines += ["## 调用与核验", "",
              f"入口为 [API 调用目录]({rel}/POLICY_CATALOG.md)、[完整接口文档]({rel}/CALLING.md) 和 [交接约定]({rel}/HANDOFF.json)。训练后的清单见 [library.json](../runs/push_letters/training_v3/library.json)。", "",
              f"核验 {account['API_calls']} 次 API 调用和 {len(library['API_source_authorship'])} 个源码/文档文件的来源，费用估算 ${account['estimated_usd']:.6f}（非账单）。检查有限梯度、权重更新、数据/权重哈希和固定随机种子的 EMA 重载。原 cut_v3/v4/v5 及旧训练框架保持不变。", "",
              "这是实现与训练完成证据，不是任务成功率或泛化证据；本轮没有真机执行。实际调用接口检查见 completion_receipt.json 中单独记录。", "",
              "通过仓库本地 `real_robot.policy_training_v3.inference.PolicyProcess` 加载，显式传入 `gpu=4`（配置允许 4/5/7）。每个模型的 `last.pt` 路径和哈希在 library.json 中。`inventory` 提供当前几何实例，`act` 接收 waypoint 或物体空间目标及独立调用状态；`close` 结束隔离进程。具体参数和交接语义以 API 文档为准。", ""]
    if receipt["interface_check"]:
        lines += ["| Policy | 有限六维候选动作 | 过期输入拒绝 | 非刚体目标拒绝 | 中断 | 解码状态 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for c in receipt["interface_check"]["cases"]:
            lines.append(f"| {c['policy_id']} | {c.get('finite_candidate', False)} | {c.get('stale_input_rejected', False)} | {c.get('invalid_goal_rejected', False)} | {c.get('interruption_supported', False)} | {c['status']} |")
        lines += ["", "该检查仅使用一帧已有配对观测和当前 TCP/物体原位目标，确认实际入口可加载并处理输入；没有任务 rollout、性能评分或额外训练。动作解码仍禁用：API 未获得经核验的原始控制器/follower，返回的是候选 v/w，尚不能直接下发真机。", ""]
    lines += ["## API 声明的适配与限制", "", metadata["adaptations"], "", metadata["limitations"], ""]
    (ROOT / "real_robot/reports/PUSH_TRAINING_V3.md").write_text("\n".join(lines))
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
