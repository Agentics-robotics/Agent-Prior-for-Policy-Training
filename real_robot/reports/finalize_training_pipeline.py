"""Verify completed generic runs and report measured artifacts, without redesign."""
import argparse
from pathlib import Path
import time

from appl.io import ROOT, atomic, digest, read
from real_robot.training_pipeline.runner import audit_designs


def finalize(name):
    cfg = read(ROOT / f"real_robot/configs/{name}.json")
    root = ROOT / cfg["run"]
    state = read(root / "workflow.json")
    if state["status"] != "complete":
        raise ValueError("Run is not complete: " + name)
    package = root / f"package_{state['revision']:02d}"
    library = read(root / "library.json")
    submission = read(package / "submission.json")
    spec = read(package / "source/package.json")
    prepared = read(package / "prepared/result.json")
    metadata = read(package / "prepared/metadata.json")
    writes, submissions, account = audit_designs(cfg)
    assert submission["package_hash"] in submissions
    for filename, sha in submission["files"].items():
        assert digest(package / "source" / filename) == sha
        assert (filename, sha) in writes
    for filename, sha in submission["assets"].items():
        assert digest(root / "assets" / filename) == sha
    assert digest(package / "prepared/result.json") == submission["preparation_hash"]
    assert digest(package / "prepared/metadata.json") == prepared["metadata_sha256"]
    assert prepared["source_hashes"] == submission["files"]
    for record in prepared["arrays"].values():
        assert digest(package / "prepared" / record["file"]) == record["sha256"]
    for session in root.glob("design_*"):
        for filename, sha in read(session / "framework.json").items():
            assert digest(ROOT / filename) == digest(session / "interface_snapshot" / filename) == sha
    for filename, sha in read(package / "executor_freeze.json").items():
        assert digest(ROOT / filename) == digest(package / "executor_snapshot" / filename) == sha
    results = {}
    for policy_id in spec["policies"]:
        folder = package / "training" / policy_id
        result = read(folder / "result.json")
        assert digest(folder / "last.pt") == result["checkpoint_sha256"]
        assert result["source_hashes"] == submission["files"]
        assert result["neural_training_verified"] and result["weight_l2_change"] > 0
        assert result["reload_max_prediction_error"] <= 1e-6 and result["calling_check"]["passed"]
        enforcement = read(folder / "enforcement.json")
        assert enforcement["seccomp"] and enforcement["landlock_abi"] > 0 and not enforcement["network"]
        results[policy_id] = result
    archive = ROOT / "real_robot/prompts/archive/20260922_before_representation_revision"
    for old in read(archive / "manifest.json")["runs"]:
        for filename, sha in old["files"].items():
            assert digest(ROOT / filename) == digest(archive / filename) == sha
    old_root = ROOT / "real_robot/runs/push_letters/training_v4/package_00"
    for filename, sha in read(old_root / "submission.json")["files"].items():
        assert digest(old_root / "source" / filename) == sha
    old_training = read(old_root / "training/shape_push_v1/result.json")
    assert digest(old_training["checkpoint"]) == old_training["checkpoint_sha256"]
    receipt = dict(passed=True, checked=time.time(), run=name, revision=state["revision"],
                   API_accounting=account, num_examples=prepared["num_examples"],
                   source_coverage=prepared["source_coverage"], policy_examples=prepared["policy_examples"],
                   models=results, old_cut_prompts_and_plans_unchanged=True,
                   old_training_v4_source_and_checkpoint_unchanged=True,
                   physical_robot_execution=False, generalization_evaluation=False)
    atomic(root / "completion_receipt.json", receipt)
    relative = "../" + str(root.relative_to(ROOT / "real_robot"))
    lines = [f"# {name}：API 实现与正式训练", "", "核对日期：2026-09-22。",
        "", f"完成的实现修订：{state['revision']}；实际 API 调用 {account['API_calls']} 次，均为 Astra/xhigh。",
        f"API 费用估算 ${account['estimated_usd']:.6f}，非实际账单。未知调用预留费用 ${account['unknown_reserved_usd']:.6f}。",
        "", "本轮承接原 cut/prior；没有运行新版 cut prompt。预处理、模型、训练和调用代码由 API 提交。",
        "", "| Policy | 架构 | 可训练参数 | 准备的可用例 | 实际更新 | 优化器 | 重载误差 |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: |"]
    for policy_id, result in results.items():
        family = spec["policies"][policy_id]["model_family"].replace("|", "/").replace("\n", " ")
        lines.append(f"| {policy_id} | {family} | {result['trainable_parameters']} | {result['training_examples']} | {result['optimizer_steps']} | {result['optimizer_class']} | {result['reload_max_prediction_error']} |")
    lines += ["", f"准备 {prepared['num_examples']} 个衍生训练例；唯一原始锚点 {sum(v['unique_supervised_anchors'] for v in prepared['source_coverage'].values())} 个。两者不能混同为独立示范数量。",
        f"覆盖 {sum(v > 0 for v in prepared['segment_examples'].values())}/{len(prepared['segment_examples'])} 个衍生片段。",
        "表中可用例按 policy_weight>0 统计；API sampler 可能另划训练/验证用途，该数不等于实际用于梯度更新的独立样本数。实际划分见 API 数据与采样代码。",
        "", "以下为 API 对实际数据的评估与限制：", ""]
    for key in ("coverage_assessment", "data_recovery_assessment", "training_rationale", "limitations"):
        lines += [f"**{key}**：{submission['readiness'][key]}", ""]
    lines += ["实际训练权重已更新，checkpoint 重载和 API 提供的调用案例通过。调用案例是接口证据，不是真机成功或独立泛化评估。",
        "", f"- [API 调用目录]({relative}/package_{state['revision']:02d}/source/POLICY_CATALOG.md)",
        f"- [API 调用说明]({relative}/package_{state['revision']:02d}/source/CALLING.md)",
        f"- [API 模型/训练配置]({relative}/package_{state['revision']:02d}/source/package.json)",
        f"- [覆盖与质量记录]({relative}/package_{state['revision']:02d}/prepared/metadata.json)",
        f"- [训练库]({relative}/library.json)", f"- [完成核验]({relative}/completion_receipt.json)", ""]
    path = ROOT / "real_robot/reports" / (name.upper() + ".md")
    path.write_text("\n".join(lines))
    return dict(run=name, report=str(path), receipt=str(root / "completion_receipt.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()
    for name in args.configs:
        print(finalize(name), flush=True)


if __name__ == "__main__":
    main()
