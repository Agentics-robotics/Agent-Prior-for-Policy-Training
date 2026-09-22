"""Audit completed artifacts and write the developer's factual Chinese report."""

from collections import Counter

from appl.io import ROOT, atomic, digest, read


def main():
    run = ROOT / "real_robot/runs/push_letters/training_v4"
    package = run / "package_00"
    library = read(run / "library.json")
    result = read(package / "training/shape_push_v1/result.json")
    prepared = read(package / "prepared/result.json")
    metadata = read(package / "prepared/metadata.json")
    calling = read(run / "calling_check/result.json")
    assert read(run / "execution_status.json")["status"] == "complete"
    assert result["neural_training_verified"] and calling["passed"]
    assert digest(result["checkpoint"]) == result["checkpoint_sha256"]
    for name, record in prepared["arrays"].items():
        assert digest(package / "prepared" / record["file"]) == record["sha256"], name
    assert digest(package / "prepared/metadata.json") == prepared["metadata_sha256"]
    for name, sha in read(package / "submission.json")["files"].items():
        assert digest(package / "source" / name) == sha
    frozen = read(run / "design_00/framework.json")
    for name, sha in frozen.items():
        assert digest(ROOT / name) == sha, name
    old_cuts = {}
    for version in ("cut_v3", "cut_v4", "cut_v5", "cut_v6"):
        folder = ROOT / "real_robot/data/push_letters" / version
        files = read(folder / "manifest.json")["files"]
        for name, sha in files.items():
            assert digest(folder / name) == sha, (version, name)
        old_cuts[version] = len(files)
    origin = read(run / "setup/framework_origin.json")
    for name, sha in origin["source_files"].items():
        assert digest(ROOT / origin["source_directory"] / name) == sha
    previous = read(ROOT / "real_robot/runs/push_letters/training_v3/library.json")
    for policy in previous["policies"]:
        assert digest(policy["checkpoint"]) == policy["checkpoint_sha256"]
        for name, sha in policy["source_hashes"].items():
            assert digest(ROOT / policy["source"] / name) == sha
    reasons = Counter(a["reason"] for a in metadata["anchor_audit"] if a["status"] == "excluded")
    counts = prepared["segment_examples"]
    covered = sum(c["policy_loss_rows"] > 0 for c in metadata["coverage"] if c["kind"] == "sparse")
    checks = [read(p) for p in (run / "design_00/preprocessing_checks").glob("*/check_receipt.json")]
    receipt = {
        "review_date": "2026-09-21",
        "status": "implementation_preprocessing_training_and_local_calling_complete",
        "API_accounting": library["API_accounting"],
        "API_source_files": len(library["API_source_authorship"]),
        "actual_updates": result["optimizer_steps"],
        "trainable_parameters": result["trainable_parameters"],
        "retained_decision_anchors": prepared["num_examples"],
        "frozen_decision_anchors": prepared["frozen_sparse_anchors"],
        "excluded_decision_anchors": prepared["excluded_sparse_anchors"],
        "covered_learning_segments": covered,
        "total_learning_segments": 12,
        "exclusion_reasons": dict(reasons),
        "preprocessing_checks": {"attempts": len(checks), "failed": sum(c["returncode"] != 0 for c in checks), "optimizer_updates": 0},
        "checkpoint_sha256": result["checkpoint_sha256"],
        "reload_max_prediction_error": result["reload_max_prediction_error"],
        "local_calling_check": str(run / "calling_check/result.json"),
        "frozen_design_framework_verified": True,
        "prepared_cache_verified": True,
        "previous_cut_files_verified": old_cuts,
        "training_v3_source_and_three_checkpoints_verified": True,
        "generalization_evaluated": False,
        "robot_execution": False,
        "limitations": ["Tiny inferred-label training subset; no coverage/generalization claim", "Chromatic perception, inferred calibration, upstream semantic goals and external planner binding required"],
    }
    atomic(run / "completion_receipt.json", receipt)
    account = library["API_accounting"]
    lines = [
        "# Push training_v4：实现、预处理及训练记录",
        "",
        "核对日期：2026-09-21。输入为冻结的 cut_v6；本轮完成 Runtime API 编写代码、实际数据转换、模型训练和本地调用检查。",
        "",
        f"**最主要的限制：47 个决策锚点最终只有 {prepared['num_examples']} 个留下有效弱标签，覆盖 {covered}/12 个学习片段。** 这轮是可检查的训练原型，尚无证据说明已学会完整任务或获得泛化能力。原始 16,745 行及 executor-only 证据保留。",
        "",
        "## 实际模型",
        "",
        "唯一学习策略 `shape_push_v1` 对物体边界上的联合候选打分。原始第三视角 RGB 经颜色分割、相机/平面几何转换成轮廓；网络输入为 256×9 点特征，以及 2,688×29 个候选特征和有效性 mask。轮廓、目标和短历史在物体/目标相关坐标中表达，原始图像不直接进入可训练网络。",
        "",
        f"点编码器 9→64→64 加 max pooling；候选评分器 93→64→64→1；共 **{result['trainable_parameters']:,} 个可训练参数**。候选组合来自 128 个边界点、7 个方向和3种短行程。输出为接触点二维坐标、推动方向、短行程长度及边界编号；`predict` 的六个数不是六轴 EE 指令。抬起、移动、下降由外部 planner/controller 执行。",
        "",
        "API 选择小型候选评分网络，并说明其比完整轨迹 Diffusion 更符合少量几何决策监督。实现采用颜色/轮廓方案，未获取或运行 SAM/Qwen 权重；这与 cut 阶段的感知依赖设想不同，限制了颜色/材质泛化。",
        "",
        "## 数据处理已经执行",
        "",
        "API 从图像与 flange 变换推断工具偏移、有效水平顶面、物体轮廓和目标变换，再提取接触点及短平移前缀；检查接触关联、配准、抬升/旋转、短行程以及候选映射，最后审阅叠加图并排除遮挡/局部轮廓等不可信标签。标签和标定均为推断，没有人工接触真值。深度因尺度/配准未核实而未使用。",
        "",
        "`prepared/metadata.json` 与 `evidence/final_anchor_audit.json` 是最终覆盖依据；早期 `A_*/B_*` 图和 JSON 是复核前诊断，不可当作最终保留清单。未来帧仅用于离线目标推导，模型观测窗口不越过锚点。",
        "",
        "| 片段 | 有效决策点 |",
        "|---|---:|",
    ]
    lines += [f"| `{c['segment_id']}` | {counts[c['segment_id']]} |" for c in metadata["coverage"]]
    lines += ["", "排除原因：", ""] + [f"- {count} 个：{reason}" for reason, count in reasons.items()]
    lines += [
        "", "## 实际训练与验证", "",
        f"GPU 4、seed 0，按 API 预先声明的预算完成 **{result['optimizer_steps']:,} 次 AdamW 更新**，使用最终 EMA。采样曝光总数 {result['sample_exposures']:,} 是重复抽样次数，不是独立样本数。权重 L2 变化 {result['weight_l2_change']:.6f}；最终 checkpoint 重载预测最大误差 {result['reload_max_prediction_error']}。无性能驱动选模、额外训练或隐藏测试。",
        "",
        f"执行过 {len(checks)} 次零更新预处理检查，其中 {sum(c['returncode'] != 0 for c in checks)} 次失败保留了原始代码和错误，由 API 修复。框架有 18 项切分/因果/稀疏监督接口测试通过。训练后的独立进程加载检查返回真实几何提案，并核对缺失执行绑定、过期观测、反射目标和中断状态。",
        "",
        "调用检查使用明确标识的共同钟、离线推断标定及工作区测试夹具，仅证明接口可调用。直接传入原录制 t/recv_time 会因约 3.2 秒钟差得到 stale_scene。没有更改录制时间，也没有把离线标定认作实机标定。",
        "",
        f"本轮 **{account['API_calls']} 次 gpt-6-astra/xhigh 调用**；请求与响应均核对；估算 API 成本 **${account['estimated_usd']:.6f}**，不是账单金额。{len(library['API_source_authorship'])} 个提交文件有对应 API write_file 来源。数据/模型科学内容由 API 编写，外层执行和本报告由开发者编写。",
        "", "## Agent 接口与成品", "",
        "上层 Agent 提供物体身份、参考轮廓、几何目标和实时标定；策略输出接触提案；执行转换器构造 planner 请求。附带的几何调度器支持 prepare→重观测→push→重观测；文字识别、布局规划、持续跟踪及真实 planner/controller 绑定仍由宿主提供。",
        "",
        "- [本地调用入口](../policy_training_v4/README.md)",
        "- [API 参数与调用说明](../runs/push_letters/training_v4/package_00/source/shape_push_v1_USAGE.md)",
        "- [API 源码与目录](../runs/push_letters/training_v4/package_00/source/POLICY_CATALOG.md)",
        "- [最终权重](../runs/push_letters/training_v4/package_00/training/shape_push_v1/last.pt)",
        "- [模型来源与训练库](../runs/push_letters/training_v4/library.json)",
        "- [最终逐锚点审计](../runs/push_letters/training_v4/package_00/prepared/evidence/final_anchor_audit.json)",
        "- [独立调用检查](../runs/push_letters/training_v4/calling_check/result.json)",
        "- [完成凭据](../runs/push_letters/training_v4/completion_receipt.json)",
        "",
        "早期 cut_v3/v4/v5/v6 发布文件、training_v3 源码及三个 checkpoint 再次通过哈希检查。未进行机器人动作、提交/push 或独立部署打包。",
    ]
    (ROOT / "real_robot/reports/PUSH_TRAINING_V4.md").write_text("\n".join(lines) + "\n")
    print(receipt)


if __name__ == "__main__":
    main()
