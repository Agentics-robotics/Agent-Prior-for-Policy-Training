"""Guard checks and in-memory formatting; no real test artifact access."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def module():
    path = Path(__file__).resolve().parents[2] / "scripts/round3_chinese_summary.py"
    spec = importlib.util.spec_from_file_location("chinese_summary", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


class MarkerPath:
    def __init__(self, exists):
        self.exists = exists

    def is_file(self):
        return self.exists


def test_missing_summary_rejects_before_any_read_or_gate(module):
    def forbidden(*args):
        pytest.fail("Missing final marker must not read artifacts or call gate")

    with pytest.raises(module.NotReady, match="缺少"):
        module.final_summary_before_gate(MarkerPath(False), forbidden, forbidden)


@pytest.mark.parametrize("marker", [False, None, 1, "true"])
def test_nonfinal_summary_never_calls_gate(module, marker):
    reads = []

    def read(path):
        reads.append(path)
        return {"final": marker}

    with pytest.raises(module.NotReady, match="final 必须"):
        module.final_summary_before_gate(MarkerPath(True), lambda: pytest.fail("Gate called early"), read)
    assert len(reads) == 1


def test_final_marker_cannot_bypass_rejected_gate(module):
    events = []

    def read(path):
        events.append("marker")
        return {"final": True}

    def gate():
        events.append("gate")
        raise AssertionError("synthetic rejected global freeze")

    with pytest.raises(AssertionError, match="rejected global freeze"):
        module.final_summary_before_gate(MarkerPath(True), gate, read)
    assert events == ["marker", "gate"]


def fixture(module):
    tasks, ns = [f"synthetic{i}" for i in range(6)], [2, 5, 10, 20]
    config = dict(tasks=tasks, demonstration_counts=ns, candidates=["B0", "P1", "P2", "P3"],
                  initial_runs=96, maximum_formal_runs=108, test_episodes={"IID": 20, "C": 40, "E": 40},
                  updates_per_system=20000, primary_checkpoint="20k EMA", design_model="unavailable",
                  session_id="unavailable", token_count="unavailable", api_cost="unavailable",
                  gpu_devices=[0, 1, 2, 3], maximum_simultaneous_gpus=4, batch_chunks_per_system_update=128, train_seeds=[0])
    rows, deltas, groups, feedback = [], {}, {}, []
    for task in tasks:
        for n in ns:
            candidates = ["B0", "P1", "P2", "P3"] + (["P4"] if task == tasks[1] and n in (5, 20) else [])
            dev = []
            for c in candidates:
                rate = {"B0": .5, "P1": .75, "P2": .25, "P3": 1., "P4": .25}[c]
                rid = module.row_id(task, c, n)
                row = dict(run_id=rid, task=task, candidate_id=c, train_n=n, status="evaluated", reason="",
                           ood_success_rate=rate, ood_delta_B0_pp=100*(rate-.5), train_wall_seconds=3600,
                           gpu_active_work_seconds=1800, parameter_count=100, mean_latency_ms=1, mean_replans=10)
                for split, count in config["test_episodes"].items():
                    row.update({split+"_success_rate": rate, split+"_episodes": count, split+"_successes": int(rate*count)})
                rows.append(row)
                deltas[rid] = dict(delta_pp=100*(rate-.5), paired_bootstrap_95ci_pp=[100*(rate-.5)-1, 100*(rate-.5)+1], draws=5000, scope="synthetic fixed models")
                development_rate = {"B0": .9 if task == tasks[0] else .4, "P1": .6, "P2": .3, "P3": .1, "P4": .8}[c]
                dev.append(dict(run_id=rid, candidate_id=c, ood_success_rate=development_rate, iid_success_rate=.5, parameter_count=100))
            initial = next(r for r in dev if r["candidate_id"] == "P1")
            final = max(dev, key=lambda r: r["ood_success_rate"])
            groups[f"{task}:n{n}"] = dict(initial_selected=initial, final_system=final, candidates=dev,
                                          b0_fallback=final["candidate_id"] == "B0")
        p4_count = 2 if task == tasks[1] else 0
        feedback.append(dict(task=task, decision="P4" if p4_count else "no_revision", feedback_cycles_used=1,
                             feedback_development_episodes=200, feedback_packet_frames=8,
                             p4_planned_trainings=p4_count, p4_completed_trainings=p4_count,
                             p4_development_episodes=50*p4_count, p4_train_wall_seconds=3600*p4_count,
                             p4_gpu_active_work_seconds=1800*p4_count,
                             p4_base_candidate_id="P2" if p4_count else None,
                             p4_principal_revision_intent="Synthetic representation change" if p4_count else None,
                             p4_debug_session_updates=20 if p4_count else 0,
                             p4_debug_session_wall_seconds=5.4 if p4_count else 0,
                             p4_debug_sessions=[dict(start_step=0, end_step=20, wall_seconds=5.4)] if p4_count else []))
    effects = []
    by_id = {row["run_id"]: row for row in rows}
    for item in feedback:
        if item["decision"] != "P4":
            continue
        for n in (5, 20):
            p4 = by_id[module.row_id(item["task"], "P4", n)]
            for kind, candidate in (("base-candidate", "P2"), ("initial-selected", "P1")):
                reference = by_id[module.row_id(item["task"], candidate, n)]
                delta = 100 * (p4["ood_success_rate"] - reference["ood_success_rate"])
                effects.append(dict(task=item["task"], train_n=n, reference_kind=kind,
                                    reference_candidate_id=candidate, reference_run_id=reference["run_id"], p4_run_id=p4["run_id"],
                                    reference_ood_success_rate=reference["ood_success_rate"], p4_ood_success_rate=p4["ood_success_rate"],
                                    paired_ood_delta=dict(delta_pp=delta, paired_bootstrap_95ci_pp=[delta-1, delta+1], draws=5000, scope="synthetic fixed models")))
    summary = dict(final=True, stage="test", selection_frozen=True, pending_training=[], results=rows,
                   invalid=[], planned_runs=98, evaluated_runs=98, completed_formal_runs=98,
                   completed_main_runs=96, completed_p4_runs=2, scored_episodes=9800,
                   paired_ood_deltas=deltas, feedback_accounting=feedback, feedback_effects=effects)
    selection = dict(frozen=True, unavailable=[], groups=groups)
    proposals = {t: dict(agent_backend="codex_session", designer={"name": "synthetic_designer"}) for t in tasks}
    bundles = {t: dict(demo_count=2, frame_count=16) for t in tasks}
    return summary, selection, config, proposals, bundles


def test_render_uses_development_choice_and_separates_p4_costs(module):
    args = fixture(module)
    result = module.render(*args, reduction_audit=True)
    assert "98/98" in result and "96/96" in result
    assert "GPU 0, 1, 2, 3（最多 4 卡" in result
    assert "gpu_reduction_to_0_3.json" in result
    assert "P1；75.0 / 75.0 / 75.0 / 75.0" in result
    assert "P3；" not in result  # P3 wins test only; it must never replace dev P1.
    assert "| 2 | 初始开发选优 | 6/6 | 75.0 / 75.0 / 75.0 / 75.0 | +25.00 |" in result
    assert "| 全部初始候选 P1/P2/P3 | 48 | 24 | 0 | 72 |" in result
    assert "| 最终系统 | 18 | 2 | 4 | 24 |" in result
    assert "| synthetic1 | 5 | 实际修订基线 | P2 | 25.0 / 25.0 / 25.0 / 25.0 | +0.0 [-1.0, +1.0] | -25.0 [-26.0, -24.0] |" in result
    assert "| synthetic1 | 5 | 初始开发选优 | P1 | 25.0 / 25.0 / 25.0 / 25.0 | -50.0 [-51.0, -49.0] | -25.0 [-26.0, -24.0] |" in result
    assert "| synthetic1 | P2 | Synthetic representation change | 20 | 5.400 |" in result
    assert "2 / 2 | 100 | 2.000 / 1.000" in result
    assert "没有 P4 干预" in result


def test_pending_or_inconsistent_tables_cannot_render(module):
    args = fixture(module)
    args[0]["results"][0]["status"] = "pending"
    with pytest.raises(module.NotReady, match="pending"):
        module.render(*args)
    args = fixture(module)
    args[1]["groups"]["synthetic0:n2"]["initial_selected"] = args[1]["groups"]["synthetic0:n2"]["final_system"]
    with pytest.raises(module.NotReady, match="开发规则"):
        module.render(*args)


def test_p4_baseline_and_debug_cost_mismatches_cannot_render(module):
    args = fixture(module)
    args[0]["feedback_effects"][0]["reference_run_id"] = module.row_id("synthetic1", "P1", 5)
    with pytest.raises(module.NotReady, match="混淆"):
        module.render(*args)
    args = fixture(module)
    args[0]["feedback_accounting"][1]["p4_debug_session_updates"] = 0
    with pytest.raises(module.NotReady, match="调试账目"):
        module.render(*args)
