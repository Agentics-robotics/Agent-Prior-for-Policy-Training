"""Fingerprint accepted Round 3 artifacts around a canonical resume invocation.

Run via Pixi. This operational check complements the numerical acceptance audit;
it uses stat signatures for its accepted inventory and hashes sources/logs/videos.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R3 = ROOT / "round3"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signature(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def snapshot(acceptance_path, render_path):
    acceptance, render = read(acceptance_path), read(render_path)
    assert acceptance["passed"] and acceptance["numerical_delivery_complete"]
    assert acceptance["stage"] == "test" and not any(acceptance["pending_counts"].values())
    assert render["passed"] and render["devices"] == [0, 1, 2, 3]
    manifest = R3 / "run_manifest.json"
    assert digest(manifest) == acceptance["run_manifest_snapshot_sha256"]
    runs = read(manifest)["runs"]
    assert len(runs) == len({r["run_id"] for r in runs}) == 98
    assert all(r[k] == "completed" for r in runs for k in
               ("status", "train_status", "dev_status", "test_status"))
    assert all(r.get(k) is None for r in runs for k in
               ("pid", "worker_pid", "launcher_pid", "scheduler_pid"))
    reviewed = read(R3 / "audits/canonical_resume_reuse_review.json")["source_sha256"]
    assert all(digest(ROOT / p) == h for p, h in reviewed.items())
    critical = {ROOT / p for p in reviewed}
    critical.update([manifest, R3 / "global_freeze.json", R3 / "selection.json",
                     Path(acceptance_path).resolve(), Path(render_path).resolve()])
    critical.update((R3 / "configs").rglob("*.json"))
    for name in ("initial_freeze.json", "proposal.json", "revision.json"):
        critical.update((R3 / "design_records").rglob(name))
    logs = set(R3.rglob("train.jsonl")) | set(R3.rglob("sessions.jsonl")) | {R3 / "events.jsonl"}
    videos, tasks = set(), set()
    for child in render["children"]:
        assert child["passed"] and child["policy_calls"] == child["scored_evaluation_episodes_added"] == 0
        tasks.update(child["tasks"])
        for record in child["records"]:
            assert record.get("status") != "pending"
            if "path" in record:
                path = ROOT / record["path"]
                assert digest(path) == record["video_sha256"]
                assert record["identity"]["renderer_sha256"] == digest(ROOT / "src/round3/report.py")
                assert read(path.with_suffix(".json")) == record
                videos.update([path, path.with_suffix(".json")])
    assert len(tasks) == 6
    inventories = {}
    episode_counts = {}
    for stage, directory, expected in (("dev", "dev_results", 4900), ("test", "locked_test_results", 9800)):
        paths = sorted(p for p in (R3 / directory).rglob("*") if p.is_file())
        inventories[stage] = [relative(p) for p in paths]
        episode_counts[stage] = sum(p.suffix == ".npz" for p in paths)
        assert episode_counts[stage] == expected
        assert sum(p.name == "complete.json" for p in paths) == 98
    return dict(
        accepted_stats={p: signature(ROOT / p) for p in acceptance["artifact_hashes"]},
        hashes={relative(p): digest(p) for p in sorted(critical | logs | videos)},
        log_and_video_stats={relative(p): signature(p) for p in sorted(logs | videos)},
        log_line_counts={relative(p): sum(1 for _ in p.open("rb")) for p in sorted(logs)},
        stage_file_inventories=inventories,
        episode_counts=episode_counts, completed_runs=len(runs),
        video_count=len(videos) // 2, video_tasks=sorted(tasks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["before", "after"])
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--render-audit", type=Path, required=True)
    args = parser.parse_args()
    state = snapshot(args.acceptance, args.render_audit)
    now = datetime.now(timezone.utc).isoformat()
    before_path = R3 / "audits/final_resume_reuse_before.json"
    if args.mode == "before":
        assert not before_path.exists(), "Preserve the original baseline"
        before_path.write_text(json.dumps(dict(created_at=now, snapshot=state)))
        print(json.dumps(dict(snapshot=str(before_path), accepted_files=len(state["accepted_stats"]),
                              episodes=state["episode_counts"], videos=state["video_count"])))
        return
    before = read(before_path)["snapshot"]
    differences = [k for k in state if state[k] != before.get(k)]
    detail = {}
    for field in differences:
        if isinstance(state[field], dict):
            keys = set(state[field]) | set(before.get(field, {}))
            detail[field] = [k for k in sorted(keys) if state[field].get(k) != before.get(field, {}).get(k)]
        else:
            detail[field] = dict(before=before.get(field), after=state[field])
    summary = read(R3 / "reports/test_summary.json")
    integrity = read(R3 / "reports/integrity.json")
    assert summary["final"] and integrity["final"] and integrity["stage"] == "test"
    assert integrity["all_completed_results_validated"]
    assert integrity["report_sha256"] == digest(R3 / "ROUND3_REPORT.md")
    result = dict(passed=not differences, completed_at=now,
                  baseline_sha256=digest(before_path), differences=detail,
                  accepted_files=len(state["accepted_stats"]),
                  unchanged_completed_runs=state["completed_runs"], episodes=state["episode_counts"],
                  videos=state["video_count"],
                  optimizer_updates_added=0 if not differences else None,
                  scored_episodes_added=0 if not differences else None,
                  video_replays_added=0 if not differences else None,
                  evidence="Unchanged accepted-file stat signatures, source/freeze/log/video hashes, log counts and episode inventories; canonical source cache/empty-queue review. This is not a kernel-level profiler.",
                  report_integrity_sha256=digest(R3 / "reports/integrity.json"))
    output = R3 / "audits/final_resume_reuse.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    assert not differences, "Canonical recovery changed an accepted artifact; inspect saved differences"


if __name__ == "__main__":
    main()
