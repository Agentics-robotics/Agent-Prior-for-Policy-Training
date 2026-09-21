"""Record final preservation and API-tool evidence for the submitted cut_v5."""

from collections import Counter
import json
import sqlite3

from appl.io import atomic, digest, read
from real_robot.data import ROOT


def main():
    run = ROOT / "real_robot/runs/push_letters/cut_v5"
    status = read(run / "status.json")
    validation = read(run / "validation.json")
    review = read(run / "review.json")
    assert status["status"] == "complete" and validation["passed"] and review["passed"]
    previous = {}
    for version in ("cut_v3", "cut_v4"):
        frozen = read(ROOT / "real_robot/runs/push_letters" / version / "interface_freeze.json")
        manifest = read(ROOT / "real_robot/data/push_letters" / version / "manifest.json")
        for name, h in frozen["files"].items():
            assert digest(ROOT / name) == h, name
        for name, h in manifest["files"].items():
            assert digest(ROOT / "real_robot/data/push_letters" / version / name) == h, name
        previous[version] = dict(frozen_files=len(frozen["files"]), published_files=len(manifest["files"]))
    db = sqlite3.connect("file:" + str(run / "_session/journal.sqlite") + "?mode=ro", uri=True)
    calls = [dict(name=n, args=json.loads(a)) for n, a in db.execute("SELECT name,args FROM tools ORDER BY rowid")]
    states = dict(db.execute("SELECT status,COUNT(*) FROM api GROUP BY status"))
    db.close()
    assert set(states) == {"consumed"}
    ledger = read(run / "cost_ledger.json")
    assert ledger["stopped"] is None and all("usd" in r for r in ledger["records"])
    result = dict(
        date="2026-09-21", status="complete", previous_versions_unchanged=previous,
        api_states=states, tool_counts=dict(Counter(c["name"] for c in calls)),
        motion_inspections=[c["args"] for c in calls if c["name"] == "read_motion"],
        state_inspection_counts={tid: len({i for c in calls if c["name"] == "read_steps"
                                         and c["args"]["trajectory_id"] == tid
                                         for i in c["args"]["indices"]})
                                 for tid in review["coverage"]},
        groups=len(review["groups"]), policies=review["policies"], segments=review["segments"],
        estimated_usd=sum(r["usd"] for r in ledger["records"]),
        input_preprocessing_implemented=False, training_runs=0, robot_execution=False,
        interpretation="Exact API design preserved; review and counts are developer-authored",
        standing_authorization="real_robot/authorizations/openai_data_processing_20260921.json",
    )
    atomic(run / "completion_receipt.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
