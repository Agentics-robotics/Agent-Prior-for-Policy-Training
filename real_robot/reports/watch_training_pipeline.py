"""Compact read-only status for the authorized implementation/training runs."""
import argparse
import json
from pathlib import Path
import sqlite3

from appl.io import ROOT, read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="*", default=["push_training_v5", "flip_egg_training_v1"])
    args = parser.parse_args()
    for name in args.configs:
        cfg = read(ROOT / f"real_robot/configs/{name}.json")
        root = ROOT / cfg["run"]
        result = dict(run=name)
        for filename in ("workflow.json", "execution_status.json"):
            if (root / filename).exists():
                value = read(root / filename)
                result[filename] = {k:value[k] for k in ("status", "revision", "training_started", "error_type", "reason") if k in value}
        result["sessions"] = []
        for folder in sorted(root.glob("design_*")):
            journal = folder / "_session/journal.sqlite"
            if not journal.exists(): continue
            db = sqlite3.connect(f"file:{journal}?mode=ro", uri=True)
            session = dict(name=folder.name, API_states=dict(db.execute("SELECT status,COUNT(*) FROM api GROUP BY status")),
                           tools=dict(db.execute("SELECT name,COUNT(*) FROM tools GROUP BY name")))
            last = db.execute("SELECT seq,request,response FROM api ORDER BY seq DESC LIMIT 1").fetchone()
            if last:
                q = json.loads(last[1])
                session["last_request"] = dict(seq=last[0], model=q["model"], reasoning=q.get("reasoning"),
                                               same_prompt=q["instructions"] == (ROOT/cfg["prompt"]).read_text())
                if last[2]:
                    session["last_outputs"] = [dict(type=o["type"], name=o.get("name")) for o in json.loads(last[2])["output"]]
            tool = db.execute("SELECT name,status,result FROM tools ORDER BY rowid DESC LIMIT 1").fetchone()
            if tool:
                summary = None
                if tool[2]:
                    value = json.loads(tool[2])
                    if isinstance(value, dict):
                        summary = {k: value[k] for k in ("error", "path", "package_hash", "returncode", "num_examples", "coverage", "submitted", "status") if k in value}
                        if not summary:
                            summary = dict(keys=list(value))
                    else:
                        summary = "Image/evidence response"
                session["last_tool"] = dict(name=tool[0], status=tool[1], result=summary)
            db.close()
            if (folder / "cost_ledger.json").exists():
                ledger = read(folder / "cost_ledger.json")
                session["estimated_usd"] = sum(r.get("usd", 0) for r in ledger["records"])
                session["transport_stop"] = ledger["stopped"]
            session["work_files"] = sorted(p.name for p in (folder / "_session/work").glob("*"))
            result["sessions"].append(session)
        result["progress"] = {}
        for pattern in ("**/progress.json", "**/preparation_progress.json"):
            for p in root.glob(pattern):
                value = read(p)
                result["progress"][str(p.relative_to(root))] = {k:value[k] for k in (
                    "step", "maximum_updates", "loss_mean", "elapsed_seconds", "policy_id", "stage", "examined", "retained", "episode", "accepted", "candidate") if k in value}
        if (root / "launcher.log").exists() and result.get("workflow.json", {}).get("status") == "stopped":
            result["launcher_tail"] = (root / "launcher.log").read_text()[-1800:]
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
