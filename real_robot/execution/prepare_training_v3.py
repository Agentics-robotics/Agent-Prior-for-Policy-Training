"""Run submitted API preprocessing once, retaining a coverage review before fit."""
import argparse
import json
import time
from appl.io import ROOT, atomic, read
from real_robot.policy_training_v3.design import configuration
from real_robot.policy_training_v3.runner import freeze_executor, start_job, finish_job, audit_designs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", type=int, default=0)
    args = parser.parse_args()
    cfg = configuration(ROOT / "real_robot/configs/push_training_v3.json")
    root = ROOT / cfg["run"]
    assert read(root / f"design_{args.revision:02d}/status.json")["status"] == "complete"
    _, _, account = audit_designs(cfg)
    package = root / f"package_{args.revision:02d}"
    freeze_executor(cfg, package)
    atomic(root / "execution_status.json", dict(status="preparing", revision=args.revision,
                                               started=time.time(), API_accounting=account))
    result = finish_job(start_job(cfg, package, "prepare", package / "prepared", cfg["devices"][0]))
    atomic(root / "execution_status.json", dict(
        status="preparation_failed" if result["returncode"] else "prepared_awaiting_coverage_review",
        revision=args.revision, result=result, API_accounting=account))
    print(json.dumps(result))
    if result["returncode"]:
        raise RuntimeError("API preprocessing failed; retained exact evidence for API-authored repair")


if __name__ == "__main__":
    main()
