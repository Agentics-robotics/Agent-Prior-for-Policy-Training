"""Task-independent entry including bounded API-owned data/design recovery."""
import argparse
from pathlib import Path
import shutil

from appl.io import ROOT, atomic, digest
from real_robot.training_pipeline.contract import configuration
from real_robot.training_pipeline.workflow import run as initial_run
from .continue_training_pipeline import run as recovery_run


def run(cfg):
    root = ROOT / cfg["run"]
    root.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__).resolve(), Path(__file__).with_name("continue_training_pipeline.py")]
    marker = root / "workflow_driver.json"
    if marker.exists():
        raise ValueError("Existing recorded driver requires explicit reconciliation")
    atomic(marker, {str(p.relative_to(ROOT)):digest(p) for p in paths})
    for p in paths:
        target = root / "workflow_driver_snapshot" / p.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    result = initial_run(cfg)
    while (result["status"] == "API_declared_blocked" and
           result["revision"] + 1 < cfg["max_implementation_revisions"]):
        result = recovery_run(cfg)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(configuration(ROOT / args.config)), flush=True)


if __name__ == "__main__":
    main()
