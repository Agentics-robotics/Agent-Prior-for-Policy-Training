"""Launch authorized generic training workflows with persistent local logs."""
import argparse
from pathlib import Path
import subprocess
import time

from appl.io import ROOT, atomic
from real_robot.training_pipeline.contract import configuration
from real_robot.training_pipeline.runner import command, environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()
    for path in args.configs:
        cfg = configuration(ROOT / path)
        root = ROOT / cfg["run"]
        root.mkdir(parents=True, exist_ok=True)
        if (root / "launcher.json").exists() or (root / "workflow.started").exists():
            raise ValueError("A recorded run cannot be silently relaunched")
        with (root / "launcher.log").open("x") as log:
            proc = subprocess.Popen(command(["run", "--config", path]), cwd=ROOT,
                env=environment(), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                start_new_session=True)
        atomic(root / "launcher.json", dict(pid=proc.pid, started=time.time(), config=path,
            automatic_retry=False, model=cfg["model"], reasoning_effort=cfg["reasoning_effort"]))
        print(f"{path}: pid={proc.pid}, log={root / 'launcher.log'}", flush=True)


if __name__ == "__main__":
    main()
