"""Detach the generic API-owned implementation/training driver with provenance."""
import argparse
import subprocess
import time

from appl.io import ROOT, atomic, digest
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
            raise ValueError("Existing recorded run needs explicit reconciliation")
        cmd = command(["--config", path])
        cmd[cmd.index("real_robot.training_pipeline")] = "real_robot.execution.train_policies"
        with (root / "launcher.log").open("x") as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=environment(),
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        atomic(root / "launcher.json", dict(pid=proc.pid, started=time.time(), config=path,
            command=cmd, launcher_sha256=digest(__file__), automatic_retry=False,
            model=cfg["model"], reasoning_effort=cfg["reasoning_effort"]))
        print(dict(config=path, pid=proc.pid, log=str(root / "launcher.log")), flush=True)


if __name__ == "__main__":
    main()
