"""One-command API design -> preparation review -> training -> calling checks."""

import ast
from pathlib import Path
import time

from appl.io import ROOT, atomic, digest, object_hash, read
from real_robot.data import Sources
from .design import design
from . import runner


def preflight(cfg):
    """Read-only preflight; never constructs a transport or creates an API session."""
    if cfg.get("test_cpu"):
        raise ValueError("CPU fixtures cannot launch a scientific API workflow")
    if Path(cfg["environment_manifest"]) != runner.MANIFEST.relative_to(ROOT):
        raise ValueError("This adapter uses the declared existing locked environment")
    cut_cfg = read(ROOT / cfg["cut_config"])
    if cut_cfg["run"] != cfg["cut_run"] or cut_cfg["output"] != cfg["cut_output"]:
        raise ValueError("Prior cut configuration and artifacts differ")
    manifest = read(ROOT / cfg["cut_output"] / "manifest.json")
    plan = read(ROOT / cfg["cut_output"] / "plan.json")
    if object_hash(plan) != manifest["plan_hash"]:
        raise ValueError("Prior API plan identity changed")
    source_manifest = read(ROOT / cfg["cut_run"] / "source_manifest.json")
    if digest(ROOT / cfg["cut_run"] / "source_manifest.json") != manifest["source_manifest_sha256"]:
        raise ValueError("Original source manifest identity changed")
    sources = Sources(cut_cfg)
    sources.verify_metadata(source_manifest)
    for path in (ROOT / "real_robot/training_pipeline").glob("*.py"):
        ast.parse(path.read_text())
    documents = {key: dict(path=cfg[key], sha256=digest(ROOT / cfg[key]))
                 for key in ("prompt", "interface", "task_specification", "config_path")}
    return dict(passed=True, documents=documents, prior_plan_hash=manifest["plan_hash"],
                source_manifest_sha256=manifest["source_manifest_sha256"],
                source_episodes=len(sources.episodes), source_rows=sum(e["n"] for e in sources.episodes.values()),
                model=cfg["model"], reasoning_effort=cfg["reasoning_effort"],
                network_calls=0, training_runs=0, robot_execution=False,
                media_verification="Full source media hashes are checked at workflow launch")


def run(cfg):
    checked = preflight(cfg)
    root = ROOT / cfg["run"]
    root.mkdir(parents=True, exist_ok=True)
    if any(root.glob("design_*")) or any(root.glob("package_*")):
        raise ValueError("Existing experiment requires explicit reconciliation; no automatic replay")
    # Exclusive marker remains after exit, including interruption/uncertain transport.
    with (root / "workflow.started").open("x") as file:
        file.write(str(time.time()) + "\n")
    atomic(root / "preflight.json", checked)
    feedback, previous = None, None
    try:
        sources = Sources(read(ROOT / cfg["cut_config"]))
        sources.verify_media(read(ROOT / cfg["cut_run"] / "source_manifest.json"))
        for revision in range(cfg["max_implementation_revisions"]):
            spent = 0.0
            for ledger_path in root.glob("design_*/cost_ledger.json"):
                ledger = read(ledger_path)
                if ledger["stopped"] or any("usd" not in r for r in ledger["records"]):
                    raise RuntimeError("Unresolved API transport/cost accounting; no retry")
                spent += sum(r["usd"] for r in ledger["records"])
            remaining = cfg["execution_guard_usd"] - spent
            if remaining <= 0:
                raise RuntimeError("Workflow execution cost guard reached")
            atomic(root / "workflow.json", dict(status="API_design", revision=revision,
                   remaining_execution_guard_usd=remaining, feedback=feedback))
            # A repair is a new API design session with exact failure evidence.
            # Transport exceptions from design are never caught as repairable failures.
            submission = design(dict(cfg, execution_guard_usd=remaining), revision, feedback, previous)
            package = root / f"package_{revision:02d}"
            if submission["readiness"]["decision"] == "blocked":
                result = dict(status="API_declared_blocked", revision=revision,
                              readiness=submission["readiness"], training_started=False)
                atomic(root / "workflow.json", result)
                return result
            try:
                library = runner.execute(cfg, revision)
            except runner.ImplementationFailure as error:
                feedback = dict(kind="implementation_execution_failure", revision=revision,
                                diagnostics=error.diagnostics,
                                instruction="Repair implementation or reassess readiness from these actual diagnostics.")
                previous = package / "source"
                atomic(package / "automatic_repair_feedback.json", feedback)
                continue
            result = dict(status="complete", revision=revision, library=library)
            atomic(root / "workflow.json", result)
            return result
        result = dict(status="implementation_repair_limit", revisions=cfg["max_implementation_revisions"], feedback=feedback)
        atomic(root / "workflow.json", result)
        return result
    except Exception as error:
        atomic(root / "workflow.json", dict(status="stopped", error_type=type(error).__name__,
               reason=str(error), automatic_transport_retry=False))
        raise
