"""Bounded, generic design-recovery continuation of a completed API assessment.

This records a NEW API revision. It never replays interrupted transport, mutates
submitted science or prescribes task-specific representations/labels.
"""
import argparse
from pathlib import Path
import time

from appl.io import ROOT, atomic, read
from real_robot.training_pipeline.contract import configuration
from real_robot.training_pipeline.design import design
from real_robot.training_pipeline.runner import ImplementationFailure, execute


RECOVERY_INSTRUCTION = """Complete the authorized policy implementation and training using the supplied
cut/prior and original evidence. Your preceding assessment identifies problems
in the current derived targets or implementation; treat recoverable design and
data-processing problems as work you own. Determine which can be addressed with
the available observations, model/perception capabilities, target derivation,
uncertainty treatment, representation or supervision choices. Carry out reasonable
evidence-based recovery and inspect the actual result, rather than handing an
unimplemented recovery plan back as a finished package. You may refine the prior's
implementation and derived data plan, recording why; there is no required target
schema, policy count or sample whitelist beyond the generic interface.

Distinguish information needed to construct defensible training targets from
information needed to commission physical robot execution. Missing deployment
bindings need not prevent offline training when the learning problem has valid
supervision. Conversely, do not claim unreliable labels are verified or force
training on them. If completing training truly requires unavailable external
information after the supported recovery attempts, identify the precise missing
input, the recovery attempted and why remaining feasible learning choices do not
meet the task. Keep the original cut, completed assessment and all failed attempts.
No new cut-stage experiment or physical robot execution is requested."""


def run(cfg):
    root = ROOT / cfg["run"]
    state = read(root / "workflow.json")
    if state["status"] != "API_declared_blocked":
        raise ValueError("This continuation starts only from a completed API data/design assessment")
    previous_revision = state["revision"]
    with (root / f"continuation_after_{previous_revision:02d}.started").open("x") as file:
        file.write(str(time.time()) + "\n")
    atomic(root / f"assessment_before_continuation_{previous_revision:02d}.json", state)
    previous = root / f"package_{previous_revision:02d}/source"
    feedback = dict(kind="continue_API_owned_data_and_design_recovery",
                    instruction=RECOVERY_INSTRUCTION, previous_assessment=state["readiness"],
                    authority="User 2026-09-22 authorized both trainings and task-independent workflow refinements")
    try:
        for revision in range(previous_revision + 1, cfg["max_implementation_revisions"]):
            spent = 0.0
            for path in root.glob("design_*/cost_ledger.json"):
                ledger = read(path)
                if ledger["stopped"] or any("usd" not in r for r in ledger["records"]):
                    raise RuntimeError("Unresolved transport/cost accounting; no automatic replay")
                spent += sum(r["usd"] for r in ledger["records"])
            remaining = cfg["execution_guard_usd"] - spent
            if remaining <= 0:
                raise RuntimeError("Existing workflow cost guard exhausted")
            atomic(root / "workflow.json", dict(status="API_design_recovery", revision=revision,
                   remaining_execution_guard_usd=remaining, feedback=feedback))
            submission = design(dict(cfg, execution_guard_usd=remaining), revision, feedback, previous)
            package = root / f"package_{revision:02d}"
            if submission["readiness"]["decision"] == "blocked":
                result = dict(status="API_declared_blocked", revision=revision,
                              readiness=submission["readiness"], training_started=False,
                              recovery_instruction=RECOVERY_INSTRUCTION)
                atomic(root / "workflow.json", result)
                return result
            try:
                library = execute(cfg, revision)
            except ImplementationFailure as error:
                feedback = dict(kind="implementation_execution_failure", diagnostics=error.diagnostics,
                                instruction="Repair the implementation from these recorded diagnostics.")
                previous = package / "source"
                atomic(package / "automatic_repair_feedback.json", feedback)
                continue
            result = dict(status="complete", revision=revision, library=library)
            atomic(root / "workflow.json", result)
            return result
        result = dict(status="implementation_repair_limit", feedback=feedback)
        atomic(root / "workflow.json", result)
        return result
    except Exception as error:
        atomic(root / "workflow.json", dict(status="stopped", error_type=type(error).__name__,
               reason=str(error), automatic_transport_retry=False))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(configuration(ROOT / args.config)), flush=True)


if __name__ == "__main__":
    main()
