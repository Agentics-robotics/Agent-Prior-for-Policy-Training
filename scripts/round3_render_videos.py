"""Render accepted Round 3 cached trajectories on physical GPUs 0–3.

Run through Pixi after the full test numerical audit, then run round3-report.
This refuses incomplete/stale acceptance and busy device leases. Four isolated
Pixi children share six fixed task jobs, with no policy, expert, or score calls.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
R3 = ROOT / "round3"
SCRIPT = Path(__file__).resolve()
GPUS = (0, 1, 2, 3)
THREADS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
EXPECTED_RUNS = 98


def require(condition, message):
    if not condition:
        raise ValueError(message)


def configure_device(gpu=None):
    """Called inside Pixi before importing NumPy, Torch, or the simulator."""
    require(gpu is None or gpu in GPUS, f"Unauthorized physical GPU: {gpu}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    os.environ["MUJOCO_GL"] = "egl"
    if gpu is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
    for variable in THREADS:
        os.environ[variable] = "1"
    # A renderer always owns its leases, even if launched inside another worker.
    os.environ.pop("ROUND3_GPU_LEASE_PID", None)


def assignments():
    from round3.common import TASKS
    return {gpu: list(TASKS[gpu::len(GPUS)]) for gpu in GPUS}


def input_fingerprint(audit_path):
    from relative_dp.utils import sha256
    return dict(audit_path=str(Path(audit_path).resolve()), audit_sha256=sha256(audit_path),
                global_freeze_sha256=sha256(R3 / "global_freeze.json"),
                selection_sha256=sha256(R3 / "selection.json"),
                driver_sha256=sha256(SCRIPT), renderer_sha256=sha256(ROOT / "src/round3/report.py"),
                accepted_test_runs=EXPECTED_RUNS)


def accepted_inputs(audit_path):
    """Revalidate audit-bound bytes without rerunning numerical acceptance.

    Only existence is inspected under locked_test_results until the complete
    numerical acceptance and every current global-freeze binding pass. Results
    from other tasks are hashed, never loaded into a worker's result objects.
    """
    from relative_dp.utils import read_json, sha256
    from round3.common import TASKS, NS
    from round3.evaluate import validate_test_gate

    manifest_path = R3 / "run_manifest.json"
    runs = read_json(manifest_path)["runs"]
    require(len(runs) == EXPECTED_RUNS and all(r["status"] != "invalid" for r in runs),
            "Renderer requires the current 98 valid formal runs")
    identifiers = {r["run_id"] for r in runs}
    require(len(identifiers) == EXPECTED_RUNS, "Duplicate formal run IDs")
    missing = [rid for rid in sorted(identifiers)
               if not (R3 / "locked_test_results" / rid / "complete.json").is_file()]
    require(not missing, f"Rendering waits for all 98 test results; missing {len(missing)}: {missing[:3]}")
    audit = read_json(audit_path)
    for field in ("passed", "audit_scope_complete", "numerical_delivery_complete",
                  "main_matrix_training_complete", "main_matrix_dev_complete", "main_matrix_test_complete",
                  "global_freeze_validated", "locked_test_contents_accessed"):
        require(audit.get(field) is True, f"Numerical test acceptance is incomplete: {field}")
    require(audit.get("stage") == "test" and audit.get("pending_counts") == dict(training=0, dev=0, test=0),
            "Numerical acceptance must cover completed locked tests")
    require(audit["counts"].get("test_runs_checked") == EXPECTED_RUNS and
            audit["counts"].get("test_episodes_checked") == EXPECTED_RUNS * 100 and
            audit["counts"].get("frozen_selection_groups_checked") == 24 and
            audit["counts"].get("feedback_decisions_checked") == 6 and not audit.get("invalid_runs"),
            "Numerical acceptance does not cover all 98 tests, 24 selections and six feedback decisions")
    require(set(audit["result_totals"]["test"]) == identifiers, "Accepted test run matrix differs")
    require(audit["auditor_sha256"] == sha256(ROOT / "scripts/verify_round3_delivery.py"),
            "Numerical auditor source changed; use a current acceptance audit")
    require(audit["run_manifest_snapshot_sha256"] == sha256(manifest_path),
            "Run ledger changed since numerical acceptance")
    bound = audit["artifact_hashes"]
    gate_path = R3 / "global_freeze.json"
    require(bound.get("round3/global_freeze.json") == sha256(gate_path),
            "Global freeze differs from numerical acceptance")
    gate = validate_test_gate()
    require(gate["planned_runs"] == EXPECTED_RUNS and gate["selection_groups"] == 24 and
            all(bound.get(name) == digest for name, digest in gate["bound_files"].items()),
            "Global freeze is not fully bound by numerical acceptance")
    require(bound.get("round3/selection.json") == sha256(R3 / "selection.json"),
            "Development selection differs from numerical acceptance")
    selection = read_json(R3 / "selection.json")
    require(selection.get("frozen") is True and
            set(selection["groups"]) == {f"{task}:n{n}" for task in TASKS for n in NS},
            "Development selection is incomplete or unfrozen")
    # The accepted complete aggregates bind trajectory hashes; also preserve
    # every standalone row/trajectory byte checked in the numerical audit.
    for rid in sorted(identifiers):
        prefix = f"round3/locked_test_results/{rid}/"
        files = {name: digest for name, digest in bound.items() if name.startswith(prefix)}
        require(prefix + "complete.json" in files and
                sum(name.endswith(".npz") for name in files) == 100 and
                sum(name.endswith(".json") for name in files) == 101,
                f"Acceptance omits complete test evidence: {rid}")
        for name, digest in files.items():
            path = (ROOT / name).resolve()
            require(path.is_relative_to(R3 / "locked_test_results" / rid), "Accepted result path escapes its run")
            require(sha256(path) == digest, f"Accepted test artifact changed: {name}")
    return selection, input_fingerprint(audit_path)


def needed_results(task, selection):
    from round3.common import run_id
    from round3.report import _checked_result, _selected
    results = {}
    for n in (20, 5, 2, 10):
        chosen = _selected(selection, task, n, "initial_selected")
        require(chosen is not None, f"No frozen initial-selected model: {task}/N{n}")
        for rid in (run_id(task, "B0", n), chosen):
            result = _checked_result(R3 / "locked_test_results" / rid / "complete.json")
            require(result["identity"]["stage"] == "test" and result["identity"]["run_id"] == rid,
                    f"Wrong cached test identity: {rid}")
            results[rid] = result
    return results


@contextlib.contextmanager
def device_leases(gpu):
    from round3.common import lock
    require(gpu in GPUS, f"Unauthorized physical GPU: {gpu}")
    with lock(f"gpu-{gpu}", blocking=False), lock(f"gpu-{gpu}-slot-1", blocking=False):
        yield


def worker(gpu, audit_path, expected, output):
    from relative_dp.utils import atomic_json
    from round3.common import now
    from round3.report import test_videos
    require(gpu in GPUS, f"Unauthorized physical GPU: {gpu}")
    with device_leases(gpu):
        selection, current = accepted_inputs(audit_path)
        require(current == expected, "Renderer inputs/source changed after parent preflight")
        records = []
        for task in assignments()[gpu]:
            records.extend(test_videos(needed_results(task, selection), selection,
                                       tasks=[task], write_manifest=False))
        require(records and not any(r.get("status") == "pending" for r in records), "Task videos remain pending")
        require(input_fingerprint(audit_path) == expected, "Renderer inputs/source changed during rendering")
        require(all(r["identity"]["renderer_sha256"] == expected["renderer_sha256"]
                    for r in records if "path" in r), "Mixed renderer identities in task videos")
        atomic_json(output, dict(passed=True, completed_at=now(), physical_gpu=gpu,
                                tasks=assignments()[gpu], inputs=current, records=records,
                                policy_calls=0, scored_evaluation_episodes_added=0))


def child_command(gpu, audit_path, expected_path, output):
    from round3.cli import pixi_binary
    require(gpu in GPUS, f"Unauthorized physical GPU: {gpu}")
    return [pixi_binary(), "run", "env", f"CUDA_VISIBLE_DEVICES={gpu}",
            f"MUJOCO_EGL_DEVICE_ID={gpu}", "MUJOCO_GL=egl", *[f"{name}=1" for name in THREADS],
            "python", str(SCRIPT), "--worker", "--gpu", str(gpu), "--audit", str(audit_path),
            "--expected", str(expected_path), "--output", str(output)]


def wait_children(children):
    from round3.cli import _process_group_live
    while children:
        remaining = []
        for gpu, process in children:
            code = process.poll()
            if code is not None and code != 0:
                raise RuntimeError(f"GPU {gpu} video child exited {code}; inspect its render log")
            if code is None or _process_group_live(process.pid):
                remaining.append((gpu, process))
        children = remaining
        if children:
            time.sleep(.2)


def launch(audit_path):
    from relative_dp.utils import atomic_json, read_json
    from round3.common import lock, now
    from round3.cli import _interrupt_and_wait, _process_group_live
    # Wait/refuse via nonblocking lease: a live test scheduler always keeps this.
    with lock("orchestrator", blocking=False):
        _, expected = accepted_inputs(audit_path)
        directory = R3 / "audits" / ("video_render_" + now().replace(":", "").replace(".", ""))
        directory.mkdir(parents=True, exist_ok=False)
        expected_path = directory / "inputs.json"
        atomic_json(expected_path, expected)
        children, receipts = [], []
        try:
            for gpu in GPUS:
                output = directory / f"gpu_{gpu}.json"
                receipts.append(output)
                with (directory / f"gpu_{gpu}.log").open("w") as log:
                    process = subprocess.Popen(child_command(gpu, audit_path, expected_path, output),
                                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                children.append((gpu, process))
            wait_children(children)
        except BaseException:
            # Signals are confined to these newly created child process groups.
            # Keep the orchestrator lease until every owned descendant exits.
            for _, process in children:
                if process.poll() is None or _process_group_live(process.pid):
                    _interrupt_and_wait(process)
            raise
        completed = [read_json(path) for path in receipts]
        require(all(r["passed"] and r["inputs"] == expected and r["physical_gpu"] == gpu and
                    r["tasks"] == assignments()[gpu] for gpu, r in zip(GPUS, completed)),
                "Child completion receipts do not match the accepted rendering jobs")
        require(input_fingerprint(audit_path) == expected, "Renderer inputs/source changed before final completion")
        result = dict(passed=True, completed_at=now(), inputs=expected, devices=list(GPUS),
                      assignments=assignments(), children=completed, policy_calls=0,
                      scored_evaluation_episodes_added=0,
                      next_step="pixi run round3-report validates/reuses these videos and writes the full manifest")
        atomic_json(directory / "complete.json", result)
        print(json.dumps(dict(passed=True, render_audit=str(directory / "complete.json"),
                              next_step=result["next_step"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True, help="Successful complete numerical --stage test audit")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu", type=int, choices=GPUS, help=argparse.SUPPRESS)
    parser.add_argument("--expected", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(not args.worker or all(x is not None for x in (args.gpu, args.expected, args.output)),
            "Internal worker requires GPU, expected inputs and receipt output")
    require(args.worker or all(x is None for x in (args.gpu, args.expected, args.output)),
            "GPU/input/output options are internal to isolated task workers")
    configure_device(args.gpu if args.worker else None)
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from relative_dp.utils import read_json

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Video rendering interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    if args.worker:
        worker(args.gpu, args.audit.resolve(), read_json(args.expected), args.output.resolve())
    else:
        launch(args.audit.resolve())


if __name__ == "__main__":
    main()
