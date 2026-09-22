"""Developer-owned scheduling and provenance for geometric decision training."""

from pathlib import Path
import json
import shutil
import sqlite3
import subprocess
import time

from appl.io import ROOT, PIXI, MANIFEST, atomic, digest, read
from appl.gpu import identity

MANIFEST = ROOT / "real_robot/training_v4_environment/pixi.toml"


def command(args):
    return [
        PIXI,
        "run",
        "--manifest-path",
        str(MANIFEST),
        "--locked",
        "--no-install",
        "python",
        "-u",
        "-m",
        "real_robot.policy_training_v4",
        *args,
    ]


def environment():
    return dict(
        PATH="/usr/bin:/bin",
        LANG="C.UTF-8",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        PYTHONPATH=str(ROOT) + ":" + str(ROOT / "src"),
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )


def freeze_executor(cfg, package):
    source = package / "source"
    submission = read(package / "submission.json")
    for name, sha in submission["files"].items():
        if digest(source / name) != sha:
            raise ValueError("API-authored source changed: " + name)
    for name, sha in submission["assets"].items():
        if digest(ROOT / cfg["run"] / "assets" / name) != sha:
            raise ValueError("Declared asset changed: " + name)
    paths = list((ROOT / "real_robot/policy_training_v4").glob("*.py")) + [
        ROOT / cfg["config_path"],
        ROOT / "real_robot/data.py",
        ROOT / "src/appl/gpu.py",
        ROOT / "src/appl/kernel.py",
        ROOT / "src/appl/io.py",
        MANIFEST,
        MANIFEST.with_name("pixi.lock"),
    ]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    marker = package / "executor_freeze.json"
    if marker.exists() and read(marker) != hashes:
        raise ValueError(
            "Executor changed after training launch; retain a new explicit revision"
        )
    atomic(marker, hashes)
    for p in paths:
        target = package / "executor_snapshot" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(p, target)


def start_job(cfg, package, operation, output, gpu, policy_id=None):
    if gpu not in cfg["devices"]:
        raise ValueError("GPU outside run allocation")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    device = identity(gpu)
    occupancy = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    )
    atomic(
        output / "gpu_occupancy.json",
        dict(device=device, compute_processes=occupancy, checked=time.time()),
    )
    args = [
        "worker",
        "--config",
        cfg["config_path"],
        "--package",
        str(package),
        "--operation",
        operation,
        "--output",
        str(output),
        "--gpu",
        str(gpu),
    ]
    if policy_id:
        args += ["--policy-id", policy_id]
    stdout = (output / "stdout.log").open("w")
    stderr = (output / "stderr.log").open("w")
    proc = subprocess.Popen(
        command(args), cwd=ROOT, env=environment(), stdout=stdout, stderr=stderr
    )
    atomic(
        output / "process.json",
        dict(
            pid=proc.pid,
            gpu=gpu,
            started=time.time(),
            operation=operation,
            policy_id=policy_id,
        ),
    )
    return dict(
        process=proc,
        output=output,
        stdout=stdout,
        stderr=stderr,
        started=time.monotonic(),
        policy_id=policy_id,
    )


def finish_job(job):
    code = job["process"].wait()
    job["stdout"].close()
    job["stderr"].close()
    result = dict(
        returncode=code,
        wall_seconds=time.monotonic() - job["started"],
        policy_id=job["policy_id"],
        automatic_retry=False,
    )
    atomic(job["output"] / "process_result.json", result)
    if code:
        result["error"] = (job["output"] / "stderr.log").read_text()[-16000:]
        atomic(job["output"] / "failure.json", result)
        return result
    result["result"] = read(job["output"] / "result.json")
    return result


def execute(cfg, revision):
    root = ROOT / cfg["run"]
    package = root / f"package_{revision:02d}"
    freeze_executor(cfg, package)
    atomic(
        root / "execution_status.json",
        dict(status="preparing", revision=revision, started=time.time()),
    )
    cache = package / "prepared"
    if (cache / "result.json").exists():
        manifest = read(cache / "result.json")
        for name, record in manifest["arrays"].items():
            if digest(cache / record["file"]) != record["sha256"]:
                raise ValueError("Cache changed: " + name)
        if digest(cache / "metadata.json") != manifest["metadata_sha256"]:
            raise ValueError("Prepared metadata changed")
    else:
        prepared = finish_job(
            start_job(cfg, package, "prepare", cache, cfg["devices"][0])
        )
        if prepared["returncode"]:
            atomic(
                root / "execution_status.json",
                dict(status="preparation_failed", revision=revision, result=prepared),
            )
            raise RuntimeError(
                "API preprocessing failed; receipt retained for explicit API repair"
            )
    atomic(
        root / "execution_status.json",
        dict(status="training", revision=revision, policies=cfg["policies"]),
    )
    jobs = []
    for policy_id, gpu in zip(cfg["policies"], cfg["devices"], strict=True):
        output = package / "training" / policy_id
        if (output / "result.json").exists():
            continue
        jobs.append(start_job(cfg, package, "train", output, gpu, policy_id))
    results = [finish_job(job) for job in jobs]
    if any(r["returncode"] for r in results):
        atomic(
            root / "execution_status.json",
            dict(status="training_failed", revision=revision, results=results),
        )
        raise RuntimeError(
            "Training failure retained; no automatic restart or API retry"
        )
    return publish(cfg, revision)


def audit_designs(cfg):
    root = ROOT / cfg["run"]
    writes = {}
    submissions = []
    calls = 0
    cost = 0
    unknown = 0
    for folder in sorted(root.glob("design_[0-9][0-9]")):
        db = sqlite3.connect(
            f"file:{folder / '_session/journal.sqlite'}?mode=ro", uri=True
        )
        prompt = read(folder / "prompt.json")
        for seq, status, request, response in db.execute(
            "SELECT seq,status,request,response FROM api ORDER BY seq"
        ):
            if status != "consumed":
                raise ValueError("Unreconciled API call")
            q = json.loads(request)
            r = json.loads(response)
            if (
                q["model"] != "gpt-6-astra"
                or q["reasoning"]["effort"] != "xhigh"
                or r.get("model") != "gpt-6-astra"
                or r.get("reasoning", {}).get("effort") != "xhigh"
                or q["instructions"] != prompt["instructions"]
                or json.loads(q["input"][0]["content"]) != prompt["initial_message"]
            ):
                raise ValueError("Actual API request identity or prompt mismatch")
            calls += 1
            for item in r["output"]:
                if item.get("type") != "function_call":
                    continue
                args = json.loads(item["arguments"])
                if item["name"] == "write_file":
                    writes[(args["path"], hashlib_bytes(args["content"]))] = dict(
                        session=folder.name, seq=seq
                    )
                elif item["name"] == "submit_package":
                    submissions.append(args["expected_hash"])
        db.close()
        ledger = read(folder / "cost_ledger.json")
        cost += sum(r.get("usd", 0) for r in ledger["records"])
        unknown += sum(r["reserved_usd"] for r in ledger["records"] if "usd" not in r)
    return (
        writes,
        submissions,
        dict(
            API_calls=calls,
            model="gpt-6-astra",
            reasoning_effort="xhigh",
            estimated_usd=cost,
            unknown_reserved_usd=unknown,
            cost_is_invoice=False,
        ),
    )


def hashlib_bytes(text):
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def publish(cfg, revision):
    root = ROOT / cfg["run"]
    package = root / f"package_{revision:02d}"
    submission = read(package / "submission.json")
    writes, submissions, account = audit_designs(cfg)
    if submission["package_hash"] not in submissions:
        raise ValueError("Missing explicit API submission")
    authorship = {}
    for name, sha in submission["files"].items():
        if digest(package / "source" / name) != sha or (name, sha) not in writes:
            raise ValueError("Missing exact API source authorship: " + name)
        authorship[name] = writes[(name, sha)]
    policies = []
    for policy_id in cfg["policies"]:
        result = read(package / "training" / policy_id / "result.json")
        if (
            result["optimizer_steps"] != read(package / "source/package.json")["policies"][policy_id]["updates"]
            or not result["neural_training_verified"]
            or digest(result["checkpoint"]) != result["checkpoint_sha256"]
        ):
            raise ValueError("Incomplete or changed trained policy")
        policies.append(
            dict(
                policy_id=policy_id,
                source=str(package / "source"),
                checkpoint=result["checkpoint"],
                checkpoint_sha256=result["checkpoint_sha256"],
                source_hashes=submission["files"],
                training_result=str(package / "training" / policy_id / "result.json"),
                API_calling_document=str(package / "source/CALLING.md"),
                **{
                    k: result[k]
                    for k in [
                        "optimizer_steps",
                        "trainable_parameters",
                        "training_examples",
                        "reload_max_prediction_error",
                        "device",
                    ]
                },
            )
        )
    library = dict(
        schema="real_robot.trained_library.v4",
        policies=policies,
        cut_plan_hash=submission["source_plan_hash"],
        API_source_authorship=authorship,
        API_accounting=account,
        stage="geometric_decision_policy_trained",
        robot_execution=False,
        generalization_performance="not evaluated",
    )
    atomic(root / "library.json", library)
    atomic(
        root / "execution_status.json",
        dict(
            status="complete",
            revision=revision,
            finished=time.time(),
            library=str(root / "library.json"),
            API_accounting=account,
        ),
    )
    return library


def status(cfg):
    root = ROOT / cfg["run"]
    value = {}
    if (root / "execution_status.json").exists():
        value["execution"] = read(root / "execution_status.json")
    for folder in sorted(root.glob("design_[0-9][0-9]")):
        if not (folder / "_session/journal.sqlite").exists():
            continue
        db = sqlite3.connect(
            f"file:{folder / '_session/journal.sqlite'}?mode=ro", uri=True
        )
        current = dict(
            API_states=dict(
                db.execute("SELECT status,COUNT(*) FROM api GROUP BY status")
            ),
            tools=dict(db.execute("SELECT name,COUNT(*) FROM tools GROUP BY name")),
        )
        if (folder / "status.json").exists():
            current["status"] = read(folder / "status.json")["status"]
        if (folder / "cost_ledger.json").exists():
            ledger = read(folder / "cost_ledger.json")
            current["estimated_usd"] = sum(r.get("usd", 0) for r in ledger["records"])
            current["transport_stop"] = ledger["stopped"]
        value[folder.name] = current
        db.close()
    for package in sorted(root.glob("package_[0-9][0-9]")):
        results = {}
        for path in package.glob("**/progress.json"):
            results[str(path.relative_to(package))] = read(path)
        for path in package.glob("**/preparation_progress.json"):
            results[str(path.relative_to(package))] = read(path)
        for path in package.glob("**/failure.json"):
            results[str(path.relative_to(package))] = read(path)
        value[package.name] = results
    return value
