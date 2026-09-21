"""Precisely inventory and relocate legacy experiments; remove unfinished Round 4."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .data import DEFAULT_ROOT, REPO_ROOT, file_hash, freeze_json, object_hash, read_json


def _now():
    return datetime.now(timezone.utc).isoformat()


def process_audit(repo: Path) -> dict:
    """Report only identity and membership, never command-line credentials."""
    listing = subprocess.run(["ps", "-u", str(os.getuid()), "-o", "pid=,comm="],
                             check=True, capture_output=True, text=True).stdout
    processes = []
    for line in listing.splitlines():
        pid_text, program = line.strip().split(maxsplit=1)
        if not (program.startswith("python") or program == "pixi"):
            continue
        proc = Path("/proc") / pid_text
        if not proc.exists():
            continue
        arguments = (proc / "cmdline").read_bytes().split(b"\0")
        cwd = Path(os.readlink(proc / "cwd"))
        project = cwd == repo or cwd.is_relative_to(repo)
        round4 = any(b"round4" in argument.lower() for argument in arguments[1:])
        processes.append({"pid": int(pid_text), "program": program, "project_cwd": project,
                          "round4_argument": round4, "confirmed_project_round4": project and round4})
    confirmed = [p["pid"] for p in processes if p["confirmed_project_round4"]]
    return {"time": _now(), "processes": processes, "confirmed_project_round4_pids": confirmed,
            "credentials_or_full_arguments_recorded": False}


def targets(repo: Path) -> list[dict]:
    """A finite mapping of observed experiment ownership; no name-wide deletion."""
    selected = {}

    def add(relative: str | Path, action: str, round_number: int):
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Migration target must be a repository-relative path")
        if relative in selected:
            raise ValueError(f"Duplicate migration target: {relative}")
        selected[relative] = {"source": str(relative), "action": action, "round": round_number,
                              "destination": str(Path("archive/legacy_rounds") / f"round{round_number}" / "repository" / relative) if action == "archive" else None}

    for number in (2, 3):
        for prefix in ("src", "tests", "docs", "configs", "artifacts", "data", "scripts", "results"):
            path = Path(prefix) / f"round{number}"
            if (repo / path).exists():
                add(path, "archive", number)
    if (repo / "round3").exists():
        add("round3", "archive", 3)
    for number in (1, 2, 3):
        for path in sorted(repo.glob(f"ROUND{number}_*")):
            if path.is_file():
                add(path.relative_to(repo), "archive", number)
    for path in sorted((repo / "scripts").glob("*.py")):
        number = 3 if "round3" in path.name else 1
        add(path.relative_to(repo), "archive", number)
    for name in ("DATA_SEMANTICS.md", "RUNTIME_RECOVERY.md"):
        if (repo / "docs" / name).exists():
            add(Path("docs") / name, "archive", 1)
    # These root trees are Round 1. Their specifically owned Round 2/3
    # children above move first; then remaining immediate items retain paths.
    for dirname in ("artifacts", "data", "results"):
        for path in sorted((repo / dirname).iterdir()):
            if path.relative_to(repo) not in selected:
                add(path.relative_to(repo), "archive", 1)
    for relative in ("round4", "src/round4", "tests/round4", "ROUND4_SPEC.md", "ROUND4_REPORT.md"):
        if (repo / relative).exists():
            add(relative, "delete", 4)
    return list(selected.values())


def _describe(repo: Path, relative: str) -> dict:
    path = repo / relative
    if path.is_symlink() or not path.resolve().is_relative_to(repo):
        raise ValueError(f"Refusing symlink/external migration target: {relative}")
    if not path.exists():
        raise ValueError(f"Missing migration source during inventory: {relative}")
    files = []
    entries = [path] if path.is_file() else sorted(path.rglob("*"))
    for child in entries:
        if child.is_symlink():
            raise ValueError(f"Migration tree contains a symlink: {child.relative_to(repo)}")
        if child.is_file():
            files.append({"path": str(child.relative_to(repo)), "bytes": child.stat().st_size,
                          "sha256": file_hash(child), "checkpoint": child.suffix in (".pt", ".pth", ".ckpt")})
    return {"type": "file" if path.is_file() else "directory", "bytes": sum(f["bytes"] for f in files),
            "files": files, "file_count": len(files), "tree_hash": object_hash(files),
            "checkpoint_files": [f["path"] for f in files if f["checkpoint"]]}


def _append(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def inventory(repo: Path = REPO_ROOT, output: Path = DEFAULT_ROOT) -> dict:
    repo, output = Path(repo).resolve(), Path(output).resolve()
    path = output / "migration_inventory.json"
    if path.exists():
        return read_json(path)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                              capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=repo,
                            check=True, capture_output=True, text=True).stdout.splitlines()
    audit = process_audit(repo)
    value = {"schema_version": "experiment1.migration_inventory.v1", "time": _now(), "repo_root": str(repo),
             "revision": revision, "preexisting_worktree_paths": [line[3:] for line in status],
             "process_audit": audit, "targets": [{**target, **_describe(repo, target["source"])} for target in targets(repo)],
             "preserved_shared_paths": ["src/relative_dp", "src/experiment1", "src/experiment_interfaces", "tests/interfaces",
                                        ".pixi", "pixi.toml", "pixi.lock", "outputs/interfaces", "examples/interfaces"],
             "new_shared_environment": {str(p.relative_to(repo)): file_hash(p) for p in sorted((repo / "src/experiment1/metaworld").glob("*.py"))},
             "missing_checkpoint_disclosure": "Inventory records only existing checkpoints. Missing/deleted legacy weights are not recreated and archives are not represented as fully loadable."}
    value["hash"] = object_hash(value)
    freeze_json(path, value)
    return value


def execute(repo: Path = REPO_ROOT, output: Path = DEFAULT_ROOT) -> dict:
    repo, output = Path(repo).resolve(), Path(output).resolve()
    value = inventory(repo, output)
    if value["repo_root"] != str(repo):
        raise ValueError("Inventory belongs to another repository")
    audit = process_audit(repo)
    _append(output / "migration_log.jsonl", {"time": _now(), "action": "process_audit", "audit": audit})
    if audit["confirmed_project_round4_pids"]:
        raise RuntimeError("Confirmed project Round 4 jobs must stop before their files can be deleted")
    completed = {}
    started_deletions = set()
    log_path = output / "migration_log.jsonl"
    for line in log_path.read_text().splitlines():
        record = json.loads(line)
        if record.get("status") == "completed":
            completed[record["source"]] = record
        if record.get("status") == "started" and record["action"] == "delete":
            started_deletions.add(record["source"])
    mapping = []
    mapping_path = repo / "archive/legacy_rounds/path_mapping.json"
    relocations = read_json(mapping_path).get("post_archive_relocations", []) if mapping_path.exists() else []
    for target in value["targets"]:
        source = repo / target["source"]
        destination = repo / target["destination"] if target["destination"] else None
        for correction in relocations:
            if target["destination"] == correction["source"]:
                destination = repo / correction["destination"]
        if source.is_symlink() or not source.resolve().is_relative_to(repo):
            raise ValueError("Source escaped the frozen repository target")
        if target["source"] in completed:
            if source.exists():
                raise ValueError("Completed migration source was recreated; refusing to delete new work")
            if destination is not None and not destination.exists():
                raise ValueError("Previously archived destination is missing")
            mapping.append({k: target[k] for k in ("source", "destination", "action", "round")})
            continue
        if not source.exists():
            # An interrupted atomic rename is recognized by all frozen hashes.
            if destination is None and target["source"] in started_deletions:
                pass  # The logged exact deletion finished before its receipt.
            elif destination is None or not destination.exists():
                raise ValueError("Unrecorded migration source is missing")
            else:
                _verify_destination(repo, target)
        else:
            actual = _describe(repo, target["source"])
            if actual["tree_hash"] != target["tree_hash"]:
                expected_files = {record["path"]: record for record in target["files"]}
                remaining_original_files = all(record == expected_files.get(record["path"]) for record in actual["files"])
                if target["source"] not in started_deletions or not remaining_original_files:
                    raise ValueError(f"Source changed since inventory: {target['source']}")
            _append(log_path, {"time": _now(), "status": "started", "source": target["source"],
                               "action": target["action"], "destination": target["destination"], "tree_hash": target["tree_hash"]})
            if destination is not None:
                if destination.exists():
                    raise ValueError("Archive destination already exists; never overwrite")
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.rename(destination)
                _verify_destination(repo, target)
            elif source.is_dir():
                shutil.rmtree(source)
            else:
                source.unlink()
        _append(log_path, {"time": _now(), "status": "completed", "source": target["source"],
                           "action": target["action"], "destination": target["destination"],
                           "bytes": target["bytes"], "tree_hash": target["tree_hash"], "file_count": target["file_count"]})
        mapping.append({k: target[k] for k in ("source", "destination", "action", "round")})
    result = {"schema_version": "experiment1.archive_mapping.v1", "inventory_hash": value["hash"], "paths": mapping,
              "existing_checkpoints": [p for t in value["targets"] if t["action"] == "archive" for p in t["checkpoint_files"]],
              "missing_checkpoints": "Historical deletion receipts and original run manifests are retained; absent weights remain absent.",
              "shared_support_root": "archive/legacy_rounds/round3/repository/round3/data"}
    if relocations:
        result["post_archive_relocations"] = relocations
    freeze_json(repo / "archive/legacy_rounds/path_mapping.json", result)
    readme = """# Legacy experiment archive\n\nRound 1, 2 and 3 retain their original repository-relative paths under `roundN/repository/`. Existing checkpoints, reports, configuration, data, logs and historical source were moved, not copied. `path_mapping.json` maps all targets. The full original file hashes and sizes are in `experiments/experiment1/migration_inventory.json`; actual actions are in `migration_log.jsonl`.\n\nSome legacy checkpoints were already absent or deleted. Only the inventoried existing files are preserved; this archive does not claim that every historical run is loadable. Historical deletion receipts and retention records remain with Round 3. Shared `src/relative_dp` remains in the active source tree because Experiment 1 reuses its frozen DP implementation.\n\nExperiment 1 references only original successful Round 3 support episodes in `round3/repository/round3/data/<task>/train20/`; historical candidates, scores and conversations are excluded from RuntimePriorAPI evidence. Unfinished Round 4 was deleted according to the exact inventory; it is not represented as a completed experiment or renamed as Experiment 1.\n"""
    path = repo / "archive/legacy_rounds/README.md"
    if path.exists() and path.read_text() != readme:
        raise ValueError("Existing archive README differs")
    if not path.exists():
        path.write_text(readme)
    return result


def correct_round2_results(repo: Path = REPO_ROOT, output: Path = DEFAULT_ROOT) -> dict:
    """Append an ownership correction without changing the original inventory."""
    from .records import atomic_json
    repo, output = Path(repo).resolve(), Path(output).resolve()
    source_name = "archive/legacy_rounds/round1/repository/results/round2"
    destination_name = "archive/legacy_rounds/round2/repository/results/round2"
    mapping_path = repo / "archive/legacy_rounds/path_mapping.json"
    mapping = read_json(mapping_path)
    previous = [r for r in mapping.get("post_archive_relocations", []) if r["source"] == source_name]
    if previous:
        if (repo / source_name).exists() or not (repo / destination_name).exists():
            raise ValueError("Completed archive ownership correction changed")
        return previous[0]
    source, destination = repo / source_name, repo / destination_name
    if destination.exists():
        raise ValueError("Ownership correction destination already exists")
    description = _describe(repo, source_name)
    correction = {"time": _now(), "source": source_name, "destination": destination_name,
                  "reason": "Original results/round2 is Round 2-specific; initial archive placed this nested directory under Round 1.",
                  "tree_hash": description["tree_hash"], "bytes": description["bytes"],
                  "file_count": description["file_count"], "original_inventory_unchanged": True}
    _append(output / "migration_log.jsonl", {**correction, "action": "archive_ownership_correction", "status": "started"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.rename(destination)
    _verify_destination(repo, dict(source=source_name, destination=destination_name, **description))
    mapping.setdefault("post_archive_relocations", []).append(correction)
    atomic_json(mapping_path, mapping)
    _append(output / "migration_log.jsonl", {**correction, "action": "archive_ownership_correction", "status": "completed"})
    return correction


def _verify_destination(repo: Path, target: dict):
    destination = repo / target["destination"]
    source_prefix = Path(target["source"])
    for record in target["files"]:
        relative = Path(record["path"]).relative_to(source_prefix)
        path = destination / relative if target["type"] == "directory" else destination
        if path.is_symlink() or file_hash(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError("Archived file hash differs from inventory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    result = inventory() if args.inventory_only else execute()
    print(json.dumps({"action": "inventory" if args.inventory_only else "execute", "records": len(result.get("targets", result.get("paths", [])))}))
