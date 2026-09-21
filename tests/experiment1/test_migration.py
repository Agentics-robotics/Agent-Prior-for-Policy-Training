"""Temporary repositories exercise destructive-operation boundaries and resume."""
from pathlib import Path

import pytest

from experiment1 import migration
from experiment1.data import freeze_json, object_hash


def fixture_inventory(repo):
    for relative, content in (("round3/data/support.npz", b"original shared support"),
                              ("round3/checkpoints/run.pt", b"existing legacy weight"),
                              ("round4/uncompleted.json", b"unfinished"),
                              ("src/relative_dp/model.py", b"shared diffusion implementation"),
                              ("unrelated/keep.txt", b"unrelated")):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    targets = []
    for source, action, number in (("round3", "archive", 3), ("round4", "delete", 4)):
        target = {"source": source, "action": action, "round": number,
                  "destination": f"archive/legacy_rounds/round{number}/repository/{source}" if action == "archive" else None}
        targets.append({**target, **migration._describe(repo, source)})
    inventory = {"repo_root": str(repo), "targets": targets}
    inventory["hash"] = object_hash(inventory)
    out = repo / "experiments/experiment1"
    freeze_json(out / "migration_inventory.json", inventory)
    return out


def test_exact_migration_retains_hashes_and_is_idempotent(tmp_path, monkeypatch):
    out = fixture_inventory(tmp_path)
    monkeypatch.setattr(migration, "process_audit", lambda repo: {"confirmed_project_round4_pids": []})
    first = migration.execute(tmp_path, out)
    assert not (tmp_path / "round3").exists()
    assert not (tmp_path / "round4").exists()
    assert (tmp_path / "archive/legacy_rounds/round3/repository/round3/data/support.npz").read_bytes() == b"original shared support"
    assert (tmp_path / "src/relative_dp/model.py").read_bytes() == b"shared diffusion implementation"
    assert (tmp_path / "unrelated/keep.txt").read_bytes() == b"unrelated"
    assert migration.execute(tmp_path, out) == first


def test_completed_target_recreated_by_new_work_is_never_deleted(tmp_path, monkeypatch):
    out = fixture_inventory(tmp_path)
    monkeypatch.setattr(migration, "process_audit", lambda repo: {"confirmed_project_round4_pids": []})
    migration.execute(tmp_path, out)
    (tmp_path / "round4").mkdir()
    (tmp_path / "round4/new.txt").write_text("new work")
    with pytest.raises(ValueError, match="recreated"):
        migration.execute(tmp_path, out)
    assert (tmp_path / "round4/new.txt").read_text() == "new work"


def test_source_changed_after_inventory_is_not_destroyed(tmp_path, monkeypatch):
    out = fixture_inventory(tmp_path)
    monkeypatch.setattr(migration, "process_audit", lambda repo: {"confirmed_project_round4_pids": []})
    (tmp_path / "round3/data/support.npz").write_bytes(b"new user data")
    with pytest.raises(ValueError, match="changed since inventory"):
        migration.execute(tmp_path, out)
    assert (tmp_path / "round3/data/support.npz").read_bytes() == b"new user data"


def test_archive_interruption_after_rename_is_recovered_by_original_hashes(tmp_path, monkeypatch):
    out = fixture_inventory(tmp_path)
    monkeypatch.setattr(migration, "process_audit", lambda repo: {"confirmed_project_round4_pids": []})
    destination = tmp_path / "archive/legacy_rounds/round3/repository/round3"
    destination.parent.mkdir(parents=True)
    (tmp_path / "round3").rename(destination)
    result = migration.execute(tmp_path, out)
    assert len(result["existing_checkpoints"]) == 1
    assert destination.exists()


def test_confirmed_project_round4_job_blocks_deletion(tmp_path, monkeypatch):
    out = fixture_inventory(tmp_path)
    monkeypatch.setattr(migration, "process_audit", lambda repo: {"confirmed_project_round4_pids": [123]})
    with pytest.raises(RuntimeError, match="must stop"):
        migration.execute(tmp_path, out)
    assert (tmp_path / "round4/uncompleted.json").exists()


def test_interrupted_exact_deletion_resumes_without_touching_other_paths(tmp_path, monkeypatch):
    out = fixture_inventory(tmp_path)
    monkeypatch.setattr(migration, "process_audit", lambda repo: {"confirmed_project_round4_pids": []})
    migration._append(out / "migration_log.jsonl", {"status": "started", "source": "round4", "action": "delete"})
    (tmp_path / "round4/uncompleted.json").unlink()
    migration.execute(tmp_path, out)
    assert not (tmp_path / "round4").exists()
    assert (tmp_path / "unrelated/keep.txt").exists()


def test_external_symlink_target_is_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "round4").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink/external"):
        migration._describe(repo, "round4")
