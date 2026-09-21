"""Archived jobs must remain in the historical training cost accounting."""
import json

from appl.report import records


def test_report_includes_archived_jobs_through_original_paths(tmp_path):
    active=tmp_path/'runs';archive=tmp_path/'archive/D1'
    (active/'checks').mkdir(parents=True)
    (active/'diagnostics').mkdir()
    (archive/'training').mkdir(parents=True)
    (active/'checks/result.json').write_text('{"optimizer_steps": 8}')
    (archive/'training/result.json').write_text('{"optimizer_steps": 5000}')
    (active/'diagnostics/D1').symlink_to(archive,target_is_directory=True)
    paths=records(active,'result.json')
    assert [str(p.relative_to(active)) for p in paths]==[
        'checks/result.json','diagnostics/D1/training/result.json']
    assert sum(json.loads(p.read_text())['optimizer_steps'] for p in paths)==5008
