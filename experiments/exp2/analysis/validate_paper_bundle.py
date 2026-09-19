"""Validate the updated paper bundle and its single authorized reporting change."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
from urllib.parse import unquote
import zipfile

from appl.io import ROOT, read, digest, atomic


def main():
    base = ROOT / 'experiments/exp2'
    paper = base / 'paper'
    receipt = read(paper / 'completion.json')
    assert receipt['status'] == 'complete'
    for name, sha in receipt['files'].items():
        assert digest(paper / name) == sha, name
    expected = set(receipt['files']) | {'completion.json'}
    archive = base / 'Exp2_paper_bundle.zip'
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(set(z.namelist()))
        assert set(z.namelist()) == {'Exp2_paper/' + p for p in expected}
        for p in expected:
            assert hashlib.sha256(z.read('Exp2_paper/' + p)).hexdigest() == digest(paper / p), p
    links = 0
    for doc in list(paper.glob('*.md')) + [paper / 'recovery_20260919/REPORT.md']:
        for target in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
            target = target.strip('<>').split('#')[0]
            if not target or '://' in target or target.startswith('mailto:'):
                continue
            assert (doc.parent / unquote(target)).exists(), (doc.name, target)
            links += 1
    examples = read(paper / 'execution_examples/index.json')['examples']
    assert len(examples) == 40
    for row in examples:
        for filename, item in row['files'].items():
            assert digest(ROOT / item['source']) == item['sha256']
            assert digest(paper / row['bundle_folder'] / filename) == item['sha256']
    recovery = read(paper / 'recovery_20260919/completion.json')
    for item in read(paper / 'recovery_20260919/index.json')['files']:
        assert digest(item['source']) == item['sha256']
        assert digest(paper / 'recovery_20260919' / item['bundle_file']) == item['sha256']
    original = read(ROOT / 'runs/exp2/single_policy_astra_xhigh/results.json')
    current = read(paper / 'results.json')
    differences = [(a, b) for a, b in zip(original['episodes'], current['episodes']) if a != b]
    assert len(differences) == 1
    previous, new = differences[0]
    assert not previous['completed'] and new == recovery['replacement']
    assert len(current['episodes']) == 1200 and all(r['completed'] for r in current['episodes'])
    assert len(current['groups']) == 40 and all(g['completed'] == 30 and g['unknown'] == 0 for g in current['groups'])
    attempts = read(paper / 'evaluation_attempts.json')
    assert attempts['recorded_reset_attempts'] == 1292 and attempts['interrupted_attempts'] == 92
    assert sum(r['selected_for_main_matrix'] for r in attempts['attempts']) == 1200
    old_paths = read(ROOT / 'runs/exp2/M1_scaleup/evaluation_recovery_20260919/original_artifacts.json')
    for p, sha in old_paths.items():
        assert digest(p) == sha, p
    value = dict(passed=True, reviewed_utc=datetime.now(timezone.utc).isoformat(),
        planned_cells=1200, complete_outcomes=1200, retained_unknown=0,
        retained_reset_attempts=1292, retained_interrupted_attempts=92,
        authorized_replacement_cells=1, original_results_match=True,
        original_results_scope='Frozen canonical reports and all original attempt files unchanged; exactly one previously unknown paper cell replaced by its authorized reset.',
        all_bundle_hashes_match=True, zip_integrity='passed', bundle_files=len(expected),
        checked_primary_document_links=links, portable_examples=40, additional_recovery_videos=1,
        recovery_success=new['success'], recovery_steps=new['steps'],
        additional_recovery_API_calls=recovery['API_accounting']['states']['consumed'],
        no_added_training=True, report_build_added_API_calls=0, report_build_added_simulator_steps=0,
        archive_bytes=archive.stat().st_size, archive_sha256=digest(archive),
        completion_sha256=digest(paper / 'completion.json'))
    atomic(base / 'paper_bundle_validation.json', value)
    print(value, flush=True)


if __name__ == '__main__':
    main()
