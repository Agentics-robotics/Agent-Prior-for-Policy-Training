"""Read the sole current Exp2 study; no API, training, or simulator entry point."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'runs/exp2/current/manifest.json'
METHODS = ('naive_DP', 'SinglePrior_6_xhigh', 'APPL_6_xhigh')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def summary(manifest):
    rows = manifest['episodes']
    return {task: {method: dict(successes=sum(r['success'] for r in rows
                if r['task'] == task and r['method'] == method),
                episodes=sum(r['task'] == task and r['method'] == method for r in rows))
            for method in METHODS} for task in [t['task'] for t in manifest['tasks']]}


def verify(manifest):
    rows, models = manifest['episodes'], manifest['models']
    assert len(rows) == 225 and len(models) == 46
    keys = [(r['task'], r['condition'], r['seed'], r['method']) for r in rows]
    assert len(set(keys)) == 225 and all(r['condition'] == 'OOD' for r in rows)
    expected = {(t['task'], 'OOD', seed, method) for t in manifest['tasks']
                for seed in t['selected_OOD_seeds'] for method in METHODS}
    assert set(keys) == expected
    assert Counter(m['method'] for m in models) == dict(naive_DP=5, SinglePrior_6_xhigh=5, APPL_6_xhigh=36)
    paired = defaultdict(set)
    for row in rows:
        p = ROOT / row['root']
        assert p.is_dir() and not p.is_symlink(), p
        for name, key in [('result.json', 'result_sha256'), ('initial_state.json', 'initial_sha256'), ('trace.jsonl', 'trace_sha256')]:
            assert digest(p / name) == row[key], p / name
        result = json.loads((p / 'result.json').read_text())
        assert result['success'] == row['success'] and result['steps'] == row['steps']
        assert json.loads((p / 'audit.json').read_text())['passed'], p
        assert (p / 'replay.mp4').stat().st_size > 0
        assert Path(row['original_root']).resolve() == p.resolve()
        paired[(row['task'], row['seed'])].add(row['initial_sha256'])
    assert len(paired) == 75 and all(len(h) == 1 for h in paired.values())
    for model in models:
        p = ROOT / model['folder']
        assert p.is_dir() and not p.is_symlink(), p
        assert digest(ROOT / model['checkpoint']) == model['checkpoint_sha256'], p
        if model['method'] != 'naive_DP':
            assert json.loads((p / 'authorship_audit.json').read_text())['passed'], p
            assert (p / 'source/PRIOR.md').is_file() and (p / 'submission.json').is_file()
            submission = json.loads((p / 'submission.json').read_text())
            for name, expected_hash in submission['files'].items():
                assert digest(p / 'source' / name) == expected_hash, p / 'source' / name
            assignment = submission['assignment']
            assert digest(Path(assignment['dataset'])) == assignment['dataset_sha256'], p
            normalizer = assignment['normalization']
            assert digest(Path(normalizer['path'])) == normalizer['sha256'], p
    return dict(passed=True, episodes=225, matched_layouts=75, models=46,
                original_result_trace_checkpoint_hashes=True, API_submissions_verified=41,
                recorded_episode_audits=True,
                experiment_execution=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Read and check original hashes, paired resets, models and episode receipts.')
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    result = dict(study=manifest['name'], scope=manifest['scope'], summary=summary(manifest))
    if args.verify:
        result['validation'] = verify(manifest)
    print(json.dumps(result, indent=2))
