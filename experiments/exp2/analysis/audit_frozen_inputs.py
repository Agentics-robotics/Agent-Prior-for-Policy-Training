"""Read-only data and static-layout audit for the predeclared five-task study."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

from appl.io import ROOT, read, digest, atomic, object_hash, source_manifest
from appl.envs.scene import CABINET_BOXES, DRAWER_BOXES, DRAWER_ORIGIN
from appl.scaleup.protocol import BASE, NAMES, config, initial_layout


def main():
    frozen = read(BASE / 'study_freeze.json')['study']
    assert object_hash(source_manifest()) == frozen['framework_source']
    files = {}
    layouts = []

    def expect(path, sha):
        path = str(Path(path).resolve())
        if path in files:
            assert files[path] == sha
        files[path] = sha

    for name in NAMES:
        cfg = config(name)
        record = frozen['tasks'][name]
        manifest_path = cfg['dataset'] / 'manifest.json'
        expect(manifest_path, record['dataset_manifest_sha256'])
        expect(cfg['_path'], record['configuration_sha256'])
        expect(cfg['completion_contract'], record['completion_contract_sha256'])
        expect(cfg['output'] / 'normalization.json', record['normalization_sha256'])
        manifest = read(manifest_path)
        for relative, sha in manifest['files'].items():
            expect(cfg['dataset'] / relative, sha)
        for source in manifest['sources'].values():
            original = Path(source['path'])
            expect(original, source['sha256'])
            for relative, sha in source['assets'].items():
                expect(original.parent / relative, sha)
        spec = read(BASE / name / 'task.json')
        expect(BASE / name / 'task.json', record['task_spec_sha256'])
        fixtures = [(np.array([0, 0, -.035]), np.array([.95, .65, .035]))]
        if name == 'drawer_exchange':
            nominal = dict(red=(np.array(DRAWER_ORIGIN) + [-.085, -.065, .028]), blue=np.array([-.4, .3, .02]))
            widths = dict(red=.012, blue=.015)
            fixtures += [(np.array(p), np.array(h)) for p, h in CABINET_BOXES]
            fixtures += [(np.array(p) + DRAWER_ORIGIN, np.array(h)) for p, h in DRAWER_BOXES]
        else:
            nominal = {c: np.array(spec['initial'][c]) for c in ('red', 'blue')}
            widths = dict(red=spec['id_half_width_m'], blue=spec['id_half_width_m'])
            fixtures += [(np.array(box['position']), np.array(box['half'])) for box in spec['fixtures']]
            full = BASE / name / 'naive_DP/full_trajectories'
            for relative, sha in read(full / 'manifest.json')['files'].items():
                expect(full / relative, sha)
        for condition, plans in record['layouts'].items():
            for plan in plans:
                positions = plan['layout']
                assert positions == initial_layout(name, plan['seed'], condition)
                offsets = {}
                for color in ('red', 'blue'):
                    point = np.array(positions[color])
                    delta = point - nominal[color]
                    assert abs(delta[2]) < 1e-10
                    maximum = float(np.abs(delta[:2]).max())
                    if condition == 'ID':
                        assert maximum <= widths[color] + 1e-10
                    else:
                        assert .022 - 1e-10 <= maximum <= .04 + 1e-10
                        assert maximum > widths[color]
                    offsets[color] = delta.tolist()
                    for center, half in fixtures:
                        overlap = .02 + half - abs(point - center)
                        assert not np.all(overlap > 1e-7), (name, condition, plan['seed'], color, center)
                overlap = .04 - abs(np.array(positions['red']) - positions['blue'])
                assert not np.all(overlap > 1e-7), (name, condition, plan['seed'], 'object_object')
                if name == 'unstack_sort':
                    assert np.allclose(offsets['red'], offsets['blue'], atol=1e-12, rtol=0)
                layouts.append(dict(task=name, condition=condition, seed=plan['seed'], offsets=offsets,
                    fixed_CAD_no_volume_intersection=True))
    original_data = ROOT / 'data/exp2/demonstrations_v2'
    for row in read(ROOT / 'runs/exp2/m0/training/request.json')['data']:
        expect(original_data / (row['id'] + '.json'), row['source_sha256'])

    def check(item):
        name, sha = item
        path = Path(name)
        return name if not path.is_file() or digest(path) != sha else None

    with ThreadPoolExecutor(max_workers=4) as pool:
        changed = [name for name in pool.map(check, files.items()) if name]
    target = ROOT / 'experiments/exp2/analysis/frozen_inputs_audit.json'
    atomic(target, dict(reviewed_utc=datetime.now(timezone.utc).isoformat(), files=len(files),
        expected_manifest_sha256=object_hash(files), changed=changed, passed=not changed,
        declared_layouts=len(layouts), layouts=layouts,
        physical_steps=0, optimizer_updates=0, api_requests=0,
        scope='Original demonstrations and image assets, API-published segments, naive full-trajectory copies, configurations and all 300 paired initial layouts.',
        geometry_limit='Static axis-aligned cube/fixture overlap checks only, allowing supporting contact; not a planning, reachable-grasp or success claim. Stacked blocks retain a common XY offset.'))
    assert not changed, changed
    print(dict(passed=True, files=len(files), paired_layouts=len(layouts), output=str(target)), flush=True)


if __name__ == '__main__':
    main()
