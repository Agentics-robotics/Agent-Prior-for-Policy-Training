"""One recorded, user-authorized layout migration; never runs an experiment.

prepare writes an exact inventory. apply consumes it, preserves original paths
as compatibility links, and deletes only the listed redundant writing exports.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / 'archive/Exp2_history_20260920'
RECEIPTS = ARCHIVE / 'migration'
RUNS = ROOT / 'runs/exp2'
EXTERNAL = Path('/home/storage/oscar/appl_exp2/legacy/Exp2_history_20260920/runs')
EXP = ROOT / 'experiments/exp2'
EXPORT = EXP / 'Exp2_new_paper'
CURRENT = RUNS / 'current'


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def inventory(base):
    """Do not traverse nested links; retain exact regular-file identity."""
    rows = []
    for directory, dirs, files in os.walk(base):
        for name in sorted(dirs + files):
            p = Path(directory) / name
            if p.is_symlink():
                rows.append(dict(path=str(p), link=os.readlink(p)))
            elif p.is_file():
                s = p.stat()
                rows.append(dict(path=str(p), size=s.st_size, mtime_ns=s.st_mtime_ns,
                                 inode=s.st_ino, device=s.st_dev, mode=s.st_mode))
    return rows


def move(src, dst, alias=False):
    src, dst = Path(src), Path(dst)
    assert src.exists() or src.is_symlink(), src
    assert not dst.exists() and not dst.is_symlink(), dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    assert src.lstat().st_dev == dst.parent.stat().st_dev, 'Only same-filesystem renames are allowed'
    src.rename(dst)
    if alias:
        try:
            src.symlink_to(dst)
        except BaseException:
            dst.rename(src)
            raise
    with (RECEIPTS/'moves.jsonl').open('a') as f:
        f.write(json.dumps(dict(source=str(src),destination=str(dst),alias=alias))+'\n')
        f.flush(); os.fsync(f.fileno())


def prepare():
    assert not RECEIPTS.exists(), 'Migration already prepared'
    RECEIPTS.mkdir(parents=True)
    results = read(EXPORT / 'results.json')
    episodes = [r for r in results if r['condition'] == 'OOD']
    assert len(episodes) == 225
    selected = list(csv.DictReader((RUNS/'Exp2_new/ood_resource_resume_20260920/episodes.csv').open()))
    assert {str(Path(r['source_root']).resolve()) for r in episodes if r['method']=='APPL_6_xhigh'} == {str(Path(r['root']).resolve()) for r in selected}
    baseline = read(RUNS/'Exp2_new/baseline_subset.json')
    assert {str(Path(r['source_root']).resolve()) for r in episodes if r['method']!='APPL_6_xhigh'} == {str(Path(r['root']).resolve()) for r in baseline if r['condition']=='OOD'}
    for r in episodes:
        p = Path(r['source_root'])
        for name, key in [('result.json','result_sha256'), ('initial_state.json','initial_sha256'), ('trace.jsonl','trace_sha256')]:
            assert sha(p/name) == r[key], p/name
        r['original_root'] = r.pop('source_root')
        r['root'] = str(Path('runs/exp2/current') / r.pop('bundle_root'))
        r['video'] = r['root'] + '/replay.mp4'
    models = read(EXPORT/'models.json')
    assert len(models) == 46
    for r in models:
        assert sha(ROOT/r['checkpoint']) == r['checkpoint_sha256'], r['checkpoint']
        old = r['folder']
        new = str(Path('runs/exp2/current/models') / r['method'] / r['task'] / r['policy_id'])
        r['original_folder'] = old
        for k, v in list(r.items()):
            if k != 'original_folder' and isinstance(v, str) and (v == old or v.startswith(old+'/')):
                r[k] = new + v[len(old):]
        r.pop('bundle_folder', None)
        r.pop('checkpoint_included', None)
    frozen = []
    for base in ['src/appl','src/experiment1','src/relative_dp','src/experiment_interfaces',
                 'experiments/exp2/exp2_new','experiments/exp2/binary_gripper','experiments/exp2/single_policy']:
        frozen += [p for p in (ROOT/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    frozen += [ROOT/p for p in ['pixi.toml','pixi.lock','pyproject.toml','environments/exp2/pixi.toml',
                 'environments/exp2/pixi.lock','experiments/exp2/EXP2_NEW.md','experiments/exp2/SINGLE_POLICY.md',
                 'experiments/experiment1/EXPERIMENT1_REPORT.md','experiments/experiment1/protocol.lock.json']]
    write(RECEIPTS/'frozen_sha256.json', {str(p.relative_to(ROOT)):sha(p) for p in frozen})
    protected = []
    for base in ['experiments/experiment1','src/experiment1','src/relative_dp','src/experiment_interfaces','archive/legacy_rounds']:
        protected += inventory(ROOT/base)
    write(RECEIPTS/'exp1_before.json', protected)
    write(RECEIPTS/'runtime_before.json', inventory(RUNS))
    write(RECEIPTS/'data_before.json', inventory(ROOT/'data/exp2'))
    for p in [ROOT/'README.md', ROOT/'PROGRESS.md', EXP/'AGENTS.md', EXP/'README.md']:
        dest=ARCHIVE/'navigation_before'/p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p,dest)
    deletions = [EXP/n for n in ['Exp2_new_paper','Exp2_new_writing','Exp2_new_paper_bundle.zip',
        'Exp2_new_paper_bundle.zip.sha256','Exp2_new_paper_bundle.validation.json',
        'Exp2_new_writing_bundle.zip','Exp2_new_writing_bundle.zip.sha256','Exp2_new_writing_bundle.validation.json',
        'Exp2_paper_bundle.zip','paper_bundle_validation.json',
        'analysis/build_exp2_new_paper.py','analysis/build_exp2_new_writing_companion.py','analysis/validate_paper_bundle.py']]
    deletions += [ROOT/'archive/Exp2_paper_before_20260919_recovery'/n for n in ['Exp2_paper_bundle.zip','paper_bundle_validation.json']]
    deletion_files = []
    for p in deletions:
        assert p.exists(), p
        deletion_files += inventory(p) if p.is_dir() else [dict(path=str(p),size=p.stat().st_size,sha256=sha(p))]
    write(RECEIPTS/'deletion_inventory.json',deletion_files)
    keep_analysis = {'audit_scaleup.py','binary_gripper_completion.py','exp2_new_completion.py',
        'exp2_new_ood_completion.py','exp2_new_ood_continue.py','exp2_new_ood_final.py','exp2_new_ood_resume.py','__init__.py'}
    local_moves=[]
    for p in sorted((EXP/'analysis').iterdir()):
        if p.name not in keep_analysis and p not in deletions:
            local_moves.append(dict(source=str(p),destination=str(ARCHIVE/'experiments/analysis'/p.name),alias=False))
    for p in sorted((EXP/'astra').glob('*.py')):
        if p.name not in {'prepare.py','runner.py','report.py','transport.py','__init__.py'}:
            local_moves.append(dict(source=str(p),destination=str(ARCHIVE/'experiments/astra'/p.name),alias=False))
    for name in ['PROTOCOL.md','PRIOR_POLICIES.md','INFERENCE_5000.md','REPORT.md','M0_REPORT.md',
                 'M0_LAYOUT.json','M1_INITIAL_LAYOUT.json','incidents.json','migration_sources.json']:
        p=EXP/name
        local_moves.append(dict(source=str(p),destination=str(ARCHIVE/'experiments'/name),alias=True))
    for name in ['m1_v1.json','m1_v2.json','m1_v2_3000.json','m1_v2_5000.json','prior_policies.json','main.json']:
        p=EXP/'configs'/name
        # Older initial studies already have valid archive aliases.
        if not p.is_symlink():
            local_moves.append(dict(source=str(p),destination=str(ARCHIVE/'experiments/configs'/name),alias=True))
    for rel in ['demonstrations','processed'] + ['scaleup/'+task+'/processed' for task in ['two_block_sort','buffer_swap','unstack_sort','tray_pack']]:
        p=ROOT/'data/exp2'/rel
        if p.exists() and not p.is_symlink():
            local_moves.append(dict(source=str(p),destination=str(ARCHIVE/'data'/rel),alias=True))
    write(RECEIPTS/'plan.json',dict(episodes=episodes,models=models,tasks=read(EXPORT/'tasks.json'),
        runtime_tops=[str(p) for p in sorted(RUNS.iterdir()) if not p.is_symlink()],
        local_moves=local_moves,deletions=list(map(str,deletions)),
        preserved_reports=['REPORT.md','FAILURE_ANALYSIS.md','RECOVERY_CASES.md','PRIOR_CATALOG.md'],
        authorization='User 2026-09-20: archive obsolete Exp2, keep 225 selected OOD outcomes; delete writing ZIPs/exports, preserve reports; Exp1 untouched.'))
    print(json.dumps(dict(prepared=True,episodes=225,models=46,local_moves=len(local_moves),
        deletion_files=len(deletion_files),deletion_bytes=sum(r['size'] for r in deletion_files if 'size' in r))))


def preserve_reports(plan):
    reports = EXP/'reports'
    reports.mkdir()
    # Retain all scientific report text/tables/figures. Export-specific navigation,
    # duplicate source, copied trajectories and ZIP mechanics are not retained.
    for name in plan['preserved_reports'] + ['tables','analysis','figures','models.json','tasks.json','results.json',
                                          'contracts','task_definitions','normalization']:
        src=EXPORT/name; dst=reports/name
        if src.is_dir(): shutil.copytree(src,dst)
        else: shutil.copy2(src,dst)
    write(RECEIPTS/'report_original_hashes.json',{
        str(p.relative_to(reports)):sha(p) for p in reports.rglob('*') if p.is_file()})
    # Preserve original versions of all six narrative documents from both exports.
    for label in ['Exp2_new_paper','Exp2_new_writing']:
        for p in (EXP/label).glob('*.md'):
            dst=ARCHIVE/'reports/export_narratives'/label/p.name
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    shutil.copy2(EXPORT/'provenance/copied_files.json',RECEIPTS/'export_original_sources.json')
    # Earlier report revisions may only exist in the old ZIP. Preserve their
    # actual report assets, excluding duplicate framework and packaging scripts.
    oldzip=ROOT/'archive/Exp2_paper_before_20260919_recovery/Exp2_paper_bundle.zip'
    with zipfile.ZipFile(oldzip) as z:
        for n in z.namelist():
            rel=Path(n).relative_to('Exp2_paper')
            if '..' in rel.parts or n.endswith('/') or rel.suffix in {'.py','.pyc'}: continue
            dst=ARCHIVE/'reports/before_recovery'/rel
            dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(z.read(n))
    # The earlier four-method report, its figures and evidence remain available.
    move(EXP/'paper',ARCHIVE/'reports/previous_four_method',alias=True)


def apply():
    plan=read(RECEIPTS/'plan.json')
    assert not CURRENT.exists() and not EXTERNAL.exists()
    # Preflight every source before the first move. Promote current assets while
    # their original parents still exist; archive those parents only afterwards.
    for row in plan['episodes']:
        assert Path(row['original_root']).is_dir()
        assert not (ROOT/row['root']).exists()
    for row in plan['models']:
        assert (ROOT/row['original_folder']).is_dir()
        assert not (ROOT/row['folder']).exists()
    for row in plan['local_moves']:
        assert Path(row['source']).exists()
        assert not Path(row['destination']).exists()
    for src in plan['runtime_tops']:
        assert Path(src).exists() and not Path(src).is_symlink()
    assert RUNS.stat().st_dev == EXTERNAL.parent.parent.stat().st_dev
    preserve_reports(plan)
    CURRENT.mkdir()
    # Only complete selected outcomes are promoted. Partial ID and interrupted
    # prefixes remain in the immutable history, never counted in this manifest.
    for row in plan['episodes']:
        src=Path(row['original_root']).resolve()
        move(src,ROOT/row['root'],alias=True)
    for row in plan['models']:
        src=(ROOT/row['original_folder']).resolve();dst=ROOT/row['folder']
        if row['task']=='drawer_exchange' and row['method']=='naive_DP':
            dst.mkdir(parents=True)
            for name in ['training','checks','configuration.json','check_device.json','train_device.json','source_versions']:
                move(src/name,dst/name,alias=True)
        else:
            move(src,dst,alias=True)
    for task in plan['tasks']:
        task=task['task']
        for name in ['normalization.json','segmentation','task.json','source_versions','inference_checks']:
            src=(RUNS/'M1_astra_xhigh'/task/name).resolve()
            move(src,CURRENT/'inputs/appl'/task/name,alias=True)
        src=(RUNS/'M1_scaleup'/task/'task.json').resolve()
        move(src,CURRENT/'inputs/tasks'/f'{task}.json',alias=True)
    # Selected leaves now exist at their canonical locations and original paths
    # are links. Parent archival can no longer invalidate promotion sources.
    EXTERNAL.mkdir(parents=True)
    (ARCHIVE/'runs').symlink_to(EXTERNAL)
    for src in plan['runtime_tops']:
        move(src,EXTERNAL/Path(src).name,alias=True)
    for row in plan['local_moves']:
        move(row['source'],row['destination'],alias=row['alias'])
    write(CURRENT/'manifest.json',dict(name='Exp2_new',review_date='2026-09-20',
        scope='Five tasks x 15 matched position-OOD initial states x three methods',
        episodes=plan['episodes'],models=plan['models'],tasks=plan['tasks'],
        history='archive/Exp2_history_20260920',
        historical_paths='Compatibility links retain frozen requests/configurations and all original reports.'))
    write(RECEIPTS/'applied.json',dict(applied=True,episodes=225,models=46,
        runtime_archive=str(EXTERNAL),source_behavior_changed=False,API_calls=0,training_updates=0,
        writing_exports_deleted=False))
    print('Reversible migration applied. No writing exports deleted; validate before prune.')


def prune():
    plan=read(RECEIPTS/'plan.json')
    assert read(RECEIPTS/'validation.json')['passed'], 'Validate the migration before deleting duplicates'
    for name in plan['preserved_reports']:
        assert (EXP/'reports'/name).is_file()
        original=ARCHIVE/'reports/export_narratives/Exp2_new_paper'/name
        assert sha(original)==sha(EXPORT/name)
    # Deletion is limited to the prepared exact inventory, after validation.
    for name in plan['deletions']:
        p=Path(name)
        if p.is_dir(): shutil.rmtree(p)
        else: p.unlink()
    packaging=[]
    for p in (ARCHIVE/'reports/previous_four_method').glob('*.py'):
        packaging.append(dict(path=str(p),sha256=sha(p),size=p.stat().st_size));p.unlink()
    write(RECEIPTS/'removed_old_report_builders.json',packaging)
    write(RECEIPTS/'pruned.json',dict(deleted=True,validated_first=True,report_originals_preserved=True))
    print('Listed duplicate writing exports deleted; all scientific reports retained.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','apply','prune'])
    args=parser.parse_args()
    dict(prepare=prepare,apply=apply,prune=prune)[args.phase]()
