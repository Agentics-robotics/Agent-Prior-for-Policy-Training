"""Finalize evidence after all fixed queues exit; no new scientific attempts."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import subprocess
import time

from appl.io import ROOT, read, digest, atomic, source_manifest, object_hash
from appl.scaleup.protocol import BASE, NAMES, config, naive_paths
from appl.prior_policies.deploy import library
import audit_scaleup
import analyze_outcomes


def verify_video(root):
    provenance = read(root / 'replay_provenance.json')
    frames = [root / 'initial.png', *sorted(root.glob('frame_*.png'))]
    if (root / 'final.png').exists():
        frames.append(root / 'final.png')
    assert provenance['frames'] == {p.name: digest(p) for p in frames}, str(root)
    video = root / 'replay.mp4'
    assert provenance['video_sha256'] == digest(video)
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_frames,codec_name', '-of', 'json', str(video)],
        capture_output=True, text=True, check=True)
    stream = json.loads(result.stdout)['streams'][0]
    assert (stream['width'], stream['height'], stream['codec_name'], int(stream['nb_frames'])) == (128, 128, 'h264', len(frames))
    return dict(episode=str(root.relative_to(BASE)), frames=len(frames), video_sha256=provenance['video_sha256'])


def main():
    while not (BASE / 'batch/completed.json').exists() or any(
        not (p.parent / 'completed.json').exists() for p in (BASE / 'batch').glob('supplement_DP*/plan.json')):
        time.sleep(20)
    results = read(BASE / 'results.json')
    assert results['planned'] == 600
    assert len(results['groups']) == 20 and all(g['completed'] + g['unknown'] == 30 for g in results['groups'])
    jobs = list((BASE / 'batch/jobs').glob('*/*/*/*/process_result.json'))
    assert len(jobs) == 600
    completed = list(BASE.glob('*/evaluation/*/*/*/result.json'))
    interruptions = list(BASE.glob('*/evaluation/*/*/*/failure.json'))
    assert len(completed) == results['completed'] and len(completed) + len(interruptions) == 600
    assert sum(read(p)['returncode'] != 0 for p in jobs) == len(interruptions)
    frozen = read(BASE / 'study_freeze.json')
    assert object_hash(source_manifest()) == frozen['study']['framework_source']
    models = {}
    for name in NAMES:
        cfg = config(name)
        policies = library(cfg)
        assert {k: dict(version=v['version'], checkpoint_sha256=v['checkpoint_sha256']) for k, v in policies.items()} == frozen['study']['tasks'][name]['policies']
        source, checkpoint = naive_paths(name)
        assert digest(checkpoint) == frozen['study']['tasks'][name]['naive_checkpoint_sha256']
        if source is not None:
            authored = read(source.parent / 'framework_source.json')
            assert not authored['api_authored'] and all(digest(source / k) == v for k, v in authored['files'].items())
        models[name] = dict(API_priors=len(policies), naive_checkpoint_sha256=digest(checkpoint))
    output = ROOT / 'experiments/exp2/analysis'
    audit_scaleup.main(output / 'scaleup_execution_audit.json')
    analyze_outcomes.main()
    execution = read(output / 'scaleup_execution_audit.json')
    assert execution['all_600_attempts_audited']
    assert read(output / 'outcome_diagnostics.json')['completed'] == len(completed)
    roots = [p.parent for p in completed + interruptions]
    with ThreadPoolExecutor(max_workers=4) as pool:
        videos = list(pool.map(verify_video, roots))
    assert len(videos) == 600
    atomic(BASE / 'video_validation.json', dict(passed=True, videos=videos, count=len(videos)))
    artifacts = {str(p): digest(p) for p in [BASE / 'REPORT.md', BASE / 'results.json', BASE / 'episodes.csv',
        BASE / 'comparison.png', BASE / 'replays.html', BASE / 'video_validation.json',
        output / 'scaleup_execution_audit.json', output / 'outcome_diagnostics.json']}
    atomic(BASE / 'completion.json', dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        status='complete' if not interruptions else 'all_planned_attempts_finished_with_unknown_outcomes',
        evidence_checks_passed=True, attempted_episodes=600, completed_episodes=len(completed),
        interrupted_episodes=len(interruptions), all_outcomes_known=not interruptions,
        interruption_records=[str(p.relative_to(BASE)) for p in interruptions],
        independent_execution_audit=True, validated_videos=600,
        frozen_models=models, study_sha256=frozen['study_sha256'], artifacts=artifacts,
        finalizer_source_sha256=digest(__file__), additional_training=0, additional_physical_steps=0, additional_API_calls=0))
    print(dict(evidence_checks_passed=True, completed_episodes=len(completed), unknown_outcomes=len(interruptions),
        validated_videos=600, receipt=str(BASE / 'completion.json')), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        atomic(BASE / 'finalization_failure.json', dict(error=str(error), exception=type(error).__name__,
            time=datetime.now(timezone.utc).isoformat(), automatic_retry=False))
        raise
