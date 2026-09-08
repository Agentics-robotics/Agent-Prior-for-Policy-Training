import argparse
import subprocess
import sys
import time
from relative_dp.utils import ROOT, atomic_json, read_json, sha256, object_hash


def child(args):
    # Caller is already a Pixi task. Every new process explicitly re-enters the
    # same locked Pixi environment and has independent CUDA compile state.
    command=['/home/users/oscar/.pixi/bin/pixi','run','python',*args]
    subprocess.run(command,cwd=ROOT,check=True)


def debug_rollout_check(run_id):
    import numpy as np
    import torch
    from .learning import Loaded,RUNS
    from .geometry import GeometryEnv
    from .calibrate import record
    output=ROOT/f'artifacts/round2/debug/{run_id}/rollout_check.json'
    checkpoint=ROOT/f'artifacts/round2/debug/{run_id}/checkpoints/step_000200.pt'
    if output.exists():
        result=read_json(output);assert result['checkpoint_hash']==sha256(checkpoint) and result['passed'];return result
    begun=time.monotonic()
    run=next(r for r in RUNS if r['run_id']==run_id)
    loaded=Loaded(checkpoint);env=GeometryEnv(run['task'])
    try:
        obs,_=env.reset(record(run['task'],29000200,50))
        history=[obs.copy(),obs.copy()]
        generator=torch.Generator(device='cuda').manual_seed(29000200)
        for _ in range(2):
            chunk=loaded.actions(np.asarray(history,np.float32),generator)
            assert chunk.shape==(16,4) and np.isfinite(chunk).all()
            for action in chunk[:4]:
                obs,_,_,_,_=env.step(np.clip(action,-1,1).astype(np.float32))
                assert np.isfinite(obs).all()
                history=[history[-1],obs.copy()]
        result=dict(passed=True,steps=8,run_id=run_id,checkpoint_hash=sha256(checkpoint),
                    wall_seconds=time.monotonic()-begun,purpose='finite diffusion/decode/step integration only; no success requirement or angle selection')
        atomic_json(output,result);return result
    finally:env.close()


def preflight():
    child(['-m','pytest','-q','tests/round2'])
    from .learning import code_hash
    result=dict(passed=True,code_hash=code_hash(),time=time.time())
    atomic_json(ROOT/'artifacts/round2/preflight.json',result)
    return result


def formal_training():
    from .learning import RUNS,code_hash
    from .collect import audit
    manifest=read_json(ROOT/'data/round2/manifest.json')
    if any(v.get('blocked') for v in manifest['tasks'].values()):
        raise RuntimeError('A task is blocked in calibration; inspect calibration evidence before formal training')
    preflight()
    report=read_json(ROOT/'artifacts/round2/data_audit.json')
    assert report['passed'] and report['full_demo_replays']==40
    assert report['manifest_hash']==sha256(ROOT/'data/round2/manifest.json')
    # Exactly 200 actual debug updates per configuration, split at 100 to
    # exercise serialized optimizer/EMA/RNG restoration in a fresh process.
    for run in RUNS:
        child(['-m','round2.learning',run['run_id'],'--debug','--stop-after','100'])
        child(['-m','round2.learning',run['run_id'],'--debug'])
    frozen=dict(code_hash=code_hash(),manifest_hash=sha256(ROOT/'data/round2/manifest.json'),
                protocol_hash=sha256(ROOT/'ROUND2_PROTOCOL.md'),
                configs={r['run_id']:read_json(ROOT/f'artifacts/round2/debug/{r["run_id"]}/config.json') for r in RUNS})
    path=ROOT/'configs/round2/freeze.json'
    if path.exists():assert read_json(path)==frozen
    else:atomic_json(path,frozen)
    for run in RUNS:
        child(['-m','round2.learning',run['run_id']])
    for task in ('drawer','door'):
        a=read_json(ROOT/f'runs/round2/r2_{task}_world_n20_s0/complete.json')
        b=read_json(ROOT/f'runs/round2/r2_{task}_frame_n20_s0/complete.json')
        for k in ('initial_weights_hash','pairing_digest','parameter_count'):assert a[k]==b[k],k
    atomic_json(ROOT/'artifacts/round2/pairing_audit.json',dict(passed=True,actual_formal_updates=80000,actual_debug_updates=800))


def evaluation():
    from .learning import RUNS
    from .evaluate import freeze_selection
    for run in RUNS:
        child(['-c','import sys; from round2.cli import debug_rollout_check; debug_rollout_check(sys.argv[1])',run['run_id']])
    for run in RUNS:
        for step in (5000,10000,20000):
            child(['-m','round2.evaluate',run['run_id'],'dev','--step',str(step)])
    freeze_selection()
    for run in RUNS:child(['-m','round2.evaluate',run['run_id'],'test'])


def main():
    import fcntl
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['preflight','calibrate','collect','train','evaluate','report','all'])
    args=p.parse_args()
    lock=(ROOT/'artifacts/round2/pipeline.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Another Round2 pipeline is active; inspect its logs before resuming.')
    stage=args.stage
    if stage=='preflight':preflight()
    if stage in ('calibrate','all'):
        from .calibrate import calibrate
        calibrate()
    if stage in ('collect','all'):
        from .collect import collect
        collect()
    if stage in ('train','all'):formal_training()
    if stage in ('evaluate','all'):evaluation()
    if stage in ('report','all'):
        from .report import report
        report()
        child(['scripts/round2/verify.py'])


if __name__=='__main__':main()
