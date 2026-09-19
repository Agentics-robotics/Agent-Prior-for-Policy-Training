"""An additional API-model study using the unchanged Exp2 scientific framework."""
import argparse
from collections import Counter, deque
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

from appl.io import ROOT, PIXI, MANIFEST, atomic, read, digest, object_hash, source_manifest

BASE = ROOT / 'runs/exp2/M1_astra_xhigh'
OLD = ROOT / 'runs/exp2/M1_scaleup'
CONFIGS = ROOT / 'experiments/exp2/configs/astra_xhigh'
MODULE = 'experiments.exp2.astra.runner'
MODEL, EFFORT = 'gpt-6-astra', 'xhigh'
DEVICES = [1, 2, 3, 5]
NAMES = ['drawer_exchange', 'two_block_sort', 'buffer_swap', 'unstack_sort', 'tray_pack']


def config(name):
    from appl.prior_policies.data import configuration
    if name not in NAMES:
        raise ValueError('Unknown task')
    cfg = configuration(CONFIGS / (name + '.json'))
    if (cfg['api']['model'], cfg['api']['reasoning_effort']) != (MODEL, EFFORT):
        raise ValueError('Unexpected API model or effort')
    if cfg['output'] != (BASE / name).resolve() or cfg['devices'] != DEVICES:
        raise ValueError('Study output/allocation mismatch')
    return cfg


def original_config(name):
    return read(ROOT / 'experiments/exp2/configs/scaleup' / (name + '.json'))


def baseline_paths(name):
    if name == 'drawer_exchange':
        return None, ROOT / 'runs/exp2/m0/training/last.pt'
    p = OLD / name / 'naive_DP'
    return p / 'source', p / 'training/last.pt'


def bind_evaluator():
    """Change only outer-run paths; execute the original evaluator function."""
    import appl.scaleup.protocol as protocol
    protocol.BASE = BASE
    protocol.CONFIGS = CONFIGS
    protocol.config = config
    protocol.naive_paths = baseline_paths
    from appl.scaleup import evaluate
    evaluate.BASE = BASE
    evaluate.config = config
    evaluate.naive_paths = baseline_paths
    return evaluate


def unchanged_framework():
    expected = read(OLD / 'study_freeze.json')['study']['framework_source']
    manifest = source_manifest()
    actual = object_hash(manifest)
    if actual != expected:
        incident = BASE / 'incidents/buffer_red_h02_reload'
        receipt = incident / 'framework_revision.json'
        if not receipt.exists():
            raise ValueError('Original scientific framework changed without a recorded correction')
        revision = read(receipt)
        key = 'src/appl/prior_policies/engine.py'
        original = (incident / 'original_engine.py').read_text()
        needle = "reloaded.eval()\n    replay=sample(reloaded,raw,spec,torch.Generator(device='cuda:0').manual_seed(913))"
        replacement = "reloaded.eval().requires_grad_(False)\n    replay=sample(reloaded,raw,spec,torch.Generator(device='cuda:0').manual_seed(913))"
        if original.count(needle) != 1 or (ROOT / key).read_text() != original.replace(needle,replacement):
            raise ValueError('Framework correction exceeds the single reload-gradient-flag change')
        reconstructed = dict(manifest)
        reconstructed[key] = digest(incident / 'original_engine.py')
        if (object_hash(reconstructed) != expected
                or revision['original_framework_source'] != expected
                or revision['corrected_framework_source'] != actual
                or revision['corrected_engine_sha256'] != manifest[key]):
            raise ValueError('Framework correction provenance mismatch')
    return actual


def immutable_json(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError('Existing preparation changed: ' + str(path))
    else:
        atomic(path, value)


def prepare():
    from appl.prior_policies.data import fit_full_normalizer
    from appl.gpu import identity
    source = unchanged_framework()
    BASE.mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(parents=True, exist_ok=True)
    tasks = {}
    for name in NAMES:
        old = original_config(name)
        cfg = copy.deepcopy(old)
        cfg['api'].update(model=MODEL, reasoning_effort=EFFORT)
        cfg['devices'] = DEVICES
        cfg['output'] = str((BASE / name).relative_to(ROOT))
        cfg['dataset'] = 'data/exp2/scaleup_astra_xhigh/' + name + '/processed'
        cfg['evaluation']['output'] = cfg['output'] + '/evaluation'
        if name == 'drawer_exchange':
            cfg.pop('task_id', None)
        path = CONFIGS / (name + '.json')
        immutable_json(path, cfg)
        immutable_json(BASE / name / 'task.json', read(OLD / name / 'task.json'))
        current = config(name)
        fit_full_normalizer(current)
        norm = read(current['output'] / 'normalization.json')
        original_norm = read(ROOT / old['output'] / 'normalization.json')
        if norm != original_norm:
            raise ValueError('Original full-demo normalizer did not reproduce exactly')
        # A read-only comparison link lets the original evaluator verify paired resets.
        link = BASE / name / 'evaluation/naive_DP'
        target = (OLD / name / 'evaluation/naive_DP').resolve()
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if link.resolve() != target:
                raise ValueError('Comparator link changed')
        elif link.exists():
            raise ValueError('Comparator path must be an explicit reuse link')
        else:
            link.symlink_to(target, target_is_directory=True)
        tasks[name] = dict(configuration_sha256=digest(path),
            original_configuration_sha256=digest(ROOT / 'experiments/exp2/configs/scaleup' / (name + '.json')),
            changed_top_level_fields=[k for k in sorted(set(old) | set(cfg)) if old.get(k) != cfg.get(k)],
            sources=norm['sources'],normalizer_sha256=norm['normalizer_sha256'],
            comparator=str(target),training=cfg['training'],evaluation_seeds=cfg['evaluation']['seeds'],
            ood_seeds=cfg['evaluation']['ood_seeds'])
    immutable_json(BASE / 'preparation.json', dict(model=MODEL,reasoning_effort=EFFORT,
        new_APPL_episodes=300,reused_DP_episodes=300,original_APPL_comparison=str(OLD),
        original_results_sha256=digest(OLD / 'results.json'),framework_source=source,tasks=tasks,
        test_scope='Paired follow-up on previously analyzed layouts; no test feedback supplied to design.',
        modifications='API model and effort; fresh API-owned cuts, priors and policies. Same scientific framework.'))
    if not (BASE / 'allocation.json').exists():
        atomic(BASE / 'allocation.json', dict(time=time.time(),devices=DEVICES,
            occupancy={i:identity(i) for i in DEVICES},user_limit_physical_devices=4))
    print(dict(prepared=str(BASE),model=MODEL,reasoning_effort=EFFORT),flush=True)


def api_identity(path, require_consumed=True):
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
    rows = db.execute("SELECT json_extract(request,'$.model'),json_extract(request,'$.reasoning.effort'),"
        "json_extract(response,'$.model'),json_extract(response,'$.reasoning.effort'),"
        "status,COUNT(*) FROM api GROUP BY 1,2,3,4,5").fetchall()
    db.close()
    if not rows:
        raise ValueError('No real API requests: ' + str(path))
    for model, effort, returned, returned_effort, state, count in rows:
        if (model, effort) != (MODEL, EFFORT):
            raise ValueError('Unexpected requested model/effort')
        if returned is not None and returned != MODEL:
            raise ValueError('Unexpected returned model')
        if state == 'consumed' and returned != MODEL:
            raise ValueError('Consumed response lacks expected model identity')
        if state == 'consumed' and returned_effort != EFFORT:
            raise ValueError('Consumed response lacks expected reasoning effort')
        if require_consumed and state != 'consumed':
            raise ValueError('Unresolved API attempt retained; no implicit retry')
    return [dict(model=m,effort=e,response_model=r,response_effort=re,status=s,calls=n)
            for m,e,r,re,s,n in rows]


def segment(name):
    from appl.prior_policies.segment import run
    cfg = config(name)
    if (cfg['output'] / 'segmentation/completed.json').exists():
        api_identity(cfg['dataset'] / '_session/journal.sqlite')
        return
    if (cfg['output'] / 'segmentation').exists() or cfg['dataset'].exists():
        raise ValueError('An existing segmentation attempt cannot be retried implicitly')
    run(cfg)
    atomic(cfg['output'] / 'segmentation/model_identity.json',
           api_identity(cfg['dataset'] / '_session/journal.sqlite'))


def entries():
    from appl.prior_policies.data import catalog
    return [(name, entry) for name in NAMES for entry in catalog(config(name))]


def command(args):
    return [PIXI, 'run', '--manifest-path', str(MANIFEST), '--locked', 'python', '-m', MODULE, *args]


def train_queue(gpu, shard):
    from appl.prior_policies.design import design
    from appl.prior_policies.runner import train_submitted
    from appl.prior_policies.report import authorship
    for index, (name, entry) in enumerate(entries()):
        if index % len(DEVICES) != shard:
            continue
        cfg = config(name)
        folder = cfg['output'] / 'policies' / entry['skill_id'] / f"heuristic_{entry['heuristic_index']:02d}"
        immutable_json(folder / 'assignment.json', entry)
        if (folder / 'failure.json').exists():
            raise ValueError('Recorded policy failure requires reconciliation')
        try:
            print(dict(stage='design',task=name,policy=entry['policy_id'],gpu=gpu),flush=True)
            submission = design(cfg, folder, gpu)
            atomic(folder / 'model_identity.json', api_identity(folder / 'design/journal.sqlite'))
            authorship(folder, submission)
            train_submitted(cfg, entry['policy_id'], gpu)
            subprocess.run(command(['check','--task',name,'--policy-id',entry['policy_id'],'--gpu',str(gpu)]),
                cwd=ROOT,check=True)
            print(dict(stage='trained_and_checked',task=name,policy=entry['policy_id']),flush=True)
        except Exception as error:
            atomic(folder / 'failure.json', dict(error=str(error),time=time.time(),automatic_retry=False))
            raise


def check(name, policy_id):
    import numpy as np
    from appl.gpu import verify_cuda
    from appl.prior_policies.data import catalog
    from appl.prior_policies.runner import PolicyProcess
    cfg = config(name)
    entry, = [e for e in catalog(cfg) if e['policy_id'] == policy_id]
    folder = cfg['output'] / 'policies' / entry['skill_id'] / f"heuristic_{entry['heuristic_index']:02d}"
    out = cfg['output'] / 'inference_checks' / policy_id
    if (out / 'result.json').exists():
        if read(out / 'result.json')['checkpoint_sha256'] != digest(folder / 'training/last.pt'):
            raise ValueError('Checked checkpoint changed')
        return
    out.mkdir(parents=True,exist_ok=False)
    worker = None
    try:
        dataset = read(entry['dataset']); s = dataset['segments'][0]
        source = Path(entry['dataset']).parent / s['file']
        observations = read(source)['observations'][:9]
        atomic(out / 'inputs.json',dict(source=str(source),source_sha256=digest(source),
            start_index=s['start'],observations=observations,seed=913,physical_steps=0))
        worker = PolicyProcess(cfg,folder,out / 'worker',913)
        actions = [worker.action(o['state'],reset=i==0) for i,o in enumerate(observations)]
        if len(actions) != 9 or np.shape(actions) != (9,8) or not np.isfinite(actions).all():
            raise ValueError('Invalid B=1 deployment actions')
        worker.close();worker=None
        closed=read(out / 'worker/closed.json')
        if closed['returncode'] != 0 or closed['inference_chunks'] != 2:
            raise ValueError('Deployment worker protocol failed')
        atomic(out / 'actions.json',actions)
        atomic(out / 'result.json',dict(policy_id=policy_id,passed=True,generated_actions=9,
            denoising_chunks=2,checkpoint_sha256=digest(folder / 'training/last.pt'),device=verify_cuda(),
            optimizer_updates=0,simulator_steps=0,original_training_observations_only=True))
    except Exception as error:
        atomic(out / 'failure.json',dict(error=str(error),automatic_retry=False))
        raise
    finally:
        if worker is not None:
            worker.close()


def freeze():
    from appl.prior_policies.deploy import library
    from appl.prior_policies.report import authorship
    evaluator = bind_evaluator()
    original = read(OLD / 'study_freeze.json')['study']
    rows = {}
    for name in NAMES:
        cfg=config(name);policies=library(cfg);checks=[]
        api_identity(cfg['dataset'] / '_session/journal.sqlite')
        for key,p in policies.items():
            authorship(p['folder'],read(p['folder'] / 'submission.json'))
            api_identity(p['folder'] / 'design/journal.sqlite')
            record=read(cfg['output'] / 'inference_checks' / key / 'result.json')
            if not record['passed'] or record['checkpoint_sha256'] != p['checkpoint_sha256']:
                raise ValueError('Missing exact-checkpoint deployment check')
            checks.append(record)
        immutable_json(cfg['output'] / 'inference_checks/summary.json',dict(passed=True,
            all_policies_checked=True,prior_policies=len(policies),records=checks,simulator_steps=0))
        _, checkpoint=baseline_paths(name)
        if digest(checkpoint) != original['tasks'][name]['naive_checkpoint_sha256']:
            raise ValueError('Frozen original baseline changed')
        rows[name]=dict(configuration_sha256=digest(cfg['_path']),
            task_spec_sha256=digest(BASE / name / 'task.json'),
            completion_contract_sha256=digest(cfg['completion_contract']),
            dataset_manifest_sha256=digest(cfg['dataset'] / 'manifest.json'),
            normalization_sha256=digest(cfg['output'] / 'normalization.json'),
            naive_checkpoint_sha256=digest(checkpoint),
            policies={k:dict(version=p['version'],checkpoint_sha256=p['checkpoint_sha256']) for k,p in policies.items()},
            layouts=original['tasks'][name]['layouts'])
        for condition,layouts in rows[name]['layouts'].items():
            if any(evaluator.initial_layout(name,v['seed'],condition) != v['layout'] for v in layouts):
                raise ValueError('Original paired layouts changed')
    study=dict(tasks=rows,framework_source=unchanged_framework(),runner_sha256=digest(__file__),
        max_steps=5000,episodes=300,model=MODEL,effort=EFFORT,
        comparator_results_sha256=digest(OLD / 'results.json'))
    path=BASE / 'study_freeze.json'
    if path.exists():
        if read(path)['study'] != study:
            raise ValueError('New study freeze changed')
    else:
        if any(BASE.glob('*/evaluation/APPL/*/*/plan.json')):
            raise ValueError('Cannot freeze after testing begins')
        atomic(path,dict(study=study,study_sha256=object_hash(study),frozen=time.time()))
        shutil.copyfile(__file__,BASE / 'frozen_runner.py')
    return study


def evaluate(name, condition, seed):
    frozen=read(BASE / 'study_freeze.json')['study']
    if frozen['runner_sha256'] != digest(__file__):
        raise ValueError('Frozen additional-study runner changed')
    from . import transport
    if digest(transport.__file__) != '0296eea2c9148a12ab64f7323b367a31d0e85ab6f5f3539fe2b955a997652cb1':
        raise ValueError('Declared API capacity gate source changed')
    out=BASE / name / 'evaluation/APPL' / condition / str(seed)
    transport.install_episode_gate(BASE,out,slots=4)
    evaluator=bind_evaluator()
    result=evaluator.evaluate(name,'APPL',condition,seed)
    atomic(out / 'model_identity.json',api_identity(out / 'api/journal.sqlite'))
    return result


def launch_job(label,args,gpu=None):
    from appl.gpu import identity
    out=BASE / 'jobs' / label
    if out.exists():
        if (out / 'process_result.json').exists():
            result=read(out / 'process_result.json')
            if result['returncode'] == 0:
                return None
        raise ValueError('Existing job is retained, not automatically retried: ' + label)
    out.mkdir(parents=True)
    stdout=(out / 'stdout.log').open('w');stderr=(out / 'stderr.log').open('w')
    occupancy=None if gpu is None else identity(gpu)
    args=command(args)
    proc=subprocess.Popen(args,cwd=ROOT,stdout=stdout,stderr=stderr,start_new_session=True)
    atomic(out / 'process.json',dict(pid=proc.pid,command=args,time=time.time(),occupancy=occupancy))
    return dict(process=proc,out=out,stdout=stdout,stderr=stderr,label=label,gpu=gpu)


def run_jobs(stage,jobs,slots,stop_on_failure):
    from appl.gpu import identity
    pending=deque(jobs);active={};failed=[];finished=0
    while pending or active:
        for slot,job in list(active.items()):
            code=job['process'].poll()
            if code is None:
                continue
            job['stdout'].close();job['stderr'].close()
            atomic(job['out'] / 'process_result.json',dict(returncode=code,finished=time.time()))
            if code:
                failed.append(job['label'])
            del active[slot];finished+=1
        for slot,gpu in enumerate(slots):
            if slot in active or not pending or (failed and stop_on_failure):
                continue
            required=10000 if stage=='evaluation' else 8000
            if gpu is not None and identity(gpu)['free_mib'] < required:
                continue
            label,args=pending.popleft()
            job=launch_job(label,args + ([] if gpu is None else ['--gpu',str(gpu)]),gpu)
            if job is not None:
                active[slot]=job
            else:
                finished+=1
        atomic(BASE / 'status.json',dict(stage=stage,pending=len(pending),finished=finished,failed=failed,
            active=[dict(label=j['label'],pid=j['process'].pid,gpu=j['gpu']) for j in active.values()],time=time.time()))
        if failed and stop_on_failure and not active:
            raise RuntimeError('Stage failure retained without retries: ' + str(failed))
        if pending or active:
            time.sleep(5)
    return failed


def run():
    prepare()
    with (BASE / 'owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_jobs('segmentation',[(f'segmentation/{n}',['segment','--task',n]) for n in NAMES],
                 [None]*4,True)
        # One independent queue per physical GPU; queues preserve exact API packages.
        run_jobs('design_training',[(f'train_queue/{i}',['queue','--shard',str(i)]) for i in range(4)],
                 DEVICES,True)
        evaluate_study()


def evaluate_study():
    """Enter the original test matrix only after every required artifact exists."""
    missing=[]
    for name,entry in entries():
        folder=config(name)['output'] / 'policies' / entry['skill_id'] / f"heuristic_{entry['heuristic_index']:02d}"
        required=[folder / 'submission.json',folder / 'training/result.json',
                  folder / 'training/process_result.json',
                  BASE / name / 'inference_checks' / entry['policy_id'] / 'result.json']
        missing.extend(str(p.resolve().relative_to(BASE.resolve())) for p in required if not p.exists())
        if (folder / 'training/process_result.json').exists() and read(folder / 'training/process_result.json')['returncode'] != 0:
            raise ValueError('Retained training failure has not been reconciled: '+str(folder))
    if missing:
        raise ValueError('Evaluation remains gated by missing artifacts: '+str(missing))
    freeze()
    jobs=[]
    for index in range(30):
        for condition in ('ID','OOD'):
            for name in NAMES:
                seed=config(name)['evaluation']['seeds' if condition=='ID' else 'ood_seeds'][index]
                jobs.append((f'evaluation/{name}/{condition}/{seed}',
                    ['evaluate','--task',name,'--condition',condition,'--seed',str(seed)]))
    # Two episodes per GPU can overlap API waits, with a memory gate per launch.
    failures=run_jobs('evaluation',jobs,DEVICES+DEVICES,False)
    from .report import generate
    result=generate()
    atomic(BASE / 'completion.json',dict(finished=time.time(),new_attempts=300,
        process_failures=failures,summary=result,automatic_retries=0))


def run_evaluation():
    """Explicit continuation; preserve original failed queue/process receipts."""
    with (BASE / 'owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX | fcntl.LOCK_NB)
        evaluate_study()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','run','run-evaluation','segment','queue','check','freeze','evaluate','report'])
    p.add_argument('--task',choices=NAMES);p.add_argument('--gpu',type=int)
    p.add_argument('--shard',type=int);p.add_argument('--policy-id')
    p.add_argument('--condition',choices=['ID','OOD']);p.add_argument('--seed',type=int)
    p.add_argument('--device-isolated',action='store_true')
    args=p.parse_args()
    if args.gpu is not None and args.gpu not in DEVICES:
        p.error('GPU outside the declared four-device pool')
    if args.command in ('check','evaluate'):
        if not args.device_isolated:
            from appl.gpu import launch
            return launch(args.gpu,sys.argv[1:],module=MODULE)
        nodes={v.name for v in Path('/dev').glob('nvidia*') if v.name.removeprefix('nvidia').isdigit()}
        if nodes != {'nvidia'+os.environ['APPL_GPU_MINOR']}:
            raise RuntimeError('Expected one isolated GPU device')
        os.environ['CUDA_VISIBLE_DEVICES']=os.environ['APPL_GPU_UUID']
        os.environ['MUJOCO_EGL_DEVICE_ID']='0'
    unchanged_framework()
    if args.command=='prepare':prepare()
    elif args.command=='run-evaluation':run_evaluation()
    elif args.command=='segment':segment(args.task)
    elif args.command=='queue':train_queue(args.gpu,args.shard)
    elif args.command=='check':check(args.task,args.policy_id)
    elif args.command=='freeze':freeze()
    elif args.command=='evaluate':evaluate(args.task,args.condition,args.seed)
    elif args.command=='report':
        from .report import generate
        generate()
    else:run()


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        print(type(error).__name__ + ': ' + str(error),file=sys.stderr,flush=True)
        raise
