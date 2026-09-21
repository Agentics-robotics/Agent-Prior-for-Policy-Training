"""Immutable inputs and explicit reuse of completed corrected baselines."""
import copy
import csv
import time
from pathlib import Path

from appl.io import ROOT, atomic, digest, read, source_manifest
from appl.prior_policies.data import configuration
from appl.prior_policies.deploy import library
from experiments.exp2.binary_gripper.common import BASE as BASELINES, TASKS

BASE = ROOT / 'runs/exp2/Exp2_new'
OLD = ROOT / 'runs/exp2/M1_astra_xhigh'
MODULE = 'experiments.exp2.exp2_new.run'
METHOD = 'APPL_6_xhigh'


def sources():
    paths = list(Path(__file__).parent.glob('*.py')) + [ROOT / 'experiments/exp2/EXP2_NEW.md']
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def config(task):
    return configuration(BASE / 'configurations' / (task + '.json'))


def episode_root(cell):
    return BASE / 'APPL_GPT6_astra_xhigh' / cell['task'] / cell['condition'] / str(cell['seed'])


def policy_identity(policies):
    return {k: dict(version=p['version'], checkpoint_sha256=p['checkpoint_sha256']) for k, p in policies.items()}


def prepare():
    from appl.scaleup.evaluate import prompt
    from experiments.exp2.astra.runner import unchanged_framework
    from experiments.exp2.binary_gripper.common import verify as baseline_verify
    if BASE.exists():
        raise ValueError('Exp2_new already exists; do not overwrite its frozen plan')
    unchanged_framework(); baseline_verify()
    baseline_rows = list(csv.DictReader((BASELINES / 'episodes.csv').open()))
    old_freeze = read(OLD / 'study_freeze.json')['study']['tasks']
    BASE.mkdir(parents=True)
    tasks = {}
    for task in TASKS:
        original = ROOT / 'experiments/exp2/configs/astra_xhigh' / (task + '.json')
        cfg = copy.deepcopy(read(original))
        cfg['api'] = dict(base_url='https://api.openai.com/v1', model='gpt-6-astra',
                          reasoning_effort='xhigh', timeout_seconds=600, service_tier='default')
        cfg['devices'] = list(range(8))
        for key in ('seeds', 'ood_seeds'):
            cfg['evaluation'][key] = cfg['evaluation'][key][:15]
        cfg['evaluation']['output'] = str(BASE / 'APPL_GPT6_astra_xhigh' / task)
        path = BASE / 'configurations' / (task + '.json')
        atomic(path, cfg)
        current = config(task); policies = library(current)
        if policy_identity(policies) != old_freeze[task]['policies']:
            raise ValueError('Original policy identity changed')
        files = {}
        for p in policies.values():
            folder = p['folder']
            selected = [folder/'submission.json', folder/'training/last.pt',
                        folder/'training/policy_context.json', folder/'training/handoff_evidence.json']
            selected += [p for p in (folder/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
            for p in selected:
                files[str(p)] = digest(p)
        for p in [current['dataset']/'manifest.json', current['output']/'normalization.json',
                  current['completion_contract'], original, OLD/task/'task.json']:
            files[str(p)] = digest(p)
        atomic(BASE/'prompts'/(task+'.json'), dict(instructions=prompt(task)))
        tasks[task] = dict(policies=policy_identity(policies), frozen_files=files,
                          configuration_sha256=digest(path), prompt=prompt(task))
    cells = []; reused = []
    for index in range(15):
        for condition in ('ID', 'OOD'):
            for task in TASKS:
                seed = config(task)['evaluation']['seeds' if condition == 'ID' else 'ood_seeds'][index]
                peers = [r for r in baseline_rows if r['task']==task and r['condition']==condition and int(r['seed'])==seed]
                if len(peers)!=2 or any(r['completed']!='True' for r in peers):
                    raise ValueError('Missing completed corrected comparator')
                references = {}
                for row in peers:
                    root = Path(row['episode_root']); audit = read(root/'audit.json')
                    if not audit['passed'] or digest(root/'result.json')!=audit['result_sha256']:
                        raise ValueError('Corrected baseline audit changed')
                    references[row['method']] = str(root)
                    reused.append(dict(task=task, condition=condition, seed=seed, method=row['method'],
                                       result=read(root/'result.json'), root=str(root),
                                       result_sha256=digest(root/'result.json'), initial_sha256=digest(root/'initial_state.json')))
                initial = [read(Path(p)/'initial_state.json') for p in references.values()]
                if initial[0]!=initial[1]:
                    raise ValueError('Baseline paired reset mismatch')
                cells.append(dict(index=len(cells), task=task, condition=condition, seed=seed,
                                  baseline_roots=references, initial_sha256=digest(Path(next(iter(references.values())))/'initial_state.json')))
    (BASE/'corrected_baselines').symlink_to(BASELINES.resolve(), target_is_directory=True)
    atomic(BASE/'baseline_subset.json', reused)
    plan = dict(study='Exp2_new', created=time.time(), tasks=tasks, cells=cells,
                selection='First 15 entries of every original ordered split; no outcome-based selection.',
                order='Seed index, ID then OOD, then fixed five-task order; one active episode.',
                intervention='Only final gripper sign, zero opens. Original APPL prompt/tools/history/DDPM100/5000-step cap.',
                provider_change='Official OpenAI endpoint with new credential; model gpt-6-astra, effort xhigh, standard tier.',
                framework_source=source_manifest(), executor_source=sources(),
                allowed_devices=list(range(8)), maximum_active_devices=5,
                training_updates=0, segmentation_calls=0, policy_design_calls=0, automatic_retry=False,
                baseline_subset_sha256=digest(BASE/'baseline_subset.json'),
                original_freeze_sha256=digest(OLD/'study_freeze.json'))
    atomic(BASE/'plan.json', plan)
    atomic(BASE/'plan_sha256.json', dict(sha256=digest(BASE/'plan.json')))
    print(dict(prepared=True, APPL_trials=len(cells), reused_baseline_trials=len(reused),
               frozen_policies=sum(len(t['policies']) for t in tasks.values())), flush=True)


def verify(task=None):
    plan = read(BASE/'plan.json')
    if digest(BASE/'plan.json')!=read(BASE/'plan_sha256.json')['sha256']:
        raise ValueError('Frozen plan changed')
    if sources()!=plan['executor_source'] or source_manifest()!=plan['framework_source']:
        raise ValueError('Frozen source changed')
    if digest(BASE/'baseline_subset.json')!=plan['baseline_subset_sha256']:
        raise ValueError('Comparator subset changed')
    if task is not None:
        record = plan['tasks'][task]
        if digest(BASE/'configurations'/(task+'.json'))!=record['configuration_sha256']:
            raise ValueError('Frozen configuration changed')
        for path, sha in record['frozen_files'].items():
            if digest(path)!=sha:
                raise ValueError('Frozen policy/input changed: '+path)
    return plan
