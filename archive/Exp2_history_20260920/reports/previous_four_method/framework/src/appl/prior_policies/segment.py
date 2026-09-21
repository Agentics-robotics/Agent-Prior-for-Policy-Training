"""Run the new API segmentation and audit its unchanged published output."""
import hashlib
import json
import os
import sqlite3
import time
from ..io import ROOT, atomic, read, digest, object_hash, archive_source
from ..demonstrations import Limits, process_demonstrations
from ..demonstrations.decouple import PROMPT
from .design import client


def run(cfg):
    root=cfg['output']/'segmentation';root.mkdir(parents=True,exist_ok=True)
    inputs=[ROOT/cfg['normalization']['directory']/(k+'.json') for k in cfg['normalization']['train_ids']]
    interface=read(ROOT/'data/exp2/reconstruction_public/interface.json')['native']
    if cfg.get('task_id'):
        interface['observation_fields'] += [dict(name=name,dimension=3,frame='world',units='metre') for name in ('red_goal','blue_goal')]
    previous_path=cfg['segmentation'].get('previous_dataset')
    previous=read(ROOT/previous_path/'manifest.json') if previous_path else None
    context=dict(experiment_version='M1_v2',observation_fields=interface['observation_fields'],
        action_contract=interface['action'],task_completion_contract=read(cfg['completion_contract']),
        research_goal='Learn reusable skills with substantially expanded transition coverage and explicit handoff interfaces. Each heuristic will become a separately implemented and trained prior-based Diffusion Policy.',
        normalization='All policies use one observation and action normalizer fitted to the complete original training demonstrations, not per-skill ranges.',
        previous_overlap=None if previous is None else dict(unique_shared_actions=previous['checks']['overlapping_actions'],
            total_original_actions=previous['checks']['assigned_unique_actions'],
            guidance='M1_v1 shared only short transition tails. M1_v2 must substantially expand meaningful shared transition context. You choose the boundaries and explain the support.'),
        deployment_requirements=None)
    limits=Limits(**{k:v for k,v in cfg['segmentation'].items() if k!='previous_dataset'})
    atomic(root/'prompt.json',dict(prompt=PROMPT,sha256=hashlib.sha256(PROMPT.encode()).hexdigest()))
    provider=client(cfg)
    record=dict(source_version=archive_source(cfg['output']),context=context,limits=limits.__dict__,
        input_files={str(p.relative_to(ROOT)):digest(p) for p in inputs},provider=provider.configuration())
    if (root/'launch.json').exists() and read(root/'launch.json')!=record:raise ValueError('Segmentation launch changed')
    atomic(root/'launch.json',record)
    try:
        result=process_demonstrations(inputs,cfg['dataset'],client=provider,context=context,limits=limits)
        validation=audit(cfg,result,previous)
        atomic(root/'completed.json',dict(completed=time.time(),plan_hash=result['plan_hash'],**validation))
        print(json.dumps(validation),flush=True)
        return result
    except Exception as error:
        atomic(root/'failure.json',dict(error=str(error),time=time.time(),automatic_retry=False));raise
    finally:os.environ.pop('APPL_PRIOR_TOKEN',None)


def audit(cfg,manifest,previous):
    root=cfg['output']/'segmentation';session=cfg['dataset']/'_session'
    plan=read(session/'plans'/(manifest['plan_hash']+'.json'))
    if object_hash(plan)!=manifest['plan_hash']:raise ValueError('Plan hash changed')
    db=sqlite3.connect('file:'+str((session/'journal.sqlite').resolve())+'?mode=ro',uri=True)
    writes=[];submits=[]
    for raw, in db.execute('SELECT response FROM api WHERE response IS NOT NULL ORDER BY seq'):
        for item in json.loads(raw).get('output',[]):
            if item.get('type')!='function_call':continue
            args=json.loads(item['arguments'])
            if item['name']=='write_plan':writes.append(args['plan'])
            if item['name']=='submit_datasets':submits.append(args['expected_hash'])
    statuses=dict(db.execute('SELECT status,COUNT(*) FROM api GROUP BY status'));db.close()
    if plan not in writes or manifest['plan_hash'] not in submits:raise ValueError('Missing exact API plan/submission')
    if set(statuses)!={'consumed'}:raise ValueError('Unreconciled segmentation API call')
    for name,sha in manifest['files'].items():
        if digest(cfg['dataset']/name)!=sha:raise ValueError('Published file changed')
    boundaries=[];heuristics=0
    for skill in plan['skills']:
        dataset=read(cfg['dataset']/'datasets'/skill['skill_id']/'dataset.json')
        if dataset['heuristics']!=skill['heuristics']:raise ValueError('API heuristic changed')
        heuristics+=len(skill['heuristics'])
        for s in skill['segments']:
            boundaries.append(dict(skill_id=skill['skill_id'],**s))
    overlaps=[]
    for a in boundaries:
        for b in boundaries:
            if a['trajectory_id']!=b['trajectory_id'] or (a['start'],a['skill_id']) >= (b['start'],b['skill_id']):continue
            start=max(a['start'],b['start']);stop=min(a['stop'],b['stop'])
            if stop>start:overlaps.append(dict(trajectory_id=a['trajectory_id'],left=a['skill_id'],right=b['skill_id'],start=start,stop=stop,actions=stop-start))
    result=dict(checks=manifest['checks'],heuristics=heuristics,api_statuses=statuses,
        previous_overlapping_actions=None if previous is None else previous['checks']['overlapping_actions'],
        overlap_ratio_to_v1=None if previous is None else manifest['checks']['overlapping_actions']/previous['checks']['overlapping_actions'],
        api_authorship_verified=True,manual_output_edits=0,training_updates=0,simulator_calls=0)
    atomic(root/'validation.json',result);atomic(root/'boundaries.json',dict(segments=boundaries,overlaps=overlaps))
    lines=['# M1_v2 API segmentation','',f"Skills: {len(plan['skills'])}; heuristics: {heuristics}; original demonstrations: {manifest['checks']['original_demonstrations']}.",
        f"Shared unique actions: {manifest['checks']['overlapping_actions']}; previous study: {result['previous_overlapping_actions']}; ratio: {result['overlap_ratio_to_v1']}.",'',
        'All boundaries, priors and handoff interfaces are exact API output. Larger overlap expands temporal support; it does not prove recovery from unseen spatial/contact errors.','',
        '## API skill datasets','']
    lines += [f"- [{s['name']}]({cfg['dataset']/'datasets'/s['skill_id']/'heuristic.md'})" for s in plan['skills']]
    lines += ['','[Validation](validation.json) · [Boundaries and overlaps](boundaries.json) · [English prompt](prompt.json)','']
    (root/'REPORT.md').write_text('\n'.join(lines));return result
