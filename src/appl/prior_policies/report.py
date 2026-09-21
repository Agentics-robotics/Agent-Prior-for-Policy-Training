"""Execution and API-authorship accounting, separate from generated prior docs."""
import csv
import json
import sqlite3
from pathlib import Path
from collections import Counter
from ..io import read,atomic,digest
from .data import catalog,evaluation_output


def api_accounting(path):
    if not path.exists():return dict(calls=0,states={},usage={},seconds=0.)
    db=sqlite3.connect('file:'+str(path.resolve())+'?mode=ro',uri=True)
    states=dict(db.execute('SELECT status,COUNT(*) FROM api GROUP BY status'))
    usage=Counter();seconds=0.
    for raw, in db.execute("SELECT payload FROM events WHERE kind='api_cost'"):
        row=json.loads(raw);seconds+=row['elapsed_seconds'];u=row.get('usage') or {}
        for key in ('input_tokens','output_tokens','total_tokens'):usage[key]+=u.get(key,0)
        usage['cached_input_tokens']+=(u.get('input_tokens_details') or {}).get('cached_tokens',0)
    db.close();return dict(calls=sum(states.values()),states=states,usage=dict(usage),seconds=seconds)


def authorship(folder,submission):
    db=sqlite3.connect('file:'+str((folder/'design/journal.sqlite').resolve())+'?mode=ro',uri=True)
    writes={};submitted=[]
    for seq,raw in db.execute('SELECT seq,response FROM api WHERE response IS NOT NULL ORDER BY seq'):
        response=json.loads(raw)
        for item in response.get('output',[]):
            if item.get('type')!='function_call':continue
            args=json.loads(item['arguments'])
            if item['name']=='write_file':writes[args['path']]=dict(content=args['content'],api_request=seq,tool_call_id=item['call_id'])
            if item['name']=='submit_policy':submitted.append(args['expected_hash'])
    db.close()
    if submission['version'] not in submitted:raise ValueError('No explicit API submission')
    records={}
    for name,sha in submission['files'].items():
        file=folder/'source'/name
        if digest(file)!=sha or file.read_text()!=writes[name]['content']:raise ValueError('API authorship mismatch: '+str(file))
        records[name]={k:v for k,v in writes[name].items() if k!='content'}
    receipt=dict(passed=True,version=submission['version'],files=records,manual_output_edits=0,
        repository_authored_policy_content=0,prior_document_api_authored=True)
    atomic(folder/'authorship_audit.json',receipt);return receipt


def episode_report(folder,episode,account):
    """Render measurements and verbatim API reasons without authoring priors."""
    goals=('drawer_open','red_on_pad','blue_inside','success')
    first={key:None for key in goals}
    trace_path=folder/'trace.jsonl'
    if trace_path.exists():
        for line in trace_path.read_text().splitlines():
            row=json.loads(line)
            for key in goals:
                if first[key] is None and row['metrics'][key]:first[key]=row['step']
    lines=[f"# Trial {episode['seed']}",'',
        f"Measured success: {episode['success']}; physical steps: {episode['steps']}; status: {episode['status']}.",'',
        f"Final predicates: `{json.dumps(episode['final'])}`.",
        f"First observed goal steps: `{json.dumps(first)}`.",
        f"API requests: {account['calls']}; token usage: `{json.dumps(account['usage'])}`.",'',
        '## API-selected invocations','',
        '| End step | Policy | Drawer | Red | Blue |','| ---: | --- | --- | --- | --- |']
    step=0
    for call in episode['invocations']:
        step+=call['executed_steps'];g=call['task_goals']
        lines.append(f"| {step} | {call['policy_id']} | {g['drawer_open']} | {g['red_on_pad']} | {g['blue_inside']} |")
    lines+=['','## Selection reasons (verbatim API output)','']
    for index,call in enumerate(episode['invocations'],1):
        lines+=[f"### {index}. {call['policy_id']} ({call['executed_steps']} executed steps)",'',
            '> '+call['reason'].replace('\n','\n> '),'']
    lines+=['[Full invocation states and checkpoint hashes](invocations.json) · [Result](result.json) · [Raw API journal](api/journal.sqlite)',
        '', '![Final simulator frame](final.png)', '']
    (folder/'REPORT.md').write_text('\n'.join(lines))
    return first


def generate(cfg,verify=False):
    if evaluation_output(cfg)!=cfg['output']:return generate_reevaluation(cfg,verify)
    root=cfg['output'];rows=[];usage=Counter();calls=0;api_seconds=0.;check_updates=0
    for entry in catalog(cfg):
        folder=root/'policies'/entry['skill_id']/f"heuristic_{entry['heuristic_index']:02d}"
        row=dict(policy_id=entry['policy_id'],skill_id=entry['skill_id'],heuristic_index=entry['heuristic_index'],
            folder=str(folder),status='pending',updates=0,parameters=None,checkpoint_sha256=None)
        if (folder/'assignment.json').exists():row['status']='designing'
        if (folder/'submission.json').exists():
            sub=read(folder/'submission.json');row['status']='submitted';row['version']=sub['version']
            row['api_metadata']=read(folder/'source/pipeline.json')
            if verify:authorship(folder,sub)
        if (folder/'training/progress.json').exists():
            p=read(folder/'training/progress.json');row.update(status='training',updates=p['step'],loss=p['loss_mean'])
        if (folder/'training/result.json').exists():
            r=read(folder/'training/result.json')
            if verify and digest(folder/'training/last.pt')!=r['checkpoint_sha256']:raise ValueError('Checkpoint changed')
            row.update(status='trained',updates=r['optimizer_steps'],parameters=r['trainable_parameters'],
                checkpoint_sha256=r['checkpoint_sha256'],training_seconds=r['training_elapsed_seconds'])
        if (folder/'failure.json').exists():row.update(status='failed',failure=read(folder/'failure.json'))
        for path in (folder/'checks').glob('*/result.json') if (folder/'checks').exists() else []:
            check_updates+=read(path)['optimizer_steps']
        account=api_accounting(folder/'design/journal.sqlite');row['api']=account
        calls+=account['calls'];api_seconds+=account['seconds'];usage.update(account['usage']);rows.append(row)
    episodes=[]
    for seed in cfg['evaluation']['diagnostic_seeds']+cfg['evaluation']['seeds']:
        folder=root/'evaluation'/str(seed)
        account=api_accounting(folder/'api/journal.sqlite')
        if (folder/'result.json').exists():
            episode=read(folder/'result.json')
            episode['first_goal_steps']=episode_report(folder,episode,account)
            episode['api']=account;episodes.append(episode)
        calls+=account['calls'];api_seconds+=account['seconds'];usage.update(account['usage'])
    segmentation_account=api_accounting(cfg['dataset']/'_session/journal.sqlite') if cfg.get('experiment_version')=='M1_v2' else dict(calls=0,usage={},seconds=0.)
    calls+=segmentation_account['calls'];usage.update(segmentation_account['usage']);api_seconds+=segmentation_account['seconds']
    summary=dict(experiment_version=cfg.get('experiment_version','M1_v1'),
        policies=rows,policy_states=dict(Counter(r['status'] for r in rows)),episodes=episodes,
        independent_ID=dict(completed=sum(r['seed'] in cfg['evaluation']['seeds'] for r in episodes),
            successes=sum(r['success'] for r in episodes if r['seed'] in cfg['evaluation']['seeds']),planned=len(cfg['evaluation']['seeds'])),
        api_calls=calls,api_usage=dict(usage),api_seconds=api_seconds,interface_optimizer_updates=check_updates,
        framework_fixture_updates=0 if cfg.get('experiment_version')=='M1_v2' else 2,
        segmentation_api=segmentation_account,authorship_verified=verify,automatic_transport_retries=0)
    atomic(root/'summary.json',summary)
    with (root/'policies.csv').open('w') as f:
        keys=['policy_id','status','updates','parameters','checkpoint_sha256','folder']
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows({k:r[k] for k in keys} for r in rows)
    lines=[f"# {summary['experiment_version']}: API-authored prior Diffusion Policies",'',
        f"Source: the API segmentation with {len({r['skill_id'] for r in rows})} skills and {len(rows)} heuristics. Each policy has an independent source package, prior document and checkpoint.",
        '', f"Training completed: {summary['policy_states'].get('trained',0)}/{len(rows)} policies. Independent ID success: {summary['independent_ID']['successes']}/{summary['independent_ID']['completed']} completed trials ({len(cfg['evaluation']['seeds'])} planned).",
        '', '## Policies','', '| Policy | State | Updates | Parameters | API prior document |','| --- | --- | ---: | ---: | --- |']
    for r in rows:
        doc=Path(r['folder'])/'source/PRIOR.md'
        lines.append(f"| {r['policy_id']} | {r['status']} | {r['updates']} | {r['parameters'] or '—'} | "+(f'[PRIOR.md]({doc})' if doc.exists() else 'pending')+' |')
    lines+=['','## Inference-time API trials','','| Seed | Scope | Success | Drawer | Red | Blue | Steps | Invocations |','| --- | --- | --- | --- | --- | --- | ---: | ---: |']
    for e in episodes:
        scope='training-reset diagnostic' if e['seed'] in cfg['evaluation']['diagnostic_seeds'] else 'independent ID'
        g=e['final']
        lines.append(f"| [{e['seed']}](evaluation/{e['seed']}/REPORT.md) | {scope} | {e['success']} | {g['drawer_open']} | {g['red_on_pad']} | {g['blue_inside']} | {e['steps']} | {len(e['invocations'])} |")
    lines+=['',f"Independent ID: {summary['independent_ID']['successes']} successes in {summary['independent_ID']['completed']} completed trials; {len(cfg['evaluation']['seeds'])} planned.",
        '', 'Success uses only the agreed simultaneous drawer-open, red-on-pad and blue-inside predicates. This does not retroactively change frozen M0 results.',
        '', '## Accounting','',f"API calls: {calls}; input tokens: {usage['input_tokens']:,} (cached {usage['cached_input_tokens']:,}); output tokens: {usage['output_tokens']:,}; API request seconds: {api_seconds:.1f}.",
        f"Completed interface-check updates: {check_updates}; synthetic framework fixture updates: {summary['framework_fixture_updates']}; segmentation API requests included: {segmentation_account['calls']}; no automatic transport retries. Partial failures remain separately recorded. Dollar invoice unavailable.",
        '',f"Authorship audit performed in this report: {verify}. The audit matches policy code, pipeline metadata and PRIOR.md against raw API write_file responses and explicit submission hashes.",
        '', '[Machine-readable summary](summary.json) · [Policy table](policies.csv)', '']
    if cfg.get('experiment_version')=='M1_v2':
        lines+=['## Normalization and handoff','',
            'Every policy uses the same full-training-demonstration observation/action normalizer. Each package includes API-authored source/HANDOFF.json and measured training/handoff_evidence.json; training/policy_context.json and the checkpoint preserve their hashes and identity.',
            '', '[Shared normalization](normalization.json) · [API segmentation](segmentation/REPORT.md) · [M1_v1 checkpoint deletion receipt](setup/v1_checkpoint_deletion_receipt.json)', '']
    (root/'REPORT.md').write_text('\n'.join(lines));return summary


def generate_reevaluation(cfg,verify=False):
    """Incremental evaluation costs/results; never rewrite the parent run."""
    root=evaluation_output(cfg);parent=cfg['output']
    frozen=read(root/'library_freeze.json');original=read(parent/'library_freeze.json')
    if digest(parent/'library_freeze.json')!=frozen['library']['parent_library_freeze_sha256']:
        raise ValueError('Parent library freeze changed')
    if verify:
        from .deploy import library
        current={k:dict(version=p['version'],checkpoint_sha256=p['checkpoint_sha256']) for k,p in library(cfg).items()}
        if current!=frozen['library']['policies']:raise ValueError('Reevaluated policy library changed')
    calls=0;seconds=0.;usage=Counter();episodes=[];failures=[];rows=[]
    baseline_steps=original['library']['evaluation']['max_steps']
    for seed in cfg['evaluation']['diagnostic_seeds']+cfg['evaluation']['seeds']:
        folder=root/'evaluation'/str(seed);account=api_accounting(folder/'api/journal.sqlite')
        calls+=account['calls'];seconds+=account['seconds'];usage.update(account['usage'])
        old=read(parent/'evaluation'/str(seed)/'result.json')
        row=dict(seed=seed,baseline_success=old['success'],baseline_steps=old['steps'],status='pending')
        if (folder/'result.json').exists():
            episode=read(folder/'result.json');episode['first_goal_steps']=episode_report(folder,episode,account)
            episode['api']=account;episodes.append(episode)
            row.update(success=episode['success'],steps=episode['steps'],status=episode['status'],
                success_by_baseline_budget=episode['success'] and episode['steps']<=baseline_steps,
                success_after_baseline_budget=episode['success'] and episode['steps']>baseline_steps)
        elif (folder/'failure.json').exists():
            failure=dict(seed=seed,**read(folder/'failure.json'));failures.append(failure);row['status']='execution_failure'
        rows.append(row)
    ids=[r for r in rows if r['seed'] in cfg['evaluation']['seeds'] and 'success' in r]
    summary=dict(experiment_version=cfg.get('experiment_version'),scope='same_seed_budget_reevaluation',
        policy_states={'reused_frozen':len(frozen['library']['policies'])},parent_output=str(parent),
        library_hash=frozen['library_hash'],baseline_max_steps=baseline_steps,max_steps=cfg['evaluation']['max_steps'],
        episodes=episodes,comparison=rows,failures=failures,
        independent_ID=dict(completed=len(ids),successes=sum(r['success'] for r in ids),planned=len(cfg['evaluation']['seeds'])),
        success_by_baseline_budget=sum(r['success_by_baseline_budget'] for r in ids),
        success_after_baseline_budget=sum(r['success_after_baseline_budget'] for r in ids),
        api_calls=calls,api_usage=dict(usage),api_seconds=seconds,additional_training_updates=0,
        automatic_transport_retries=0,library_verified=verify)
    atomic(root/'summary.json',summary)
    lines=[f"# M1_v2: {cfg['evaluation']['max_steps']}-step budget reevaluation",'',
        f"The same {len(frozen['library']['policies'])} frozen policies and initial seeds are reused. Additional training updates: 0.",
        'These are new episodes from reset, not physical continuation of the old traces. The inference API can plan differently when given more remaining steps.',
        '',f"ID successes: {summary['independent_ID']['successes']}/{len(ids)} completed ({len(cfg['evaluation']['seeds'])} planned). Of those, {summary['success_by_baseline_budget']} succeeded within the first {baseline_steps} steps, and {summary['success_after_baseline_budget']} required later steps.",
        '', '| Seed | Scope | Original success | New success | New steps | Within original budget | Status |',
        '| --- | --- | --- | --- | ---: | --- | --- |']
    if cfg['evaluation'].get('feedback_protocol'):
        lines[0]=f"# M1_v2: API stopping conditions with private {cfg['evaluation']['max_steps']}-step limit"
        lines[3]='These are new episodes from reset. The total and remaining physical budget are hidden from API input. This is a system revision with a new prompt, API-authored stop conditions and bounded context; it is not a single-factor budget comparison.'
    for r in rows:
        scope='training reset' if r['seed'] in cfg['evaluation']['diagnostic_seeds'] else 'same-seed ID retest'
        lines.append(f"| [{r['seed']}](evaluation/{r['seed']}/REPORT.md) | {scope} | {r['baseline_success']} | {r.get('success','pending')} | {r.get('steps','—')} | {r.get('success_by_baseline_budget','—')} | {r['status']} |")
    lines+=['','## Interpretation','',
        'Success remains the simultaneous drawer-open, red-on-pad, blue-inside conjunction. No terminal release, speed or hold condition is added. A new success beyond the old budget shows that this new trajectory needed additional execution time; it does not prove that the old episode would have followed the same continuation.',
        '', '## Incremental accounting','',
        f"Inference API requests: {calls}; input tokens: {usage['input_tokens']:,} (cached {usage['cached_input_tokens']:,}); output tokens: {usage['output_tokens']:,}; cumulative request seconds: {seconds:.1f}. Original design/training costs remain in the parent report and are not charged again. Dollar invoice unavailable.",
        f"Execution failures: {len(failures)}. No automatic transport retries. Current library hashes verified: {verify}.",
        '', '[Original 1500-step report](../REPORT.md) · [Machine-readable comparison](summary.json) · [New freeze](library_freeze.json) · [Budget amendment](setup/amendment.json)', '']
    if cfg['evaluation'].get('feedback_protocol'):
        lines+=['## Feedback protocol','',
            'The API supplies each invocation duration, numeric stopping conditions and notebook. The executor checks those exact conditions after every physical action and returns control at their first match; it never selects a successor or invents a skill threshold. All state/action trajectories and original API outputs remain saved.',
            'API requests retain the original initial message, latest complete read of every policy, latest invocation/notebook and recent complete exchanges. Older invocations are available through read_invocation. No API text is rewritten. The private API request ceiling is recorded in the configuration and costs are incremental.', '']
    (root/'REPORT.md').write_text('\n'.join(lines));return summary
