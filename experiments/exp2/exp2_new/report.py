"""Matched completed subsets, raw API provenance, action replay and cost ledger."""
import csv
import html
import json
from pathlib import Path
import sqlite3

import numpy as np
from appl.io import atomic, digest, read
from experiments.exp2.binary_gripper.common import execute_action
from experiments.exp2.binary_gripper.report import video
from experiments.exp2.analysis.audit_scaleup import audit_api, budget_fields
from appl.scaleup.report import audit_episode
from .common import BASE, TASKS, METHOD, config, episode_root, verify
from .budget import NANO


def audit(cell):
    plan=verify(cell['task']);root=episode_root(cell)
    result=read(root/'result.json') if (root/'result.json').exists() else None
    if not (root/'initial_state.json').exists():
        atomic(root/'audit.json',dict(passed=False,reason='No initialized physical state',completed=False))
        return
    contract=read(config(cell['task'])['completion_contract'])
    initial=read(root/'initial_state.json');environment=read(root/'environment.json')
    trace=[json.loads(line) for line in (root/'trace.jsonl').read_text().splitlines()] if (root/'trace.jsonl').exists() else []
    for row in trace:
        expected=execute_action(row['raw_action'],np.asarray(environment['action_low'],np.float32),
                                np.asarray(environment['action_high'],np.float32))
        if not np.array_equal(expected,np.asarray(row['action'],np.float32)):
            raise ValueError('Executed action differs from the declared gripper correction')
    for reference in cell['baseline_roots'].values():
        if initial!=read(Path(reference)/'initial_state.json'):raise ValueError('Paired reset mismatch')
    physical=audit_episode(root,contract,result)
    db=sqlite3.connect((root/'api/journal.sqlite').resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    statuses={};usage=[]
    for seq,status,request_text,response_text in db.execute('SELECT * FROM api ORDER BY seq'):
        request=json.loads(request_text);response=json.loads(response_text) if response_text else None
        if (request['model'],request['reasoning']['effort'],request.get('service_tier'))!=('gpt-6-astra','xhigh','default'):
            raise ValueError('Actual wire model/effort/tier differs')
        if request!=read(root/'api/api'/f'{seq:04d}.request.json'):raise ValueError('Wire/journal request mismatch')
        for item in request['input']:
            if (item.get('role')=='user' or item.get('type')=='function_call_output') and budget_fields(item):
                raise ValueError('Executor-only physical budget disclosed')
        if '5000' in request['instructions'] or '5,000' in request['instructions']:
            raise ValueError('Physical cap disclosed in instructions')
        if status=='consumed':
            if (response.get('model'),(response.get('reasoning') or {}).get('effort'))!=('gpt-6-astra','xhigh'):
                raise ValueError('Consumed response identity mismatch')
            wire=read(root/'api/transport'/f'{seq:04d}'/'response.json')
            if response!=wire['response']:raise ValueError('API response changed')
        if response and response.get('usage'):usage.append(response['usage'])
        statuses[status]=statuses.get(status,0)+1
    db.close()
    if result is not None:
        literal=audit_api(root,contract,trace,initial,result,plan['tasks'][cell['task']])
    else:
        literal=dict(full_terminal_audit=False,reason='Interrupted attempt retained; no terminal task score')
    for worker in (root/'workers').glob('*'):
        if read(worker/'closed.json')['returncode']!=0:raise ValueError('Policy worker exit failure')
        if read(worker/'enforcement.json')['network'] is not False:raise ValueError('Policy worker network enabled')
    value=dict(passed=True,completed=result is not None,physical=physical,api=literal,request_statuses=statuses,
               usage=usage,all_actions_match_decoder=True,paired_initial_equal=True,
               result_sha256=digest(root/'result.json') if result else None,
               trace_sha256=digest(root/'trace.jsonl') if trace else None,
               learned_optimizer_updates=0)
    atomic(root/'audit.json',value)


def write_csv(path,rows):
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def generate():
    plan=verify();baseline=read(BASE/'baseline_subset.json');rows=[];cards=[]
    for b in baseline:
        root=Path(b['root'])
        if digest(root/'result.json')!=b['result_sha256'] or digest(root/'initial_state.json')!=b['initial_sha256']:
            raise ValueError('Reused baseline changed')
        rows.append(dict(task=b['task'],condition=b['condition'],seed=b['seed'],method=b['method'],
                         state='completed',success=b['result']['success'],steps=b['result']['steps'],root=str(root)))
    complete_cells=set();unknown=0
    for cell in plan['cells']:
        root=episode_root(cell);result=None
        proof=read(root/'audit.json') if (root/'audit.json').exists() else {}
        process=BASE/'supervisor/evaluate'/str(cell['index'])/'process_result.json'
        completed=(proof.get('passed') and proof.get('completed') and process.exists()
                   and read(process)['returncode']==0 and (root/'replay_provenance.json').exists()
                   and not (root/'cleanup_failure.json').exists())
        if completed:
            result=read(root/'result.json')
            if digest(root/'result.json')!=proof['result_sha256']:raise ValueError('Audited result changed')
            if proof['trace_sha256'] and digest(root/'trace.jsonl')!=proof['trace_sha256']:raise ValueError('Audited trace changed')
            complete_cells.add((cell['task'],cell['condition'],cell['seed']))
        state='completed' if completed else 'interrupted' if (root/'interruption.json').exists() else 'running' if root.exists() else 'not_started'
        unknown+=int(state=='interrupted')
        rows.append(dict(task=cell['task'],condition=cell['condition'],seed=cell['seed'],method=METHOD,
                         state=state,success=result['success'] if result else None,
                         steps=result['steps'] if result else None,root=str(root)))
        if (root/'replay.mp4').exists():
            proof_video=read(root/'replay_provenance.json')
            if digest(root/'replay.mp4')!=proof_video['video_sha256']:raise ValueError('Video changed')
            title=f"{cell['task']} / {cell['condition']} / {cell['seed']}: {state}"
            if result:title+=f" — {'success' if result['success'] else 'failure'}, {result['steps']} steps"
            url=str((root/'replay.mp4').relative_to(BASE))
            cards.append(f'<article><h3>{html.escape(title)}</h3><video controls preload="none" src="{url}"></video></article>')
    groups=[]
    for task in TASKS+['all']:
        for condition in ('ID','OOD'):
            for method in ('naive_DP','SinglePrior_6_xhigh',METHOD):
                selected=[r for r in rows if (task=='all' or r['task']==task) and r['condition']==condition and r['method']==method]
                done=[r for r in selected if r['state']=='completed']
                matched=[r for r in done if (r['task'],r['condition'],r['seed']) in complete_cells]
                groups.append(dict(task=task,condition=condition,method=method,planned=len(selected),completed=len(done),
                                   successes=sum(r['success'] for r in done),matched_completed=len(matched),
                                   matched_successes=sum(r['success'] for r in matched)))
    ledger=read(BASE/'budget/ledger.json') if (BASE/'budget/ledger.json').exists() else None
    costs=None
    if ledger:
        records=ledger['records'];known=sum(r.get('settled_nano_usd',0) for r in records)
        held=sum(r['reserved_nano_usd'] for r in records if 'settled_nano_usd' not in r)
        costs=dict(cap_usd=ledger['cap_nano_usd']/NANO,usage_based_conservative_usd=known/NANO,
                   unknown_charge_reservations_usd=held/NANO,remaining_unreserved_usd=(ledger['cap_nano_usd']-known-held)/NANO,
                   generation_attempts=len(records),stopped=ledger['stopped'],
                   input_tokens=sum((r.get('usage') or {}).get('input_tokens',0) for r in records),
                   output_tokens=sum((r.get('usage') or {}).get('output_tokens',0) for r in records),
                   note='Usage estimate at frozen standard rates; missing cache-write detail uses the higher write tariff. Not an invoice.')
    atomic(BASE/'results.json',dict(study='Exp2_new',planned_APPL=150,completed_APPL=len(complete_cells),
           interrupted_APPL=unknown,pending_APPL=150-len(complete_cells)-unknown,costs=costs,groups=groups,
           learned_optimizer_updates=0,segmentation_calls=0,policy_design_calls=0))
    write_csv(BASE/'episodes.csv',rows);write_csv(BASE/'summary.csv',groups)
    lines=['# Exp2_new: corrected-gripper APPL deployment','',
           f'APPL completed: **{len(complete_cells)}/150**. Interrupted: **{unknown}**. Unstarted/running: **{150-len(complete_cells)-unknown}**.','',
           '## Protocol','',
           'Five tasks, the first 15 original ID and 15 position-OOD layouts per task. Reuse all 36 frozen GPT-6 Astra/xhigh policies and all original API-authored priors/handoff documents. No new segmentation, candidate generation or training. The original English prompt, policy tools, observation history/reset rules, DDPM100, eight-step action chunks and hidden 5,000-step physical cap are retained.',
           'The executor maps the predicted gripper sign to full open/close (+1 at zero), matching the corrected naive-DP and single-prior baselines. The seven arm targets retain native clipping. Raw predictions and executed actions are recorded separately. A new credential uses the official OpenAI endpoint and standard service tier. The API still selects every policy, duration, numerical stop condition and finish decision; its outputs are retained verbatim.',
           'The two baselines reuse 300 already validated episodes from the completed 600-episode corrected study. No baseline is rerun. Original artifacts stay at their immutable paths, with a comparison link in this directory.',
           '','## Results on the preselected 15-layout groups','',
           '| Task | Split | Method | Successes / completed | Planned | Matched APPL subset |',
           '| --- | --- | --- | ---: | ---: | ---: |']
    for g in groups:
        lines.append(f"| {g['task']} | {g['condition']} | {g['method']} | {g['successes']}/{g['completed']} | {g['planned']} | {g['matched_successes']}/{g['matched_completed']} |")
    lines+=['','The matched column compares only identical layouts with a completed new APPL outcome. Budget/provider interruptions are unknown outcomes, never task failures. These are previously examined layouts and one frozen training seed, not a fresh confirmatory test. Budget-limited completion can be selective; do not present partial denominators as 15 completed trials.',
            '','## Cost and execution','',
            json.dumps(costs,indent=2) if costs else 'No authorized monetary ledger yet; paid deployment has not started.',
            '','Costs cover only this new runtime deployment. Historical segmentation/training/design costs are not billed again. Each generation request has a token-count receipt, worst-case reservation and provider usage receipt where available. No retries, fallback credentials, model substitutions or rewritten API outputs.',
            '','## Evidence','',
            '- [Frozen plan and original policy/input hashes](plan.json)',
            '- [Explicit baseline subset](baseline_subset.json)',
            '- [Per-layout status](episodes.csv)',
            '- [Machine-readable results](results.json)',
            '- [Videos, including interrupted prefixes](replays.html)',
            '- [Complete original corrected baselines](corrected_baselines/REPORT.md)']
    (BASE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (BASE/'START_HERE.md').write_text('# Exp2_new\n\nStart with [the report](REPORT.md), [per-layout results](episodes.csv) and [videos](replays.html).\n\nOnly completed, audited APPL outcomes count in success denominators. Historical artifacts remain immutable.\n')
    (BASE/'replays.html').write_text('<!doctype html><meta charset="utf-8"><title>Exp2_new replays</title><style>body{font:16px sans-serif;max-width:1100px;margin:auto}article{display:inline-block;vertical-align:top;width:48%;margin:1%}video{width:100%;image-rendering:pixelated}</style><h1>Exp2_new</h1><p>Source frames every 20 control steps, played at approximately 6× simulation speed. Interrupted prefixes are not task outcomes.</p>'+''.join(cards))
    print(dict(completed_APPL=len(complete_cells),interrupted=unknown,costs=costs),flush=True)
