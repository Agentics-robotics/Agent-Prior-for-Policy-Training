"""Finalize a budget-limited Exp2_new run without sending any API request."""
from collections import Counter
import csv
import json
from pathlib import Path
import sqlite3
import subprocess
import time

from appl.io import atomic, digest, read
from appl.prior_policies.feedback import measurements, matches, update_ranges
from appl.scaleup.tasks import measure
from experiments.exp2.exp2_new.common import BASE, TASKS, config, verify, episode_root
from experiments.exp2.exp2_new.budget import NANO, usage_cost
from experiments.exp2.exp2_new.report import generate


def partial_invocations(root, cell, frozen):
    """Audit completed physical invocations before the terminal unsent/error call."""
    contract=read(config(cell['task'])['completion_contract'])
    state=read(root/'initial_state.json')
    trace=[json.loads(s) for s in (root/'trace.jsonl').read_text().splitlines()] if (root/'trace.jsonl').exists() else []
    calls=read(root/'invocations.json') if (root/'invocations.json').exists() else []
    db=sqlite3.connect((root/'api/journal.sqlite').resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    executed=[(json.loads(a),json.loads(r)) for a,r in db.execute(
        "SELECT args,result FROM tools WHERE name='invoke_policy' AND status='completed' ORDER BY rowid")
        if 'error' not in json.loads(r)]
    db.close()
    assert len(calls)==len(executed)
    pos=0
    for call,(args,output) in zip(calls,executed):
        for key in ('policy_id','reason','stop_when','notebook'):assert args[key]==call[key]
        assert args['steps']==call['requested_steps'] and call['before']==state
        policy=frozen['policies'][call['policy_id']]
        assert policy['version']==call['source_version'] and policy['checkpoint_sha256']==call['checkpoint_sha256']
        count=call['executed_steps'];chunk=trace[pos:pos+count]
        assert 1<=count<=min(args['steps'],5000-pos) and len(chunk)==count
        initial=measurements(state,measure(state,contract));ranges={};update_ranges(ranges,initial,pos)
        for index,row in enumerate(chunk):
            assert row['policy_id']==args['policy_id']
            current=measurements(row['state'],row['metrics']);update_ranges(ranges,current,row['step'])
            matched=matches(args['stop_when'],current,initial)
            if index+1<count:assert not matched and not row['metrics']['success']
        pos+=count;last=chunk[-1];state=last['state']
        reason='task_success' if last['metrics']['success'] else 'api_condition' if matched else 'executor_limit' if pos==5000 else 'requested_duration'
        assert call['stop_reason']==reason and call['matched_stop_rules']==matched
        if reason=='requested_duration':assert count==args['steps']
        assert call['after']==state and call['task_goals']==last['metrics'] and call['metric_ranges']==ranges
        assert output['last_invocation']['invocation_id']==call['invocation_id']
    assert pos==len(trace)
    return dict(passed=True,invocations=len(calls),physical_steps=pos,
                API_arguments_unchanged=True,terminal_task_outcome_known=False)


def main():
    supervisor=read(BASE/'supervisor/evaluate/completion.json')
    assert supervisor['all_workers_exited']
    plan=verify()
    for task in TASKS:verify(task)
    generate()
    summary=read(BASE/'results.json');ledger=read(BASE/'budget/ledger.json')
    assert ledger['stopped'] or summary['completed_APPL']==150
    records={r['key']:r for r in ledger['records']}
    assert len(records)==len(ledger['records'])
    matched=set();statuses=Counter();http=Counter();totals=Counter();cost_rows=[];episodes=[];workers=0
    for cell in plan['cells']:
        root=episode_root(cell)
        if not root.exists():continue
        job=BASE/'supervisor/evaluate'/str(cell['index'])
        process=read(job/'process_result.json')
        audit=read(root/'audit.json');assert audit['passed']
        result=read(root/'result.json') if (root/'result.json').exists() else None
        if result:
            assert process['returncode']==0 and audit['completed']
            assert digest(root/'result.json')==audit['result_sha256']
        else:
            assert process['returncode']!=0 and (root/'interruption.json').exists()
            atomic(root/'interrupted_invocation_audit.json',partial_invocations(root,cell,plan['tasks'][cell['task']]))
        if audit['trace_sha256']:assert digest(root/'trace.jsonl')==audit['trace_sha256']
        movie=read(root/'replay_provenance.json');assert digest(root/'replay.mp4')==movie['video_sha256']
        for name,sha in movie['frames'].items():assert digest(root/name)==sha
        for worker in (root/'workers').glob('*'):
            assert read(worker/'closed.json')['returncode']==0
            assert read(worker/'enforcement.json')['network'] is False
            workers+=1
        db=sqlite3.connect((root/'api/journal.sqlite').resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
        cost=0;calls=0;local_tokens=Counter();finish=[]
        for seq,status,request_text,response_text in db.execute('SELECT * FROM api ORDER BY seq'):
            statuses[status]+=1;request=json.loads(request_text)
            assert (request['model'],request['reasoning']['effort'],request['service_tier'])==('gpt-6-astra','xhigh','default')
            assert request==read(root/'api/api'/f'{seq:04d}.request.json')
            transport=root/'api/transport'/f'{seq:04d}'
            attempted=(transport/'generation_attempt.json').exists()
            key=str((root/'api').resolve().relative_to(BASE.resolve()))+f'/{seq:04d}'
            if not attempted:
                assert key not in records and status=='not_sent'
                continue
            assert key in records and key not in matched
            matched.add(key);record=records[key];calls+=1
            count=read(transport/'count_response.json')
            assert count['http_status']==200 and count['response']['input_tokens']==record['input_tokens_counted']
            if (transport/'response.json').exists():
                wire=read(transport/'response.json');http[str(wire['http_status'])]+=1
                body=wire['response']
                if response_text:assert json.loads(response_text)==body
                if status=='consumed':
                    assert body['model']=='gpt-6-astra' and body['reasoning']['effort']=='xhigh'
                usage=body.get('usage')
                if usage:
                    assert usage==record['usage']
                    amount,_=usage_cost(usage);assert amount==record['settled_nano_usd']
                    cost+=amount
                    detail=usage.get('input_tokens_details') or {}
                    local_tokens.update(input_tokens=usage['input_tokens'],output_tokens=usage['output_tokens'],
                        cached_input_tokens=detail.get('cached_tokens',0),cache_write_tokens=detail.get('cache_write_tokens',0),
                        reasoning_tokens=(usage.get('output_tokens_details') or {}).get('reasoning_tokens',0))
        finish=[json.loads(row[0])['reason'] for row in db.execute("SELECT args FROM tools WHERE name='finish' AND status='completed'")]
        db.close();totals.update(local_tokens)
        row=dict(task=cell['task'],condition=cell['condition'],seed=cell['seed'],completed=bool(result),
            success=result['success'] if result else None,steps=result['steps'] if result else read(root/'interruption.json')['steps'],
            generation_attempts=calls,reported_cost_usd=cost/NANO,**local_tokens,API_finish_reason=finish[-1] if finish else '')
        cost_rows.append(row);episodes.append(dict(index=cell['index'],root=str(root),passed=True,completed=bool(result)))
    assert matched==set(records)
    settled=sum(r.get('settled_nano_usd',0) for r in records.values())
    held=sum(r['reserved_nano_usd'] for r in records.values() if 'settled_nano_usd' not in r)
    assert settled+held<=ledger['cap_nano_usd']
    occupancy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
    own=[]
    for line in occupancy.splitlines():
        if not line.strip():continue
        pid=line.split(',')[0].strip();path=Path('/proc')/pid/'cmdline'
        try:argv=path.read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except (FileNotFoundError,PermissionError):continue
        if 'experiments.exp2.exp2_new' in argv or str(BASE) in argv or str(BASE.resolve()) in argv:own.append(pid)
    assert not own
    with (BASE/'episode_costs.csv').open('w') as stream:
        fields=list(dict.fromkeys(k for r in cost_rows for k in r))
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(cost_rows)
    costs=dict(reported_usd=settled/NANO,unknown_charge_reservations_usd=held/NANO,
               cap_usd=ledger['cap_nano_usd']/NANO,authorized_SGD=100,**totals)
    value=dict(passed=True,reviewed=time.time(),completed_APPL=summary['completed_APPL'],
        interrupted_APPL=summary['interrupted_APPL'],unstarted_APPL=summary['pending_APPL'],planned_APPL=150,
        all_150_complete=summary['completed_APPL']==150,stopping_reason=ledger['stopped'],
        generation_attempts=len(records),request_statuses=dict(statuses),HTTP_statuses=dict(http),costs=costs,
        all_scheduled_parent_processes_exited=True,closed_policy_workers=workers,remaining_own_GPU_processes=own,
        peak_active_physical_devices=1,physical_GPU=7,new_training_updates=0,transport_retries=0,
        original_frozen_inputs_unchanged=True,plan_sha256=digest(BASE/'plan.json'),
        result_sha256=digest(BASE/'results.json'),analysis_source_sha256=digest(__file__),episodes=episodes,
        zero_request_setup_incident=str(BASE/'incidents/path_resolution_20260920/diagnosis.json'))
    atomic(BASE/'completion.json',value)
    lines=['# Exp2_new: budget-limited execution and findings','',
        f"Completed outcomes: **{summary['completed_APPL']}/150**; interrupted: **{summary['interrupted_APPL']}**; unstarted: **{summary['pending_APPL']}**.",
        '',f"Stop reason: `{ledger['stopped']}`.",
        '',f"Recorded usage cost: **USD {settled/NANO:.6f}**. Unknown-charge reservations: **USD {held/NANO:.6f}**. The user authorized SGD 100; the ledger cap was USD 78, rounded below the retrieved conversion. These are token-tariff estimates, not the account invoice.",
        '', '## Matched completed layouts','',
        'Use exactly the layouts with completed new APPL results for comparisons. Budget interruption is not a task failure. The frozen 15-per-task/split plan remains incomplete unless every listed outcome has been validated.',
        '', '| Split | Method | Successes / matched completed |','| --- | --- | ---: |']
    for g in summary['groups']:
        if g['task']=='all':lines.append(f"| {g['condition']} | {g['method']} | {g['matched_successes']}/{g['matched_completed']} |")
    lines+=['','## Per-episode cost','',
        '| Task | Split | Seed | Outcome | Steps | Generation attempts | Usage cost (USD) |',
        '| --- | --- | ---: | --- | ---: | ---: | ---: |']
    for r in cost_rows:
        outcome='success' if r['success'] else 'failure' if r['completed'] else 'interrupted / unknown'
        lines.append(f"| {r['task']} | {r['condition']} | {r['seed']} | {outcome} | {r['steps']} | {r['generation_attempts']} | {r['reported_cost_usd']:.6f} |")
    lines+=['','## Interpretation','',
        'All 36 policy packages were reused without retraining or new design calls. The API selected all deployed policies, invocation durations, numeric stop conditions and finish decisions. The executor only applied the declared binary gripper decoder and original geometric predicates.',
        'The expensive recovery episodes retain all their attempts. A high request count is not evidence that a learned policy can correct the remaining placement error. Conversely, early failures do not establish that the 5,000-step physical cap was too short: the raw finish decisions and actual step counts identify who ended each episode.',
        'These are small, budget-limited, previously examined layout subsets from one training seed. Do not claim the full 15-ID/15-OOD comparison is complete or infer statistical superiority from these partial rates.',
        '', '## Preserved setup incident','',
        'The initial local path-resolution error occurred before any HTTP request or robot action. The old source, plan and zero-step attempt are retained. The one-line canonical-path repair passed its regression test (12 framework tests total); original policy/prompt/data/seed/checkpoint inputs remained unchanged. This was not a transport retry.',
        '', '## Evidence','',
        '- [Result table](REPORT.md)', '- [Completion and accounting audit](completion.json)',
        '- [Every episode cost and original API finish reason](episode_costs.csv)',
        '- [Request-level budget ledger](budget/ledger.json)',
        '- [Budget authorization and conversion](budget/authorization.json)',
        '- [All videos, including interrupted prefixes](replays.html)',
        '- [Preserved local setup incident](incidents/path_resolution_20260920/diagnosis.json)']
    (BASE/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
    (BASE/'START_HERE.md').write_text('# Exp2_new\n\nStart with [results](REPORT.md), [costs and interpretation](ANALYSIS.md), [completion audit](completion.json) and [videos](replays.html).\n\nThis is a budget-limited run. Use the explicit completed, interrupted and unstarted counts; incomplete cells are not task failures.\n')
    print({k:value[k] for k in ['passed','completed_APPL','interrupted_APPL','unstarted_APPL','generation_attempts','costs','remaining_own_GPU_processes']},flush=True)


if __name__=='__main__':main()
