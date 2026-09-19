"""Read-only verification of completed scale-up executions and API decisions.

This analysis runs outside the frozen executor. It never changes an episode,
an API output, a policy, or a test schedule.
"""
import argparse
import json
import sqlite3
from collections import Counter

from appl.io import ROOT, atomic, digest, read
from appl.prior_policies.feedback import measurements, matches, update_ranges
from appl.scaleup.protocol import BASE, NAMES, config
from appl.scaleup.report import audit_episode
from appl.scaleup.tasks import measure


def budget_fields(value, path=''):
    found=[]
    if isinstance(value,dict):
        for key,item in value.items():
            if key in ('max_steps','remaining_steps','remaining_budget','total_step_budget'):
                found.append(path+'.'+key)
            found.extend(budget_fields(item,path+'.'+key))
    elif isinstance(value,list):
        for i,item in enumerate(value):found.extend(budget_fields(item,path+f'[{i}]'))
    elif isinstance(value,str) and value[:1] in ('{','['):
        try:decoded=json.loads(value)
        except json.JSONDecodeError:return found
        found.extend(budget_fields(decoded,path))
    return found


def audit_api(root,contract,trace,initial,result,frozen,allow_terminal_http_failure=False):
    database=(root/'api/journal.sqlite').resolve()
    wal=database.with_name(database.name+'-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError('Completed journal has an uncheckpointed WAL; a durable snapshot is required')
    # Only exited episode processes reach this audit. With no pending WAL data,
    # immutable mode reads the durable database without creating shared-memory
    # locks or writing to the original journal directory.
    db=sqlite3.connect('file:'+str(database)+'?mode=ro&immutable=1',uri=True)
    db.row_factory=sqlite3.Row
    states=Counter();requests=0;last_api_status=None
    for row in db.execute('SELECT * FROM api ORDER BY seq'):
        states[row['status']]+=1;requests+=1;request=json.loads(row['request'])
        last_api_status=row['status']
        # Inspect framework-visible inputs. API-generated arguments/reasoning are
        # retained in the journal and are not treated as framework disclosures.
        for item in request['input']:
            if item.get('role')=='user' or item.get('type')=='function_call_output':
                if budget_fields(item):raise ValueError('Private budget field in API input')
        if '5000' in request['instructions'] or '5,000' in request['instructions']:
            raise ValueError('Private physical cap in instructions')
    executed=[]
    for row in db.execute("SELECT * FROM tools WHERE name='invoke_policy' AND status='completed' ORDER BY rowid"):
        output=json.loads(row['result'])
        if 'error' not in output:
            executed.append((json.loads(row['args']),output))
    db.close()
    calls=read(root/'invocations.json') if (root/'invocations.json').exists() else []
    if len(executed)!=len(calls):raise ValueError('API invocation receipts differ from recorded calls')
    position=0;previous=initial;early=0
    for (args,output),call in zip(executed,calls):
        for key in ('policy_id','reason','stop_when','notebook'):
            if args[key]!=call[key]:raise ValueError('API invocation output was rewritten: '+key)
        if args['steps']!=call['requested_steps'] or not 1<=args['steps']<=300:
            raise ValueError('Invocation duration mismatch')
        if call['before']!=previous:raise ValueError('Invocation starts from a different observed state')
        selected=frozen['policies'][call['policy_id']]
        if selected['version']!=call['source_version'] or selected['checkpoint_sha256']!=call['checkpoint_sha256']:
            raise ValueError('Invocation did not use the frozen selected policy')
        count=call['executed_steps'];chunk=trace[position:position+count]
        if len(chunk)!=count or not 1<=count<=min(args['steps'],5000-position):
            raise ValueError('Invalid executed action count')
        start=measurements(previous,measure(previous,contract));ranges={};update_ranges(ranges,start,position)
        for index,row in enumerate(chunk):
            if row['policy_id']!=args['policy_id']:raise ValueError('Actions came from a different policy')
            current=measurements(row['state'],row['metrics']);update_ranges(ranges,current,row['step'])
            matched=matches(args['stop_when'],current,start)
            if index+1<len(chunk) and (matched or row['metrics']['success']):
                raise ValueError('Executor continued past the literal stop condition')
        position+=count;last=chunk[-1];previous=last['state']
        reason=('task_success' if last['metrics']['success'] else 'api_condition' if matched else
                'executor_limit' if position==5000 else 'requested_duration')
        if reason!=call['stop_reason'] or matched!=call['matched_stop_rules']:
            raise ValueError('Stop reason differs from literal per-step replay')
        if reason=='requested_duration' and count!=args['steps']:raise ValueError('Unexplained short invocation')
        if call['after']!=previous or call['task_goals']!=last['metrics'] or call['metric_ranges']!=ranges:
            raise ValueError('Invocation boundary/metric feedback differs from physical trace')
        if output['last_invocation']['invocation_id']!=call['invocation_id']:
            raise ValueError('API received a different invocation identifier')
        early+=int(reason=='api_condition')
    if position!=len(trace):raise ValueError('Some physical actions lack an API-selected invocation')
    if allow_terminal_http_failure:
        if states['http_failed']!=1 or last_api_status!='http_failed' or set(states)-{'consumed','http_failed'}:
            raise ValueError('Expected exactly one retained final HTTP failure after consumed requests')
    elif set(states)-{'consumed'}:raise ValueError('Completed trial contains an unresolved API request')
    if result is not None and result['invocations']!=len(calls):raise ValueError('Result invocation count mismatch')
    return dict(requests=requests,states=dict(states),invocations=len(calls),literal_condition_returns=early,
        terminal_http_failure_retained=allow_terminal_http_failure,
        API_arguments_unchanged=True,private_budget_fields_absent=True,all_actions_use_selected_frozen_policy=True)


def main(output):
    freeze=read(BASE/'study_freeze.json');rows=[];interrupted=[];skipped=[]
    for name in NAMES:
        cfg=config(name);contract=read(cfg['completion_contract'])
        for method in ('naive_DP','APPL'):
            for condition in ('ID','OOD'):
                seeds=cfg['evaluation']['seeds' if condition=='ID' else 'ood_seeds']
                for seed in seeds:
                    root=BASE/name/'evaluation'/method/condition/str(seed)
                    job=BASE/'batch/jobs'/name/method/condition/str(seed)
                    if not (root/'result.json').exists() or not (job/'process_result.json').exists():
                        if (root/'failure.json').exists() and (job/'process_result.json').exists():
                            failure=read(root/'failure.json');initial=read(root/'initial_state.json')
                            if method!='APPL' or failure['error']!='Provider HTTP failure retained; no automatic retry':
                                raise ValueError('Unexpected interrupted-attempt type requires separate reconciliation')
                            if read(job/'process_result.json')['returncode']==0:raise ValueError('Failure/process mismatch')
                            audited=audit_episode(root,contract,None)
                            if audited['steps']!=failure['physical_steps'] or audited['first_success'] is not None:
                                raise ValueError('Interrupted physical prefix mismatch')
                            trace=[json.loads(line) for line in (root/'trace.jsonl').read_text().splitlines()] if (root/'trace.jsonl').exists() else []
                            api=audit_api(root,contract,trace,initial,None,freeze['study']['tasks'][name],True)
                            paired=BASE/name/'evaluation/naive_DP'/condition/str(seed)/'initial_state.json'
                            if paired.exists() and read(paired)!=initial:raise ValueError('Interrupted paired reset mismatch')
                            interrupted.append(dict(task=name,method=method,condition=condition,seed=seed,
                                prefix_evidence_passed=True,final_outcome_known=False,**audited,api=api))
                            continue
                        skipped.append(dict(task=name,method=method,condition=condition,seed=seed));continue
                    if read(job/'process_result.json')['returncode']!=0:raise ValueError('Completed result has a nonzero process exit')
                    result=read(root/'result.json');initial=read(root/'initial_state.json')
                    audited=audit_episode(root,contract,result)
                    if result['steps']>5000:raise ValueError('Physical cap exceeded')
                    if result['success'] and audited['first_success']!=result['steps']:
                        raise ValueError('Executor continued beyond geometric task success')
                    record=dict(task=name,method=method,condition=condition,seed=seed,passed=True,**audited)
                    if method=='APPL':
                        trace=[json.loads(line) for line in (root/'trace.jsonl').read_text().splitlines()] if (root/'trace.jsonl').exists() else []
                        record['api']=audit_api(root,contract,trace,initial,result,freeze['study']['tasks'][name])
                    other='APPL' if method=='naive_DP' else 'naive_DP'
                    paired=BASE/name/'evaluation'/other/condition/str(seed)/'initial_state.json'
                    if paired.exists() and read(paired)!=initial:raise ValueError('Paired initial state mismatch')
                    rows.append(record)
    value=dict(passed=True,audited_completed=len(rows),planned=600,all_600_audited=len(rows)==600,
        audited_interrupted_prefixes=len(interrupted),interrupted_episodes=interrupted,
        all_600_attempts_audited=len(rows)+len(interrupted)==600,
        skipped_not_completed_or_process_running=skipped,episodes=rows,
        analysis_source_sha256=digest(__file__),study_sha256=freeze['study_sha256'])
    atomic(output,value)
    print(dict(passed=True,audited_completed=len(rows),audited_interrupted_prefixes=len(interrupted),planned=600),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(BASE/'execution_audit.json'))
    main(parser.parse_args().output)
