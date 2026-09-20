"""Explicit one-attempt transport recovery; original attempt remains immutable."""
import copy
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

from appl.agent import AgentLoop
from appl.io import ROOT,atomic,digest,object_hash,read
from appl.prior_policies.design import client
from appl.prior_policies.report import authorship
from appl.prior_policies.runner import gpu_job
from experiments.exp2.astra.runner import api_identity
from experiments.exp2.astra.transport import install_episode_gate
from experiments.exp2.single_policy.common import BASE,NAMES,config,folder,freeze,verify_preparation
from experiments.exp2.single_policy.design import FullTaskTools,PROMPT
from experiments.exp2.single_policy.runner import command,phase_run

INCIDENT=BASE/'incidents/tray_design_http502'
PROPOSAL=ROOT/'experiments/exp2/analysis/single_policy_tray_recovery_proposal.json'


def inventory(root):
    return {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}


def design_continuation():
    authorization=read(INCIDENT/'authorization.json');proposal=read(PROPOSAL)
    if not authorization['authorized'] or authorization['proposal_sha256']!=digest(PROPOSAL):
        raise ValueError('Exact user authorization required')
    verify_preparation();f=folder('tray_pack');original=f/'design'
    if (INCIDENT/'continuation.json').exists(): raise RuntimeError('Authorized recovery already attempted')
    if (original/'journal.sqlite-wal').exists(): raise ValueError('Original journal is not quiescent')
    if digest(original/'journal.sqlite')!=proposal['original_journal_sha256']:
        raise ValueError('Original interrupted journal changed')
    before=inventory(original)
    original_db=sqlite3.connect((original/'journal.sqlite').resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    api=original_db.execute('SELECT * FROM api ORDER BY seq').fetchall()
    tools=original_db.execute('SELECT * FROM tools ORDER BY id').fetchall()
    history=json.loads(original_db.execute("SELECT value FROM state WHERE key='history'").fetchone()[0])
    original_db.close()
    if len(api)!=21 or api[-1][1]!='http_failed' or any(r[1]!='consumed' for r in api[:-1]):
        raise ValueError('Original request states differ from the proposal')
    retained=INCIDENT/'original_attempt/design';retained.parent.mkdir(parents=True,exist_ok=False)
    original.rename(retained)
    (f/'failure.json').rename(INCIDENT/'original_attempt/policy_failure.json')
    shutil.copytree(retained,original)
    if inventory(retained)!=before or inventory(original)!=before:
        raise ValueError('Exact original preservation/copy failed')
    # Only the derived active journal drops the unresolved transport row. The
    # complete executed original, including that row and cost event, is retained.
    # The consumed conversation, every tool result and all API file bytes remain.
    with sqlite3.connect(original/'journal.sqlite') as db:
        db.execute("DELETE FROM api WHERE seq=21 AND status='http_failed'")
        db.execute("DELETE FROM events WHERE kind='api_cost' AND json_extract(payload,'$.seq')=21")
        db.commit()
        if db.execute('SELECT * FROM api ORDER BY seq').fetchall()!=api[:-1]:
            raise ValueError('Consumed API prefix changed')
        if db.execute('SELECT * FROM tools ORDER BY id').fetchall()!=tools:
            raise ValueError('Original tool output changed')
    cfg=copy.deepcopy(config('tray_pack'))
    # 20 inherited consumed calls + at most 15 new calls. The retained failed
    # request is the additional charged attempt: 20 + 15 + 1 <= original 36.
    cfg['design']['max_api_calls']=35
    receipt=dict(status='starting',started=time.time(),original_inventory=before,
        original_attempt=str(retained),original_api_calls=21,consumed_prefix_calls=20,
        additional_API_limit=15,original_failed_attempts_charged_separately=1,
        effective_active_journal_limit=35,original_total_limit=36,
        source_sha256=digest(__file__),authorization_sha256=digest(INCIDENT/'authorization.json'),
        original_history_sha256=object_hash(history),automatic_retry=False,policy_output_edits=0)
    atomic(INCIDENT/'continuation.json',receipt)
    t=FullTaskTools(cfg,f,7);install_episode_gate(BASE,f/'design',slots=4)
    c=client(cfg);respond=c.respond;new_calls=0
    def checked(request):
        nonlocal new_calls
        if (request['model'],request['reasoning']['effort'])!=('gpt-6-astra','xhigh'):
            raise ValueError('Model or effort changed')
        if new_calls==0 and object_hash(request)!=proposal['failed_request_sha256']:
            raise ValueError('First recovery request must exactly match the failed request')
        new_calls+=1
        if new_calls>15: raise ValueError('Authorized remaining API budget exceeded')
        return respond(request)
    c.respond=checked
    try:
        AgentLoop(t.j,t,c,PROMPT).run(dict(policy_id='SinglePrior_6_xhigh',
            task='Design, implement, check and submit one full-task prior Diffusion Policy from the supplied training demonstrations.'))
        if [tuple(r) for r in t.j.db.execute('SELECT * FROM api WHERE seq<=20 ORDER BY seq')]!=api[:-1]:
            raise ValueError('Original consumed API prefix changed during continuation')
        after_tools={r[0]:tuple(r) for r in t.j.db.execute('SELECT * FROM tools')}
        if any(after_tools[r[0]]!=r for r in tools): raise ValueError('Original tool results changed')
        if t.j.get('history')[:len(history)]!=history: raise ValueError('Conversation prefix changed')
        if inventory(retained)!=before: raise ValueError('Retained original attempt changed')
        submitted=read(f/'submission.json')
        atomic(INCIDENT/'continuation.json',dict(receipt,status='submitted',finished=time.time(),
            additional_API_calls=new_calls,original_attempt_unchanged=True,
            original_history_preserved=True,exact_first_request_verified=True,version=submitted['version']))
    except Exception as error:
        atomic(INCIDENT/'continuation.json',dict(receipt,status='failed',finished=time.time(),
            additional_API_calls=new_calls,error=str(error),original_attempt_unchanged=inventory(retained)==before))
        atomic(f/'failure.json',dict(error=str(error),automatic_retry=False,authorized_recovery_failed=True));raise
    finally:
        t.j.db.close();os.environ.pop('APPL_PRIOR_TOKEN',None)
    api_identity(f/'design/journal.sqlite');authorship(f,submitted)
    if (f/'training').exists(): raise ValueError('Original candidate must not already have formal training')
    gpu_job(config('tray_pack'),f,f/'source',f/'training',7,60000)
    subprocess.run(command('check','--task','tray_pack','--gpu',7),cwd=ROOT,check=True)
    atomic(INCIDENT/'model_ready.json',dict(completed=time.time(),training_sha256=digest(f/'training/result.json'),
        deployment_check_sha256=digest(f/'deployment_check/result.json')))


def account_retained_failure():
    """Add only the retained failed call; inherited consumed prefix is not doubled."""
    snapshot=INCIDENT/'report_before_transport_accounting';snapshot.mkdir()
    for name in ('results.json','REPORT.md','completion.json'):
        shutil.copyfile(BASE/name,snapshot/name)
    result=read(BASE/'results.json');summary=result['summary']
    states=summary['design_API_states'];states['http_failed']=states.get('http_failed',0)+1
    summary.update(total_actual_design_API_requests=sum(states.values()),
        retained_failed_requests_without_reported_usage=1,authorized_transport_recoveries=1)
    result['retained_failed_API_accounting']=dict(calls=1,states={'http_failed':1},usage=None,
        note='Only original attempt request 21. Consumed requests 1-20 are inherited unchanged and counted once in the active journal.',
        original_journal_sha256=digest(INCIDENT/'original_attempt/design/journal.sqlite'),
        continuation_sha256=digest(INCIDENT/'continuation.json'))
    atomic(BASE/'results.json',result)
    report=(BASE/'REPORT.md').read_text()
    previous=read(snapshot/'results.json')['summary']['design_API_states']
    report=report.replace(f"request states: `{previous}`",f"request states: `{states}`")
    report+='\n## Explicit transport recovery\n\nOne tray-design HTTP 502 is retained with unknown usage. The user authorized one exact-request continuation of the same unsubmitted candidate within the original 36-call budget; the consumed prefix and API source bytes were preserved. No other candidate was repeated. The active journal counts the inherited consumed prefix once; the retained failed request is charged separately. See [continuation receipt](incidents/tray_design_http502/continuation.json).\n'
    (BASE/'REPORT.md').write_text(report)
    completed=read(BASE/'completion.json');completed.update(summary)
    completed['artifacts'].update({name:digest(BASE/name) for name in ('results.json','REPORT.md')})
    completed['transport_recovery_receipt_sha256']=digest(INCIDENT/'continuation.json')
    atomic(BASE/'completion.json',completed)
    return summary


def main():
    if (INCIDENT/'recovery_started.json').exists(): raise RuntimeError('Controlled recovery already started')
    atomic(INCIDENT/'recovery_started.json',dict(started=time.time(),pid=os.getpid(),source_sha256=digest(__file__)))
    try:
        design_continuation()
        # The original coordinator finishes its four unaffected jobs and keeps
        # its failed exit. This continuation does not relabel that execution.
        while not (BASE/'supervisor/training_failed.json').exists():
            if (BASE/'supervisor/failure.json').exists(): raise RuntimeError('Unexpected original supervisor failure')
            print('Waiting for the four original training/check jobs to finish.',flush=True)
            time.sleep(30)
        ended=read(BASE/'supervisor/training_failed.json')
        if ended['active'] or ended['pending'] or ended['failures']!=[dict(task='tray_pack',gpu=1,returncode=1)]:
            raise ValueError('Original supervisor has additional failures or unresolved work')
        freeze()
        jobs=[]
        for index in range(30):
            for condition in ('ID','OOD'):
                for name in NAMES:
                    seed=config(name)['evaluation']['seeds' if condition=='ID' else 'ood_seeds'][index]
                    jobs.append(dict(task=name,condition=condition,seed=seed))
        phase_run('evaluation',jobs,[0,1,7,0,1,7])
        from experiments.exp2.single_policy.report import generate
        generate();summary=account_retained_failure()
        if inventory(INCIDENT/'original_attempt/design')!=read(INCIDENT/'continuation.json')['original_inventory']:
            raise ValueError('Original failed attempt changed')
        atomic(INCIDENT/'recovery_completed.json',dict(completed=time.time(),summary=summary,
            original_attempt_preserved=True,automatic_retries=0))
        print(dict(completed=True,summary=summary),flush=True)
    except Exception as error:
        atomic(INCIDENT/'recovery_failed.json',dict(error=str(error),time=time.time(),automatic_retry=False));raise


if __name__=='__main__': main()
