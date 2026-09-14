"""One explicitly user-authorized exact recovery of the OOM-failed A4 slot.

No candidate/recipe changes, extra seeds or hidden feedback. Retain the original
failure/checkpoint/selection, resume the saved state, and refresh only q4 before
global freezing. No automatic retries of failed training or transport calls.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import time

from experiment1.records import ROOT, EXPERIMENT, PIXI, Records, atomic_json, immutable_json, read_json, file_hash, digest, encode, locked, now
from experiment1.protocol import verify_frozen, candidate_runtime_sources
from experiment1.runner import run_system, feedback_for

INSTANCE = 'assembly__N10__rep0'
DEST = EXPERIMENT / 'recoveries' / 'assembly_N10_A4_20260913'


def remaining_instances():
    active=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            args=(path/'cmdline').read_bytes().split(b'\0')
            cwd=(path/'cwd').resolve()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if cwd==ROOT and b'experiment1.cli' in args and b'instance' in args:
            active.append(int(path.name))
    return active


def recover():
    os.environ.update(CUDA_VISIBLE_DEVICES='0',CUDA_DEVICE_ORDER='PCI_BUS_ID',MUJOCO_EGL_DEVICE_ID='0')
    records=Records()
    protocol=verify_frozen()
    with locked(EXPERIMENT/'locks/runner.lock'), locked(EXPERIMENT/'locks/gpu-0.lock'), locked(EXPERIMENT/'locks'/('instance-'+INSTANCE+'.lock')):
        if (EXPERIMENT/'global_freeze.json').exists():
            raise ValueError('This authorized recovery requires hidden testing not to have started')
        slot=dict(records.db.execute("SELECT * FROM slots WHERE instance=? AND system='A4'",(INSTANCE,)).fetchone())
        if slot['state']!='failed':
            raise ValueError('One-shot recovery requires the original failed slot, not a new retry')
        if DEST.joinpath('before.json').exists():
            raise ValueError('Recovery already started; inspect saved state before any explicit resumption')
        package=records.submission(INSTANCE,'A4')
        candidate=EXPERIMENT/'generated'/INSTANCE/'A4/submitted'
        if any(file_hash(candidate/name)!=sha for name,sha in package['files'].items()):
            raise ValueError('Original submitted candidate changed')
        directory=EXPERIMENT/'runs'/INSTANCE/'A4'
        import torch
        checkpoint=torch.load(directory/'latest.pt',map_location='cpu',weights_only=False)
        if checkpoint['step']!=1 or checkpoint['protocol_hash']!=file_hash(EXPERIMENT/'protocol.lock.json'):
            raise ValueError('Expected the inspected step-one original recovery checkpoint')
        session=dict(records.db.execute('SELECT * FROM sessions WHERE instance=?',(INSTANCE,)).fetchone())
        if session['phase']!='selection':
            raise ValueError('Expected original completed selection phase')
        inventory=subprocess.run([str(PIXI),'run','--locked','nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader','-i','0'],capture_output=True,text=True,check=True)
        if inventory.stdout.strip():
            raise ValueError('GPU 0 has an existing compute process; recovery did not launch')
        DEST.mkdir(parents=True,exist_ok=True)
        immutable_json(DEST/'before.json',dict(time=now(),authority='User explicitly asked to fill the failed slot',
            slot=slot,submission=package,session=session,
            selections=[dict(row) for row in records.db.execute('SELECT * FROM selections WHERE instance=?',(INSTANCE,))],
            checkpoint_sha256=file_hash(directory/'latest.pt'),resumed_step=1,
            additional_scientific_slots=0,new_training_attempt=True,exact_saved_state=True,
            gpu=0,original_failure_retained=True,hidden_testing_started=False,
            original_protocol_sha256=file_hash(EXPERIMENT/'protocol.lock.json')))
        for source,target in [(directory/'latest.pt',DEST/'original_step1.pt'),
                              (directory/'training_costs.jsonl',DEST/'original_training_costs.jsonl'),
                              (EXPERIMENT/'feedback'/INSTANCE/'final.json',DEST/'original_final_feedback.json'),
                              (EXPERIMENT/'selections'/INSTANCE/'frozen.json',DEST/'original_frozen_selection.json')]:
            shutil.copy2(source,target)
        atomic_json(DEST/'status.json',dict(time=now(),status='resuming_training',step=1,gpu=0))
        records.event('authorized_oom_recovery_started',instance=INSTANCE,system='A4',gpu=0,
                      resume_step=1,original_failure_receipt=str(DEST/'before.json'),automatic_retry=False)
        records.set_slot(INSTANCE,'A4','training',gpu=0,authorized_recovery=str(DEST/'before.json'),resumed_step=1)
        outcome=run_system(records,'assembly',10,'A4',0)
        if outcome['status']!='completed':
            raise ValueError('Authorized recovery did not complete; failure retained, no automatic retry')
        if any(file_hash(candidate/name)!=sha for name,sha in package['files'].items()):
            raise ValueError('Candidate changed during recovery')
        feedback=feedback_for(records,INSTANCE,('A1','A2','A3','A4'))
        atomic_json(DEST/'recovered_final_feedback.json',feedback)
        # Explicit supersession, backed by the complete pre-recovery records.
        # Keep all earlier API requests/responses/tools/costs and original q1/q3.
        atomic_json(EXPERIMENT/'feedback'/INSTANCE/'final.json',feedback)
        message=dict(role='user',content=encode(dict(
            authorized_recovery=True,instance_id=INSTANCE,development_feedback=feedback,
            instruction='The user explicitly authorized exact recovery of the resource-OOM A4 failure. The original submitted code/config and seed are unchanged. A4 has now completed training and development before any hidden test. The earlier q4 selection and failure feedback are preserved but superseded. Record q4 using ONLY these updated development numbers and the SAME fixed rule. No code/design changes, new candidate, or further training. q1 and q3 stay unchanged. Existing tool/API budgets remain in force.')))
        with records.db:
            records.db.execute('DELETE FROM selections WHERE instance=? AND budget=4',(INSTANCE,))
            records.db.execute('UPDATE sessions SET history=? WHERE instance=?',
                (encode(json.loads(session['history'])+[message]),INSTANCE))
        records.event('q4_reopened_after_authorized_resource_recovery',instance=INSTANCE,
            preserved_original=str(DEST/'original_frozen_selection.json'),hidden_test_feedback=False)
        atomic_json(DEST/'status.json',dict(time=now(),status='refreshing_q4',development_completed=True))
        credential=Path.home()/'.config/experiment1/runtime-api-credential.json'
        if credential.stat().st_mode & 0o077:
            raise ValueError('Saved credential permissions are not private')
        os.environ.update(read_json(credential))
        from experiment1.api_client import APIConfig,ToolResponsesClient
        from experiment1.runtime_api import RuntimePriorAPI
        from experiment1.design_tools import DesignTools
        from experiment1.data import load_common_spec
        from experiment1.preflight import checker_for,support_slice
        common=load_common_spec('assembly',EXPERIMENT)
        tools=DesignTools(records,INSTANCE,'selection',EXPERIMENT/'public',EXPERIMENT/'evidence'/INSTANCE,
            common,checker_for(support_slice(EXPERIMENT,'assembly',10),candidate_runtime_sources()),feedback=feedback)
        RuntimePriorAPI(records,ToolResponsesClient(APIConfig(**protocol['api']))).run_phase(tools,
            phase_message=dict(instance_id=INSTANCE,development_feedback=feedback,authorized_resource_recovery=True))
        selected={str(row['budget']):json.loads(row['record']) for row in records.db.execute('SELECT * FROM selections WHERE instance=?',(INSTANCE,))}
        old=read_json(DEST/'original_frozen_selection.json')
        assert all(selected[q]==old['selections'][q] for q in ('1','3'))
        atomic_json(EXPERIMENT/'selections'/INSTANCE/'frozen.json',dict(instance=INSTANCE,selections=selected,
            submissions={c:records.submission(INSTANCE,c)['submission_hash'] for c in ('A1','A2','A3','A4')},
            feedback_hashes=dict(initial=file_hash(EXPERIMENT/'feedback'/INSTANCE/'initial.json'),
                                 final=file_hash(EXPERIMENT/'feedback'/INSTANCE/'final.json'))))
        records.event('authorized_oom_recovery_complete',instance=INSTANCE,system='A4',gpu=0,
            original_failure_preserved=True,q4=selected['4'],candidate_unchanged=True)
        atomic_json(DEST/'status.json',dict(time=now(),status='recovered_and_selection_frozen',q4=selected['4']['candidate_id']))
    # Let the already-running independent instance finish without interference.
    while remaining_instances():
        time.sleep(10)
    with (DEST/'restart.log').open('x') as log:
        subprocess.run([str(PIXI),'run','--locked','python',str(Path.home()/'.config/experiment1/launch.py')],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    atomic_json(DEST/'status.json',dict(time=now(),status='recovered_and_outer_runner_restarted'))


if __name__=='__main__':
    try:
        recover()
    except Exception as error:
        atomic_json(DEST/'status.json',dict(time=now(),status='blocked',reason=str(error),automatic_retry=False))
        raise
