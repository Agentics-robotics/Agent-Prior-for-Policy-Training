"""Read-only scientific validation and final handoff for the gripper ablation.

Run only after the 600-rollout supervisor has exited successfully. This command
adds reporting artifacts; it never starts physics, training, or an API client.
"""
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import time

from appl.io import ROOT,read,atomic,digest
from experiments.exp2.binary_gripper.common import BASE,TASKS,METHODS,verify,model_record,episode_root


def complete():
    plan=verify();summary=read(BASE/'results.json');closed=read(BASE/'supervisor/completed.json')
    assert summary['completed']==600 and summary['unknown']==0 and closed['active_workers']==0
    for task in TASKS:
        for method in METHODS:
            cell=next(c for c in plan['cells'] if c['task']==task and c['method']==method)
            model_record(cell)
            check=read(BASE/'checks'/task/method/'result.json')
            assert check['passed'] and check['max_action_error']==0 and check['actions']==17 and check['physical_steps']==0
            if method=='SinglePrior_6_xhigh':
                worker=BASE/'checks'/task/method/'worker'
                assert read(worker/'closed.json')['returncode']==0
                assert read(worker/'enforcement.json')['network'] is False
    for files in read(BASE/'fixed_inputs_audit.json')['task_inputs_match_original_freeze'].values():
        for path,sha in files.items():assert digest(path)==sha,path
    snapshot=BASE/'executor_snapshot'
    for name,sha in read(snapshot/'manifest.json')['files'].items():assert digest(snapshot/name)==sha

    phases=Counter();indices=defaultdict(list);events=[];jobs=[]
    for path in sorted((BASE/'supervisor/jobs').glob('*/*/process.json')):
        launch=read(path);exit_record=read(path.parent/'process_result.json')
        assert exit_record['returncode']==0
        assert exit_record['cell']==launch['cell'] and exit_record['gpu']==launch['gpu']
        phase=exit_record['phase'];phases[phase]+=1;indices[phase].append(launch['cell']['index'])
        events.extend([(launch['started'],1,launch['gpu']),(exit_record['finished'],-1,launch['gpu'])])
        jobs.append(dict(phase=phase,index=launch['cell']['index'],gpu=launch['gpu'],pid=launch['pid'],
            started=launch['started'],finished=exit_record['finished'],returncode=0))
    assert phases=={'check':10,'evaluate':600}
    assert sorted(indices['evaluate'])==list(range(600)) and len(set(indices['check']))==10
    active=Counter();peak=0
    for timestamp,delta,gpu in sorted(events):
        active[gpu]+=delta;assert active[gpu]>=0
        peak=max(peak,sum(v>0 for v in active.values()))
        assert sum(v>0 for v in active.values())<=5
    assert not any(active.values())

    groups=defaultdict(lambda:dict(episodes=0,success=0,failed=0,any_red_goal=0,any_blue_goal=0,
        failed_after_red_goal=0,failed_without_either_object_goal=0,after_1500=0,after_3000=0))
    paired=[];regressions=[];gains=[];physical=0;worker_exits=0
    for cell in plan['cells']:
        root=episode_root(cell);result=read(root/'result.json');audit=read(root/'audit.json');replay=read(root/'replay_provenance.json')
        assert audit['passed'] and audit['all_executed_actions_match_rule'] and audit['reference_initial_state_equal']
        assert digest(root/'trace.jsonl')==audit['trace_sha256'] and digest(root/'result.json')==audit['result_sha256']
        assert digest(root/'replay.mp4')==replay['video_sha256']
        assert int(replay['ffprobe']['nb_read_frames'])==len(replay['frames'])
        reference=Path(cell['reference_root'])
        for name,key in [('result.json','reference_result_sha256'),('trace.jsonl','reference_trace_sha256'),('initial_state.json','reference_initial_sha256')]:
            assert digest(reference/name)==cell[key]
        assert read(root/'initial_state.json')==read(reference/'initial_state.json')
        assert result['API_calls']==0 and result['learned_optimizer_updates']==0
        if cell['method']=='SinglePrior_6_xhigh':
            assert read(root/'worker/closed.json')['returncode']==0
            enforcement=read(root/'worker/enforcement.json')
            assert enforcement['network'] is False and enforcement['credentials'] is False
            worker_exits+=1
        physical+=result['steps'];g=groups[(cell['task'],cell['method'],cell['condition'])]
        red='red_on_pad' if cell['task']=='drawer_exchange' else 'red_at_goal'
        blue='blue_inside' if cell['task']=='drawer_exchange' else 'blue_at_goal'
        reached_red=red in result['first_goals'];reached_blue=blue in result['first_goals'];success=result['success']
        g['episodes']+=1;g['success']+=int(success);g['failed']+=int(not success)
        g['any_red_goal']+=int(reached_red);g['any_blue_goal']+=int(reached_blue)
        g['failed_after_red_goal']+=int(not success and reached_red)
        g['failed_without_either_object_goal']+=int(not success and not reached_red and not reached_blue)
        g['after_1500']+=int(success and result['steps']>1500);g['after_3000']+=int(success and result['steps']>3000)
        row={k:cell[k] for k in ['index','task','method','condition','seed','original_success','original_steps','reference_root']}
        row.update(success=success,steps=result['steps'],first_goals=result['first_goals'],episode_root=str(root))
        paired.append(row)
        if cell['original_success'] and not success:regressions.append(row)
        if not cell['original_success'] and success:gains.append(row)
    assert worker_exits==300 and physical==summary['physical_steps']

    ancestors={os.getpid()};pid=os.getppid()
    while pid>1:
        ancestors.add(pid)
        try:pid=int(Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[1])
        except FileNotFoundError:break
    live=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) in ancestors:continue
        try:argv=(path/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        text=' '.join(v.decode(errors='replace') for v in argv)
        if any(v==b'experiments.exp2.binary_gripper.run' for v in argv) or (
                any(v==b'appl.prior_policies' for v in argv) and str(BASE) in text):
            live.append(dict(pid=int(path.name),command=text))
    assert not live,live
    assert not list(BASE.rglob('journal.sqlite'))
    result=dict(passed=True,reviewed=time.time(),complete_physical_trials=600,preflight_checks=10,
        original_action_max_error=0,validated_replays=600,unknown=0,automatic_retries=0,
        API_calls=0,learned_optimizer_updates=0,physical_steps=physical,clean_policy_worker_exits=worker_exits,
        clean_preflight_policy_worker_exits=5,
        maximum_simultaneously_active_physical_gpus=peak,devices_used=sorted({j['gpu'] for j in jobs}),
        live_evaluation_workers=live,models_and_original_evidence_unchanged=True,
        paired_gains=len(gains),paired_losses=len(regressions),jobs=jobs,
        plan_sha256=digest(BASE/'plan.json'),results_sha256=digest(BASE/'results.json'))
    atomic(BASE/'final_validation.json',result)
    rows=[dict(task=k[0],method=k[1],condition=k[2],**v) for k,v in sorted(groups.items())]
    atomic(BASE/'failure_stage_counts.json',dict(groups=rows,paired_gains=gains,paired_losses=regressions,
        interpretation='Goal crossings describe the saved trajectories; they do not identify the cause of failure.'))
    lines=['# Outcome analysis and interpretation','',
        'This analysis uses all 600 completed new physical attempts, with no repeat selection and no new API calls. '
        'The two corrected methods reuse their original frozen weights; only executed gripper decoding changes.','',
        '## Full paired comparison','',
        '| Task | Split | Method | Original / 30 | Binary / 30 | Gained | Lost | Successes after 1500 | After 3000 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for g in summary['groups']:
        if g['task']=='all':continue
        a=groups[(g['task'],g['method'],g['condition'])]
        lines.append(f"| {g['task']} | {g['condition']} | {g['method']} | {g['original_success']} | {g['new_success']} | {g['paired_gains']} | {g['paired_losses']} | {a['after_1500']} | {a['after_3000']} |")
    lines+=['','## Remaining failed trajectories','',
        'A goal can be reached and subsequently lost. The counts below describe whether it was reached at any point; '
        'they are not grasp-contact labels or a causal diagnosis. Drawer has an additional drawer-open requirement.','',
        '| Task | Split | Method | Failed / 30 | Failures that reached red goal | Failures reaching neither object goal |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['task']} | {r['condition']} | {r['method']} | {r['failed']} | {r['failed_after_red_goal']} | {r['failed_without_either_object_goal']} |")
    lines+=['','## What this establishes','',
        'The action audit verifies that every final gripper command is exactly +1 or −1, with unchanged arm clipping. '
        'The gradual partial-opening command mechanism is therefore removed by construction. Paired outcome gains and '
        'losses measure the practical effect of this fixed decoder intervention on these already examined layouts.',
        'The API-authored single-prior policies also use the continuous gripper interface in their historical runs. '
        'They can be reevaluated with this correction entirely offline: there is no need to regenerate code, train new '
        'weights or invoke an agent at deployment.',
        'Remaining failures are evidence that this intervention is not sufficient for every layout. Earlier controlled '
        'diagnostics demonstrated narrow intermediate-state coverage for tray packing; the goal-crossing table alone '
        'does not prove that coverage explains each remaining failure, nor does it isolate training duration, network '
        'capacity or DDPM step count. Those variables were held fixed here.',
        'The original comparison should acknowledge its weak continuous-gripper baseline. Historical APPL scores '
        'remain unchanged and use the old execution interface. Both APPL variants were deliberately paused; a matched '
        'four-method corrected-executor comparison is not available from this run.',
        'These are paired single resets of previously inspected layouts, with one trained checkpoint per task. '
        'Fixed seeds and identical initial states do not establish zero native contact-simulation variability: the '
        'earlier expert replay audit found all sixty successful but only thirty-two full-state traces bitwise identical. '
        'Do not treat every individual gain/loss as a deterministic causal attribution or claim variability across '
        'training seeds was measured.','',
        '## Evidence and cost','',
        f"- {physical:,} new physical steps; 600 complete outcomes and videos; 0 unknown outcomes.",
        '- 0 Runtime API requests, 0 learned-model updates. Historical policy-design costs are not spent again.',
        '- Ten original-action reproduction checks: 17 actions per model, maximum difference 0.',
        f'- All 610 evaluation/check parent processes and 305 local restricted policy workers (300 evaluation + 5 preflight) exited successfully. Peak device use: {peak}.',
        '- [Final validation and process/resource audit](final_validation.json)',
        '- [Goal-crossing counts and every gained/lost pair](failure_stage_counts.json)',
        '- [Main report](REPORT.md), [all videos](replays.html), [fixed first-seed paired videos](paired_examples.html).','']
    (BASE/'ANALYSIS.md').write_text('\n'.join(lines))
    with (BASE/'REPORT.md').open('a') as stream:
        stream.write('\n[Outcome interpretation and remaining failures](ANALYSIS.md) · [Final validation](final_validation.json)\n')
    print({k:v for k,v in result.items() if k!='jobs'})


if __name__=='__main__':complete()
